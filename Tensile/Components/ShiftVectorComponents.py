################################################################################
# Copyright 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from .. import Code
from ..Component import ShiftVectorComponents
from ..AsmUtils import vgpr, sgpr, accvgpr, staticMultiply, vectorStaticDivide, vectorStaticRemainder, log2

class ShiftVectorComponentsMFMA(ShiftVectorComponents):
    kernel = {"EnableMatrixInstruction": True}

    """
    Shift Vector Components d0,1
    """
    def __call__(self, writer, kernel, tP):
        """ when we enable shift ptr with vectorwidth(2), we shift global read on edge block when size % vectorwidth != 0.
            For example if M size == 3 vector width == 2, we want to do global read for [0-1] and [2-3].
            But 3 is not in memory object, so we shift to do global read [0-1] and [1-2].
            So buffer become [0, 1, 1, 2], assume result in register is same as input [0, 1, 1, 2]
            We need to shift it back to [0, 1, 2].

            In MFMA outputs, We have numContinuousOutput(4) for each thread.
            We have numThreadInWave(64) threads.
            number of thread in N is sames as kernel["MatrixInstN"] (32)
            number of thread in M is numThreadInWave/numOutputThreads1 = 2
            stride of continuous output for each thread (numSubOutputPerWave0) is numOutputThreads0 * numContinuousOutput, (8).
            we have numSubOutputGroupsPerWave0 which is 4 (kernel[tP["mt"]](64) // numSubOutputPerWave0(8))

            So we do shift back by below algorithm.
            1. check if M_size % GlobalLoadVectorWidth != 0, return if == 0
            2. decide which subgroup we need to shift, M_size(3) means 3/8 = group 0
            3. decide which thread we need to shift, we have different groups of thread, (0-31) for first group, (32-63) for second group.
            4. decide which shift block (subTile1) we want to shift. for ex [0-1], [1-2], we want to shift second subtile
        """

        # TODO: use this for non SourceSwap for B?
        # this part can  support non SourceSwap for B
        # But let non SourceSwap for B go original shiftptr path
        if (not kernel["SourceSwap"]) and tP["isB"]:
            return Code.Module("ShiftVectorComponentsMFMA (Empty)")

        # common parameter
        regPerElem      = kernel["MIRegPerOut"]
        glvw            = tP["glvw"]
        numThreadInWave = writer.kernel["WavefrontSize"]
        accImOffset     = writer.AccVgprImagNumOffset(kernel)
        vectorWidth     = kernel["VectorWidth"] if (kernel["SourceSwap"] and tP["isA"]) else 1

        # use to handle MatrixInst 4x4
        matrixInstM     = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        matrixInstN     = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
        matrixInstBM    = 1 if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
        matrixInstBN    = 1 if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

        # unify process for dimension M/N
        matrixInstCoal  = matrixInstM              if tP["isA"] else matrixInstN
        matrixInstPrep  = matrixInstN              if tP["isA"] else matrixInstM
        matrixInstBCoal = matrixInstBM             if tP["isA"] else matrixInstBN
        matrixInstBPrep = matrixInstBN             if tP["isA"] else matrixInstBM
        miWaveGroupCoal = kernel["MIWaveGroup"][0] if tP["isA"] else kernel["MIWaveGroup"][1]
        miWGIdStride    = numThreadInWave          if tP["isA"] else (numThreadInWave * kernel["MIWaveGroup"][0])
        miWaveTitleCoal = kernel["MIWaveTile"][0]  if tP["isA"] else kernel["MIWaveTile"][1]
        miWaveTitlePrep = kernel["MIWaveTile"][1]  if tP["isA"] else kernel["MIWaveTile"][0]

        # unify process for SourceSwap and non-SourceSwap
        conThInProcDim  = kernel["SourceSwap"] ^ tP["isB"] # continuous threads in processed dimension(Coalesced dimension)

        threadInterval  = 1 if conThInProcDim else matrixInstPrep
        numThreadInCoal = matrixInstCoal if conThInProcDim else (numThreadInWave // matrixInstPrep)

        numContOutCoal  = 1 if conThInProcDim else kernel["MIOutputVectorWidth"]
        allContOutCoal  = numContOutCoal * vectorWidth

        OutBlocksInMI   = (matrixInstCoal * matrixInstPrep) // numThreadInWave // numContOutCoal
        OutBlocksInMI   = 1 if conThInProcDim else OutBlocksInMI

        subMBShapeCoal  = (matrixInstCoal * vectorWidth) if conThInProcDim else ((numThreadInWave // matrixInstPrep) * numContOutCoal)
        MBShapeCoal     = subMBShapeCoal * OutBlocksInMI
        MIBShapeCoal    = MBShapeCoal * matrixInstBCoal
        WGShapeCoal     = MIBShapeCoal * miWaveGroupCoal
        miOuterTTCoal   = miWaveTitleCoal // vectorWidth

        numOutputsPrep  = (matrixInstCoal * matrixInstPrep // numThreadInWave) if conThInProcDim else 1
        numOutputsPrep  = numOutputsPrep * matrixInstBPrep * miWaveTitlePrep
        complexMultiplier = 2 if kernel["ProblemType"]["DataType"].isComplex() else 1

        # unify process for dimension M/N
        regStrideCoal = 1                                                                if tP["isA"] else numOutputsPrep
        regStridePrep = miOuterTTCoal * matrixInstBCoal * OutBlocksInMI * allContOutCoal if tP["isA"] else 1


        # labels for shiftptr
        glvwLabels = []
        MBblockLabels = []
        VWBlockLabels = []
        for i in range(0, glvw): # grvw block
            r = (i+1) % glvw    # r = [1,2,3,...,glvw-1, 0], the last one glvwLabels[glvw-1] stores for r=0 -> no shift
            comment = "end shift0" if i == glvw-1 else ""
            label = Code.Label(writer.labels.getName("ShiftVectorComponents%u_GLVW%u" % (tP["idx"], r) ), comment)
            glvwLabels.append(label)
            subMBLabels = []
            subVWBlockLabels = []
            for mb in range(0, OutBlocksInMI * matrixInstBCoal * miOuterTTCoal): # unit block of each thread
                label = Code.Label(writer.labels.getName("ShiftVectorComponents%u_GLVW%u_BM%u" % (tP["idx"], r, mb)), "")
                subMBLabels.append(label)
                sub2VWBlockLabels = []
                for vw in range(0, max(1, allContOutCoal//glvw)): # vw block of glvw
                    label = Code.Label(writer.labels.getName("ShiftVectorComponents%u_GLVW%u_BM%u_VW%u" % (tP["idx"], r, mb, vw)), "")
                    sub2VWBlockLabels.append(label)
                subVWBlockLabels.append(sub2VWBlockLabels)
            MBblockLabels.append(subMBLabels)
            VWBlockLabels.append(subVWBlockLabels)

        # wgMT value
        tmpSgpr = writer.getTmpSgpr(writer.laneSGPRCount).idx()
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2)
        dummy   = writer.vgprPool.checkOut(1)
        wgMT    = writer.vgprPool.checkOut(1)
        wg      = tP["wg"]

        module = Code.Module("ShiftVectorComponentsMFMA")

        # get M size of edge block
        mtReg = writer.vgprPool.checkOut(1)
        module.addInst("v_mov_b32"    , vgpr(wgMT), sgpr(wg), "")
        module.addInst("v_mul_i32_i24", vgpr(wgMT), hex(-kernel[tP["mt"]]), vgpr(wgMT), "wg*MT")
        module.addInst("_v_add_co_u32", vgpr(wgMT), writer.vcc, sgpr("SizesFree+%u"%tP["idx"]), vgpr(wgMT), "wgMT = Size - wg*MT")
        module.addInst("v_mov_b32"    , vgpr(mtReg), hex(kernel[tP["mt"]]), "MT")
        module.addInst("v_cmp_lt_u32" , sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(wgMT), vgpr(mtReg), "wgMT < MT" )
        module.addInst("v_cndmask_b32", vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), sgpr(tmpSgpr,writer.laneSGPRCount), "wgMT = (wgMT < MT) ? wgMT : MT" )

        # identify which wave have to process
        wReg = writer.vgprPool.checkOut(1)
        sReg = writer.vgprPool.checkOut(1)
        module.addCode(vectorStaticDivide(wReg, "Serial", miWGIdStride, tmpVgpr, tmpSgpr))
        module.addCode(vectorStaticRemainder(dummy, wReg, wReg, miWaveGroupCoal, tmpVgpr, tmpSgpr))
        module.addCode(vectorStaticDivide(sReg, wgMT, MIBShapeCoal, tmpVgpr, tmpSgpr))
        module.addCode(vectorStaticRemainder(dummy, sReg, sReg, miWaveGroupCoal, tmpVgpr, tmpSgpr))
        module.addInst("v_cmp_eq_u32" , sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(sReg), vgpr(wReg), "wave_id == block_belong_to_wave?" )
        module.addInst("v_cndmask_b32", vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), sgpr(tmpSgpr,writer.laneSGPRCount), "wgMT = (wgMT < MT) ? wgMT : MT" )
        writer.vgprPool.checkIn(mtReg)
        writer.vgprPool.checkIn(sReg)

        # mbReg: which mb block meed to shift, mb(matrixInstM*VectorWidth)
        module.addComment1("mbReg: which mb block need to shift, mb(matrixInstCoal(%u) * VectorWidth(%u))" % (matrixInstCoal, vectorWidth))
        mbReg = writer.vgprPool.checkOut(1)
        tReg  = writer.vgprPool.checkOut(1)
        module.addCode(vectorStaticDivide(mbReg, wgMT, subMBShapeCoal, tmpVgpr, tmpSgpr))
        module.addCode(staticMultiply(vgpr(tReg), vgpr(wReg), (matrixInstBCoal * OutBlocksInMI), sgpr(tmpSgpr)))
        module.addInst("_v_sub_u32", vgpr(mbReg), vgpr(mbReg), vgpr(tReg), "")
        writer.vgprPool.checkIn(tReg)

        # gbReg: glvw block id
        module.addComment1("gbReg: glvw block id")
        gbReg = writer.vgprPool.checkOut(1)
        module.addCode(vectorStaticDivide(gbReg, wgMT, glvw, tmpVgpr, tmpSgpr))

        # tgbReg: thread in glvw block
        module.addComment1("tgbReg: glvw block id")
        tgbReg = writer.vgprPool.checkOut(1)
        module.addCode(vectorStaticDivide(tgbReg, "Serial", threadInterval, tmpVgpr, tmpSgpr))
        module.addCode(vectorStaticRemainder(dummy, tgbReg, tgbReg, numThreadInCoal, tmpVgpr, tmpSgpr))
        module.addCode(staticMultiply(vgpr(tgbReg), vgpr(tgbReg), allContOutCoal, sgpr(tmpSgpr)))
        module.addCode(vectorStaticDivide(tgbReg, tgbReg, glvw, tmpVgpr, tmpSgpr))
        module.addCode(staticMultiply(vgpr(wReg), vgpr(wReg), MIBShapeCoal//glvw, sgpr(tmpSgpr)))
        module.addInst("_v_add_co_u32", vgpr(tgbReg), writer.vcc, vgpr(wReg), vgpr(tgbReg), "tgbReg = (tid_coal * continOut) / GLVW")
        module.addInst("_v_sub_u32", vgpr(gbReg), vgpr(gbReg), vgpr(tgbReg), "")
        writer.vgprPool.checkIn(wReg)
        writer.vgprPool.checkIn(tgbReg)

        # vw block of glvw
        module.addComment1("vwReg: glvw in which vw block?")
        vwReg = writer.vgprPool.checkOut(1)
        module.addInst("v_and_b32", vgpr(vwReg), allContOutCoal-1, vgpr(wgMT), "permute register between threads")
        module.addInst("v_lshrrev_b32", vgpr(vwReg), log2(glvw), vgpr(vwReg), "permute register between threads")

        # rReg : reminder of M_size % vectorwidth
        # decide to jump to block which handle this case, M_size % vector width
        module.addComment1("rReg : reminder of M_size % GlobalLoadVectorWidth")
        rReg = writer.vgprPool.checkOut(1)
        module.addCode(vectorStaticRemainder(dummy, rReg, wgMT, glvw, tmpVgpr, tmpSgpr))
        for r in range(1, glvw):
            module.addInst("v_cmp_eq_u32", writer.vcc, vgpr(rReg), hex(r), "wgMT%%VW == %u"%r )
            module.addCode(Code.BranchInst("s_cbranch_vccnz", glvwLabels[(r-1)].getLabelName(), "branch to shift d%u r=%u"%(tP["idx"], r)))
        module.addCode(Code.BranchInst("s_branch", glvwLabels[glvw-1].getLabelName(), "no shifting" ))
        writer.vgprPool.checkIn(rReg)

        _, arch2acc = writer.AccToArchMapper(kernel)

        # blocks for handle M_size % vector width
        for r in range(1, glvw):
            module.addComment2("shift d%u r=%u"%(tP["idx"], r))
            module.addCode(glvwLabels[r-1])
            for tt in range(0, miOuterTTCoal):
                for bm in range(0, matrixInstBCoal):
                    for ob in range(0, OutBlocksInMI):
                        label  = ob + OutBlocksInMI * (bm + matrixInstBCoal * tt)
                        target = ob + OutBlocksInMI * (bm + matrixInstBCoal * miWaveGroupCoal * tt)
                        module.addInst("v_cmp_eq_u32", writer.vcc, vgpr(mbReg), hex(target), "")
                        module.addCode(Code.BranchInst("s_cbranch_vccnz", MBblockLabels[r-1][label].getLabelName(), "branch to shift d%u r%u mb%u" % (tP["idx"], r, label)))

        for r in range(1, glvw):
            for mb in range(0, miOuterTTCoal * matrixInstBCoal * OutBlocksInMI):
                module.addComment2("shift d%u r=%u mb=%u"%(tP["idx"], r, mb))
                MBblockLabels[r-1][mb].comment = "r%u mb%u"%(r, mb)
                module.addCode(MBblockLabels[r-1][mb])
                for vw in range(0, max(1, allContOutCoal//glvw)):
                    module.addInst("v_cmp_eq_u32", writer.vcc, vgpr(vwReg), hex(vw), "")
                    module.addCode(Code.BranchInst("s_cbranch_vccnz", VWBlockLabels[r-1][mb][vw].getLabelName(), "branch to shift d%u r%u mb%u vw%u" % (tP["idx"], r, mb, vw)))

        # blocks for handle M_size % vector width
        tReg  = writer.vgprPool.checkOut(min(glvw, allContOutCoal))
        for r in range(1, glvw):
            for tt in range(0, miOuterTTCoal):
                for bm in range(0, matrixInstBCoal):
                    for ob in range(0, OutBlocksInMI):
                        mb = ob + OutBlocksInMI * (bm + matrixInstBCoal * tt)
                        for vw in range(0, max(1, allContOutCoal//glvw)):
                            module.addComment2("shift d%u r=%u mb=%u vw%d"%(tP["idx"], r, mb, vw))
                            VWBlockLabels[r-1][mb][vw].comment = "r%u mb%u vw%u"%(r, mb, vw)
                            module.addCode(VWBlockLabels[r-1][mb][vw])
                            module.addInst("s_mov_b32", sgpr(tmpSgpr), (((ob*subMBShapeCoal + bm*MBShapeCoal + tt*WGShapeCoal) // glvw) + vw), "")
                            module.addInst("v_cmpx_eq_u32", sgpr(tmpSgpr, writer.laneSGPRCount), vgpr(gbReg), sgpr(tmpSgpr), "is thread in edge glvw region" )
                            module.addInst("v_and_b32", vgpr(tmpVgpr), kernel["WavefrontSize"]-1, vgpr("Serial"), "permute register between threads")
                            module.addInst("v_lshlrev_b32", vgpr(tmpVgpr), log2(writer.bpr), vgpr(tmpVgpr), "permute register between threads")

                            for ot in range(numOutputsPrep):
                                for c  in range(complexMultiplier):
                                    for nr in range(regPerElem):
                                        vgprOffsetForSCIU = 0
                                        copyInstStr = "v_accvgpr_read_b32" if not kernel["MIArchVgpr"] else "v_mov_b32"
                                        for e in range(min(r, allContOutCoal)):
                                            src = (e+(glvw-r)) % allContOutCoal
                                            srcVgpr = (src + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                            srcVgpr = srcVgpr + ot * regStridePrep
                                            srcVgpr = arch2acc[srcVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                            srcStr = accvgpr(srcVgpr) if not kernel["MIArchVgpr"] else vgpr(srcVgpr)
                                            module.addInst(copyInstStr, vgpr(tReg+e), srcStr, "glvw %u mb %u tt1 %u r %u" % (r, mb, ot, nr))

                                        if not kernel["MIArchVgpr"]:
                                            module.addInst("s_nop", "1", "v_accvgpr read vgpr after write vgpr: 2 wait states")

                                        needWait = False
                                        for e in range(min(r, allContOutCoal)):
                                            crossThread = (e+(glvw-r)) // allContOutCoal
                                            if crossThread != 0:
                                                module.addInst("ds_bpermute_b32", vgpr(tReg+e), vgpr(tmpVgpr), vgpr(tReg+e), "offset:{}".format(crossThread*threadInterval*4), "permute edge values")
                                                needWait = True

                                        if needWait:
                                            module.addWaitCnt(waitAll=True, comment="wait for swizzle operation")

                                        copyInstStr = "v_accvgpr_write_b32" if not kernel["MIArchVgpr"] else "v_mov_b32"
                                        for e in range(min(r, allContOutCoal)):
                                            dstVgpr = (e + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                            dstVgpr = dstVgpr + ot * regStridePrep
                                            dstVgpr = arch2acc[dstVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                            dstStr = accvgpr(dstVgpr) if not kernel["MIArchVgpr"] else vgpr(dstVgpr)
                                            module.addInst(copyInstStr, dstStr, vgpr(tReg+e), "")

                            # end shift reset mask and jump out
                            all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                            module.addInst("s_mov_b{}".format(kernel["WavefrontSize"]), sgpr(tmpSgpr, writer.laneSGPRCount), all1mask, "to restore all threads active")
                            module.addInst("s_or_saveexec_b{}".format(kernel["WavefrontSize"]), writer.vcc, sgpr(tmpSgpr,writer.laneSGPRCount), "all threads active")
                            module.addCode(Code.BranchInst("s_branch", glvwLabels[glvw-1].getLabelName(), "done shifting" ))
                            module.addSpaceLine()

        module.addCode(glvwLabels[glvw-1])
        writer.vgprPool.checkIn(tReg)

        # checkin scratch vgprs
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(wgMT)
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(gbReg)
        writer.vgprPool.checkIn(vwReg)
        writer.vgprPool.checkIn(mbReg)

        return module
