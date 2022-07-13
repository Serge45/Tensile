################################################################################
# Copyright 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
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

from . import Code
from .AsmUtils import vgpr, sgpr, log2
from .Common import globalParameters

##############################################################################
# Fields associated with computing address
##############################################################################
class AddrCalculation:
    # rowInc is number of rows to add to the base address
    # coord0Vgpr : This is VGPR that holds coord0.  Coord0 is element-space
    #    packed index for the 0 coordinate of the C/D matrix.
    # coord1Vgpr : VGPR which tracks the last coord1 calculation.
    #          If this is new coord1, just overwrite it with latest calc.
    def __init__(self, kernelWriter, ss, addrCVgpr, addrDVgpr, element, \
        coordOffset0, coord1Vgpr, coordOffset1, rowInc, newCoord1):
        self.kernelWriter = kernelWriter

        # vgprs for address, could be more than one (for flat)
        self.addrDVgpr = addrDVgpr
        self.addrCVgpr = addrCVgpr
        self.coord1Vgpr = coord1Vgpr # vgpr that stores coord1Vgpr

        self.element = element
        self.coordOffset0 = coordOffset0
        self.coordOffset1 = coordOffset1
        self.rowInc = rowInc
        self.rowIncDirtyRowPtr = 0 # rowInc was used to modify rowPtr, need to recompute addr
        self.newCoord1 = newCoord1 # vgpr that stores newCoord1

        if ss.optSingleColVgpr:
            # optimized stores use the load offset for coordOffset0 calculations.
            self.globalOffset = coordOffset0 * kernelWriter.bpeCexternal
        else:
            # else non-opt stores include the coord0 offset into VGPR address calcs
            self.globalOffset = 0

    def addScaled(self, destV, src0, src1, scale1, tmpS01, comment=""):
        """
        Use minimally efficient instructions to add stride*scale
        """

        module = Code.Module("addScaled")
        if scale1 == 1:
            module.addInst("_v_add_u32", destV, src0, src1, comment)
        else:
            module.addInst("s_mul_i32", sgpr(tmpS01), src1, scale1, "scale stride")
            module.addInst("_v_add_u32", destV, src0,  sgpr(tmpS01), comment)
        return module


    def emitAddressCoordIncrement(self, kernel, ss, tmpVgpr, tmpS01, updateCoord1):
        """
        Emit code that computes the coord0 and coord1 for this element
        sets self.coord0Vgpr with the address that holds the coord0 value for this element.
        Input:
          - tmpVgpr is a 1 temporary VGPR used for coord0 calculation on edges
        """

        module = Code.Module("emitAddressCoordIncrement")
        kw = self.kernelWriter
        (d1,d0,vc1,vc0) = self.element
        self.coord0Vgpr = None # will set below

        # module.addText(self.kernelWriter.comment1("store addr=v%u coordOffset0=%u"% \
        #    (self.addr, self.coordOffset0)))
        module.addText(self.kernelWriter.comment1("(d1,vc1,d0,vc0)=(%u,%u,%u,%u)"\
            % (d1,vc1,d0,vc0)))
        if ss.optSingleColVgpr:
            self.coord0Vgpr = kw.coord0
        elif not ss.optSharedColVgpr or (d1 == vc1 == 0):
            # not share mode or first row always does the address calc math:

            if self.coordOffset0 == 0:
                self.coord0Vgpr = kw.coord0
            elif self.coordOffset0 <= 64:
                self.coord0Vgpr = tmpVgpr
                module.addInst("_v_add_co_u32", vgpr(self.coord0Vgpr), self.kernelWriter.vcc, vgpr(kw.coord0), self.coordOffset0, \
                          "coord0.1: coord0 += d0*sg0*VW + vc0")
            else:
                self.coord0Vgpr = tmpVgpr
                module.addInst("s_mov_b32", sgpr(tmpS01), self.coordOffset0, "coordOffset0 d0=%u vc0=%u"%(d0, vc0))
                module.addInst("_v_add_co_u32", vgpr(self.coord0Vgpr), self.kernelWriter.vcc, vgpr(kw.coord0), sgpr(tmpS01), \
                          "coord0.2: coord0 += d0*sg0*VW + vc0")

            if self.newCoord1:
                if not kernel["BufferStore"] or updateCoord1:
                    if self.rowInc== 0:
                        None
                    elif self.rowInc <= 64:
                        # rowInc fits in instruction:
                        module.addInst("_v_add_co_u32", vgpr(self.coord1Vgpr), self.kernelWriter.vcc, \
                                  vgpr(self.kernelWriter.coord1), self.rowInc, \
                                  "coord1.1: coord1Vgpr += d1*sg1*VW + vc1")
                    else:
                        module.addInst("s_mov_b32", sgpr(tmpS01), self.rowInc, "rowInc d1=%u vc1=%u"%(d0, vc0))
                        module.addInst("_v_add_co_u32", vgpr(self.coord1Vgpr), self.kernelWriter.vcc, \
                                  vgpr(self.kernelWriter.coord1), sgpr(tmpS01), \
                                  "coord1.2: coord1 += d1*sg1*VW + vc1")
        return module

    # storeChar is 'C' or 'D'
    # elementVgpr is coord0Vgpr*strideCD0, or optimized to just coord0Vgpr if strideCD0 is unit const
    def emitExtractAndScalePackedDims(self, kernel, ss, tmpVgpr, storeChar):
        module = Code.Module("emitExtractAndScalePackedDims")
        kw = self.kernelWriter
        packedIndices = kernel["PackedC0IndicesX"]
        packedBits = self.coord0Vgpr # start with coord0, will move to temp below
        rowPtr = kw.cinRowPtr if (storeChar == 'C') else kw.coutRowPtr
        addrVgpr = self.addrCVgpr if (storeChar == 'C') else self.addrDVgpr

        for i,idx in enumerate(packedIndices[:-1]):
            # vgprTmp assignments:
            #   - tmp+0 may be the incoming packed coordinate 0, used on replay too
            #   - tmp+1 is DIV output
            #   - tmp+2 is scratch
            idxChar= globalParameters["IndexChars"][idx]
            module.addText(kw.comment1("extract %s"%kw.sizeRef(idx)))
            assert(tmpVgpr+1 != packedBits) # bad since we still need packedBits below for remainder (can't overwrite here)
            module.addInst("V_MAGIC_DIV", \
                           tmpVgpr+1, vgpr(packedBits), sgpr("MagicNumberSize%s"%idxChar), \
                           sgpr("MagicShiftSize%s"%idxChar), sgpr("MagicAbitSize%s"%idxChar) if kernel["MagicDivAlg"]==2 else "0", "")
            # tmpVgpr+1 returns the quotient, tmpVgpr+2 is overwritten

            # compute remainder, packedBits % sizeIdx - this is the 'extracted' index that must be scaled
            # remainder is mul and sub
            module.addInst("v_mul_lo_u32", vgpr(tmpVgpr+2), vgpr(tmpVgpr+1), kw.sizeRef(idx), \
                           "remainder part 1")
            module.addInst("_v_sub_u32", vgpr(tmpVgpr+2), vgpr(packedBits), vgpr(tmpVgpr+2),
                           "remainder part 2")

            if i==0:
                module.addInst("v_mul_lo_u32", vgpr(addrVgpr), vgpr(tmpVgpr+2), \
                          kw.strideRef(storeChar, idx), "addrCalc <- scaled extracted dim")
            else:
                module.addInst("v_mul_lo_u32", vgpr(tmpVgpr+2), vgpr(tmpVgpr+2), \
                          kw.strideRef(storeChar, idx), "scale extracted dim")
                module.addInst("_v_add_u32", vgpr(addrVgpr), vgpr(addrVgpr), \
                          vgpr(tmpVgpr+2), "addrCalc += scaled extracted dim ")

            if i < len(packedIndices)-2:
                # TODO - might be able to eliminate this
                module.addInst("v_mov_b32", vgpr(tmpVgpr+0), vgpr(tmpVgpr+1), \
                          "Copy remaining bits for next divide")
                packedBits = tmpVgpr+0

        if len(packedIndices)>1:
            # if we unpacked something, then scale it to BPE
            module.addText(kw.comment1("extract final %s"%kw.sizeRef(packedIndices[-1])))
            module.addInst("v_mul_lo_u32", vgpr(tmpVgpr+2), vgpr(tmpVgpr+1), \
                      kw.strideRef(storeChar, packedIndices[-1]), "scale final extracted dim")
            module.addInst("_v_add_u32", vgpr(addrVgpr), vgpr(addrVgpr), \
                      vgpr(tmpVgpr+2), "addrCalc += scaled extracted dim ")

            module.addInst("_v_add_lshl_u32", vgpr(addrVgpr), \
                      vgpr(rowPtr), \
                      vgpr(addrVgpr), \
                      hex(log2(kw.bpeCexternal)), \
                      "packed: add rowPtr and scaleToBpe")

        return module

    def emitScaleToBpe(self, kernel, ss, tmpVgpr, singleUpdate, tc):
        """
        Needs 3 temporary VGPRs
        """

        module = Code.Module("emitScaleToBpe")
        kw = self.kernelWriter
        (d1,d0,vc1,vc0) = self.element
        rowPtr = kw.cinRowPtr if (tc == 'C') else kw.coutRowPtr
        addrVgpr = self.addrCVgpr if (tc == 'C') else self.addrDVgpr
        # set when we generate code that updates the address
        # optSingleColVgpr and optSharedColVgpr attempt to minimize these updates
        updatedAddr = False

        # scale and set final address:
        stride0 = kw.strideRef(tc, 0)
        if kw.isConstUnitStride(stride0):
            elementVgpr = self.coord0Vgpr
        else:
            module.addInst("v_mul_lo_u32", \
                vgpr(addrVgpr), \
                vgpr(self.coord0Vgpr), \
                stride0, \
                "scale element by non-unit stride")
            elementVgpr = addrVgpr

        if ss.optSingleColVgpr:
            # This is first element in the first batch, create a byte address that will
            # be re-used by subsequent elements:
            # if this element is firstInBatch - may need to set up a bpe-scaled row pointer for the batch:
            #  - need row-ptr start of each batch
            assert (kw.coord0 == self.coord0Vgpr) # elementAddr assignment above assumes these are the same
            if singleUpdate:
                updatedAddr = True
                singleColAddrUpdated = ss.singleColCAddrUpdated if (tc == 'C') else ss.singleColDAddrUpdated
                if not singleColAddrUpdated or not ss.optSrdIncForRow:
                    if tc == 'C':
                        ss.singleColCAddrUpdated = True
                    else:
                        ss.singleColDAddrUpdated = True
                    module.addInst("_v_add_lshl_u32", \
                      vgpr(addrVgpr), \
                      vgpr(rowPtr), \
                      vgpr(elementVgpr), \
                      hex(log2(kw.bpeCexternal)), \
                      "optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=%d, coord0Vgpr=%d"%(kw.coord0, self.coord0Vgpr))
        elif ss.optSharedColVgpr:
            # Need an address calculation for the first address in each row:
            if d1==0 and vc1==0:
                packedIndices = kernel["PackedC0IndicesX"]
                if len(packedIndices) > 1:
                    updatedAddr = True
                    module.addCode(self.emitExtractAndScalePackedDims(kernel, ss, tmpVgpr, tc))
                else:
                    updatedAddr = True
                    module.addInst("_v_add_lshl_u32", \
                      vgpr(addrVgpr), \
                      vgpr(rowPtr), \
                      vgpr(elementVgpr), \
                      hex(log2(kw.bpeCexternal)), \
                      "optSharedColVgpr scaleToBpe for first row: col addr <- cinRowPtr + coord0, scaled by BPE")
        else:
            # Generate final address calculation (to bytes) for each element
            # The unpacking takes 8-10 instructions so could be worth optimizing someday :
            # each col has same offset so could create a class to hold column-specific state including
            # the byte address offset for that col and the mask in/out.
            packedIndices = kernel["PackedC0IndicesX"]
            if len(packedIndices) > 1:
                updatedAddr = True
                module.addCode(self.emitExtractAndScalePackedDims(kernel, ss, tmpVgpr, tc))
            else:
                updatedAddr = True
                module.addInst("_v_add_lshl_u32", \
                    vgpr(addrVgpr), \
                    vgpr(rowPtr), \
                    vgpr(elementVgpr), \
                    hex(log2(kw.bpeCexternal)), \
                    "scaleToBpe: accumulate d0 lower and *= bpe into Cin addr")

        # if not optSrdIncForRow then we may have moved the row pointer
        # and depending on paths above may not have refreshed addrVgpr already.
        # if so - do it here:
        if self.rowIncDirtyRowPtr and not updatedAddr:
            module.addInst("_v_add_lshl_u32", \
              vgpr(addrVgpr), \
              vgpr(rowPtr), \
              vgpr(kw.coord0), \
              hex(log2(kw.bpeCexternal)), \
              "scaleToBpe: Update address with new rowPtr")

        return module

    def edgeProtectCode(self, kernel, edge, beta, atomic, mask, tmpSgpr):
        """
        Generate code to protect address offset in edge case
        """

        module = Code.Module("edgeProtectCode")
        kw = self.kernelWriter
        tmpS01 = tmpSgpr
        tmpS23 = tmpSgpr+self.kernelWriter.laneSGPRCount

        laneSGPRCount = self.kernelWriter.laneSGPRCount
        wavefrontSize = kernel["WavefrontSize"]

        # Now do the edge check and compute the address in bytes:
        if kernel["BufferStore"]:
            if edge and (not kernel["StoreRemapVectorWidth"] or (kernel["StoreRemapVectorWidth"] and beta)):
                # Set address to -1 if OOB on either dimension
                # and only check the x/coord0 index here, save a couple inst
                sizeBoundary = [0,0]
                sizeBoundary[0] = \
                    sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
                    else kw.sizeRef(kernel["ProblemType"]["Index0"])
                sizeBoundary[1] = \
                    sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
                    else kw.sizeRef(kernel["ProblemType"]["Index1"])

                module.addInst("v_cmp_lt_u32", sgpr(tmpS01,laneSGPRCount), vgpr(self.coord0Vgpr), sizeBoundary[0], "coord0 < size0" )
                module.addInst("v_cmp_lt_u32", sgpr(mask,laneSGPRCount), vgpr(self.coord1Vgpr), sizeBoundary[1], "coord1 < size1" )
                module.addInst("s_and_b{}".format(wavefrontSize), sgpr(mask,laneSGPRCount), sgpr(tmpS01,laneSGPRCount), sgpr(mask,laneSGPRCount), "in0 && in1" )
        else:
            module.addInst("v_cmp_lt_u32", sgpr(tmpS01,laneSGPRCount), vgpr(self.coord0Vgpr), sgpr("SizesFree+0"), "coord0 < size0" )
            module.addInst("v_cmp_lt_u32", sgpr(tmpS23,laneSGPRCount), vgpr(self.coord1Vgpr), sgpr("SizesFree+1"), "coord1 < size1" )
            module.addInst("s_and_b{}".format(wavefrontSize),  sgpr(mask,laneSGPRCount), sgpr(tmpS01,laneSGPRCount), sgpr(tmpS23,laneSGPRCount), "in0 && in1" )

            if (beta or atomic):
                module.addInst("s_mov_b{}".format(wavefrontSize), self.kernelWriter.exec, sgpr(mask,laneSGPRCount), "sgprs -> exec" )

        return module

    # TODO - mask should be part of AddrCalc state not passed as parm
    def emitAddressSetupCode(self, kernel, ss, tmpVgpr, tmpS01, edge, beta, atomic, elementIdx, addrVgpr):
        """
        Generate code to set up the address vgpr
        Input:
          tmpVgpr : two temp vgprs
        Output:
          Returns kStr with appropriate setup code
          Sets self.coord0Vgpr with vgpr that contains the coord0 for this element.  This enables
            optimization - if no setup code is required the coord0 can be the input.
        """

        module = Code.Module("emitAddressSetupCode")
        kw = self.kernelWriter

        updateCoord1 = (edge or len(kernel["PackedC1IndicesX"]) > 1)
        module.addCode(self.emitAddressCoordIncrement(kernel, ss, tmpVgpr, tmpS01, updateCoord1))

        # calculate flat load offset
        if not kernel["BufferStore"]:
            # flat: in-bounds exec mask
            # global offset macro (requires 3 tmpVgpr)
            # final address = C + index*bytes
            inst = Code.Inst("GLOBAL_OFFSET_C", "") # FIXME: Workaround
            params = ["%u" % addrVgpr]
            for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
                if i == kernel["ProblemType"]["Index0"]:
                    params.append("%s" % (self.coord0Vgpr))
                elif i == kernel["ProblemType"]["Index1"]:
                    params.append("%s" % (self.coord1Vgpr))
                else: # just a group index
                    params.append("sgprWorkGroup%u"%i)
            params.append("%s" % (tmpVgpr+2))
            inst.params = params
            module.addCode(inst)
            module.addInst("v_mov_b32", vgpr(tmpVgpr+2), vgpr(addrVgpr+0), "temp store offset 0")
            module.addInst("v_mov_b32", vgpr(tmpVgpr+3), vgpr(addrVgpr+1), "temp store offset 1")

        # Move the row ptr VGPR
        # optSrdIncForRow moves the SRD so don't move here
        if not ss.optSrdIncForRow and kernel["BufferStore"]:
            if self.rowInc > 0:
                self.rowIncDirtyRowPtr = 1
                #assert (not kernel["ProblemType"]["UseInitialStridesCD"])
                module.addText(kw.comment("Fix for UseInitialStridesCD, emitAddressSetupCode"))

                if len(kernel["PackedC1IndicesX"]) == 1:
                    strideChar = self.kernelWriter.indexChars[kernel["PackedC1IndicesX"][0]]
                    module.addCode(self.addScaled(vgpr(kw.cinRowPtr),  vgpr(kw.cinRowPtr),  \
                              sgpr("StrideC%s"%strideChar), self.rowInc, tmpS01, "ROWINC- Move cinRowPtr to next row"))
                    module.addCode(self.addScaled(vgpr(kw.coutRowPtr), vgpr(kw.coutRowPtr), \
                              sgpr("StrideD%s"%strideChar), self.rowInc, tmpS01, "Move coutRowPtr to next row"))
                elif len(kernel["PackedC1IndicesX"]) > 1:
                    module.addCode(kw.extractPackedCoord1ToRowStart(kernel, kernel["PackedC1IndicesX"] , self.coord1Vgpr, 'D'))

        # Shift Pointer for MFMA:
        #   For MFMA shift pointer, correct data is stored in another thread.
        #   Therefore, MFMA cannot use v_mov to amend store data
        #   It needs to modify the coord1 of thread directly.
        if (not kernel["SourceSwap"]) and (not kernel["GuaranteeNoPartialB"]) and kw.readTileDimVectorB and kernel["EnableMatrixInstruction"] and edge:
            (d1,d0,vc1,vc0) = self.element
            if (d1 == vc1 == d0 == vc0 == 0) or self.newCoord1:
                sgprCnt = self.kernelWriter.laneSGPRCount
                waveSize = kernel["WavefrontSize"]
                packedC1 = kernel["PackedC1IndicesX"]
                strideC1 = "StrideC%s" % (kw.indexChars[packedC1[0]])
                strideD1 = "StrideD%s" % (kw.indexChars[packedC1[0]])

                module.addText(kw.comment("shift vector components d1"))
                vw = kernel["GlobalLoadVectorWidthB"]
                vTmp1 = tmpVgpr
                vTmp2 = tmpVgpr+1
                sTmp1 = tmpS01
                sTmp2 = tmpS01+sgprCnt
                # check conditions
                module.addInst("v_bfi_b32", vgpr(vTmp1), vw-1, 0, vgpr(self.coord1Vgpr), "coord1 & ~(vw-1)")
                module.addInst("v_bfi_b32", vgpr(vTmp2), vw-1, 0, sgpr("SizesFree+%u"%kw.tPB["idx"]), "sizeFree & ~(vw-1)")
                module.addInst("v_cmp_eq_u32", sgpr(sTmp1,sgprCnt), vgpr(vTmp1), vgpr(vTmp2), "if coord1 is in edge glvw")
                module.addInst("v_and_b32", vgpr(vTmp2), sgpr("SizesFree+%u"%kw.tPB["idx"]), vw-1, "sizeFree mod VW")
                module.addInst("v_cmp_gt_u32", sgpr(sTmp2,sgprCnt), vgpr(vTmp2), 0, "this problem is not multiple size of glvw")
                module.addInst("s_and_b{}".format(waveSize), sgpr(sTmp1,sgprCnt), sgpr(sTmp1,sgprCnt), sgpr(sTmp2,sgprCnt), "AND both conditions")
                # calculate new coord
                module.addInst("_v_add_u32", vgpr(vTmp1), vgpr(self.coord1Vgpr), vgpr(vTmp2), "shift coord1")
                module.addInst("v_bfi_b32", vgpr(vTmp1), vw-1, vgpr(vTmp1), sgpr("SizesFree+%u"%kw.tPB["idx"]), "new coord1 = (shift coord1 & (vw-1)) |  (sizeFree & ~(vw-1))")
                module.addInst("_v_sub_i32", vgpr(vTmp2), vgpr(vTmp1), vgpr(self.coord1Vgpr), "shift how many column")
                module.addInst("v_cndmask_b32", vgpr(self.coord1Vgpr), vgpr(self.coord1Vgpr), vgpr(vTmp1), \
                              sgpr(sTmp1,sgprCnt), "set new coord1 if meet conditions" )

                module.addInst("v_mad_i32_i24", vgpr(vTmp1), sgpr(strideC1), vgpr(vTmp2), vgpr(kw.cinRowPtr), \
                             "new rowStart address += shift column * StridesC")
                module.addInst("v_cndmask_b32", vgpr(kw.cinRowPtr), vgpr(kw.cinRowPtr), vgpr(vTmp1), sgpr(sTmp1,sgprCnt), \
                             "set new rowStart if meet conditions" )
                module.addInst("v_mad_i32_i24", vgpr(vTmp1), sgpr(strideD1), vgpr(vTmp2), vgpr(kw.coutRowPtr), \
                             "new rowStart address += shift column * StridesD")
                module.addInst("v_cndmask_b32", vgpr(kw.coutRowPtr), vgpr(kw.coutRowPtr), vgpr(vTmp1), sgpr(sTmp1,sgprCnt), \
                             "set new rowStart if meet conditions" )

                if kernel["StoreRemapVectorWidth"]:
                    ldsPad = max(kernel["StoreRemapVectorWidth"],kernel["MIOutputVectorWidth"])
                    module.addInst("v_mov_b32", vgpr(vTmp1), hex((kernel["MacroTile0"]+ldsPad)*kw.bpeCexternal), \
                                "lds byte stride = (MT0 + PAD) * bpe")
                    module.addInst("v_mad_i32_i24", vgpr(vTmp1), vgpr(vTmp1), vgpr(vTmp2), vgpr(kw.storeRemapLW), \
                                "new lds write address += shift column * Lds byte Stride")
                    module.addInst("v_cndmask_b32", vgpr(kw.storeRemapLW), vgpr(kw.storeRemapLW), vgpr(vTmp1), \
                                  sgpr(sTmp1,sgprCnt), "set new rowStart if meet conditions" )

                module.addSpaceLine()

        return module

    def emitLdChange(self, kernel, ss, tc, edge, beta, mask, singleUpdate, tmpVgpr, addrVgpr, BufAddr):
        """
        Generate code for final C read/D write address
        """

        laneSGPRCount = self.kernelWriter.laneSGPRCount

        module = Code.Module("emitLdChange")
        if kernel["BufferStore"]:
            module.addCode(self.emitScaleToBpe(kernel, ss, tmpVgpr, singleUpdate, tc))
            if edge and (not kernel["StoreRemapVectorWidth"] or (kernel["StoreRemapVectorWidth"] and beta)):
                module.addInst("v_cndmask_b32", vgpr(addrVgpr), -1, vgpr(addrVgpr), \
                               sgpr(mask,laneSGPRCount), "LD%s clip if OOB. offset" % tc )
        else:
            # store a copy of the offset in 2 of the tmpVgpr for D
            module.addInst("_v_add_co_u32",  vgpr(addrVgpr+0), self.kernelWriter.vcc, vgpr(BufAddr+0), vgpr(tmpVgpr+2), \
                           "addrVgpr = C(D) + index*bytes (lo)" )
            module.addInst("_v_addc_co_u32", vgpr(addrVgpr+1), self.kernelWriter.vcc, vgpr(BufAddr+1), vgpr(tmpVgpr+3), \
                           self.kernelWriter.vcc, "addrVgpr = C(D) + index*bytes (hi)")
        return module

    def incrementToNextRow(self, kernel, tc, ss, stmp):
        """
        Generate code to move to the next row(s)
        If optSrdIncForRow, this will move the SRD forward
        If not, this could generate some other instructions
        """

        module = Code.Module("incrementToNextRow")
        numRows = self.rowInc
        tmpBpe = self.kernelWriter.bpeCexternal
        if ss.optSrdIncForRow:
            if numRows:
                packedC1 = kernel["PackedC1IndicesX"]
                assert(len(packedC1) == 1)  # would need to extract each dim and scale
                strideCD1 = "Stride%s%s"%(tc,self.kernelWriter.indexChars[packedC1[0]])
                if numRows > 1:
                    module.addInst("s_mul_i32", sgpr(stmp), \
                                   sgpr(strideCD1), \
                                   numRows*tmpBpe, \
                                   "scale Stride%s *= numRows(%u) * bpe"%(tc,numRows))
                else:
                    module.addInst("s_lshl_b32 ", \
                          sgpr(stmp), \
                          sgpr(strideCD1), \
                          log2(tmpBpe), \
                          "incToNextRow: Scale by BPE")

                module.addInst("s_add_u32 ", \
                     sgpr("Srd%s+0"%(tc)), \
                     sgpr("Srd%s+0"%(tc)), \
                     sgpr(stmp), \
                     "incToNextRow: gra SRD += inc(lower)" )
                module.addInst("s_addc_u32 ", \
                     sgpr("Srd%s+1"%(tc)), \
                     sgpr("Srd%s+1"%(tc)), \
                     0, \
                     "incToNextRow: gra SRD += inc(upper)" )

            None

        return module
