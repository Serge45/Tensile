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
from ..Component import ComputeStoreVgprs
from ..AsmUtils import log2, staticMultiply, vgpr, sgpr, vectorStaticDivide, vectorStaticRemainder

class ComputeStoreVgprsMFMA(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": False}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        # writer.coord0
        # writer.coord1
        # writer.cinRowPtr  : C buffer coulmn offset
        # writer.coutRowPtr : D buffer coulmn offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1)
        tid1 = writer.vgprPool.checkOut(1)
        if kernel["BufferStore"]:
            writer.cinRowPtr    = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.coutRowPtr = writer.vgprPool.checkOut(1, "coutRowPtr")

        wave_id = writer.vgprPool.checkOut(1)

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpVgpr1 = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
        dummy    = writer.vgprPool.checkOut(1,"dummy")
        tmpSgpr  = writer.getTmpSgpr(1).idx()

        # constant
        MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
        MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

        # matrixInstM = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        matrixInstN = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]

        module = Code.Module("ComputeStoreVgprsMFMA")

        # coord 1 : wave part
        module.addCode(vectorStaticDivide(wave_id, "Serial", writer.kernel["WavefrontSize"], tmpVgpr1, tmpSgpr))
        module.addCode(vectorStaticDivide(tid1, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr))
        module.addInst("v_mul_lo_u32", vgpr(tid1), hex(MIBShape1), vgpr(tid1), "wave coordination offset 1")

        # coord 1 : thread part
        module.addCode(vectorStaticRemainder(dummy, tmpVgpr0, "Serial", matrixInstN, tmpVgpr1, tmpSgpr))
        module.addInst("_v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "coordination 1 = wave_id1 + tid1")

        # coord 1 : offset part
        packedC1 = kernel["PackedC1IndicesX"]
        strideC1 = "StrideC%s" % (writer.indexChars[packedC1[0]])
        strideD1 = "StrideD%s" % (writer.indexChars[packedC1[0]])
        module.addInst("v_mul_lo_u32", vgpr(writer.cinRowPtr), vgpr(tid1), sgpr(strideC1), " offset 1")
        module.addInst("v_mul_lo_u32", vgpr(writer.coutRowPtr), vgpr(tid1), sgpr(strideD1), " offset 1")

        # coord 0 : wave part
        module.addCode(vectorStaticRemainder(dummy, tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr))
        module.addInst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape0), vgpr(tmpVgpr0), "wave coordination offset 0")

        # coord 0 : thread part
        module.addCode(vectorStaticRemainder(dummy, tid0, "Serial", writer.kernel["WavefrontSize"], tmpVgpr1, tmpSgpr))
        module.addCode(vectorStaticDivide(tid0, tid0, matrixInstN, tmpVgpr1, tmpSgpr))
        module.addCode(staticMultiply(vgpr(tid0), vgpr(tid0), kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "thread0 * continuous_output"))
        module.addInst("_v_add_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), "coordination 0 = wave_id0 + tid0")

        wg0="WorkGroup0"
        wg1="WorkGroup1"

        # macro tile 0 part
        module.addInst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile0"], sgpr(wg0), "wgp0 * MT0")
        module.addInst("_v_add_u32", vgpr(tid0), sgpr(tmpSgpr), vgpr(tid0), "coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0")

        # macro tile 1 part
        module.addInst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile1"], sgpr(wg1), "wgp1 * MT1")
        module.addInst("_v_add_u32", vgpr(tid1), sgpr(tmpSgpr), vgpr(tid1), "coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1")

        # extract packed rowStart vgpr
        if len(packedC1) > 1:
          module.addCode(writer.extractPackedCoord1ToRowStart(kernel, packedC1, tid1, 'D'))

        # release resource
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(tmpVgpr1)
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            module.addCode(writer.storeRemapComputeStoreVgprs(kernel))

        writer.coord0 = tid0
        writer.coord1 = tid1

        return module

class ComputeStoreVgprsMFMASwap(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": True}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        # writer.coord0
        # writer.coord1
        # writer.cinRowPtr  : C buffer coulmn offset
        # writer.coutRowPtr : D buffer coulmn offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1)
        tid1 = writer.vgprPool.checkOut(1)
        if kernel["BufferStore"]:
            writer.cinRowPtr    = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.coutRowPtr = writer.vgprPool.checkOut(1, "coutRowPtr")

        wave_id = writer.vgprPool.checkOut(1)

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpVgpr1 = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
        dummy    = writer.vgprPool.checkOut(1,"dummy")
        tmpSgpr  = writer.getTmpSgpr(1).idx()

        # constant
        MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
        MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

        matrixInstM = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        # matrixInstN = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]

        module = Code.Module("ComputeStoreVgprsMFMASwap")

        module.addCode(vectorStaticDivide(wave_id, "Serial", writer.kernel["WavefrontSize"], tmpVgpr1, tmpSgpr))

        # coord 1 : wave part
        module.addCode(vectorStaticDivide(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr))
        module.addInst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape1), vgpr(tmpVgpr0), "wave coordination offset 1")

        # coord 1 : thread part
        module.addCode(vectorStaticRemainder(dummy, tid1, "Serial", writer.kernel["WavefrontSize"], tmpVgpr1, tmpSgpr))
        module.addCode(vectorStaticDivide(tid1, tid1, matrixInstM, tmpVgpr1, tmpSgpr))
        module.addCode(staticMultiply(vgpr(tid1), vgpr(tid1), kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "thread0 * continuous_output"))
        module.addInst("v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "coordination 1 = wave_id1 + tid1")
        if writer.allowLRVWforTLUandMI and writer.lrvwB > 1:
          module.addCode(staticMultiply(vgpr(tid1), vgpr(tid1), writer.lrvwB, sgpr(tmpSgpr), "coordination 1 *= lrvwB"))

        # coord 1 : offset part
        packedC1 = kernel["PackedC1IndicesX"]
        strideC1 = "StrideC%s" % (writer.indexChars[packedC1[0]])
        strideD1 = "StrideD%s" % (writer.indexChars[packedC1[0]])
        module.addInst("v_mul_lo_u32", vgpr(writer.cinRowPtr), vgpr(tid1), sgpr(strideC1), " offset 1")
        module.addInst("v_mul_lo_u32", vgpr(writer.coutRowPtr), vgpr(tid1), sgpr(strideD1), " offset 1")

        # coord 0 : wave part
        module.addCode(vectorStaticRemainder(dummy, tid0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr))
        module.addInst("v_mul_lo_u32", vgpr(tid0), hex(MIBShape0), vgpr(tid0), "wave coordination offset 0")

        # coord 0 : thread part
        module.addCode(vectorStaticRemainder(dummy, tmpVgpr0, "Serial", matrixInstM, tmpVgpr1, tmpSgpr))
        module.addInst("_v_add_lshl_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), log2(kernel["VectorWidth"]), "coordination 0 = wave_id0 + tid0")

        wg0="WorkGroup0"
        wg1="WorkGroup1"

        # macro tile 0 part
        module.addInst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile0"], sgpr(wg0), "wgp0 * MT0")
        module.addInst("v_add_u32", vgpr(tid0), sgpr(tmpSgpr), vgpr(tid0), "coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0")

        # macro tile 1 part
        module.addInst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile1"], sgpr(wg1), "wgp1 * MT1")
        module.addInst("v_add_u32", vgpr(tid1), sgpr(tmpSgpr), vgpr(tid1), "coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1")

        # release resource
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(tmpVgpr1)
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            module.addCode(writer.storeRemapComputeStoreVgprs(kernel))

        writer.coord0 = tid0
        writer.coord1 = tid1

        return module
