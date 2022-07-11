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

from ..Component import PackData
from ..Code import Module
from ..DataType import DataType
from ..AsmUtils import vgpr, sgpr, SaturateCastType, saturateCastInt

class PackData_F16(PackData):
    kernel = {"ProblemType": {"DestDataType": DataType(DataType.half)}}
    def __call__(self, gwvw, elementSumIdx, inputPrefix=""):
        assert (gwvw % 2 == 0)
        module = Module("PackData F16")
        formatting = (inputPrefix + "%u") if inputPrefix else "%u"
        for vi in range(0, gwvw):
            sumIdxV = elementSumIdx + vi
            module.addInst("v_cvt_f16_f32", vgpr(formatting%sumIdxV), vgpr(formatting%sumIdxV), "convert C to fp16")
            if vi%2 == 1:
              d = elementSumIdx + vi//2
              module.addInst("v_pack_b32_f16", vgpr(d), vgpr(formatting%(sumIdxV-1)), vgpr(formatting%sumIdxV), "Pack with neighbor" )
        return module

class PackData_BF16(PackData):
    kernel = {"ProblemType": {"DestDataType": DataType(DataType.bfloat16)}}
    def __call__(self, gwvw, elementSumIdx, bf16CVTVgprStruct, tmpS01, laneSGPRC, inputPrefix=""):
        assert (gwvw % 2 == 0)

        vgprBf16Temp = bf16CVTVgprStruct.vgprBf16Temp
        vgprBf16Inc = bf16CVTVgprStruct.vgprBf16Inc
        vgprFp32Nan = bf16CVTVgprStruct.vgprFp32Nan
        vgprBf16Mask = bf16CVTVgprStruct.vgprBf16Mask

        module = Module("PackData BF16")
        formatting = (inputPrefix + "%u") if inputPrefix else "%u"
        for vi in range(0, gwvw):
            sumIdxV = elementSumIdx + vi

            module.addInst("v_cmp_u_f32", sgpr(tmpS01,laneSGPRC), vgpr(formatting%sumIdxV), vgpr(formatting%sumIdxV), "check Nan" )
            module.addInst("v_bfe_u32", vgpr(vgprBf16Temp), vgpr(formatting%sumIdxV), "16", "1", "Non-Nan case: store lsb of bf16" )
            module.addInst("v_add3_u32", vgpr(vgprBf16Temp), vgpr(formatting%sumIdxV), vgpr(vgprBf16Temp), vgpr(vgprBf16Inc), "Non-Nan case: add lsb and the increment for rounding" )
            module.addInst("v_cndmask_b32", vgpr(formatting%sumIdxV), vgpr(vgprBf16Temp), vgpr(vgprFp32Nan), sgpr(tmpS01,laneSGPRC), "" )
            if vi%2 == 0:
              module.addInst("v_lshrrev_b32", vgpr(formatting%sumIdxV), "16", vgpr(formatting%sumIdxV), "convert C to bf16" )
            elif vi%2 == 1:
              d = elementSumIdx + vi//2
              module.addInst("v_and_or_b32", vgpr(d), vgpr(formatting%sumIdxV), vgpr(vgprBf16Mask), vgpr(formatting%(sumIdxV-1)), "pack two bf16 to dword")
        return module

class PackData_INT8(PackData):
    kernel = {"ProblemType": {"DestDataType": DataType(DataType.int8)}}
    def __call__(self, gwvw, elementSumIdx, tmpVgpr, tmpS01, SaturateTypeInt8 = SaturateCastType.NORMAL, inputPrefix=""):
        assert (gwvw % 4 == 0)
        module = Module("PackData int8")
        formatting = (inputPrefix + "%u") if inputPrefix else "%u"
        for vi in range(0, gwvw):
            sumIdxV = elementSumIdx + vi
            if vi%4 == 0:
              d = elementSumIdx + vi//4
              for i in range(0, 4):
                module.addCode(saturateCastInt(sumIdxV+i, tmpVgpr, tmpS01, -128, 127, type=SaturateTypeInt8, initGpr=(i%4 == 0)))
              module.addInst("v_lshlrev_b16", vgpr(formatting%(sumIdxV+1)), 8, vgpr(formatting%(sumIdxV+1)), "" )
              module.addInst("v_lshlrev_b16", vgpr(formatting%(sumIdxV+3)), 8, vgpr(formatting%(sumIdxV+3)), "" )
              module.addInst("v_or_b32", vgpr(formatting%(sumIdxV)), vgpr(formatting%(sumIdxV)), vgpr(formatting%(sumIdxV+1)),
                             "dst_sel:DWORD", "dst_unused:UNUSED_PAD", "src0_sel:BYTE_0", "src1_sel:DWORD", "" )
              module.addInst("v_or_b32", vgpr(formatting%(sumIdxV+1)), vgpr(formatting%(sumIdxV+2)), vgpr(formatting%(sumIdxV+3)),
                             "dst_sel:WORD_1", "dst_unused:UNUSED_PAD", "src0_sel:BYTE_0", "src1_sel:DWORD", "" )
              module.addInst("v_or_b32", vgpr(d), vgpr(formatting%(sumIdxV)), vgpr(formatting%(sumIdxV+1)),
                             "dst_sel:DWORD", "dst_unused:UNUSED_PAD", "src0_sel:WORD_0", "src1_sel:DWORD", "" )
        return module
