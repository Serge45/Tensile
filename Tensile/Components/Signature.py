################################################################################
# Copyright 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
from ..Component import Signature
from ..Common import globalParameters, gfxName

from math import ceil

class SignatureCOV3(Signature):
    kernel = {"CodeObjectVersion": "V3"}

    def __call__(self, writer):
        kernel = writer.kernel

        signature = Code.Signature("v3", self.commentHeader(), writer.kernelName)

        # kern arg size
        kernArgReg = 0
        kernArgReg += 3*writer.rpga
        kernArgReg += max(1,int(writer.bpeAB/4)) # alpha
        if kernel["ProblemType"]["UseBeta"]:
            kernArgReg += max(1,int(writer.bpeCexternal/4)) # beta
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsA"]) # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsB"]) # strides
        if not kernel["ProblemType"]["UseInitialStridesAB"]:
            kernArgReg -= 2 # strides
        if not kernel["ProblemType"]["UseInitialStridesCD"]:
            kernArgReg -= 2 # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesSummation"]
        kernArgReg += kernel["ProblemType"]["NumIndicesC"]
        if globalParameters["DebugKernel"]:
            kernArgReg += writer.rpga # debug buffer
        # kernArgBytes = kernArgReg * 4 # bytes/reg

        # register allocation
        totalVgprs = writer.vgprPool.size()
        totalSgprs = writer.sgprPool.size()

        # accumulator offset for Unified Register Files
        vgprCount = totalVgprs
        agprStart = None
        if writer.archCaps["ArchAccUnifiedRegs"]:
            agprStart = ceil(totalVgprs/8)*8
            vgprCount = agprStart + writer.agprPool.size()

        group_segment_size = kernel["LdsNumElements"] * writer.bpeAB

        sgprWgZ = 1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0
        signature.addKernelDescriptor(Code.SignatureKernelDescriptorV3(target=gfxName(writer.version),
                                                                       accumOffset=agprStart,
                                                                       totalVgprs=vgprCount,
                                                                       totalSgprs=totalSgprs,
                                                                       groupSegSize=group_segment_size,
                                                                       reserved="",
                                                                       hasWave32=writer.archCaps["HasWave32"],
                                                                       waveFrontSize=kernel["WavefrontSize"],
                                                                       sgprWg=[1, 1, sgprWgZ],
                                                                       vgprWi=0,
                                                                       name=writer.kernelName))

        signature.addOptConfigComment(tt=[kernel["ThreadTile0"], kernel["ThreadTile1"]],
                                      sg=[kernel["SubGroup0"], kernel["SubGroup1"]],
                                      vw=kernel["VectorWidth"],
                                      glvwA=kernel["GlobalLoadVectorWidthA"],
                                      glvwB=kernel["GlobalLoadVectorWidthB"],
                                      d2lA=kernel["DirectToLdsA"],
                                      d2lB=kernel["DirectToLdsB"],
                                      useSgprForGRO=kernel["_UseSgprForGRO"])

        srcValueType = kernel["ProblemType"]["DataType"].toNameAbbrev()
        dstValueType = kernel["ProblemType"]["DestDataType"].toNameAbbrev()
        cptValueType = kernel["ProblemType"]["ComputeDataType"].toNameAbbrev()
        actValueType = kernel["ProblemType"]["DestDataType"].toNameAbbrev()
        cptByte = kernel["ProblemType"]["ComputeDataType"].numBytes()


        codeMeta = Code.SignatureCodeMetaV3(groupSegSize=group_segment_size,
                                            totalVgprs=totalVgprs,
                                            totalSgprs=totalSgprs,
                                            flatWgSize=(kernel["SubGroup0"] * kernel["SubGroup1"] * kernel["LocalSplitU"]),
                                            waveFrontSize=kernel["WavefrontSize"],
                                            name=writer.kernelName)

        if globalParameters["DebugKernel"]:
            codeMeta.addArg(8, "AddressDbg", "global_buffer",     "struct", "generic")
        codeMeta.addArg(    8,      "sizeC",      "by_value",        "u64")
        codeMeta.addArg(    8,      "sizeA",      "by_value",        "u64")
        codeMeta.addArg(    8,      "sizeB",      "by_value",        "u64")
        codeMeta.addArg(    8,          "D", "global_buffer", dstValueType, "generic")
        codeMeta.addArg(    8,          "C", "global_buffer", dstValueType, "generic")
        codeMeta.addArg(    8,          "A", "global_buffer", srcValueType, "generic")
        codeMeta.addArg(    8,          "B", "global_buffer", srcValueType, "generic")

        useSize = max(4, cptByte)
        codeMeta.addArg(useSize,    "alpha", "by_value", cptValueType)
        if kernel["ProblemType"]["UseBeta"]:
            codeMeta.addArg(useSize, "beta", "by_value", cptValueType)

        if ((kernel["ProblemType"]["ActivationType"] != 'none') and (kernel["_GlobalAccumulation"] != 'MultipleBuffer') \
            and kernel["ActivationFused"]):
          activationSize = max(4, kernel["ProblemType"]["DestDataType"].numBytes())
          activationValueType = actValueType
          for name in kernel["ProblemType"]["ActivationType"].getAdditionalArgStringList():
            codeMeta.addArg(activationSize,             name, "by_value", activationValueType)
          if kernel["ProblemType"]["ActivationType"] == 'all':
            codeMeta.addArg(             4,            "activationType", "by_value",               "u32")

        for i in range(0, writer.numSgprStridesD):
            codeMeta.addArg(             4,               "strideD%u"%i, "by_value",               "u32")

        for i in range(0, writer.numSgprStridesC):
            codeMeta.addArg(             4,               "strideC%u"%i, "by_value",               "u32")

        for i in range(0, writer.numSgprStridesA):
            codeMeta.addArg(             4,               "strideA%u"%i, "by_value",               "u32")

        for i in range(0, writer.numSgprStridesB):
            codeMeta.addArg(             4,               "strideB%u"%i, "by_value",               "u32")

        for i in range(0, writer.numSgprSizesFree):
            codeMeta.addArg(             4,             "SizesFree%u"%i, "by_value",               "u32")

        for i in range(0, writer.numSgprSizesSum):
            codeMeta.addArg(             4,              "SizesSum%u"%i, "by_value",               "u32")

        for idxChar in kernel["PackedC0IdxChars"][:-1]:
            codeMeta.addArg(             4, "MagicNumberSize%s"%idxChar, "by_value",               "u32")
            codeMeta.addArg(             4,  "MagicShiftSize%s"%idxChar, "by_value",               "u32")

        codeMeta.addArg(                 4,           "OrigStaggerUIter", "by_value",              "i32")

        codeMeta.addArg(                 4,             "NumWorkGroups0", "by_value",              "u32")
        codeMeta.addArg(                 4,             "NumWorkGroups1", "by_value",              "u32")

        codeMeta.addArg(                 4,              "NumFullBlocks", "by_value",              "u32")
        codeMeta.addArg(                 4,              "WgmRemainder1", "by_value",              "u32")
        codeMeta.addArg(                 4,   "MagicNumberWgmRemainder1", "by_value",              "u32")

        codeMeta.addArg(                 4,                    "OffsetD", "by_value",              "u32")
        codeMeta.addArg(                 4,                    "OffsetC", "by_value",              "u32")
        codeMeta.addArg(                 4,                    "OffsetA", "by_value",              "u32")
        codeMeta.addArg(                 4,                    "OffsetB", "by_value",              "u32")

        codeMeta.addArg(                 4,                    "padding", "by_value",              "u32")

        signature.addCodeMeta(codeMeta)

        return signature
