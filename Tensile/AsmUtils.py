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

from math import log
from enum import Enum
from .Code import HolderContainer, RegisterContainer, Module, Inst, Item
from .Common import printExit
import random
import string

########################################
# Format GPRs
########################################

def gpr(*args):
    gprType = args[0]
    args = args[1]
    if isinstance(args[0], Holder):
        idx = args[0].idx
        if len(args) == 1:
            return HolderContainer(gprType, idx, 1)
        elif len(args) == 2:
            return HolderContainer(gprType, idx, args[1])
    if isinstance(args[0], int):
        if len(args) == 1:
            return RegisterContainer(gprType, None, args[0], 1)
        elif len(args) == 2:
            return RegisterContainer(gprType, None, args[0], args[1])
    if isinstance(args[0], str):
        if len(args) == 1:
            return RegisterContainer(gprType, args[0], None, 1)
        elif len(args) == 2:
            return RegisterContainer(gprType, args[0], None, args[1])

def vgpr(*args):
    return gpr("v", args)

def sgpr(*args):
    return gpr("s", args)

def accvgpr(*args):
    return gpr("acc", args)

def mgpr(*args):
    return gpr("m", args)

class Holder:
    def __init__(self, idx):
        self.idx = idx

########################################
# Log 2
########################################

def log2(x):
    return int(log(x, 2) + 0.5)

########################################
# Compound instructions
########################################

# Perform 32-bit scalar mul and save 64-bit result in two SGPR
# src0 and src1 are 32-bit ints in scalar sgpr or small int constants (<64?))
# signed indicates if input and output data is signed
# return returns in dst0:dest (lower 32-bit in dst0, high 64-bit in dst1))
# Requires 2 tmp vgprs
def s_mul_int_64_32(hasSMulHi, dst0, dst1, src0, src1, signed, vtmp0, comment):
    module = Module("s_mul_int_64_32")
    sign = "i" if signed else "u"
    assert(dst1 != src0) # no worky since dst1 overwritten by first mul operations
    assert(dst1 != src1) # no worky since dst1 overwritten by first mul operations
    # the else path below has less restrictions but prefer consistency
    if hasSMulHi:
        module.addInst("s_mul_hi_{}32".format(sign), dst1, src0, src1, comment)
        module.addInst("s_mul_i32", dst0, src0, src1, comment)
    else:
        if type(src1) != 'str' or not src1.startswith("s"):
            # Swap operands, need a scalar sgpr in src1 (not a constant)
            t = src0
            src0 = src1
            src1 = t
        vtmp1 = vtmp0+1
        module.addInst("v_mov_b32", vgpr(vtmp0), src0, comment)
        module.addInst("v_mul_hi_{}32".format(sign), vgpr(vtmp1), vgpr(vtmp0), src1, comment)
        module.addInst("v_readfirstlane_b32", dst1, vgpr(vtmp1), comment)
        module.addInst("v_mul_lo_u32", vgpr(vtmp1), vgpr(vtmp0), src1, comment)
        module.addInst("v_readfirstlane_b32", dst0, vgpr(vtmp1), comment)
    return module

########################################
# Divide & Remainder
# quotient register, remainder register, dividend register, divisor, tmpVgprx2, tmpSgpr
########################################

def vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, doRemainder=True, comment=""):

    dComment = "%s = %s / %s"    % (vgpr(qReg), vgpr(dReg), divisor) if (comment=="") else comment
    rComment = "%s = %s %% %s" % (vgpr(rReg), vgpr(dReg), divisor) if (comment=="") else comment

    module = Module("vectorStaticDivideAndRemainder")
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        divisor_log2 = log2(divisor)
        module.addInst("v_lshrrev_b32", vgpr(qReg), divisor_log2, vgpr(dReg), dComment)
        if doRemainder:
            module.addInst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), rComment)
    else:
        """
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 7:
            shift = 32+3
        elif divisor >= 6:
            shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        """
        shift = 32+1
        magic = ((2**shift) // divisor) + 1
        module.addInst("s_mov_b32", sgpr(tmpSgpr), hex(magic), dComment)
        module.addInst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), dComment)
        module.addInst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), dComment)
        module.addInst("v_lshrrev_b64", vgpr(tmpVgpr,2), hex(shift), vgpr(tmpVgpr,2), dComment)
        module.addInst("v_mov_b32", vgpr(qReg), vgpr(tmpVgpr), dComment)
        if doRemainder:
            module.addInst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), rComment)
            module.addInst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(qReg), sgpr(tmpSgpr), rComment)
            module.addInst("_v_sub_u32", vgpr(rReg), vgpr(dReg), vgpr(tmpVgpr), rComment)
    return module

def vectorStaticDivide(qReg, dReg, divisor, tmpVgpr, tmpSgpr, comment=""):
    rReg = -1 # unused
    module = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, False, comment)
    module.name = "vectorStaticDivide (reg=-1)"
    return module

def vectorStaticRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, comment=""):
    if comment == "":
        comment = "%s = %s %% %s" % (vgpr(rReg), vgpr(dReg), divisor)

    module = Module("vectorStaticRemainder")
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        module.addInst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), comment)
    else:
        """
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 7:
            shift = 32+3
        elif divisor >= 6:
            shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        """
        shift = 32+1
        magic = ((2**shift) // divisor) + 1
        module.addInst("s_mov_b32", sgpr(tmpSgpr), hex(magic), comment)
        module.addInst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), comment)
        module.addInst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), comment)
        module.addInst("s_mov_b32", sgpr(tmpSgpr), hex(shift), comment)
        module.addInst("v_lshrrev_b64", vgpr(tmpVgpr,2), sgpr(tmpSgpr), vgpr(tmpVgpr,2), comment)
        module.addInst("v_mov_b32", vgpr(qReg), vgpr(tmpVgpr), comment)
        module.addInst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), comment)
        module.addInst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(qReg), sgpr(tmpSgpr), comment)
        module.addInst("_v_sub_u32", vgpr(rReg), vgpr(dReg), vgpr(tmpVgpr), comment)
    return module

# only used for loop unroll and GlobalSplitU
# doRemainder==0 : compute quotient only
# doRemainder==1 : compute quotient and remainder
# doRemainder==2 : only compute remainder (not quotient unless required for remainder)
# dreg == dividend
# tmpSgpr must be 2 SPGRs
# qReg and dReg can be "sgpr[..]" or names of sgpr (will call sgpr)
def scalarStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr, \
        doRemainder=1):

    assert (qReg != tmpSgpr)


    qRegSgpr = qReg if isinstance(qReg, RegisterContainer) and qReg.regType == 's' else sgpr(qReg)

    dRegSgpr = dReg if isinstance(dReg, RegisterContainer) and dReg.regType == 's' else sgpr(dReg)

    module = Module("scalarStaticDivideAndRemainder")
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        divisor_log2 = log2(divisor)
        if doRemainder != 2:
            module.addInst("s_lshr_b32", qRegSgpr, dRegSgpr, divisor_log2, \
                    "%s = %s / %u"%(qRegSgpr, dRegSgpr, divisor) )
        if doRemainder:
            module.addInst("s_and_b32", sgpr(rReg), (divisor-1), dRegSgpr, \
                    "%s = %s %% %u"%(sgpr(rReg), dRegSgpr, divisor) )
    else:
        """
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 6:
            shift = 32+3
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        """
        shift = 32+1
        magic = ((2**shift) // divisor) + 1
        magicHi = magic // (2**16)
        magicLo = magic & (2**16-1)

        module.addInst("s_mov_b32", sgpr(tmpSgpr+1), hex(0), "STATIC_DIV: divisior=%s"%divisor)
        module.addInst("s_mul_i32", sgpr(tmpSgpr+0), hex(magicHi), dRegSgpr, "tmp1 = dividend * magic hi")
        module.addInst("s_lshl_b64", sgpr(tmpSgpr,2), sgpr(tmpSgpr,2), hex(16), "left shift 16 bits")
        module.addInst("s_mul_i32", qRegSgpr, dRegSgpr, hex(magicLo), "tmp0 = dividend * magic lo")
        module.addInst("s_add_u32", sgpr(tmpSgpr+0), qRegSgpr, sgpr(tmpSgpr+0), "add lo")
        module.addInst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), hex(0), "add hi")
        module.addInst("s_lshr_b64", sgpr(tmpSgpr,2), sgpr(tmpSgpr,2), hex(shift), "tmp1 = (dividend * magic) << shift")
        module.addInst("s_mov_b32", qRegSgpr, sgpr(tmpSgpr), "quotient")
        if doRemainder:
            module.addInst("s_mul_i32", sgpr(tmpSgpr), qRegSgpr, hex(divisor), "quotient*divisor")
            module.addInst("s_sub_u32", sgpr(rReg), dRegSgpr, sgpr(tmpSgpr), "rReg = dividend - quotient*divisor")
    return module

########################################
# Scalar Magic Div
# product register, operand register, multiplier
########################################

# dividend is a symbol (constant or sgpr).  Used directly not inside automatic sgpr(..)
# dst is 2 consecutive SGPR
#   result returned in dst0. dst1 is used as a temp,
# dst[1] cannot be same as divident, dst[0] can be same as dividend and this can be useful
def scalarMagicDivExplicit(dst, dividend, magicNumber, magicAbit, magicShift):
    module = Module("scalarMagicDivExplicit")
    module.addComment1("dst1:0 = dividend(%s) / magicTag(%s)" % (dividend, magicNumber))
    module.addInst("s_mul_hi_u32", sgpr(dst+1), dividend, sgpr(magicNumber), "scalar magic div (magicnum)")
    module.addInst("s_mul_i32", sgpr(dst+0), dividend, sgpr(magicAbit), "scalar magic div (abit)")
    module.addInst("s_add_u32", sgpr(dst+0), sgpr(dst+0), sgpr(dst+1), "scalar magic div (combine)")
    module.addInst("s_lshr_b32", sgpr(dst+0), sgpr(dst+0), sgpr(magicShift), \
                   "scalar magic div (shift), quotient in s%s"%dst)
    return module

def scalarMagicDiv(dst, dividend, magicTag):
    return scalarMagicDivExplicit(dst, dividend,
                                  magicNumber="MagicNumberSize"+magicTag,
                                  magicAbit="MagicAbitSize"+magicTag,
                                  magicShift="MagicShiftSize"+magicTag)

########################################
# Multiply
# product register, operand register, multiplier
########################################

def staticMultiply(product, operand, multiplier, tmpSgpr=None, comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    module = Module("staticMultiply")
    if multiplier == 0:
        module.addInst("v_mov_b32", product, hex(multiplier), comment)
    elif ((multiplier & (multiplier - 1)) == 0): # pow of 2
        multiplier_log2 = log2(multiplier)
        if multiplier_log2==0 and product == operand:
            module.addInst("", comment + " (multiplier is 1, do nothing)")
        else:
            module.addInst("v_lshlrev_b32", product, hex(multiplier_log2), operand, comment)
    else:
        if product == operand:
            module.addInst("s_mov_b32", tmpSgpr, hex(multiplier), comment)
            module.addInst("v_mul_lo_u32", product, tmpSgpr, operand, comment)
        else:
            module.addInst("v_mov_b32", product, hex(multiplier), comment)
            module.addInst("v_mul_lo_u32", product, product, operand, comment)
    return module


########################################
# Multiply scalar for 64bit
# product register, operand register, multiplier
########################################

def scalarStaticMultiply(product, operand, multiplier, tmpSgpr=None, comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    if multiplier == 0:
            return Inst("s_mov_b64", product, hex(multiplier), comment)

    # TODO- to support non-pow2, need to use mul_32 and mul_hi_32 ?
    assert ((multiplier & (multiplier - 1)) == 0) # assert pow of 2

    multiplier_log2 = log2(multiplier)
    if multiplier_log2==0 and product == operand:
        return Inst("", comment + " (multiplier is 1, do nothing)")
    else:
        # notice that the src-order of s_lshl_b64 is different from v_lshlrev_b32.
        return Inst("s_lshl_b64", product, operand, hex(multiplier_log2), comment)

def sBranchIfZero(sgprName, computeDataType, tmpSgpr, laneSC, label, waveFrontSize, vcc):
    module = Module("sBranchIfZero")
    sgprStr = "s[{}]".format(sgprName)
    if computeDataType.isDoubleComplex():
        module.addInst("v_cmp_eq_f64", sgpr(tmpSgpr, laneSC), sgpr(sgprName, 2), 0.0, "%s.real == 0.0 ?" % sgprStr)
        sgprVar = "%s+2" % sgprName if isinstance(sgprName, str) else sgprName + 2
        module.addInst("v_cmp_eq_f64", vcc, sgpr(sgprVar, 2), 0.0, "%s.imag == 0.0 ?" % sgprStr)
        module.addInst(f"s_and_b{waveFrontSize}", sgpr(tmpSgpr, laneSC), vcc, sgpr(tmpSgpr, laneSC), "%s == 0 ?" % sgprStr)
        module.addInst(f"s_cmp_eq_u{waveFrontSize}", sgpr(tmpSgpr, laneSC), hex(0), "branch if %s == 0" % sgprStr)
        module.addInst("s_cbranch_scc0 %s" % (label.getLabelName()), "branch if %s == 0" % sgprStr)
    elif computeDataType.isDouble():
        module.addInst("v_cmp_eq_f64", vcc, sgpr(sgprName, 2), 0.0, "%s == 0.0 ?" % sgprStr)
        module.addInst("s_cbranch_vccnz %s" % (label.getLabelName()), "branch if %s == 0" % sgprStr)
    elif computeDataType.isSingleComplex():
        module.addInst("v_cmp_eq_f32", sgpr(tmpSgpr, laneSC), sgpr(sgprName), 0.0, "%s.real == 0.0f ?" % sgprStr)
        sgprVar = "%s+1" % sgprName if isinstance(sgprName, str) else sgprName + 1
        module.addInst("v_cmp_eq_f32", vcc, sgpr(sgprVar), 0.0, "%s.imag == 0.0f ?" % sgprStr)
        module.addInst(f"s_and_b{waveFrontSize}", sgpr(tmpSgpr, laneSC), vcc, sgpr(tmpSgpr, laneSC), "%s == 0 ?" % sgprStr)
        module.addInst(f"s_cmp_eq_u{waveFrontSize}", sgpr(tmpSgpr, laneSC), hex(0), "branch if %s == 0" % sgprStr)
        module.addInst("s_cbranch_scc0 %s" % (label.getLabelName()), "branch if %s == 0" % sgprStr)
    elif computeDataType.isSingle() or computeDataType.isHalf() or computeDataType.isBFloat16():
        module.addInst("v_cmp_eq_f32", vcc, sgpr(sgprName), 0.0, "%s == 0.0f ?" % sgprStr)
        module.addInst("s_cbranch_vccnz %s" % (label.getLabelName()), "branch if %s == 0" % sgprStr)
    elif computeDataType.isInt32(): # int32
        module.addInst("s_cmp_eq_u32", sgpr(sgprName), 0, "%s == 0 ?" % sgprStr)
        module.addInst("s_cbranch_scc1 %s" % (label.getLabelName()), "branch if %s == 0" % sgprStr)
    else:
      printExit("Unsupported compute data type: %s" % str(computeDataType))
    return module

########################################
# Saturate Cast Integer
########################################

class SaturateCastType(Enum):
    NORMAL = 1
    DO_NOTHING = 2
    UPPER = 3
    LOWER = 4

def saturateCastInt(sumIdxV, tmpVgpr, tmpSgpr, lowerBound, upperBound, type=SaturateCastType.NORMAL, initGpr=True):
    # SaturateCastType = 0, normal case
    # SaturateCastType = 1, do nothing
    # SaturateCastType = 2, upperbound only
    # SaturateCastType = 3, lowerbound only
    initGprStr = "with init gpr" if initGpr else "without init gpr"
    module = Module("SaturateCastInt %s"%(initGprStr))
    if type == SaturateCastType.NORMAL:
        tmpLowerBound = tmpSgpr
        tmpUpperBound = tmpVgpr
        if initGpr:
            lowerBoundHex = hex(lowerBound)
            upperBoundHex = hex(upperBound)
            module.addInst("s_movk_i32", sgpr(tmpLowerBound), lowerBoundHex, "%d"%lowerBound )
            module.addInst("v_mov_b32", vgpr(tmpUpperBound), upperBoundHex, "%d"%upperBound )
        module.addInst("v_med3_i32", vgpr("ValuC+%u"%(sumIdxV)), vgpr("ValuC+%u"%(sumIdxV)), sgpr(tmpLowerBound), vgpr(tmpUpperBound), "x= min(%d, max(%d, x))"%(upperBound, lowerBound) )
    elif type == SaturateCastType.DO_NOTHING:
        pass
    elif type == SaturateCastType.UPPER:
        module.addInst("v_min_i32", vgpr("ValuC+%u"%(sumIdxV)), upperBound, vgpr("ValuC+%u"%(sumIdxV)), "x = min(%d, x)"%upperBound )
    elif type == SaturateCastType.LOWER:
        module.addInst("v_max_i32", vgpr("ValuC+%u"%(sumIdxV)), lowerBound, vgpr("ValuC+%u"%(sumIdxV)), "x = max(%d, x)"%lowerBound )
    return module

def replacePlaceHolder(module, srcStr, dstStr):
    assert(isinstance(module, Item))
    if isinstance(module, Module):
        for item in module.items():
            replacePlaceHolder(item, srcStr, dstStr)
    elif isinstance(module, Inst):
        for param in module.params:
            param.replaceRegName(srcStr, dstStr)
    return module

########################################
# Label Manager
########################################

def magicGenerator(chars=(string.ascii_uppercase + string.digits)):
    return ''.join(random.choice(chars) for _ in range(16))

class LabelManager():
    def __init__(self):
        self.labelDict = dict()

    def addName(self, name):
        if name not in self.labelDict:
            self.labelDict[name] = 0
        else:
            self.labelDict[name] += 1

    def getUniqueName(self):
        name = magicGenerator()
        while 1:
            if name not in self.labelDict:
                break
            name = magicGenerator()
        return self.getName(name)

    def getUniqueNamePrefix(self, prefix):
        name = prefix + "_" + magicGenerator()
        while 1:
            if name not in self.labelDict:
                break
            name = prefix + "_" + magicGenerator()
        return self.getName(name)

    def getName(self, name):
        if name not in self.labelDict:
            self.labelDict[name] = 0
        return name + "_" + str(self.labelDict[name])

    def getNameInc(self, name):
        self.addName(name)
        return name + "_" + str(self.labelDict[name])

    def getNameIndex(self, name, index):
        if name not in self.labelDict:
            printExit("You have to add a label first to get a label name with specific index.")
        if index > self.labelDict[name]:
            printExit("The index %u exceeded. (> %u)"%(index, self.labelDict[name]))
        return name + "_" + str(index)
