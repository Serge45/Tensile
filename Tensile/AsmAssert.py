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

def bomb(self, scratchVgpr, cookie=None):
    """
    Cause a GPUVM fault.
    Instruction after the bomb will write the cookie to SGPR0, so you can see the cookie in the
    backtrace. Useful for locating which spot in code generated the bomb
    vgprAddr controls which vgpr to overwrite with the null pointer address
    """

    module = Code.Module("bomb")
    vgprAddr = scratchVgpr

    if cookie != None:
        if cookie < 0:
            module.addLabel("bomb_neg%u" % abs(cookie), "")
        else:
            module.addLabel("bomb_%u" % abs(cookie), "")
    module.addInst("v_mov_b32", vgpr(vgprAddr+0), 0, "")
    module.addInst("v_mov_b32", vgpr(vgprAddr+1), 0, "")
    #module.addInst("s_trap",1,  "")
    module.addInst("_flat_load_b32", vgpr(vgprAddr), vgpr(vgprAddr,2), "bomb - force fault" )

    # This move does not execute but appears in the instruction stream immediately following
    # the faulting load:
    if cookie != None:
        module.addInst("s_mov_b32", sgpr(0), cookie, "bomb cookie=%d(0x%x)"%(cookie,cookie&0xffffffff))

    return module

class Assert():
    def __init__(self, laneSGPRCount, wavefrontSize):
        self.printedAssertCnt = 0
        self.laneSGPRCount = laneSGPRCount
        self.wavefrontSize = wavefrontSize

    ##############################################################################
    # assertCommon : Common routine for all assert functions.
    # On entry, we have already set the exec-mask so any enabled lanes should bomb
    ##############################################################################
    def assertCommon(self, vtmp, cookie=-1):
        module = Code.Module("assertCommon")
        if self.db["EnableAsserts"]:
            self.printedAssertCnt += 1
            # Default cookie for asserts is negative of printed #asserts
            # Can be used to roughly identify which assert in the code is firing
            module.addCode(self.bomb(vtmp, cookie if cookie != -1 else -self.printedAssertCnt))
        return module

    ##############################################################################
    # assertCmpCommon : Common routine for all assert comparison functions
    ##############################################################################
    def assertCmpCommon(self, cond, val0, val1, vtmp, cookie=-1):
        module = Code.Module("assertCmpCommon")
        if self.db["EnableAsserts"]:
            module.addInst("s_or_saveexec_b{}".format(self.wavefrontSize), sgpr("SaveExecMask",self.laneSGPRCount), 0, \
                "assert: saved execmask")
            module.addInst("_v_cmpx_%s"%cond, self.vcc, val0, val1, "v_cmp" )
            module.addCode(self.assertCommon(vtmp, cookie))
            module.addInst("s_or_saveexec_b{}".format(self.wavefrontSize), self.vcc, sgpr("SaveExecMask",self.laneSGPRCount), \
                "assert: restore execmask")
        return module

    ##############################################################################
    # Handle different conditions for the asserts:
    # These support uin32 compare, float could be added later
    # Asserts currently modify vcc
    ##############################################################################
    def eq(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon("ne_u32", val0, val1, vtmp, cookie)

    def eq_u16(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon("ne_u16", val0, val1, vtmp, cookie)

    def ne(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon("eq_u32", val0, val1, vtmp, cookie)

    def lt_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon("ge_u32", val0, val1, vtmp, cookie)

    def gt_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon("le_u32", val0, val1, vtmp, cookie)

    def le_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon("gt_u32", val0, val1, vtmp, cookie)

    def ge_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon("lt_u32", val0, val1, vtmp, cookie)

    def ge_i32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon("lt_i32", val0, val1, vtmp, cookie)

    # can left shift w/o losing non-zero bits:
    def no_shift_of(self, val0, shift, stmp, vtmp, cookie=-1):
        module = Code.Module("Assert no shift of")
        # TODO - use BFE here:
        module.addInst("s_mov_b32", stmp, hex((shift-1) << (32-log2(shift))), "assert_no_shift_of - compute mask")
        module.addInst("s_and_b32", stmp, stmp, val0, "assert_no_shift_of")
        module.addCode(self.eq(stmp, 0, vtmp, cookie))
        return module

    # asserts if val0 is not an integer multiple of multiple2
    # multiple2 must be a constant and power of 2
    # for example assert_multiple(A, 8) will assert if A is not multiple of 8
    def multiple_b32(self, sval, multiple2, vtmp, cookie=-1):
        module = Code.Module("Assert multiple b32")
        if self.db["EnableAsserts"]:

            stmp = sgpr("SaveExecMask") # repurpose to get a tmp sgpr

            module.addInst("s_and_b{}".format(self.wavefrontSize), stmp, sval, multiple2-1, "mask" )
            module.addInst("s_cmp_eq_u32", stmp, 0, "if maskedBits==0 then SCC=1 == no fault" )
            module.addInst("s_mov_b{}".format(self.wavefrontSize), sgpr("SaveExecMask",self.laneSGPRCount), -1, "")
            module.addInst("s_cmov_b{}".format(self.wavefrontSize), sgpr("SaveExecMask", self.laneSGPRCount),  0, "Clear exec mask")

            module.addInst("s_and_saveexec_b{}".format(self.wavefrontSize), sgpr("SaveExecMask",self.laneSGPRCount), sgpr("SaveExecMask",self.laneSGPRCount), \
                "assert: saved execmask")

            module.addCode(self.assertCommon(vtmp, cookie))

            module.addInst("s_or_saveexec_b{}".format(self.wavefrontSize), self.vcc, sgpr("SaveExecMask",self.laneSGPRCount), \
                "assert: restore execmask")

        return module

    # assert v0 + expectedScalarDiff == v1
    # Verify that each element in v1 is scalar offset from v0
    def assert_vector_diff(self, v0, v1, expectedScalarDiff, cmpvtmp, vtmp, cookie=-1):
        module = Code.Module("assert_vector_diff")
        module.addInst("_v_add_co_u32", \
                       vgpr(cmpvtmp), self.vcc, \
                       expectedScalarDiff, \
                       v0, \
                       "assert_vector_diff add expectedScalarDiff")
        module.addCode(self.eq(vgpr(cmpvtmp), v1, vtmp, cookie))
        return module
