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

from posixpath import isabs
from . import Code
from .Common import globalParameters

# format macro
def macroRegister(name, value):
    return Code.Inst(".set", name, value, "")

class InstMacros():
    def __init__(self, version, isa, macInst, asmCaps, archCaps, asmBugs, wavefrontSize, vcc):
        self.version       = version
        self.isa           = isa
        self.macInst       = macInst
        self.asmCaps       = asmCaps
        self.archCaps      = archCaps
        self.asmBugs       = asmBugs
        self.wavefrontSize = wavefrontSize
        self.vcc           = vcc

    def defineVALUMacros(self):
        """
          Defines cross-architecture compatibility macros.
        """
        module = Code.Module("VALU macros")

        macro = Code.Macro("_v_add_co_u32", "dst:req", "cc:req", "src0:req", "src1:req", "dpp=")
        if self.asmBugs["ExplicitCO"]:
            macro.addInst("v_add_co_u32", "\\dst", "\\cc", "\\src0", "\\src1", "\\dpp", "")
        else:
            macro.addInst("v_add_u32", "\\dst", "\\cc", "\\src0", "\\src1", "\\dpp", "")
        module.addCode(macro)

        # add w/o carry-out.  On older arch, vcc is still written
        macro = Code.Macro("_v_add_u32", "dst:req", "src0:req", "src1:req", "dpp=")
        if self.asmBugs["ExplicitNC"]:
            macro.addInst("v_add_nc_u32", "\\dst", "\\src0", "\\src1", "\\dpp", "")
        elif self.asmBugs["ExplicitCO"]:
            macro.addInst("v_add_u32", "\\dst", "\\src0", "\\src1", "\\dpp", "")
        else:
            macro.addInst("v_add_u32", "\\dst", "vcc", "\\src0", "\\src1", "\\dpp", "")
        module.addCode(macro)

        # add w/o carry-out.  On older arch, vcc is still written
        macro = Code.Macro("_v_add_i32", "dst:req", "src0:req", "src1:req", "dpp=")
        if self.asmBugs["ExplicitNC"]:
            macro.addInst("v_add_nc_i32", "\\dst", "\\src0", "\\src1", "\\dpp", "")
        elif self.asmBugs["ExplicitCO"]:
            macro.addInst("v_add_i32", "\\dst", "\\src0", "\\src1", "\\dpp", "")
        else:
            macro.addInst("v_add_i32", "\\dst", "vcc", "\\src0", "\\src1", "\\dpp", "")
        module.addCode(macro)

        macro = Code.Macro("_v_addc_co_u32", "dst:req", "ccOut:req", "src0:req", "ccIn:req", "src1:req", "dpp=")
        if self.asmBugs["ExplicitNC"]:
            macro.addInst("v_add_co_ci_u32" "\\dst", "\\ccOut", "\\src0", "\\ccIn", "\\src1", "\\dpp", "")
        elif self.asmBugs["ExplicitCO"]:
            macro.addInst("v_addc_co_u32", "\\dst", "\\ccOut", "\\src0", "\\ccIn", "\\src1", "\\dpp", "")
        else:
            macro.addInst("v_addc_u32", "\\dst", "\\ccOut", "\\src0", "\\ccIn", "\\src1", "\\dpp", "")
        module.addCode(macro)

        macro = Code.Macro("_v_sub_co_u32", "dst:req", "cc:req", "src0:req", "src1:req", "dpp=")
        if self.asmBugs["ExplicitCO"]:
            macro.addInst("v_sub_co_u32", "\\dst", "\\cc", "\\src0", "\\src1", "\\dpp", "")
        else:
            macro.addInst("v_sub_u32", "\\dst", "\\cc", "\\src0", "\\src1", "\\dpp", "")
        module.addCode(macro)

        # sub w/o carry-out.  On older arch, vcc is still written.
        macro = Code.Macro("_v_sub_u32", "dst:req", "src0:req", "src1:req", "dpp=")
        if self.asmBugs["ExplicitNC"]:
            macro.addInst("v_sub_nc_u32", "\\dst", "\\src0", "\\src1", "\\dpp", "")
        elif self.asmBugs["ExplicitCO"]:
            macro.addInst("v_sub_u32", "\\dst", "\\src0", "\\src1", "\\dpp", "")
        else:
            macro.addInst("v_sub_u32", "\\dst", "vcc", "\\src0", "\\src1", "\\dpp", "")
        module.addCode(macro)

        # sub w/o carry-out.  On older arch, vcc is still written.
        macro = Code.Macro("_v_sub_i32", "dst:req", "src0:req", "src1:req", "dpp=")
        if self.asmBugs["ExplicitNC"]:
            macro.addInst("v_sub_nc_i32", "\\dst", "\\src0", "\\src1", "\\dpp", "")
        elif self.asmBugs["ExplicitCO"]:
            macro.addInst("v_sub_i32", "\\dst", "\\src0", "\\src1", "\\dpp", "")
        else:
            macro.addInst("v_sub_i32", "\\dst", "vcc", "\\src0", "\\src1", "\\dpp", "")
        module.addCode(macro)

        # Use combined add+shift, where available:
        macro = Code.Macro("_v_add_lshl_u32", "dst:req", "src0:req", "src1:req", "shiftCnt:req")
        if globalParameters["AsmCaps"][self.version]["HasAddLshl"]:
            macro.addInst("v_add_lshl_u32", "\\dst", "\\src0", "\\src1", "\\shiftCnt", "")
        else:
            if self.asmBugs["ExplicitCO"]:
                macro.addInst("v_add_co_u32", "\\dst", "vcc", "\\src0", "\\src1", "")
            else:
                macro.addInst("v_add_u32", "\\dst", "vcc", "\\src0", "\\src1", "")
            macro.addInst("v_lshlrev_b32", "\\dst", "\\shiftCnt", "\\dst", "")
        module.addCode(macro)

        # Use combined shift+add, where available:
        macro = Code.Macro("_v_lshl_add_u32", "dst:req", "src0:req", "src1:req", "shiftCnt:req")
        if globalParameters["AsmCaps"][self.version]["HasAddLshl"]:
            macro.addInst("v_lshl_add_u32", "\\dst", "\\src0", "\\src1", "\\shiftCnt", "")
        else:
            macro.addInst("v_lshlrev_b32", "\\dst", "\\shiftCnt", "\\dst", "")
            if self.asmBugs["ExplicitCO"]:
                macro.addInst("v_add_co_u32", "\\dst", "vcc", "\\src0", "\\src1", "")
            else:
                macro.addInst("v_add_u32", "\\dst", "vcc", "\\src0", "\\src1", "")
        module.addCode(macro)

        # Use combined shift+or, where available:
        macro = Code.Macro("_v_lshl_or_b32", "dst:req", "src0:req", "shiftCnt:req", "src1:req")
        if globalParameters["AsmCaps"][self.version]["HasLshlOr"]:
            macro.addInst("v_lshl_or_b32", "\\dst", "\\src0", "\\shiftCnt", "\\src1", "")
        else:
            macro.addInst("v_lshlrev_b32", "\\dst", "\\shiftCnt", "\\src0", "")
            macro.addInst("v_or_b32", "\\dst", "\\dst", "\\src1", "")
        module.addCode(macro)

        # v_dot2acc & v_dot4_acc
        inst = 'v_dot2c_f32_f16' if (self.version[0] < 11) else 'v_dot2acc_f32_f16'
        macro = Code.Macro("_v_dot2acc_f32_f16", "dst", "src0", "src1")
        macro.addInst(inst, "\\dst", "\\src0", "\\src1", "")
        module.addCode(macro)

        return module


    def defineCMPXMacros(self):
        """
        Navi's cmpx instruction writes only to EXEC, not to SGPRs or to VCC.
        For now, replicate old behaviour with two instructions.
        """
        def macro(op, dtype):
            dict = {'op': op, 'dtype': dtype}
            macro = Code.Macro("_v_cmpx_{op}_{dtype}".format(**dict), "dst", "src0", "src1=")
            if self.archCaps["CMPXWritesSGPR"]:
                macro.addInst("v_cmpx_{op}_{dtype}".format(**dict), "\\dst", "\\src0", "\\src1", "")
            else:
                macro.addInst("v_cmp_{op}_{dtype}".format(**dict), "\\dst", "\\src0", "\\src1", "")
                if self.wavefrontSize == 64:
                    macro.addInst("s_mov_b64", "exec", "\\dst", "")
                else:
                    macro.addInst("s_mov_b32", "exec_lo", "\\dst", "")
            return macro

        ops = ['lt', 'eq', 'le', 'gt', 'ne', 'lg', 'ge', 'o', 'u']
        dtypes = list([sg + ln for sg in ['i','u'] for ln in ['16', '32', '64']])

        module = Code.Module("CMPX macros")
        for op in ops:
            for dtype in dtypes:
                module.addCode(macro(op, dtype))
        return module


    def defineMACInstructionMacros(self):
        macro = Code.Macro("_v_mac_f32", "c:req", "a:req", "b:req")

        if self.macInst == "FMA":
            if self.asmCaps["v_fmac_f32"]:
                macro.addInst("v_fmac_f32", "\\c", "\\a", "\\b", "")
            elif self.asmCaps["v_fma_f32"]:
                macro.addInst("v_fmac_f32", "\\c", "\\a", "\\b", "\\c", "")
            else:
                raise RuntimeError("FMA instruction specified but not supported on {}".format(self.isa))
        elif self.asmCaps["v_mac_f32"]:
            macro.addInst("v_mac_f32", "\\c", "\\a", "\\b", "")
        else:
            raise RuntimeError("MAC instruction specified but not supported on {}".format(self.isa))

        return macro


    def generalMacro(self, prefix, origin, replace, *args):
        macro = Code.Macro("_%s%s" % (prefix, origin), *args)
        inst_args = list(["\\%s"%arg for arg in args])
        macro.addInst("%s%s" % (prefix, replace), *inst_args, "")
        return macro


    def defineSLoadMacros(self):
        module = Code.Module("S load macros")
        macro_list = {'b32' :'dword',
                      'b64' :'dwordx2',
                      'b128':'dwordx4',
                      'b256':'dwordx8',
                      'b512':'dwordx16'}
        module.addComment1('scale global load macros')
        for key in macro_list:
            origin = key
            replace = macro_list[key] if (self.version[0] < 11) else key
            module.addCode(self.generalMacro("s_load_", origin, replace, 'dst', 'base', 'offset'))
        return module


    def defineDSMacros(self):
        module = Code.Module("DS macros")
        module.addComment1('ds operation macros')

        width = ('u8', 'u8_d16_hi', 'u16', 'u16_d16_hi', 'b32', 'b64', 'b128')
        for w in width:
            origin = f'load_{w}'
            replace = f'read_{w}' if (self.version[0] < 11) else f'load_{w}'
            module.addCode(self.generalMacro('ds_', origin, replace, 'dst', 'src', 'offset'))

        width = ('b8', 'b8_d16_hi', 'b16', 'b16_d16_hi', 'b32', 'b64', 'b128')
        for w in width:
            origin = f'store_{w}'
            replace = f'write_{w}' if (self.version[0] < 11) else f'store_{w}'
            module.addCode(self.generalMacro('ds_', origin, replace, 'dst', 'src', 'offset'))

        width = ('b32', 'b64')
        op = {'load2' : 'read2',
              'store2': 'write2'}
        for key in op:
            for w in width:
                origin = f'{key}_{w}'
                replace = f'{op[key]}_{w}' if (self.version[0] < 11) else f'{key}_{w}'
                module.addCode(self.generalMacro('ds_', origin, replace, 'dst', 'src', 'offset1', 'offset2'))

        return module


    def defineBufferMemoryMacros(self):
        module = Code.Module("Buffer memory operation macros")
        module.addComment1('buffer memory operation macros')

        type_list = {
          'b32'       : 'dword',
          'b64'       : 'dwordx2',
          'b96'       : 'dwordx3',
          'b128'      : 'dwordx4',
          'd16_b16'   : 'short_d16',
          'd16_hi_b16': 'short_d16_hi',
          'd16_u8'    : 'ubyte_d16',
          'd16_hi_u8' : 'ubyte_d16_hi',
          'u16'       : 'ushort'
        }
        for t in type_list:
            origin  = f'{t}'
            replace = f'{type_list[t]}' if (self.version[0] < 11) else f'{t}'
            module.addCode(self.generalMacro('buffer_load_', origin, replace, 'dst', 'voffset', 'base', 'soffset', 'offen', 'ioffset', 'md0', 'md1', 'md2'))

        type_list = {
          'b32'       : 'dword',
          'b64'       : 'dwordx2',
          'b96'       : 'dwordx3',
          'b128'      : 'dwordx4',
          'b16'       : 'short',
          'd16_hi_b16': 'short_d16_hi',
          'b8'        : 'byte',
          'd16_hi_b8' : 'byte_d16_hi',
        }
        for t in type_list:
            origin  = f'{t}'
            replace = f'{type_list[t]}' if (self.version[0] < 11) else f'{t}'
            module.addCode(self.generalMacro('buffer_store_', origin, replace, 'src', 'voffset', 'base', 'soffset', 'offen', 'ioffset', 'md0', 'md1', 'md2'))

        type_list = {'_b32': '',
                     '_b64': '_x2'}
        for t in type_list:
            origin  = f'{t}'
            replace = f'{type_list[t]}' if (self.version[0] < 11) else f'{t}'
            module.addCode(self.generalMacro('buffer_atomic_cmpswap', origin, replace, 'dst', 'voffset', 'base', 'soffset', 'offen', 'ioffset', 'md0', 'md1', 'md2'))

        return module


    def defineFlatMemoryMacros(self):
        module = Code.Module("Flat memory operation macros")
        module.addComment1('Flat memory operation macros')

        type_list = {
          'b32'       : 'dword',
          'b64'       : 'dwordx2',
          'b96'       : 'dwordx3',
          'b128'      : 'dwordx4',
          'd16_b16'   : 'short_d16',
          'd16_hi_b16': 'short_d16_hi',
          'd16_u8'    : 'ubyte_d16',
          'd16_hi_u8' : 'ubyte_d16_hi',
          'u16'       : 'ushort'
        }
        for t in type_list:
            origin  = f'{t}'
            replace = f'{type_list[t]}' if (self.version[0] < 11) else f'{t}'
            module.addCode(self.generalMacro('flat_load_', origin, replace, 'dst', 'base', 'md0', 'md1', 'md2'))
            module.addCode(self.generalMacro('flat_store_', origin, replace, 'base', 'src', 'md0', 'md1', 'md2'))

        type_list = {'_b32': ''}
        for t in type_list:
            origin  = f'{t}'
            replace = f'{type_list[t]}' if (self.version[0] < 11) else f'{t}'
            module.addCode(self.generalMacro('flat_atomic_cmpswap', origin, replace, 'tmp', 'base', 'data', 'md'))

        return module


    def defineFeatureMacros(self):
        """
          Defines cross-architecture compatibility macros.
        """
        module = Code.Module("Feature macros")
        module.addComment2("Asm syntax workarounds")
        module.addCode(self.defineVALUMacros())
        module.addCode(self.defineCMPXMacros())
        module.addCode(self.defineMACInstructionMacros())
        module.addCode(self.defineSLoadMacros())
        module.addCode(self.defineDSMacros())
        module.addCode(self.defineBufferMemoryMacros())
        module.addCode(self.defineFlatMemoryMacros())

        return module

    # Performs a division using 'magic number' computed on host
    # Argument requirements:
    #   - dstIdx must be two consecutive registers ; on exit the lower one will contain the quotient.  The upper is used as a temp.
    #   - First parm is passed as an integer vgpr index ; remaining are vgpr or sgpr symbolic names
    #   - dstIdx+1 cannot be same as dividend.  dividend+0 can be same as dividend and this may be useful for chaining divides.
    def defineMagicDivMacros(self, magicDivAlg):
        module = Code.Module("defineMagicDivMacros")
        module.addComment1("Magic div and mod functions")
        macro = Code.Macro("V_MAGIC_DIV", "dstIdx:req", "dividend:req", "magicNumber:req", "magicShift:req", "magicA:req")
        if magicDivAlg==1: # TODO: remove me
            macro.addInst("v_mul_hi_u32", "v[\\dstIdx+1]", "\\dividend", "\\magicNumber", "")
            macro.addInst("v_mul_lo_u32", "v[\\dstIdx+0]", "\\dividend", "\\magicNumber", "")
            macro.addInst("v_lshrrev_b64", "v[\\dstIdx:\\dstIdx+1]", "\\magicShift", "v[\\dstIdx:\\dstIdx+1]", "")
        elif magicDivAlg==2:
            macro.addInst("v_mul_hi_u32", "v[\\dstIdx+1]", "\\dividend", "\\magicNumber", "")
            macro.addInst("v_mul_lo_u32", "v[\\dstIdx+0]", "\\dividend", "\\magicA", "")
            macro.addInst("_v_add_u32", "v[\\dstIdx+0]", "v[\\dstIdx+0]", "v[\\dstIdx+1]", "")
            macro.addInst("v_lshrrev_b32", "v[\\dstIdx+0]", "\\magicShift", "v[\\dstIdx+0]", "")
        module.addCode(macro)
        return module

    def defineDynamicScalarDivMacros(self):
        module = Code.Module("Dynamic scalar divide macros")
        module.addComment1("Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor;")
        macro = Code.Macro("DYNAMIC_VECTOR_DIVIDE", "vQuotient", "vRemainder", "vDividend", "vDivisor", "vTmp0", "vTmp1", "sTmp")
        sTmpStr = "s[\\sTmp]" if (self.wavefrontSize == 32) else "s[\\sTmp:\\sTmp+1]"
        macro.addInst("v_cvt_f32_u32", "v[\\vQuotient]",  "v[\\vDivisor]",  "" )
        macro.addInst("v_rcp_f32",     "v[\\vQuotient]",  "v[\\vQuotient]", "" )
        macro.addInst("v_mul_f32",     "v[\\vQuotient]",  "0x4f800000",     "v[\\vQuotient]", "" )
        macro.addInst("v_cvt_u32_f32", "v[\\vQuotient]",  "v[\\vQuotient]", "" )
        macro.addInst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vDivisor]", "v[\\vQuotient]", "" )
        macro.addInst("v_mul_hi_u32",  "v[\\vTmp0]",      "v[\\vDivisor]", "v[\\vQuotient]", "" )
        macro.addInst("_v_sub_co_u32",     "v[\\vTmp1]",      self.vcc, hex(0),    "v[\\vRemainder]", "" )
        macro.addInst("v_cmp_ne_i32",  sTmpStr, hex(0),        "v[\\vTmp0]", "" )
        macro.addInst("v_cndmask_b32", "v[\\vRemainder]", "v[\\vTmp1]",     "v[\\vRemainder]", sTmpStr, "" )
        macro.addInst("v_mul_hi_u32",  "v[\\vRemainder]", "v[\\vRemainder]", "v[\\vQuotient]", "" )
        macro.addInst("_v_sub_co_u32",     "v[\\vTmp0]",      self.vcc,            "v[\\vQuotient]", "v[\\vRemainder]", "" )
        macro.addInst("_v_add_co_u32",     "v[\\vQuotient]",  self.vcc,            "v[\\vQuotient]", "v[\\vRemainder]", "" )
        macro.addInst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vTmp0]", sTmpStr, "" )
        macro.addInst("v_mul_hi_u32",  "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vDividend]", "" )
        macro.addInst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
        macro.addInst("_v_sub_co_u32",     "v[\\vTmp0]",      self.vcc,            "v[\\vDividend]", "v[\\vRemainder]", "" )
        macro.addInst("v_cmp_ge_u32",  sTmpStr, "v[\\vDividend]", "v[\\vRemainder]", "" )
        macro.addInst("_v_add_co_u32",     "v[\\vRemainder]", self.vcc,            hex(1), "v[\\vQuotient]", "" )
        macro.addInst("_v_add_co_u32",     "v[\\vTmp1]",      self.vcc, -1,        "v[\\vQuotient]", "" )
        macro.addInst("v_cmp_le_u32",  self.vcc,             "v[\\vDivisor]", "v[\\vTmp0]", "" )
        macro.addInst("s_and_b{}".format(self.wavefrontSize),     self.vcc,             sTmpStr,         self.vcc,     "" )
        macro.addInst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vRemainder]", self.vcc, "" )
        macro.addInst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vTmp1]",     "v[\\vQuotient]", sTmpStr, "" )
        macro.addInst("v_cmp_ne_i32",  self.vcc, hex(0),     "v[\\vDivisor]", "" )
        macro.addInst("v_cndmask_b32", "v[\\vQuotient]",  -1, "v[\\vQuotient]", self.vcc, "final result" )
        macro.addInst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
        macro.addInst("_v_sub_co_u32",     "v[\\vRemainder]", self.vcc,            "v[\\vDividend]", "v[\\vRemainder]", "final result" )
        module.addCode(macro)
        return module
