################################################################################
# Copyright 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

import ctypes
from copy import deepcopy
import math
import re
import struct
from collections import OrderedDict

from .AsmRegisterPool import RegisterPool
from .AsmUtils import vgpr, sgpr, Holder
from .Common import printExit, printWarning
from .Code import HolderContainer, RegisterContainer, Module, Inst, TextBlock

################################################################################
# How to add an activation
# 1. Add a new type in ActivationType
# 2. Create a new getXXXAssembly function in class Activation
# 3. Add if-else condition in generateAssembly in class Activation
# 4. Add if-else condition in generateInlineAssemblyBody in class
#    ActivationInline
#
# Helper function(s)
# 1. getRegAndInitAssembly
#    ```
#    getRegAndInitAssembly(<v for vgpr/ s for sgpr>,
#                          <False for reg pool/ True for tmp reg pool>,
#                          <size of reg>,
#                          <init value>,
#                          <key>,
#                          <comment>)
#    ```
#    Returns
#    1. sgprinf: The original checkOut return value
#    2. regInitStr: The init instruction string
#
#    Example,
#    ```
#    sgprinf, regInitStr = self.getRegAndInitAssembly('s', False, 1, \
#        "0x3f4c422a", "FloatGeluK0", "float gelu k0")
#    ```
#    this will generate ``regInitStr`` as
#    ```
#    s_mov_b32 sXX, "0x3f4c422a" // float16 max
#    ```
#    if the key "FloatGeluK0" is not found in sgprDict
# 2. class ActivationRegisterPool
#    A wrapper of RegisterPool. All the checkOut-ed registers will be checkIn-ed
#    at the end of the numBatches for loop.
#    When ActivationType is set to 'all', the registers will be checkIn-ed after
#    activation's gwvw for loop.
################################################################################

################################################################################
# This is the ActivationType class
# stringList:
#   This list stores the names of extra arguments, e.g.
#   y = (x > 0) ? x : x * alpha
# lookup:
#   This dict stores the supported activation types as keys and number of
#   arguments as values. Insert any new type before 'none' and 'all'. The
#   sequence of the table should match the enum in Activation.hpp.
#
# To add an activation type, see the instruction in Activation.py.
################################################################################

class ActivationAvailable:
    def __init__(self, canHalf=False, canSingle=False, canDouble=False, canBFloat16=False, canInt8=False, canInt16=False, canInt32=False):
        self.half = canHalf
        self.single = canSingle
        self.double = canDouble
        self.bfloat16 = canBFloat16
        self.int8 = canInt8
        self.int16 = canInt16
        self.int32 = canInt32

class ActivationTypeRegister:
    def __init__(self, name, extraArgs, canHalf=False, canSingle=False, canDouble=False, canBFloat16=False, canInt8=False, canInt16=False, canInt32=False):
        self.name = name
        self.extraArgs = extraArgs
        self.can = ActivationAvailable(canHalf, canSingle, canDouble, canBFloat16, canInt8, canInt16, canInt32)
    def typeAvailable(self, dataType):
        if dataType.isHalf() and self.can.half:
            return True
        elif dataType.isSingle() and self.can.single:
            return True
        elif dataType.isDouble() and self.can.double:
            return True
        elif dataType.isBFloat16() and self.can.bfloat16:
            return True
        elif dataType.isInt8() and self.can.int8:
            return True
        elif dataType.isInt32() and self.can.int32:
            return True
        return False

class ActivationType:
    stringList = ['alpha', 'beta', 'gamma', 'delta' ]
    # Exp is only for verification. So we will not return exp in the supported list.
                                                                             # Half,Single,Double,BFloat16,  Int8, Int16, Int32
    lookupVeri = OrderedDict([('exp',       ActivationTypeRegister('exp', 0,       True,  True, False,   False, False, False, False)) ])

    # Note: The BFloat16 gemm uses Single type activations. The int8 gemm uses int32 type activations.
                                                                                 # Half,Single,Double,BFloat16,  Int8, Int16, Int32
    lookup = OrderedDict([('abs',         ActivationTypeRegister('abs', 0,         True,  True,  True,    True, False, False,  True)), \
                          ('clippedrelu', ActivationTypeRegister('clippedrelu', 2, True,  True,  True,   False, False, False,  True)), \
                          ('gelu',        ActivationTypeRegister('gelu', 0,        True,  True, False,   False, False, False, False)), \
                          ('leakyrelu',   ActivationTypeRegister('leakyrelu', 1,   True,  True,  True,   False, False, False,  True)), \
                          ('relu',        ActivationTypeRegister('relu', 0,        True,  True,  True,   False, False, False,  True)), \
                          ('sigmoid',     ActivationTypeRegister('sigmoid', 0,     True,  True, False,   False, False, False, False)), \
                          ('tanh',        ActivationTypeRegister('tanh', 2,        True,  True, False,   False, False, False, False)), \
                          ('none',        ActivationTypeRegister('none', 0)), \
                          ('all',         ActivationTypeRegister('all', 0)) ])
    def __init__(self, value):
        if isinstance(value, str):
            strValue = value.lower()
            if strValue in self.lookup:
                self.value = strValue
            elif strValue in self.lookupVeri:
                self.value = strValue
            else:
                raise RuntimeError("Unrecognized activation type %s"%value)
        elif isinstance(value, ActivationType):
            self.value = value.value
        else:
            raise RuntimeError("Unrecognized input type %s, should be string or ActivationType"%str(value))
    def getAdditionalArgNum(self):
        if self.value == 'all':
            maxArgNum = 0
            for key, activationInst in self.lookup.items():
                maxArgNum = max(maxArgNum, activationInst.extraArgs)
            return maxArgNum
        elif self.value in self.lookup:
            return self.lookup[self.value].extraArgs
        return 0
    def getAdditionalArgStringList(self, addPrefix=True):
        list = []
        for i in range(0, self.getAdditionalArgNum()):
            if addPrefix:
                list.append("activation" + self.stringList[i].capitalize())
            else:
                list.append(self.stringList[i])
        return list
    @classmethod
    def getEnumIndex(cls, enumStr):
        return list(cls.lookup.keys()).index(enumStr)
    @classmethod
    def getEnumStrList(cls, dataType, includeNone = True):
        enumList = []
        for key, activationInst in cls.lookup.items():
            if (((key != 'none') or includeNone) and (key != 'all')):
                if activationInst.typeAvailable(dataType):
                    enumList.append(key)
        if not enumList:
            printWarning("No available activation for this data type %s.\n"%str(dataType))
        return enumList
    def state(self): return self.value.capitalize()
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return self.value.capitalize()
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other.lower()
        elif isinstance(other, ActivationType):
            return self.value == other.value
        else:
            raise RuntimeError("Unrecognized type in rhs, should be string or ActivationType")
    def toEnum(self):
        return self.value.capitalize()

ActivationMagicNumbers = {"FloatGeluK0": 0x3f4c422a, \
                          "FloatGeluK1": 0x3d372713, \
                          "Float16GeluK1": 0x29b9 }

# float32 union
class floatUnion(ctypes.Union):
    _fields_ = [('u', ctypes.c_uint), ('f', ctypes.c_float)]

class ActivationModule:
    ################################################################################
    ################################################################################
    ###
    ###   Public Functions
    ###
    ################################################################################
    ################################################################################

    def __init__(self, vcc) -> None:
        self.vcc = vcc
        self.usePK = True
        self.vgprCounter = 0
        self.sgprCounter = 0
        self.saturateI8 = False
        self.vgprPrefixFormat = ""

    # Public function
    def getModule(self, cDataType, activationType, vgprIdx):
        module = ""
        self.resetGprCounter()
        if (activationType == 'abs'):
            module = self.getAbsModule(cDataType, vgprIdx)
        elif (activationType == 'clippedrelu'):
            module = self.getClippedReluModule(cDataType, vgprIdx, "activationAlpha", "activationBeta")
        elif (activationType == 'exp'):
            module = self.getExpModule(cDataType, vgprIdx)
        elif (activationType == 'gelu'):
            module = self.getGeluModule(cDataType, vgprIdx)
        elif (activationType == 'leakyrelu'):
            module = self.getLeakyReluModule(cDataType, vgprIdx, "activationAlpha")
        elif (activationType == 'relu'):
            module = self.getReluModule(cDataType, vgprIdx)
        elif (activationType == 'sigmoid'):
            module = self.getSigmoidModule(cDataType, vgprIdx)
        elif (activationType == 'tanh'):
            module = self.getTanhModule(cDataType, vgprIdx, "activationAlpha", "activationBeta")
        elif (activationType == 'none'):
            module = Module("No activation")
        else:
            module = Module("%s not implemented"%activationType)

        return self.postProcess(cDataType, module)

    def postProcess(self, cDataType, module):
        CombineInstructions(module)
        module = ConvertCoeffToHex(module, cDataType, self.usePK)
        return module

    def assignGpr(self, module, vgprIdx, sgprIdx):
        patternPrefix = ["v", "s"]
        gprIdx = [vgprIdx, sgprIdx]
        for idx, pf in enumerate(patternPrefix):
            module = HolderToGpr(module, gprIdx[idx], pf)
        return module

    def setUsePK(self, usePK):
        self.usePK = usePK

    def setSaturationForInt8(self, sat):
        self.saturateI8 = sat

    def setVgprPrefixFormat(self, formatting):
        self.vgprPrefixFormat = formatting

    ################################################################################
    ################################################################################
    ###
    ###   Internal Helper Functions
    ###
    ################################################################################
    ################################################################################

    def resetGprCounter(self):
        self.vgprCounter = 0
        self.sgprCounter = 0

    def getVgpr(self, num):
        value = self.vgprCounter
        self.vgprCounter += num
        return value

    def getSgpr(self, num):
        value = self.sgprCounter
        self.sgprCounter += num
        return value

    def vgprPrefix(self, *args):
        if isinstance(args[0], int) and self.vgprPrefixFormat:
            vgprStr = self.vgprPrefixFormat%args[0]
        else:
            vgprStr = args[0]

        if len(args) == 1:
            return vgpr(vgprStr)
        else:
            args = args[1]
            return vgpr(vgprStr, args)

    ################################################################################
    ################################################################################
    ###
    ###   Activation Functions
    ###
    ################################################################################
    ################################################################################

    def getAbsModule(self, cDataType, vgprIdx):
        module = Module("Abs")
        if cDataType.isHalf() or cDataType.isBFloat16():
            absMagic = "0x7fff7fff" if self.usePK else "0x7fff"
            module.addInst("v_and_b32", self.vgprPrefix(vgprIdx), absMagic, self.vgprPrefix(vgprIdx), "Remove sign bit")
        elif cDataType.isSingle():
            module.addInst("v_and_b32", self.vgprPrefix(vgprIdx), "0x7fffffff", self.vgprPrefix(vgprIdx), "Remove sign bit")
        elif cDataType.isDouble():
            module.addInst("v_and_b32", self.vgprPrefix(vgprIdx+1), "0x7fffffff", self.vgprPrefix(vgprIdx+1), "Remove sign bit")
        elif cDataType.isInt32():
            vgprTemp = self.getVgpr(1)
            module.addInst("v_sub_i32", vgpr(Holder(idx=vgprTemp)), 0, self.vgprPrefix(vgprIdx), "x2 = -x")
            if self.saturateI8:
                vgprTemp2 = self.getVgpr(1)
                module.addInst("v_mov_b32", vgpr(Holder(idx=vgprTemp2)), hex(127), "value = 127")
                module.addInst("v_med3_i32", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), vgpr(Holder(idx=vgprTemp2)), "y = min(127, max(x, x2))" )
            else:
                module.addInst("v_sub_i32", self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), "y = max(x, x2)")
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getClippedReluModule(self, cDataType, vgprIdx, activationAlpha, activationBeta):
        module = Module("ClippedRelu")
        if cDataType.isHalf():
            for i in range(0, 2):
                module.addInst("v_cmp_ge_f16", self.vcc, self.vgprPrefix(vgprIdx), sgpr(activationAlpha),
                               "src0_sel:WORD_%d"%i, "src1_sel:WORD_0", "x > alpha ?")
                module.addInst("v_min_f16", self.vgprPrefix(vgprIdx), sgpr(activationBeta), self.vgprPrefix(vgprIdx),
                               "dst_sel:WORD_%d"%i, "dst_unused:UNUSED_PRESERVE", "src0_sel:WORD_%d"%i, "src1_sel:WORD_%d"%i,
                               "min(x, beta)")
                module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx), 0.0, self.vgprPrefix(vgprIdx), self.vcc,
                               "dst_sel:WORD_%d"%i, "dst_unused:UNUSED_PRESERVE", "src0_sel:WORD_%d"%i, "src1_sel:WORD_%d"%i,
                               "set x to 0 if < alpha")
            module.addInst("s_nop", 0, "1 wait states")#workaround for emulator
        elif cDataType.isSingle():
            module.addInst("v_cmp_ge_f32", self.vcc, self.vgprPrefix(vgprIdx), sgpr(activationAlpha), "x >= alpha ?")
            module.addInst("v_min_f32", self.vgprPrefix(vgprIdx), sgpr(activationBeta), self.vgprPrefix(vgprIdx), "min(x, beta)")
            module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx), 0.0, self.vgprPrefix(vgprIdx), self.vcc, "set x to 0 if < alpha")
        elif cDataType.isDouble():
            module.addInst("v_cmp_ge_f64", self.vcc, self.vgprPrefix(vgprIdx, 2), sgpr(activationAlpha, 2), "x >= alpha ?")
            module.addInst("v_min_f64", self.vgprPrefix(vgprIdx, 2), sgpr(activationBeta, 2), self.vgprPrefix(vgprIdx, 2), "min(x, beta)")
            module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx), 0, self.vgprPrefix(vgprIdx), self.vcc, "set x to 0 if < 0")
            module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx+1), 0, self.vgprPrefix(vgprIdx+1), self.vcc, "set x to 0 if < 0")
        elif cDataType.isInt32():
            module.addInst("v_cmp_ge_i32", self.vcc, self.vgprPrefix(vgprIdx), sgpr(activationAlpha), "x >= alpha ?")
            module.addInst("v_min_i32", self.vgprPrefix(vgprIdx), sgpr(activationBeta), self.vgprPrefix(vgprIdx), "min(x, beta)")
            module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx), 0.0, self.vgprPrefix(vgprIdx), self.vcc, "set x to 0 if < alpha")
        return module

    def getExpModule(self, cDataType, vgprIdx):
        module = Module("Exp")
        if cDataType.isHalf():
            sgprMagic = self.getSgpr(1)
            module.addInst("s_mov_b32", sgpr(Holder(idx=sgprMagic)), math.log(math.e,2), "exp magic" )
            if self.usePK:
                module.addInst("v_pk_mul_f16", self.vgprPrefix(vgprIdx), sgpr(Holder(idx=sgprMagic)), self.vgprPrefix(vgprIdx), "exp step 1")
                for i in range(0, 2):
                    vgprCtrl = "dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d"%(i, i)
                    module.addInst("v_exp_f16", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), vgprCtrl, "exp step 2")
                    module.addInst("s_nop", 0, "1 wait state") #workaround for emulator
            else:
                module.addInst("v_mul_f16", self.vgprPrefix(vgprIdx), sgpr(Holder(idx=sgprMagic)), self.vgprPrefix(vgprIdx), "exp step 1")
                module.addInst("v_exp_f16", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), "exp step 2")
                module.addInst("s_nop", 0, "1 wait state") #workaround for emulator
        elif cDataType.isSingle():
            module.addInst("v_mul_f32", self.vgprPrefix(vgprIdx), math.log(math.e,2), self.vgprPrefix(vgprIdx), "exp step 1")
            module.addInst("v_exp_f32", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), "exp step 2" )
            module.addInst("s_nop", 0, "1 wait states")#workaround for emulator
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getGeluModule(self, cDataType, vgprIdx):
        module = Module("Gelu")
        # Gelu(x) = 0.5 * x * (1 + tanh(k0 * x * (1 + k1 * x * x)))
        if cDataType.isHalf():
            pkStr = "_pk" if self.usePK else ""
            flt16GeluK1Str = HexToStr(cDataType, self.usePK, ActivationMagicNumbers["Float16GeluK1"])
            sgprMagicK1 = self.getSgpr(1)
            module.addInst("s_mov_b32", sgpr(Holder(idx=sgprMagicK1)), flt16GeluK1Str, "Float16GeluK1" )
            vgprTemp = self.getVgpr(1)
            module.addInst("v%s_mul_f16"%pkStr, vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), "x * x" )
            vgprCtrl = "op_sel_hi:[1,1,0,1]" if self.usePK else ""
            module.addInst("v%s_fma_f16"%pkStr, vgpr(Holder(idx=vgprTemp)), vgpr(Holder(idx=vgprTemp)), sgpr(Holder(idx=sgprMagicK1)), 1.0, vgprCtrl, "x^2 * k1 + 1")
            module.addInst("v%s_mul_f16"%pkStr, vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), "x * (x^2 * k1 + 1)")
            coef = floatUnion(u=ActivationMagicNumbers["FloatGeluK0"])
            module.addInst("v%s_mul_f16"%pkStr, vgpr(Holder(idx=vgprTemp)), coef.f, vgpr(Holder(idx=vgprTemp)), "k0 * x * (x^2 * k1 + 1)")
            module.addCode(self.getTanhModule(cDataType, Holder(idx=vgprTemp), "", ""))
            vgprCtrl = "op_sel_hi:[0,1,1]" if self.usePK else ""
            module.addInst("v%s_add_f16"%pkStr, vgpr(Holder(idx=vgprTemp)), 1.0, vgpr(Holder(idx=vgprTemp)), vgprCtrl, "1 + tanh(...)" )
            module.addInst("v%s_mul_f16"%pkStr, vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), "x * (1 + tanh(...))")
            module.addInst("v%s_mul_f16"%pkStr, self.vgprPrefix(vgprIdx), 0.5, vgpr(Holder(idx=vgprTemp)), vgprCtrl, "0.5 * x * (1 + tanh(...))")
        elif cDataType.isSingle():
            vgprTemp = self.getVgpr(1)
            flt16GeluK1Str = HexToStr(cDataType, self.usePK, ActivationMagicNumbers["FloatGeluK1"])
            module.addInst("v_mul_f32", vgpr(Holder(idx=vgprTemp)), flt16GeluK1Str, self.vgprPrefix(vgprIdx), "k1 * x")
            module.addInst("v_fma_f32", vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), 1.0, "1 + (k1 * x * x)")
            module.addInst("v_mul_f32", vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), "x * (1 + k1 * x * x)")
            coef = floatUnion(u=ActivationMagicNumbers["FloatGeluK0"])
            module.addInst("v_mul_f32", vgpr(Holder(idx=vgprTemp)), coef.f, vgpr(Holder(idx=vgprTemp)), "k0 * x * (x^2 * k1 + 1)")
            module.addCode(self.getTanhModule(cDataType, Holder(idx=vgprTemp), "", ""))
            module.addInst("v_add_f32", vgpr(Holder(idx=vgprTemp)), 1.0, vgpr(Holder(idx=vgprTemp)), "1 + tanh(...)" )
            module.addInst("v_mul_f32", vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), "x * (1 + tanh(...))")
            module.addInst("v_mul_f32", self.vgprPrefix(vgprIdx), 0.5, vgpr(Holder(idx=vgprTemp)), "0.5 * x * (1 + tanh(...))")
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getLeakyReluModule(self, cDataType, vgprIdx, activationAlpha):
        module = Module("LeakyRelu")
        if cDataType.isHalf():
            vgprTemp = self.getVgpr(1)
            module.addInst("v_pk_mul_f16", vgpr(Holder(idx=vgprTemp)), sgpr(activationAlpha), self.vgprPrefix(vgprIdx), "tmp = x * alpha")
            for i in range(0, 2):
                module.addInst("v_cmp_ge_f16", self.vcc, self.vgprPrefix(vgprIdx), 0.0, "src0_sel:WORD_%d"%i, "src1_sel:WORD_0", "x > 0 ?")
                module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx),
                               self.vcc, "dst_sel:WORD_%d"%i, "dst_unused:UNUSED_PRESERVE", "src0_sel:WORD_%d"%i, "src1_sel:WORD_%d"%i,
                               "set x to tmp if < 0")
        elif cDataType.isSingle():
            vgprTemp = self.getVgpr(1)
            module.addInst("v_mul_f32", vgpr(Holder(idx=vgprTemp)), sgpr(activationAlpha), self.vgprPrefix(vgprIdx), "tmp = x * alpha")
            module.addInst("v_cmp_ge_f32", self.vcc, self.vgprPrefix(vgprIdx), 0.0, "x >= 0 ?")
            module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), self.vcc, "set x to tmp if < 0")
        elif cDataType.isDouble():
            vgprTemp = self.getVgpr(2)
            module.addInst("v_mul_f64", vgpr(Holder(idx=vgprTemp), 2), sgpr(activationAlpha, 2), self.vgprPrefix(vgprIdx, 2), "tmp = x * alpha")
            module.addInst("v_cmp_ge_f64", self.vcc, self.vgprPrefix(vgprIdx, 2), 0.0, "x >= 0 ?")
            module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), self.vcc, "set x to tmp if < 0")
            module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx+1), vgpr(Holder(idx=vgprTemp+1)), self.vgprPrefix(vgprIdx+1), self.vcc, "set x to tmp if < 0")
        elif cDataType.isInt32():
            vgprTemp = self.getVgpr(1)
            module.addInst("v_mul_lo_u32", vgpr(Holder(idx=vgprTemp)), sgpr(activationAlpha), self.vgprPrefix(vgprIdx), "tmp = x * alpha")
            module.addInst("v_cmp_ge_i32", self.vcc, self.vgprPrefix(vgprIdx), 0, "x >= 0 ?")
            module.addInst("v_cndmask_b32", self.vgprPrefix(vgprIdx), vgpr(Holder(idx=vgprTemp)), self.vgprPrefix(vgprIdx), self.vcc, "set x to tmp if < 0")
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getReluModule(self, cDataType, vgprIdx):
        module = Module("LeakyRelu")
        if cDataType.isHalf():
            module.addInst("v_pk_max_f16", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), 0, "x = max(0, x)" )
        elif cDataType.isSingle():
            module.addInst("v_max_f32", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), 0, "x = max(0, x)" )
        elif cDataType.isDouble():
            module.addInst("v_max_f64", self.vgprPrefix(vgprIdx, 2), self.vgprPrefix(vgprIdx, 2), 0, "x = max(0, x)" )
        elif cDataType.isInt32():
            if self.saturateI8:
                vgprTemp = self.getVgpr(1)
                module.addInst("v_mov_b32", vgpr(Holder(idx=vgprTemp)), hex(127), "value = 127")
                module.addInst("v_med3_i32", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), 0, vgpr(Holder(idx=vgprTemp)), "x = min(127, max(0, x))" )
            else:
                module.addInst("v_max_i32", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), 0, "x = max(0, x)" )
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getSigmoidModule(self, cDataType, vgprIdx):
        module = Module("Sigmoid")
        if cDataType.isHalf():
            pkStr = "_pk" if self.usePK else ""
            module.addInst("v%s_mul_f16"%pkStr, self.vgprPrefix(vgprIdx), -1.0, self.vgprPrefix(vgprIdx), " x = -x")
            module.addCode(self.getExpModule(cDataType, vgprIdx))
            if self.usePK:
                module.addInst("v_pk_add_f16", self.vgprPrefix(vgprIdx), 1.0, self.vgprPrefix(vgprIdx), "op_sel_hi:[0,1,1]", "1 + exp(-x)")
                for i in range(0, 2):
                    module.addInst("v_rcp_f16", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx),
                                   "dst_sel:WORD_%d"%i, "dst_unused:UNUSED_PRESERVE", "src0_sel:WORD_%d"%i, "1 / (1 + exp(-x))")
                module.addInst("s_nop", 0, "1 wait state") #workaround for emulator
            else:
                module.addInst("v_add_f16", self.vgprPrefix(vgprIdx), 1.0, self.vgprPrefix(vgprIdx), "1 + exp(-x)")
                module.addInst("v_rcp_f16", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), "1 / (1 + exp(-x))")
                module.addInst("s_nop", 0, "1 wait state") #workaround for emulator
        elif cDataType.isSingle():
            module.addCode(self.getExpModule(cDataType, vgprIdx))
            module.addInst("v_add_f32", self.vgprPrefix(vgprIdx), 1.0, self.vgprPrefix(vgprIdx), "1 + exp(-x)" )
            module.addInst("v_rcp_f32", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), "1 / (1 + exp(-x))" )
            module.addInst("s_nop", 0, "1 wait states")#workaround for emulator
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getTanhModule(self, cDataType, vgprIdx, activationAlpha, activationBeta):
        module = Module("Tanh")
        if cDataType.isHalf():
            pkStr = "_pk" if self.usePK else ""
            # We don't need s_pack_ll_b32_b16 cause the input is already duplicated
            if activationAlpha:
                module.addInst("v%s_mul_f16"%pkStr, self.vgprPrefix(vgprIdx), sgpr(activationAlpha), self.vgprPrefix(vgprIdx), "x * alpha")
            module.addInst("v%s_mul_f16"%pkStr, self.vgprPrefix(vgprIdx), 2, self.vgprPrefix(vgprIdx), " x = 2 * x")
            module.addCode(self.getExpModule(cDataType, vgprIdx))
            if self.usePK:
                module.addInst("v_pk_add_f16", self.vgprPrefix(vgprIdx), 1.0, self.vgprPrefix(vgprIdx), "op_sel_hi:[0,1,1]", "e^2x + 1")
                for i in range(0, 2):
                    vgprCtrl = "dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d"%(i, i)
                    module.addInst("v_rcp_f16", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), vgprCtrl, "1 / (1 + exp(-x))")
                    module.addInst("s_nop", 0, "1 wait state") #workaround for emulator
            else:
                module.addInst("v_add_f16", self.vgprPrefix(vgprIdx), 1.0, self.vgprPrefix(vgprIdx), "e^2x + 1")
                module.addInst("v_rcp_f16", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), "1 / (1 + exp(-x))")
                module.addInst("s_nop", 0, "1 wait state") #workaround for emulator
            vgprCtrl = "op_sel_hi:[0,1,0,1]" if self.usePK else ""
            module.addInst("v%s_fma_f16"%pkStr, self.vgprPrefix(vgprIdx), -2.0, self.vgprPrefix(vgprIdx), 1.0, vgprCtrl, "tanh(x) = (1 / (e^2x + 1)) * (-2) + 1")
            if activationBeta:
                module.addInst("v%s_mul_f16"%pkStr, self.vgprPrefix(vgprIdx), sgpr(activationBeta), self.vgprPrefix(vgprIdx), "beta * tanh(x)")
        elif cDataType.isSingle():
            if activationAlpha:
                module.addInst("v_mul_f32", self.vgprPrefix(vgprIdx), sgpr(activationAlpha), self.vgprPrefix(vgprIdx), "x * alpha")
            module.addInst("v_mul_f32", self.vgprPrefix(vgprIdx), 2, self.vgprPrefix(vgprIdx), " x = 2 * x")
            module.addCode(self.getExpModule(cDataType, vgprIdx))
            module.addInst("v_add_f32", self.vgprPrefix(vgprIdx), 1.0, self.vgprPrefix(vgprIdx), "e^2x + 1")
            module.addInst("v_rcp_f32", self.vgprPrefix(vgprIdx), self.vgprPrefix(vgprIdx), "1 / (e^2x + 1)")
            module.addInst("s_nop", 0, "1 wait states") #workaround for emulator
            module.addInst("v_fma_f32", self.vgprPrefix(vgprIdx), -2.0, self.vgprPrefix(vgprIdx), 1.0, "(-2) * (1 / (e^2x + 1)) + 1")
            if activationBeta:
                module.addInst("v_mul_f32", self.vgprPrefix(vgprIdx), sgpr(activationBeta), self.vgprPrefix(vgprIdx), "beta * tanh(x)")
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

################################################################################
################################################################################
###
###   Post Functions
###
################################################################################
################################################################################

__FUSE_MAGIC_NAME__       = "dbiw@I$HONIhnjf4_fused"
__DEUPLICATE_MAGIC_NAME__ = "W*@($GUBUOJ_duplicated"

# Public
def CombineInstructions(module, fuseDebug = False):
    moduleAndIndex = dict()
    CombineInstructionsBetweenModules(module, moduleAndIndex, fuseDebug)
    # Remove Empty Blocks
    module = RemoveEmptyBlocks(module)
    return module

# Does not support modules with branches
def CombineInstructionsBetweenModules(module, moduleAndIndex, fuseDebug):
    index = 0
    while index < len(module.items()):
        item = module.items()[index]
        if isinstance(item, Module):
            CombineInstructionsBetweenModules(item, moduleAndIndex, fuseDebug)
            index = module.items().index(item)
        elif isinstance(item, Inst):
            if moduleAndIndex:
                FuseInstruction(item, moduleAndIndex, fuseDebug)
                index = module.items().index(item)
            outGpr = item.params[0] if len(item.params) > 0 else ""
            if isinstance(outGpr, RegisterContainer):
                # Update the dict
                moduleAndIndex[outGpr] = item
        index += 1

def RemoveEmptyBlocks(module):
    for idx, item in enumerate(module.items()):
        if isinstance(item, Module):
            newItem = RemoveEmptyBlocks(item)
            module.items()[idx] = newItem
    module.removeItemsByName(__DEUPLICATE_MAGIC_NAME__)
    if len(module.items()) == 1 and isinstance(module.items()[0], Module):
        return module.items()[0]
    return module

def FuseInstruction(currentInst, moduleAndIndex, fuseDebug):
    assert(isinstance(currentInst, Inst))
    backInst = deepcopy(currentInst)
    # Fuses if v_add_f16 to v_fma_f16 if v_add_f16 is a self adding instruction.
    # Currently, we only fuse when the vgpr is add by 1 in both instructions.
    # ex. v_add_f16 v0, 1.0, v0
    #     +  v_fma_f16 v0, -2.0, v0, 1.0
    #     => v_fma_f16 v0, -2.0, v0, 2.0
    if re.match(r"v[_pk]*_add_", currentInst.inst):
        outVgpr = currentInst.params[0]
        addConst = ""
        isSelfAddConst = False
        for param in currentInst.params[1:]:
            if param == outVgpr:
                isSelfAddConst = True
            if (isinstance(param, float) or isinstance(param, int)):
                if param == 1:
                    addConst = param

        if isSelfAddConst and addConst:
            oldInst = moduleAndIndex.get(outVgpr)
            if isinstance(oldInst, Inst):
                oldOutVgpr = oldInst.params[0] if len(oldInst.params) > 0 else ""
                if re.match(r"v[_pk]*_fma_", oldInst.inst) and oldInst.params[3] == 1.0:
                    # Cannot fuse if the target instruction has any rvalue reassigned or its lvalue
                    # used before the current instruction
                    if not FindAssignAndUse(oldInst, currentInst, outVgpr, outVgpr):
                        newInst = deepcopy(oldInst)
                        newInst.params[3] = addConst + newInst.params[3]
                        newInst.comment += " ( + 1 (fused))"
                        currentInst.copy(newInst)
                        removeOldInst(oldInst, backInst, currentInst, fuseDebug)
    # Fuses if v_mul_f16 to v_mul_f16 if the later one is a self multiplying instruction.
    # Only fuses when both instructions multiply constant
    elif re.match(r"v[_pk]*_mul_", currentInst.inst):
        outVgpr = currentInst.params[0]
        mulConst = ""
        newFuseInst = ""
        isSelfMulConst = False
        for param in currentInst.params[1:]:
            if param == outVgpr:
                isSelfMulConst = True
            # The constant may be an sgpr
            if isinstance(param, RegisterContainer) and param.regType == 's':
                oldInst = moduleAndIndex.get(param)
                if isinstance(oldInst, Inst):
                    oldparam = oldInst.params[1]
                    if oldInst.inst == "s_mov_b32" and \
                        oldInst.params[0] == param and (isinstance(oldparam, float) or isinstance(oldparam, int)):
                        # Cannot fuse if another instruction is using the same sgpr before a new assignment occurs
                        if not FindUse(oldInst, currentInst, param):
                            mulConst = oldparam
                            newFuseInst = oldInst
            if (isinstance(param, float) or isinstance(param, int)):
                mulConst = param

        if isSelfMulConst and mulConst:
            oldInst = moduleAndIndex.get(outVgpr)
            if isinstance(oldInst, Inst):
                oldOutVgpr = oldInst.params[0] if len(oldInst.params) > 0 else ""
                if re.match(r"v[_pk]*_mul_", oldInst.inst):
                    # Cannot fuse if the target instruction has any rvalue reassigned or its lvalue
                    # used before the current instruction
                    if not FindAssignAndUse(oldInst, currentInst, outVgpr, outVgpr):
                        for paramIdx, param in enumerate(oldInst.params[1:]):
                            if (isinstance(param, float) or isinstance(param, int)):
                                newInst = deepcopy(oldInst)
                                newValue = param * mulConst
                                formatting = " (fused %f)" if isinstance(param, float) else " (fused %d)"
                                if newFuseInst:
                                    newFuseInst.params[1] = newValue
                                    newInst.params[paramIdx+1] = newFuseInst.params[0]
                                    newFuseInst.comment += formatting%newValue
                                else:
                                    newInst.params[paramIdx+1] = newValue
                                newInst.comment += formatting%newValue
                                currentInst.copy(newInst)
                                removeOldInst(oldInst, backInst, currentInst, fuseDebug)
                                break

# This only works for Activation.py
def FindUse(startInst, targetInst, varTarget):
    _, isUse = FindUseIter(startInst, targetInst, varTarget)
    return isUse

# This only works for Activation.py
def FindUseIter(startItem, targetInst, varTarget):
    module = startItem
    idx = -1
    isEnd = False
    isUse = False
    if isinstance(startItem, Inst):
        module = startItem.parent
        idx = module.items().index(startItem)
    assert(isinstance(module, Module))
    if idx + 1 < len(module.items()[idx + 1:]):
        for item in module.items()[idx + 1:]:
            if item is targetInst:
                pass
            elif isinstance(item, Inst) and (len(item.params) > 0):
                if len(item.params) > 1:
                    for param in item.params[1:]:
                        if param == varTarget:
                            isEnd = True
                            isUse = True
                            break
                elif item.params[0] == varTarget:
                    isEnd = True
                    isUse = False
            elif isinstance(item, Module):
                isEnd, isUse = FindUseIter(item, targetInst, varTarget)
            if isEnd:
                return isEnd, isUse
    return False, isUse

# This only works for Activation.py
def FindAssignAndUse(startInst, endInst, assignVar, useVar):
    _, isUse = FindAssignAndUseIter(startInst, endInst, assignVar, useVar)
    return isUse

# This only works for Activation.py
def FindAssignAndUseIter(startItem, endInst, assignVar, useVar):
    module = startItem
    idx = -1
    isEnd = False
    isUse = False
    if isinstance(startItem, Inst):
        module = startItem.parent
        idx = module.items().index(startItem)
    assert(isinstance(module, Module))
    if idx + 1 < len(module.items()[idx + 1:]):
        for item in module.items()[idx + 1:]:
            # Use
            if item is endInst:
                isEnd = True
            elif isinstance(item, Inst) and (len(item.params) > 0):
                if item.params[0] == assignVar:
                    isEnd = True
                    isUse = True
                # Check use
                if len(item.params) > 1:
                    for param in item.params[1:]:
                        if param == useVar:
                            isEnd = True
                            isUse = True
                            break
            elif isinstance(item, Module):
                isEnd, isUse = FindAssignAndUseIter(item, endInst, assignVar, useVar)
            if isEnd:
                return isEnd, isUse
    return isEnd, isUse

def removeOldInst(removeInst, dstInst, fusedInst, debug):
    module = removeInst.parent
    targetIdx = -1
    for idx, item in enumerate(module.items()):
        if item == removeInst:
            if debug:
                tb = TextBlock("\n/* Fused to block %s + %s -> %s */\n"%(str(removeInst), str(dstInst), fusedInst))
                tb.name = __FUSE_MAGIC_NAME__
                module.items()[idx] = tb
            else:
                targetIdx = idx
            break
    if targetIdx > -1:
        module.removeItemByIndex(targetIdx)

################################################################################
################################################################################
###
###   Helper Functions
###
################################################################################
################################################################################

def getMagic(cDataType, value, isPack=False):
    if cDataType.isDouble():
        printExit("Currently magic does not support double.")
    elif cDataType.isHalf():
        fu = value if isinstance(value, floatUnion) else floatUnion(f=value)
        magicNum = struct.unpack('<H', struct.pack('<e', fu.f))[0]
        if isPack:
            magicNum = ctypes.c_uint(magicNum).value
            magicNum = ((magicNum << 16) | magicNum)
    elif cDataType.isSingle():
        fu = value if isinstance(value, floatUnion) else floatUnion(f=value)
        magicNum = fu.u
    return hex(magicNum)

def getMagicStr(cDataType, value, isPack=False):
    return str(getMagic(cDataType, value, isPack))

def HexToStr(cDataType, isPack, *args):
    if len(args) == 1:
        magicNum = args[0]
        uint32 = ctypes.c_uint(magicNum).value
        if isPack and cDataType.isHalf():
            uint32 = ((uint32 << 16) | uint32)
        hexStr = str(hex(uint32))
    else:
        raise RuntimeError("Currently does not support multiple args.")
    return hexStr

def ConvertCoeffToHex(module, cDataType, isPack):
    if (module.name == "Exp"):
        param = module.items()[0].params[1]
        module.items()[0].params[1] = getMagic(cDataType, param, isPack)
        return module
    for itemIdx, item in enumerate(module.items()):
        if isinstance(item, Module):
            newItem = ConvertCoeffToHex(item, cDataType, isPack)
            module.items()[itemIdx] = newItem
    return module

def HolderToGpr(module, idx, pf):
    for itemIdx, item in enumerate(module.items()):
        if isinstance(item, Module):
            newItem = HolderToGpr(item, idx, pf)
            module.items()[itemIdx] = newItem
        elif isinstance(item, Inst):
            for itemIdx, param in enumerate(item.params):
                if isinstance(param, HolderContainer) and param.regType == pf:
                    param.setRegNum(idx)
                    item.params[itemIdx] = param.getCopiedRC()
    return module

def addSpace(alignStr, str):
  totalLength = len(alignStr) + len(str)
  return '{message: >{width}}'.format(message=str, width=totalLength)

class ActivationInline:
  def __init__(self, wavefrontSize, dataType) -> None:
    self.wavefrontSize = wavefrontSize
    self.dataType = dataType
    self.asmStr = "asm("

  # Public Function
  def generateInlineAssemblyFunction(self, activationType):
    kStr = ""
    if activationType == 'none':
      return kStr

    ptrStr = self.dataType.toDevice("HIP")
    names = ""
    if activationType == 'all':
      names += ",\n"
      names += "  Tensile::ActivationType const activationType"
    for name in activationType.getAdditionalArgStringList(False):
      names += ",\n"
      names += "  %s const %s"%(ptrStr, name)
    changeLine = "\n  " if names else ""
    kStr += "__device__ inline %s activation(%s%s value%s)\n{\n"%(ptrStr, changeLine, ptrStr, names)
    # function body
    if activationType == 'all':
      for index, enumStr in enumerate(ActivationType.getEnumStrList(self.dataType, includeNone=False)):
        if index == 0:
          kStr += "  if (activationType == Tensile::ActivationType::%s) {\n"%(ActivationType(enumStr).toEnum())
        else:
          kStr += " else if (activationType == Tensile::ActivationType::%s) {\n"%(ActivationType(enumStr).toEnum())
        kStr += self.generateInlineAssemblyBody(4, enumStr)
        kStr += "  }"
      kStr += "\n"
    else:
      kStr += self.generateInlineAssemblyBody(2, activationType)
    # function body end
    kStr += "  return value;\n"
    kStr += "}\n"
    return kStr

  def replaceGpr(self, module):
    for item in module.items():
        if isinstance(item, Inst):
            for idx, param in enumerate(item.params):
                if isinstance(param, RegisterContainer):
                    if not param.regName:
                        param.setInlineAsm(True)

  def getActivationAsmStr(self, activation, module, spaces):
    module = activation.postProcess(self.dataType, module)
    self.replaceGpr(module)
    activation.assignGpr(module, 0, 0)
    module.setInlineAsmPrintMode(True)
    kStr = str(module)
    newStr = ""
    for instStr in kStr.split("\n"):
        if instStr:
            newStr += spaces + instStr + "\n"
    return newStr

  # Internal Function
  def generateInlineAssemblyBody(self, spaces, activationType):
    ptrStr = self.dataType.toDevice("HIP")
    activation = ActivationModule(self.vcc)
    activation.setUsePK(False)
    activation.resetGprCounter()
    kStr = ""
    padSpacesStr = ' ' * spaces
    asm = padSpacesStr + self.asmStr
    if (activationType == 'abs'):
      if self.dataType.isHalf() or self.dataType.isBFloat16():
        unionDataTypeStr = "_Float16" if self.dataType.isHalf() else "BFloat16"
        unionName = "f16_union" if self.dataType.isHalf() else "bf16_union"
        kStr += (padSpacesStr + "union {\n")
        kStr += (padSpacesStr + "  %s f;\n"%unionDataTypeStr)
        kStr += (padSpacesStr + "  short s;\n")
        kStr += (padSpacesStr + "} %s;\n"%unionName)
        kStr += (padSpacesStr + "%s.f = value;\n"%unionName)
        kStr += (padSpacesStr + "%s.s = %s.s & 0x7fff;\n"%(unionName, unionName))
        kStr += (padSpacesStr + "value = %s.f;\n"%unionName)
      elif (self.dataType.isSingle() or self.dataType.isDouble() or self.dataType.isInt32()):
        kStr += (padSpacesStr + "value = abs(value);\n")
      else:
        raise RuntimeError("Unrecognized data type %s."%self.dataType)
    elif (activationType == 'clippedrelu'):
      if (self.dataType.isSingle() or self.dataType.isHalf() or self.dataType.isDouble()):
        kStr += (padSpacesStr + "value = (value >= alpha) ? min(value, beta) : 0.0;\n")
      elif self.dataType.isInt32():
        kStr += (padSpacesStr + "value = (value >= alpha) ? min(value, beta) : 0;\n")
    elif (activationType == 'exp'):
      kStr += (asm + " // Exp\n")
      module = activation.getExpModule(self.dataType, 0)
      kStr += self.getActivationAsmStr(activation, module, (len(asm) * " "))
      kStr += addSpace(asm, ": \"+v\"(value) : \n")
      kStr += self.getRequiredRegStr(asm, activation.vgprCounter, activation.sgprCounter)
    elif (activationType == 'gelu'):
      kStr += (asm + " // gelu\n")
      module = activation.getGeluModule(self.dataType, 0)
      kStr += self.getActivationAsmStr(activation, module, (len(asm) * " "))
      kStr += addSpace(asm, ": \"+v\"(value) : \n")
      kStr += self.getRequiredRegStr(asm, activation.vgprCounter, activation.sgprCounter)
    elif (activationType == 'leakyrelu'):
      if (self.dataType.isSingle() or self.dataType.isHalf() or self.dataType.isDouble()):
        kStr += (padSpacesStr + "value = (value >= 0.0) ? value : (value * alpha);\n")
      elif self.dataType.isInt32():
        kStr += (padSpacesStr + "value = (value >= 0) ? value : (value * alpha);\n")
      else:
        raise RuntimeError("Unsupported data type %s."%ptrStr)
    elif (activationType == 'relu'):
      if (self.dataType.isSingle() or self.dataType.isHalf() or self.dataType.isDouble()):
        kStr += (padSpacesStr + "value = max(0.0, value);\n")
      elif self.dataType.isInt32():
        kStr += (padSpacesStr + "value = max(0, value);\n")
      else:
        raise RuntimeError("Unsupported data type %s."%ptrStr)
    elif (activationType == 'sigmoid'):
      kStr += (asm + " // Sigmoid\n")
      module = activation.getSigmoidModule(self.dataType, 0)
      kStr += self.getActivationAsmStr(activation, module, (len(asm) * " "))
      kStr += addSpace(asm, ": \"+v\"(value) : \n")
      kStr += self.getRequiredRegStr(asm, activation.vgprCounter, activation.sgprCounter)
    elif (activationType == 'tanh'):
      kStr += (asm + " // tanh\n")
      module = activation.getTanhModule(self.dataType, 0, 1, 2)
      kStr += self.getActivationAsmStr(activation, module, (len(asm) * " "))
      kStr += addSpace(asm, ": \"+v\"(value) : \"s\"(alpha), \"s\"(beta)\n")
      kStr += self.getRequiredRegStr(asm, activation.vgprCounter, activation.sgprCounter)
    else:
      if (activationType != 'none'):
        raise RuntimeError("Unrecognized type %s."%activationType)
    return kStr
  # Internal use. Automatically gets the required vgprs and sgprs for inline assembly
  def getRequiredRegStr(self, spaceAlignStr, numOfVgpr, numOfSgpr):
    requiredReg = []
    for i in range(0, numOfVgpr):
      requiredReg.append("\"v%d\""%i)
    for i in range(0, numOfSgpr):
      requiredReg.append("\"s%d\""%i)
    requiredStr = ""
    if (len(requiredReg) > 0):
      requiredStr = requiredReg[0]
      for i in range(1, len(requiredReg)):
        requiredStr += ", %s"%requiredReg[i]
    kStr = ""
    kStr += addSpace(spaceAlignStr,":%s);\n"%requiredStr)
    return kStr

  # FIXME: Copy from KernelWriterAssembly
  # Internal use.
  @property
  def vcc(self) -> str:
    if self.wavefrontSize == 64:
      return "vcc"
    else:
      return "vcc_lo"
