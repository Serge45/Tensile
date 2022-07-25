################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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
from .Common import printExit

import re

branchInstList = ["s_setpc_b64", "s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz"]


def GetAssignmentDict(module):
    assignmentDict = dict()
    GetAssignmentDictIter(module, assignmentDict)
    return assignmentDict

def RemoveDuplicateAssignment(module, assignmentDict):
    gprAssignValue = dict()
    RemoveDuplicateAssignmentIter(module, assignmentDict, gprAssignValue)

################################################################################
################################################################################
###
###   Internal Functions
###
################################################################################
################################################################################

def GetAssignmentDictIter(module, assignmentDict):
    for item in module.items():
        if isinstance(item, Code.Module):
            GetAssignmentDictIter(item, assignmentDict)
        elif isinstance(item, Code.Inst):
            if item.inst == ".set":
                if re.match(r'^[sv]gpr', item.params[0]):
                    if isinstance(item.params[1], int):
                        assignmentDict[item.params[0]] = item.params[1]
                    else:
                        # Format must be .set AAAAA, BBBBB+0
                        m = re.findall(r'^([a-zA-Z]+)\+([0-9])*', item.params[1])
                        assert(m)
                        m = m[0]
                        num = assignmentDict[m[0]] + int(m[1])
                        assignmentDict[item.params[0]] = num

# Currently only removes s_mov_b32, does not support 2 sgpr at lvalue
def RemoveDuplicateAssignmentIter(module, assignmentDict, gprAssignValue):
    index = 0
    while index < len(module.items()):
        item = module.items()[index]
        if isinstance(item, Code.Module):
            RemoveDuplicateAssignmentIter(item, assignmentDict, gprAssignValue)
        elif isinstance(item, Code.Inst):
            if item.inst in branchInstList: # Reset dict
                gprAssignValue.clear()
            elif item.inst == "s_mov_b32":
                gpr      = item.params[0]
                gprValue = item.params[1]
                setName2RegNum(gpr, assignmentDict)
                if gpr.regType in gprAssignValue:
                    if gpr.regIdx in gprAssignValue[gpr.regType]:
                        if gprValue == gprAssignValue[gpr.regType][gpr.regIdx]:
                            if item.comment:
                                item.inst = ""
                                item.params = []
                                item.comment += " (dup assign opt.)"
                            else:
                                module.removeItemByIndex(index)
                                index -= 1
                else:
                    gprAssignValue[gpr.regType] = dict()
                gprAssignValue[gpr.regType][gpr.regIdx] = gprValue
            # These macros does not follow the pattern :(
            elif re.match(r'GLOBAL_OFFSET_', item.inst) or \
                 item.inst == "V_MAGIC_DIV" or \
                 item.inst == "DYNAMIC_VECTOR_DIVIDE": # Remove if global read use the register
                m = re.findall(r'^(.+?)\+(\d+)', item.params[0])[0]
                num = assignmentDict[m[0]] + int(m[1])
                if ("v" in gprAssignValue) and \
                    (num in gprAssignValue["v"]):
                    del gprAssignValue["v"][num]
            elif len(item.params) > 1:
                gpr = item.params[0]
                if isinstance(gpr, Code.RegisterContainer):
                    gprList = setName2RegNum(gpr, assignmentDict)
                    if gpr.regType in gprAssignValue:
                        for gprIdx in gprList:
                            if gprIdx in gprAssignValue[gpr.regType]: # Remove if anyone use the register
                                del gprAssignValue[gpr.regType][gprIdx]
        elif isinstance(item, Code.CompoundInst):
            if not isinstance(item, Code.WaitCnt):
                printExit("Currently does not support any Item that is a Code.CompoundInst but \
                           not a Code.WaitCnt.")
        elif isinstance(item, Code.Label):  # Reset dict
            gprAssignValue.clear()
        index += 1

################################################################################
################################################################################
###
###   Helper Functions
###
################################################################################
################################################################################

# Find ".set AAAAA 0" and convert "s[AAAAA]" into "s0"
def setName2RegNum(gpr, assignmentDict):
    assert(isinstance(gpr, Code.RegisterContainer))
    if gpr.regIdx == None and gpr.regName:
        name = gpr.getRegNameWithType()
        m = re.findall(r'^(.+?)\+(\d+)', name)
        if m:
            m = m[0]
            num = assignmentDict[m[0]] + int(m[1])
        else:
            num = assignmentDict[name]
        gpr.regIdx = num
    RegNumList = []
    for i in range(0, gpr.regNum):
        RegNumList.append(i + gpr.regIdx)
    return RegNumList
