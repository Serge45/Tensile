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
                gprList = setName2RegNum(gpr, assignmentDict)
                if gprList[0] in gprAssignValue:
                    if gprList[1][0] in gprAssignValue[gprList[0]]:
                        if gprValue == gprAssignValue[gprList[0]][gprList[1][0]]:
                            if item.comment:
                                item.inst = ""
                                item.params = []
                                item.comment += " (dup assign opt.)"
                            else:
                                module.removeItemByIndex(index)
                                index -= 1
                else:
                    gprAssignValue[gprList[0]] = dict()
                gprAssignValue[gprList[0]][gprList[1][0]] = gprValue
            # These macros does not follow the pattern :(
            elif re.match(r'GLOBAL_OFFSET_', item.inst) or \
                 item.inst == "V_MAGIC_DIV" or \
                 item.inst == "DYNAMIC_VECTOR_DIVIDE": # Remove if global read use the register
                m = re.findall(r'^(.+?)\+(\d+)', item.params[0])[0]
                num = assignmentDict[m[0]] + int(m[1])
                if ("V" in gprAssignValue) and \
                    (num in gprAssignValue[gprList[0]]):
                    del gprAssignValue["v"][num]
            elif len(item.params) > 1:
                gpr = item.params[0]
                if isinstance(gpr, str) and (gpr != "vcc") and re.match(r"^[sv]", gpr):
                    gprList = setName2RegNum(gpr, assignmentDict)
                    for gprIdx in gprList[1]:
                        if (gprList[0] in gprAssignValue) and \
                           (gprIdx in gprAssignValue[gprList[0]]): # Remove if anyone use the register
                            del gprAssignValue[gprList[0]][gprIdx]
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
    RegNumList = []
    m = re.findall(r'^[svm]\[(.+?):(.+?)\]', gpr)
    if m:
        m = list(m[0])
        n1 = re.findall(r'^(.+?)(\+)*(\d+)*', m[0])
        n2 = re.findall(r'^(.+?)\+(\d+)', m[1])
        if n2:
            n1 = list(n1[0])
            n2 = list(n2[0])
            num = assignmentDict[n2[0]]
            assert(n1[0] in n2[0])
            lower = int(n1[1]) if n1[1] else 0
            RegNumList = [gpr[0], list(i for i in range(num + lower, num + int(n2[1]) + 1))]
        else:
            RegNumList = [gpr[0], list(i for i in range(int(m[0]), int(m[1]) + 1))]
    else:
        m = re.findall(r'^[svm]\[(.+?)(\+\d+)*\]', gpr)
        if m:
            m = list(item for item in m[0] if item)
            num = assignmentDict[m[0]]
            if len(m) == 2:
                bound = re.findall(r'\d+', m[1])[0]
                RegNumList = [gpr[0], list(i for i in range(num, num + int(bound) + 1))]
            else:
                RegNumList = [gpr[0], [num]]
        else:
            m = re.findall(r'^[svm](\d+)', gpr)
            RegNumList = [gpr[0], [int(m[0])]]
    return RegNumList
