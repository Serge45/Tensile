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
from typing import NamedTuple

def GetAssignmentDict(module):
    assignmentDict = dict()
    GetAssignmentDictIter(module, assignmentDict)
    return assignmentDict

def BuildGraph(module, vgprMax, sgprMax, assignmentDict):
    graph = dict()
    graph["v"] = [[] for _ in range(vgprMax)]
    graph["s"] = [[] for _ in range(sgprMax)]
    graph["m"] = [[] for _ in range(1)]
    RecordGraph(module, graph, assignmentDict)
    return graph

def RemoveDuplicateAssignment(graph):
    RemoveDuplicateAssignmentGPR(graph, "s")

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
        elif isinstance(item, Code.RegSet):
            num = 0
            if item.ref:
                num = assignmentDict[item.ref] + item.offset
            else:
                num = item.value
            assignmentDict[item.name] = num

class graphProp(NamedTuple):
    lrv: int = 0
    item: Code.Item = None

def RecordGraph(module, graph, assignmentDict):
    for item in module.items():
        if isinstance(item, Code.Module):
            RecordGraph(item, graph, assignmentDict)
        elif isinstance(item, Code.BranchInst) or \
             isinstance(item, Code.Label) or \
             isinstance(item, Code.CompoundInst):
            if isinstance(item, Code.CompoundInst) and (not isinstance(item, Code.WaitCnt)):
                printExit("Currently does not support any Item that is a Code.CompoundInst but \
                           not a Code.WaitCnt.")
            for i in range(len(graph["v"])):
               graph["v"][i].append(item)
            for i in range(len(graph["s"])):
               graph["s"][i].append(item)
        elif isinstance(item, Code.Inst):
            branchInstList = ["s_setpc_b64", "s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz"]
            if item.inst in branchInstList:
                assert("Should not add branch inst without Code.BranchInst.")
            for p in item.params:
                if isinstance(p, Code.RegisterContainer):
                    setName2RegNum(p, assignmentDict)
                    if p.regType == "acc":
                        continue
                    for i in range(p.regIdx, p.regIdx + p.regNum):
                        if graph[p.regType][i] and graph[p.regType][i][-1] == item:
                            continue
                        # print("[%s] Index %d %d" %(p.regType, i, len(graph[p.regType])))
                        graph[p.regType][i].append(item)
        elif isinstance(item, Code.Macro):
            # Only push when registers are used in the macro
            for p in item.macro.params:
                if isinstance(p, Code.RegisterContainer):
                    setName2RegNum(p, assignmentDict)
                    for i in range(p.regIdx, p.regIdx + p.regNum + 1):
                        if graph[p.regType][i] and graph[p.regType][i][-1] == item:
                            continue
                        graph[p.regType][i].append(item)

# Currently only removes s_mov_b32, does not support 2 sgpr at lvalue
def RemoveDuplicateAssignmentGPR(graph, regType):
    for idx, sList in enumerate(graph[regType]):
        assignValue = None
        newList = []
        for item in sList:
            isRemoved = False
            if isinstance(item, Code.BranchInst) or \
               isinstance(item, Code.Label) or \
               isinstance(item, Code.CompoundInst):
               assignValue = None
            elif isinstance(item, Code.Inst):
                if item.inst == "s_mov_b32":
                    gpr      = item.params[0]
                    gprValue = item.params[1]
                    if gpr.regIdx == idx and gprValue == assignValue:
                        if item.comment:
                            item.inst = ""
                            item.params = []
                            item.comment += " (dup assign opt.)"
                        else:
                            module = item.parent
                            module.removeItem(item)
                        isRemoved = True
                    assignValue = gprValue
                # These macros does not follow the pattern :(
                elif re.match(r'GLOBAL_OFFSET_', item.inst) or \
                     item.inst == "V_MAGIC_DIV" or \
                     item.inst == "DYNAMIC_VECTOR_DIVIDE": # Remove if global read use the register
                    assignValue = None
                elif len(item.params) > 1:
                    gpr = item.params[0]
                    if isinstance(gpr, Code.RegisterContainer) and (gpr.regType == regType):
                        for i in range(gpr.regIdx, gpr.regIdx + gpr.regNum):
                            if i == idx:
                                assignValue = None
                                break
            if not isRemoved:
                newList.append(item)

        if len(newList) != len(sList):
            graph["s"][idx] = newList

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

def graphDebugSaveToTxt(graph, kernelName):
    f = open('%s.txt' % kernelName, 'w')
    f.write("VGPR\n")
    i = 0
    for d in graph["v"]:
        f.write("[%d]\n" % i)
        for dd in d:
            ss = str(dd)
            f.write(ss)
        f.write("\n")
        i += 1
    i = 0
    f.write("SGPR\n")
    for d in graph["s"]:
        f.write("[%d]\n" % i)
        for dd in d:
            ss = str(dd)
            f.write(ss)
        f.write("\n")
        i += 1
    f.close()
