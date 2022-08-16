################################################################################
# Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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
from . import CodePass
from . import Common
from .AsmUtils import replacePlaceHolder
from .Common import globalParameters, CHeader, roundUp, Backup, print2, printExit
from .Component import Component
from .CustomKernels import isCustomKernelConfig
from .SolutionStructs import Solution

import abc
import os
import subprocess
import copy

################################################################################
# Kernel Writer
################################################################################
class KernelWriter(metaclass=abc.ABCMeta):
  #__metaclass__=abc.ABCMeta

  ##############################################################################
  # Init
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    self.kernelMinNaming = kernelMinNaming
    self.kernelSerialNaming = kernelSerialNaming
    self.overflowedResources = 0

  @property
  def asmCaps(self):
    """
    Assembler capabilities for the current ISA version.
    """
    return globalParameters["AsmCaps"][self.version]

  @property
  def archCaps(self):
    """
    Architectural capabilities for the current ISA version.
    """
    return globalParameters["ArchCaps"][self.version]

  @property
  def globalParams(self):
    """
    Global parameters for current configuration.
    """
    return globalParameters

  ##############################################################################
  # makeSchedule:  Schedule work into interations.

  # Tensile uses a two-level scheduler.  This the first-level, which
  # schedules global reads, global incs, and local writes into iteration.
  # Then makeSubIterSchedule schedules the instructions within the iteration.
  #
  # Inputs:
  #   localWriteEndIter: loop iteration where last writes should be inserted
  #      If scheduleLocalWrite=0, all writes will be be placed in this iteration.
  #      If scheduleLocalWrite=1, the scheduler will work backwards from this
  #      iteration.
  #
  # Outputs:
  #   self.unrollLoopHeaderCode:
  #      - Code module that should be added into the unroll loop header
  #        In unscheduled code this contains global loads and global address increment
  #   self.perIterGlobalReadCode[], self.perIterLocalWriteCode[]
  #      - List indexed by unroll iteration.
  #        Each entry in the list is a code module that should be added into that iteration.
  #        May be None, indicating no extra code for that iteration
  #   self.grEndMfmaIndex
  #   self.lwStartMfmaIndex
  #   self.lwEndMfmaIndex
  #   self.barrierMfmaIndex
  #   self.numMfmaForNextLoopLR
  # This routine is responsible for setting the schedule including determining
  # that all necessary dependency are met.  The driver code in kernelBody
  # blindly follows the plan set in unrollLoopHeaderCode and perIterCode
  ##############################################################################
  def makeSchedule(self, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu=0, skipGlobalReadInc=False, firstIter=False, lastLoop=False, lastLc=False):

    currentIsa = globalParameters["CurrentISA"]
    maxVmcnt = globalParameters["AsmCaps"][currentIsa]["MaxVmcnt"]

    self.unrollLoopHeaderCode = Code.Module()
    # schedule of work for each local_read iteration:
    self.perIterGlobalReadCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    if lastLc:
      self.perIterLocalWriteCodeNGLL = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
    self.perIterGlobalReadCodeDTV = [ Code.Module() for i in range (kernel["LoopIters"]) ] # global read for DirectToVgpr
    assert([item.name for item in self.globalReadIncrements.itemList] == ['globalReadIncrementA', 'globalReadIncrementB'])

    globalReadIncACode  = self.globalReadIncrements.findNamedItem("globalReadIncrementA")
    globalReadIncBCode  = self.globalReadIncrements.findNamedItem("globalReadIncrementB")

    if uDu < kernel["DepthULdsDivisor"] - 1 and kernel.enabledSplitLDS and kernel["PrefetchGlobalRead"] \
       or skipGlobalReadInc:
      globalReadIncACode  = Code.Module()
      globalReadIncBCode  = Code.Module()

    grBackup = None
    if uDu != kernel["DepthULdsDivisor"] - 2 and kernel.enabledSplitLDS:
      # hack RAII object for auto restore
      # withhold issuing global read codes until in the 2nd last subloop, meaning we empty the code
      # modules in other subloops.
      grBackup = Backup(self, globalReadACode = self.globalReadACode, globalReadBCode = self.globalReadBCode)
      self.globalReadACode = Code.StructuredModule() # empty
      self.globalReadBCode = Code.StructuredModule() # empty

    siaComponent = Component.SIA.find(self)
    siaComponent.schedIntoIteration(self, kernel, tensorParametersA, tensorParametersB, \
      localWriteEndIter, uDu, firstIter, lastLoop, lastLc, maxVmcnt, globalReadIncACode, \
      globalReadIncBCode)

    if grBackup is not None:
      del grBackup

  ##############################################################################
  # Schedule work into the each unroll loop iteration
  # localReadCode is the local reads for this loop iteration
  #  (returned by localReadDo). The instructions in localReadCode
  #  will retain their relative order, but may be interleaved
  #  with instructions from otherCode.

  # globalReadCode is the 'other' buffer loads and addr increments
  # localWriteCode is the 'other' local writes
  #  to schedule in with the ds reads.  The instructions
  #  will retain their relative order, but may be interleaved
  #  with instructions from localReadCode.

  # pointerCode contains local pointer changes (if needed)
  # waitCode contains s_waitcnt before macs.
  #   - Cannot be "" or None
  #   - may be empty Module if not waiting is desired (perhaps for debug)
  #   - may be multiple instructions (ConservativeWaitCnt)
  #   - typically is a single Code.WaitCnt.  This routine will
  #     modify the lgkmcnt to account for any scheduling decisions.
  #     If this is not desired, add the waitCnt to pointerCode and
  #     set waitCode to an empty module
  # macIterCode contains the mac iters.  May be a macro call.
  #
  # returns: a Module with the combined, optimally scheduled
  #  localReadCode + otherCode
  ##############################################################################
  def makeSubIterSchedule(self, kernel, localReadCode, iteration, pointerLWCode, pointerLRCode, waitCode, macIterCode, \
      waitLWCode = Code.Module(), syncCode = Code.Module(), packCode = Code.Module(), isDTVodd = False, NLLlast = False):

    iterCode = Code.Module()
    globalReadCode = copy.deepcopy(self.perIterGlobalReadCode[iteration])
    globalReadCodeDTV = self.perIterGlobalReadCodeDTV[iteration]
    origLenGlobalReadCodeDTV = len(list(self.perIterGlobalReadCodeDTV[iteration].items()))
    localWriteCode = self.perIterLocalWriteCode[iteration]
    isBarrier = kernel["LoopIters"] - self.numItersPLR
    hasLocalRead = localReadCode.countType(Code.LocalReadInst)
    # Default schedule is other, local reads, then local writes:
    if self.scheduleIterAlg==0:
      # simple schedule, just add the modules in-order
      iterCode.addCode(globalReadCode)
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      iterCode.addCode(localReadCode)
      iterCode.addCode(localWriteCode)
      iterCode.addCode(pointerLWCode)
      iterCode.addCode(pointerLRCode)
      iterCode.addCode(waitCode)
      iterCode.addCode(packCode)
      iterCode.addCode(macIterCode)
    elif self.scheduleIterAlg == 1:
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      #import pdb
      #pdb.set_trace()
      # simple algorithm - do half the reads first:
      readsToSchedule = localReadCode.countType(Code.LocalReadInst) / 2
      #localReadCode.prettyPrint()
      readItems = localReadCode.flatitems()
      while readItems:
        item = readItems.pop(0)
        #print "readsToSchedule=", readsToSchedule, "item=", item
        iterCode.addCode(item)
        readsThisItem = item.countType(Code.LocalReadInst)
        if readsThisItem:
          assert readsThisItem==1, "Scheduler assumes 1 read per item"
          readsToSchedule = readsToSchedule - 1
          if readsToSchedule == 0:
            break

      iterCode.addCode(globalReadCode)

      # add rest of the reads here
      for item in readItems:
        iterCode.addCode(item)

      #move down write to be the last
      iterCode.addCode(localWriteCode)
      # tack on the pointer and mac code:
      iterCode.addCode(pointerLWCode)
      iterCode.addCode(pointerLRCode)
      iterCode.addCode(waitCode)
      iterCode.addCode(packCode)
      iterCode.addCode(macIterCode)
    elif self.scheduleIterAlg == 2:
    # SIA2 use only 1 iteration and separate compute and fetch by raising compute priority
    # 2 workgroup interleave, while WG0/WG1 doing compute, WG1/WG0 doing fetch
    # EPS need to be 1, or valu instruction will break interleave
      iterCode.addCode(globalReadCode)
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      iterCode.addCode(localReadCode)
      iterCode.addCode(waitCode)

      # interleave pack code
      # BF16 or FP16: each packCode is for one 32-bit reg,  1 packing inst: half-to-single x1
      # INT8        : each packCode is for one 32-bit regs, 3 packing inst: byte-to-half x2 + half-to-single x1
      instPerRegPack = 1 / kernel["ProblemType"]["DataType"].numRegisters() - 1
      instPerPack    = int(kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numRegisters() * instPerRegPack)
      packItems = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)) ]
        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Code.Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Code.Module()
        packBItems = packB.flatitems()
        if packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)):
          packItems += packINtems.pop(0)

      macIterItem = macIterCode.flatitems()
      # pop the first code which is s_nop 1 for packing
      item = macIterItem.pop(0)

      numMfmaPerIter = self.numMfmaPerIter
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0

      for i in range(numMfmaPerIter):
        if packItems:
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          packAIdx += instPerPack if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packBIdx += instPerPack if i % kernel["MIWaveTileA"] == 0 else 0
          # blockWidth < 1, means 0.5 or 0.25 (BF,H,Int8)
          packAIdx = packAIdx if self.tPA["localReadInstruction"].blockWidth < 1 else 0
          packBIdx = packBIdx if self.tPB["localReadInstruction"].blockWidth < 1 else 0
          numPack = (packAIdx + packBIdx)
          iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))
          # we put 2 pack in each mfma, "2" means A & B
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPack):
                iterCode.addCode(packItems.pop(0))
                curPackIdx += 1
            else:
              iterCode.addInst("s_nop ","0","VALU packing writes to be consumed by matrix instruction")
              curPackIdx += 1
        if i == 0:
          if not packItems:
            tmpVgpr = self.vgprPool.checkOut(1)
            iterCode.addInst("v_mov_b32 ","v%u"%(tmpVgpr),"0x0","valu operation to have different priority")
            self.vgprPool.checkIn(tmpVgpr)
          iterCode.addInst("s_setprio ","3","Raise priority while processing macs")
        item = macIterItem.pop(0)
        iterCode.addCode(item)

      iterCode.addInst("s_setprio ","1","Raise priority while processing macs")
      if kernel["1LDSBuffer"]:
        barrier = Code.Module()
        barrier.addComment0("1 LDS buffer: read-sync-write")
        barrier.addWaitCnt(lgkmcnt=0, comment="")
        barrier.addInst("s_barrier","")
        iterCode.addCode(barrier)
      iterCode.addCode(localWriteCode)
      iterCode.addCode(pointerLWCode)
      iterCode.addCode(pointerLRCode)
      iterCode.addInst("s_setprio ","2","Raise priority while processing macs")
      pass
    elif self.scheduleIterAlg == 3:
      iterCode.addComment0(" grEndMfmaIndex:%u, lwStartMfmaIndex:%u, lwEndMfmaIndex:%u " %(self.grEndMfmaIndex,self.lwStartMfmaIndex,self.lwEndMfmaIndex))
      iterCode.addComment0(" numMfmaForLR:%u, barrierMfmaIndex:%u " %(self.numMfmaForNextLoopLR,self.barrierMfmaIndex))
      #####
      # Prepare and Assign parameter
      ####
      if iteration == 0:
        self.localReadsVacancy = []
        self.localReadsWait = [ [] for j in range(kernel["LoopIters"])]
      self.localReadsWait[iteration] = waitCode
      numMfmaPerIter = self.numMfmaPerIter
      isBarrier = kernel["LoopIters"] - self.numItersPLR
      writeItems = list(localWriteCode.items())
      macIterItems = macIterCode.flatitems()
      skipLocalWriteWaitcnt = 0
      localReadsWaitcnt = 0
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0

      #####
      # Prepare localReadCode
      ####
      localReadCodeAB = Code.Module()
      for iui in range(kernel["InnerUnroll"]):
        localReadCodeA = localReadCode.findNamedItem("LocalReadDoA_I%s"%(iui))
        localReadCodeB = localReadCode.findNamedItem("LocalReadDoB_I%s"%(iui))
        # In case localReadDo not generate localReadCode Module
        # and findNamedItem will return None type
        # TODO: findNamedItem return Code.Module() if not found
        if not localReadCodeA:
          localReadCodeA = Code.Module()
        if not localReadCodeB:
          localReadCodeB = Code.Module()
        if localReadCodeA.items():
          localReadCodeAB.addCode(localReadCodeA.items().pop(0))
        if localReadCodeB.items():
          localReadCodeAB.addCode(localReadCodeB.items().pop(0))
        while localReadCodeA.items():
          localReadCodeAB.addCode(localReadCodeA.items().pop(0))
        while localReadCodeB.items():
          localReadCodeAB.addCode(localReadCodeB.items().pop(0))
      localReadItems = localReadCodeAB.flatitems()
      localReadItemsThisLoop = localReadItems if iteration < isBarrier else []
      localReadItemsNextLoop = localReadItems if iteration >= isBarrier else []

      #####
      # Prepare pack Code                for B:
      # since the mfma reuse B first =>    for A: mfma[A][B]
      # we need 1 vector A and 1 vector B for first mfma
      # then we prepare remaining A, then remaining B
      # BF16 or FP16: each packCode is for one 32-bit reg,  1 packing inst: half-to-single x1
      # INT8        : each packCode is for one 32-bit regs, 3 packing inst: byte-to-half x2 + half-to-single x1
      ####
      instPerRegPack = 1 / kernel["ProblemType"]["DataType"].numRegisters() - 1
      instPerPack    = int(kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numRegisters() * instPerRegPack)
      packItems = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)) ]
        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Code.Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Code.Module()
        packBItems = packB.flatitems()
        if packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)):
          packItems += packINtems.pop(0)

      # remove s_nop for packing
      # we will add s_nop if needed
      if macIterItems:
        macIterItems.pop(0)

      ####
      # scheduled local read to previous iterations
      ####
      if self.numVgprBuffer >= kernel["LoopIters"]:
        for vacancy in self.localReadsVacancy:
          # {"items","latencyLeft","atIter","atMfmaIndex","noReadsAtThisIter"}
          for localRead in list(localReadItemsThisLoop):
            if vacancy["latencyLeft"] > localRead.IssueLatency * 2:
              if not localRead.readToTempVgpr:
                vacancy["latencyLeft"] -= localRead.IssueLatency * 2
                vacancy["items"].addCode(localRead)
                localReadItemsThisLoop.remove(localRead)
                if vacancy["atMfmaIndex"] > self.lwStartMfmaIndex - 1 and kernel["1LDSBuffer"]:
                  self.overflowedResources = 5
                # update waitCnt
                if self.numItersPLR:
                  for readsIter in range(vacancy["atIter"], iteration + self.numItersPLR):
                    if (vacancy["atMfmaIndex"] % numMfmaPerIter == 0 or readsIter != vacancy["atIter"]) and \
                        (vacancy["noReadsAtThisIter"] or readsIter <= vacancy["atIter"] + self.numItersPLR):
                      if isinstance(self.localReadsWait[readsIter], Code.WaitCnt):
                        self.localReadsWait[readsIter].lgkmcnt += 1
                        # This line is added for backward compatibility
                        self.localReadsWait[readsIter].vscnt = self.localReadsWait[readsIter].vmcnt \
                          if self.localReadsWait[readsIter].lgkmcnt != -1 and \
                             self.localReadsWait[readsIter].vmcnt != -1 else -1
            else:
              # make sure the localread sequence remain the same
              vacancy["latencyLeft"] = 0
      numReadsInst = len(localReadItemsThisLoop) if iteration < isBarrier else len(localReadItemsNextLoop)

      for i in range(numMfmaPerIter):
        mfmaIndex = iteration * numMfmaPerIter + i
        iterCode.addComment0(" mfmaIndex:%u " %(mfmaIndex))

        ####
        # scheduled local read
        ####
        readLeft = numReadsInst
        latencyLeft = self.miLatencyLeft
        # with PrefetchLocalRead, localreads can interleave with mfma
        if self.numItersPLR and iteration < isBarrier:
          # take ds_write into account to schedule ds_read, assume A and B localwrite have same width (TLDS=1)
          if (mfmaIndex >= self.lwStartMfmaIndex) and not globalReadCode.countType(Code.GlobalReadInst):
            for j in range(min(len(writeItems),self.numLocalWriteModPerMfma)):
              if writeItems[j].countType(Code.LocalWriteInst):
                latencyLeft -= (self.tPA["localWriteInstruction"].IssueLatency*2)
          readLeftLROPT = 0
          for j in range(len(localReadItemsThisLoop)):
            latencyLeft -= localReadItemsThisLoop[j].IssueLatency*2
            readLeftLROPT += 1 if latencyLeft >= 0 else 0
          # at least 1 instruction
          readLeftLROPT = max(readLeftLROPT,1)
          # evenly schedule localread with each mfma
          readLeftLREven = numReadsInst // numMfmaPerIter
          if (numReadsInst % (numMfmaPerIter)) > i:
            readLeftLREven += 1
          # we want no localreads at first mfma
          if (iteration == 0) and numMfmaPerIter != 1:
            numMfmaForLR = numMfmaPerIter - 1
            if i < numMfmaPerIter - numMfmaForLR:
              readLeftLREven = 0
              readLeftLROPT = 0
            # rest mfma help to schedule those localReads
            else:
              readLeftLREven = numReadsInst // (numMfmaPerIter-1)
              if (numReadsInst % (numMfmaPerIter-1)) >= i:
                readLeftLREven += 1
          # if there are too many localreads, change strategy to even.
          readLeft = max(readLeftLREven,readLeftLROPT)
        if not self.numItersPLR and iteration < isBarrier:
          for j in range(len(localReadItemsThisLoop)):
            latencyLeft -= localReadItemsThisLoop[j].IssueLatency*2
        # if start to schedule localwrite, but still have localreads not scheduled yet,
        # reject to use 1LDSB, since it will write and read same lds buffer at same time.
        # TODO: force to schedule all remaining localreads before start to schedule localwrite.
        if mfmaIndex >= self.lwStartMfmaIndex and mfmaIndex <= max(self.lwEndMfmaIndex,self.barrierMfmaIndex) and \
          localReadItemsThisLoop and localWriteCode.countType(Code.LocalWriteInst) and kernel["1LDSBuffer"]:
          self.overflowedResources = 5
        # DirectToVgpr case, localReadItemsThisLoop and localWriteCode.countType(Code.LocalWriteInst) do not satisfy at the same time.
        # However, it is still invaid if localReadItemsThisLoop exists when mfmaIndex > lwStartMfmaIndex
        elif (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and \
          mfmaIndex > self.lwStartMfmaIndex and mfmaIndex <= max(self.lwEndMfmaIndex,self.barrierMfmaIndex) and \
          localReadItemsThisLoop and kernel["1LDSBuffer"]:
          self.overflowedResources = 5
        for j in range(readLeft):
          if localReadItemsThisLoop:
            item = localReadItemsThisLoop.pop(0)
            iterCode.addCode(item)
            if (i == 0):
              localReadsWaitcnt += 1
        if not localReadItemsThisLoop and latencyLeft > 0 and iteration < isBarrier and \
            not(mfmaIndex > self.lwStartMfmaIndex and kernel["1LDSBuffer"]):
          item = Code.Module()
          item.addComment0("localReadsVacancy: latencyLeft %d"%(latencyLeft))
          iterCode.addCode(item)
          self.localReadsVacancy.append({ "items": item, \
                                          "latencyLeft": latencyLeft, \
                                          "atIter": iteration, \
                                          "atMfmaIndex": mfmaIndex, \
                                          "noReadsAtThisIter": numReadsInst == 0, \
                                        })

        ####
        # scheduled global read
        ####
        for j in range(self.numGlobalReadInsPerMfma):
          if globalReadCode.items():
            loadModule = globalReadCode.items().pop(0)
            if isDTVodd:
              # need to swap Vgpr set for odd code
              loadModule = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadModule)
            iterCode.addCode(loadModule)
        # schedule remaining globalReadInst
        if mfmaIndex == self.grEndMfmaIndex:
          while globalReadCode.items() and \
              (globalReadCode.countType(Code.GlobalReadInst) or kernel["PrefetchGlobalRead"] == 2):
            loadModule = globalReadCode.items().pop(0)
            if isDTVodd:
              # need to swap Vgpr set for odd code
              loadModule = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadModule)
            iterCode.addCode(loadModule)
        # schedule remaining globalReadIncInst
        if i == numMfmaPerIter - 1:
          while globalReadCode.items():
            loadModule = globalReadCode.items().pop(0)
            if isDTVodd:
              # need to swap Vgpr set for odd code
              loadModule = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadModule)
            iterCode.addCode(loadModule)

        ####
        # scheduled local write
        ####
        if kernel["1LDSBuffer"] and mfmaIndex == self.lwStartMfmaIndex - 1:
          barrier = Code.Module()
          barrier.addComment0("1 LDS buffer: read-sync-write")
          barrier.addWaitCnt(lgkmcnt=0, comment="")
          barrier.addInst("s_barrier","")
          iterCode.addCode(barrier)

        if kernel["StorePriorityOpt"]:
          flagInsert = False
          if kernel["PrefetchGlobalRead"] == 2:
            lwStartOffset = 0
            if kernel["DirectToLds"]:
              lwStartOffset = 2
            #  if (mfmaIndex == self.lwStartMfmaIndex or mfmaIndex == self.barrierMfmaIndex+2):
            if (mfmaIndex == self.lwStartMfmaIndex + lwStartOffset or mfmaIndex == self.barrierMfmaIndex+1) :
              flagInsert = True
          elif kernel["PrefetchGlobalRead"] == 1 and numMfmaPerIter >= 4:
            # this setting is good for fixed clock, but not good for auto clock
            #if (mfmaIndex == self.grEndMfmaIndex or mfmaIndex == self.barrierMfmaIndex+1) :
            withGL = (not NLLlast)
            withDTLload = kernel["DirectToLds"] and withGL
            startIndex = 0 if withDTLload else 1
            if (mfmaIndex == startIndex or withGL and mfmaIndex == self.barrierMfmaIndex+1):
              flagInsert = True
          if flagInsert:
            iterCode.addInst("s_setprio 3","store optimization")

        if (mfmaIndex >= self.lwStartMfmaIndex):
          for j in range(self.numLocalWriteModPerMfma):
            # in case there are localWrite and globalread in same iteration
            # we need to make sure globalRead before localWrite
            if writeItems and not globalReadCode.countType(Code.GlobalReadInst):
              writeItem = writeItems.pop(0)
              iterCode.addCode(writeItem)
              # if there is localWrite at first mfma, need to skip it in waitcnt.
              if i == 0:
                skipLocalWriteWaitcnt += writeItem.countType(Code.LocalWriteInst)
              if not localReadItemsThisLoop:
                self.perIterLocalWriteCanSkip[iteration] += writeItem.countType(Code.LocalWriteInst)
        if mfmaIndex == self.lwEndMfmaIndex:
          while writeItems:
            writeItem = writeItems.pop(0)
            # generate all remaining pre code before the first Store C
            iterCode.addCode(writeItem)
            if i == 0:
              skipLocalWriteWaitcnt += writeItem.countType(Code.LocalWriteInst)
            if not localReadItemsThisLoop:
              self.perIterLocalWriteCanSkip[iteration] += writeItem.countType(Code.LocalWriteInst)

        ####
        # scheduled pointer
        ####
        if mfmaIndex == self.lwEndMfmaIndex:
          iterCode.addCode(pointerLWCode)
        if i == numMfmaPerIter - 1:
          iterCode.addCode(pointerLRCode)

        ####
        # scheduled sync
        ####
        if mfmaIndex == self.barrierMfmaIndex and self.numItersPLR:
          iterCode.addCode(waitLWCode)
          iterCode.addCode(syncCode)

        ####
        # scheduled local read for next loop
        # localReads for next loop should after barrier
        ####
        latencyLeft = self.miLatencyLeft
        if self.numItersPLR and iteration >= isBarrier:
          readLeftLROPT = 0
          for j in range(len(localReadItemsNextLoop)):
            latencyLeft -= localReadItemsNextLoop[j].IssueLatency*2
            readLeftLROPT += 1 if latencyLeft >= 0 else 0
          # at least 1 instruction
          readLeftLROPT = max(readLeftLROPT,1)
          # evenly schedule localread with each mfma
          readLeftLREven = numReadsInst // numMfmaPerIter
          if (numReadsInst % (numMfmaPerIter)) > i:
            readLeftLREven += 1
          # we want no localreads at barrier mfma
          if (iteration == isBarrier) and numMfmaPerIter != 1:
            numMfmaForLR = self.numMfmaForNextLoopLR
            if i < numMfmaPerIter - numMfmaForLR:
              readLeftLREven = 0
              readLeftLROPT = 0
            # rest mfma help to schedule those localReads
            else:
              readLeftLREven = numReadsInst // (numMfmaPerIter-1)
              if (numReadsInst % (numMfmaPerIter-1)) >= i:
                readLeftLREven += 1
          # if there are too many localreads, change strategy to even.
          readLeft = max(readLeftLREven,readLeftLROPT)
        for j in range(readLeft):
          if localReadItemsNextLoop:
            item = localReadItemsNextLoop.pop(0)
            iterCode.addCode(item)
            if (i == 0):
              localReadsWaitcnt += 1

        ####
        # scheduled wait localReads
        ####
        if i == 0:
          iterCode.addCode(waitCode)

        ####
        # scheduled pack
        ####
        if packItems:
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          packAIdx += instPerPack if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packBIdx += instPerPack if i % kernel["MIWaveTileA"] == 0 else 0
          # blockWidth < 1, means 0.5 or 0.25 (BF,H,Int8)
          packAIdx = packAIdx if self.tPA["localReadInstruction"].blockWidth < 1 else 0
          packBIdx = packBIdx if self.tPB["localReadInstruction"].blockWidth < 1 else 0
          numPack = (packAIdx + packBIdx)
          iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))
          # we put 2 pack in each mfma
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPack):
                iterCode.addCode(packItems.pop(0))
                curPackIdx += 1
            else:
              iterCode.addInst("s_nop ","0","VALU packing writes to be consumed by matrix instruction")
              curPackIdx += 1
        if i == numMfmaPerIter - 1:
          while packItems:
            iterCode.addCode(packItems.pop(0))

        ####
        # scheduled mfma
        ####
        iterCode.addCode(macIterItems.pop(0) if macIterItems else Code.Module())

        ####
        # scheduled global read for DirectToVgpr (PGR=2 only)
        ####
        numLoadVgpr = len(list(globalReadCodeDTV.items()))
        if numLoadVgpr > 0:
          interval = roundUp(numMfmaPerIter / origLenGlobalReadCodeDTV)
          tileIndex = 0 if kernel["DirectToVgprA"] else 1
          if (kernel["MIWaveTile"][tileIndex] // kernel["VectorWidth"]) > 1:
            if kernel["ProblemType"]["DataType"].isDoubleComplex():
              # adjustment for double complex
              # limit the max of interval up to 4 if (kernel["MIWaveTile"][0] // kernel["VectorWidth"]) > 1
              interval = min(4, interval)
            elif kernel["ProblemType"]["DataType"].isDouble():
              # adjustment for double
              # in this case, interval must be 1 to avoid overwritting vreg by global read
              interval = 1
          # DirectToVgprA + TLU=False + VW > 1 case, need to use interval = 1
          if kernel["DirectToVgprA"] and (not kernel["ProblemType"]["TLUA"]) and kernel["VectorWidth"] > 1:
            interval = 1
          # if number of mfma after self.grEndMfmaIndex is smaller than numMfmaPerIter, we need to use smaller interval to insert DTV load.
          # this is to ensure DTV load is generated after lwStartMfmaIndex
          intervalAfterGrEnd = kernel["LoopIters"] * numMfmaPerIter - self.lwStartMfmaIndex
          intervalMfma = min(numMfmaPerIter, intervalAfterGrEnd)
          numInstToInsert = roundUp(origLenGlobalReadCodeDTV / intervalMfma)
          remainingTimesToInsert = roundUp(numLoadVgpr / numInstToInsert)
          insertMfmaIndex = kernel["LoopIters"] * numMfmaPerIter - 1 - interval * (remainingTimesToInsert - 1)
          # avoid insertMfmaIndex getting smaller than (kernel["LoopIters"] - 1) * numMfmaPerIter
          insertMfmaIndex = max(insertMfmaIndex, (kernel["LoopIters"] - 1) * numMfmaPerIter)
          if mfmaIndex == insertMfmaIndex:
            for i in range(min(numLoadVgpr, numInstToInsert)):
              loadDTVModule = globalReadCodeDTV.items().pop(0)
              if isDTVodd:
                # need to swap Vgpr set for odd code
                loadDTVModule = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadDTVModule)
              iterCode.addCode(loadDTVModule)

        if kernel["StorePriorityOpt"]:
          flagInsert = False
          if kernel["PrefetchGlobalRead"] == 2:
            #  if (mfmaIndex == self.barrierMfmaIndex or mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)):
            if (mfmaIndex == self.barrierMfmaIndex - 1 or (not NLLlast) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) :
                flagInsert = True
          elif kernel["PrefetchGlobalRead"] == 1 and numMfmaPerIter >= 4:
            # this setting is good for fixed clock, but not good for auto clock
            #if (mfmaIndex == mfmaIndex == self.barrierMfmaIndex - 1 or mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) :
            insertPos1 = self.grEndMfmaIndex
            if not kernel["NoLdsWriteCode"]:
              insertPos1 = self.lwStartMfmaIndex - 1
            withGL = (not NLLlast)
            if withGL and (mfmaIndex == insertPos1 or (not NLLlast) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) or \
               (not withGL) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter // 2 - 1):
              flagInsert = True
          if flagInsert:
            iterCode.addInst("s_setprio 0","store optimization")
    else:
      assert 0, "Unsupported scheduleIterAlg=%u"%self.scheduleIterAlg

    if isinstance(waitCode, Code.WaitCnt):

      # Set the waitCount, based on the new iter schedule
      lgkmcnt = waitCode.lgkmcnt
      localReads = 0
      localWrites = 0
      if kernel["EnableMatrixInstruction"]:
        # dataAtIter      : the data we wait is read at which iteration
        # numReadsIter    : in this loop, number of iteration we have read (data used in current loop)
        dataAtIterA = iteration//self.numIterPerCoalescedReadA - self.numItersPLR
        dataAtIterB = iteration//self.numIterPerCoalescedReadB - self.numItersPLR
        numReadsIterA = min(iteration+1, kernel["LoopIters"]//self.numIterPerCoalescedReadA - self.numItersPLR)
        numReadsIterB = min(iteration+1, kernel["LoopIters"]//self.numIterPerCoalescedReadB - self.numItersPLR)
        skipReadsIterA = numReadsIterA - dataAtIterA - 1 if not dataAtIterA < max(dataAtIterA,dataAtIterB) else 0
        skipReadsIterB = numReadsIterB - dataAtIterB - 1 if not dataAtIterB < max(dataAtIterA,dataAtIterB) else 0
        # numPrefetchIter : in this loop, number of prefetch iteration we have read (data used in next loop)
        # currently we have localReadA and localReadB if iteration >= isBarrier
        # some case will not have localReads if PGR=0 or NoLoadLoop
        # known bug: wider localread + numItersPLR>1 may have chance to fail.
        numPrefetchIter = (iteration//(kernel["LoopIters"]-self.numItersPLR))*((iteration+1)-(kernel["LoopIters"]-self.numItersPLR)) if kernel["PrefetchGlobalRead"] else 0
        numPrefetchIter = 0 if iteration >= isBarrier and not hasLocalRead else numPrefetchIter
        skipReadsIterA += numPrefetchIter
        skipReadsIterB += numPrefetchIter
        # here the reads are prefetches so can skip them in the waitcnt
        # how many localreads can skip is based on how many iterations we prefetch.
        localReads += self.numReadsPerIterA * skipReadsIterA + localReads + self.numReadsPerIterB * skipReadsIterB
        # some of localReads is interleaved after waitcnt in SIA3
        if kernel["ScheduleIterAlg"] == 3 and self.numItersPLR and\
          (iteration < numReadsIterA or iteration < numReadsIterB or numPrefetchIter):
          if (iteration < numReadsIterA and not dataAtIterA < max(dataAtIterA,dataAtIterB)) or numPrefetchIter:
            localReads -= self.numReadsPerIterA
          if (iteration < numReadsIterB and not dataAtIterB < max(dataAtIterA,dataAtIterB)) or numPrefetchIter:
            localReads -= self.numReadsPerIterB
          localReads += localReadsWaitcnt
        lgkmcnt += localReads
        iterCode.addComment0("numPrefetchIter=%u" % numPrefetchIter)
        iterCode.addComment0("dataAtIterA=%u numReadsIterA=%u skipReadsIterA=%u readsPerIterA=%u" % (dataAtIterA, numReadsIterA, skipReadsIterA, self.numReadsPerIterA))
        iterCode.addComment0("dataAtIterB=%u numReadsIterB=%u skipReadsIterB=%u readsPerIterB=%u" % (dataAtIterB, numReadsIterB, skipReadsIterB, self.numReadsPerIterB))
        if kernel["ScheduleIterAlg"] == 0 or kernel["ScheduleIterAlg"] == 1:
          for i in range (max(dataAtIterA,dataAtIterB),iteration+1):
            localWrites += self.perIterLocalWriteCode[i].countType(Code.LocalWriteInst)
        # ScheduleIterAlg=2, localwrite is after waitCnt, no need to count it's current iteration.
        if kernel["ScheduleIterAlg"] == 3:
          for i in range (max(dataAtIterA,dataAtIterB)+1,iteration):
            localWrites += self.perIterLocalWriteCode[i].countType(Code.LocalWriteInst)
          if kernel["ScheduleLocalWrite"] > 0:
            # current iteration localWrite count
            localWrites += skipLocalWriteWaitcnt
            # dataAtIter iteration localWrite count
            if self.numItersPLR:
              skipPreIterLW = self.perIterLocalWriteCanSkip[max(dataAtIterA,dataAtIterB)]
              if kernel["PrefetchGlobalRead"] == 2 and kernel["LocalReadVectorWidth"] == 2 and \
                 (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
                # PGR==2 and LRVW==2 and DirectToVgpr enabled case, count local write before max(dataAtIterA,dataAtIterB)
                # NOTE: This logic assumes that local write is scheduled after local read.
                for up in range(max(dataAtIterA,dataAtIterB)):
                  skipPreIterLW += self.perIterLocalWriteCanSkip[up]
              localWrites += skipPreIterLW
        lgkmcnt += localWrites
      else:
        for item in list(iterCode.items()):
          localReads  = item.countType(Code.LocalReadInst)
          localWrites = item.countType(Code.LocalWriteInst)
          if self.numVgprBuffer:
            # SQ: If PrefetchLocalRead = 1 and DepthU == LocalSplitU, then there is no double
            #  buffering and we must wait for all localReads but not localWrites.
            #  In that case, LoopIters == 1:
            if kernel["LoopIters"] > 1:
              # here the reads are prefetches so can skip them in the waitcnt
              lgkmcnt += localReads
            # and the writes are targetting another section of LDS and are
            # synchronized through a different waitcnt than this one
            # (which is always just before the macs)
            lgkmcnt += localWrites
          else:
            # if UnrollLoopEfficiencyEnable == True  use waitCode passed lgkmCnt
            # else:
            # we need to wait for all preceding reads before the macs
            # so only opportunity for optimization is if the writes are at the end
            if globalParameters["UnrollLoopEfficiencyEnable"]:
              lgkmcnt = waitCode.lgkmcnt
            else:
              if localReads:
                lgkmcnt = 0 # reset to wait for all reads
              else:
                lgkmcnt = localWrites  # this only survives if writes are at the end

      waitCode.comment += " old=%u, new=%u newLW=%u newLR=%u" % (waitCode.lgkmcnt, lgkmcnt,localWrites,localReads)
      waitCode.lgkmcnt = lgkmcnt
      # This line is added for backward compatibility
      waitCode.vscnt = waitCode.vmcnt if waitCode.lgkmcnt != -1 and waitCode.vmcnt != -1 else -1

    return iterCode

  ##############################################################################
  # returns list of modules or text
  ##############################################################################
  def setupNewTile(self, kernel, tensorParametersA, tensorParametersB, isOptNLL=False, forceNoTileCode=False, forceNoGRCode=False):
    module = Code.Module("setupNewTile")

    ####################################
    # Global Read Addresses
    ####################################
    module.addComment2("Begin setupNewTile")

    # work-group assignments
    module.addComment1("global read addresses: work-group")
    if not forceNoTileCode:
      module.addCode(self.graWorkGroup(kernel))

    self.dontAppendCode = forceNoTileCode
    # tile assignments
    module.addComment1("global read addresses: tile offset assignment a")
    module.addCode(self.graTileAssignment(kernel, tensorParametersA))
    module.addComment1("global read addresses: tile offset assignment b")
    module.addCode(self.graTileAssignment(kernel, tensorParametersB))

    # unroll assignments
    module.addComment1("global read addresses: unroll assignment a")
    module.addCode(self.graUnrollAssignment(kernel, tensorParametersA))
    module.addComment1("global read addresses: unroll assignment b")
    module.addCode(self.graUnrollAssignment(kernel, tensorParametersB))

    # other free indices
    if kernel["ProblemType"]["NumIndicesC"] > 2:
      module.addComment1("global read addresses: other free assignments")
      module.addCode(self.graOtherFreeAssignments(kernel))

    # other summation indices
    if self.otherSummations:
      module.addComment1("global read addresses: other summation assignments")
      module.addCode(self.graOtherSummationAssignments(kernel))

    # tile offsets
    module.addComment1("global read addresses: tile offsets a")
    module.addCode(self.graTileOffsets(kernel, tensorParametersA))
    module.addComment1("global read addresses: tile offsets b")
    module.addCode(self.graTileOffsets(kernel, tensorParametersB))

    # unroll offsets
    module.addComment1("global read addresses: unroll offsets a")
    module.addCode(self.graUnrollOffsets(kernel, tensorParametersA))
    module.addComment1("global read addresses: unroll offsets b")
    module.addCode(self.graUnrollOffsets(kernel, tensorParametersB))

    # tile edges
    if kernel["EdgeType"] == "ShiftPtr":
      # Shift here has two purposes:
      #  1. Ensure the loads are in-bounds to prevent fault.
      #     BufferLoad uses the buffer limit hardware and does not require bounds checking for this case
      #  2. Shift-left a wide vector load to ensure it is completely in-bounds.
      #     If this occurs we need to 'unshift' the C values (see shiftVectorComponents)
      #     BufferLoad does support this shifting, but if GuaranteeNoPartial=1 then
      #     it can be guaranteed that no shifting is required.
      if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"]) and not forceNoTileCode:
        module.addComment1("global read addresses: shift a")
        module.addCode(self.graShift(kernel, tensorParametersA))
      if not (kernel["BufferLoad"] and  kernel["GuaranteeNoPartialB"]) and not forceNoTileCode:
        module.addComment1("global read addresses: shift b")
        module.addCode(self.graShift(kernel, tensorParametersB))

    # final offsets
    module.addComment1("global read addresses: final offsets a")
    module.addCode(self.graFinalOffsets(kernel, tensorParametersA))
    module.addComment1("global read addresses: final offsets b")
    module.addCode(self.graFinalOffsets(kernel, tensorParametersB))
    self.dontAppendCode = False
    self.dontAppendCode = self.dontAppendCode or forceNoTileCode

    # addresses
    if not forceNoTileCode:
      module.addComment1("global read addresses: addresses a")
      module.addCode(self.graAddresses(kernel, tensorParametersA))
      module.addComment1("global read addresses: addresses b")
      module.addCode(self.graAddresses(kernel, tensorParametersB))

    # increments
    module.addComment1("global read addresses: increments a")
    for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
      module.addCode(self.graIncrements(kernel, i, tensorParametersA))
    module.addComment1("global read addresses: increments b")
    for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
      module.addCode(self.graIncrements(kernel, i, tensorParametersB))

    ####################################
    # Local Write Addresses
    ####################################
    module.addComment2("Local Write Addresses")

    # tile assignments
    module.addCode(self.lwaTileAssignment(kernel, tensorParametersA))
    module.addCode(self.lwaTileAssignment(kernel, tensorParametersB))

    # unroll assignments
    module.addCode(self.lwaUnrollAssignment(kernel, tensorParametersA))
    module.addCode(self.lwaUnrollAssignment(kernel, tensorParametersB))

    # first offsets
    module.addComment1("local write addresses: first offset a")
    module.addCode(self.lwaFirstOffset(kernel, tensorParametersA))
    module.addComment1("local write addresses: first offset b")
    module.addCode(self.lwaFirstOffset(kernel, tensorParametersB))
    self.dontAppendCode = False
    self.dontAppendCode = self.dontAppendCode or forceNoTileCode

    ###########################################################################
    # summations loops: open
    ###########################################################################

    # declare loop num iter
    if not forceNoTileCode:
      module.addComment0("declare loop num iterations")

    # perform initC in the shadow of the prefetch
    # Prefetch occurs at start of unroll loop
    # If we have multiple summation indices (otherSummationLoops>0),
    # we can't init in shadow of this prefetch
    # since that would initC inside the other summation loops

    if self.doShadowInit != 2:
      module.addCode(self.initC(kernel))

    # open non-unrolled summation loops
    if not forceNoTileCode:
      for i in range(kernel["ProblemType"]["NumIndicesSummation"]-1):
        module.addComment1("summation loop %u"%i)
        module.addCode(self.calculateLoopNumIter(kernel, i))
        if self.actualSummationLoops>1:
          module.addCode(self.openLoop(kernel, i))
      module.addCode(self.calculateLoopNumIter(kernel, self.unrollIdx))

    if not forceNoTileCode:
      if self.staggerU:
        module.addCode(self.declareStaggerParms(kernel))
        module.addCode(self.calculateStagger(kernel, tensorParametersA))
        module.addCode(self.calculateStagger(kernel, tensorParametersB))

    # LRO and LWA as assigned
    # init lds read pointers before each unrolled loop
    module.addComment0("local read addresses: init pointers a")
    module.addCode(self.localReadInitPointers(kernel, tensorParametersA))
    module.addComment0("local read addresses: init pointers b")
    module.addCode(self.localReadInitPointers(kernel, tensorParametersB))

    ####################################
    # prefetch: unrolled loop prefix
    ####################################
    if kernel["PrefetchGlobalRead"]:
      pfi = 1
      module.addComment1("prefetch: global -> local")
      module.addCode(self.openSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=isOptNLL))
      # if DirectToVgprA is enabled, swap the order of global read (B->A)
      tensorParameters1st = tensorParametersA
      tensorParameters2nd = tensorParametersB
      if kernel["DirectToVgprA"]:
        tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
      moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters1st, usePlaceHolder=False)
      module.addCode(replacePlaceHolder(moduleTmp, "__placeholder__", 0))
      module.addCode(self.globalReadDo(kernel, 0, tensorParameters1st, 0))
      moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters2nd, usePlaceHolder=False)
      module.addCode(replacePlaceHolder(moduleTmp, "__placeholder__", 0))
      module.addCode(self.globalReadDo(kernel, 0, tensorParameters2nd, 0))
      module.addCode(self.globalReadIncrementAB(kernel, self.unrollIdx, pfi))

    module.addComment2("End setupNewTile")

    return module


  ##############################################################################
  # get conditions to skip local write wait
  ##############################################################################
  def getConditionToSkipLocalWriteWait( self, kernel, u):
    cond1 = not (u == 0 and kernel["PrefetchLocalRead"] != 0 and \
       (kernel["DirectToVgprA"] and kernel["DirectToLdsB"] or kernel["DirectToVgprB"] and kernel["DirectToLdsA"])) \
      or kernel["PrefetchGlobalRead"]==2
    # no need local read wait if LocalReadVectorWidth==2 and u is odd.
    # In that case, Prefetch local read covers both u = 0 and 1 (limit to MFMA+double+DirectToVgpr only)
    # (The other side of numReadsIterCoalesced must be 0 to skip local read wait)
    condSkip = kernel["LocalReadVectorWidth"]==2 and (u%2 != 0) and kernel["EnableMatrixInstruction"] and \
               kernel["ProblemType"]["DataType"].isDouble() and \
              (kernel["DirectToVgprA"] and self.numReadsIterCoalescedB % 2 == 0 or \
               kernel["DirectToVgprB"] and self.numReadsIterCoalescedA % 2 == 0)
    return cond1 and (not condSkip)

  ##############################################################################
  # No Load Loop Body
  ##############################################################################
  def noLoadLoopBody( self, kernel, tensorParametersA, tensorParametersB, pack, isOptNLL, isNGLL, NLLfirst, NLLlast, isDTVodd=False):
    module = Code.Module("noLoadLoopBody")
    expand = kernel["ExpandPointerSwap"]
    lastuIdx = False
    pflr     = self.numItersPLR
    localWriteEndIter = kernel["LoopIters"] - self.numItersPLR - 1

    for uIdx in range(0, kernel["LoopIters"]*kernel["DepthULdsDivisor"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      uDu = uIdx // kernel["LoopIters"] # uDu: index of compute loop
      isLastLoop = (uDu == kernel["DepthULdsDivisor"] -1 ) and not isNGLL
      if u == 0:
        if uDu > 0:
          assert len(self.globalReadACode.items()) > 0 and len(self.globalReadBCode.items()) > 0 # already issued in first uDu
          self.globalReadACode = Code.StructuredModule() # empty
          self.globalReadBCode = Code.StructuredModule() # empty
          self.globalReadIncrements = Code.Module() # empty
          self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
          self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))
        if not isLastLoop:
          self.localWriteACode = self.localWriteDo(kernel, tensorParametersA, (uDu+1)%kernel["DepthULdsDivisor"])  # local write in loopcnt N targets data for loopcnt N+1
          self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB, (uDu+1)%kernel["DepthULdsDivisor"])
        else:
          self.localWriteACode = Code.Module()
          self.localWriteBCode = Code.Module()

        # TODO schedule waitcnt/barrier in makeSubIterSchedule()
        if kernel["PrefetchGlobalRead"] and kernel["LoopIters"] in [1, 2] and uDu > 0:
          module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "wait for local write"))
          module.addCode(self.syncThreads(kernel, "sync for local read after write"))

        if not isNGLL:
          # PAP would have GlobalRead and GlobalInc, but no localWrite
          # Get the perIterGlobalReadCode code for PAP (if PAP=On), else would be empty
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu, skipGlobalReadInc=False, lastLoop=NLLlast)
          module.addCode(self.unrollLoopHeaderCode)

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = (u == localWriteEndIter)
      isSwapAndResetLwoIter = isResetLroIter
      isSwapLroIter = isResetLroIter
      if kernel["ScheduleIterAlg"] == 3:
          isSwapAndResetLwoIter = (u == self.lwEndMfmaIndex//(self.numMfmaPerIter))

      extraComment = ""
      if isLastLoop:
        extraComment += " (last unrolled loop)"
      else:
        if kernel.enabledSplitLDS:
            extraComment += f" (uDu={uDu}) "
        if isResetLroIter:
            extraComment += " (reset local read pointers iteration) "
        if isSwapAndResetLwoIter:
            extraComment += " (swap and reset local write pointers iteration) "
        if isSwapLroIter:
            extraComment += " (swap local read pointers iteration) "

      module.addComment1("iter %u%s"%(u,extraComment))
      plrIdx = ((u+pflr) % (self.numVgprBuffer+1)) % kernel["LoopIters"]
      localReads = Code.Module()

      pointerLWCode = Code.Module()
      pointerLRCode = Code.Module()
      waitCode = Code.Module()  # may be overwritten (not added to) below
      macIterCode = Code.Module()
      waitLWCode = Code.Module()
      syncCode = Code.Module()

      hasLiveLdsData = kernel["PrefetchGlobalRead"] or (uDu < kernel["DepthULdsDivisor"]-1)
      hasLiveLdsData = hasLiveLdsData and not isLastLoop
      # reads for current loop are done in previous iteration because of wider local read
      doReadA = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadA - self.numItersPLR)
      doReadB = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadB - self.numItersPLR)
      # reads for next loop
      doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
      doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
      # disable LocalRead if DirectToVgpr is enabled
      doReadA = doReadA and (not kernel["DirectToVgprA"])
      doReadB = doReadB and (not kernel["DirectToVgprB"])
      for iui in range(0,kernel["InnerUnroll"]):
        doReadA = doReadA and iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]
        doReadB = doReadB and iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]
        if doReadA:
          localReads.addComment1("local read a")
          localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
          localReads.addCode(localReadCodeA)
          pack[plrIdx*self.numIterPerCoalescedReadA].addCode(packCodeA)
        if doReadB:
          localReads.addComment1("local read b")
          localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
          localReads.addCode(localReadCodeB)
          pack[plrIdx*self.numIterPerCoalescedReadB].addCode(packCodeB)
        if (not isResetLroIter or iui != kernel["InnerUnroll"]-1):
          if doReadA:
            localReads.addComment1("local read increment a")
            localReads.addCode(self.localReadInc(kernel, iui, tensorParametersA))
          if doReadB:
            localReads.addComment1("local read increment b")
            localReads.addCode(self.localReadInc(kernel, iui, tensorParametersB))

      if not isLastLoop:
        if kernel["PrefetchGlobalRead"]:
          # put barrier at localWriteEndIter+1
          if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
            # skip local write wait if DirectToVgpr + DirectToLds is enabled
            if not kernel["NoLdsWriteCode"]:
              waitLWCode.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
            if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
              # DirectToVgpr + DirectToLds case, add waitcnt vmcnt before s_barrier
              retInst = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, firstIter=False, beforeBarrier=True)
              waitLWCode.addCode(retInst)
            syncCode.addCode(self.syncThreads(kernel))

          if isSwapAndResetLwoIter: # ResetLroIter
            # local write for next iter, used to have local writes here
            pointerLWCode.addComment1("local write swap offsets a")
            pointerLWCode.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
            pointerLWCode.addComment1("local write swap offsets b")
            pointerLWCode.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

          if isSwapLroIter: # ResetLroIter
            # Swap, reset, or increment the LRO:
            # force internalPointerSwap = False in NGLL case
            internalPointerSwap = expand and not isNGLL
            pointerLRCode.addComment1("local read swap offsets a")
            pointerLRCode.addCode(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersA))
            pointerLRCode.addComment1("local read swap offsets b")
            pointerLRCode.addCode(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersB))

        if isResetLroIter: # ResetLroIter
          pointerLRCode.addComment1("local read init pointers a")
          pointerLRCode.addCode(self.localReadInitPointers(kernel, tensorParametersA))
          pointerLRCode.addComment1("local read init pointers b")
          pointerLRCode.addCode(self.localReadInitPointers(kernel, tensorParametersB))

      # we initiate lgkmcnt to 0, then assigning it correct value in makeSubIterSchedule()
      waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, \
          -1, 0, 0, \
          "wait for prior local read local write")
      # DirectToVgpr case, wait for global read as well as local read/write
      if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
        # not generate wait here
        #  1) local write code in previous u (u-1) has waitcnt vmcnt
        prevVmcnt = False
        prevLocalWrite = ""
        if (u > 0):
          prevLocalWrite = ' '.join([str(x) for x in self.perIterLocalWriteCode[u-1].flatitems()])
          prevVmcnt = "vmcnt" in prevLocalWrite
        if not prevVmcnt:
          retInst = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, False, isNGLL, NLLlast=NLLlast)
          module.addCode(retInst)

      luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
      if kernel["EnableMatrixInstruction"]:
        # NGLL case, use first set
        setId = 0 if isNGLL else 1
        # flip setId if isDTVodd is True
        if isDTVodd:
           setId = 1 - setId
        # use second set for DirectToVGPR
        vregSetIdxMFMA = setId # use first set for NGLL, second set for other cases
        macIterCode.addCode(self.mfmaIter(kernel, u, kernel["InnerUnroll"], vregSetIdxMFMA,lastuIdx))
      else:
        printExit("TensileLite does not support MAC instructions.")

      subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                      u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx], isDTVodd, NLLlast)
      module.addCode(subIterCode)
      # vgpr.checkin for all the checked-out vgpr in LocalRead
      for item in list(pack[luIdx].items()):
        if item.tempVgpr != None:
          self.vgprPool.checkIn(item.tempVgpr)
          item.tempVgpr = None
      pack[luIdx] = Code.Module()
    return module

  ##############################################################################
  # noLoadLoop
  # Create the no load loop (NLL)
  #
  # isOptNLL : the NLL is to be optimized for the alpha=1 and non-edge case
  ##############################################################################
  def noLoadLoop( self, kernel, tensorParametersA, tensorParametersB, isOptNLL, isNGLL, pack ):
    module = Code.Module("noLoadLoop")
    LoopNameComment = "NoGlobalLoadLoop" if isNGLL else "NoLoadLoop"
    isOptNLLComment = "Opt" if isOptNLL else "Ord"
    startComment = "%s. %s - Begin " % (isOptNLLComment, LoopNameComment)
    module.addComment2(startComment)
    NLLfirst = True
    NLLlast = True
    if kernel["PrefetchGlobalRead"] == 2:
      # PGR=2 case NoLoadLoop(NLL) is generated twice
      # we need to distinguish them to generate proper code at each NLL
      if isNGLL:
        NLLlast = False
      else:
        # PGR=2 and not isNGLL means second NoLoadLoop for PGR2.
        # Need to avoid generating duplicated code which is already generated in NGLL(first NoLoadLoop for PGR=2)
        NLLfirst = False
    if isNGLL:
      self.perIterLocalWriteCode = self.perIterLocalWriteCodeNGLL
      self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
    #else:
    if not isNGLL:
      self.dtlsM0UpdateACode = Code.StructuredModule()
      self.globalReadACode = Code.StructuredModule() # empty
      self.dtlsM0UpdateBCode = Code.StructuredModule()
      self.globalReadBCode = Code.StructuredModule() # empty
      self.globalReadIncrements = Code.Module()
      self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
      self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))
      self.localWriteACode = Code.Module()
      self.localWriteBCode = Code.Module()

    kStrOpenSum = self.openSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=isOptNLL)

    # skip generating OpenSum code here for SingleNLLOpt
    if not (isOptNLL and self.enableSingleNLLOpt):
      module.addCode(kStrOpenSum)
      kStrOpenSum = "" # empty OpenSum str to avoid inserting it again

    if not self.numItersPLR:
      if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
        module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "10wait for global read"))
      # TODO: need to check if we correctly checked-in the temp VGPR used for Int8 LocalWrite (uDu, PGR=2)
      module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "4wait for local write"))
      module.addCode(self.syncThreads(kernel))

    # if DirectToVgpr and  ASEM is not multiple of DepthU*2, generate noLoadLoopBody twice for odd and even exit separately
    if ( kernel["DirectToVgprA"] or  kernel["DirectToVgprB"]) and (kernel["AssertSummationElementMultiple"] % (kernel["DepthU"] * 2) != 0):
      # generate additional No Load Loop Body code for odd case (to use the other Vreg set for DirectToVgpr)
      # 1. generate odd check
      name = ""
      if isNGLL:
        name += "NoGlobalLoadLoop"
      else:
        name += "NoLoadLoop"
      if isOptNLL:
        name += "Opt"
      else:
        name += "Ord"
      module.addCode(self.openOddNoLoadLoopForDTV(kernel, isNGLL, name))
      # 2. generate  no Load Loop Body code for odd
      # backup
      self.saveLocalPointers(kernel)
      # deepCopy packCode for OptNLL noLoadLoop
      deepCopyPack = copy.deepcopy(pack)
      module.addCode(self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, deepCopyPack, isOptNLL, isNGLL, NLLfirst, NLLlast, isDTVodd=True))
      # restore
      self.restoreLocalPointers(kernel)
      # 3. generate even start label
      module.addCode(self.closeOddNoLoadLoopForDTV(kernel, isNGLL, name))
      # 4. generate  no Load Loop Body code for odd
      # need to re-initialize perIterLocalWriteCanSkip to avoid having incorrect lgkmcnt
      self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
      module.addCode(self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, pack, isOptNLL, isNGLL, NLLfirst, NLLlast))
      # 5. generate even end label
      module.addCode(self.generateEvenEndLabeNoLoadLoopForDTV(kernel, isNGLL, name))
    else:
      # generate no Load Loop Body code
      module.addCode(self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, pack, isOptNLL, isNGLL, NLLfirst, NLLlast))

    # add OpenSum code here if it is not empty
    if kStrOpenSum != "":
      module.addCode(kStrOpenSum)

    # Close code is necessary for both first and last (NGLL case(=NLLfirst) needs label)
    module.addCode(self.closeSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=isOptNLL, isNGLL=isNGLL))

    return module

  ##############################################################################
  # Loop Body
  ##############################################################################
  def loopBody( self, kernel, tensorParametersA, tensorParametersB, pack, lc, loopCopies, finalLoop, firstIter=False ):
    module = Code.Module("loopBody")
    expand = kernel["ExpandPointerSwap"]

    # not generate openLoop for firstIter
    if not firstIter:
      module.addComment2("Unrolled Loop %u/%u - Begin" % (lc+1, loopCopies))
      module.addCode(self.openLoopCopy(kernel, lc))
    if kernel["PrefetchGlobalRead"] and not self.numItersPLR and not kernel["ScheduleIterAlg"] == 2:
      if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
        module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "11wait for global read"))
      module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "1wait for local write"))
      module.addCode(self.syncThreads(kernel, "4sync for global read"))

    module.addComment1("Begin Each Unroll: Check VGPR.checkin for INT8 LW")

    # if DirectToVgprA is enabled, swap the order of global read (B->A)
    tensorParameters1st = tensorParametersA
    tensorParameters2nd = tensorParametersB
    tc1 = 'A'
    tc2 = 'B'
    if kernel["DirectToVgprA"]:
      tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
      tc1, tc2 = tc2, tc1
    # unrolled loop: global read A, B
    # M0 update for directToLds
    vregSetIdxGR = 0
    if (kernel["DirectToVgpr%s"%tc1]):
      vregSetIdxGR = (kernel["PrefetchGlobalRead"] + lc ) % 2 # toggle vreg set for DirectToVgpr.
    self.dtlsM0UpdateACode = self.directToLdsM0Update(kernel, 1, tensorParameters1st, usePlaceHolder=True)
    self.globalReadACode  = self.globalReadDo(kernel, 1, tensorParameters1st, vregSetIdxGR)
    vregSetIdxGR = 0
    if (kernel["DirectToVgpr%s"%tc2]):
      vregSetIdxGR = (kernel["PrefetchGlobalRead"] + lc ) % 2 # toggle vreg set for DirectToVgpr.
    self.dtlsM0UpdateBCode = self.directToLdsM0Update(kernel, 1, tensorParameters2nd, usePlaceHolder=True)
    self.globalReadBCode = self.globalReadDo(kernel, 1, tensorParameters2nd, vregSetIdxGR)

    # unrolled loop: increment global read addresses
    self.globalReadIncrements = self.globalReadIncrementAB(kernel, self.unrollIdx, 0)

    if not kernel["NoLdsWriteCode"]:
      self.localWriteACode = self.localWriteDo(kernel, tensorParametersA)
      self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB)
    else:
      self.localWriteACode = Code.Module()
      self.localWriteBCode = Code.Module()

    # localWriteEndIter is used to determine which iteration to put sync
    # if PGR=0, GR,LW,sync,LR will put at front of loop.
    localWriteEndIter = kernel["LoopIters"] - self.numItersPLR - 1

    # Schedule the global read, global read inc, and writes:
    unrollLoopHeaderCodeScheduled = False
    if not kernel["PrefetchGlobalRead"]:
      unrollLoopHeaderCodeScheduled = True
      self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter=firstIter)
      module.addCode(self.unrollLoopHeaderCode)

    # if not prefetch global, localWrite before mac's
    if not kernel["PrefetchGlobalRead"]:
      # unrolled loop: local write A, B
      module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "5wait for global read"))
      module.addCode(self.syncThreads(kernel, "PGR=0, prior iter done reading lds"))
      if not kernel["NoLdsWriteCode"]:
        module.addComment1("local write a")
        tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA)
        module.addCode(tempLWCodeModA)
        module.addComment1("local write b")
        tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB)
        module.addCode(tempLWCodeModB)
      module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
      module.addCode(self.syncThreads(kernel))
      # debug Local state
      """
      module.addCode("    /* print Local state */" + self.endLine)
      module.addCode("    for (unsigned int i = serial; i < LDS_NUM_ELEMENTS; i+=NUM_THREADS) {%s" % self.endLine)
      module.addCode("      printf(\\\"localMemory[%%06u] = %%.0f\\\\n\\\", i, localMemory[i]);%s" )
          % self.endLine
      module.addCode("    }" + self.endLine)
      """

    # unrolled loop: prefetch local
    if self.numItersPLR and not kernel["PrefetchGlobalRead"]:
      for plrIdx in range(0, self.numItersPLR):
        pack[plrIdx] = Code.Module()
        for iui in range(0,kernel["InnerUnroll"]):
          if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"] and (not kernel["DirectToVgprA"]) : # no local read code if DirectToVgpr is enabled
            module.addComment1("prefetch local a")
            localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
            module.addCode(localReadCodeA)
            pack[plrIdx].addCode(packCodeA)
          if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"] and (not kernel["DirectToVgprB"]) : # no local read code if DirectToVgpr is enabled
            module.addComment1("prefetch local b")
            localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
            module.addCode(localReadCodeB)
            pack[plrIdx].addCode(packCodeB)
          if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"] and (not kernel["DirectToVgprA"]) : # no local read code if DirectToVgpr is enabled
            module.addComment0("local read increment a")
            module.addCode(self.localReadInc(kernel, iui, tensorParametersA))
          if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]  and (not kernel["DirectToVgprB"]) : # no local read code if DirectToVgpr is enabled
            module.addComment0("local read increment b")
            module.addCode(self.localReadInc(kernel, iui, tensorParametersB))

    pflr     = self.numItersPLR  # how many pf already done above

    ############################################################################
    # unrolled loop: mac iterations
    ############################################################################

    # double/quadruple the number of compute loop for each DepthU's worth of data read
    for uIdx in range(0, kernel["LoopIters"]*kernel["DepthULdsDivisor"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      uDu = uIdx // kernel["LoopIters"] # uDu: index of compute loop
      if u==0: # if at start of subloop...
        # ...update local write code
        if not kernel["NoLdsWriteCode"]:
          self.localWriteACode = self.localWriteDo(kernel, tensorParametersA, (uDu+1)%kernel["DepthULdsDivisor"])  # local write in loopcnt N targets data for loopcnt N+1
          self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB, (uDu+1)%kernel["DepthULdsDivisor"])
        else:
          self.localWriteACode = Code.Module()
          self.localWriteBCode = Code.Module()

        # TODO schedule waitcnt/barrier in makeSubIterSchedule()
        if kernel["PrefetchGlobalRead"] and kernel["LoopIters"] in [1, 2] and uDu > 0:
          module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "wait for local write"))
          module.addCode(self.syncThreads(kernel, "sync for local read after write"))

        if not unrollLoopHeaderCodeScheduled:
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu, firstIter=firstIter, lastLoop=False, lastLc=(lc==loopCopies-1))
          module.addCode(self.unrollLoopHeaderCode)

      # for PGR=0 where generator can't schedule the instructions (yet),
      # we duplicate the local write codegen and append to string list directly
      if not kernel["PrefetchGlobalRead"]:
        doWrite = False
        if uDu<kernel["DepthULdsDivisor"]-1 and u==kernel["LoopIters"]-self.numItersPLR:
          doWrite = True
          writeForNextLoop = 1
        if uDu>0 and self.numItersPLR==0 and u==0:
          assert doWrite==False # should be exclusive with the previous condition
          doWrite = True
          writeForNextLoop = 0
        # unrolled loop: local write A, B
        if doWrite:
          module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "5wait for local read"))
          module.addCode(self.syncThreads(kernel, "PGR=0, prior iter done reading lds"))
          if not kernel["NoLdsWriteCode"]:
            module.addComment1("local write a")
            tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA, (uDu+writeForNextLoop)%kernel["DepthULdsDivisor"])
            module.addCode(tempLWCodeModA)
            module.addComment1("local write b")
            tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB, (uDu+writeForNextLoop)%kernel["DepthULdsDivisor"])
            module.addCode(tempLWCodeModB)
          module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
          module.addCode(self.syncThreads(kernel))

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = (u == localWriteEndIter)
      isSwapAndResetLwoIter = isResetLroIter
      isSwapLroIter = isResetLroIter
      if kernel["ScheduleIterAlg"] == 3:
        isSwapAndResetLwoIter = (u == self.lwEndMfmaIndex//(self.numMfmaPerIter))
      extraComment = ""
      if kernel.enabledSplitLDS:
        extraComment += f" (uDu={uDu}) "
      if isResetLroIter:
        extraComment += " (reset local read pointers iteration) "
      if isSwapAndResetLwoIter:
        extraComment += " (swap and reset local write pointers iteration) "
      if isSwapLroIter:
        extraComment += " (swap local read pointers iteration) "

      module.addComment1("iter %u%s"%(u,extraComment))
      plrIdx = ((u+pflr) % (self.numVgprBuffer+1)) % kernel["LoopIters"]

      localReads = Code.Module()
      localReadsA = Code.Module()
      localReadsB = Code.Module()

      pointerLWCode = Code.Module()
      pointerLRCode = Code.Module()
      waitCode = Code.Module()  # may be overwritten (not added to) below
      macIterCode = Code.Module()
      waitLWCode = Code.Module()
      syncCode = Code.Module()

      hasLiveLdsData = kernel["PrefetchGlobalRead"] or (uDu < kernel["DepthULdsDivisor"]-1)
      # reads for current loop are done in previous iteration because of wider local read
      doReadA = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadA - self.numItersPLR)
      doReadB = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadB - self.numItersPLR)
      # reads for next loop
      doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
      doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
      # disable LocalRead if DirectToVgpr is enabled
      doReadA = doReadA and (not kernel["DirectToVgprA"])
      doReadB = doReadB and (not kernel["DirectToVgprB"])
      # double the number of VgprValu if self.vgprValuDouble is true
      plrIdxLR = plrIdx
      if self.vgprValuDouble and (lc & 1) == 0:
        # use the next buffer set (do not change the index of pack[])
        plrIdxLR += 1
      for iui in range(0,kernel["InnerUnroll"]):
        doReadA = doReadA and iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]
        doReadB = doReadB and iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]
        if doReadA:
          localReads.addComment1("local read a")
          localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdxLR*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
          localReads.addCode(localReadCodeA)
          localReadsA.addCode(localReadCodeA)
          pack[plrIdx*self.numIterPerCoalescedReadA].addCode(packCodeA)
        if doReadB:
          localReads.addComment1("local read b")
          localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdxLR*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
          localReads.addCode(localReadCodeB)
          localReadsB.addCode(localReadCodeB)
          pack[plrIdx*self.numIterPerCoalescedReadB].addCode(packCodeB)
        # Don't increment the LRO if we are going to reset them below:
        if not isResetLroIter or iui != kernel["InnerUnroll"]-1:
          if doReadA:
            localReads.addComment1("local read increment a")
            localReads.addCode(self.localReadInc(kernel, iui, tensorParametersA))
          if doReadB:
            localReads.addComment1("local read increment b")
            localReads.addCode(self.localReadInc(kernel, iui, tensorParametersB))

      if kernel["PrefetchGlobalRead"]:
        # wait code for DirectToVgpr
        if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
          # not generate wait here
          #  1) local write code in previous u (u-1) has waitcnt vmcnt
          prevVmcnt = False
          prevLocalWrite = ""
          if (u > 0):
            for up in range(u):
              prevLocalWrite += ' '.join([str(x) for x in self.perIterLocalWriteCode[up].flatitems()])
            prevVmcnt = "vmcnt" in prevLocalWrite
          if not prevVmcnt:
            retInst = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, firstIter)
            module.addCode(retInst)
        # put barrier at localWriteEndIter+1
        if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
          if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
            # skip generating wait for global read again here in DirectToVgpr case
            if not(kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
              module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "12wait for global read"))
            else:
              # DirectToVgpr + DirectToLds case, add waitcnt vmcnt before s_barrier
              retInst = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, firstIter, beforeBarrier=True)
              waitLWCode.addCode(retInst)
          # skip local write wait if DirectToVgpr + DirectToLds is enabled
          # (no local write code. Global read wait for DirectToLds is already done)
          if not kernel["NoLdsWriteCode"]:
            waitLWCode.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
          if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
            # put only barrier for DirectToVgpr (to avoid generating waitcnt for global read)
            syncCode.addCode("s_barrier" + self.endLine)
          else:
            syncCode.addCode(self.syncThreads(kernel))

        if isSwapAndResetLwoIter: # ResetLroIter
          # local write for next iter, used to have local writes here
          pointerLWCode.addComment1("local write swap offsets a")
          pointerLWCode.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
          pointerLWCode.addComment1("local write swap offsets b")
          pointerLWCode.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

        if isSwapLroIter: # ResetLroIter
          # Swap, reset, or increment the LRO:
          pointerLRCode.addComment1("local read swap offsets a")
          pointerLRCode.addCode(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
          pointerLRCode.addComment1("local read swap offsets b")
          pointerLRCode.addCode(self.localReadSwapOffsets(kernel, expand, tensorParametersB))

      if isResetLroIter: # ResetLroIter
        pointerLRCode.addComment1("local read init pointers a")
        pointerLRCode.addCode(self.localReadInitPointers(kernel, tensorParametersA))
        pointerLRCode.addComment1("local read init pointers b")
        pointerLRCode.addCode(self.localReadInitPointers(kernel, tensorParametersB))

      # we initiate lgkmcnt to 0, then assigning it correct value in makeSubIterSchedule()
      if self.getConditionToSkipLocalWriteWait(kernel, u):
        waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, \
            -1, 0, 0, \
            "wait for prior local read local write")

      luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
      if kernel["EnableMatrixInstruction"]:
        vregSetIdxMFMA = lc
        macIterCode.addCode(self.mfmaIter(kernel, u, kernel["InnerUnroll"], vregSetIdxMFMA, firstIter=firstIter and u == 0))
      else:
        printExit("TensileLite does not support MAC instructions.")

      ###### unroll loop efficiency implementation######################################
      # unroll loop efficiency implementation
      ## split A&B fetch&MAC code into multiple groups
      ## splitting strategy   based on TT size
      ## 6x4 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
      ##        LDS fetch(es) into group of A{1-2)B(0) , A(3),B(1) (not implemented yet)
      ## 4x6 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
      ##        LDS fetch(es) into group of B{1-2)A(0) , B(3),A(1)
      ## 4x4 -> split into group of 8 and 8  MAC(s)
      ## 6x6 -> split into group of 12 MAC(s)
      ## 8x4/4x8 -> split into group of 16 and 16  MAC(s)
      ## 8x8 -> split into group of 16 MAC(s)
      ## supports only PLR=0
      ###############################################################################
      if self.numItersPLR or (not globalParameters["UnrollLoopEfficiencyEnable"]):
        subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                        u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx])
        module.addCode(subIterCode) # add scheduled "other", local reads, local writes
        for item in list(pack[luIdx].items()):
          if item.tempVgpr != None:
            self.vgprPool.checkIn(item.tempVgpr)
            item.tempVgpr = None
        pack[luIdx] = Code.Module()
      else:
        printExit("TensileLite does not support MAC instructions.")

    # close unrolled loop
    if expand:
      if not finalLoop:
        module.addComment2("Unrolled Loop - End %u/%u"%(lc+1, loopCopies))
      else:
        module.addComment2("Unrolled Loop - End %u/%u (final)"%(lc+1, loopCopies))

    else:
      module.addComment2("Unrolled Loop - End")

    oddLabel = lc == 0
    module.addCode(self.closeLoop(kernel, self.unrollIdx, finalLoop, oddLabel=oddLabel))
    return module

  ##############################################################################
  # Kernel Body
  ##############################################################################
  def kernelBody( self, kernel, tensorParametersA, tensorParametersB ):
    expand = kernel["ExpandPointerSwap"]

    ####################################
    # Begin String
    moduleKernelBody = Code.Module("kernelBody")

    ####################################
    # Function Signature
    ####################################
    moduleKernelBody.addComment2("Begin Kernel")

    module = Code.Module("body")
    module.addComment2("Allocate Resources")
    module.addCode(self.allocateResources(kernel))

    ####################################
    # Local Read Addresses
    ####################################
    module.addComment2("Local Read Addresses")

    # tile assignments
    module.addComment1("local read addresses: tile assignments a/b")
    module.addCode(self.lraTileAssignment(kernel, tensorParametersA, tensorParametersB))

    # final offsets
    module.addComment1("local read addresses: final offsets a")
    module.addCode(self.lraFinalOffset(kernel, tensorParametersA))
    module.addComment1("local read addresses: final offsets b")
    module.addCode(self.lraFinalOffset(kernel, tensorParametersB))

    # declare addresses
    module.addComment1("local read addresses: declare addresses a")
    module.addCode(self.lraDeclareAddresses(kernel, tensorParametersA))
    module.addComment1("local read addresses: declare addresses b")
    module.addCode(self.lraDeclareAddresses(kernel, tensorParametersB))

    # doShadowInit performs initialization in the 'shadow' of the global mem prefetch
    self.doShadowInit = 0
    if kernel["PrefetchGlobalRead"]:
      if self.actualSummationLoops == 1:
        self.doShadowInit = 2 # 2 is both store setup and initC
      else:
        # can't do shadow initC with multiple summation since this resets the ValuC counters
        # on each unroll iteration.
        self.doShadowInit = 1 # 1 is just store setup

    module.addCode(self.setupNewTile(kernel, tensorParametersA, tensorParametersB, isOptNLL=False))

    pack = [ Code.Module() for i in range (self.numVgprBuffer+1) ]
    self.preLoopLocalWriteCode = None

    if kernel["PrefetchGlobalRead"]:
      if self.doShadowInit:
        module.addCode(self.openShadowInit(kernel))
        module.addCode(self.globalWriteWorkGroupInit(kernel))
        if self.doShadowInit == 2:
          module.addCode(self.initC(kernel)) # initC while waiting for global reads
        module.addCode(self.closeShadowInit(kernel))

      module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "8wait for global read"))
      # These cases loop back and run the prefetch loop again
      # we need an extra barrier to ensure that the ds_reads (either for SR or MFMA) from previous iteration
      # have finished before we generate the prefetch for the next summation index.
      if self.actualSummationLoops>1:
        module.addInst(self.syncStr, "") #FIXME: Why an inst here?

      # local write
      self.preLoopLocalWriteCode = self.preLoopLocalWriteDo(kernel, tensorParametersA, tensorParametersB)
      module.addCode(self.preLoopLocalWriteCode)
      # swap local ptrs
      module.addComment1("local write swap a")
      module.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
      module.addComment1("local write swap b")
      module.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

      if kernel["PrefetchGlobalRead"] == 2:
        module.addCode(self.openPrefetchGlobalRead2(kernel))
        # if DirectToVgprA is enabled, swap the order of global read (B->A)
        tensorParameters1st = tensorParametersA
        tensorParameters2nd = tensorParametersB
        if kernel["DirectToVgprA"]:
          tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
        module.addCode(self.directToLdsM0Update(kernel, 1, tensorParameters1st))
        module.addCode(self.globalReadDo(kernel, 0, tensorParameters1st, 1))
        module.addCode(self.directToLdsM0Update(kernel, 1, tensorParameters2nd))
        module.addCode(self.globalReadDo(kernel, 0, tensorParameters2nd, 1))

        # swap local ptrs again if DirectToLds is enabled
        if kernel["DirectToLdsA"]:
          module.addComment1("local write swap a")
          module.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
        if kernel["DirectToLdsB"]:
          module.addComment1("local write swap b")
          module.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

        module.addCode(self.closePrefetchGlobalRead2(kernel))

      # prefetch-local
      if self.numItersPLR:
        # not generate wait for local write if LDS write code is not generated
        if not kernel["NoLdsWriteCode"]:
          # TODO: need to check if we correctly checked-in the temp VGPR used for Int8 LocalWrite (uDu, PGR=2)
          module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "0prefetch wait for local write"))
        module.addCode(self.syncThreads(kernel))

        # in some cases need an extra copy of the LDS read with appropriate double buffer offsets
        for plrIdx in range(0, self.numItersPLR):
          pack[plrIdx] = Code.Module()
          for espi in range(0, 1):
            for iui in range(0,kernel["InnerUnroll"]):
              if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"] and (not kernel["DirectToVgprA"]) : # no local read code if DirectToVgpr is enabled
                module.addComment1("local read prefetch a")
                localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, espi, tensorParametersA)
                module.addCode(localReadCodeA)
                pack[plrIdx].addCode(packCodeA)
              if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"] and (not kernel["DirectToVgprB"]) : # no local read code if DirectToVgpr is enabled
                module.addComment1("local read prefetch b")
                localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, espi, tensorParametersB)
                module.addCode(localReadCodeB)
                pack[plrIdx].addCode(packCodeB)
              if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"] and (not kernel["DirectToVgprA"]) : # no local read code if DirectToVgpr is enabled
                module.addComment1("local read inc a")
                module.addCode(self.localReadInc(kernel, iui, tensorParametersA))
              if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"] and (not kernel["DirectToVgprB"]) : # no local read code if DirectToVgpr is enabled
                module.addComment1("local read inc b")
                module.addCode(self.localReadInc(kernel, iui, tensorParametersB))
      module.addCode(self.closeSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=False, isNGLL=False))

    loopCopies = 2 if expand else 1

    if self.useInitAccVgprOpt:
      # generate first iteration code for init accvgpr opt
      module.addComment2("First Unrolled Iter for InitAccVgprOpt - Begin")
      # open loop without Label
      module.addCode(self.openLoop(kernel, self.unrollIdx, noLabelGen=True))
      module.addCode(self.loopBody( kernel, tensorParametersA, tensorParametersB, pack, 0, loopCopies, False, firstIter=True ))

    # open unrolled summation loop
    module.addComment2("Unrolled Loop(s) - Begin")
    module.addCode(self.openLoop(kernel, self.unrollIdx, beginLabelOnly=False))

    lcStart = 0
    if self.useInitAccVgprOpt:
      lcStart = 1 if loopCopies == 2 else 0
    for lc in range(0, loopCopies):
      loopIndex = lcStart + lc
      if loopIndex >= loopCopies:
        loopIndex -= loopCopies
      # loop body code generation
      finalLoop = lc == loopCopies - 1
      module.addCode(self.loopBody( kernel, tensorParametersA, tensorParametersB, pack, loopIndex, loopCopies, finalLoop ))

    module.addComment1("Before NLL: Check VGPR.checkin for INT8 LW")

    # swap local write, read again before noLoadLoop if PrefetchGlobalRead and DirectToLds is enabled
    # In DirectToLds enabled case, local write address is necessary for prefetch global read (for m0).
    # However, even exit with DirectToLds will not pass with this code (limitation).
    # So far, this code is to make odd exit case (i.e. k is multiple of 2*depthU) pass for DirectToVgpr
    if not self.useInitAccVgprOpt and kernel["PrefetchGlobalRead"] and kernel["ExpandPointerSwap"]:
      # local write for next iter, used to have local writes here
      if(kernel["DirectToLdsA"]):
        module.addComment1("local write swap offsets a")
        module.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
      if(kernel["DirectToLdsB"]):
        module.addComment1("local write swap offsets b")
        module.addCode(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
    # swap local read point for self.useInitAccVgprOpt
    if self.useInitAccVgprOpt and kernel["ExpandPointerSwap"]:
      module.addComment1("local read swap offsets a")
      module.addCode(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
      module.addComment1("local read swap offsets b")
      module.addCode(self.localReadSwapOffsets(kernel, expand, tensorParametersB))

    if kernel["PrefetchGlobalRead"] == 2:
      module.addCode(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=True, pack=pack))

    # This "NoLoad" loop is a copy of the unroll loop but with global loads + LDS writes removed
    # doShadowInit is required since this pushes up the store SRD initialization before the NLL
    # OptNLL only allowed for single summation index  - for multiple summation we (currently)
    # execute the NLL inside each unroll iteration not just once at the end.
    if kernel["PrefetchGlobalRead"]:
      if not kernel["SuppressNoLoadLoop"]:

        firstNLLgenerated = False
        if kernel["KernelLanguage"] == "Assembly" and kernel["OptNoLoadLoop"] and \
           kernel["BufferLoad"] and kernel["BufferStore"] and self.doShadowInit and \
           kernel["LocalSplitU"]==1 and kernel["GlobalSplitU"] == 1 and \
           self.actualSummationLoops==1:

          firstNLLgenerated = True

          # two different noLoadLoops:
          # 1. OptNLL & PAP global-read interleaved (only for PAP=ON)
          # (2. OptNLL : No PAP global-read (For PAP=OFF, or PAP=ON but the last tile))
          #  -> this is unified with 1. global-read is invalidated at the last tile.
          # 3. OrdinaryNLL (Not Opt.)
          self.saveLocalPointers(kernel)
          # deepCopy packCode for OptNLL noLoadLoop
          deepCopyPack = copy.deepcopy(pack)
          module.addCode(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=True, isNGLL=False, pack=deepCopyPack))
          self.restoreLocalPointers(kernel)

        # skip second NLL code if enableSingleNLLOpt
        if not (self.enableSingleNLLOpt and firstNLLgenerated):
          module.addCode(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=False, pack=pack))
        else:
          # generate PrefetchGlobalLastIterEnd label
          module.addCode(self.closeSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=False, isNGLL=False))

      # if PGR, last few iterations will have PLR,
      # and those PLR will not be used(register not checkIn) if without NoLoadLoop
      else:
        for i in range(self.numVgprBuffer):
          for item in list(pack[i].items()):
            if item.tempVgpr != None:
              self.vgprPool.checkIn(item.tempVgpr)
              item.tempVgpr = None

    if self.staggerU and self.actualSummationLoops>1:
      module.addComment1("remove stagger offsets")
      module.addCode(self.removeStagger(kernel, tensorParametersA))
      module.addCode(self.removeStagger(kernel, tensorParametersB))

    if not self.noTailLoop:
      ########################################
      # Tail Loop
      # which means tail loop not needed.
      ########################################
      self.inTailLoop = True
      module.addComment2("Tail Loop")

      # Update local write pointers in case the upcoming global reads are writing directly to LDS:
      if kernel["PrefetchGlobalRead"]:
        module.addComment1("local write reset offsets a")
        module.addCode(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersA))
        if kernel["ExpandPointerSwap"]:
          # reset local write offset in asm code as well
          module.addCode(self.localWriteResetOffsets(kernel, False, tensorParametersA))
        module.addComment1("local write reset offsets b")
        module.addCode(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersB))
        if kernel["ExpandPointerSwap"]:
          # reset local write offset in asm code as well
          module.addCode(self.localWriteResetOffsets(kernel, False, tensorParametersB))

      # tail: global read
      module.addCode(self.calculateLoopNumIter(kernel, -1))
      if self.staggerU and self.actualSummationLoops==1:
        module.addComment1("remove stagger offsets for tail loop")
        module.addCode(self.removeStagger(kernel, tensorParametersA))
        module.addCode(self.removeStagger(kernel, tensorParametersB))

      # if DirectToVgprA is enabled, swap the order of global read (B->A)
      tensorParameters1st = tensorParametersA
      tensorParameters2nd = tensorParametersB
      tc1 = 'a'
      tc2 = 'b'
      if kernel["DirectToVgprA"]:
        tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
        tc1, tc2 = tc2, tc1
      module.addComment1("Update M0 for DTLDS")
      moduleTmp = self.directToLdsM0Update(kernel, 1, tensorParameters1st)
      module.addCode(replacePlaceHolder(moduleTmp, "__placeholder__", 0))
      module.addComment1("global read %s"%tc1)
      vregSetIdx = 0
      module.addCode(self.globalReadDo(kernel, 2, tensorParameters1st, vregSetIdx))
      module.addComment1("Update M0 for DTLDS")
      moduleTmp = self.directToLdsM0Update(kernel, 1, tensorParameters2nd)
      module.addCode(replacePlaceHolder(moduleTmp, "__placeholder__", 0))
      module.addComment1("global read %s"%tc2)
      vregSetIdx = 0
      module.addCode(self.globalReadDo(kernel, 2, tensorParameters2nd, vregSetIdx))
      module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "2wait for global read"))
      module.addCode(self.syncThreads(kernel))

      # the following read/write addresses could be modified in recalcLocal(Read|Write)Addresses due to policy change
      self.oriLraA = None # back up original local read address vgpr
      self.oriLraB = None
      self.oriLwaA = None # back up original local write address vgpr
      self.oriLwaB = None
      for uDu in range(0, kernel["DepthULdsDivisor"]):
        if kernel.enabledSplitLDS:
          # change local write policy from interleave-K to fractional as tail loop
          # iterate LDS read address one unit of K at a time
          module.addComment1("Recalc local write offsets")
          module.addCode(self.recalcLocalWriteAddresses(kernel, tensorParametersA, uDu))
          module.addCode(self.recalcLocalWriteAddresses(kernel, tensorParametersB, uDu))
        if uDu > 0:
          module.addComment1("sync before local write")
          module.addCode(self.syncThreads(kernel))
        if not kernel["NoLdsWriteCode"]:
          # tail: local write
          module.addComment1("local write a")
          tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA, None)
          module.addCode(tempLWCodeModA)
          module.addComment1("local write b")
          tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB, None)
          module.addCode(tempLWCodeModB)
        # change local read policy from wider local read to one unit of K at a time
        module.addComment1("Recalc local read offsets")
        module.addCode(self.recalcLocalReadAddressesAB(kernel))
        # TODO: need to check if we correctly checked-in the temp VGPR used for Int8 LocalWrite (uDu, PGR=2)
        module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "5wait for local write"))
        module.addCode(self.syncThreads(kernel))
        #module.addCode(self.dumpLds(kernel, 0, 8))

        # tail: re-init local read addresses
        if kernel["PrefetchGlobalRead"]:
          module.addComment1("local read reset offsets a")
          module.addCode(self.localReadResetOffsets(kernel, tensorParametersA))
          module.addComment1("local read reset offsets b")
          module.addCode(self.localReadResetOffsets(kernel, tensorParametersB))
          module.addComment1("local read init pointers a")
          module.addCode(self.localReadInitPointers(kernel, tensorParametersA))
          module.addComment1("local read init pointers b")
          module.addCode(self.localReadInitPointers(kernel, tensorParametersB))
        # tail: macs
        module.addComment1("tail loop: macs")
        module.addCode(self.openLoop(kernel, -1, uDu if kernel.enabledSplitLDS else None))

        # Try to use InnerUnroll in the tail loop if allowed:
        KinInnerUnroll = kernel["InnerUnroll"]
        if kernel["EnableMatrixInstruction"]:
          KinInnerUnroll *= kernel["MatrixInstK"]

        tailLoopInnerUnroll = 1
        if (kernel["AssertSummationElementMultiple"] % KinInnerUnroll == 0):
          tailLoopInnerUnroll = kernel["InnerUnroll"]
        # need to unroll tail loop for the following cases
        mEnd = 1
        if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
          mEnd = kernel["DepthU"]//KinInnerUnroll
        elif kernel["DirectToLds"] and kernel["EnableMatrixInstruction"] and kernel["InnerUnroll"] == 1 and\
             (kernel["GlobalLoadVectorWidthA"] * self.bpeAB > 4 or kernel["GlobalLoadVectorWidthB"] * self.bpeAB > 4) and \
             kernel["DepthU"] // kernel["MatrixInstK"] > 2:
          mEnd = kernel["DepthU"] // (kernel["MatrixInstK"] * 2)

        for mValue in range(mEnd):
          pack[0] = Code.Module()
          for iui in range(0, tailLoopInnerUnroll):
            doReadA = not kernel["DirectToVgprA"]
            doReadB = not kernel["DirectToVgprB"]
            if doReadA:
              # Reading 16-bit data from LDS requires packing when ECC enabled
              module.addComment1("local read a")
              localReadCodeA, packCodeA = self.localReadDo(kernel, 0, iui, 0, tensorParametersA)
              module.addCode(localReadCodeA)
              pack[0].addCode(packCodeA)
            if doReadB:
              module.addComment1("local read b")
              localReadCodeB, packCodeB = self.localReadDo(kernel, 0, iui, 0, tensorParametersB)
              module.addCode(localReadCodeB)
              pack[0].addCode(packCodeB)
            # adjustment for DirectToLds case
            iuiParam = iui + tailLoopInnerUnroll * mValue
            if doReadA:
              module.addComment1("local read inc a")
              module.addCode(self.localReadInc(kernel, iuiParam, tensorParametersA))
            if doReadB:
              module.addComment1("local read inc b")
              module.addCode(self.localReadInc(kernel, iuiParam, tensorParametersB))
          module.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "4wait for local read"))

          if kernel["EnableMatrixInstruction"]:
            module.addCode(pack[0])
            # vgpr.checkin for all the checked-out vgpr in LocalRead
            for item in list(pack[0].items()):
              if item.tempVgpr != None:
                self.vgprPool.checkIn(item.tempVgpr)
                item.tempVgpr = None
            pack[0] = Code.Module()

          if kernel["EnableMatrixInstruction"]:
            # DirectToVgpr is not applicable for tail loop
            vregSetIdxMFMA = 0
            module.addCode(self.mfmaIter(kernel, 0, tailLoopInnerUnroll, vregSetIdxMFMA, False, True))
          else:
            printExit("TensileLite does not support MAC instructions.")

          finalLoop = mValue == mEnd - 1
          module.addCode(self.closeLoop(kernel, -1, finalLoop, uDu if kernel.enabledSplitLDS else None))
      # always emit the skip-tail-loop label
      module.addCode(self.closeLoop(kernel, -1, None, emitEndLabelOnly=True))
      # tail: close
      self.inTailLoop = False

    # extra summation loops: global increment and close
    for i in reversed(range(self.otherSummationLoops)):
      module.addComment1("global read inc AB")
      module.addCode(self.globalReadIncrementAB(kernel, i, 0))
      module.addCode(self.closeLoop(kernel, i, True))

    module.addCode(self.endSummation(kernel))
    if not self.doShadowInit:
      module.addCode(self.globalWriteWorkGroupInit(kernel))

    ####################################
    # Shift Vector Components
    ####################################
    if kernel["EdgeType"] == "ShiftPtr":
      # GuaranteeNoPartial means each component in the vector loads is always valid.  In this case we
      # don't need the unshift code

      # shift vector components d0
      if not kernel["GuaranteeNoPartialA"] and self.readTileDimVectorA:
        module.addComment1("shift vector components d0")
        module.addCode(self.shiftVectorComponents(kernel, tensorParametersA))

      # shift vector components d1, for MFMA version, B never entered this
      if not kernel["GuaranteeNoPartialB"] and self.readTileDimVectorB:
        module.addComment1("shift vector components d1")
        module.addCode(self.shiftVectorComponents(kernel, tensorParametersB))

    # complex declare tmp registers
    module.addCode(self.complexDeclareTmpRegisters(kernel))

    ####################################
    # LocalSplitU reduction
    ####################################
    #if kernel["NumThreads"]%kernel["MacroTile0"] == 0:
    if kernel["LocalSplitU"] > 1:
      module.addComment2("LocalSplitU Reduction")
      module.addCode(self.syncThreads(kernel))

      # LocalSplitU: local write
      module.addComment1("LocalSplitU: local write")
      module.addCode(self.localSplitULocalWrite(kernel))

      # LocalSplitU: local read
      module.addComment1("LocalSplitU: local read")
      module.addCode(self.localSplitULocalRead(kernel))

      # LocalSplitU: local read
      module.addComment1("LocalSplitU: reduction")
      module.addCode(self.localSplitUReduction(kernel))

      # LocalSplitU: global write indices
      module.addComment1("LocalSplitU: global write indices")
      module.addCode(self.localSplitUGlobalWriteIndices(kernel))

      # LocalSplitU: global write
      module.addComment1("LocalSplitU: global write")
      module.addCode(self.localSplitUGlobalWrite(kernel))


    else:
      ####################################
      # NOT LocalSplitU
      ####################################

      # global write indices
      module.addComment1("not-LocalSplitU: global write indices")
      module.addCode(self.notLocalSplitUGlobalWriteIndices(kernel))

      # global write
      module.addComment1("not-LocalSplitU: global write")
      module.addCode(self.notLocalSplitUGlobalWrite(kernel))

    # function suffix
    module.addCode(self.functionEnd(kernel, True))
    module.addCode(self.functionSuffix(kernel))

    error = self.overflowedResources
    # function signature last since it needs to know how many gprs were actually used
    fs = self.functionSignature(kernel)
    assignDict = CodePass.GetAssignmentDict(fs)
    graph = CodePass.BuildGraph(module, self.vgprPool.size(), self.sgprPool.size(), assignDict)
    CodePass.RemoveDuplicateAssignment(graph)
    moduleKernelBody.addCode(fs)
    moduleKernelBody.addCode(module)
    return (error, str(moduleKernelBody))

  ##############################################################################
  # Init Kernel
  ##############################################################################
  @abc.abstractmethod
  def initKernel(self, kernel, tensorParametersA, tensorParametersB ):

    self.staggerU = kernel["StaggerU"] and (kernel["KernelLanguage"]=="Source" or kernel["BufferLoad"])
    self.tPA = tensorParametersA
    self.tPB = tensorParametersB

    # Only assembly supports scheduling
    self.canSchedule = (kernel["KernelLanguage"] == "Assembly")

    if self.canSchedule:
      self.scheduleGlobalRead = kernel["ScheduleGlobalRead"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"] # flat updates lgkmcnt counts = hard to schedule flat loads
    else:
      self.scheduleGlobalRead = 0

    if self.canSchedule:
      self.scheduleLocalWrite = kernel["ScheduleLocalWrite"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"]  # flat updates lgkmcnt counts = hard to schedule writes and loads?
    else:
      self.scheduleLocalWrite = 0

    if self.canSchedule:
      self.scheduleIterAlg = kernel["ScheduleIterAlg"]
    else:
      self.scheduleIterAlg = 0

    self.noTailLoop = kernel["NoTailLoop"]

    self.actualSummationLoops = kernel["ProblemType"]["NumIndicesSummation"]
    self.otherSummationLoops  = self.actualSummationLoops-1
    self.otherSummations      = kernel["ProblemType"]["NumIndicesSummation"]-1 # not loops but summations vars

    if kernel["KernelLanguage"] == "Source":
      self.language = globalParameters["RuntimeLanguage"]
    else:
      self.language = "ASM"
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.indexChars[kernel["ProblemType"]["Index0"]]
    self.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.indexChars[kernel["ProblemType"]["Index1"]]
    self.unrollIdx = kernel["ProblemType"]["NumIndicesSummation"]-1
    self.unrollChar = \
        self.indexChars[kernel["ProblemType"]["IndicesSummation"][\
        self.unrollIdx]]
    self.tileChar0 = self.indexChars[kernel["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[kernel["ProblemType"]["Index1"]]
    self.tileCharA = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==0) \
        else self.tileChar1
    self.tileCharB = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==1) \
        else self.tileChar1

    """
    if kernel["ProblemType"]["Tensor0"]==0:
      kernel["ThreadTileA"] = kernel["ThreadTile0"]
      kernel["ThreadTileB"] = kernel["ThreadTile1"]
      kernel["SubGroupA"] = kernel["SubGroup0"]
      kernel["SubGroupB"] = kernel["SubGroup1"]
      kernel["MacroTileA"] = kernel["MacroTile0"]
      kernel["MacroTileB"] = kernel["MacroTile1"]
    else:
      kernel["ThreadTileB"] = kernel["ThreadTile0"]
      kernel["ThreadTileA"] = kernel["ThreadTile1"]
      kernel["SubGroupB"] = kernel["SubGroup0"]
      kernel["SubGroupA"] = kernel["SubGroup1"]
      kernel["MacroTileB"] = kernel["MacroTile0"]
      kernel["MacroTileA"] = kernel["MacroTile1"]
    """

    """
    # original parameters
    NumLoadsCoalesced -> NumLoadsPerpendicular
    # new intermediate parameters
    numReadsTile # nrt
    numReadsUnroll # nru
    numReadsTileVecComp # nrvt
    numReadsUnrollVecComp # nrvu
    numWritesCoal # nwc
    numWritesPerp # nwp
    numWritesCoalVecComp # nwvc
    numWritesPerpVecComp # nwvp
    readTileComponents (based on grcv)
    readTileVector
    """

    # TODO load sub-vector
    vwa = kernel["GlobalLoadVectorWidthA"]
    vwb = kernel["GlobalLoadVectorWidthB"]

    # allow LocalReadVectorWidth for TLU + MatrixInstruction
    self.allowLRVWforTLUandMI = kernel["allowLRVWforTLUandMI"]

    self.numItersPLR = kernel["PrefetchLocalRead"]%kernel["LoopIters"]
    self.numVgprBuffer = kernel["LoopIters"] if kernel["PrefetchLocalRead"] > kernel["LoopIters"] else kernel["PrefetchLocalRead"]
    # merge N iteration's read into 1 iteration if can't coalesce read
    # ex, A can coalesce read, B can't
    # MergeRead 0: ds_readAx1 ds_readBx1 mfma | ds_readAx1 ds_readBx1 mfma | => ds_readAx2 ds_readBx1 mfma | ds_readBx1 mfma |
    # MergeRead 1: ds_readAx1 ds_readBx1 mfma | ds_readAx1 ds_readAx1 mfma | => ds_readAx2 ds_readBx1 ds_readBx1 mfma | mfma |
    MergeRead = 0
    if not kernel["ProblemType"]["TLUA"] or MergeRead or self.allowLRVWforTLUandMI:
      if (not kernel["ProblemType"]["TLUA"]) and kernel["DirectToVgprA"]:
        # DirectToVgpr + TLU=False case, ignore LocalReadVectorWidth and use GlobalLoadVectorWidth instead.
        self.lrvwA = vwa
      else:
        self.lrvwA = kernel["LocalReadVectorWidth"]
    else:
      if kernel["EnableMatrixInstruction"]:
        self.lrvwA = kernel["MIInputPerThread"]
      else:
        self.lrvwA = 1
    if not kernel["ProblemType"]["TLUB"] or MergeRead or self.allowLRVWforTLUandMI:
      if (not kernel["ProblemType"]["TLUB"]) and kernel["DirectToVgprB"]:
        # DirectToVgpr + TLU=False case, ignore LocalReadVectorWidth and use GlobalLoadVectorWidth instead.
        self.lrvwB = vwb
      else:
        self.lrvwB = kernel["LocalReadVectorWidth"]
    else:
      if kernel["EnableMatrixInstruction"]:
        self.lrvwB = kernel["MIInputPerThread"]
      else:
        self.lrvwB = 1

    # DirectToVgprB + VW > 1 case, set lrvwB = VW
    # DirectToVgprB case, global load data directly goes to Vgpr.
    # If VW=2, it means lrwvB is 2.
    if kernel["DirectToVgprB"] and kernel["VectorWidth"] > 1:
      self.lrvwB = kernel["VectorWidth"]
    # DirectToVgpr + TLU=False case
    # set lrvw = VW
    self.vgprValuDouble = False
    #if kernel["DirectToVgprA"] and kernel["PrefetchLocalRead"] > 1 and (not kernel["ProblemType"]["TLUA"]) and kernel["VectorWidth"] > 1:
    if kernel["DirectToVgprA"] and (not kernel["ProblemType"]["TLUA"]) and (not kernel["ProblemType"]["TLUB"]) or \
       kernel["DirectToVgprB"] and (not kernel["ProblemType"]["TLUB"]) and (not kernel["ProblemType"]["TLUA"]):
      self.lrvwA = max(self.lrvwA, self.lrvwB)
      self.lrvwB = self.lrvwA
      if kernel["DepthU"] // kernel["MatrixInstK"] <= 2 and self.lrvwA > 1:
        # need to double vgprValu to avoid local read overwritting vgprValu registers
        self.vgprValuDouble = True

    # Wider LocalRead
    if kernel["EnableMatrixInstruction"]:
      self.numReadsIterCoalescedA = self.lrvwA // kernel["MIInputPerThread"]
      self.numReadsIterCoalescedB = self.lrvwB // kernel["MIInputPerThread"]
      if self.allowLRVWforTLUandMI:
        self.numReadsIterCoalescedA = 1
        self.numReadsIterCoalescedB = 1
    else:
      self.numReadsIterCoalescedA  = 1
      self.numReadsIterCoalescedB  = 1
    self.numIterPerCoalescedReadA = max(1,self.numReadsIterCoalescedA//kernel["InnerUnroll"])
    self.numIterPerCoalescedReadB = max(1,self.numReadsIterCoalescedB//kernel["InnerUnroll"])

    if kernel["ScheduleIterAlg"] == 3 or kernel["ScheduleIterAlg"] == 2:
      self.numMfmaPerIter = kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1] * kernel["InnerUnroll"]
      if kernel["ProblemType"]["DataType"].isComplex(): self.numMfmaPerIter *= 4

    ########################################
    # read vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"]: # NT no transpose
      self.numReadsTileA = kernel["NumLoadsCoalescedA"]
      self.numReadsUnrollA = kernel["NumLoadsPerpendicularA"]
      self.readTileDimComponentsA = False # Vector
      self.readTileDimVectorA = True # Vector
      self.readUnrollDimComponentsA = False # Scalar
      self.readUnrollDimVectorA = False # Scalar
      self.numReadsTileVecCompA = vwa
      self.numReadsUnrollVecCompA = 1
    else: # TN yes transpose
      self.numReadsTileA = kernel["NumLoadsPerpendicularA"]
      self.numReadsUnrollA = kernel["NumLoadsCoalescedA"]
      self.readTileDimComponentsA = False # Scalar
      self.readTileDimVectorA = False # Scalar
      self.readUnrollDimComponentsA = False # Vector
      self.readUnrollDimVectorA = True # Vector
      self.numReadsUnrollVecCompA = vwa
      self.numReadsTileVecCompA = 1

    ########################################
    # write vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"] != kernel["UnrollMajorLDSA"]: # NT no transpose
      self.numWritesCoalA = kernel["NumLoadsCoalescedA"]
      self.writeUnrollDimComponentsA = False # Scalar
      self.writeTileDimComponentsA = False # Vector
      writeCoal = True
    else: # TN yes transpose
      self.numWritesCoalA = kernel["NumLoadsPerpendicularA"]
      self.writeUnrollDimComponentsA = False # Scalar
      self.writeTileDimComponentsA = kernel["GlobalReadVectorWidth"] > 1 # Components
      writeCoal = False

    # writeCoal indicates writes should be done in the coal dim
    # else in perp
    if writeCoal:
      self.numWritesCoalVecCompA = vwa // kernel["DepthULdsDivisor"]
      self.numWritesPerpVecCompA = 1
    else:
      self.numWritesCoalVecCompA = 1
      self.numWritesPerpVecCompA = vwa
    del writeCoal

    self.numReadVectorComponentsA = kernel["GlobalLoadVectorWidthA"] \
        if (self.readTileDimComponentsA \
        or self.readUnrollDimComponentsA) else 1
    # self.numWriteVectorComponentsA = kernel["GlobalLoadVectorWidthA"] \
    #     if (self.writeTileDimComponentsA \
    #     or self.writeUnrollDimComponentsA) else 1
    # self.numReadTileVectorComponentsA = kernel["GlobalLoadVectorWidthA"] \
    #     if self.readTileDimComponentsA else 1 # for branches
    # convert tile/unroll to para/perp
    if kernel["ProblemType"]["TLUA"]:
      self.numReadsCoalVecCompA = self.numReadsTileVecCompA
      self.numReadsPerpVecCompA = self.numReadsUnrollVecCompA
      # for asm
      self.readCoalescedComponentsA  = self.readTileDimComponentsA
      # self.readCoalescedVectorA      = self.readTileDimVectorA  # Not Used
      self.readPerpendicularComponentsA  = self.readUnrollDimComponentsA
      # self.readPerpendicularVectorA      = self.readUnrollDimVectorA  # Not Used
    else:
      self.numReadsCoalVecCompA = self.numReadsUnrollVecCompA
      self.numReadsPerpVecCompA = self.numReadsTileVecCompA
      # for asm
      self.readCoalescedComponentsA  = self.readUnrollDimComponentsA
      # self.readCoalescedVectorA      = self.readUnrollDimVectorA  # Not Used
      self.readPerpendicularComponentsA  = self.readTileDimComponentsA
      # self.readPerpendicularVectorA      = self.readTileDimVectorA  # Not Used

    ####################################
    # read vectors or vector components b
    ####################################
    if kernel["ProblemType"]["TLUB"]: # NT no transpose
      self.numReadsTileB = kernel["NumLoadsCoalescedB"]
      self.numReadsUnrollB = kernel["NumLoadsPerpendicularB"]
      self.readTileDimComponentsB = False # Vector
      self.readTileDimVectorB = True # Vector
      self.readUnrollDimComponentsB = False # Scalar
      self.readUnrollDimVectorB = False # Scalar
      self.numReadsTileVecCompB = vwb
      self.numReadsUnrollVecCompB = 1
    else: # TN yes transpose
      self.numReadsTileB = kernel["NumLoadsPerpendicularB"]
      self.numReadsUnrollB = kernel["NumLoadsCoalescedB"]
      self.readTileDimComponentsB = False # Scalar
      self.readTileDimVectorB = False # Scalar
      self.readUnrollDimComponentsB = False # Vector
      self.readUnrollDimVectorB = True # Vector
      self.numReadsUnrollVecCompB = vwb
      self.numReadsTileVecCompB = 1

    ####################################
    # write vectors or vector components b
    ####################################
    if kernel["ProblemType"]["TLUB"] != kernel["UnrollMajorLDSB"]: # NT no transpose
      self.numWritesCoalB = kernel["NumLoadsCoalescedB"]
      self.writeUnrollDimComponentsB = False # Vector
      self.writeTileDimComponentsB = False # Vector
      writeCoal = True
    else: # TN yes transpose
      self.numWritesCoalB = kernel["NumLoadsPerpendicularB"]
      self.writeUnrollDimComponentsB = False
      self.writeTileDimComponentsB = kernel["GlobalReadVectorWidth"] > 1 # Components
      writeCoal = False

    # writeCoal indicates writes should be done in the coal dim
    # else in perp
    if writeCoal:
      self.numWritesCoalVecCompB = vwb // kernel["DepthULdsDivisor"]
      self.numWritesPerpVecCompB = 1
    else:
      self.numWritesCoalVecCompB = 1
      self.numWritesPerpVecCompB = vwb
    del writeCoal

    # numReadVectorComponentsB is refers to global reads
    self.numReadVectorComponentsB = kernel["GlobalLoadVectorWidthB"] \
        if (self.readTileDimComponentsB \
        or self.readUnrollDimComponentsB) else 1
    # self.numWriteVectorComponentsB = kernel["GlobalLoadVectorWidthB"] \
    #     if (self.writeTileDimComponentsB \
    #     or self.writeUnrollDimComponentsB) else 1
    # self.numReadTileVectorComponentsB = kernel["GlobalLoadVectorWidthB"] \
    #     if self.readTileDimComponentsB else 1 # for branches
    # convert tile/unroll to para/perp
    if kernel["ProblemType"]["TLUB"]:
      self.numReadsCoalVecCompB = self.numReadsTileVecCompB
      self.numReadsPerpVecCompB = self.numReadsUnrollVecCompB
      # for asm
      self.readCoalescedComponentsB  = self.readTileDimComponentsB
      # self.readCoalescedVectorB      = self.readTileDimVectorB  # Not Used
      self.readPerpendicularComponentsB  = self.readUnrollDimComponentsB
      # self.readPerpendicularVectorB      = self.readUnrollDimVectorB  # Not Used
    else:
      self.numReadsCoalVecCompB = self.numReadsUnrollVecCompB
      self.numReadsPerpVecCompB = self.numReadsTileVecCompB
      # for asm
      self.readCoalescedComponentsB  = self.readUnrollDimComponentsB
      # self.readCoalescedVectorB      = self.readUnrollDimVectorB  # Not Used
      self.readPerpendicularComponentsB  = self.readTileDimComponentsB
      # self.readPerpendicularVectorB      = self.readTileDimVectorB  # Not Used

    ####################################
    # load sizes
    """
    if kernel["ProblemType"]["TLUA"]:
      kernel["LSCA"] = kernel["MacroTileA"] \
          / kernel["NumLoadsCoalescedA"]
      kernel["LSPA"] = kernel["DepthU"] / kernel["NumLoadsPerpendicularA"]
    else:
      kernel["LSCA"] = kernel["DepthU"] / kernel["NumLoadsCoalescedA"]
      kernel["LSPA"] = kernel["MacroTileA"] \
          / kernel["NumLoadsPerpendicularA"]
    if kernel["ProblemType"]["TLUB"]:
      kernel["LSCB"] = kernel["MacroTileB"] \
          / kernel["NumLoadsCoalescedB"]
      kernel["LSPB"] = kernel["DepthU"] / kernel["NumLoadsPerpendicularB"]
    else:
      kernel["LSCB"] = kernel["DepthU"] / kernel["NumLoadsCoalescedB"]
      kernel["LSPB"] = kernel["MacroTileB"] \
          / kernel["NumLoadsPerpendicularB"]
    kernel["LVCA"] = kernel["LSCA"] / kernel["GlobalLoadVectorWidthA"]
    kernel["LVCB"] = kernel["LSCB"] / kernel["GlobalLoadVectorWidthB"]
    kernel["LVPA"] = kernel["LSPA"] / kernel["GlobalLoadVectorWidthA"]
    kernel["LVPB"] = kernel["LSPB"] / kernel["GlobalLoadVectorWidthB"]
    """

    self.getTensorParameters(tensorParametersA, kernel, True)
    self.getTensorParameters(tensorParametersB, kernel, False)

    tensorParametersA["PackedIndices"] = kernel["PackedC%uIndicesX"%self.tPA["tile01Idx"]]
    tensorParametersB["PackedIndices"] = kernel["PackedC%uIndicesX"%self.tPB["tile01Idx"]]

    # condition(s) to enable init accvgpr opt (initialize only the last set of accvgpr instead of whole accvgpr)
    self.useInitAccVgprOpt = False
    # enable for the following conditions
    if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
      self.useInitAccVgprOpt = True
    # force to disable for the following conditions
    if self.useInitAccVgprOpt:
      if kernel["PrefetchGlobalRead"] == 2:
        # PGR=2 case, K > DepthU * 2 is necessary ( if not noTailLoop, need > DepthU * 3)
        # (kernel["AssertSizeGreaterThan"][3] > DepthU * 2 (or 3)
        minDUnum = 2 if self.noTailLoop else 3
        if not (3 in kernel["AssertSizeGreaterThan"].keys() and kernel["AssertSizeGreaterThan"][3] >= kernel["DepthU"] * minDUnum):
          print2("InitAccVgprOpt is disabled because AssertSizeGreaterThan for K is not greater than DepthU * %u"%minDUnum)
          self.useInitAccVgprOpt = False
      if kernel["PrefetchGlobalRead"] == 1:
        # PGR=1 case, K > DepthU * 1 is necessary ( if not noTailLoop, need > DepthU * 2)
        # (kernel["AssertSizeGreaterThan"][3] > DepthU * 2 (or 3)
        minDUnum = 1 if self.noTailLoop else 2
        if not (3 in kernel["AssertSizeGreaterThan"].keys() and kernel["AssertSizeGreaterThan"][3] >= kernel["DepthU"] * minDUnum):
          print2("InitAccVgprOpt is disabled because AssertSizeGreaterThan for K is not greater than DepthU * %u"%minDUnum)
          self.useInitAccVgprOpt = False

    # condition(s) to enable singleNLL opt
    self.enableSingleNLLOpt = False
    if self.noTailLoop:
      pass
      # so far, not enabled for DirectToVgpr
      # Performance is better with Tensile, but does not perform better with HPL
      #if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
      #  self.enableSingleNLLOpt = True

  ##############################################################################
  # Function Signature
  ##############################################################################
  @abc.abstractmethod
  def functionSignature(self, kernel ):
    return ""

  ##############################################################################
  # Allocate Resources
  ##############################################################################
  @abc.abstractmethod
  def allocateResources(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  @abc.abstractmethod
  def graWorkGroup(self, kernel):
    return ""

  ##############################################################################
  # Get Params For Tensor A/B
  ##############################################################################
  def getTensorParameters(self, tP, kernel, tA):
    tP["mirror"] = bool(kernel["ProblemType"]["MirrorDims%s" % ("A" if tA else "B")])
    if tA: # A
      tP["isA"] = True                                      # is this tensor A
      tP["isB"] = False                                     # is this tensor B
      tP["bpe"] = int(4*kernel["ProblemType"]["DataType"].numRegisters())
      tP["tensorChar"] = "A"                                # tensor character A/B
      tP["tensorIdx"] = 0                                   # tensor index A=0, B=1
      tP["tileChar"] = self.tileCharA                       # tile char I0 or J1
      tP["tileIdx"] = kernel["ProblemType"]["Index01A"]     # is the tile dimension of A the 0th or 1th index, i.e. Aki, tileIdx=0
      tP["tile01Idx"] = 1 if tP["tileIdx"] else 0
      tP["lsc"] = "LSCA"                                    # load size coalesced A, number of elements that get loaded along coalesced dimension with each load
      tP["lsp"] = "LSPA"                                    # load size perpendicular A, number of elements that get loaded along non-coalesced dimension with each load
      tP["lvc"] = "LVCA"                                    # "load size" in terms of number of short-vectors and not elements
      tP["lvp"] = "LVPA"                                    # "load size" in terms of number of short-vectors and not elements
      tP["rtv"] = self.readTileDimVectorA                   # bool in the tile dimension, reads will read vectors
      tP["rtc"] = self.readTileDimComponentsA               # bool in the tile dimension, reads will read vector components
      #tP["ruv"] = self.readUnrollDimVectorA
      #tP["nlvc"] = self.numReadVectorComponentsA
      #tP["nwvc"] = self.numWriteVectorComponentsA
      tP["wg"] = "WorkGroup%u" % (tP["tile01Idx"])# these are storing the actual strong to lookup the number from kernel dictionary
      tP["prevWg"] = "PrevWorkGroup0"                       # used for prefetch-across-persistent #NHWC TO-do
      tP["sg"] = "SubGroup%u" % (tP["tile01Idx"])
      tP["tt"] = "ThreadTile%u" % (tP["tile01Idx"])
      tP["mt"] = "MacroTile%u" % (tP["tile01Idx"])
      tP["tlu"] = kernel["ProblemType"]["TLUA"]             # thread stride is less than unroll stride, i.e., not transposing matrix
      tP["ia"] = kernel["ProblemType"]["IndexAssignmentsA"] # array of index assignments
      #tP["nlc"] = kernel["NumLoadsCoalescedA"]
      #tP["nlp"] = kernel["NumLoadsPerpendicularA"]
      #tP["nlcv"] = self.numReadsCoalVecCompA
      tP["nlpv"] = self.numReadsPerpVecCompA                # num vector components perpendicular to coalesced; =1 or VW
      # NEW
      tP["nrt"] = self.numReadsTileA                        # number of reads along tile dimension
      tP["nrtv"] = self.numReadsTileVecCompA                # number of vector components along tile dimension; =1 or VW
      tP["nru"] = self.numReadsUnrollA                      # number of reads along unroll dimension
      tP["nruv"] = self.numReadsUnrollVecCompA              # number of vector components along unroll dimension; =1 or VW
      tP["nrc"] = kernel["NumLoadsCoalescedA"]              # number of reads along coalesced dimension
      tP["nrcv"] = self.numReadsCoalVecCompA                # number of vector components along coalesced dimension
      tP["nrp"] = kernel["NumLoadsPerpendicularA"]          # number of reads along perpendicular dimension
      tP["nrpv"] = self.numReadsPerpVecCompA                # number of vector components along perpendicular dimension
      tP["nwcv"] = self.numWritesCoalVecCompA               # number of vector component writes along coalesced dimension
      tP["nwpv"] = self.numWritesPerpVecCompA               # number of vector component writes along perpendicular dimension
      tP["glvw"] = kernel["GlobalLoadVectorWidthA"]
      # asm
      tP["rcc"] = self.readCoalescedComponentsA             # read vector components along coalesced dimensions
      # tP["rcv"] = self.readCoalescedVectorA                 # read vector along coalesced dimension
      tP["rpc"] = self.readPerpendicularComponentsA         # read vector components along perpendicular dimension
      # tP["rpv"] = self.readPerpendicularVectorA             # read vector along perpendicular dimension
      tP["ruc"] = self.readUnrollDimComponentsA             # read vector components along unroll dimension
      tP["wtc"] = self.writeTileDimComponentsA              # write vector components along tile dimension
      tP["wuc"] = self.writeUnrollDimComponentsA            # write vector components along unroll dimension
      tP["idx"] = kernel["ProblemType"]["Index0"]           # index 0 is tile dimension belonging to A. Note 'idx' may not be in tP['ia'].
      tP["rc"] = kernel["ProblemType"]["IndexAssignmentsA"][0] \
          in [kernel["ProblemType"]["Index01A"], \
          kernel["ProblemType"]["IndexUnroll"]]             # can read coalesced
      tP["NonTemporal"] = kernel["NonTemporalA"]            # non-temporal read type
    else: # B
      tP["isA"] = False
      tP["isB"] = True
      tP["bpe"] = int(4*kernel["ProblemType"]["DataType"].numRegisters())
      tP["tensorChar"] = "B"
      tP["tensorIdx"] = 1
      tP["tileChar"] = self.tileCharB
      tP["tileIdx"] = kernel["ProblemType"]["Index01B"]
      tP["tile01Idx"] = 1 if tP["tileIdx"] else 0
      tP["lsc"] = "LSCB"
      tP["lsp"] = "LSPB"
      tP["lvc"] = "LVCB"
      tP["lvp"] = "LVPB"
      tP["rtv"] = self.readTileDimVectorB
      tP["rtc"] = self.readTileDimComponentsB
      #tP["ruv"] = self.readUnrollDimVectorB
      #tP["nlvc"] = self.numReadVectorComponentsB
      #tP["nwvc"] = self.numWriteVectorComponentsB
      tP["wg"] = "WorkGroup%u" % (tP["tile01Idx"])
      tP["prevWg"] = "PrevWorkGroup1"
      tP["sg"] = "SubGroup%u" % (tP["tile01Idx"])
      tP["tt"] = "ThreadTile%u" % (tP["tile01Idx"])
      tP["mt"] = "MacroTile%u" % (tP["tile01Idx"])
      tP["tlu"] = kernel["ProblemType"]["TLUB"]
      tP["ia"] = kernel["ProblemType"]["IndexAssignmentsB"]
      # NEW
      tP["nrt"] = self.numReadsTileB
      tP["nrtv"] = self.numReadsTileVecCompB
      tP["nru"] = self.numReadsUnrollB
      tP["nruv"] = self.numReadsUnrollVecCompB
      tP["nrc"] = kernel["NumLoadsCoalescedB"]
      tP["nrcv"] = self.numReadsCoalVecCompB
      tP["nrp"] = kernel["NumLoadsPerpendicularB"]
      tP["nrpv"] = self.numReadsPerpVecCompB
      tP["nwcv"] = self.numWritesCoalVecCompB
      tP["nwpv"] = self.numWritesPerpVecCompB
      tP["glvw"] = kernel["GlobalLoadVectorWidthB"]
      # asm
      tP["rcc"] = self.readCoalescedComponentsB
      # tP["rcv"] = self.readCoalescedVectorB
      tP["rpc"] = self.readPerpendicularComponentsB
      # tP["rpv"] = self.readPerpendicularVectorB
      tP["ruc"] = self.readUnrollDimComponentsB
      tP["wtc"] = self.writeTileDimComponentsB
      tP["wuc"] = self.writeUnrollDimComponentsB
      tP["idx"] = kernel["ProblemType"]["Index1"]
      tP["rc"] = kernel["ProblemType"]["IndexAssignmentsB"][0] \
          in [kernel["ProblemType"]["Index01B"], \
          kernel["ProblemType"]["IndexUnroll"]] # can read coalesced
      tP["NonTemporal"] = kernel["NonTemporalB"]

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherFreeAssignments(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherSummationAssignments(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A/B
  ##############################################################################
  @abc.abstractmethod
  def graShift(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graFinalOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def graAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Increments A/B
  # This function declares the increments
  ##############################################################################
  @abc.abstractmethod
  def graIncrements(self, kernel, loopIdx, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaTileAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaUnrollAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaFirstOffset(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Local Read Addresses: Tile Assignment
  ##############################################################################
  @abc.abstractmethod
  def lraTileAssignment(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lraFinalOffset(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read Addresses offset conversion for DTL + NLC > 1
  ##############################################################################
  @abc.abstractmethod
  def lraOffsetConversionForDTLandNLC(self, kernel, tP, offset_val, generateAsm=False, \
                                      finalVgpr=None, tmp1=None, tmp2=None):
    return ""

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def lraDeclareAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Recalculate local read addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalReadAddressesAB(self, kernel):
    return ""

  ##############################################################################
  # Recalculate local write addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalWriteAddresses(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Define stagger parms that will be used in calculateStagger
  ##############################################################################
  @abc.abstractmethod
  def declareStaggerParms(self, kernel):
    return ""


  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  @abc.abstractmethod
  def calculateStagger(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Remove stagger offset (before tail loop)
  ##############################################################################
  @abc.abstractmethod
  def removeStagger(self, kernel):
    return ""

  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  @abc.abstractmethod
  def calculateLoopNumIter(self, kernel, loopIdx):
    return ""


  ##############################################################################
  # openShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def openShadowInit(self, kernel):
    return ""

  ##############################################################################
  # closeShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def closeShadowInit(self, kernel):
    return ""

  ##############################################################################
  # Initialize C
  ##############################################################################
  @abc.abstractmethod
  def initC(self, kernel):
    return ""

  ##############################################################################
  # Open Loop
  # loopIdx<0 : tail loop
  ##############################################################################
  @abc.abstractmethod
  def openLoop(self, kernel, loopIdx, uDu, noLabelGen, beginLabelOnly):
    return ""

  ##############################################################################
  # Close Loop
  ##############################################################################
  @abc.abstractmethod
  def closeLoop(self, kernel, loopIdx, finalLoop, uDu, emitEndLabelOnly, oddLabel=False):
    return ""

  ##############################################################################
  # Open Loop Copy
  ##############################################################################
  @abc.abstractmethod
  def openLoopCopy(self, kernel, lc):
      return ""

  ##############################################################################
  # End Summation
  ##############################################################################
  @abc.abstractmethod
  def endSummation(self, kernel, label = None, isOptNLL = False):
    return ""

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  @abc.abstractmethod
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL):
    return ""

  @abc.abstractmethod
  def closeSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isNGLL):
    return ""

  ##############################################################################
  # Global Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def globalReadIncrementAB(self, kernel, loopIdx, prefetchIndex, incs=1):
    return ""

  ##############################################################################
  # Global Read: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def globalReadDo(self, kernel, mode, tP, vregSetIdx=0):
    return ""

  ##############################################################################
  # directToLds m0 update: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def directToLdsM0Update(self, kernel, mode, tP, usePlaceHolder=False):
    return ""

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Write: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteResetOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Write in Prefetch Pass (PreLoop): Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def preLoopLocalWriteDo(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteDo(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Read: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadResetOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInitPointers(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInc(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadDo(self, kernel, bufferIdx, innerUnrollIndex, epsi, tP):
    return ""

  ##############################################################################
  # Shift Vector Components d0/1
  ##############################################################################
  @abc.abstractmethod
  def shiftVectorComponents(self, kernel, tP):
    return ""

  ##############################################################################
  # Complex Declare Tmp Registers
  ##############################################################################
  @abc.abstractmethod
  def complexDeclareTmpRegisters(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  @abc.abstractmethod
  def localSplitULocalWrite(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  @abc.abstractmethod
  def localSplitULocalRead(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  @abc.abstractmethod
  def localSplitUReduction(self, kernel):
    return ""

  ##############################################################################
  # globalWriteWorkGroupInitBeforePersistentLoop:
  ##############################################################################
  @abc.abstractmethod
  def globalWriteWorkGroupInitBeforePersistentLoop(self, kernel):
    return ""

  ##############################################################################
  # globalWriteWorkGroupInit:
  # Perform work-group granularity init
  ##############################################################################
  @abc.abstractmethod
  def globalWriteWorkGroupInit(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWrite(self, kernel):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWrite(self, kernel):
    return ""

  ##############################################################################
  # openOddNoLoadLoopForDTV
  # generate open code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  @abc.abstractmethod
  def openOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    return ""

  ##############################################################################
  # closeOddNoLoadLoopForDTV
  # generate close code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  @abc.abstractmethod
  def closeOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    return ""

  ##############################################################################
  # generateEvenEndLabeNoLoadLoopForDTV
  # generate even end label for DirectToVgpr
  ##############################################################################
  @abc.abstractmethod
  def generateEvenEndLabeNoLoadLoopForDTV(self, kernel, isNGLL, name):
    return ""

  ##############################################################################
  # generateOddEndVgprCopyForDTV
  # generate odd end vgpr copy for DirectToVgpr
  ##############################################################################
  @abc.abstractmethod
  def generateOddEndVgprCopyForDTV(self, kernel):
    return ""

  ##############################################################################
  # PrefetchGlobalRead2
  ##############################################################################
  @abc.abstractmethod
  def openPrefetchGlobalRead2(self, kernel):
    return ""

  @abc.abstractmethod
  def closePrefetchGlobalRead2(self, kernel):
    return ""

  ##############################################################################
  # Function End
  ##############################################################################
  @abc.abstractmethod
  def functionEnd(self, kernel, addLabel=True):
    return ""

  ##############################################################################
  # Function Suffix
  ##############################################################################
  @abc.abstractmethod
  def functionSuffix(self, kernel):
    return ""

  ##############################################################################
  # WaitCnt
  ##############################################################################
  @abc.abstractmethod
  def wait(self, kernel, tPA, tPB, globalRead, localWrite, localRead, comment):
    return ""

  ##############################################################################
  # SyncThreads
  ##############################################################################
  @abc.abstractmethod
  def syncThreads(self, kernel):
    return ""

  ##############################################################################
  # MapAcctoArch
  ##############################################################################
  @abc.abstractmethod
  def MapAcctoArchRegs(self, kernel, option):
    return ""

  ##############################################################################
  #
  #   Entry Functions
  #
  ##############################################################################


  ##############################################################################
  # get kernel name
  ##############################################################################
  def getKernelFileBase(self, kernel):
    if isCustomKernelConfig(kernel):
      fileBase = kernel["CustomKernelName"]
    elif globalParameters["ShortNames"]:
      fileBase = Solution.getNameSerial(kernel, self.kernelSerialNaming)
    else:
      fileBase = self.shortenFileBase(kernel)
    return fileBase

  def getKernelName(self, kernel):
    kernelName = Solution.getNameMin(kernel, self.kernelMinNaming)
    return kernelName

  def getKernelSource(self, kernel):
    """
    Returns the source of the kernel, either C++ or assembly.
    """


    fileString = ""
    self.tPA = tensorParametersA = {}
    self.tPB = tensorParametersB = {}
    self.initKernel(kernel, tensorParametersA, tensorParametersB )
    self.stringIdx = 0
    (error, kb) = self.kernelBody( kernel, tensorParametersA, tensorParametersB)
    fileString += str(kb)

    if error != 0:
      if globalParameters["ForceGenerateKernel"]:
        print ("warning: Generating kernel source resulted in error {}, but ForceGenerateKernel=1 so saving source".format(error))
      else:
        raise RuntimeError("Generating kernel source resulted in error {}".format(error))

    return fileString

  def getAssemblyDirectory(self):
      return Common.ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly"))

  def byteArrayScriptSource(self):
    return """
#!/usr/bin/env python

fileString = ""
fileString += "/*******************************************************************************\\n"
fileString += "* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.\\n"
fileString += "*\\n"
fileString += "* Permission is hereby granted, free of charge, to any person obtaining a copy\\n"
fileString += '* of this software and associated documentation files (the \"Software\"), to deal\\n'
fileString += "* in the Software without restriction, including without limitation the rights\\n"
fileString += "* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-\\n"
fileString += "* ies of the Software, and to permit persons to whom the Software is furnished\\n"
fileString += "* to do so, subject to the following conditions:\\n"
fileString += "*\\n"
fileString += "* The above copyright notice and this permission notice shall be included in all\\n"
fileString += "* copies or substantial portions of the Software.\\n"
fileString += "*\\n"
fileString += '* THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-\\n'
fileString += "* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS\\n"
fileString += "* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR\\n"
fileString += "* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER\\n"
fileString += "* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-\\n"
fileString += "* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\\n"
fileString += "*******************************************************************************/\\n\\n"
fileString += "/**************************************************\\n"
fileString += "* This file was generated by Tensile:             *\\n"
fileString += "* https://github.com/ROCmSoftwarePlatform/Tensile *\\n"
fileString += "**************************************************/\\n\\n\\n"
import os.path
fileString += '#include "Kernels.h"\\n\\n'
fileString += "/* code object byte array */\\n\\n"
codeObjectFileNames = [f for f in os.listdir(".") if (os.path.isfile(f) and f.endswith(".co"))]
for codeObjectFileName in codeObjectFileNames:
  print codeObjectFileName
  print "\\n"
  kernelName=os.path.splitext(codeObjectFileName)[0]
  codeObjectFile = open(codeObjectFileName, "r")
  codeObjectByteArray = bytearray(codeObjectFile.read())
  codeObjectFile.close()
# write code object byte array for asm
  fileString += "const unsigned char %s_coba[%u] = {\\n" % (kernelName, len(codeObjectByteArray))
  for byteIdx in range(0, len(codeObjectByteArray)):
    byte = codeObjectByteArray[byteIdx]
    fileString += "0x%02x" % byte
    if byteIdx < len(codeObjectByteArray)-1:
      fileString += ","
    else:
      fileString += "};\\n"
    if byteIdx % 16 == 15:
      fileString += "\\n"
  text_file = open("Kernels.cpp", "w")
  text_file.write("%s" % fileString)
  text_file.close()
"""

  def writeByteArrayScript(self):
    asmPath = self.getAssemblyDirectory()

    bytearrayFileName = os.path.join(asmPath,"insert_byte_array.py")
    if not os.path.isfile(bytearrayFileName):
      with open(bytearrayFileName, 'w') as bytearrayFile:
        bytearrayFile.write(self.byteArrayScriptSource())
      os.chmod(bytearrayFileName, 0o777)
    return bytearrayFileName

  def shortenFileBase(self, kernel):
    base = self.getKernelName(kernel)
    if len(base) <= globalParameters["MaxFileName"]:
      return base

    import hashlib
    import base64

    pivot = globalParameters["MaxFileName"] * 3 // 4
    firstPart = base[:pivot]
    secondPart = base[pivot:]

    secondHash = hashlib.sha256(secondPart.encode()).digest()
    secondPart = base64.b64encode(secondHash, b'_-').decode()

    return firstPart + secondPart

  def getKernelObjectAssemblyFile(self, kernel):
    asmPath = self.getAssemblyDirectory()
    # write assembly file to assembly directory
    kernelName = self.getKernelFileBase(kernel)
    fileBase = os.path.join(asmPath, kernelName )
    assemblyFileName = "%s.s" % fileBase

    kernelSource = self.getKernelSource(kernel)

    if globalParameters["PrintLevel"] >= 2:
      print("write_assemblyFilename %s" % assemblyFileName)

    with open(assemblyFileName, 'w') as assemblyFile:
      assemblyFile.write(kernelSource)

    return assemblyFileName

  def getAssembledKernelObjectFile(self, kernel):
    assemblyFileName = self.getKernelObjectAssemblyFile(kernel)

    base, ext = os.path.splitext(assemblyFileName)
    objectFileName = base + '.o'

    args = self.getCompileArgs(assemblyFileName, objectFileName)
    if globalParameters["PrintCodeCommands"]:
      print (' '.join(args), " && ")

    subprocess.check_call(args, cwd=self.getAssemblyDirectory())

    return objectFileName

  def getSingleCodeObjectFile(self, kernel):
    objectFileName = self.getAssembledKernelObjectFile(kernel)

    base, ext = os.path.splitext(objectFileName)
    coFileName = base + '.co'

    args = self.getLinkCodeObjectArgs([objectFileName], coFileName)
    if globalParameters["PrintCodeCommands"]:
      print (' '.join(args))

    subprocess.check_call(args, cwd=self.getAssemblyDirectory())

    return coFileName

  ##############################################################################
  def getSourceFileString(self, kernel):
    """
    Returns a string suitable for placing in Kernels.cpp.  This means the actual kernel source in the case
    of a source kernel, or an assembled code object byte array definition in the case of an assembly kernel,
    or an empty string in the case that CodeFromFiles is true.

    In the case of an assembly kernel, this function has the side effect of creating the following files:
     * An assembly source file
     * An object file
     * A code object file
     * A Python script which can create byte array variable definitions.
    """

    try:
      if kernel["KernelLanguage"] == "Assembly":
        # asmPath = self.getAssemblyDirectory()
        # kernelName = self.getKernelName(kernel)

        if globalParameters["GenerateSourcesAndExit"]:
          # only create the assembly file.
          self.getKernelObjectAssemblyFile(kernel)
          return (0, "")
        else:
          self.writeByteArrayScript()
          self.getSingleCodeObjectFile(kernel)

          # I guess in this case we are making sure that the code object file exists by executing the code
          # above but we aren't placing it into the source.
          return (0, "")

      else:
        return (0, self.getKernelSource(kernel))

    except subprocess.CalledProcessError as exc:
      print(exc)
      return (-1, "")
    except RuntimeError as exc:
      if globalParameters["PrintSolutionRejectionReason"]:
        print(exc)
      return (-2, "")

  ##############################################################################
  # header file string
  ##############################################################################
  def getHeaderFileString(self, kernel):
    kernelName = self.getKernelName(kernel)
    fileString = "" # CHeader
    if self.language == "HIP" or self.language == "OCL":
      if not globalParameters["MergeFiles"]:
        fileString += CHeader
        fileString += "#pragma once\n\n"
        if self.language == "HIP":
          fileString += "#include <hip/hip_runtime.h>\n"
          fileString += "#include <hip/hip_fp16.h>\n"
          fileString += "#include <KernelHeader.h>\n"
          fileString += "\n"
        else:
          fileString += "#include <string>\n"
      if self.language == "OCL":
        fileString += "extern const char * const %s_src;\n" % kernelName
      else:
        fileString += self.functionSignature(kernel)
        fileString += ";\n"
    else:
      if not globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1:
        fileString += "#pragma once\n\n"
      if not globalParameters["CodeFromFiles"]:
        fileString += "extern const unsigned char %s_coba[]; // code object byte array\n" % kernelName

    return fileString

  ##############################################################################
  # flip Vreg set for DirectToVgpr in global read
  ##############################################################################
  def flipVregSetForDirectToVgprInGlobalRead(self, kernel, item):
    # need to swap VGPR register set for odd code
    baseName = "G2LA" if kernel["DirectToVgprA"] else "G2LB" # only one of them is enabled
    set0 = baseName + "0"
    set1 = baseName + "1"
    itemStr = str(item)
    if set0 in itemStr:
      # replace set0 with set1
      replacePlaceHolder(item, set0, set1)
    elif set1 in itemStr:
      # replace set1 with set0
      replacePlaceHolder(item, set1, set0)
    return item

  ##############################################################################
  # waitcnt code for DirectToVgpr
  ##############################################################################
  def getWaitcntCodeForDirectToVgpr(self, kernel, localWriteEndIter, u, firstIter, beforeBarrier=False, NLLlast=False, oddLast=False):
    inst = Code.Inst("", "No need to wait.")
    # generate wait only if BufferLoad is True (this logic does not work with FlatLoad)
    if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and kernel["BufferLoad"]:
      pgr2 = kernel["PrefetchGlobalRead"] == 2
      numGlobalReadA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"] * self.numReadVectorComponentsA
      numGlobalReadB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"] * self.numReadVectorComponentsB
      numGlobalRead = numGlobalReadA if kernel["DirectToVgprA"] else numGlobalReadB
      numGlobalReadAll = numGlobalReadA + numGlobalReadB
      numGlobalStoreC = 0
      numReadsIterCoalesced = self.numReadsIterCoalescedA if kernel["DirectToVgprA"] else self.numReadsIterCoalescedB
      waitComment = "global read wait for DirectToVgpr"
      # delay DirectToVgpr global read (from previous iteration) which is not referred yet
      numRegsIn1set = (numGlobalRead // kernel["LoopIters"]) * numReadsIterCoalesced
      numSet = (u + numReadsIterCoalesced) // numReadsIterCoalesced
      numSetMod = (u + numReadsIterCoalesced) % numReadsIterCoalesced
      if numSetMod > 0:
        # if mod > 0, wait is already done by mod == 0 case and no need to wait for same set of global read
        return inst
      needToWait = numGlobalRead - numSet * numRegsIn1set
      # No global load A, B in no load loop. Reset numGlobalReadAll and numGlobalRead
      numGlobalReadAll = 0
      numGlobalRead = 0
      if pgr2:
        # PGR=2 case, add numGlobalReadAll for second set of prefetch
        needToWait += numGlobalReadAll
      if u > 0:
        # count number of global read for i < u
        count = 0
        for i in range(u):
          globalReadStr = ' '.join([str(x) for x in self.perIterGlobalReadCode[i].flatitems()])
          count += globalReadStr.count("_buffer_load")
          # PGR=2 case, global read is in LocalWriteCode
          localWriteStr = ' '.join([str(x) for x in self.perIterLocalWriteCode[i].flatitems()])
          count += localWriteStr.count("_buffer_load")
        needToWait += count
        if u == localWriteEndIter + 1 and beforeBarrier:
          # beforeBarrier case, reduce the amount of non-Vgpr global read
          needToWait -= (numGlobalReadAll - numGlobalRead)
      # adjustment for oddLast
      # oddLast case, ignore all of above and set 0
      if oddLast:
        needToWait = 0
      inst = Code.WaitCnt(vmcnt=needToWait, comment=waitComment)
    return inst
