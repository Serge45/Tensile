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
from .Common import gfxName, globalParameters, print2, printExit, printWarning, roundUp
from .Component import Component
from .KernelWriter import KernelWriter
from .SolutionStructs import isPackedIndex
from .Utils import ceil_divide
from .AsmMemoryInstruction import MemoryInstruction
from .AsmRegisterPool import RegisterPool, SmartPoolContainer
from .AsmStoreState import StoreState
from .AsmAssert import Assert, bomb
from .AsmMacros import InstMacros, macroRegister
from .AsmUtils import vgpr, sgpr, accvgpr, mgpr, log2, s_mul_int_64_32, \
                      vectorStaticDivideAndRemainder, vectorStaticDivide, vectorStaticRemainder, \
                      scalarStaticDivideAndRemainder, staticMultiply, scalarStaticMultiply, sBranchIfZero, \
                      replacePlaceHolder, \
                      SaturateCastType, LabelManager
from .Activation import ActivationModule, ActivationType

from math import ceil, log
from copy import deepcopy
from typing import NamedTuple
import collections
import os
import shlex


################################################################################
# Assembly Kernel
################################################################################

class KernelWriterAssembly(KernelWriter):

  ##############################################################################
  # Init
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    super(KernelWriterAssembly, self).__init__( \
        kernelMinNaming, kernelSerialNaming)
    self.do = {}
    self.do["PreLoop"]     = True
    self.do["GlobalReadA"] = True
    self.do["GlobalReadB"] = True
    self.do["GlobalInc"]   = True
    self.do["LocalWrite"]  = True
    self.do["LocalReadA"]  = True
    self.do["LocalReadB"]  = True
    self.do["Wait"]        = True
    self.do["Sync"]        = True
    self.do["MAC"]         = True
    self.do["PostLoop"]    = True
    self.do["ApplyAlpha"]  = True
    self.do["GlobalWrite"] = True
    self.do["EdgeWrite"]   = True
    self.do["KeepDirectToLdsAlloc"] = False  # If true, keep regs used for LDS alloc even if not used

    # Remove me if 906 can work with beta in SGPR
    # Also can push alpha/beta recalc back to host for HPA mode
    self.betaInSgpr = True

    # Various debug flags and modes
    self.db = {}
    self.db["EnableAsserts"]       = globalParameters["EnableAsserts"]  # Enable assertion codegen. Requires 2 SGPR.
    self.db["DebugKernelMaxItems"] = 16  # Capture first N(=16) print values, ignore subsequent.  If -1, debug writing is faster but writing more than 16 values is undefined.

    # Chicken bit to add conservative synchronization at strategic points:
    # 0x01 = waitcnt + barrier after vector load
    # 0x02 = waitcnt at self.wait() for globalRead
    # 0x04 = waitcnt at self.wait() for localWrite
    # 0x08 = waitcnt at self.wait() for localRead
    # 0x10 = waitcnt after summation iteration, this can catch lingering ds or vm activity from summation loop
    # 0x20 = waitcnt before each write batch
    # 0x40 = waitcnt after each write batch
    self.db["ConservativeWaitCnt"] = 0x00

    self.db["InitLds"]     = False  # Initialize LDS at start of kernel
    self.initLdsValue     = 0xFFFFFFFF  # Value to use for LDS Init, if enabled

    # InitSgpr and InitVgpr can initialize at various points:
    #  0x1: Init at kernel start
    #  0x2: Init at end of summation loop (after tail too) - this is just before store loop
    self.db["InitSgpr"]   = 0x0  # init SGPRs
    self.initSgprValue    = 0x0  # Value to use for Sgpr Init, if enabled

    self.db["InitVgpr"]   = 0x0  # init VGPRs
    self.initVgprValue    = 0xFFFFFFFF  # Value to use for Vgpr Init, if enabled

    # Debug and Check flags:
    # Check A and B values loaded from memory to ensure they are 1
    # Requires DataInitTypeAB=1.
    # Only works if the problem uses full tiles (no edges)
    # Mismatches will assert (generate GPUVM fault)
    self.db["CheckValue1A"] = globalParameters["EnableDebugA"]
    self.db["CheckValue1B"] = globalParameters["EnableDebugB"]

    # Check value in C matrix.
    # Caveats:
    #  - Only works for single, or Half/BF with HPA.
    #  - Checks after alpha calc for each element.  Later elements (in the TT) will not yet have applied their alpha.
    #  - Only works if matrix is integral multiple of macro-tile (no edges) - check is dumb so doesn't know
    #    which work-items are outside the valid edge.
    #  - Does not work in OptNoLoadLoop
    self.db["CheckValueC"]  = globalParameters["EnableDebugC"]
    # value expected if CheckValueC is set. Use '.' for FP.
    # For example could be 16.0 if U=8 and alpha=2
    self.db["ValueCExpectedValue"] = globalParameters["ExpectedValueC"]

    # Force an expected value for all C outputs.
    # May be useful for checking store path
    # See same caveats as CheckValueC
    self.db["ForceExpectedValue"]  = globalParameters["ForceCExpectedValue"]

    # Force VSerial value into the output, this will
    # not match reference but can be useful to see which work-items are
    # storing which values
    # See same caveats as CheckValueC
    self.db["ForceVSerial"] = False

    # can't do both of these since they both override output
    assert (not (self.db["ForceExpectedValue"] and self.db["ForceVSerial"]))

    self.db["ForceInputValueA"] = False
    self.db["ForceInputValueB"] = False
    self.db["ForceValueA"] = 1.0
    self.db["ForceValueB"] = 1.0

    self.db["CheckStoreC"] = -1 # -1 disables, reload and verify output data.  Specify expected constant value.
    #self.db["CheckStoreC"] = 1024.0 # possible value

    self.db["ForceEdgeStores"] = 0 # 1=force use of edge store path for all tiles,  2=add assert in non-edge stores
    self.db["AssertNoEdge"] = 0 # Add assert in edge store code so crashes if executed

    # print vgpr register pool checkins and checkouts
    self.db["PrintRP"] = 0
    self.db["AssertOnSgprOverflow"] = False
    self.db["PrintStoreRegisterDb"] = False

    # Number of times localReadDo(localWriteDo) has been called by the code-generator.
    # Used to control debug enablement.
    # Note this increments as the assembly code is generated not as it executes
    # so it can be used to determine which iteration of the unroll is being generated
    self.localReadDoCnt   = 0
    self.localWriteDoCnt  = 0

    self.maxVgprs = 256
    # max allowed is 112 out of 112 , 6 is used by hardware 4 SGPRs are wasted
    self.maxSgprs = 102
    self.maxOccupancy = 10

    self.syncStr = "s_barrier"
    self.labels = LabelManager()
    self.localReadOffsetA = 0
    self.localReadOffsetB = 0
    self.inTailLoop = False

    # Activation related
    # A dict to prevent generating duplicate names in assembly
    self.globalWriteIfStateLabelSuffixDict = dict()

  @property
  def vcc(self) -> str:
    if self.kernel["WavefrontSize"] == 64:
      return "vcc"
    else:
      return "vcc_lo"

  @property
  def exec(self) -> str:
    if self.kernel["WavefrontSize"] == 64:
      return "exec"
    else:
      return "exec_lo"

  @property
  def laneSGPRCount(self) -> int:
    """ How many SGPRs does it take to have one bit per lane? """
    if self.kernel["WavefrontSize"] == 64:
      return 2
    else:
      return 1

  def getCompileArgs(self, sourceFileName, objectFileName, *moreArgs, isa=None, wavefrontSize=None):
    if isa is None:
      isa = self.version
    if wavefrontSize is None:
      wavefrontSize = self.kernel["WavefrontSize"]

    archHasV3 = globalParameters["AsmCaps"][isa]["HasCodeObjectV3"]

    launcher = shlex.split(os.environ.get('Tensile_ASM_COMPILER_LAUNCHER', ''))

    rv = launcher + [globalParameters['AssemblerPath'],
          '-x', 'assembler',
          '-target', 'amdgcn-amd-amdhsa']

    if archHasV3:
      rv += ['-mcode-object-version=2' if globalParameters["CodeObjectVersion"] == "V2" else '-mcode-object-version=4']

    rv += ['-mcpu=' + gfxName(isa)]

    if wavefrontSize == 64:
      rv += ['-mwavefrontsize64']
    else:
      rv += ['-mno-wavefrontsize64']

    rv += moreArgs

    rv += ['-c', '-o', objectFileName, sourceFileName]

    return rv

  def getLinkCodeObjectArgs(self, objectFileNames, coFileName, *moreArgs):
    rv = [globalParameters['AssemblerPath'],
          '-target', 'amdgcn-amd-amdhsa']
    rv += moreArgs
    rv += ['-o', coFileName] + objectFileNames
    return rv

  def getVgprOccupancy(self, numThreads, vgprs, unifiedVgprRegs=False):
    multiplier = int(ceil(max(numThreads, 256) / 256.0)) # example: wg=512 multiplier=2, 1024=4
    maxOccupancy = self.maxOccupancy//multiplier

    vgprAllocateAligned = 4    if not unifiedVgprRegs else 8
    totalVgprs = self.maxVgprs if not unifiedVgprRegs else self.maxVgprs*2
    vgprsAligned = int(ceil(vgprs/vgprAllocateAligned))*vgprAllocateAligned
    vgprsAligned *= multiplier

    if   vgprsAligned > totalVgprs:  return 0
    elif vgprsAligned < 1:           return maxOccupancy
    occupancy = min(totalVgprs//vgprsAligned, maxOccupancy)

    #print("vgprs = ", vgprs, "vgprsAligned = ", vgprsAligned, "unifiedVgprRegs = " ,unifiedVgprRegs, "Occupancy = ", occupancy)

    return occupancy

  ########################################
  def getOccupancy(self, numThreads, vgprs, ldsSize, accvgprs=0, unifiedVgprRegs=False):

    ldsLimitedOccupancy = self.getLdsLimitedOccupancy(ldsSize)

    if not unifiedVgprRegs:
      vgprLimitedOccupancy    = self.getVgprOccupancy(numThreads, vgprs,          unifiedVgprRegs)
      accvgprLimitedOccupancy = self.getVgprOccupancy(numThreads, accvgprs,       unifiedVgprRegs)
    else:
      vgprLimitedOccupancy    = self.getVgprOccupancy(numThreads, vgprs+accvgprs, unifiedVgprRegs)
      accvgprLimitedOccupancy = vgprLimitedOccupancy

    return min(ldsLimitedOccupancy, vgprLimitedOccupancy, accvgprLimitedOccupancy)

  # TODO: also consider sgpr
  def getMaxRegsForOccupancy(self, numThreads, vgprs, ldsSize, accvgprs=0, unifiedVgprRegs=False):
    lastVgprs = vgprs
    considerAccVgprs = 0       if not unifiedVgprRegs else accvgprs
    totalVgprs = self.maxVgprs if not unifiedVgprRegs else self.maxVgprs*2

    initOccupancy = self.getOccupancy(numThreads, vgprs, ldsSize, accvgprs, unifiedVgprRegs)
    if initOccupancy == 0: return lastVgprs

    while (vgprs + considerAccVgprs) < totalVgprs and vgprs < self.maxVgprs:
      vgprs += 1
      if self.getVgprOccupancy(numThreads, vgprs + considerAccVgprs, unifiedVgprRegs) >= initOccupancy:
        lastVgprs = vgprs
        next
      else:
        break

    return lastVgprs

  @staticmethod
  def getLdsLimitedOccupancy(ldsSize):
    maxLds = 65536
    # As ldsSize gets large, rounding might push us slightly higher than maxLds.
    # Clamp at maxLds
    ldsSize = min(ldsSize + 255, maxLds) & 0x1ff00 # 256-byte granularity

    ldsLimitedOccupancy = maxLds//ldsSize
    return ldsLimitedOccupancy

  @staticmethod
  def getLdsSize(kernel):
    ldsSize = kernel["LdsNumElements"] * kernel["ProblemType"]["DataType"].numBytes()
    return ldsSize

  ########################################
  def sizeRef(self, idx):
    """
    Return sgpr() or const with the specified size
    See above definitions for how these are mapped to Free or Sum sizes
    based on the problem definition.
    """
    idxChar= globalParameters["IndexChars"][idx]
    return sgpr("Size%s"%idxChar)

  def loopChar(self, kernel, loopIdx):
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    return globalParameters["IndexChars"][loopDim]

  def loopSizeRef(self, kernel, loopIdx):
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    return self.sizeRef(loopDim)

  def loopCounterName(self, kernel, loopIdx):
    return "LoopCounter%s"%(self.loopChar(kernel, loopIdx))

  def loopCounter(self, kernel, loopIdx):
    """
    Return loopCounter for loopIdx wrapped in "SGPR" syntax
    loop idx is 0...unrollIdx
    """
    return sgpr(self.loopCounterName(kernel,loopIdx))

  def checkLastIter(self, kernel, comment="at last iteration?"):
    """ Return last iteration of unroll loop. """
    if self.unrollIncIsDepthU:
      return Code.Inst("s_cmp_gt_u32", "DepthU", \
          sgpr("UnrollLoopLastIter"), comment)
    else:
      return Code.Inst("s_cmp_eq_u32", self.loopCounter(kernel, self.unrollIdx), \
          0, comment)

  def isConstUnitStride(self, stride):
      if isinstance(stride, Code.RegisterContainer):
        return False
      return stride.startswith("const")

  ########################################
  def strideRef(self, tc, dim):
    """
    Return sgpr with specified stride or define starting with const if constant.
    dim is index 0...max indices and is in global index space.
    """
    problemType = self.kernel["ProblemType"]
    if tc in ['A','B']:
      if not problemType["UseInitialStridesAB"] and \
          dim == problemType["IndexAssignments%s"%tc][0]:
        return ("constStride%s%s"%(tc,self.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.indexChars[dim]))
    elif tc in ['D','C']:
      if not problemType["UseInitialStridesCD"] and dim == 0:
        return ("constStride%s%s"%(tc,self.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.indexChars[dim]))
    else:
      raise ValueError("unexpected tensorChar='%s' in stride function"%tc)

  ##############################################################################
  # Find Memory Instruction For Width and Stride
  ##############################################################################
  def findMemoryInstructionForWidthStride(self, width, strides, combine, \
      instructions):
    for i in range(0, len(instructions)):
      instruction = instructions[i]
      numAddresses = instruction.numAddresses
      numOffsets = instruction.numOffsets
      offsetMultiplier = instruction.offsetMultiplier
      blockWidth = instruction.blockWidth
      valid = True
      if width < blockWidth:
        valid = False
      if combine: # try to combine ops
        if numOffsets > 0: # if inst combines using offsets
          for stride in strides:
            if stride % offsetMultiplier != 0:
              valid = False
      else: # don't try to combine ops
        if numOffsets > 1 or numAddresses > 1:
          valid = False
      if valid:
        return i
      else:
        continue

    printWarning("Could not find valid memory instruction for width=%f" % width)
    return len(instructions)

  ##############################################################################
  # Select Memory Instruction
  # when selecting instruction, need to support stride in both dims
  ##############################################################################
  def selectMemoryInstruction(self,
      operation, # ReadGlobal, WriteLocal, ReadLocal
      width, # num registers 1 chunk
      write2, # Para, Perp, None
      para2, # NumLoadsPara >= 2
      perp2, # NumLoadsPerp >= 2
      strides ):

    #instructions = self.memoryArchitecture[operation]
    instructions = self.memoryInstructions[operation]
    # try to combine
    if (write2 == "Coalesced" and para2) \
        or (write2 == "Perpendicular" and perp2):
      instructionIdx = self.findMemoryInstructionForWidthStride( \
          width, strides, True, instructions)
    # don't or can't combine
    else:
      instructionIdx = self.findMemoryInstructionForWidthStride( \
          width, strides, False, instructions)

    if instructionIdx < len(instructions): # found
      return instructionIdx
    else:
      raise RuntimeError("Could not find valid memory instruction for operation=%s, width=%f, kernel=%s" %(operation, width, self.kernelName))

  def getTmpSgpr(self, num, align=None, tag=None):
    if align==None:
      align = 1 if num==1 else 2
    if tag==None:
      tag = "getTmpSgpr(%d)"%num

    t = SmartPoolContainer(self.sgprPool, num, align, tag)
    if t.idx()+num > self.maxSgprs:
      self.overflowedResources = 2
      if self.db["AssertOnSgprOverflow"]:
        assert(t.idx()+num <= self.maxSgprs)
    return t

  def defineSgpr(self, name, numSgprs, align=1):
    if numSgprs == 0: return

    sgprIdx = self.sgprPool.checkOutAligned(numSgprs, align, tag=name, preventOverflow=0)
    #self.sgprIdx = roundUpToNearestMultiple(self.sgprIdx,align)
    #print (name, "->", self.sgprIdx, "+", numSgprs)
    self.sgprs[name] = sgprIdx

    return sgprIdx

  def undefineSgpr(self, name):
    self.sgprPool.checkIn(self.sgprs[name])
    # later references will result in compile-time error (with odd 'error: expected relocatable expression')
    # and 'Kernel ... not found in any loaded module'
    # TODO: temporarily disable undef as it seems to have issues
    return Code.Inst(".set", "%s" % name, "UNDEF", "")

  def defineVariableSgprs(self, kernel):
    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    # self.lastPostLoopSgpr = self.sgprPool.size()

    if self.unrollIncIsDepthU:
      # product of all summation dimensions, this also will be divided if GSU is enabled
      self.defineSgpr("UnrollLoopLastIter", 1)

    if self.use64bShadowLimit:
      # If need more SGPR could overlap this with the Tensor2dSize regs
      self.defineSgpr("ShadowLimitA", 2, 2)
      self.defineSgpr("ShadowLimitB", 2, 2)

    if self.staggerU:
      self.defineSgpr("StaggerUIter", 1)  # stagger loop iterations, used for various iter counts in the code
      self.defineSgpr("WrapUA", 2)  # Bytes to add to SrdA to reset address from N-1 iter to AddressA
      self.defineSgpr("WrapUB", 2)  # Bytes to add to SrdB to reset address from N-1 iter to AddressB

    self.defineSgpr("GlobalReadIncsA", self.numSgprGlobalReadIncsA)
    self.defineSgpr("GlobalReadIncsB", self.numSgprGlobalReadIncsB)

    if kernel["LocalWriteUseSgprA"]:
        self.defineSgpr("LocalWriteAddrA", 1)
    if kernel["LocalWriteUseSgprB"]:
        self.defineSgpr("LocalWriteAddrB", 1)

    if kernel["_UseSgprForGRO"]:
      needFirstSgprOffset = kernel["DirectToLdsA"] and kernel["UseInstOffsetForGRO"]
      numberOfSgpr = self.numGlobalReadOffsetsA if needFirstSgprOffset else (self.numGlobalReadOffsetsA-1)
      self.defineSgpr("ScalarGlobalReadOffsetA", numberOfSgpr)

      needFirstSgprOffset = kernel["DirectToLdsB"] and kernel["UseInstOffsetForGRO"]
      numberOfSgpr = self.numGlobalReadOffsetsB if needFirstSgprOffset else (self.numGlobalReadOffsetsB-1)
      self.defineSgpr("ScalarGlobalReadOffsetB", numberOfSgpr)

    # debug flag to allocate dummy / unused sgpr
    # useful when comparing code that adds new kernel arguments to see what
    # was actually changed
    numDummySgpr= 0
    for i in range(numDummySgpr):
      self.defineSgpr("DummySgpr%d"%i, 1)

    if self.sgprPool.size() >= self.maxSgprs:
      print ("warning: Number of defined SGPRS (%d) overflowed max SGPRS (%d)." \
               % (self.sgprPool.size(), self.maxSgprs))

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def initKernel(self, kernel, tPA, tPB ):
    super(KernelWriterAssembly, self).initKernel(kernel, tPA, tPB)

    dkp = kernel["DisableKernelPieces"]
    self.do["NullKernel"]  = dkp >= 9 or dkp == -9

    self.kernel = kernel

    # init these here in case some kernel pieces are disabled for performance exploration:
    tPA["localReadOffset"] = 0
    tPB["localReadOffset"] = 0

    self.sgprs=collections.OrderedDict()

    self.LdsOOB = 0xF00000

    #---
    # Internal optimization and debug controls.
    # These have a default which is almost always faster so don't make a full-blown YAML parm
    # But have a control here so we can disable for debugging and also easily tell
    # which parts of the code were changed to support the new mode.
    self.globalReadIncsUseVgpr = False if kernel["BufferLoad"] else True

    # If True, GRO are expressed as offsets from the beginning of the macro-tile, and the SRD
    # is set to the beginning of the macro-tile.
    # If False, GRO are expressed as offsets from the beginning of the lowest 2 dimensions
    # in the tensor.
    # True can allow Buffer-Based logic to have significantly higher range and handle larger tensors
    # groOffsetInMacroTile doesn't work with pointer-shift because it sets the SRD to point to the
    # start of the macro-tile - if we overhang by small number of elements (<GRVW) then can't shift
    # back to get all the data.
    # groOffsetInMacroTile doesn't work with packed dims since these need to set SRD to the tensor base
    # then extract the packed dimensions from the flattened index (including the workgroup) and scale by strides
    # - the index is per-work-item so can't put work-group into the SRD
    if len(kernel["PackedC0IndicesX"])==1 and len(kernel["PackedC1IndicesX"])==1 and kernel["BufferLoad"]:
      self.groOffsetInMacroTile = 1
    else:
      self.groOffsetInMacroTile = 0

    self.use64bPackSumOffset = 0  # use 2 SGPR for extracting packed summation dims.  Not supported, but this marks eventual required changes

    # use 64-bit buffer limit shadow register
    # but not implemented or tested
    self.use64bShadowLimit = kernel["Use64bShadowLimit"] and kernel["BufferLoad"]

    # Check if the address setup code for LWA and GRO causes register growth.
    # This is not an error condition but bears further investigation.
    # Realistically we just have the GlobalToLocal VGPRs, all else is growth.
    self.preventVgprOverflowDuringNewTile = 0 and not globalParameters["ForceGenerateKernel"]

    # For Beta:
    # Rather than waiting for all loads to finish with s_waitcnt vmcnt(0), interleave
    # appropriate vmcnts into the stores so they issue as loads become available
    self.interleaveStoreVmcnt = (not kernel["GroupLoadStore"]) and kernel["BufferStore"]

    # if >0, shift the start of the SRD left by specified #elements (not bytes)
    # Gives pointer shift some room to move left, even into the previous macro-tile
    # This slightly reduces the range of the GRO since they have to include the offset
    # Pointer shift still cannot be used with very small matrices < GRVW
    self.srdShiftLeft = {}
    self.srdShiftLeft["A"] = kernel["GlobalLoadVectorWidthA"]
    self.srdShiftLeft["B"] = kernel["GlobalLoadVectorWidthB"]

    self.checkGRO = False
    # checkGRO requires useSgprForGRO=0 so that code allocates and uses
    # the VGPRs that are used for the GRO offset checking
    assert not (kernel["_UseSgprForGRO"] and self.checkGRO)

    # Debug mode to explore combining VGPRs.
    # Saves VGPRs but doesn't generate correct answer
    self.combineLocalAddresses = 0

    # ISA version, such as 803
    self.version = globalParameters["CurrentISA"]
    if "ISA" in kernel:
      self.version = tuple(kernel["ISA"])
    if not globalParameters["AsmCaps"][self.version]["SupportedISA"]:
      defaultIsa = (9,0,0)
      print("warning: ISA:", self.version, " is not supported; overriding with ", defaultIsa)
      self.version = defaultIsa

    self.unifiedVgprRegs = False
    if globalParameters["ArchCaps"][self.version]["ArchAccUnifiedRegs"]:
      self.unifiedVgprRegs = True

    if kernel["EnableMatrixInstruction"]:
      if (kernel["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() == 'f64') and (not self.asmCaps["HasMFMA_f64"]):
        raise RuntimeError("FP64 MatrixInstruction not supported for {0}".format(self.version))
      elif not self.asmCaps["HasMFMA"]:
        raise RuntimeError("MatrixInstruction not supported for {0}".format(self.version))

      if kernel["MFMA_BF16_1K"] and not self.asmCaps["HasMFMA_bf16_1k"]:
        raise RuntimeError("BF16_1k MatrixInstruction not supported for {0}".format(self.version))

    self.AsmBugs = {}
    self.AsmBugs["ExplicitCO"] = globalParameters["AsmCaps"][self.version]["HasExplicitCO"]
    self.AsmBugs["ExplicitNC"] = globalParameters["AsmCaps"][self.version]["HasExplicitNC"]

    if not globalParameters["AsmCaps"][self.version]["HasDirectToLds"]:
      kernel["DirectToLdsA"] = False
      kernel["DirectToLdsB"] = False
      kernel["LocalWriteUseSgprA"] = False # Requires DirectToLdsA
      kernel["LocalWriteUseSgprB"] = False # Requires DirectToLdsB

    # The inst HasAtomicAdd is using is not compatible with int32.
    self.useAtomicAdd = (self.asmCaps["HasAtomicAdd"] and kernel["ProblemType"]["ComputeDataType"].isSingle()) and \
                        (kernel["_GlobalAccumulation"] == 'SingleBuffer')

    #######################################L
    # Available Memory Instructions
    ########################################

    # name, numAddresses, numOffsets, offsetMultiplier, blockWidth, formatting):
    ########################################
    # Local Read
    _ds_load_b128 = MemoryInstruction("_ds_load_b128",  1, 1, 4, 4, \
        "%s, %s offset:%s" )
    _ds_load2_b64 = MemoryInstruction("_ds_load2_b64",  1, 2, 2, 2, \
        "%s, %s offset0:%s, offset1:%s" )
    _ds_load_b64 = MemoryInstruction("_ds_load_b64",    1, 1, 2, 2, \
        "%s, %s offset:%s" )
    _ds_load2_b32 = MemoryInstruction("_ds_load2_b32",  1, 2, 1, 1, \
        "%s, %s offset0:%s offset1:%s" )
    _ds_load_b32 = MemoryInstruction("_ds_load_b32",    1, 1, 1, 1, \
        "%s, %s offset:%s" )
    _ds_load_u16 = MemoryInstruction("_ds_load_u16",    1, 1, 1, 0.5, \
        "%s, %s offset:%s" )
    _ds_load_u8 = MemoryInstruction("_ds_load_u8",      1, 1, 1, 0.25, \
        "%s, %s offset:%s" )
    ########################################
    # Local Write
    _ds_store_b128 = MemoryInstruction("_ds_store_b128",  1, 1, 4, 4, \
        "%s, %s offset:%s" )
    _ds_store2_b64 = MemoryInstruction("_ds_store2_b64",  1, 2, 2, 2, \
        "%s, %s, %s offset0:%s, offset1:%s" )
    _ds_store_b64 = MemoryInstruction("_ds_store_b64",    1, 1, 2, 2, \
        "%s, %s offset:%s" )
    _ds_store2_b32 = MemoryInstruction("_ds_store2_b32",  1, 2, 1, 1, \
        "%s, %s, %s offset0:%s offset1:%s" )
    _ds_store_b32 = MemoryInstruction("_ds_store_b32",    1, 1, 1, 1, \
        "%s, %s offset:%s" )
    _ds_store_b16 = MemoryInstruction("_ds_store_b16",    1, 1, 1, 0.5, \
        "%s, %s offset:%s" )
    _ds_store_b8 = MemoryInstruction("_ds_store_b8",      1, 1, 1, 0.25, \
        "%s, %s offset:%s" )
    ########################################
    # Global Read
    _flat_load_b128 = MemoryInstruction("_flat_load_b128", 1, 0, 0, 4, \
        "UNUSED %s, %s" )
    _flat_load_b64 = MemoryInstruction("_flat_load_b64",   1, 0, 0, 2, \
        "UNUSED %s, %s" )
    _flat_load_b32 = MemoryInstruction("_flat_load_b32",   1, 0, 0, 1, \
        "UNUSED %s, %s" )

    _buffer_load_b128 = MemoryInstruction("_buffer_load_b128", 1, 0, 0, 4, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s" )
    _buffer_load_b64 = MemoryInstruction("_buffer_load_b64", 1, 0, 0, 2, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s" )
    _buffer_load_b32 = MemoryInstruction("_buffer_load_b32", 1, 0, 0, 1, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s" )
    # generate half directly w/o using the format string to handle hi/lo correctly
    _buffer_load_d16_b16 = MemoryInstruction("_buffer_load_d16_b16", 1, 0, 0, 0.5, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s" )
    # generate byte directly w/o using the format string to handle hi/lo correctly
    _buffer_load_d16_u8 = MemoryInstruction("_buffer_load_d16_u8", 1, 0, 0, 0.25, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s" )

    self.buff_load_inst_offset_max = 4096

    ########################################
    # Global Write
    _flat_store_b128 = MemoryInstruction("_flat_store_b128", 1, 0, 0, 4, \
        "%s, %s" )
    _flat_store_b64  = MemoryInstruction("_flat_store_b64",  1, 0, 0, 2, \
        "%s, %s" )
    _flat_store_b32  = MemoryInstruction("_flat_store_b32",  1, 0, 0, 1, \
        "%s, %s" )

    ########################################
    # Available Memory Instructions per Architecture
    # gfx701 "Hawaii"
    # gfx801 "Carrizo"
    # gfx802 "Tonga"
    # gfx803 "Fiji"
    # gfx900
    ########################################
    if (kernel["BufferLoad"]):
      chosen_load_b128 = _buffer_load_b128
      chosen_load_b64  = _buffer_load_b64
      chosen_load_b32  = _buffer_load_b32
      chosen_load_b16  = _buffer_load_d16_b16
      chosen_load_b8   = _buffer_load_d16_u8
    else:
      chosen_load_b128 = _flat_load_b128
      chosen_load_b64  = _flat_load_b64
      chosen_load_b32  = _flat_load_b32
      chosen_load_b16  = _flat_load_b32 # not supported
      chosen_load_b8   = _flat_load_b32 # not supported

    chosen_store_b128 = _flat_store_b128
    chosen_store_b64  = _flat_store_b64
    chosen_store_b32  = _flat_store_b32

    self.memoryInstructions = {
          "GlobalRead" : [ chosen_load_b128, chosen_load_b64, chosen_load_b32,
                           chosen_load_b16, chosen_load_b8 ],
          "GlobalWrite": [ chosen_store_b128, chosen_store_b64, chosen_store_b32 ],
          "LocalRead"  : [ _ds_load_b128, _ds_load2_b64, _ds_load_b64,
                           _ds_load2_b32, _ds_load_b32, _ds_load_u16, _ds_load_u8 ],
          "LocalWrite" : [ _ds_store_b128, _ds_store2_b64, _ds_store_b64, _ds_store2_b32,
                           _ds_store_b32, _ds_store_b16, _ds_store_b8 ]
        }

    if self.asmCaps["v_fma_mix_f32"]:
      self.mixinst = "v_fma_mix_f32"
    elif self.asmCaps["v_mad_mix_f32"]:
      self.mixinst = "v_mad_mix_f32"
    else:
      self.mixinst = "NOT_SUPPORTED"

    self.overflowedResources = 0 # if true, comment out whole kernel

    self.kernelName = self.getKernelName(kernel)
    self.inTailLoop = False
    self.serializedStore = False
    self.codeAccVgprRead = None
    self.codeMulAlpha = None

    # condition(s) to allocate tile offset and unroll offset registers for PK kernel # FIXME: Remove these?
    self.useGlobalReadTileVgpr = False

    # registers per element
    self.bpr = 4 # all registers are 32bit

    # default setup
    # AB=DataType / Cexternal=DestDataType / Cinternal=Accumulation (MAC or MFMA)
    self.bpeAB = int(self.bpr * kernel["ProblemType"]["DataType"].numRegisters())

    # Cexternal = the "current" kernel output type,
    # - default: the "current" kernel is a non-GSU-kernel,
    #     Cexternal (= DestDataType) and is the final gemm result
    #
    # - For GSU: the "current" kernel is a GSU-kernel,
    #     this kernel returns a temp buffer with same type as Cinternal.
    #     Later, another kernel will accumulate this buffer
    #     and convert the final result to Cexternal (= DestDataType) as the gemm result
    self.bpeCexternal = int(self.bpr * kernel["ProblemType"]["DestDataType"].numRegisters())

    # already covers: dgemm, cgemm, zgemm, sgemm
    #               : hgemm  + !HPA ([H/H/H] compute = internal = f16)
    #               : hgemm  +  HPA ([H/H/S] or [H/S/S] compute = internal = f32)
    #               : bfgemm +  HPA ([B/B/S] or [H/S/S] compute = internal = f32)
    #               : int8x4-gemm   (internal = i32)
    self.bpeCinternal = int(self.bpr * kernel["ProblemType"]["ComputeDataType"].numRegisters())

    #jgolds Need to check device for support
    # HPA not allowed in dgemm, cgemm, zgemm, sgemm
    if kernel["ProblemType"]["HighPrecisionAccumulate"] and \
       not (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16() or \
          kernel["ProblemType"]["DataType"].isInt8x4() or kernel["ProblemType"]["DataType"].isInt8()):
        print("HighPrecisionAccumulate only valid when DataType is half, bf16, Int8x4, Int8. Forcing HPA to False")
        kernel["ProblemType"]["HighPrecisionAccumulate"] = False

    self.bpeCexternal = self.bpeCinternal if kernel["_GlobalAccumulation"] else self.bpeCexternal

    assert self.bpeAB == tPA["bpe"]
    assert self.bpeAB == tPB["bpe"]
    # registers per global address
    self.rpga = 2 # 64-bit
    # registers per local address
    self.rpla = 1 # 32-bit
    # registers per global 32-bit offset (some intructions only support 32-bit offset)
    self.rpgo = 1 # 32-bit

    ####################################
    # choose memory instructions
    ####################################

    ########################################
    # globalReadA instruction; no flat_load2_*
    self.globalReadWidthA = float(tPA["nrcv"]*tPA["bpe"])/self.bpr
    self.globalRead2CoalescedA = kernel["NumLoadsCoalescedA"]>1 \
        or self.readCoalescedComponentsA
    self.globalRead2PerpendicularA = kernel["NumLoadsPerpendicularA"] > 1 \
        or self.readPerpendicularComponentsA
    self.globalReadInstructionIdxA = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthA, \
        False, \
        self.globalRead2CoalescedA, self.globalRead2PerpendicularA, [] )
    ########################################
    # globalReadB instruction; no flat_load2_
    self.globalReadWidthB = float(tPB["nrcv"]*tPB["bpe"])/self.bpr
    self.globalRead2CoalescedB = kernel["NumLoadsCoalescedB"]>1 \
        or self.readCoalescedComponentsB
    self.globalRead2PerpendicularB = kernel["NumLoadsPerpendicularB"] > 1 \
        or self.readPerpendicularComponentsB
    self.globalReadInstructionIdxB = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthB, \
        False, \
        self.globalRead2CoalescedB, self.globalRead2PerpendicularB, [] )

    ########################################
    # localWriteA instruction
    # for local, tile->para, unroll->perp
    #self.localWriteWidthA = 1 if (self.writeTileDimComponentsA \
    #    or self.writeUnrollDimComponentsA) else kernel["VectorWidth"]
    self.localWriteWidthA = tPA["nwcv"]*tPA["bpe"]//self.bpr
    if self.localWriteWidthA < 1:
      self.localWriteWidthA = (1.0*tPA["nwcv"]*tPA["bpe"])/self.bpr
    self.localWrite2CoalescedA = tPA["nrc"]>1 \
        or self.writeTileDimComponentsA
    self.localWrite2PerpendicularA = tPA["nrp"]>1 \
        or self.writeUnrollDimComponentsA
    # localWriteA stride tile
    if kernel["ProblemType"]["TLUA"]:
      if self.writeTileDimComponentsA:
        self.localWriteStrideTileA = 1
        self.localWriteJoinTileA = "Components"
      else:
        self.localWriteStrideTileA = kernel["LSCA"]
        self.localWriteJoinTileA = "Coalesced"
    else:
      if self.writeUnrollDimComponentsA:
        self.localWriteStrideTileA = 1
        self.localWriteJoinTileA = "Components"
      else:
        self.localWriteStrideTileA = kernel["LSPA"]
        self.localWriteJoinTileA = "Perpendicular"
    self.localWriteStrideTileA = self.localWriteStrideTileA*tPA["bpe"]//self.bpr
    # localWriteA stride unroll
    if kernel["ProblemType"]["TLUA"]:
      if self.writeUnrollDimComponentsA:
        self.localWriteStrideUnrollA = 1*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Components"
      else:
        self.localWriteStrideUnrollA = kernel["LSCA"]*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Perpendicular"
    else:
      if self.writeTileDimComponentsA:
        self.localWriteStrideUnrollA = 1*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Components"
      else:
        self.localWriteStrideUnrollA = kernel["LSCA"]*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Coalesced"
    self.localWriteStrideUnrollA = \
        (self.localWriteStrideUnrollA*tPA["bpe"])//self.bpr
    self.localWriteInstructionIdxA = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthA, \
        False, \
        self.localWrite2CoalescedA, self.localWrite2PerpendicularA,
        [self.localWriteStrideTileA, self.localWriteStrideUnrollA] )

    ########################################
    # localWriteB instruction
    # for local, tile->para, unroll->perp
    #self.localWriteWidthB = 1 if (self.writeTileDimComponentsB \
    #    or self.writeUnrollDimComponentsB) else kernel["VectorWidth"]
    self.localWriteWidthB = tPB["nwcv"]*tPB["bpe"]//self.bpr
    if self.localWriteWidthB < 1:
      self.localWriteWidthB = (1.0*tPB["nwcv"]*tPB["bpe"])/self.bpr
    self.localWrite2CoalescedB = tPB["nrc"]>1 \
        or self.writeTileDimComponentsB
    self.localWrite2PerpendicularB = tPB["nrp"]>1 \
        or self.writeUnrollDimComponentsB
    # localWriteB stride tile
    if kernel["ProblemType"]["TLUB"]:
      if self.writeTileDimComponentsB:
        self.localWriteStrideTileB = 1
        self.localWriteJoinTileB = "Components"
      else:
        self.localWriteStrideTileB = kernel["LSCB"]
        self.localWriteJoinTileB = "Coalesced"
    else:
      if self.writeUnrollDimComponentsB:
        self.localWriteStrideTileB = 1
        self.localWriteJoinTileB = "Components"
      else:
        self.localWriteStrideTileB = kernel["LSPB"]
        self.localWriteJoinTileB = "Perpendicular"
    self.localWriteStrideTileB = (self.localWriteStrideTileB*tPB["bpe"])//self.bpr
    # localWriteB stride unroll
    if kernel["ProblemType"]["TLUB"]:
      if self.writeUnrollDimComponentsB:
        self.localWriteStrideUnrollB = 1*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Components"
      else:
        self.localWriteStrideUnrollB = kernel["LSCB"]*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Perpendicular"
    else:
      if self.writeTileDimComponentsB:
        self.localWriteStrideUnrollB = 1*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Components"
      else:
        self.localWriteStrideUnrollB = kernel["LSCB"]*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Coalesced"
    self.localWriteStrideUnrollB = \
        (self.localWriteStrideUnrollB*tPB["bpe"])//self.bpr
    self.localWriteInstructionIdxB = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthB, \
        False, \
        self.localWrite2CoalescedB, self.localWrite2PerpendicularB,
        [self.localWriteStrideTileB, self.localWriteStrideUnrollB] )

    ########################################
    # localRead A
    localReadWidth = (kernel["VectorWidth"] * tPA["bpe"]) // self.bpr
    if kernel["EnableMatrixInstruction"]:
      if tPA["tlu"] and self.allowLRVWforTLUandMI:
        localReadWidth = (self.lrvwA * tPA["bpe"]) // self.bpr
      else:
        localReadWidth = tPA["bpe"] / self.bpr
    if kernel["UnrollMajorLDSA"]:
      localReadWidth = (self.lrvwA * tPA["bpe"]) // self.bpr
    # for directToLds x2/x4 support
    if kernel["DirectToLdsA"]:
      localReadWidth  = 1    # for fp64 its f32

    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedA = \
        kernel["ThreadTile0"] * tPA["bpe"]//self.bpr
    self.localRead2CoalescedA = kernel["ThreadTile0"]//kernel["VectorWidth"] > 1
    self.localReadInstructionIdxA = \
        self.selectMemoryInstruction("LocalRead", localReadWidth, \
        False, \
        self.localRead2CoalescedA, localRead2Perpendicular,
        [self.localReadStrideCoalescedA] )
    tPA["localReadSwapByteOffset"] = 0
    tPB["localReadSwapByteOffset"] = 0
    tPA["localWriteSwapByteOffset"] = 0
    tPB["localWriteSwapByteOffset"] = 0

    ########################################
    # localRead B
    localReadWidth = (kernel["VectorWidth"] * tPB["bpe"]) // self.bpr
    if kernel["EnableMatrixInstruction"]:
      if tPB["tlu"] and self.allowLRVWforTLUandMI:
        localReadWidth = (self.lrvwB * tPB["bpe"]) // self.bpr
      else:
        localReadWidth = tPB["bpe"] / self.bpr
    if kernel["UnrollMajorLDSB"]:
      localReadWidth = (self.lrvwB * tPB["bpe"]) // self.bpr
    # for directToLds x2/x4 support
    if kernel["DirectToLdsB"]:
      localReadWidth  = 1    # for fp64 its f32

    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedB = \
    kernel["ThreadTile1"] * tPB["bpe"]//self.bpr
    self.localRead2CoalescedB = kernel["ThreadTile1"]//kernel["VectorWidth"] > 1
    self.localReadInstructionIdxB = \
        self.selectMemoryInstruction("LocalRead", localReadWidth, \
        False, \
        self.localRead2CoalescedB, localRead2Perpendicular,
        [self.localReadStrideCoalescedB] )

    instructions = self.memoryInstructions
    self.globalReadInstructionA = instructions["GlobalRead"][ \
        self.globalReadInstructionIdxA]
    self.globalReadInstructionB = instructions["GlobalRead"][ \
        self.globalReadInstructionIdxB]
    self.localWriteInstructionA = instructions["LocalWrite"][ \
        self.localWriteInstructionIdxA]
    self.localWriteInstructionB = instructions["LocalWrite"][ \
        self.localWriteInstructionIdxB]
    self.localReadInstructionA = instructions["LocalRead"][ \
        self.localReadInstructionIdxA]
    self.localReadInstructionB = instructions["LocalRead"][ \
        self.localReadInstructionIdxB]
    # global reads per instruction
    tPA["nrcvpi"] = int((self.globalReadInstructionA.totalWidth*self.bpr)/tPA["bpe"])
    tPB["nrcvpi"] = int((self.globalReadInstructionB.totalWidth*self.bpr)/tPB["bpe"])
    tPA["nwcvpi"] = int((self.localWriteInstructionA.totalWidth*self.bpr)/tPA["bpe"])
    tPB["nwcvpi"] = int((self.localWriteInstructionB.totalWidth*self.bpr)/tPB["bpe"])
    ####################################
    # VGPR Allocation
    ####################################

    ####################################
    # num vgprs: valu
    #jgolds bpeCinternal because we are allocating accumulation registers here
    self.numVgprValuC = (kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.bpeCinternal)//self.bpr

    PLR = kernel["PrefetchLocalRead"] if kernel["PrefetchLocalRead"] < kernel["LoopIters"] else kernel["LoopIters"] - 1
    valuBlocks = (1+PLR) * kernel["InnerUnroll"]
    # double the number of VgprValu if self.vgprValuDouble is true
    if self.vgprValuDouble:
      valuBlocks *= 2
    if kernel["EnableMatrixInstruction"]:
      self.numVgprValuAPerBlock = kernel["MIWaveTileA"] * kernel["MIInputPerThread"] * tPA["bpe"] // self.bpr
      self.numVgprValuBPerBlock = kernel["MIWaveTileB"] * kernel["MIInputPerThread"] * tPB["bpe"] // self.bpr
    else:
      printExit("TensileLite does not support non MFMA.")

    # change numVgprValuAPerBlock to 0 for A if DirectToVgpr is enabled
    if kernel["DirectToVgprA"]:
      self.numVgprValuAPerBlock = 0
    self.numVgprValuA = self.numVgprValuAPerBlock * valuBlocks
    # change numVgprValuBPerBlock to 0 for B if DirectToVgpr is enabled
    if kernel["DirectToVgprB"]:
      self.numVgprValuBPerBlock = 0
    self.numVgprValuB = self.numVgprValuBPerBlock * valuBlocks

    ####################################
    # num vgprs: global -> local elements
    self.numVgprG2LA = 0
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      self.numVgprG2LA = roundUp((kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"] * \
        kernel["GlobalLoadVectorWidthA"] * tPA["bpe"]) / (float)(self.bpr))
    # using _ds_store_b8: need one more vgpr space to do lshr
    if self.localWriteInstructionA.blockWidth == 0.25:
      self.numVgprG2LA = self.numVgprG2LA * 2
    # double numVgprG2LA if DirectToVgpr is enabled
    if kernel["DirectToVgprA"]:
      self.numVgprG2LA = self.numVgprG2LA * 2

    self.numVgprG2LB = 0
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      self.numVgprG2LB = roundUp((kernel["NumLoadsCoalescedB"] * kernel["NumLoadsPerpendicularB"] * \
        kernel["GlobalLoadVectorWidthB"] * tPB["bpe"]) / (float)(self.bpr))
    # using _ds_store_b8: need one more vgpr space to do lshr
    if self.localWriteInstructionB.blockWidth == 0.25:
      self.numVgprG2LB = self.numVgprG2LB * 2
    # double numVgprG2LB if DirectToVgpr is enabled
    if kernel["DirectToVgprB"]:
      self.numVgprG2LB = self.numVgprG2LB * 2

    ####################################
    # num vgprs: local read addresses
    self.numVgprLocalReadAddressesA = 1 * self.rpla
    self.numVgprLocalReadAddressesB = 1 * self.rpla
    # do not allocate local read address register if DirectToVgpr is enabled
    if kernel["DirectToVgprA"]:
      self.numVgprLocalReadAddressesA = 0
    if kernel["DirectToVgprB"]:
      self.numVgprLocalReadAddressesB = 0

    ####################################
    # num vgprs: local write addresses
    #numLocalWritesA = kernel["NumLoadsCoalescedA"] \
    #    * nlp * self.numWriteVectorComponentsA
    #numLocalWriteInstructionsA = numLocalWritesA \
    #    / self.localWriteInstructionA[self.instructionIdxNumOffsets]
    self.numVgprLocalWriteAddressesA = 0 if kernel["LocalWriteUseSgprA"] else 1 * self.rpla

    #numLocalWritesB = kernel["NumLoadsCoalescedB"] \
    #    * nlp * self.numWriteVectorComponentsB
    #numLocalWriteInstructionsB = numLocalWritesB \
    #    / self.localWriteInstructionB[self.instructionIdxNumOffsets]
    self.numVgprLocalWriteAddressesB = 0 if kernel["LocalWriteUseSgprB"] else 1 * self.rpla

    # do not allocate local write address register if DirectToVgpr is enabled
    if kernel["DirectToVgprA"]:
      self.numVgprLocalWriteAddressesA = 0
    if kernel["DirectToVgprB"]:
      self.numVgprLocalWriteAddressesB = 0

    ####################################
    # num vgprs: global read addresses
    numGlobalReadsA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["GlobalLoadVectorWidthA"] \
        * self.numReadVectorComponentsA
    numGlobalReadInstructionsA = (numGlobalReadsA * tPA["bpe"])//\
        (self.globalReadInstructionA.blockWidth * 4)

    if kernel["BufferLoad"]:
      self.numGlobalReadOffsetsA = roundUp(numGlobalReadInstructionsA * self.rpgo)
    else:
      numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.rpga

    numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["GlobalLoadVectorWidthB"] \
        * self.numReadVectorComponentsB
    numGlobalReadInstructionsB = (numGlobalReadsB * tPB["bpe"])// \
        (self.globalReadInstructionB.blockWidth * 4)
    if kernel["BufferLoad"]:
      self.numGlobalReadOffsetsB = roundUp(numGlobalReadInstructionsB * self.rpgo)
    else:
      numVgprGlobalReadAddressesB = numGlobalReadInstructionsB * self.rpga
    if self.globalReadIncsUseVgpr:
      numVgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga
      numVgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga
    else:
      numVgprGlobalReadIncsA = 0
      numVgprGlobalReadIncsB = 0

    numVgprAddressDbg = self.rpga if globalParameters["DebugKernel"] else 0

    ####################################
    # num vgprs: c write address
    # 1 address where to write first value
    # 1 tmp address where to write current value

    ####################################
    # VGPR Assignment
    ####################################
    vgprIdx = 0
    self.totalAgprs = 0
    self.startVgprValuC = vgprIdx; vgprIdx += self.numVgprValuC

    if kernel["EnableMatrixInstruction"]:
      # MI kernels can overlap C-tile w/ AB-tile up until writeback. Illustrated below:
      # |<-------------- valuC -------------->|
      # |------------|-----------|xx|---------|
      #   lastValuAB ^           ^  ^         ^
      #         lastVgprForReads ^  ^         ^
      #              startVgprReuse ^         ^
      #                             lastValuC ^
      # TODO a bit tricky. Better to manage all GPRs solely through RegisterPool
      self.serializedStore = True

      ########################################
      # AGPR Allocation
      ########################################
      if not kernel["MIArchVgpr"]:
        self.totalAgprs = self.numVgprValuC
        vgprIdx = 0
        self.numVgprValuC = 0

      self.startaccValuC0 = None
      self.startaccValuC1 = None

    # TODO: alignment hack, figure out a better solution
    vgprIdx = ((vgprIdx+1)//2)*2
    # Avoid bank conflict between VgprA and VgprC
    if (self.version[0] == 10) and ((vgprIdx % 4) == (self.startVgprValuC % 4)):
      vgprIdx += 1
    self.startVgprValuA = vgprIdx; vgprIdx += self.numVgprValuA
    self.startVgprG2LA = None
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      # if PGR = True, PAP could be possibly enabled, we move G2LA later to prevent it from being reclaimed
      # otherwise, put G2L here since it can overlap valu
      if not kernel["PrefetchGlobalRead"] and not kernel.enabledSplitLDS: # g2l can overlap valu
        self.startVgprG2LA = self.startVgprValuA
        vgprIdx = self.startVgprValuA \
            + max(self.numVgprValuAPerBlock*valuBlocks, self.numVgprG2LA)

    # TODO: alignment hack, figure out a better solution
    vgprIdx = ((vgprIdx+1)//2)*2
    self.startVgprValuB = vgprIdx; vgprIdx += self.numVgprValuB
    self.startVgprG2LB = None
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      # if PGR = True, PAP could be possibly enabled, we move G2LB later to prevent it from being reclaimed
      # otherwise, put G2L here since it can overlap valu
      if not kernel["PrefetchGlobalRead"] and not kernel.enabledSplitLDS: # g2l can overlap valu
        self.startVgprG2LB = self.startVgprValuB
        vgprIdx = self.startVgprValuB \
            + max(self.numVgprValuBPerBlock*valuBlocks, self.numVgprG2LB)

    # Registers allocated above this point can be used as temps during setup
    # Registers above here are reserved in initC, near the end of the setup
    # code
    self.lastValuAB = vgprIdx
    #----------------------------------
    # Point at last VGPR that can be reclaimed for use in the summation loop
    # If more VGPRs are added here be aware of the register reclaim code in
    # endSummation - registers that should be preserved after lastVgprForReads
    #
    # For PAP: decide the reclaim case
    # if we're not doing PAP, then the GlobalRead, LocalWrite, LocalRead, VgprG2L can be reclaimed
    # (and we'll extend the "lastVgprForReads" value later)
    # otherwise if we have PAP, they can't be reclaimed so we simply use the current vgprIdx
    self.lastVgprForReads = vgprIdx
    #----------------------------------

    if not kernel["LocalWriteUseSgprA"]:
      if self.combineLocalAddresses:
        self.startVgprLocalWriteAddressesA = self.startVgprLocalReadAddressesA
      else:
        self.startVgprLocalWriteAddressesA = vgprIdx
        vgprIdx += self.numVgprLocalWriteAddressesA

    if not kernel["LocalWriteUseSgprB"]:
      if self.combineLocalAddresses:
        self.startVgprLocalWriteAddressesB = self.startVgprLocalReadAddressesA
      else:
        self.startVgprLocalWriteAddressesB = vgprIdx
        vgprIdx += self.numVgprLocalWriteAddressesB

    # BufferLoad:
    # Uses a resource descriptor (SRD) which is stored in 4 SGPRs and thus shared by all work-items.
    # Each work-item also uses  a unique 32-bit offset into vgprGlobalReadOffset.  These offsets are set when
    # the tile is initialized and stay constant through the execution of the kernel.
    # The base address in the SRD is updated when the algorithm moves to a new tile
    # BufferLoad disables the gptGlobalReadAddr used in flat addressing.
    if kernel["BufferLoad"]:
      self.startVgprGlobalReadOffsetA = vgprIdx
      vgprIdx += 1 if kernel["_UseSgprForGRO"] else self.numGlobalReadOffsetsA
      self.startVgprGlobalReadOffsetB = vgprIdx
      vgprIdx += 1 if kernel["_UseSgprForGRO"] else self.numGlobalReadOffsetsB
      # allocate tile offset and unroll offset registers for PK kernel
      if self.useGlobalReadTileVgpr:
        self.startVgprGlobalReadTileOffsetA = vgprIdx
        vgprIdx += tPA["nrt"]
        self.startVgprGlobalReadUnrollOffsetA = vgprIdx
        vgprIdx += self.numGlobalReadOffsetsA
        self.startVgprGlobalReadTileOffsetB = vgprIdx
        vgprIdx += tPB["nrt"]
        self.startVgprGlobalReadUnrollOffsetB = vgprIdx
        vgprIdx += self.numGlobalReadOffsetsB

    else:
      # TODO: alignment hack, figure out a better solution
      vgprIdx = ((vgprIdx+1)//2)*2
      self.startVgprGlobalReadAddressesA = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesA
      self.startVgprGlobalReadAddressesB = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesB

    self.startVgprGlobalReadIncsA = vgprIdx
    vgprIdx += numVgprGlobalReadIncsA
    self.startVgprGlobalReadIncsB = vgprIdx
    vgprIdx += numVgprGlobalReadIncsB
    #-----------

    if self.startVgprG2LA is None:
      # TODO: alignment hack, figure out a better solution
      vgprIdx = ((vgprIdx+1)//2)*2
      self.startVgprG2LA = vgprIdx; vgprIdx += self.numVgprG2LA

    if self.startVgprG2LB is None:
      # TODO: alignment hack, figure out a better solution
      vgprIdx = ((vgprIdx+1)//2)*2
      self.startVgprG2LB = vgprIdx; vgprIdx += self.numVgprG2LB

    # GlobalRead, LocalWrite, LocalRead, G2L can be reclaimed, extend the "lastVgprForReads" value
    self.lastVgprForReads = vgprIdx
    #-----------

    self.startVgprLocalReadAddressesA = vgprIdx
    vgprIdx += self.numVgprLocalReadAddressesA
    if self.combineLocalAddresses:
      self.startVgprLocalReadAddressesB = self.startVgprLocalReadAddressesA
    else:
      self.startVgprLocalReadAddressesB = vgprIdx
      vgprIdx += self.numVgprLocalReadAddressesB

    self.startVgprAddressDbg = vgprIdx
    vgprIdx += numVgprAddressDbg

    # allocate VGPRS for loadC and storeC (dedicated for now)
    self.startVgprG2LC = None
    self.startVgprL2GC = None
    self.GlobalReadOffsetC = None
    self.GlobalWriteOffsetD = None
    # for zgemm + (SCIU or MIAV) case, allocate 4 vgpr for alpha calculation (cannot use tmp vgpr in unroll loop or write batch)
    if kernel["ProblemType"]["DataType"].isDoubleComplex() and kernel["MIArchVgpr"]:
      # need proper alignment
      vgprIdx = ((vgprIdx+2 - 1)//2)*2
      self.startVgprAlphaTmp = vgprIdx
      vgprIdx += 4

    self.startVgprSerial = vgprIdx
    vgprIdx += 1 # for vgpr serial id

    # tmp vgprs
    #minVgprTmp += 4
    #if globalParameters["DebugKernel"]:
    #  minVgprTmp += 2
    #vgprIdx += minVgprTmp
    #print2("%3u vgprs <- %s" % (vgprIdx, self.kernelName) )
    self.startVgprReuse = vgprIdx # for register reuse;

    self.totalVgprs = max(vgprIdx, self.numVgprValuC)
    if self.totalVgprs < kernel["MinVgprNumber"] or self.totalVgprs > kernel["MaxVgprNumber"]:
      raise RuntimeError("Generating asm kernel error: total vgpr: %u not in [%u, %u].\n" % (self.totalVgprs, kernel["MinVgprNumber"], kernel["MaxVgprNumber"]))

    ########################################
    # SGPR Allocation
    ########################################

    ####################################
    # num sgprs: initial kernel state
    self.sgprPool = RegisterPool(0, 's', defaultPreventOverflow=True, printRP=0)
    numSgprAddressD = self.rpga # til end
    numSgprAddressC = self.rpga # til end
    numSgprAddressA = self.rpga # til read offsets
    numSgprAddressB = self.rpga # til read offsets
    # would not less than 1 reg,
    # since even if ComputeType = H, we still pass the arg as a 32-bit (concate two 16-bit)
    numSgprAlpha = max(1,int(self.bpeCinternal/4))
    numSgprBeta  = max(1,int(self.bpeCinternal/4)) if kernel["ProblemType"]["UseBeta"] else 0
    self.numSgprStridesD = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprStridesC = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprStridesA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    self.numSgprStridesB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    if not kernel["ProblemType"]["UseInitialStridesCD"]:
      self.numSgprStridesD -= 1
      self.numSgprStridesC -= 1
    if not kernel["ProblemType"]["UseInitialStridesAB"]:
      self.numSgprStridesA -= 1
      self.numSgprStridesB -= 1
    self.numSgprSizesSum = kernel["ProblemType"]["NumIndicesSummation"]
    self.numSgprSizesFree = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprOffsetD = 1
    self.numSgprOffsetC = 1
    self.numSgprOffsetA = 1
    self.numSgprOffsetB = 1
    self.numActivationTypeArgSize = 0 # Will change to 1 if activationType == All
    self.numActivationArgSize = max(1, int(kernel["ProblemType"]["DestDataType"].numRegisters()))
    self.numactivationArgTotalSize = self.numActivationArgSize * kernel["ProblemType"]["ActivationType"].getAdditionalArgNum()
    self.numSgprAddressDbg = self.rpga if globalParameters["DebugKernel"] else 0

    ####################################
    # num sgprs: global read increments
    if self.globalReadIncsUseVgpr:
      self.numSgprGlobalReadIncsA = 0
      self.numSgprGlobalReadIncsB = 0
    else:
      self.numSgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] * self.rpgo
      self.numSgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] * self.rpgo

    ########################################
    # SGPR Assignment according to AMDGPU-ABI
    ########################################
    self.defineSgpr("KernArgAddress", self.rpga)
    assert(self.sgprs["KernArgAddress"] ==  0) # kernarg is passed to kernel as SGPR0

    if kernel["WorkGroupMapping"]>=0 :
      self.defineSgpr("WorkGroup0", 1)
      self.defineSgpr("WorkGroup1", 1)
    else:
      self.defineSgpr("WorkGroup1", 1)
      self.defineSgpr("WorkGroup0", 1)

    wg=2

    for idx in kernel["ProblemType"]["IndicesBatch"]:
      if not isPackedIndex(kernel,idx):
        self.defineSgpr("WorkGroup%u"%wg, 1)
        wg+=1

    # SGPR above are user SGPR which are set by GPU hardware when the kernel is launched
    self.firstInitSgpr = self.sgprPool.size()

    # To avoid corrupting tmp sgprs that may be used around the assert,
    # reserve some sgprs to save/restore the execmask
    if self.db["EnableAsserts"]:
      self.defineSgpr("SaveExecMask", 2, 2)
    self.asmAssert = Assert(self.laneSGPRCount, kernel["WavefrontSize"])

    self.defineSgpr("GSUSumIdx", 2 if kernel["GlobalSplitU"] > 1 else 0)

    # for packed batches without stride restrictions need to do something different here
    assert sorted(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]) == \
           sorted(set(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]))
    for idxChar in kernel["PackedC0IdxChars"][:-1]:
      if kernel["MagicDivAlg"]==2:
        self.defineSgpr("MagicAbitSize%s"%idxChar, 1)
    for idxChar in kernel["PackedC1IdxChars"][:-1]:
      if kernel["MagicDivAlg"]==2:
        self.defineSgpr("MagicAbitSize%s"%idxChar, 1)

    # product of all packed dims in the 0 or 1 dimensions:
    if len(kernel["PackedC0IndicesX"]) > 1:
      self.defineSgpr("PackedSize0", 1)
    if len(kernel["PackedC1IndicesX"]) > 1:
      self.defineSgpr("PackedSize1", 1)

    # contractions with multiple summations will use multiple LoopCounters, if PSD=0
    for i in range(kernel["ProblemType"]["NumIndicesSummation"]):
      self.defineSgpr(self.loopCounterName(kernel,i), 1)

    self.defineSgpr("OrigLoopCounter", 1)

    if globalParameters["DebugKernel"]:
      self.defineSgpr("AddressDbg", self.numSgprAddressDbg)
      self.defineSgpr("DebugKernelItems", 1)

    if kernel["BufferLoad"]:
       # resource descriptor (SRD) A and B, must be aligned on 4-SGPR boundary
      self.defineSgpr("SrdA", 4, 4)
      self.defineSgpr("SrdB", 4, 4)
    if kernel["BufferStore"]:
      self.defineSgpr("SrdD", 4, 4)
      self.defineSgpr("SrdC", 4, 4)

    ###################################
    # Get kernel argument start here
    self.defineSgpr("Tensor2dSizeA", 2,4)
    # fill empty Sgpr slot caused by Sgpr alignment,
    # because we need following defineSgpr use continuous sgpr
    SgprSlot = []
    currentSize = self.sgprPool.size()
    while (1):
      tempSgpr = self.sgprPool.checkOut(1,"fill empty slot temporarily",preventOverflow=0)
      if tempSgpr >= currentSize:
        self.sgprPool.checkIn(tempSgpr)
        break
      SgprSlot.append(tempSgpr)
    self.defineSgpr("Tensor2dSizeB", 2, 2)
    self.argAddressOffset = 6 * 4 # 8 bytes C, A, B

    self.defineSgpr("AddressD", numSgprAddressD)
    self.defineSgpr("AddressC", numSgprAddressC)
    self.defineSgpr("AddressA", numSgprAddressA)
    self.defineSgpr("AddressB", numSgprAddressB)
    self.defineSgpr("Alpha", numSgprAlpha, numSgprAlpha)
    if kernel["ProblemType"]["UseBeta"]:
      self.defineSgpr("Beta", numSgprBeta, numSgprBeta)
    if ((kernel["ProblemType"]["ActivationType"] != 'none') and (kernel["_GlobalAccumulation"] != 'MultipleBuffer') \
        and kernel["ActivationFused"]):
      for name in kernel["ProblemType"]["ActivationType"].getAdditionalArgStringList():
          self.defineSgpr(name, self.numActivationArgSize, self.numActivationArgSize)
      if kernel["ProblemType"]["ActivationType"] == 'all':
        self.numActivationTypeArgSize = 1
        self.defineSgpr("ActivationType", self.numActivationTypeArgSize)
    self.defineSgpr("StridesD", self.numSgprStridesD)
    self.defineSgpr("StridesC", self.numSgprStridesC)
    self.defineSgpr("StridesA", self.numSgprStridesA)
    self.defineSgpr("StridesB", self.numSgprStridesB)
    self.defineSgpr("SizesFree", self.numSgprSizesFree)
    self.defineSgpr("SizesSum", self.numSgprSizesSum)

    # for packed batches without stride restrictions need to do something different here
    assert sorted(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]) == \
           sorted(set(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]))
    for idxChar in kernel["PackedC0IdxChars"][:-1]:
      self.defineSgpr("MagicNumberSize%s"%idxChar, 1)
      self.defineSgpr("MagicShiftSize%s"%idxChar, 1)
    for idxChar in kernel["PackedC1IdxChars"][:-1]:
      self.defineSgpr("MagicNumberSize%s"%idxChar, 1)
      self.defineSgpr("MagicShiftSize%s"%idxChar, 1)
    self.defineSgpr("OrigStaggerUIter", 1)  # Original stagger register.  Only needed for Persistent
    self.defineSgpr("NumWorkGroups0", 1)
    self.defineSgpr("NumWorkGroups1", 1)

    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    self.lastPostLoopSgpr = self.sgprPool.size()
    self.defineSgpr("NumFullBlocks", 1) # Magic number to use for div by (NumWorkGroups1 % WGM)
    self.defineSgpr("WgmRemainder1", 1) # Magic number to use for div by (NumWorkGroups1 % WGM)
    self.defineSgpr("MagicNumberWgmRemainder1", 1) # Magic number to use for div by (NumWorkGroups1 % WGM)

    self.defineSgpr("OffsetD", self.numSgprOffsetD)
    self.defineSgpr("OffsetC", self.numSgprOffsetC)
    self.defineSgpr("OffsetA", self.numSgprOffsetA)
    self.defineSgpr("OffsetB", self.numSgprOffsetB)

    self.numSgprToLoad = 2 + 2 + numSgprAddressD + numSgprAddressC + numSgprAddressA + numSgprAddressB + numSgprAlpha + \
      (numSgprBeta if kernel["ProblemType"]["UseBeta"] else 0) + self.numSgprStridesD + self.numSgprStridesC + self.numSgprStridesA + \
      self.numSgprStridesB + self.numSgprSizesFree + self.numSgprSizesSum + \
      len(kernel["PackedC0IdxChars"][:-1])*2 + len(kernel["PackedC1IdxChars"][:-1])*2 + \
      1 + \
      2 + \
      3 + \
      self.numSgprOffsetD + self.numSgprOffsetC + self.numSgprOffsetA + self.numSgprOffsetB
    if ((kernel["ProblemType"]["ActivationType"] != 'none') and (kernel["_GlobalAccumulation"] != 'MultipleBuffer') \
        and kernel["ActivationFused"]):
      self.numSgprToLoad += self.numActivationTypeArgSize + self.numactivationArgTotalSize

    self.argOffsetOffset = (self.numSgprToLoad + 2 - (self.numSgprOffsetD + self.numSgprOffsetC + self.numSgprOffsetA + self.numSgprOffsetB)) * 4

    # Get kernel argument end here
    ###################################

    # put unused Sgpr back to SgprPool
    while SgprSlot:
      tempSgpr = SgprSlot.pop(0)
      self.sgprPool.checkIn(tempSgpr)
    if not self.staggerU:
      self.undefineSgpr("OrigStaggerUIter")  # Original stagger register.  Only needed for Persistent

    ########################################
    # Register Pools
    ########################################
    #print "TotalVgprs", self.totalVgprs
    self.vgprPool = RegisterPool(self.totalVgprs, 'v', defaultPreventOverflow=False,
                                 printRP=self.db["PrintRP"])
    #print self.vgprPool.state()
    self.savedVgprPool = None
    self.savedSgprPool = None

    # C regs are not used during initialization so mark them as available -
    # we will claim then just before the start of the unroll loop:
    self.vgprPool.add(self.startVgprValuA, \
        self.lastValuAB - self.startVgprValuA, "ValuAB") # Add as available

    self.vgprPool.add(self.startVgprValuC, \
      self.numVgprValuC, "ValuC-Block") # Add as available
    #print self.vgprPool.state()
    ## accumulator Buffer for storeCinUnroll feature
    self.agprPool = RegisterPool(self.totalAgprs, 'a', defaultPreventOverflow=False, printRP=0)
    # C regs are not used during initialization so mark them as available -
    # we will claim then just before the start of the unroll loop:
    numAccvgprs = self.totalAgprs
    self.agprPool.add(0, numAccvgprs, "ValuC-Block")

    # place any of these gpr inst values into tPA, tPB for later reference
    tPA["globalReadInstruction"] = self.globalReadInstructionA
    tPA["localWriteInstruction"] = self.localWriteInstructionA
    tPA["localReadInstruction"] = self.localReadInstructionA
    tPA["gpr"] = {}

    tPB["globalReadInstruction"] = self.globalReadInstructionB
    tPB["localWriteInstruction"] = self.localWriteInstructionB
    tPB["localReadInstruction"] = self.localReadInstructionB
    tPB["gpr"] = {}

    ########################################
    # reads Per Iteration
    ########################################
    if kernel["EnableMatrixInstruction"]:
      # setting numReadPerVector to 0 for DirectToVgpr makes performance a little worse.
      # so, keep this part unchanged.
      #self.numReadPerVectorA = 0 if kernel["DirectToVgprA"] else tPA["bpe"] * self.lrvwA // int(tPA["localReadInstruction"].blockWidth * 4)
      #self.numReadPerVectorB = 0 if kernel["DirectToVgprB"] else tPB["bpe"] * self.lrvwB // int(tPB["localReadInstruction"].blockWidth * 4)
      self.numReadPerVectorA = tPA["bpe"] * self.lrvwA // int(tPA["localReadInstruction"].blockWidth * 4)
      self.numReadPerVectorB = tPB["bpe"] * self.lrvwB // int(tPB["localReadInstruction"].blockWidth * 4)
      numA = kernel["InnerUnroll"]*(kernel["MIWaveTile"][0] * self.numReadPerVectorA) // tPA["localReadInstruction"].numOffsets
      numB = kernel["InnerUnroll"]*(kernel["MIWaveTile"][1] * self.numReadPerVectorB) // tPB["localReadInstruction"].numOffsets
      # wider localread has 2 mode
      # 1. using larger IU to coalesced localread, only half of local reads in 1 iteration
      # 2. using larger PLR to read more iterations, same number local reads in 1 iteration
      if kernel["InnerUnroll"] >= self.numReadsIterCoalescedA:
        numA //= self.numReadsIterCoalescedA
        if self.allowLRVWforTLUandMI:
          numA //= self.lrvwA
      if kernel["InnerUnroll"] >= self.numReadsIterCoalescedB:
        numB //= self.numReadsIterCoalescedB
        if self.allowLRVWforTLUandMI:
          numB //= self.lrvwB
    else:
      printExit("TensileLite does not support non MFMA.")
    self.numReadsPerIterA = numA
    self.numReadsPerIterB = numB
    self.localReadDoCntA   = 0
    self.localReadDoCntB   = 0

    if kernel["EnableMatrixInstruction"]:
      self.miLatency = kernel["MatrixInstM"] // 2
      miIssueLatency = 2
      # give 1 quad-cycle buffer to prevend bubble from sync
      miLatencyBuffer = 1
      self.miLatencyLeft = max(self.miLatency - miLatencyBuffer - miIssueLatency,0)

    # pre-determine labels in order
    unrollChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
    self.labels = LabelManager()
    # shift vectors determined later

    canCheckValueC = (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and \
                      kernel["ProblemType"]["HighPrecisionAccumulate"]
    canCheckValueC = canCheckValueC or kernel["ProblemType"]["DataType"].isSingle()
    canCheckValueC = canCheckValueC or (kernel["ProblemType"]["DataType"].isInt8() and kernel["ProblemType"]["HighPrecisionAccumulate"])
    assert not self.db["CheckValueC"] or canCheckValueC

    if self.db["InitLds"] : print ("\n***WARNING: InitLds enabled, may impact performance\n")
    if self.db["InitSgpr"] : print ("\n***WARNING: InitSgpr enabled, may impact performance\n")
    if self.db["InitVgpr"] : print ("\n***WARNING: InitVgpr enabled, may impact performance\n")
    if self.db["ConservativeWaitCnt"] : print ("\n***WARNING: ConservativeWaitCnt enabled, may impact performance\n")
    if self.do["KeepDirectToLdsAlloc"] : print ("\n***WARNING: KeepDirectToLdsAlloc enabled, may impact performance\n")
    if self.db["CheckValue1A"] : print ("\n***WARNING: CheckValue1A enabled, may impact performance\n")
    if self.db["CheckValue1B"] : print ("\n***WARNING: CheckValue1B enabled, may impact performance\n")
    if self.db["CheckValueC"] : print ("\n***WARNING: CheckValueC enabled, may impact performance\n")
    if self.db["ForceExpectedValue"] : print ("\n***WARNING: ForceExpectedValue enabled, may impact functionality\n")
    if self.db["ForceVSerial"] : print ("\n***WARNING: ForceVSerial enabled, will impact functionality\n")
    if self.db["ForceInputValueA"] : print ("\n***WARNING: ForceInputValueA enabled, may impact functionality\n")
    if self.db["ForceInputValueB"] : print ("\n***WARNING: ForceInputValueB enabled, may impact functionality\n")
    if self.db["CheckStoreC"] >=0  : print ("\n***WARNING: CheckStoreC enabled, may impact performance\n")
    if self.db["ForceEdgeStores"] : print ("\n***WARNING: ForceEdgeStores enabled, may impact performance\n")
    if self.db["AssertNoEdge"] : print ("\n***WARNING: AssertNoEdge enabled, may impact functionality and performance\n")
    if self.db["PrintRP"] : print ("\n***WARNING: PrintRP enabled, may generate verbose output\n")
    if kernel["CheckTensorDimAsserts"] : print ("\n***WARNING: CheckTensorDimAsserts enabled, may impact performance\n")
    if kernel["CheckDimOverflow"] : print ("\n***WARNING: CheckDimOverflow enabled, may impact performance\n")

  ##############################################################################
  def functionSignature(self, kernel ):
    """
    Function Signature
    called after rest of code
    """
    module = Code.Module("functionSignature")

    signature = Component.Signature.find(self)
    module.addCode(signature(self))

    instMacros = InstMacros(version=self.version,isa=self.kernel["ISA"], macInst=self.kernel["MACInstruction"],
                                 asmCaps=self.asmCaps, archCaps=self.archCaps, asmBugs=self.AsmBugs,
                                 wavefrontSize=self.kernel["WavefrontSize"], vcc=self.vcc)
    module.addCode(instMacros.defineFeatureMacros())
    module.addCode(instMacros.defineMagicDivMacros(kernel["MagicDivAlg"]))

    ########################################
    # VGPR Macros
    ########################################
    module.addComment2("VGPR Assignments")
    module.addComment0("ValuC range: [%u-%u), %s"%(self.startVgprValuC, self.startVgprValuC+self.numVgprValuC, \
                           "serializedStore enabled" if self.serializedStore else ""))
    module.addCode(macroRegister("vgprValuC", self.startVgprValuC))

    module.addComment0("ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx")
    # PLR index: from X0 to X<LoopIters-1> (at most) -> VGPRs will be duplicated LoopIters times (at most)
    # eg, if LoopIters = 4, there would be at most 4*VGPRs
    # PLR = kernel["PrefetchLocalRead"] if kernel["PrefetchLocalRead"] < kernel["LoopIters"] else kernel["LoopIters"] - 1
    PLR = min(kernel["PrefetchLocalRead"], kernel["LoopIters"]-1)
    numBi = PLR+1
    # double the number of VgprValue if self.vgprValuDouble is true
    if self.vgprValuDouble:
      numBi *= 2
    ri = 0
    if self.numVgprValuA > 0: # Do not generate vgprValuA if numVgprValuA is 0
      for bi in range(0,numBi): # buffer indices
        for iui in range(0, kernel["InnerUnroll"]):
          module.addCode(macroRegister("vgprValuA_X%u_I%u"%(bi,iui), self.startVgprValuA+ri))
          ri += self.numVgprValuAPerBlock
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
        module.addCode(macroRegister("vgprG2LA", self.startVgprG2LA))
        if kernel["DirectToVgprA"]:
          # additional definition G2LA0, G2LA1 for swapping register sets
          module.addCode(macroRegister("vgprG2LA0", self.startVgprG2LA))
          module.addCode(macroRegister("vgprG2LA1", self.startVgprG2LA + self.numVgprG2LA//2))

    ri = 0
    if self.numVgprValuB > 0: # Do not generate vgprValuB if numVgprValuB is 0
      for bi in range(0,numBi): # buffer indices
        for iui in range(0, kernel["InnerUnroll"]):
          module.addCode(macroRegister("vgprValuB_X%u_I%u"%(bi,iui), self.startVgprValuB+ri))
          ri += self.numVgprValuBPerBlock
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
        module.addCode(macroRegister("vgprG2LB", self.startVgprG2LB))
        if kernel["DirectToVgprB"]:
          # additional definition G2LB0, G2LB1 for swapping register sets
          module.addCode(macroRegister("vgprG2LB0", self.startVgprG2LB))
          module.addCode(macroRegister("vgprG2LB1", self.startVgprG2LB + self.numVgprG2LB//2))
    if not kernel["LocalWriteUseSgprA"] and self.numVgprLocalWriteAddressesA > 0:
      module.addCode(macroRegister("vgprLocalWriteAddrA", \
          self.startVgprLocalWriteAddressesA))
      if self.numVgprLocalWriteAddressesA > 1:
        module.addCode(macroRegister("vgprLocalWriteAddrOverhangA", \
            self.startVgprLocalWriteAddressesA+1))
    if not kernel["LocalWriteUseSgprB"] and self.numVgprLocalWriteAddressesB > 0:
      module.addCode(macroRegister("vgprLocalWriteAddrB", \
          self.startVgprLocalWriteAddressesB))
      if self.numVgprLocalWriteAddressesB > 1:
        module.addCode(macroRegister("vgprLocalWriteAddrOverhangB", \
            self.startVgprLocalWriteAddressesB+1))
    if kernel["BufferLoad"]:
      module.addCode(macroRegister("vgprGlobalReadOffsetA", \
          self.startVgprGlobalReadOffsetA))
      module.addCode(macroRegister("vgprGlobalReadOffsetB", \
          self.startVgprGlobalReadOffsetB))
      if self.useGlobalReadTileVgpr:
        module.addCode(macroRegister("vgprGlobalReadTileOffsetA", \
            self.startVgprGlobalReadTileOffsetA))
        module.addCode(macroRegister("vgprGlobalReadUnrollOffsetA", \
            self.startVgprGlobalReadUnrollOffsetA))
        module.addCode(macroRegister("vgprGlobalReadTileOffsetB", \
            self.startVgprGlobalReadTileOffsetB))
        module.addCode(macroRegister("vgprGlobalReadUnrollOffsetB", \
            self.startVgprGlobalReadUnrollOffsetB))
    else:
      module.addCode(macroRegister("vgprGlobalReadAddrA", \
          self.startVgprGlobalReadAddressesA))
      module.addCode(macroRegister("vgprGlobalReadAddrB", \
          self.startVgprGlobalReadAddressesB))

    if self.globalReadIncsUseVgpr:
      module.addCode(macroRegister("vgprGlobalReadIncsA", \
          self.startVgprGlobalReadIncsA))
      module.addCode(macroRegister("vgprGlobalReadIncsB", \
          self.startVgprGlobalReadIncsB))
    if self.numVgprLocalReadAddressesA > 0:
      module.addCode(macroRegister("vgprLocalReadAddrA", \
          self.startVgprLocalReadAddressesA))
    if self.numVgprLocalReadAddressesB > 0:
      module.addCode(macroRegister("vgprLocalReadAddrB", \
          self.startVgprLocalReadAddressesB))

    if kernel["ProblemType"]["DataType"].isDoubleComplex() and kernel["MIArchVgpr"]:
      module.addCode(macroRegister("vgprAlphaTmp", \
          self.startVgprAlphaTmp))

    # Serial is always the last register in the pool so the store
    # code doesn't have to deal with fragmentation
    self.vgprstartSerial = self.vgprPool.size()-1
    module.addCode(macroRegister("vgprSerial", self.startVgprSerial))

    if globalParameters["DebugKernel"]:
      module.addCode(macroRegister("vgprAddressDbg", \
          self.startVgprAddressDbg))
    #module.addComment0("Occu: %u waves/simd" % self.numWavesPerSimd )
    module.addComment0("Num VGPR=%u"%self.vgprPool.size())
    module.addComment0("Num AccVGPR=%u"%self.agprPool.size())

    ########################################
    # SGPR Macros
    ########################################
    module.addComment2("SGPR Assignments")

    # Emit declarations for all sgprs allocated with defineSgpr
    # in the order they were declared
    for skey in self.sgprs:
      module.addCode(macroRegister("sgpr"+skey, self.sgprs[skey]))
    module.addComment0("max SGPR=%u"%self.sgprPool.size())

    module.addSpaceLine()
    module.addComment0("Size Assignments")
    problemType = kernel["ProblemType"]
    for idx in range(max(problemType["IndexAssignmentsA"] + problemType["IndexAssignmentsB"])+1):
      idxChar= globalParameters["IndexChars"][idx]
      if idx in problemType["IndicesFree"] or idx in problemType["IndicesBatch"]:
        idxType="Free"
      elif idx in problemType["IndicesSummation"]:
        idxType="Sum"
        idx = idx - problemType["NumIndicesC"]
      else:
        raise ValueError("unexpected index type in size assignments")

      module.addCode(macroRegister("sgprSize%s"%(idxChar), \
                  "sgprSizes%s+%u"%(idxType, idx)))

    module.addSpaceLine()
    module.addComment0("Stride Assignments")
    for tc in ('D','C'):
      for idx in range(0, problemType["NumIndicesC"]):
        i = idx
        idxChar= self.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesCD"]:
          module.addCode(macroRegister("constStride%s%s"%(tc,idxChar), 1))
        else:
          if not kernel["ProblemType"]["UseInitialStridesCD"]:
            i = i-1
          module.addCode(macroRegister("sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s+%u"%(tc, i)))

    for tc in ('A','B'):
      for i, idx in enumerate(problemType["IndexAssignments%s"%tc]):
        idxChar= self.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesAB"]:
          module.addCode(macroRegister("constStride%s%s"%(tc,idxChar), 1))
        else:
          if not kernel["ProblemType"]["UseInitialStridesAB"]:
            i = i-1
          module.addCode(macroRegister("sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s+%u"%(tc, i)))

    module.addSpaceLine()
    module.addCode(macroRegister("MT0", kernel["MacroTile0"]))
    module.addCode(macroRegister("MT1", kernel["MacroTile1"]))
    module.addCode(macroRegister("DepthU", kernel["DepthU"]))
    module.addCode(macroRegister("GSU", kernel["GlobalSplitU"]))
    module.addCode(macroRegister("BpeA", self.tPA["bpe"]))
    module.addCode(macroRegister("BpeALog2", log2(self.tPA["bpe"])))
    module.addCode(macroRegister("BpeB", self.tPB["bpe"]))
    module.addCode(macroRegister("BpeBLog2", log2(self.tPB["bpe"])))
    module.addComment0("Number of elements to shift-left SRD")
    module.addCode(macroRegister("SrdShiftLeftA", self.srdShiftLeft['A']))
    module.addCode(macroRegister("SrdShiftLeftB", self.srdShiftLeft['B']))

    if kernel["BufferLoad"] or kernel["BufferStore"]:
      module.addComment0("2GB limit - set offsets to -1 to exceed this and clamp")
      module.addCode(macroRegister("BufferLimit", "0xffffffff"))
      #TODO-64 : This is max 32-bit negative value, the tail loop
      # does incrementally step through the GRO and increment GRO
      # which are initialized with this value
      module.addCode(macroRegister("BufferOOB", "0x80000000"))

      srdUpperValue = Code.SrdUpperValue(self.version)
      module.addComment2("Bits 127:96 of SRD.\n" + srdUpperValue.desc())
      module.addCode(macroRegister("Srd127_96", str(srdUpperValue)))

    ########################################
    # Global Offsets
    ########################################
    # justOffset32 means we should only write the 32-bit offset
    # This is used in Buffer addressing modes.
    # Flat addressing modes expect the GLOBAL_OFFSET to initialize a full 64-bit address
    for (tc, indices, justOffset32, tP) in [ \
        ("C", list(range(0, kernel["ProblemType"]["NumIndicesC"])), kernel["BufferStore"], None), \
        ("A", kernel["ProblemType"]["IndexAssignmentsA"], kernel["BufferLoad"], self.tPA), \
        ("B", kernel["ProblemType"]["IndexAssignmentsB"], kernel["BufferLoad"], self.tPB) ]:

      # BufferStore does not use this macro so don't generate it:
      if tc == "C" and kernel["BufferStore"]:
        continue

      module.addComment1("Global Offset %s"%tc)
      numDim = len(indices)
      idxChars = []
      for i in indices:
        idxChars.append(self.indexChars[i])

      # macro declaration
      calcDims = [] # dimensions which are participating in the address calc (ignores other summation)
      mirrorSumDims = []
      macroArgs = []
      for i in range(0, numDim):
        if tc == 'C':
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesCD"]
          idxChar = self.indexChars[i]
        else:
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesAB"]
          idxChar = self.indexChars[tP['ia'][i]]

        # tile index or unroll vgpr or summation
        # other summation (other than unroll) are included in the GLOBAL_OFFSET macro but not used in address calc
        if     tc in ('A','C') and indices[i] == kernel["ProblemType"]["Index0"] \
            or tc in ('B','C') and indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          macroArgs.append("vgprOffset%s:req" % idxChars[i])
          calcDims.append(i)
        elif indices[i] in kernel["ProblemType"]["IndicesSummation"]:
          # other summation index (not unroll)
          if tc in ('A', 'B') and indices[i] in kernel["ProblemType"]["MirrorDims%s" % tc]:
            mirrorSumDims.append(i)
          continue
        else:
          # other batch or free index
          if isPackedIndex(kernel, indices[i]):
            calcDims.append(i)
            macroArgs.append("vgprOffset%s:req" % idxChars[i])
          elif not justOffset32: # buffer/justOffset32 scalars are included in SRD not the offset, so skip here
            calcDims.append(i)
            macroArgs.append("sgprOffset%s:req" % idxChars[i])
      macro = Code.Macro("GLOBAL_OFFSET_%s" % tc, "vgprAddr:req", *macroArgs, "vgprTmp:req")

      # Each index may be skipped, scaled by stride, or unscaled
      # If destLo is unset, no accumulation is necessary.

      # if the first index (i==0) is unscaled (UseInitialStrides),
      # it can be combined at the next update or moved at end
      # (if there is no next update)

      pendingOffset = None # offset pending for accumulation
      offsetIsVgpr = False # True if the source is VGPR ; False if SGPR
      destLo = None

      # true for first addr calc. In this case, we can directly write addr
      # rather than accumulating through a tmp
      writeDirectToAddr = 1

      # mirror other summation indices
      for i in mirrorSumDims:
        if writeDirectToAddr:
          dest = "v[\\vgprAddr+0]"
          needAdd = 0 # don't need add since writing address directly.
          writeDirectToAddr = 0
        else:
          dest = "v[\\vgprTmp+0]"
          needAdd = 1
        macro.addInst("_v_sub_u32", \
                dest,
                sgpr("Size%s"%globalParameters["IndexChars"][indices[i]]), \
                "1", \
                "mirror %s%s 1"%(tc, globalParameters["IndexChars"][indices[i]]))
        macro.addInst("v_mul_lo_u32", \
                dest,
                dest, \
                self.strideRef(tc, indices[i]), \
                "mirror %s%s 2"%(tc, globalParameters["IndexChars"][indices[i]]))

        if needAdd:
          writeDirectToAddr = 0 # safety net, once we write address can't directly overwrite it later
          destLo = "v[\\vgprAddr+0]"
          destHi = "v[\\vgprAddr+1]"

          srcLo = pendingOffset if pendingOffset else destLo
          srcHi = 0 if pendingOffset else destHi
          macro.addInst("_v_add_co_u32", \
            destLo, \
            self.vcc, \
            srcLo, \
            "v[\\vgprTmp+0]", \
            "accumulate %s lower"%idxChar)

      for i in calcDims:
        # should have eliminated these above
        idx = indices[i]
        isMirrorIdx = tc in ('A', 'B') and idx in kernel["ProblemType"]["MirrorDims%s" % tc]
        assert not (idx in kernel["ProblemType"]["IndicesSummation"] and idx != kernel["ProblemType"]["IndexUnroll"])

        if indices[i] == kernel["ProblemType"]["Index0"] \
            or indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          offsetIsVgpr = True
        # other c index sgpr (free or batch)
        elif indices[i] < kernel["ProblemType"]["NumIndicesC"]:
          if isPackedIndex(kernel, indices[i]):
            offsetIsVgpr = True
          else:
            offsetIsVgpr = False
        else:
          assert(0) # no other type allowed

        # offset is VGPR or SGPR string to use for the offset
        if offsetIsVgpr:
          offset = "v[\\vgprOffset%s]" % idxChars[i]
        else:
          offset = "s[\\sgprOffset%s]" % idxChars[i]

        # macro.addComment0("dim%s pendingOffset=%s offset=%s offsetIsVgpr=%s" \
        #    % (self.indexChars[indices[i]], pendingOffset, offset, offsetIsVgpr))

        needAdd = 0
        # should be indices[i]??
        if i==0 and not useInitialStrides:
          # slide into next address calc - can do addr = pendingOffset + nextAddrCalc
          pendingOffset = offset
          writeDirectToAddr = 0
        else:
          # tile index or unroll vgpr
          if offsetIsVgpr:
            if writeDirectToAddr:
              destLo = "v[\\vgprAddr+0]"
              destHi = "v[\\vgprAddr+1]"
              needAdd = 0 # don't need add since writing address directly.
              writeDirectToAddr = 0
            else:
              destLo = "v[\\vgprTmp+0]"
              destHi = "v[\\vgprTmp+1]"
              needAdd = 1
            if isMirrorIdx:
              macro.addInst("_v_sub_i32", \
                "v[\\vgprTmp+0]",
                sgpr("Size%s"%globalParameters["IndexChars"][idx]), \
                offset, \
                "mirror %s%s 1"%(tc, globalParameters["IndexChars"][indices[i]]))
              macro.addInst("_v_sub_i32", \
                "v[\\vgprTmp+0]",
                "v[\\vgprTmp+0]", \
                "1", \
                "mirror %s%s 2"%(tc, globalParameters["IndexChars"][indices[i]]))
              offset = "v[\\vgprTmp+0]"

            # offset * stride
            macro.addInst("v_mul_lo_u32", \
                destLo,
                self.strideRef(tc, indices[i]), \
                offset, \
                "mul d%u lower"%i)
            if not justOffset32:
              macro.addInst("v_mul_hi_u32", \
                  destHi,
                  self.strideRef(tc, indices[i]), \
                  offset, \
                  "mul d%u upper"%i)
          else: # offset is SGPR:
            assert not isMirrorIdx
            if not justOffset32:
              # buffer mode (aka justOffset32) does scalars into SRD not offset
              macro.addInst("v_mov_b32", \
                  "v[\\vgprTmp+2]", \
                  "s[\\sgprOffset%s]"%idxChars[i], \
                  "sgprOffset -> vgprTmp+2")
              # offset * stride
              macro.addInst("v_mul_lo_u32", \
                  "v[\\vgprTmp+0]", \
                  self.strideRef(tc, indices[i]), \
                  "v[\\vgprTmp+2]",  \
                  "other stride mul d%u lower"%i)
              macro.addInst("v_mul_hi_u32", \
                  "v[\\vgprTmp+1]", \
                  self.strideRef(tc, indices[i]), \
                  "v[\\vgprTmp+2]",  \
                  "mul d%u upper"%i)
              needAdd = 1

        if needAdd:
          writeDirectToAddr = 0 # safety net, once we write address can't directly overwrite it later
          destLo = "v[\\vgprAddr+0]"
          destHi = "v[\\vgprAddr+1]"
          # addr += offset * stride (lo) : accumulate just-computed address term into addr

          srcLo = pendingOffset if pendingOffset else destLo
          srcHi = 0 if pendingOffset else destHi
          macro.addInst("_v_add_co_u32", \
            destLo, \
            self.vcc, \
            srcLo, \
            "v[\\vgprTmp+0]", \
            "accumulate %s lower"%idxChar)

          # addr += offset * stride (hi)
          if not justOffset32:
            macro.addInst("_v_addc_co_u32", \
                "v[\\vgprAddr+1]", \
                self.vcc, \
                "v[\\vgprTmp+1]",  \
                srcHi, \
                self.vcc, \
                "accumulate %s upper"%idxChar)
          pendingOffset = None

      # pendingOffset but never got a chance to apply it,
      # need to just add an explicit move or add:
      # this can happen for small-order tensors
      if pendingOffset != None:
        destLo = "v[\\vgprAddr+0]"
        if writeDirectToAddr:
          macro.addInst("v_mov_b32", destLo, offset, "setup d0 lower")
          if not justOffset32:
            macro.addInst("v_mov_b32", "v[\\vgprAddr+1]", hex(0), "d0 upper")
        else:
          macro.addInst("_v_add_co_u32", \
            destLo, \
            self.vcc, \
            destLo, \
            pendingOffset, \
            "accumulate final pendingOffset")


      if tP != None and kernel["BufferLoad"] and self.srdShiftLeft[tc]:
        macro.addInst("_v_add_u32", \
            "v[\\vgprAddr+0]", \
            hex(self.srdShiftLeft[tc]), \
            "v[\\vgprAddr+0]", \
            "add prepad for pointer shift")

      # addr *= bytes/element
      if justOffset32:
        macro.addCode(staticMultiply("v[\\vgprAddr+0]", "v[\\vgprAddr+0]", self.bpeAB, None, "offset *= bytes/element"))
      else:
        macro.addInst("v_lshlrev_b64", \
            "v[\\vgprAddr+0:\\vgprAddr+1]", \
            hex(log2(self.bpeAB)), \
            "v[\\vgprAddr+0:\\vgprAddr+1]", \
            "offset *= bytes/element")
      module.addCode(macro)

    module.addCode(instMacros.defineDynamicScalarDivMacros())

    if self.overflowedResources:
      if self.overflowedResources == 1:
        msg = "too many vgprs"
      elif self.overflowedResources == 2:
        msg = "too many sgprs"
      elif self.overflowedResources == 3:
        msg = "half store requires at least two elements per batch"
      elif self.overflowedResources == 4:
        msg = "Occupancy limit"
      elif self.overflowedResources == 5:
        msg = "reading and writing LDS at same time require 2 LDS buffer"
      elif self.overflowedResources == 6:
        msg = "SIA2 better with occupancy 2"
      else:
        msg = "unknown"

      if globalParameters["PrintSolutionRejectionReason"]:
        printWarning("%s overflowed resources.  errorCode=%d, msg=\"%s\", vgprs=%u, sgprs=%u" \
          % (self.kernelName, self.overflowedResources, msg, \
          self.vgprPool.size(), self.sgprPool.size()))
      module.addInst("s_endpgm", "overflowed resources")
      module.addInst(".if", "0", "")

    return module

  ##############################################################################
  # getKernArg
  # Write an argument to specified SGPR and move the kernArgOffset
  # if writeSgpr==0, just move the kernArgOffset - this is used to skip
  # unused parms
  ##############################################################################
  def getKernArg(self, parmName, writeSgpr=1):
    item = None
    size = 1*4
    if writeSgpr:
      item = Code.Inst("_s_load_b32", sgpr(parmName), \
          sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
    else:
      item = Code.TextBlock("Move offset by %u\n" % size)
    self.kernArgOffset += size
    return item

  ##############################################################################
  # code phrase for load batched address from array of buffer pointer
  ##############################################################################
  def loadBatchedAddress(self, kernel, Batch, tmpSgpr):
    laneSC = self.laneSGPRCount
    module = Code.Module("loadBatchedAddress %s" % Batch)
    module.addSpaceLine()

    # handle Batch C/D
    if not kernel["_GlobalAccumulation"]:
      for idx in kernel["ProblemType"]["IndicesBatch"]:
        if not isPackedIndex(kernel,idx):
          module.addInst("s_mul_i32", sgpr(tmpSgpr), sgpr(Batch), 0x8, "offset of global buffer address")
          module.addInst("_s_load_b64", sgpr("AddressD", 2), sgpr("AddressD",2), sgpr(tmpSgpr), "load global buffer D address")

      endCheckLabel = Code.Label(self.labels.getName(f"label_skip_c_buffer_deref_{Batch}"), "")
      module.addCode(sBranchIfZero("Beta", kernel["ProblemType"]["ComputeDataType"], tmpSgpr, laneSC, endCheckLabel, \
                     kernel['WavefrontSize'], self.vcc))

      for idx in kernel["ProblemType"]["IndicesBatch"]:
        if not isPackedIndex(kernel,idx):
          module.addInst("s_mul_i32", sgpr(tmpSgpr), sgpr(Batch), 0x8, "offset of global buffer address")
          module.addInst("_s_load_b64", sgpr("AddressC", 2), sgpr("AddressC",2), sgpr(tmpSgpr), "load global buffer C address")

      module.addCode(endCheckLabel)

    #handle Batch A/B
    endCheckLabel = Code.Label(self.labels.getName(f"label_skip_ab_buffer_deref_{Batch}"), "")
    module.addInst("s_mov_b32", sgpr(tmpSgpr), hex(1), "check summation size")
    for i in range(0, self.numSgprSizesSum):
      module.addInst("s_mul_i32", sgpr(tmpSgpr), sgpr("SizesSum+%u"%(i)), sgpr(tmpSgpr), "check summation size")
    module.addInst("s_cmp_eq_u32", sgpr(tmpSgpr), hex(0), "skip buffer deref is size of summation is 0")
    module.addInst("s_cbranch_scc1", endCheckLabel.getLabelName(), "skip buffer deref is size of summation is 0")

    module.addCode(sBranchIfZero("Alpha", kernel["ProblemType"]["ComputeDataType"], tmpSgpr, laneSC, endCheckLabel, \
                                 kernel['WavefrontSize'], self.vcc))

    module.addInst("s_mul_i32", sgpr(tmpSgpr), sgpr(Batch), 0x8, "offset of global buffer address")
    for idx in kernel["ProblemType"]["IndicesBatch"]:
      if not isPackedIndex(kernel,idx):
        module.addInst("_s_load_b64", sgpr("AddressA", 2), sgpr("AddressA",2), sgpr(tmpSgpr), "load global buffer A address")
        module.addInst("_s_load_b64", sgpr("AddressB", 2), sgpr("AddressB",2), sgpr(tmpSgpr), "load global buffer B address")

    module.addCode(endCheckLabel)

    return module

  ##############################################################################
  def allocateResources(self, kernel):
    module = Code.Module("allocateResources")

    if kernel["StorePriorityOpt"]:
      module.addInst("s_setprio 3", "optimization store")

    if self.do["NullKernel"]:
      module.addInst("s_endpgm", "Skip the whole kernel")

    if self.do["PreLoop"]:
      if self.db["InitSgpr"] & 0x1:
        module.addComment1("Init SGPRs")
        for i in range(self.firstInitSgpr, self.sgprPool.size()):
          module.addInst("s_mov_b32", sgpr(i), hex(self.initSgprValue), "InitSgpr&0x1")
        module.addSpaceLine()

      if self.db["InitVgpr"] & 0x1:
        module.addComment1("Init VGPRs")
        for i in range(1, self.totalVgprs):
          module.addInst("v_mov_b32", vgpr(i), hex(self.initVgprValue), "InitVgpr&0x1")
        module.addSpaceLine()

      # set m0
      module.addInst("s_mov_b32", mgpr(0), hex(kernel["LdsNumElements"] \
          * self.bpeAB), "LDS clamp at %u bytes" \
          %(kernel["LdsNumElements"] * self.bpeAB) )

      # set Serial id vgpr
      module.addInst("v_mov_b32", vgpr("Serial"), vgpr(0), "thread serial id")

      if self.kernel["WavefrontSize"] == 32:
        module.addInst("s_mov_b32", "vcc_hi", "0", "Ensure hi bits are zero")

      ########################################
      # load kernel args
      module.addComment1("Load Kernel Args")
      self.kernArgOffset = 0
      if globalParameters["DebugKernel"]:
        module.addCode(self.getKernArg("AddressDbg"))
        module.addCode(self.getKernArg("AddressDbg+1"))

      self.getKernArg("Tensor2dSizeC+0",0)
      self.getKernArg("Tensor2dSizeC+1",0)

      load = self.numSgprToLoad
      sgprStart = self.sgprs["Tensor2dSizeA"]
      while load > 0:
        if load >= 16:
          load -= 16
          module.addInst("_s_load_b512", sgpr(sgprStart,16), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 16
          self.kernArgOffset += 16 * 4
          continue
        if load >= 8:
          load -= 8
          module.addInst("_s_load_b256", sgpr(sgprStart,8), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 8
          self.kernArgOffset += 8 * 4
          continue
        if load >= 4:
          load -= 4
          module.addInst("_s_load_b128", sgpr(sgprStart,4), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 4
          self.kernArgOffset += 4 * 4
          continue
        if load >= 2:
          load -= 2
          module.addInst("_s_load_b64", sgpr(sgprStart,2), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 2
          self.kernArgOffset += 2 * 4
          continue
        if load >= 1:
          load -= 1
          module.addInst("_s_load_b32", sgpr(sgprStart), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 1
          self.kernArgOffset += 1 * 4
          continue
      # currently align sgpr to kernel argument memory, and use s_load_bxxx to load argument as large as possible in one instruction
      # however, in order to match sgpr to kernel argument memory, some unnecessarily sgpr will also be defined, and caused wasting of sgpr.
      # TODO: more efficient way is to organize both sgpr and kernel argument memory in API

      if kernel.enabledSetPrioSplitLDS:
        module.addInst("s_setprio", "1", "prioritize init code so as to issue load sooner")
      module.addInst("s_waitcnt", "lgkmcnt(0)", "wait for %u bytes of kern args" % self.kernArgOffset )

      if not kernel["ProblemType"]["StridedBatched"]:
        tmpSgpr = self.getTmpSgpr(self.laneSGPRCount).idx()
        module.addCode(self.loadBatchedAddress(kernel, "WorkGroup2", tmpSgpr))
        module.addInst("s_waitcnt", "lgkmcnt(0)", "wait global buffer address ready")
    else:
      module.addCode(".if", "0")

    # add offset to buffer
    if not kernel["_GlobalAccumulation"]:
      module.addInst("s_lshl_b32", sgpr("OffsetD"), sgpr("OffsetD"), hex(log2(self.bpeCexternal)), "elements offset to bytes offset")
      module.addInst("s_add_u32",  sgpr("AddressD+0"), sgpr("AddressD+0"), sgpr("OffsetD"), "add offset to buffer address")
      module.addInst("s_addc_u32", sgpr("AddressD+1"), sgpr("AddressD+1"), 0, "add offset to buffer address")

      module.addInst("s_lshl_b32", sgpr("OffsetC"), sgpr("OffsetC"), hex(log2(self.bpeCexternal)), "elements offset to bytes offset")
      module.addInst("s_add_u32",  sgpr("AddressC+0"), sgpr("AddressC+0"), sgpr("OffsetC"), "add offset to buffer address")
      module.addInst("s_addc_u32", sgpr("AddressC+1"), sgpr("AddressC+1"), 0, "add offset to buffer address")

    module.addInst("s_lshl_b32", sgpr("OffsetA"), sgpr("OffsetA"), hex(log2(self.bpeAB)), "elements offset to bytes offset")
    module.addInst("s_add_u32",  sgpr("AddressA+0"), sgpr("AddressA+0"), sgpr("OffsetA"), "add offset to buffer address")
    module.addInst("s_addc_u32", sgpr("AddressA+1"), sgpr("AddressA+1"), 0, "add offset to buffer address")

    module.addInst("s_lshl_b32", sgpr("OffsetB"), sgpr("OffsetB"), hex(log2(self.bpeAB)), "elements offset to bytes offset")
    module.addInst("s_add_u32",  sgpr("AddressB+0"), sgpr("AddressB+0"), sgpr("OffsetB"), "add offset to buffer address")
    module.addInst("s_addc_u32", sgpr("AddressB+1"), sgpr("AddressB+1"), 0, "add offset to buffer address")

    # self.groOffsetInMacroTile == 1 case, subtract pre-pad here
    if self.groOffsetInMacroTile:
      prePad = self.srdShiftLeft["A"] * self.tPA["bpe"] # leave room in case we have to pointer shift
      module.addInst("s_sub_u32",  sgpr("AddressA+0"), sgpr("AddressA+0"), prePad, "pre-pad to make room for possible pointer shift")
      module.addInst("s_subb_u32",  sgpr("AddressA+1"), sgpr("AddressA+1"), 0, "pre-pad to make room for possible pointer shift")
      prePad = self.srdShiftLeft["B"] * self.tPB["bpe"] # leave room in case we have to pointer shift
      module.addInst("s_sub_u32",  sgpr("AddressB+0"), sgpr("AddressB+0"), prePad, "pre-pad to make room for possible pointer shift")
      module.addInst("s_subb_u32",  sgpr("AddressB+1"), sgpr("AddressB+1"), 0, "pre-pad to make room for possible pointer shift")

    # undefine Offset sgpr
    module.addSpaceLine()
    module.addCode(self.undefineSgpr("OffsetD"))
    module.addCode(self.undefineSgpr("OffsetC"))
    module.addCode(self.undefineSgpr("OffsetA"))
    module.addCode(self.undefineSgpr("OffsetB"))

    self.defineVariableSgprs(kernel)

    # Check alpha == 0, is done before kernel body
    # so if alpha/beta=Half, they haven't been converted to f32
    # This means we can use ComputeDataType as AlphaType (even <h,h,h,h,"h,h"> +"HPA")
    if self.do["ApplyAlpha"]:

      module.addComment1("Short circuit condition if Alpha == 0, then sumDims=0")
      endCheckLabel = Code.Label("AlphaNonZero", "")
      if kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
        module.addInst("v_cmp_eq_f64", self.vcc, sgpr("Alpha", 2), 0.0, "Alpha.real == 0.0 ?")
        module.addInst("s_cbranch_vccz", endCheckLabel.getLabelName(), "branch if Alpha.real != 0")
        module.addInst("v_cmp_eq_f64", self.vcc, sgpr("Alpha+2", 2), 0.0, "Alpha.imag == 0.0 ?")
        module.addInst("s_cbranch_vccz", endCheckLabel.getLabelName(), "branch if Alpha.imag != 0")

      elif kernel["ProblemType"]["ComputeDataType"].isDouble():
        module.addInst("v_cmp_eq_f64", self.vcc, sgpr("Alpha", 2), 0.0, "Alpha == 0.0 ?")
        module.addInst("s_cbranch_vccz", endCheckLabel.getLabelName(), "branch if Alpha != 0")

      elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
        module.addInst("v_cmp_eq_f32", self.vcc, sgpr("Alpha"), 0.0, "Alpha.real == 0.0f ?")
        module.addInst("s_cbranch_vccz", endCheckLabel.getLabelName(), "branch if Alpha.real != 0")
        module.addInst("v_cmp_eq_f32", self.vcc, sgpr("Alpha+1"), 0.0, "Alpha.imag == 0.0f ?")
        module.addInst("s_cbranch_vccz", endCheckLabel.getLabelName(), "branch if Alpha.imag != 0")

      # AlphaType is f32 or two-concated-f16, or two-concated-bf16(not support)
      elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
           kernel["ProblemType"]["ComputeDataType"].isHalf() or \
           kernel["ProblemType"]["ComputeDataType"].isBFloat16():
        module.addInst("v_cmp_eq_f32", self.vcc, sgpr("Alpha"), 0.0, "Alpha == 0.0f ?")
        module.addInst("s_cbranch_vccz", endCheckLabel.getLabelName(), "branch if alpha != 0")

      # AlphaType is int32
      else:
        module.addInst("s_cmp_eq_u32", sgpr("Alpha"), 0, "Alpha == 0 ?")
        module.addInst("s_cbranch_scc0", endCheckLabel.getLabelName(), "branch if alpha != 0")

      # Conditional set summation dimensions to 0 on SCC==1
      for i in range(0, self.numSgprSizesSum):
        module.addInst("s_mov_b32", sgpr("SizesSum+%u"%(i)), hex(0), "Set summation dim=0 if Alpha == 0")

      # Jump here if alpha is non-zero
      module.addCode(endCheckLabel)

    if kernel["MagicDivAlg"]==2:
      for idxChar in sorted(set(kernel["PackedC0IdxChars"][:-1] + kernel["PackedC1IdxChars"][:-1])):
          module.addInst("s_lshr_b32", sgpr("MagicAbitSize%s"%idxChar), sgpr("MagicShiftSize%s"%idxChar), 31,"extract abit")
          module.addInst("s_and_b32",  sgpr("MagicShiftSize%s"%idxChar), sgpr("MagicShiftSize%s"%idxChar), hex(0x7fffffff), "remove abit")

    ########################################
    # Debug Buffer
    if globalParameters["DebugKernel"]:
      module.addComment1("Debug Buffer")

      # nwg0 FIXME use NumWorkGroups0
      nwg0 = self.vgprPool.checkOut(1)
      tmpVgpr = self.vgprPool.checkOutAligned(2, 2)
      tmpSgpr = self.getTmpSgpr(1).idx()
      module.addComment("// nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine))
      module.addInst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile0"]-1), "MT0-1")
      module.addInst("v_mov_b32", vgpr(tmpVgpr), sgpr(tmpSgpr), "MT0-1")
      module.addInst("_v_add_co_u32", vgpr(nwg0), self.vcc, sgpr("SizesFree+0"), \
          vgpr(tmpVgpr), "%s = size0+MT0-1"%vgpr(nwg0))
      module.addCode(vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpVgpr, tmpSgpr))
      self.vgprPool.checkIn(tmpVgpr)
      self.nipt = 16 # num integers per thread
      v = self.vgprPool.checkOut(3)
      module.addInst("v_mov_b32", vgpr(v), sgpr("WorkGroup0"), "%s=wg0"%vgpr(v) )
      module.addInst("v_mov_b32", vgpr(v+1), sgpr("WorkGroup1"), "%s=wg1"%vgpr(v+1) )
      module.addInst("v_mul_lo_u32", vgpr(v+1), vgpr(v+1), vgpr(nwg0), \
          "%s=wg1*nwg0"%vgpr(v+1) )
      module.addInst("_v_add_co_u32", vgpr(v), self.vcc, vgpr(v), vgpr(v+1), \
          "%s=wg1*nwg0+wg0"%vgpr(v) )
      module.addCode(staticMultiply(vgpr(v), vgpr(v), kernel["NumThreads"], sgpr(tmpSgpr)))
      module.addInst("_v_add_co_u32", vgpr(v), self.vcc, vgpr(v), vgpr("Serial"), \
          "%s=tid+NT*(wg1*nwg0+wg0)=serial"%vgpr(v) )
      module.addInst("v_mul_lo_u32", vgpr(v), hex(self.nipt*4), vgpr(v), \
          "%s=serial*nipt*4"%vgpr(v) )
      module.addInst("v_mov_b32", vgpr(v+1), 0, "")
      module.addInst("_v_add_co_u32", vgpr("AddressDbg"), self.vcc, sgpr("AddressDbg"), \
          vgpr(v), "%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") )
      module.addInst("v_mov_b32", vgpr(v+2), sgpr("AddressDbg+1"), "%s=AddressD1"%vgpr(v+2) )
      module.addInst("_v_addc_co_u32", vgpr("AddressDbg+1"), self.vcc, vgpr(v+2), \
          vgpr(v+1), self.vcc, "%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") )
      module.addInst("s_mov_b32", sgpr("DebugKernelItems"), 0, "")
      self.vgprPool.checkIn(v)
      self.vgprPool.checkIn(nwg0)

    if self.db["InitLds"]:
      module.addCode(self.initLds(kernel, self.initLdsValue))

    if kernel["CheckTensorDimAsserts"]:
      module.addCode(self.getMultipleB32Assert(sgpr("SizesSum+%u"%(self.numSgprSizesSum-1)),
                  kernel["AssertSummationElementMultiple"], 0x1001))
      module.addCode(self.getMultipleB32Assert(sgpr("SizesFree+0"),
                  kernel["AssertFree0ElementMultiple"], 0x1002))
      module.addCode(self.getMultipleB32Assert(sgpr("SizesFree+1"),
                  kernel["AssertFree1ElementMultiple"], 0x1003))

    return module

  ##############################################################################
  # Perform a magic division (mul by magic number and shift)
  # dest is two consec SGPR, used for intermediate temp as well as final result
  # result quotient returned in sgpr(dest,1)
  ##############################################################################
  def sMagicDiv(self, kernel, dest, dividend, magicNumber, magicShift):
    module = Code.Module("sMagicDiv")
    module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(dest), sgpr(dest+1), dividend, magicNumber, "s_magic mul"))
    module.addInst("s_lshr_b64", sgpr(dest,2), sgpr(dest,2), magicShift, "sMagicDiv")
    return module

  ##############################################################################
  # Perform a sgpr version of magic division algo 2 (mul by magic number, Abit and shift)
  # dest is three consec SGPR, used for intermediate temp as well as final result
  # result quotient returned in sgpr(dest,1)
  ##############################################################################
  def sMagicDivAlg2(self, kernel, dest, dividend, magicNumber, magicShiftAbit):
    # dest+0: q,
    # dest+1: intermediate for magic div
    # dest+2: A tmpS to store the 'Abit' and the final Shift (use tmpS to save sgpr)
    tmpS = dest+2

    module = Code.Module("sMagicDivAlg2")
    module.addInst("s_mul_hi_u32", sgpr(dest+1), dividend, magicNumber, " s_magic mul, div alg 2")
    module.addInst("s_lshr_b32", sgpr(tmpS), magicShiftAbit, 31, " tmpS = extract abit")                              # tmpS = MagicAbit
    module.addInst("s_mul_i32", sgpr(dest), dividend, sgpr(tmpS), " s_magic mul, div alg 2")
    module.addInst("s_add_u32", sgpr(dest), sgpr(dest), sgpr(dest+1), "")

    module.addInst("s_and_b32",  sgpr(tmpS), magicShiftAbit, hex(0x7fffffff), " tmpS = remove abit to final shift")   # tmpS = MagicShift
    module.addInst("s_lshr_b32", sgpr(dest), sgpr(dest), sgpr(tmpS), " sMagicDiv Alg 2")
    return module

  def extractPackedCoord1ToRowStart(self, kernel, packedC1, packedCoordVgpr, storeChar):
    # calculate packed rowStart vgpr
    # vgprTmp assignments:
    #   - tmp+0 is the incoming packed coordinate 1, used on replay too
    #   - tmp+1 is DIV output
    #   - tmp+2 is scratch
    #   - tmp+3 holds thread rowStart free1 offset
    module = Code.Module("extractPackedCoord1ToRowStart")
    tmpV0 = self.vgprPool.checkOut(4)
    tmpV1 = tmpV0 + 1
    tmpV2 = tmpV0 + 2
    tmpV3 = tmpV0 + 3

    module.addInst("v_mov_b32", vgpr(tmpV0), vgpr(packedCoordVgpr),  "copy coord1 then unpack")
    for i,idx in enumerate(packedC1[:-1]):
      idxChar= globalParameters["IndexChars"][idx]
      module.addComment0("extract %s"%self.sizeRef(idx))
      module.addInst("V_MAGIC_DIV", \
                tmpV1, vgpr(tmpV0), sgpr("MagicNumberSize%s"%idxChar), \
                sgpr("MagicShiftSize%s"%idxChar), (sgpr("MagicAbitSize%s"%idxChar) if kernel["MagicDivAlg"]==2 else "0"), "")
      module.addInst("v_mul_lo_u32", vgpr(tmpV2), vgpr(tmpV1), self.sizeRef(idx), "remainder part 1")
      module.addInst("_v_sub_u32", vgpr(tmpV2), vgpr(tmpV0), vgpr(tmpV2), "remainder part 2")
      if i==0:
        module.addInst("v_mul_lo_u32", vgpr(tmpV3), vgpr(tmpV2), \
                  self.strideRef(storeChar, idx), "addrCalc <- scaled extracted dim")
      else:
        module.addInst("v_mul_lo_u32", vgpr(tmpV2), vgpr(tmpV2), \
                  self.strideRef(storeChar, idx), "scale extracted dim")
        module.addInst("_v_add_u32", vgpr(tmpV3), vgpr(tmpV3), \
                  vgpr(tmpV2), "addrCalc += scaled extracted dim ")

      if i < len(packedC1)-2:
        module.addInst("v_mov_b32", vgpr(tmpV0), vgpr(tmpV1), \
                  "Copy remaining bits for next divide")

    module.addComment0("extract final %s"%self.sizeRef(packedC1[-1]))
    module.addInst("v_mul_lo_u32", vgpr(tmpV2), vgpr(tmpV1), \
              self.strideRef(storeChar, packedC1[-1]), "scale final extracted dim")
    module.addInst("_v_add_u32", vgpr(self.coutRowPtr), vgpr(tmpV3), \
              vgpr(tmpV2), "rowStart += scaled extracted dim ")

    self.vgprPool.checkIn(tmpV0)
    return module

  ##############################################################################
  # Global Read Addresses: WorkGroup
  ##############################################################################
  def graWorkGroup(self, kernel):
    module = Code.Module("graWorkGroup")
    module.addComment0("graWorkGroup mapping")
    if kernel["GlobalSplitU"] > 1:
      module.addComment("// GSU-not-WGMapRR :nwg1 = (size%s + MT%s - 1) / MT%s;" \
          % (self.tileChar1, self.tileChar1, self.tileChar1))

      # gsuSumIdx = wg1 % GSU
      # wg1       = wg1 / GSU
      tmpSgpr = self.getTmpSgpr(3).idx() # needs 3
      divisor = tmpSgpr+2
      module.addInst("s_mov_b32", sgpr(divisor), sgpr("WorkGroup1"), \
          "copying for divisor")

      #tmp = self.vgprPool.checkOut(1)

      #module.addInst("v_mov_b32", vgpr(tmp), sgpr("WorkGroup1"), "wg1")
      #module.addCode(dump(vgpr(tmp))) # numerator

      module.addCode(scalarStaticDivideAndRemainder("WorkGroup1", "GSUSumIdx", \
          divisor, kernel["GlobalSplitU"], tmpSgpr, 1))

      #module.addInst("v_mov_b32", vgpr(tmp), sgpr("WorkGroup1"), "wg1")
      #module.addCode(dump(vgpr(tmp))) # quotient
      #module.addInst("v_mov_b32", vgpr(tmp), sgpr("GSUSumIdx"), "gsusumidx")
      #module.addCode(dump(vgpr(tmp))) # remainder
      #self.vgprPool.checkIn(tmp)
      #module.addInst("s_endpgm", "")

    ########################################
    # Blocked rows or columns
    absWgm = abs(kernel["WorkGroupMapping"])
    if abs(kernel["WorkGroupMapping"]) > 1:
      smallNumMagicShift = 31
      magicNumberWgm = ((1<<smallNumMagicShift) // absWgm + 1)

      tmpSgpr = self.getTmpSgpr(4).idx()
      blockId2  = tmpSgpr+0
      wgSerial2 = tmpSgpr+1
      wgmDivisor = tmpSgpr+2
      wgmDivisorMagicNumber = tmpSgpr+3

      module.addInst("s_mov_b32", sgpr(wgmDivisorMagicNumber), hex(magicNumberWgm)+'L', \
          "magic number for WGM==%u"%absWgm)
      # blockId and serial within block

      # note this overwrites blockId2+1
      module.addCode(self.sMagicDiv(kernel, dest=blockId2, dividend=sgpr("WorkGroup1"), \
          magicNumber=sgpr(wgmDivisorMagicNumber), magicShift=smallNumMagicShift))
      module.addInst("s_mul_i32", sgpr(wgSerial2), sgpr(blockId2), absWgm, "quotient * non-magic divisor")
      module.addInst("s_sub_u32", sgpr(wgSerial2), sgpr("WorkGroup1"), sgpr(wgSerial2), "WorkGroup1=remainder")
      module.addInst("s_mul_i32", sgpr(wgSerial2), sgpr(wgSerial2), sgpr("NumWorkGroups0"), "(wg1 % WGM)*nwg0")
      module.addInst("s_add_u32", sgpr(wgSerial2), sgpr(wgSerial2), sgpr("WorkGroup0"), "wgSerial = wg0 + (wg1 % WGM)*nwg0")

      module.addInst("s_cmp_ge_u32", sgpr(blockId2), sgpr("NumFullBlocks"), "blockId >= numFullBlocks ?")
      # reuse wgmDivisorMagicNumber - may override with remainder here:
      module.addInst("s_cmov_b32", sgpr(wgmDivisorMagicNumber), sgpr("MagicNumberWgmRemainder1"),  "")
      module.addInst("s_cselect_b32", sgpr(wgmDivisor), sgpr("WgmRemainder1"), absWgm,  "")

      if kernel["WorkGroupMapping"]>=0 :
        firstWg = "WorkGroup0"
        secondWg = "WorkGroup1"
      else:
        firstWg = "WorkGroup1"
        secondWg = "WorkGroup0"

      assert(self.sgprs[firstWg] & 0x1 == 0) # must be even and ...
      assert(self.sgprs[firstWg]+1 == self.sgprs[secondWg] ) # must be consecutive (for magic div below)
      module.addCode(self.sMagicDiv(kernel, dest=self.sgprs[firstWg], dividend=sgpr(wgSerial2), \
          magicNumber=sgpr(wgmDivisorMagicNumber), magicShift=smallNumMagicShift))
      if kernel["WorkGroupMapping"]<0 :
        module.addInst("s_mov_b32", sgpr("WorkGroup0"), sgpr(firstWg), "")
      module.addInst("s_mul_i32", sgpr("WorkGroup1"), sgpr("WorkGroup0"), sgpr(wgmDivisor), "quotient * non-magic divisor")
      module.addInst("s_sub_u32", sgpr("WorkGroup1"), sgpr(wgSerial2), sgpr("WorkGroup1"), "WorkGroup1=remainder")

      module.addInst("s_mul_i32", sgpr(blockId2), sgpr(blockId2), \
          abs(kernel["WorkGroupMapping"]), "blockId * WGM")

      module.addInst("s_add_u32", sgpr(secondWg), sgpr(secondWg), \
          sgpr(blockId2), "wg1 += blockId * WGM")

    return module

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  # global read addresses: tile offset assignment (message from .s)
  ##############################################################################
  def graTileAssignment(self, kernel, tP):
    module = Code.Module("graTileAssignment")
    tc = tP["tensorChar"]

    divisorName = tP["lvc"]
    divisor = kernel[divisorName]

    # force to swap gro-tile and gro-unroll for DirectToVgpr + TLU=False
    forceSwap = (kernel["DirectToVgpr%s"%tc] and not tP["tlu"])
    if tP["tlu"] or forceSwap:
      rReg = self.vgprPool.checkOut(1, "graTA rReg0", self.preventVgprOverflowDuringNewTile) # gro-tile = serial%divisor
      qReg = self.vgprPool.checkOut(1, "graTA qReg0", self.preventVgprOverflowDuringNewTile) # gro-unroll = serial/divisor
      tReg = rReg
      uReg = qReg
      tOpStr = "%"
      uOpStr = "/"
    else:
      qReg = self.vgprPool.checkOut(1, 'graTA qReg1', self.preventVgprOverflowDuringNewTile) # gro-tile = serial/divisor
      rReg = self.vgprPool.checkOut(1, 'graTA rReg1', self.preventVgprOverflowDuringNewTile) # gro-unroll = serial%divisor
      tReg = qReg
      uReg = rReg
      tOpStr = "/"
      uOpStr = "%"

    module.addComment0("%s = %u" % (divisorName, kernel[divisorName]))
    if self.groOffsetInMacroTile:
      tReg2 = tReg
      # treg2 and treg same register and value - we store the 'static'
      # part of the address calculation in the SRD to maximize the
      # range of the 32-bit GRO
      module.addComment0("%s = (local)gro%s-tile = serial%s%s (note (wg%s*MT%s) will be added to SRD)" \
          % (vgpr(tReg2), tc, tOpStr, divisorName, tc, tc) )
    else:
      tReg2 = self.vgprPool.checkOut(1, 'treg2', self.preventVgprOverflowDuringNewTile)
      module.addComment0("%s = gro%s-tile = serial%s%s + (wg%s*MT%s)" \
          % (vgpr(tReg2), tc, tOpStr, divisorName, tc, tc) )

    module.addComment0("%s = gro%s-unroll = serial%s%s" \
        % (vgpr(uReg), tc, uOpStr, divisorName) )

    tmpVgpr = self.vgprPool.checkOutAligned(2, 2, 'graTA vgpr', self.preventVgprOverflowDuringNewTile)
    tmpSgpr = self.getTmpSgpr(1).idx()

    dividendReg = "Serial" # local serial

    if kernel["WaveSeparateGlobalRead%s"%tc]:
      dividendReg = self.vgprPool.checkOut(1, "idInWave", self.preventVgprOverflowDuringNewTile)
      dummy       = self.vgprPool.checkOut(1, "dummy", self.preventVgprOverflowDuringNewTile)
      module.addCode(vectorStaticRemainder(dummy, dividendReg, "Serial", kernel["WavefrontSize"], tmpVgpr, tmpSgpr))

    splitRead = kernel["SplitGlobalRead"]
    # Split global read reorders reading rows within lanes of a wavefront
    # If the wavefront is reading all from a single row, then disable split global read for this tensor
    if divisor > kernel["WavefrontSize"]:
      splitRead = 1

    if kernel["DirectToVgpr%s"%tc]:
      # offset calculation for DirectToVgpr
      # ported code from local read for DirectToVgpr
      # alloc vgpr
      wReg       = self.vgprPool.checkOut(1,"wReg") # quotient
      # parameters
      tile01      = tP["tile01Idx"]
      waveWidth   = kernel["WavefrontSize"]
      num1DBlocks = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
      num1DWaves  = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
      vectorWidth = 1 # kernel["VectorWidth"] if ((tile01 == 0) and kernel["SourceSwap"]) else 1 # TODO: nonSwap VectorWidth
      strideTile  = 1 # tentative
      strideWave  = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth
      # tile offset
      module.addCode(vectorStaticRemainder(wReg, qReg, dividendReg, waveWidth, tmpVgpr, tmpSgpr))
      module.addCode(vectorStaticRemainder(wReg, rReg, qReg, kernel["MatrixInstN"], tmpVgpr, tmpSgpr))
      # block offset (no code. assuming num1DBlocks == 1)
      # unroll offset (no code here. This will be handled in GlobalOffset)
      # wave offset
      if num1DWaves > 1:
          module.addCode(vectorStaticDivide(wReg, dividendReg, waveWidth, tmpVgpr, tmpSgpr))
          module.addCode(vectorStaticRemainder(tmpVgpr, wReg, wReg, num1DWaves, tmpVgpr, tmpSgpr))
          module.addCode(staticMultiply(vgpr(wReg), vgpr(wReg), strideWave, sgpr(tmpSgpr)))
          module.addInst("_v_add_u32", vgpr(rReg), vgpr(wReg), vgpr(rReg),"")
          # need division for qReg
          module.addCode(vectorStaticDivide(qReg, qReg, kernel["MatrixInstN"], tmpVgpr, tmpSgpr))
          lrvwOther = self.lrvwB if tP["isA"] else self.lrvwA # The other side of lrvw
          if lrvwOther == 2 and not self.allowLRVWforTLUandMI and tP["tlu"]:
            # DirectToVgpr + LocalReadVectorWidth=2 case, multiply qReg by 2
            module.addCode(staticMultiply(vgpr(qReg), vgpr(qReg), lrvwOther, sgpr(tmpSgpr)))
      # release register
      self.vgprPool.checkIn(wReg)
    elif splitRead > 1:
      splitGroup = self.vgprPool.checkOut(1, "splitGroup", self.preventVgprOverflowDuringNewTile)
      splitIndex = self.vgprPool.checkOut(1, "splitIndex", self.preventVgprOverflowDuringNewTile)
      waveSize = kernel["WavefrontSize"]
      groupDivisor = waveSize // splitRead
      groupOffset = waveSize // divisor
      newDivisor = divisor // splitRead

      module.addCode(vectorStaticRemainder(tmpVgpr, splitIndex, dividendReg, groupDivisor, tmpVgpr, tmpSgpr, "Split index"))
      module.addCode(vectorStaticDivideAndRemainder(qReg, rReg, splitIndex, newDivisor, tmpVgpr, tmpSgpr))

      module.addCode(vectorStaticDivideAndRemainder(splitGroup, splitIndex, dividendReg, waveSize, tmpVgpr, tmpSgpr))

      if groupOffset > 1:
        module.addInst("v_mul_u32_u24", vgpr(splitGroup), groupOffset, vgpr(splitGroup), "Calculate wave group offset")
      module.addInst("_v_add_u32", vgpr(qReg), vgpr(splitGroup), vgpr(qReg), "Add wave group")

      module.addCode(vectorStaticDivide(splitIndex, splitIndex, groupDivisor, tmpVgpr, tmpSgpr, "Calculate index offset"))
      module.addInst("v_mul_u32_u24", vgpr(splitIndex), newDivisor, vgpr(splitIndex), "Calculate index offset")
      module.addInst("_v_add_u32", vgpr(rReg), vgpr(splitIndex), vgpr(rReg), "Add index offset")

      self.vgprPool.checkIn(splitIndex)
      self.vgprPool.checkIn(splitGroup)
    else:
      module.addCode(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgpr, tmpSgpr))

    if kernel["WaveSeparateGlobalRead%s"%tc]:
      module.addInst("v_readfirstlane_b32", sgpr(tmpSgpr), vgpr("Serial"), "WaveIdxWavefrontWidth")
      module.addInst("s_lshr_b32", sgpr(tmpSgpr), sgpr(tmpSgpr), hex(log2(kernel["WavefrontSize"])), "WaveId")
      module.addInst("s_mul_i32", sgpr(tmpSgpr), sgpr(tmpSgpr), kernel[tP["lsp"]] * tP["nrp"], \
          "Global Read Wave: each wave loads continuous lsp(%u)*nrp(%u) columns" % (kernel[tP["lsp"]], tP["nrp"]))
      module.addInst("_v_add_u32", vgpr(qReg), sgpr(tmpSgpr), vgpr(qReg), \
          "Global Read Wave: add back to column index")
      self.vgprPool.checkIn(dividendReg)
      self.vgprPool.checkIn(dummy)

    if tP["glvw"] > 1:
      if tP["tlu"]:
        module.addComment0("gro-tile *= glvw")
        module.addCode(staticMultiply(vgpr(tReg), vgpr(tReg), tP["glvw"], sgpr(tmpSgpr)))
      else:
        module.addComment0("gro-unroll *= glvw")
        module.addCode(staticMultiply(vgpr(uReg), vgpr(uReg), tP["glvw"], sgpr(tmpSgpr)))
    if forceSwap:
      # in this case, need to multiply vw to gro-tile
      module.addComment0("gro-tile *= vw")
      module.addCode(staticMultiply(vgpr(tReg), vgpr(tReg), kernel["VectorWidth"], sgpr(tmpSgpr)))

    if not self.groOffsetInMacroTile:
      # Buffer Load will set the SRD to start of the MacroTile
      # So don't add the static wg-related component here - save for later.
      module.addCode(staticMultiply(vgpr(tmpVgpr), sgpr(tP["wg"]), kernel[tP["mt"]]))  # workgroup
      module.addInst("_v_add_co_u32", vgpr(tReg2), self.vcc, vgpr(tmpVgpr), \
          vgpr(tReg), "gro%s-tile = serial%s%s*VW + (wg%s*MT%s)" \
          % (tc, tOpStr, divisorName, tc, tc) )

    if kernel["GlobalSplitU"] > 1:
      uReg2 = self.vgprPool.checkOut(1, "uReg2", self.preventVgprOverflowDuringNewTile)
      module.addInst("v_mov_b32", vgpr(uReg2), vgpr(uReg), "copy for GlobalSplitU")
      tP["gpr"]["uReg2"] = uReg2
    tP["gpr"]["lwoT"] = tReg
    tP["gpr"]["tReg"] = tReg2
    tP["gpr"]["uReg"] = uReg
    self.vgprPool.checkIn(tmpVgpr)

    return Code.Module("graTileAssignment (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Unroll Assignment
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    module = Code.Module("graUnrollAssignment")
    # note groOffsetInMacroTile rolls these into SRD so don't change here:
    if not self.groOffsetInMacroTile and kernel["GlobalSplitU"] > 1:
      gsuOffset = self.vgprPool.checkOut(1, "gsuOffset", self.preventVgprOverflowDuringNewTile)
      module.addInst("v_mov_b32", vgpr(gsuOffset), sgpr("GSUSumIdx"), "=gsuSumIdx")
      tmpSgpr = self.getTmpSgpr(1).idx()
      # graUnrollAssignment += gsuSumIdx*DepthU
      module.addCode(staticMultiply(vgpr(gsuOffset), vgpr(gsuOffset), kernel["DepthU"], sgpr(tmpSgpr)))

      module.addInst("_v_add_co_u32", vgpr(tP["gpr"]["uReg"]), self.vcc, \
          vgpr(gsuOffset), vgpr(tP["gpr"]["uReg"]), \
          "graUnrollAssignment += gsuOffset")
      self.vgprPool.checkIn(gsuOffset)
    else:
      module.addComment0(vgpr(tP["gpr"]["uReg"]))

    return Code.Module("graUnrollAssignment (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  def graOtherFreeAssignments(self, kernel):
    module = Code.Module("graOtherFreeAssignments")
    module.addComment0(sgpr("WorkGroup2"))
    return module

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  def graOtherSummationAssignments(self, kernel):
    module = Code.Module("graOtherSummationAssignments")
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      index = i
      module.addInst(".set", "globalReadOffsetA%s" % self.indexChars[index], "0")
      module.addInst(".set", "globalReadOffsetB%s" % self.indexChars[index], "0")
    return module

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  def graTileOffsets(self, kernel, tP):
    module = Code.Module("graTileOffsets")
    tc = tP["tensorChar"]
    tP["vgprPackedOffsets"] = None
    tP["vgprTileOffsetsCheckOut"] = False
    tP["numVgprTileOffsets"] = 0
    if kernel["_UseSgprForGRO"]:
      # Let the vgprTileOffsets checkin handle tReg later since these are same vgpr
      tP["vgprTileOffsets"] = tP["gpr"]["tReg"]
    else:
      numTileOffsets = tP["nrt"]
      if tP["rtc"]:
        numTileOffsets *= tP["glvw"]
      if self.useGlobalReadTileVgpr:
        tP["vgprTileOffsets"] = self.startVgprGlobalReadTileOffsetA if tP["isA"] else self.startVgprGlobalReadTileOffsetB
        tP["numVgprTileOffsets"] = numTileOffsets # keep numTileOffsets for later use
      else:
        tP["vgprTileOffsets"] = self.vgprPool.checkOut(numTileOffsets, "vgprTileOffsets", self.preventVgprOverflowDuringNewTile)
        tP["vgprTileOffsetsCheckOut"] = True
      v = tP["vgprTileOffsets"]
      numExtraPackedOffsetsPerTile = len(tP["PackedIndices"])-1
      if numExtraPackedOffsetsPerTile:
        tP["vgprPackedOffsets"] = self.vgprPool.checkOut(numExtraPackedOffsetsPerTile * numTileOffsets, "vgprPackedOffsets", self.preventVgprOverflowDuringNewTile)
      strideIdx = tP["lsc"] if tP["tlu"] else tP["lsp"]
      stride = kernel[strideIdx]
      # adjustment for DirectToVgpr + tlu=False + VW > 1 case
      strideInterleave = False
      if kernel["DirectToVgpr%c"%tc] and (not tP["tlu"]) and kernel["VectorWidth"] > 1:
        strideInterleave = True
        stride = stride * kernel["VectorWidth"] - (kernel["VectorWidth"] - 1)

      if tP["rtc"]:
        assert(numExtraPackedOffsetsPerTile == 0) # not supported here
        # l=0, s=0
        module.addInst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["tReg"]), "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], 0, 0) )
        # l=0, s>0
        for s in range(1, tP["glvw"]):
          module.addInst("_v_add_co_u32", vgpr(v+s), self.vcc, 1, \
              vgpr(v+s-1), "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], 0, s) )
        for l in range(1, tP["nrt"]):
          # l>0, s=0
          strideValue = stride
          if strideInterleave and (l & 1) != 0:
            strideValue = 1
          module.addInst("_v_add_co_u32", vgpr(v+l*tP["glvw"]), self.vcc, strideValue, \
              vgpr(v+(l-1)*tP["glvw"]), \
              "gro%s%s_%u_s%u + %s"%(tP["tensorChar"], tP["tileChar"], l, 0, strideIdx) )
          # l>0, s>0
          for s in range(1, tP["glvw"]):
            module.addInst("_v_add_co_u32", vgpr(v+l*tP["glvw"]+s), self.vcc, \
                1, vgpr(v+l*tP["glvw"]+(s-1)), \
                "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], l, s) )

      else:
        module.addInst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["tReg"]), "gro%s%s_%u"%(tP["tensorChar"], tP["tileChar"], 0) )
        for l in range(1, tP["nrt"]):
          strideValue = stride
          if strideInterleave and (l & 1) != 0:
            strideValue = 1
          module.addInst("_v_add_co_u32", vgpr(v+l), self.vcc, strideValue, \
              vgpr(v+l-1), "gro%s%s_%u += %s"%(tP["tensorChar"], tP["tileChar"], l, strideIdx) )
        if numExtraPackedOffsetsPerTile:
          tmpV = self.vgprPool.checkOutAligned(2,2,"packTmp", self.preventVgprOverflowDuringNewTile)

          for l in range(0, tP["nrt"]):
            lastGroVgpr = vgpr(v+l)
            lastGroIdx = tP["PackedIndices"][0]
            module.addSpaceLine()
            for p in range(0, numExtraPackedOffsetsPerTile):
              groIdx  = tP["PackedIndices"][p+1]
              groChar = globalParameters["IndexChars"][tP["PackedIndices"][p+1]]
              groVgpr = vgpr(tP["vgprPackedOffsets"] + l*numExtraPackedOffsetsPerTile + p)
              pChar = globalParameters["IndexChars"][tP["PackedIndices"][p]]
              module.addInst("V_MAGIC_DIV", \
                  tmpV, lastGroVgpr, sgpr("MagicNumberSize%s"%pChar), \
                  sgpr("MagicShiftSize%s"%pChar), (sgpr("MagicAbitSize%s"%pChar) if kernel["MagicDivAlg"]==2 else "0") )
              module.addInst("v_mov_b32", groVgpr, vgpr(tmpV), "extract gro%s%s_%u (%s)"%(tc,groChar,l,groVgpr))
              module.addInst("v_mul_lo_u32", vgpr(tmpV), groVgpr, sgpr("SizesFree+%u"%lastGroIdx), "remainder part 1")
              module.addInst("_v_sub_u32", lastGroVgpr, lastGroVgpr, vgpr(tmpV), \
                  "remove extracted bits from gro%s%s_%u (%s)"%(tc, globalParameters["IndexChars"][lastGroIdx], l, lastGroVgpr))
              lastGroVgpr = groVgpr
              lastGroIdx = groIdx
          self.vgprPool.checkIn(tmpV)

      # groOffsetInMacroTile uses same register for both of these, don't free it here:
      if tP["gpr"]["lwoT"] != tP["gpr"]["tReg"] :
        self.vgprPool.checkIn(tP["gpr"]["tReg"])
        tP["gpr"]["tReg"] = None
    return Code.Module("graTileOffsets (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  def graUnrollOffsets(self, kernel, tP):
    module = Code.Module("graUnrollOffsets")
    tc = tP["tensorChar"]
    if kernel["_UseSgprForGRO"]:
      tP["gpr"]["unrollOffsets"] = tP["gpr"]["uReg"]
    else:
      numUnrollOffsets = tP["nru"]
      if tP["ruc"]:
        numUnrollOffsets *= tP["glvw"]
      if self.useGlobalReadTileVgpr:
        tP["gpr"]["unrollOffsets"] = self.startVgprGlobalReadUnrollOffsetA if tP["isA"] else self.startVgprGlobalReadUnrollOffsetB
      else:
        tP["gpr"]["unrollOffsets"] = self.vgprPool.checkOut(numUnrollOffsets, "unrollOffsets", self.preventVgprOverflowDuringNewTile)
      v = tP["gpr"]["unrollOffsets"]
      strideIdx = (tP["lsp"] if tP["tlu"] else tP["lsc"])
      stride = kernel[strideIdx]
      prevStride = 0
      totalStride = 0
      lrvwOther = self.lrvwB if tP["isA"] else self.lrvwA # The other side of lrvw
      tluOther = kernel["ProblemType"]["TLUB"] if tP["isA"] else kernel["ProblemType"]["TLUA"] # The other side of tlu
      if tP["ruc"]:
        # l=0, s=0
        module.addInst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["uReg"]), "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, 0) )
        # l=0, s>0
        for s in range(1, tP["glvw"]):
          module.addInst("_v_add_co_u32", vgpr(v+s), self.vcc, 1, \
              vgpr(v+s-1), "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, s) )
        for l in range(1, tP["nru"]):
          # l>0, s=0
          totalStride += stride
          if  tP["tlu"] and kernel["DirectToVgpr%s"%tc] and lrvwOther == 2 and not tluOther:
            # DirectToVgpr + LocalReadVectorWidth=2 + other side of TLU is false case, stride * 2 is added every 2. Add 1 in odd l case
            totalStride = stride * (l - (l % 2)) + (l % 2)
          currStride = totalStride - prevStride
          prevStride = totalStride
          module.addInst("_v_add_co_u32", vgpr(v+l*tP["glvw"]), self.vcc, currStride, \
              vgpr(v+(l-1)*tP["glvw"]), \
              "gro%s%s_%u_s%u + %s"%(tP["tensorChar"], self.unrollChar, l, 0, strideIdx) )
          # l>0, s>0
          for s in range(1, tP["glvw"]):
            module.addInst("_v_add_co_u32", vgpr(v+l*tP["glvw"]+s), self.vcc, \
                1, vgpr(v+l*tP["glvw"]+(s-1)), \
                "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, s) )
      else:
        module.addInst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["uReg"]), "gro%s%s_%u"%(tP["tensorChar"], self.unrollChar, 0) )
        for l in range(1, tP["nru"]):
          totalStride += stride
          if tP["tlu"] and kernel["DirectToVgpr%s"%tc] and lrvwOther == 2 and not tluOther:
            # DirectToVgpr + LocalReadVectorWidth=2 case, stride * 2 is added every 2. Add 1 in odd l case
            totalStride = stride * (l - (l % 2)) + (l % 2)
          currStride = totalStride - prevStride
          prevStride = totalStride
          module.addInst("_v_add_co_u32", vgpr(v+l), self.vcc, currStride, \
              vgpr(v+l-1), "gro%s%s_%u + %s"%(tP["tensorChar"], self.unrollChar, l, strideIdx) )
      #self.vgprPool.checkIn(tP["gpr"]["uReg"])
    return Code.Module("graUnrollOffsets (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Shift A/B
  # See if the load (including vw) will extend past the 'free' dim of the
  # tensor.  If so clip to the last legal value which is inside the array
  ##############################################################################
  def graShift(self, kernel, tP):
    # graShift requires a vgpr for each address component (so each component
    # can be examined and shifted if necessary) - therefore does not work
    # with UseSgprForGRO.
    assert(not kernel["_UseSgprForGRO"])

    module = Code.Module("graShift")
    #tc = tP["tensorChar"]
    # edge value
    margin = tP["glvw"] if tP["rtv"] else 1
    edge = self.vgprPool.checkOut(1, "edge", self.preventVgprOverflowDuringNewTile)

    if self.groOffsetInMacroTile:
      # Subtract the static component from SizesFree:
      tmpSgpr = self.getTmpSgpr(1).idx()
      module.addInst("s_mul_i32", sgpr(tmpSgpr), sgpr(tP["wg"]), kernel[tP["mt"]], "WorkGroup[01] * MT")
      module.addInst("s_sub_u32", sgpr(tmpSgpr), self.sizeRef(tP["idx"]), sgpr(tmpSgpr), \
                "edge = Size%s - WG*MT"%(tP["tileChar"]))
      # use math here to use unsigned (to increase range)
      #  - add srdShiftLeft to tmpSgpr - ensure it is always positive
      #  - below add srdShiftLeft to a tmp copy of the offset used for the compare
      # edge = (Size - WG*MT) - margin = the last valid load position that won't cause OOB
      # offset = the current load position for this thread
      # so if offset is larger than edge, we go back to the edge position
      module.addInst("s_sub_u32", sgpr(tmpSgpr), sgpr(tmpSgpr), margin, "edge -= margin(%u)"%(margin))
      module.addInst("v_mov_b32", vgpr(edge), sgpr(tmpSgpr), \
          "edge vgpr = Size%s- WG*MT - margin(%u)"%(tP["tileChar"], margin) )
      #shiftedEdge = self.vgprPool.checkOut(1, "shiftedEdge", self.preventVgprOverflowDuringNewTile)
      #module.addInst("_v_add_co_u32", vgpr(shiftedEdge), self.vcc, vgpr(edge), self.srdShiftLeft[tc],
      #             "shiftedEdge = edge + srdShiftLeft({})".format(self.srdShiftLeft[tc]))
    else:
      tmpSgpr = self.getTmpSgpr(1).idx()
      module.addInst("s_sub_u32", sgpr(tmpSgpr), self.sizeRef(tP["idx"]), margin, \
          "edge = Size%s-%u"%(tP["tileChar"], margin) )
      module.addInst("v_mov_b32", vgpr(edge), sgpr(tmpSgpr), \
          "edge vgpr = Size%s-%u"%(tP["tileChar"], margin) )

    if kernel["CheckDimOverflow"]:
      # if tensor is really skinny (SizesFree is less then glvw) then shifting fails-
      # can detect here if the computed edge after subtracting marging is <0
      module.addCode(self.getCmpAssert(self.asmAssert.ge_i32, vgpr(edge), 0))
    #module.addCode(self.getCmpAssert(self.asmAssert.ne, sgpr("WorkGroup0"),1))

    # shift offsets
    vSrc = tP["vgprTileOffsets"]
    if self.useGlobalReadTileVgpr:
      # self.useGlobalReadTileVgpr case, use new vgpr as dst to avoid overwritting GlobalReadTileVgpr with shifted value
      tP["vgprTileOffsets"] = self.vgprPool.checkOut(tP["numVgprTileOffsets"], "vgprTileOffsets", self.preventVgprOverflowDuringNewTile)
      tP["vgprTileOffsetsCheckOut"] = True
    vDst = tP["vgprTileOffsets"]
    tmpSgpr = self.getTmpSgpr(self.laneSGPRCount).idx()
    for l in range(0, tP["nrt"]):
      # compare
      cmpCommentText = "offset < edge"
      if self.groOffsetInMacroTile:
        #shiftedOffset = self.vgprPool.checkOut(1, "shiftedOffset", self.preventVgprOverflowDuringNewTile)
        #module.addInst("_v_add_co_u32", vgpr(shiftedOffset), self.vcc, vgpr(vSrc+l), self.srdShiftLeft[tc], "shiftedOffset = offset + srdShiftLeft(%u)"%(self.srdShiftLeft[tc]))
        ## int cmp since if we are near the front of the tile this may go negative:
        #module.addInst("v_cmp_lt_u32", sgpr(tmpSgpr,self.laneSGPRCount), vgpr(shiftedOffset), vgpr(shiftedEdge),
        #             "shiftedOffset < shiftedEdge")
        #self.vgprPool.checkIn(shiftedOffset)
        module.addInst("v_min_i32", vgpr(vDst+l), vgpr(edge), vgpr(vSrc+l),
                     "offset = (%s) ? offset(v%u) : edge(v%u)"%(cmpCommentText, vSrc+l, edge))
      else:
        module.addInst("v_cmp_lt_u32", sgpr(tmpSgpr,self.laneSGPRCount), vgpr(vSrc+l), vgpr(edge),
                     "shiftedOffset < shiftedEdge")
        # shift
        module.addInst("v_cndmask_b32", vgpr(vDst+l), vgpr(edge), vgpr(vSrc+l), sgpr(tmpSgpr,self.laneSGPRCount),
                     "offset = (%s) ? offset(v%u) : edge(v%u)"%(cmpCommentText, vSrc+l, edge))
    self.vgprPool.checkIn(edge)
    #if self.groOffsetInMacroTile:
    #  self.vgprPool.checkIn(shiftedEdge)

    #if tP["isB"]:
    #  module.addInst("s_endpgm")

    return module

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  def graFinalOffsets(self, kernel, tP):
    module = Code.Module("graFinalOffsets")
    tc = tP["tensorChar"]
    tmp = self.vgprPool.checkOut(3, "tmp", self.preventVgprOverflowDuringNewTile)
    graIdx = 0
    swapPerpPara = (((tc=="A" and kernel["DirectToVgprA"]) or (tc=="B" and kernel["DirectToVgprB"])) \
                    and (not tP["tlu"]) and tP["nrp"] > 1)

    if not swapPerpPara:
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
              # single loop
              singleModule, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
              module.addCode(singleModule)
    else:
      # swap para and perp
      for para in range(0, tP["nrc"]):
        for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
          for perp in range(0, tP["nrp"]):
            for sPerp in range(0, tP["nrpv"]):
              # single loop
              singleStr, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
              module.addCode(singleModule)

    if tP["vgprTileOffsetsCheckOut"]:
      self.vgprPool.checkIn(tP["vgprTileOffsets"])
      tP["vgprTileOffsets"] = None
      tP["vgprTileOffsetsCheckOut"] = False
      # _UseSgprForGRO uses same vgpr for ureg and tP["gpr"]["unrollOffsets"] so
      # let checkin(ureg) do the checkin
      # vgprTileOffsets is renamed version of treg/lwo so checkin here

    if not kernel["_UseSgprForGRO"] and not self.useGlobalReadTileVgpr:
      self.vgprPool.checkIn(tP["gpr"]["unrollOffsets"])
      tP["gpr"]["unrollOffsets"] = None

    if tP["vgprPackedOffsets"] != None:
      self.vgprPool.checkIn(tP["vgprPackedOffsets"])
      tP["vgprPackedOffsets"] = None

    self.vgprPool.checkIn(tmp)
    #if tP["isB"]:
    #  module.addCode(self.getBomb(0x100))

    return Code.Module("Global Read Addresses: Final Offsets A/B (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B (single loop)
  ##############################################################################
  def graFinalOffsetsSingleLoop(self, kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara):
    module = Code.Module("graFinalOffsetsSingleLoop")
    problemType = kernel["ProblemType"]
    tVW = 1
    tVS = 0
    uVW = 1
    uVS = 0
    if tP["rtc"]:
      tVW = tP["glvw"]
      tVS = 1
    elif tP["ruc"]:
      uVW = tP["glvw"]
      uVS = 1

    # single loop start

    # vgpr assignments
    if tP["tlu"]:
      vgprTile   = tP["vgprTileOffsets"]   + para*tVW + sPara*tVS
      vgprUnroll = tP["gpr"]["unrollOffsets"] + perp*uVW + sPerp*uVS
    else:
      vgprTile   = tP["vgprTileOffsets"]   + perp*tVW + sPara*tVS
      vgprUnroll = tP["gpr"]["unrollOffsets"] + para*uVW + sPerp*uVS

    if graIdx==0 or not kernel["_UseSgprForGRO"]:
      # emit global offset macro
      # TODO -refactor this and macro def to pass all indices, use the ones we need
      if kernel["BufferLoad"]:
        bfArgs = ["GLOBAL_OFFSET_%s" % tP["tensorChar"], "vgprGlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx)]
      else:
        bfArgs = ["GLOBAL_OFFSET_%s" % tP["tensorChar"], "vgprGlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx)]
      packedIter = 0 #iterator through ia
      iaToGpr = [None] * problemType["TotalIndices"]
      for i in tP["ia"]:
        if i < problemType["NumIndicesC"]:
          if i == tP["tileIdx"]:
            iaToGpr[i] = vgprTile
            bfArgs.append( "%2u" % iaToGpr[i] )
          else:
            if isPackedIndex(kernel,i):
              iaToGpr[i] = tP["vgprPackedOffsets"] + \
                            (vgprTile-tP["vgprTileOffsets"])*(len(tP["PackedIndices"])-1) + \
                            packedIter
              bfArgs.append( "%2u" % (iaToGpr[i]) )
              packedIter += 1
            else:
              # just a group index
              if not kernel["BufferLoad"]:  # buffer load adds these to SRD not the GLOBAL_OFFSET here
                bfArgs.append( "sgprWorkGroup%u"%i )
        else: # summation index
          if i == problemType["IndexUnroll"]:
            iaToGpr[i] = vgprUnroll
            bfArgs.append( "%2u" % iaToGpr[i] )
          # other summation indices are ignored

      bfArgs.append( "%u" % tmp )
      bfArgs.append( "gRO%s_%u_%u_%u_%u" % (tP["tensorChar"], para, sPara, perp, sPerp) )
      module.addInst(*bfArgs)

      tmpSgpr = self.getTmpSgpr(2).idx()

      # modify start
      if (not kernel["_UseSgprForGRO"]) and kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
         # add room for instruction offset
        groVgpr = "GlobalReadOffset%s+%u" % (tP["tensorChar"], graIdx)
        module.addInst("s_mov_b32", sgpr(tmpSgpr), self.buff_load_inst_offset_max, "" )
        module.addInst("_v_add_u32", vgpr(groVgpr), vgpr(groVgpr), sgpr(tmpSgpr), "shift for UseInstOffsetForGRO")

        ldsInc = (self.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]
        if kernel["LdsBlockSizePerPad%s"%tc] != 0:
          ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
        else:
          padInterval = (self.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.bpr
          ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpe"]

        # buffer_load only support 12 bit instruction offset
        # we have to increase m0 if offset is larger thant 12 bits
        # so only keep 12 bit offset and subtract it on global address
        # global address will add back by buffer_load instruction offset
        ldsInc = (ldsInc * graIdx) % self.buff_load_inst_offset_max
        if (ldsInc != 0):
          module.addInst("s_mov_b32", sgpr(tmpSgpr), ldsInc, "" )
          module.addInst("_v_sub_u32", vgpr(groVgpr), vgpr(groVgpr), sgpr(tmpSgpr), "sub offset for buffer_load instoffset")

    needFirstSgprOffset = kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]
    if (kernel["_UseSgprForGRO"] or self.checkGRO) and (needFirstSgprOffset or graIdx > 0):
      # compute offsets for scalar global read offsets:
      if kernel["_UseSgprForGRO"]:
        tmpIdx = graIdx if needFirstSgprOffset else graIdx-1
        scalarGro = "ScalarGlobalReadOffset%s+%u"%(tc, tmpIdx)
      else:
        scalarGro = self.getTmpSgpr(1).idx()

      # this needs unroll stride in some cases and free stride in others
      # if we have multiple free strides - what is expected behavior?
      # could just extract the first free dimension from A?
      stride1 = "Stride%s%s"%(tc,self.indexChars[tP["idx"]])
      if tP["tlu"]:
        tileStride   = kernel[tP["lsc"]] * (para*tVW + sPara*tVS)
        unrollStride = kernel[tP["lsp"]] * (perp*uVW + sPerp*uVS)
        unrollSummation = [ i for i in tP["ia"] if i in problemType["IndicesSummation"] ]
        strideU = "Stride%s%s"%(tc,self.indexChars[unrollSummation[-1]])
        module.addInst("s_mul_i32", sgpr(scalarGro), sgpr(strideU), unrollStride, \
                     "compute offset diff (scaled unrollDim)")
        if tileStride:
          module.addInst("s_add_u32", sgpr(scalarGro), sgpr(scalarGro), tileStride, \
                     "compute offset diff (tileDim)")
      else:
        tileStride   = kernel[tP["lsp"]] * (perp*tVW + sPara*tVS)
        unrollStride = kernel[tP["lsc"]] * (para*uVW + sPerp*uVS)
        strideF = "Stride%s%s"%(tc,self.indexChars[tP['tileIdx']])
        module.addInst("s_mul_i32", sgpr(scalarGro), sgpr(strideF), tileStride, \
                     "compute offset diff (scaled tileDim)")
        if unrollStride:
          module.addInst("s_add_u32", sgpr(scalarGro), sgpr(scalarGro), unrollStride, \
                     "compute offset diff (unrollDim)")

      # Using offsets so GRO holds a byte offset not an element offset
      # So scale here before comparison:
      module.addInst("s_lshl_b32", \
          sgpr(scalarGro), \
          sgpr(scalarGro), \
          hex(log2(tP["bpe"])), \
          "scalar offset *= bytes/element")

      if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
        # add room for instruction offset
        module.addInst("s_add_u32", sgpr(scalarGro), sgpr(scalarGro), self.buff_load_inst_offset_max, "shift for UseInstOffsetForGRO")

        ldsInc = (self.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]
        if kernel["LdsBlockSizePerPad%s"%tc] != 0:
          ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
        else:
          padInterval = (self.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.bpr
          ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpe"]

        # buffer_load only support 12 bit instruction offset
        # we have to increase m0 if offset is larger thant 12 bits
        # so only keep 12 bit offset and subtract it on global address
        # global address will add back by buffer_load instruction offset
        ldsInc = (ldsInc * graIdx) % self.buff_load_inst_offset_max
        if (ldsInc != 0):
          module.addInst("s_sub_u32", sgpr(scalarGro), sgpr(scalarGro), ldsInc, "sub offset for buffer_load instoffset")

      if self.checkGRO:
        # Debug mode to verify that the computed offsets are offset by the expected scalar
        print(tc, "tileStride=", tileStride, "unrollStride=", unrollStride, \
              "stride=%s"%(stride1))

        module.addCode(self.getVectorDiffAssert(vgpr("GlobalReadOffset%s+%u"%(tc,0)), \
                                             vgpr("GlobalReadOffset%s+%u"%(tc,graIdx)), \
                                             sgpr(scalarGro)))

    # dump final offsets
    # BufferLoad flavor:
    #if tP["isA"]:
    #  module.addCode(self.dump(vgpr("GlobalReadOffset%s+%u+0"%(tP["tensorChar"], graIdx))))
    # Flat load flavor:
    #module.addCode(dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx))))
    #module.addCode(dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx))))
    graIdx += self.rpgo if kernel["BufferLoad"] else self.rpga

    return module, graIdx

  ##############################################################################
  # Add the constant offsets to the specified srd.
  # Srd is set to point to the base of the tile. All offsets except lowest-order
  # 2d dims are computed into the SRD.
  # GRO are offset from the tile SRD and the first GRO will be 0
  # Only called for BufferLoad=1 (or eventually BufferStore=1)
  ##############################################################################
  def computeLoadSrd(self, kernel, tP, tc, indices, bpe):
    module = Code.Module("computeLoadSrd")

    stmp = self.getTmpSgpr(2+2+1).idx()
    tileStart = stmp+2
    wroteTileStart = False
    #---
    # Compute tileStart #elements from the 2D array start
    # Add tile (and unroll if GSU) component into SRD - SRD will point to beginning of the macro-tile:
    if self.groOffsetInMacroTile:
      # packed modes can't use this mode, and code here assumes 1 index.
      assert(len(kernel["PackedC0IndicesX"])==1)
      assert(len(kernel["PackedC1IndicesX"])==1)

      wroteTileStart = True
      #tP['ia'][1]

      # This is guaranteed to fit in 32-bit since the WG*MT is a number of elements in some unsigned direction:
      module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(tP["wg"]), kernel[tP["mt"]], "WorkGroup[01] * MT"))
      if kernel["CheckDimOverflow"] >=2:
        module.addCode(self.getCmpAssert(self.asmAssert.eq, sgpr(tileStart+1),0))
      strideF = self.strideRef(tc, tP['tileIdx'])
      if not self.isConstUnitStride(strideF):
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart), sgpr(tileStart+1), sgpr(tileStart+0), \
                   strideF, "tlu=0, scaled tile-offset by stride"))

      if kernel["GlobalSplitU"] > 1:
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), kernel["DepthU"], sgpr("GSUSumIdx"), "gsuOffset = DepthU*bpe*GSUSumIdx"))
        if kernel["CheckDimOverflow"] >=2:
          module.addCode(self.getCmpAssert(self.asmAssert.eq, sgpr(stmp+1),0))
        unrollSummation = [ i for i in tP["ia"] if i in kernel["ProblemType"]["IndicesSummation"] ]
        stride = self.strideRef(tc,unrollSummation[-1])
        if tP["tlu"] and not self.isConstUnitStride(stride):
          # non-transpose case, unroll is in perp dim and should be scaled by unroll Stride
          module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp), sgpr(stmp+1), sgpr(stmp+0), \
                    stride, "tlu=1, scaled unroll-offset by stride"))

        module.addInst("s_add_u32",  sgpr(tileStart+0), sgpr(tileStart+0), sgpr(stmp+0), "accum GsuOffset term to tilestart")
        module.addInst("s_addc_u32", sgpr(tileStart+1), sgpr(tileStart+1), sgpr(stmp+1), "accum GsuOffset term to tilestart")

    # Output : tileStart[0:1] have offset in elements from the 2D start of the tile.
    # if groOffsetInMacroTile=1, 2DStart + tileStart gives the the start of the macro-tile;
    # This is used to compute the limit.
    # Later we modify tileStart to include batch and higher-order dims and add this to SRD.

    #---
    # Compute BUFFER Limit:
    prePad = self.srdShiftLeft[tc] * tP["bpe"] # leave room in case we have to pointer shift

    if not wroteTileStart:
      module.addInst("s_mov_b32", sgpr(tileStart+0), 0, "set default tileStart")
      module.addInst("s_mov_b32", sgpr(tileStart+1), 0, "set default tileStart")

    if self.use64bShadowLimit:
      limitTmp0 = "ShadowLimit%s+0"%tc
      limitTmp1 = "ShadowLimit%s+1"%tc
    else:
      limitTmp0 = stmp+0
      limitTmp1 = stmp+1

    module.addInst("s_sub_u32",  sgpr(limitTmp0), sgpr("Tensor2dSize%s"%tc), sgpr(tileStart+0), "sub tileStart")
    module.addInst("s_subb_u32", sgpr(limitTmp1), sgpr("Tensor2dSize%s+1"%tc), sgpr(tileStart+1), "sub tileStart")

    if self.use64bShadowLimit:
      # Set initial buffer limit
      # if the limit is >64bit, incrementSrd decrements the shadow as the SRD increments,
      # and when we get within 32-bit we start to step down the SRD
      # if the limit is <32bits, set it accurately here:
      # Note lshl_b64 the higher-numbered SGPR has the upper 32-bits
      module.addInst("s_lshl_b64", sgpr("ShadowLimit%s"%tc,2),  sgpr("ShadowLimit%s"%tc,2), \
          hex(log2(tP["bpe"])), "Set limit to use bytes")
      if prePad:
        module.addInst("s_add_u32",  sgpr("ShadowLimit%s+0"%tc), sgpr("ShadowLimit%s+0"%tc), prePad, "extend limit for pre-pad")
        module.addInst("s_addc_u32", sgpr("ShadowLimit%s+1"%tc), sgpr("ShadowLimit%s+1"%tc), 0, "extend limit for pre-pad")

      if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
        module.addInst("s_add_u32",  sgpr("ShadowLimit%s+0"%tc), sgpr("ShadowLimit%s+0"%tc), self.buff_load_inst_offset_max, "extend limit for directToLDS instruction offset")
        module.addInst("s_addc_u32", sgpr("ShadowLimit%s+1"%tc), sgpr("ShadowLimit%s+1"%tc), 0, "extend limit for directToLDS instruction offset")

      module.addInst("s_cmp_eq_u32", sgpr("ShadowLimit%s+1"%tc), 0, "are we within 2^32?")
      module.addInst("s_cselect_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "BufferLimit", "Move shadow to real if we are within 2^32")
    else:
      # put limit directly into SRD:
      module.addInst("s_lshl_b32", sgpr("Srd%s+2"%tc), sgpr(stmp+0), hex(log2(tP["bpe"])), "Set limit to use bytes")
      module.addInst("s_add_u32",  sgpr("Srd%s+2"%tc), sgpr("Srd%s+2"%tc), prePad, "extend limit for pre-pad")

    # Apply any high-order address components to the tileStart and eventually the SRD - batch idx for batched gemm
    if kernel["ProblemType"]["StridedBatched"]:
      numDim = len(indices)
      wg=2 # TODO - refactor since only WG2 is supported and this is always batch
      for i in range(1, numDim):
        idx = indices[i]
        if idx == kernel["ProblemType"]["Index0"] \
            or idx == kernel["ProblemType"]["Index1"] \
            or idx in kernel["ProblemType"]["IndicesSummation"] \
            or isPackedIndex(kernel, idx):
              continue # these will be captured in GRO not the SRD (or other summations are always 0)
        else:
          assert(wg==2) # can only have one wg2 with a batch. Other dimensions should be packed into wg0/wg1
          stride = "Stride%s%s"%(tc,self.indexChars[tP['ia'][i]])
          if not wroteTileStart:
            module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(stride), sgpr("WorkGroup2"), "Stride*WG"))
            wroteTileStart = True
          else:
            module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), sgpr(stride), sgpr("WorkGroup2"), "Stride*WG"))
            module.addInst("s_add_u32",  sgpr(tileStart+0), sgpr(tileStart+0), sgpr(stmp+0), "accum wg term to tilestart")
            module.addInst("s_addc_u32", sgpr(tileStart+1), sgpr(tileStart+1), sgpr(stmp+1), "accum wg term to tilestart")
          wg+=1

    # Add the tile start to the SRD
    if wroteTileStart:
      module.addCode(scalarStaticMultiply(sgpr(tileStart,2), sgpr(tileStart,2), bpe, None, "tileStart *= BPE"))
      module.addInst("s_add_u32",  sgpr("Srd%s+0"%tc), sgpr("Address%s+0"%tc), sgpr(tileStart+0), "SRD base = Address+ tileStart0")
      module.addInst("s_addc_u32", sgpr("Srd%s+1"%tc), sgpr("Address%s+1"%tc), sgpr(tileStart+1), "SRD base = Address+ tileStart1")
    else:
      module.addInst("s_mov_b32", sgpr("Srd%s+0"%tc), sgpr("Address%s+0"%tc), "init SRD base address (lower )" )
      module.addInst("s_mov_b32", sgpr("Srd%s+1"%tc), sgpr("Address%s+1"%tc), "init SRD base address (upper) + other fields" )

    # self.groOffsetInMacroTile == 1 case,  pre-pad is already subtracted from AddressA/B
    if prePad and self.groOffsetInMacroTile == 0:
      module.addInst("s_sub_u32",  sgpr("Srd%s+0"%tc), sgpr("Srd%s+0"%tc), prePad, "pre-pad to make room for possible pointer shift")
      module.addInst("s_subb_u32",  sgpr("Srd%s+1"%tc), sgpr("Srd%s+1"%tc), 0, "pre-pad to make room for possible pointer shift")

    if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
      module.addInst("s_sub_u32",  sgpr("Srd%s+0"%tc), sgpr("Srd%s+0"%tc), self.buff_load_inst_offset_max, "make room for directToLDS instruction offset")
      module.addInst("s_subb_u32",  sgpr("Srd%s+1"%tc), sgpr("Srd%s+1"%tc), 0, "make room for directToLDS instruction offset")

    module.addInst("s_mov_b32", sgpr("Srd%s+3"%tc), "Srd127_96", "Set bits 127_96 in SRD")

    #if tP["isB"]:
    #  module.addCode(self.getCmpAssert(self.asmAssert.ne, sgpr("WorkGroup1"), 0xA))

    if kernel["CheckDimOverflow"]>=2:
      # double-check to make sure the SRD limit is inside the allowed tensor:
      #   - compute size of tensor in elements (including all dimensions)
      #   - subtract the SRD base and SRD buffer limit
      #   - Make sure the 64bit result is >0
      module.addInst("s_lshl_b64", sgpr(stmp,2), sgpr("Tensor2dSize%s"%tc,2), log2(bpe), "tensor size in bytes")
      module.addInst("s_add_u32",  sgpr(stmp+0), sgpr(stmp+0), sgpr("Address%s+0"%tc), "add start ptr to compute tensor%s bot-right"%tc)
      module.addInst("s_addc_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("Address%s+1"%tc), "add start ptr to compute tensor%s bot-right"%tc)
      module.addInst("s_sub_u32",  sgpr(stmp+0), sgpr(stmp+0), sgpr("Srd%s+0"%tc), "sub SRD base")
      module.addInst("s_subb_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("Srd%s+1"%tc), "sub SRD base")
      if self.use64bShadowLimit:
        module.addInst("s_sub_u32", sgpr(stmp+0), sgpr(stmp+0), sgpr("ShadowLimit%s+0"%tc), "sub buffer size")
        module.addInst("s_subb_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("ShadowLimit%s+1"%tc), "sub buffer size")
      else:
        module.addInst("s_sub_u32",  sgpr(stmp+0), sgpr(stmp+0), sgpr("Srd%s+2"%tc), "sub buffer limit")

      module.addCode(self.getCmpAssert(self.asmAssert.eq, sgpr(stmp+1), 0))  # must be 0 or we are way OOB
      module.addCode(self.getCmpAssert(self.asmAssert.ge_u32, sgpr(stmp+0), 0)) # diff greater than zero
      if 0 and tP["isB"]:
        t = self.vgprPool.checkOut(1, "t", self.preventVgprOverflowDuringNewTile)
        module.addInst("s_add_u32", sgpr(stmp+0), sgpr("WorkGroup1"), sgpr("WorkGroup2"), "bozo, debug")
        module.addInst("v_mov_b32", vgpr(t), 0x54, "")
        module.addCode(self.getCmpAssert(self.asmAssert.ne, sgpr(stmp+0), vgpr(t) ))
        self.vgprPool.checkIn(t)

    return module

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  def graAddresses(self, kernel, tP):
    module = Code.Module("graAddresses")
    tc = tP["tensorChar"]
    graIdx = 0

    if kernel["BufferLoad"]:
      # maxAddrSgpr = size[n] * stride[n-1]
      module.addComment0("max read offset = size[n] * stride[n-1]")

      module.addCode(self.computeLoadSrd(kernel, tP, tc, kernel["ProblemType"]["IndexAssignments%s"%tc], tP["bpe"]))

      #module.addCode(self.getBomb(0x13)) # after addresses and SRD set
    else:
      tmp = self.vgprPool.checkOut(2, "tmp", self.preventVgprOverflowDuringNewTile)
      module.addInst("v_mov_b32", vgpr(tmp+0), sgpr("Address%s+0"%tP["tensorChar"]), "" )
      module.addInst("v_mov_b32", vgpr(tmp+1), sgpr("Address%s+1"%tP["tensorChar"]), "" )
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):

              comment = "gRA%s_%u_%u_%u_%u = addr%s+grO%s_%u_%u_%u_%u" \
                  % (tP["tensorChar"], para, sPara, perp, sPerp, \
                  tP["tensorChar"], tP["tensorChar"], \
                  para, sPara, perp, sPerp )
              module.addInst("_v_add_co_u32", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                  self.vcc, \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                  vgpr(tmp+0), \
                  comment+" (lower)")
              module.addInst("_v_addc_co_u32", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  self.vcc, \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  vgpr(tmp+1), \
                  self.vcc, \
                  comment+" (upper)")
              #module.addCode(dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx))))
              #module.addCode(dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx))))
              graIdx += self.rpga
      #module.addInst("s_endpgm")
      self.vgprPool.checkIn(tmp)

    return module

  ##############################################################################
  # Global Read Addresses: Increments
  # Define graIncrements, called once for each summation
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    module = Code.Module("graIncrements")
    tc = tP["tensorChar"]

    dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
    loopChar = self.indexChars[dimIdx]

    stride = self.strideRef(tc, dimIdx)
    isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]

    #print (tc, ": loopIdx=", loopIdx, "dimIdx=", dimIdx, "strideIdx=", strideIdx)

    gsu = 1
    if kernel["GlobalSplitU"] > 1:
      gsu = kernel["GlobalSplitU"]

    assert(self.unrollIdx == kernel["ProblemType"]["NumIndicesSummation"]-1)
    if loopIdx==self.unrollIdx:
      if self.globalReadIncsUseVgpr:
        tmpSgpr = self.getTmpSgpr(2).idx()
        module.addInst("s_mul_i32", sgpr(tmpSgpr+0), \
            "DepthU*%d"%(gsu*tP["bpe"]), stride, \
            "incr%s%s = %s*DepthU*bpe (unrollIdx)"%(tc, loopChar, stride) )
        # TODO - this should be mul-H??
        module.addInst("s_mov_b32", \
            sgpr(tmpSgpr+1), \
            hex(0), \
            "(carry)")
        module.addInst("v_mov_b32", \
            vgpr("GlobalReadIncs%s+%u+0"%(tc, 2*loopIdx)), \
            sgpr(tmpSgpr+0), \
            "" )
        module.addInst("v_mov_b32", \
            vgpr("GlobalReadIncs%s+%u+1"%(tc, 2*loopIdx)), \
            sgpr(tmpSgpr+1), \
            "" )
      else: # not globalReadIncsUseVgpr, ie use SGPR

        m = "DepthU*Bpe%s"%(tc)
        if gsu>1:
          m += "*%d"%gsu

        if isMirrorIdx:
          m = "-%s"%(m)

        # multiply by stride, optimizing if unit stride
        if self.isConstUnitStride(stride):
          module.addInst("s_mov_b32", sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), m, \
              "incr%s (unrollIdx)"%(tc) )
        else:
          module.addInst("s_mul_i32", sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), \
              m, stride, \
              "incr%s unrollIdx)"%(tc) )
    else:
      # other summation
      if self.globalReadIncsUseVgpr:
        printExit("NumIndicesSummation=%u not yet supported in assembly unless globalReadIncsUseVgpr==0" \
            % kernel["ProblemType"]["NumIndicesSummation"] )
      else:
        graInc = "GlobalReadIncs%s+%u"%(tc, loopIdx)
        # subtract increments done by the inner iterations
        # may be negative:
        loopIdxPrev = loopIdx + 1
        dimIdxPrev    = kernel["ProblemType"]["IndicesSummation"][loopIdxPrev] # dimension index
        loopCharPrev  = self.indexChars[dimIdxPrev]
        stridePrev = self.strideRef(tc, dimIdxPrev)
        isMirrorIdxPrev = dimIdxPrev in kernel["ProblemType"]["MirrorDims%s"%tc]

        module.addComment1("compute globalReadInc for higher-level loop")

        tmpSgpr = self.getTmpSgpr(3).idx()
        # Summations always appear in both A and B, can compute number of iterations just once:
        if loopIdxPrev==self.unrollIdx:
          loopCounterName= self.loopCounterName(kernel, self.unrollIdx)
          if tP["isA"]:
            quotient = loopCounterName
            dividend = "SizesSum+%u"%self.unrollIdx
            divisor = kernel["DepthU"]
            if self.noTailLoop and kernel["AssertSummationElementMultiple"] % kernel["DepthU"] != 0:
              # round up SizesSum/DepthU for noTailLoop case
              module.addInst("s_add_i32", sgpr(quotient), (divisor - 1), sgpr(dividend), \
                  "round up SizeSum / DepthU" )
              module.addCode(scalarStaticDivideAndRemainder(quotient, None, quotient, \
                          divisor, tmpSgpr+2, 0))
            else:
              module.addCode(scalarStaticDivideAndRemainder(quotient, None, dividend, \
                          divisor, tmpSgpr+2, 0))

            if kernel["GlobalSplitU"] > 1:
              module.addCode(self.calculateLoopNumIterGsu(kernel, loopCounterName, tmpSgpr))

            module.addInst("s_mul_i32", sgpr(loopCounterName), sgpr(loopCounterName), \
                      kernel["GlobalSplitU"]*kernel["DepthU"], \
                      "=loopCounterName*DepthU")
          module.addInst("s_mul_i32", sgpr(graInc), stridePrev, sgpr(loopCounterName), \
                "tmp <- stride%s%s * myWgUnrollIters" %(tc, loopCharPrev))
        else:
          module.addInst("s_mul_i32", sgpr(graInc), stridePrev, self.sizeRef(dimIdxPrev), \
                "tmp <- stride%s%s * size%s%s" %(tc, loopCharPrev, tc, loopCharPrev))

        # subtract amount that previous inner loop will have already incremented:
        # graInc is used as temp for the prev loop calc
        if isMirrorIdx and isMirrorIdxPrev:
          module.addInst("s_sub_i32", sgpr(graInc), \
              sgpr(graInc), \
              stride, \
              "incr%s%s = <prev-incs> - stride%s%s"%(tc, loopChar, tc, loopChar) )
        elif isMirrorIdx:
          module.addInst("s_add_i32", sgpr(graInc), \
              stride, \
              sgpr(graInc), \
              "incr%s%s = stride%s%s + <prev-incs>"%(tc, loopChar, tc, loopChar) )
          module.addInst("s_sub_i32", sgpr(graInc), \
              0, \
              sgpr(graInc), \
              "incr%s%s = - (stride%s%s + <prev-incs>)"%(tc, loopChar, tc, loopChar) )
        elif isMirrorIdxPrev:
          module.addInst("s_add_i32", sgpr(graInc), \
              stride, \
              sgpr(graInc), \
              "incr%s%s = stride%s%s + <prev-incs>"%(tc, loopChar, tc, loopChar) )
        else:
          module.addInst("s_sub_i32", sgpr(graInc), \
              stride, \
              sgpr(graInc), \
              "incr%s%s = stride%s%s - <prev-incs>"%(tc, loopChar, tc, loopChar) )

        module.addInst("s_lshl_b32", \
            sgpr(graInc), \
            sgpr(graInc), \
            "Bpe%sLog2"%tc,
            "<- scale by bpe")

        if 0 and tP["isB"] and loopIdx==0:
          module.addCode(self.getBomb())
          #module.addCode(self.getCmpAssert(self.asmAssert.ne, sgpr("WorkGroup1"),0))

    #module.addCode(dump(vgpr("GlobalReadIncs%s"%tP["tensorChar"])))
    #module.addInst("s_endpgm")
    #if tP["isB"]:
    #  module.addCode(self.getBomb(0x100))
    return Code.Module("graIncrements (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  def lwaTileAssignment(self, kernel, tP):
    module = Code.Module("lwaTileAssignment")
    module.addComment0("lwaTileAssignment%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["lwoT"])))
    return module

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    module = Code.Module("lwaUnrollAssignment")
    uReg = tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"]
    module.addComment0("lwaUnrollAssignment%s = %s" % (tP["tensorChar"], vgpr(uReg)))
    if kernel.enabledSplitLDS and kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      if self.inTailLoop:
        subIterReg = self.vgprPool.checkOut(1, "subIterReg")
        module.addComment0("Each wg writes 1/%u of G2L data to LDS"%kernel["DepthULdsDivisor"])
        module.addInst("v_lshrrev_b32", vgpr(subIterReg), log2(kernel["_DepthULds"]), vgpr(uReg), "sub_G2L_idx = uIdx / DepthU_Compute")
        module.addInst("v_and_b32", vgpr(uReg), vgpr(uReg), kernel["_DepthULds"]-1, "unrollIdx = unrollIdx % DepthU_Compute")
        tP["gpr"]["subIterReg"] = subIterReg
      else:
        module.addComment0("Each thd writes 1/%u of G2L data to LDS"%kernel["DepthULdsDivisor"])
        module.addInst("v_lshrrev_b32", vgpr(uReg), log2(kernel["DepthULdsDivisor"]), vgpr(uReg), "sub_G2L_idx = uIdx / DepthULdsDivisor")
    return module

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  # uDu: which part of G2L buffer to write to LDS
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP, uDu=0):
    module = Code.Module("lwaFirstOffset")
    tc = tP["tensorChar"]
    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    #"lwFOA = lwA%s + lwA%s*MT%s" \
    #    % (tP["tileChar"], self.unrollChar, tP["tileChar"])
    uReg = tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"]
    if kernel["LocalWriteUseSgpr%s"%tc]:
      destVgpr = self.vgprPool.checkOut(1, "destVgpr", self.preventVgprOverflowDuringNewTile)
    else:
      destVgpr = "LocalWriteAddr%s"%tc

    if kernel["UnrollMajorLDS%s" % tc]:
      lds_stride = kernel["_DepthULds"] + LdsPad
      module.addInst("v_mul_u32_u24", vgpr(destVgpr), hex(lds_stride), vgpr(tP["gpr"]["lwoT"]), \
          "lw%s%s**(DepthU_Compute + PAD)"%(tP["tensorChar"], self.unrollChar))
      module.addInst("_v_add_lshl_u32", vgpr(destVgpr), vgpr(uReg), vgpr(destVgpr), hex(log2(tP["bpe"])), \
          "lwFO%s = (lw%s%s + lw%s%s*(DepthU+PAD))*bpe" % (tc, tc, tc, tc, self.unrollChar) )
    else:
      lds_stride = kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad
      module.addInst("v_mul_u32_u24", vgpr(destVgpr), hex(lds_stride), vgpr(uReg), \
          "lw%s%s**(MT%s + PAD)"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
      module.addInst("_v_add_lshl_u32", vgpr(destVgpr), vgpr(tP["gpr"]["lwoT"]), vgpr(destVgpr), hex(log2(tP["bpe"])), \
          "lwFO%s = (lw%s%s + lw%s%s*(MT%s+PAD))*bpe" % (tc, tc, tc, tc, self.unrollChar, tP["tileChar"]) )

    # LdsBlockSizePerPad: add padding
    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
      tmpVgpr = self.vgprPool.checkOut(2)
      tmpSgpr = self.getTmpSgpr(1).idx()
      module.addCode(vectorStaticDivide(uReg, destVgpr, kernel["LdsBlockSizePerPad%s"%tc], tmpVgpr, tmpSgpr, \
        "padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
      module.addCode(staticMultiply(vgpr(uReg), vgpr(uReg), kernel["LdsPad%s"%tc] * tP["bpe"], sgpr(tmpSgpr), \
        "padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
      module.addInst("_v_add_u32", vgpr(destVgpr), vgpr(uReg), vgpr(destVgpr), \
        "add padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc]))
      self.vgprPool.checkIn(tmpVgpr)

    if tP["isB"]:
      if kernel["LdsOffsetB"] != 0: # LdsOffsetB can be 0 if DirectToVgprA is enabled
        module.addInst("_v_add_co_u32", \
            vgpr(destVgpr), \
            self.vcc, \
            hex(kernel["LdsOffsetB"]*tP["bpe"]), \
            vgpr(destVgpr), \
            "lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u*%u" % (tP["tileChar"], \
            self.unrollChar, tP["tileChar"], kernel["LdsOffsetB"], self.bpeAB) )

    self.vgprPool.checkIn(tP["gpr"]["lwoT"])
    tP["gpr"]["lwoT"] = None
    if kernel["GlobalSplitU"] > 1:
      self.vgprPool.checkIn(tP["gpr"]["uReg2"])
      tP["gpr"]["uReg2"] = None
    #LSC_ * LSP_
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    validWIPerLoad     = kernel[tP["lsc"]] * kernel[tP["lsp"]]//tP["glvw"]
    validBytesPerLoad  = kernel[tP["lsc"]] * kernel[tP["lsp"]] * numBytesPerElement
    maxBytesPerLoad    = kernel["NumThreads"] * tP["glvw"] * numBytesPerElement

    if kernel["WaveSeparateGlobalRead%s"%tc]:
      validBytesPerLoad *= (kernel["NumThreads"] // self.kernel["WavefrontSize"])

    assert (validBytesPerLoad <= maxBytesPerLoad)
    assert (kernel[tP["lsc"]] * kernel[tP["lsp"]] % tP["glvw"] == 0)

    if validBytesPerLoad != maxBytesPerLoad:
      tmpSgpr = self.getTmpSgpr(1).idx()
      module.addInst("s_mov_b32", sgpr(tmpSgpr), validWIPerLoad, \
          "lsc*lsp=%u*%u"%(kernel[tP["lsc"]],kernel[tP["lsp"]] ))
      module.addInst("v_cmp_lt_u32", \
          self.vcc, \
          vgpr("Serial"), \
          sgpr(tmpSgpr), \
          "fractional: ensure tid < global read tile elements")
      tmpVgpr = self.vgprPool.checkOut(1, "tmpVgpr", self.preventVgprOverflowDuringNewTile)
      module.addInst("v_mov_b32", vgpr(tmpVgpr), hex(self.LdsOOB), "")
      module.addInst("v_cndmask_b32", \
                  vgpr(destVgpr), \
                  vgpr(tmpVgpr), \
                  vgpr(destVgpr), \
                   self.vcc, \
                   "Mask load so out-of-gr-tile bounds returns 0")
      self.vgprPool.checkIn(tmpVgpr)

    elif self.inTailLoop and kernel.enabledSplitLDS: # where (DepthU for global read) != (DepthU for compute)
      tmpSgpr = self.getTmpSgpr(1).idx()

      # only for TN tensor + TN lds layout
      assert tP["tlu"] == 0
      module.addInst("v_cmp_eq_u32",self.vcc, vgpr(tP["gpr"]["subIterReg"]), uDu, "if sub_g2l_idx == %u ?"%uDu)

      ldsOOB = self.vgprPool.checkOut(1, "lds OOB addr", self.preventVgprOverflowDuringNewTile)
      module.addInst("v_mov_b32", vgpr(ldsOOB), hex(self.LdsOOB), "lds OOB address")
      module.addInst("v_cndmask_b32", \
                  vgpr(destVgpr), \
                  vgpr(ldsOOB), \
                  vgpr(destVgpr), \
                   self.vcc, \
                   "Mask threads not belonging to current sub_g2l_idx by assigning OOB")
      self.vgprPool.checkIn(ldsOOB)

    if kernel["LocalWriteUseSgpr%s"%tc]:
      # TODO: Can refactor code above to Compute this directly:
      module.addInst("v_readfirstlane_b32", \
          sgpr("LocalWriteAddr%s"%tc), \
          vgpr(destVgpr), \
          "Copy lds write address VGPR to SGPR")
      self.vgprPool.checkIn(destVgpr)

    self.vgprPool.checkIn(tP["gpr"]["uReg"])
    tP["gpr"]["uReg"] = None
    if "subIterReg" in tP["gpr"]:
      if tP["gpr"]["subIterReg"] is not None:
        self.vgprPool.checkIn(tP["gpr"]["subIterReg"])
      tP["gpr"]["subIterReg"] = None
    # dump lds write offsets
    #if tP["isA"]:
      #module.addCode(self.dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"])))
      #module.addCode(self.getBomb(-40))
    # do not generate local write address code if DirectToVgpr is enabled
    return Code.Module("lwaUnrollAssignment (Empty)") if self.dontAppendCode or kernel["DirectToVgpr%s"%tc] else module

  ##############################################################################
  # Local Read Addresses: Tile Assignment
  ##############################################################################
  def lraTileAssignment(self, kernel, tPA, tPB):
    module = Code.Module("lraTileAssignment")

    component = Component.LraTileAssignment.find(self)

    tP0 = tPA if tPB["tile01Idx"] else tPB
    tP1 = tPB if tPB["tile01Idx"] else tPA

    if component:
      # do not generate local read code if DirectToVgpr is enabled
      tc = tP0["tensorChar"]
      if not kernel["DirectToVgpr%s"%tc]:
        module.addCode(component(self, kernel, tP0))
      # do not generate local read code if DirectToVgpr is enabled
      tc = tP1["tensorChar"]
      if not kernel["DirectToVgpr%s"%tc]:
        module.addCode(component(self, kernel, tP1))

    return module

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    module = Code.Module("lraFinalOffset")

    # do not generate local read code if DirectToVgpr is enabled
    tc = tP["tensorChar"]
    if kernel["DirectToVgpr%s"%tc]:
      return module

    # allocate resources
    sgid    = self.vgprPool.checkOut(1) # quotient
    rReg    = self.vgprPool.checkOut(1) # remainder, unused here
    tmpVgpr = self.vgprPool.checkOutAligned(2, 2,"tmpVgpr")
    tmpSgpr = self.getTmpSgpr(1).idx()

    # constant
    tc          = tP["tensorChar"]
    tile01      = tP["tile01Idx"]
    LdsPad      = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
    divisor     = kernel["SubGroup0"] * kernel["SubGroup1"]
    mtAddPad    = kernel["MacroTile%u" % tile01] + LdsPad

    # generate instruction
    module.addCode(vectorStaticDivide(sgid, "Serial", divisor, tmpVgpr, tmpSgpr, \
      "LSU offset: sgid = Serial / subGroup(%u)" % divisor))
    module.addInst("s_mov_b32", sgpr(tmpSgpr), mtAddPad, \
      "LSU offset: stride = MT%u(%u) + PAD%u(%u)" % (tile01, kernel["MacroTile%u" % tile01], tile01, LdsPad))
    module.addInst("v_mul_lo_u32", vgpr(sgid), sgpr(tmpSgpr), vgpr(sgid), \
      "LSU offset: lsuoffset = sgid*(MT%u+PAD)"%tile01)
    if not kernel["EnableMatrixInstruction"] and kernel["VectorWidth"] > 1:
      module.addCode(staticMultiply(vgpr(tP["gpr"]["lro"]), vgpr(tP["gpr"]["lro"]), kernel["VectorWidth"], sgpr(tmpSgpr), \
      "Final Offset: lr%sOffset * VW" % tc))

    # final offset
    finalVgpr = vgpr("LocalReadAddr%s"%tc)
    if (kernel["DirectToLds%s" % tc] and \
        kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"] > 4):
      # DirectToLds + DGEMM case
      # use bpr for LSU offset instead of bpe (DirectToLds needs _ds_load_b32)
      module.addInst("v_lshlrev_b32", vgpr(sgid), hex(log2(self.bpr)), vgpr(sgid),  \
              "LSU offset: lsuoffset = lsuoffset * bpr");
      module.addInst("v_lshlrev_b32", vgpr(tP["gpr"]["lro"]), hex(log2(tP["bpe"])), vgpr(tP["gpr"]["lro"]),  \
              "Final Offset: offset = (lro%s*VW)*bpe+lsuoffset*bpr" % tile01);
      module.addInst("_v_add_u32", finalVgpr, vgpr(sgid), vgpr(tP["gpr"]["lro"]), "")
      # need magic offset calc here (after final offset)
      # offset calculation for TLU=1 when glvw * bpe * wavefrontsize > 256
      # x2/x4 directToLds stores 8/16 bytes into LDS like below
      # address offset in LDS in bytes
      # DWORD# written by LDS_DMA
      #  address offset in LDS (byte offset)
      #  0    4    8    12    16   20   24   28   32   36   40   44    48    52   56   60
      #  data dword#:
      #  0    4    8    12    2    6    10   14    1   5    9    13     3    7    11   15
      #  Noffset calculation for VW =1 (BPe=8) / VW =2 (BPE=4)
      #  use direcToLds for best VW and GRVW case; other cases requires bit more lane manipulation.
      #  offset calculation  for B might benefit from some optimization.
      #  offset calculation for x2/x4  is basically manipulation lane offset based on layout
      tmp1    = self.vgprPool.checkOut(1,"tmp1")
      tmp2    = self.vgprPool.checkOut(1,"tmp2")
      if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 8):
        # (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
        module.addInst("v_and_b32", vgpr(tmp1), "0x4", finalVgpr, "magic offset calc")
        module.addInst("v_lshlrev_b32", vgpr(tmp1),  hex(3), vgpr(tmp1), "")
        module.addInst("v_and_b32", vgpr(tmp2), "0x38", finalVgpr, "")
        module.addInst("v_lshrrev_b32", vgpr(tmp2),  hex(1), vgpr(tmp2), "")
        module.addInst("v_or_b32", vgpr(tmp1), vgpr(tmp1), vgpr(tmp2), "")
        module.addInst("v_and_b32", finalVgpr, "0xffffffc3", finalVgpr, "")
        module.addInst("v_or_b32", finalVgpr, finalVgpr, vgpr(tmp1), "")
      else:  #if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 16):  # most preferred case
        # (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
        module.addInst("v_and_b32", vgpr(tmp1), "0x4", finalVgpr, "magic offset calc")
        module.addInst("v_lshlrev_b32", vgpr(tmp1),  hex(3), vgpr(tmp1), "")
        module.addInst("v_and_b32", vgpr(tmp2), "0x8", finalVgpr, "")
        module.addInst("v_lshlrev_b32", vgpr(tmp2),  hex(1), vgpr(tmp2), "")
        module.addInst("v_or_b32", vgpr(tmp1), vgpr(tmp1), vgpr(tmp2), "")
        module.addInst("v_and_b32", vgpr(tmp2), "0x30", finalVgpr, "")
        module.addInst("v_lshrrev_b32", vgpr(tmp2),  hex(2), vgpr(tmp2), "")
        module.addInst("v_or_b32", vgpr(tmp1), vgpr(tmp1), vgpr(tmp2), "")
        module.addInst("v_and_b32", finalVgpr, "0xffffffc3", finalVgpr, "")
        module.addInst("v_or_b32", finalVgpr, finalVgpr, vgpr(tmp1), "")
      # TODO: cover other cases

      # another address conversion for DirectToLds + NumLoadsCoalesced > 1
      newModule, dummy = self.lraOffsetConversionForDTLandNLC(kernel, tP, offset_val=0, generateAsm=True, \
                                                              finalVgpr=finalVgpr, tmp1=tmp1, tmp2=tmp2)
      module.addCode(newModule)

      self.vgprPool.checkIn(tmp1)
      self.vgprPool.checkIn(tmp2)
    else:
      module.addInst("_v_add_lshl_u32", finalVgpr, vgpr(sgid), vgpr(tP["gpr"]["lro"]), hex(log2(tP["bpe"])), \
        "Final Offset: offset = (lro%s*VW+lsuoffset)*bpe" % tile01 )

    # LdsBlockSizePerPad: add padding
    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] !=0:
      module.addCode(vectorStaticDivide(rReg, "LocalReadAddr%s"%tc, kernel["LdsBlockSizePerPad%s"%tc], tmpVgpr, tmpSgpr, \
        "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
      module.addCode(staticMultiply(vgpr(rReg), vgpr(rReg), kernel["LdsPad%s"%tc] * tP["bpe"], sgpr(tmpSgpr), \
        "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
      module.addInst("_v_add_u32", vgpr("LocalReadAddr%s"%tc), vgpr(rReg), vgpr("LocalReadAddr%s"%tc), \
        "Final Offset: add padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc]))

    # release resources
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(sgid)
    self.vgprPool.checkIn(rReg)
    self.vgprPool.checkIn(tP["gpr"]["lro"])

    return module

  ##############################################################################
  # Local Read Addresses offset conversion for DTL + NLC > 1
  ##############################################################################
  def lraOffsetConversionForDTLandNLC(self, kernel, tP, offset_val, generateAsm=False, \
                                      finalVgpr=None, tmp1=None, tmp2=None):
    module = Code.Module("lraOffsetConversionForDTLandNLC")
    # another address conversion for DirectToLds + NumLoadsCoalesced > 1
    divisorName = tP["lvc"]
    divisor = kernel[divisorName]
    width = kernel["WavefrontSize"] if tP["tlu"] else kernel["DepthU"]
    if divisor < width:
      # DirectToLds + above conditions, rotate offset_val bits to adjust LDS offset
      lowerScale = tP["nrc"]
      upperScale = (kernel["WavefrontSize"] // divisor)
      # bit rotation necessary only when nrc > 1
      if lowerScale > 1:
        tile01 = tP["tile01Idx"]
        rightShift = int(log2(lowerScale)) # assuming power of 2
        leftShift = int(log2(upperScale)) # assuming power of 2
        line = kernel["MacroTile%u" % tile01] if tP["tlu"] else kernel["DepthU"]
        ldsLineSize = line * tP["bpe"] // lowerScale
        maskBitsLow = (lowerScale - 1) * ldsLineSize
        maskBitsHigh = (upperScale - 1) * lowerScale * ldsLineSize
        maskBitsAll = (maskBitsLow | maskBitsHigh)

        # offset_val conversion
        low = offset_val & maskBitsLow
        high = offset_val & maskBitsHigh
        low <<= leftShift
        high >>= rightShift
        val = low | high
        offset_val = (offset_val & (~maskBitsAll)) | val

        # generate asm code
        if generateAsm:
          tmpSgpr2 = self.getTmpSgpr(1).idx()
          module.addInst("v_and_b32", vgpr(tmp1), hex(maskBitsLow), finalVgpr, \
            "Offset rotation for DirectToLds + %s > 1"%tP["lsc"])
          module.addInst("v_and_b32", vgpr(tmp2), hex(maskBitsHigh), finalVgpr, "")
          module.addInst("v_lshlrev_b32", vgpr(tmp1), hex(leftShift), vgpr(tmp1), "")
          module.addInst("v_lshrrev_b32", vgpr(tmp2), hex(rightShift), vgpr(tmp2), "")
          module.addInst("v_or_b32", vgpr(tmp1), vgpr(tmp1), vgpr(tmp2), "")
          module.addInst("s_mov_b32", sgpr(tmpSgpr2), hex(maskBitsAll), "")
          module.addInst("v_not_b32", vgpr(tmp2), sgpr(tmpSgpr2), "")
          module.addInst("v_and_b32", finalVgpr, vgpr(tmp2), finalVgpr, "")
          module.addInst("v_or_b32", finalVgpr, vgpr(tmp1), finalVgpr, "")

    return module, offset_val

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    module = Code.Module("lraDeclareAddresses")
    if tP["isA"]:
      module.addComment0("N/A")

    else:
      # no local read code if DirectToVgpr is enabled
      # no need to generate add code if LdsOffset is 0
      if kernel["DirectToVgprB"] or kernel["LdsOffset%s"%tP["tensorChar"]] == 0:
        module = Code.Module("lraDeclareAddresses (Empty)")
      else:
        module.addInst("_v_add_co_u32", \
            vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
            self.vcc, \
            hex(kernel["LdsOffset%s"%tP["tensorChar"]]*tP["bpe"]), \
            vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
            " += LdsOffset%s (lower)"%tP["tensorChar"])
    return module

  ##############################################################################
  # openShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def openShadowInit(self, kernel):
    module = Code.Module("openShadowInit")
    module.addCode(Code.Label("ShadowInitStart", ""))
    return module

  ##############################################################################
  # closeShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def closeShadowInit(self, kernel):
    module = Code.Module("closeShadowInit")
    assert(self.doShadowInit and kernel["PrefetchGlobalRead"])

    module.addCode(self.checkLastIter(kernel))
    if kernel["SuppressNoLoadLoop"]:
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
      lastIterEnd = "LoopEnd%s"%loopChar
    else:
      lastIterEnd = "PrefetchGlobalLastIterEnd"

    # This branch could potentially be very far e.g. > SIMM16
    module.addComment1("after InitC, skip to end of prefetch last iter if numIter==0")
    # use positive offset only long jump
    module.addCode(self.longBranchScc1(lastIterEnd, positiveOnly=True))

    return module

  ##############################################################################
  # longBranch - 32 bit offset
  # s_branch class instructions take a label operand which is truncated to 16 bit
  # If the target label address offset is greater than 16 bits, then
  # we must use a longer 32 bit version.
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranch(self, label):
    module = Code.Module("longBranch label %s"%label)
    labelName = Code.Label.getFormatting(label)
    tmpSgpr = self.getTmpSgpr(3).idx()
    positiveLabel = Code.Label(self.labels.getUniqueNamePrefix("Positive"), "")
    module.addInst("s_getpc_B64", sgpr(tmpSgpr,2), "addr of next instr")
    module.addInst("s_add_i32",  sgpr(tmpSgpr+2), "%s"%labelName, hex(4), "target branch offset")
    module.addInst("s_cmp_ge_i32", sgpr(tmpSgpr+2), hex(0), "check positive or negative")
    module.addInst("s_cbranch_scc1 %s" % positiveLabel.getLabelName(), "jump when positive")

    # negative offset
    module.addInst("s_abs_i32",  sgpr(tmpSgpr+2), sgpr(tmpSgpr+2), "abs offset")
    module.addInst("s_sub_u32",  sgpr(tmpSgpr),   sgpr(tmpSgpr),   sgpr(tmpSgpr+2), "sub target branch offset")
    module.addInst("s_subb_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), 0, "sub high and carry")
    module.addInst("s_setpc_b64", sgpr(tmpSgpr,2), "branch to %s"%labelName)

    # positive offset
    module.addCode(positiveLabel)
    module.addInst("s_add_u32",  sgpr(tmpSgpr), sgpr(tmpSgpr), sgpr(tmpSgpr+2), "add target branch offset")
    module.addInst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), 0, "add high and carry")
    module.addInst("s_setpc_b64", sgpr(tmpSgpr,2), "branch to %s"%labelName)
    return module

  ##############################################################################
  # longBranchPositive - 32 bit offset (positive offset only)
  # s_branch class instructions take a label operand which is truncated to 16 bit
  # If the target label address offset is greater than 16 bits, then
  # we must use a longer 32 bit version.
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchPositive(self, label):
    module = Code.Module("longBranchPositive label %s"%label)
    labelName = Code.Label.getFormatting(label)
    tmpSgpr = self.getTmpSgpr(3).idx()
    module.addInst("s_getpc_B64", sgpr(tmpSgpr,2), "addr of next instr")
    module.addInst("s_add_i32",  sgpr(tmpSgpr+2), "%s"%labelName, hex(4), "target branch offset")

    # positive offset
    module.addInst("s_add_u32",  sgpr(tmpSgpr), sgpr(tmpSgpr), sgpr(tmpSgpr+2), "add target branch offset")
    module.addInst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), 0, "add high and carry")
    module.addInst("s_setpc_b64", sgpr(tmpSgpr,2), "branch to %s"%labelName)
    return module

  ##############################################################################
  # longBranchNegative - 32 bit offset (negative offset only)
  # s_branch class instructions take a label operand which is truncated to 16 bit
  # If the target label address offset is greater than 16 bits, then
  # we must use a longer 32 bit version.
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchNegative(self, label):
    module = Code.Module("longBranchNegative label %s"%label)
    labelName = Code.Label.getFormatting(label)
    tmpSgpr = self.getTmpSgpr(3).idx()
    module.addInst("s_getpc_B64", sgpr(tmpSgpr,2), "addr of next instr")
    module.addInst("s_add_i32",  sgpr(tmpSgpr+2), "%s"%labelName, hex(4), "target branch offset")

    # negative offset
    module.addInst("s_abs_i32",  sgpr(tmpSgpr+2), sgpr(tmpSgpr+2), "abs offset")
    module.addInst("s_sub_u32",  sgpr(tmpSgpr),   sgpr(tmpSgpr),   sgpr(tmpSgpr+2), "sub target branch offset")
    module.addInst("s_subb_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), 0, "sub high and carry")
    module.addInst("s_setpc_b64", sgpr(tmpSgpr,2), "branch to %s"%labelName)
    return module

  ##############################################################################
  # longBranchScc0 - 32 bit offset
  # Conditional branch to label when SCC == 0
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchScc0(self, label, positiveOnly=False, negativeOnly=False):
    module = Code.Module("longBranchScc0 label %s"%label)
    noBranchLabel = Code.Label(self.labels.getUniqueNamePrefix("NoBranch"), "")
    module.addInst("s_cbranch_scc1 %s" % noBranchLabel.getLabelName(), "Only branch on scc0")
    if positiveOnly:
      module.addCode(self.longBranchPositive(label))
    elif negativeOnly:
      module.addCode(self.longBranchNegative(label))
    else:
      module.addCode(self.longBranch(label))
    module.addCode(noBranchLabel)
    return module

  ##############################################################################
  # longBranchScc1 - 32 bit offset
  # Conditional branch to label when SCC == 1
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchScc1(self, label, positiveOnly=False, negativeOnly=False):
    module = Code.Module("longBranchScc1 label %s"%label)
    noBranchLabel = Code.Label(self.labels.getUniqueNamePrefix("NoBranch"), "")
    module.addInst("s_cbranch_scc0 %s" % noBranchLabel.getLabelName(), "Only branch on scc1")
    if positiveOnly:
      module.addCode(self.longBranchPositive(label))
    elif negativeOnly:
      module.addCode(self.longBranchNegative(label))
    else:
      module.addCode(self.longBranch(label))
    module.addCode(noBranchLabel)
    return module

  ##############################################################################
  # Initialize C
  ##############################################################################
  def initC(self, kernel):
    module = Code.Module("initC")
    module.addComment1("initC: remove C-tile %u-%u from pool"%(self.startVgprValuC, self.startVgprValuC+self.numVgprValuC))
    self.vgprPool.remove(self.startVgprValuC, self.numVgprValuC, "ValuC")
    numAccvgprs = self.totalAgprs
    self.agprPool.remove(0, numAccvgprs, "ValuC")
    module.addComment1("initC: remove AB-tile %u-%u from pool"%(self.startVgprValuA, self.lastValuAB))
    self.vgprPool.remove(self.startVgprValuA, self.lastValuAB - self.startVgprValuA, "ValuAB")
    numCVgpr = max(self.numVgprValuC, numAccvgprs)

    startNumCVgpr = 0
    if self.useInitAccVgprOpt:
      # init accvgpr opt. initialize only the last set of accvgpr instead of whole accvgpr
      numRegistersOut  = kernel["MIRegPerOut"]
      accs_per_wave    = kernel["MatrixInstM"] * kernel["MatrixInstN"] * kernel["MatrixInstB"] \
                         // self.kernel["WavefrontSize"] * numRegistersOut
      startNumCVgpr = numCVgpr - accs_per_wave

    if kernel["LdsInitCVgprs"]:
      tmpAddr = self.vgprPool.checkOut(1,"tmp vgpr for lds init C registers")
      module.addInst("v_mov_b32", vgpr(tmpAddr), self.LdsOOB, "set out-of-bound addr")

    for i in range(startNumCVgpr, numCVgpr):
      copyInsStr = "v_mov_b32" if self.numVgprValuC else "v_accvgpr_write"
      regStr = vgpr("ValuC+%u"%i) if self.numVgprValuC else accvgpr(i)
      if not kernel["LdsInitCVgprs"]:
        module.addInst(copyInsStr, regStr, hex(0), "initC")
      else:
        module.addInst("_ds_load_b32", regStr, vgpr(tmpAddr), "offset:0", "initC")

    if kernel["LdsInitCVgprs"]:
      self.vgprPool.checkIn(tmpAddr)

    return module

  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  def declareLoopNumIter(self, kernel):
    module = Code.Module("declareLoopNumIter")
    if self.unrollIncIsDepthU:
      if kernel["GlobalSplitU"] > 1:
        tmpSgpr = self.getTmpSgpr(3).idx()
        quotient = "UnrollLoopLastIter"
        dividend = self.loopSizeRef(kernel, self.unrollIdx) # sumSize
        divisor = kernel["DepthU"]
        module.addCode(scalarStaticDivideAndRemainder(quotient, None, dividend, divisor, tmpSgpr, 0))
        module.addCode(self.calculateLoopNumIterGsu(kernel, "UnrollLoopLastIter", tmpSgpr))
        module.addInst ("s_mul_i32", sgpr("UnrollLoopLastIter"), sgpr("UnrollLoopLastIter"), "DepthU", "scale")
      else:
        module.addInst ("s_mov_b32", sgpr("UnrollLoopLastIter"), self.loopSizeRef(kernel, self.unrollIdx), "init")

    return module

  ##############################################################################
  # Calculate and apply stagger offsets and edge
  # Output: Sets sgpr(StaggerRowMask)
  ##############################################################################
  def declareStaggerParms(self, kernel):
    module = Code.Module("declareStaggerParms")
    tmpSgpr = self.getTmpSgpr(2).idx()
    if self.staggerU:
      # this could be dynamic?
      if kernel["StaggerUMapping"] == 0:
        staggerInput = sgpr("WorkGroup0")
      elif kernel["StaggerUMapping"] == 1:
        staggerInput = sgpr("WorkGroup1")
      elif kernel["StaggerUMapping"] == 2:
        staggerInput = sgpr("WorkGroup2")
      elif kernel["StaggerUMapping"] == 3:
        # wgSerial = (nwg0*ngw1)*wg2 + (nwg0)*wg1 + wg0
        wgSerial = tmpSgpr
        tmp = tmpSgpr+1
        module.addInst("s_mul_i32", sgpr(wgSerial), sgpr("NumWorkGroups0"), sgpr("NumWorkGroups1"), \
          "wgSerial = (nwg0*ngw1)*wg2 + (nwg0)*wg1 + wg0")
        module.addInst("s_mul_i32", sgpr(wgSerial), sgpr(wgSerial), sgpr("WorkGroup2"), "")
        module.addInst("s_mul_i32", sgpr(tmp), sgpr("NumWorkGroups0"), sgpr("WorkGroup1"), "")
        module.addInst("s_add_u32", sgpr(wgSerial), sgpr(wgSerial), sgpr(tmp), "")
        module.addInst("s_add_u32", sgpr(wgSerial), sgpr(wgSerial), sgpr("WorkGroup0"), "")
        staggerInput = sgpr(wgSerial)
      elif kernel["StaggerUMapping"] == 4:
        staggerInput = -1

      module.addInst("s_and_b32", sgpr("StaggerUIter"), sgpr("OrigStaggerUIter"), \
                    staggerInput, \
                    "Compute actual stagger start for this tile")
      module.addInst("s_lshl_b32", sgpr("StaggerUIter"), sgpr("StaggerUIter"), \
                kernel["_staggerStrideShift"], "shift by StaggerUStride")
    return module

  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  def calculateStagger(self, kernel, tP):
    imod = Code.Module("calculateStagger")
    tc = tP["tensorChar"]

    if self.staggerU:
      assert (kernel["BufferLoad"])

      staggerTmp = self.getTmpSgpr(2).idx()

      #---
      imod.addComment1("SRDs += (StaggerUIter) * GlobalReadIncs%s+%u"% (tc, self.unrollIdx))

      # Calculate the stagger byte offset
      imod.addModuleAsFlatItems(self.s_mul_i64_i32(
                sgpr(staggerTmp), sgpr(staggerTmp+1), \
                sgpr("StaggerUIter"), sgpr("GlobalReadIncs%s+%u"%(tc, self.unrollIdx)), \
                " stagger byte offset"))

      # Amount of bytes to add to get back to start.
      # on the llop iteration which matches StaggerUIter, this offset added instead of GlobalReadInc
      imod.addModuleAsFlatItems(self.s_mul_i64_i32(sgpr("WrapU%s+0"%tc), sgpr("WrapU%s+1"%tc), \
                self.loopCounter(kernel, self.unrollIdx), sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                "Number of bytes accessed by the unroll loop"))

      imod.addInst("s_sub_u32", sgpr("WrapU%s+0"%tc),  \
                sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                sgpr("WrapU%s+0"%tc), \
                "remove one iteration")
      imod.addInst("s_subb_u32", sgpr("WrapU%s+1"%tc), \
                0, \
                sgpr("WrapU%s+1"%tc), \
                "remove one iteration")

      imod.addCode(self.incrementSrd(kernel, tP, sgpr(staggerTmp), sgpr(staggerTmp+1)))

      if tP["isB"]:
        # Convert passed in S' to S for easy loop comparison.  S=S-(PGR-1)'
        imod.addInst("s_add_u32", sgpr("StaggerUIter"), sgpr("StaggerUIter"), \
                  (2 if kernel["PrefetchGlobalRead"] else 1), \
                  "Subtract (PGR-1); StaggerUIter now contains target iteration to wrap")
    return imod

  ##############################################################################
  # Remove stagger offset (before tail loop)
  # |          |           |   |
  # |-- S'*I --|
  # |---------- W' --------|-I-|
  #           ^ current SRD pos
  # ^unrollLoopStart           ^tailLoopStart   (in summation0 dimension)

  #
  # S = sgprStaggerUIter = S+(PGR+1)'
  # W = sgprWrapU
  # PGR = kernel["PrefetchGlobalRead"]
  #
  # S' = StaggUIter that is passed into the kernel = -PGR+1+S
  # S'*I is also the global read offset (from unrollLoopStart) at unroll loop exit ?
  # I = GlobalReadIncs
  # W' = W

  # Need to move it to tailLoopStart

  # To compute position where tail loop should start:
  #  = W' - S'*I + I
  #  = W - (S+PGR+1)*I) + I
  #  = W - (S+PGR+1)*I + I
  #  = W - (S+2+PGR)*I
  ##############################################################################
  def removeStagger(self, kernel, tP):
    imod = Code.Module("removeStagger")
    if self.staggerU:
      tc = tP["tensorChar"]
      tmp = self.getTmpSgpr(2).idx()
      # might be able to refactor this to eliminate signed math
      imod.addInst("s_sub_i32", sgpr(tmp), 3 if kernel["PrefetchGlobalRead"] else 2, \
                  sgpr("StaggerUIter"), "")
      imod.addModuleAsFlatItems(self.s_mul_i64_i32(sgpr(tmp), sgpr(tmp+1), \
                  sgpr(tmp), sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                  "start offset S in bytes"))
      imod.addInst("s_sub_u32", sgpr(tmp), sgpr(tmp), sgpr("WrapU%s"%tc), "S - WrapU")
      imod.addInst("s_subb_u32", sgpr(tmp+1), sgpr(tmp+1), sgpr("WrapU%s+1"%(tc)), "S - WrapU")

      imod.addCode(self.incrementSrd(kernel, tP, sgpr(tmp), sgpr(tmp+1)))

    return imod

  ##############################################################################
  # Emit code to compute loop iterations for GSU.
  # See same function in KernelWriterSource.py for background explanation
  # This function is used to compute number of loop iters and also
  # for computing the global read increment for GSU case.
  # For multiple summation, the number of loop iterations needs to be reset
  # for each iteration so replicate the code in addr inc and at open of unroll loop

  # tmpSgpr is allocation of at least 3 tmpSgpr

  # Output: SGPR(destName) contains the number of unroll iterations for
  # this workgroup.
  ##############################################################################
  def calculateLoopNumIterGsu(self, kernel, destName, tmpSgpr):
    module = Code.Module("calculateLoopNumIterGsu")

    loopCounter = sgpr(destName)
    quotient = destName
    remainder = "GSUSumIdx+1" # numIterPerWgRemainder
    dividend = tmpSgpr+2 # numIterMyWg
    divisor = kernel["GlobalSplitU"]
    if log(divisor,2).is_integer():
      module.addInst("s_mov_b32", sgpr(dividend), loopCounter, "copy for divide IterGsu" )
      module.addCode(scalarStaticDivideAndRemainder(quotient, remainder, dividend, divisor, tmpSgpr, 1))
    else:
      qReg = self.vgprPool.checkOut(1,"qReg")
      rReg = self.vgprPool.checkOut(1,"rReg")
      dReg = self.vgprPool.checkOut(1,"dReg")
      tmpVgpr = self.vgprPool.checkOutAligned(2,2,"tmpReg")
      module.addInst("v_mov_b32", vgpr(dReg), loopCounter, "copy for divide IterGsu")
      module.addCode(vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr))
      module.addInst("v_readfirstlane_b32", sgpr(quotient), vgpr(qReg), "")
      module.addInst("v_readfirstlane_b32", sgpr(remainder), vgpr(rReg), "")
      self.vgprPool.checkIn(tmpVgpr)
      self.vgprPool.checkIn(dReg)
      self.vgprPool.checkIn(rReg)
      self.vgprPool.checkIn(qReg)

    # if gsuSumIdx < numIterPerWgRemainder
    module.addInst("s_add_u32", sgpr(tmpSgpr), "1", \
                  loopCounter, "tmp<-numIterMyWg+" )
    module.addInst("s_cmp_lt_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
        "gsuSumIdx < numIterPerWgRemainder" )
    module.addInst("s_cmov_b32", loopCounter, sgpr(tmpSgpr), "numIterMyWg++ if needed" )

    return module

  ##############################################################################
  # Calculate Loop Num Iter
  # loopIdx is the index of the loop (used for contractions with multiple summations)
  # 0 is outermost; self.unrollIdx is the unroll index.
  # -1 is tail loop (used only for the unroll loop)
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx):
    module = Code.Module("calculateLoopNumIter")

    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    loopChar = self.indexChars[loopDim]

    ########################################
    # Tail Loop
    if tailLoop:
      tmpSgpr = self.getTmpSgpr(4).idx()
      loopCounterName = self.loopCounterName(kernel, loopIdx)
      module.addSpaceLine()
      if kernel["SuppressNoLoadLoop"]:
        # If the tail loop is suppressed, then final iterations will have moved the Srd base forward
        # (and also moved back the srd shadow limit) and slammed Limit to 0, so need to 'undo'
        # those increments - see setTailSrd
        assert(kernel["PrefetchGlobalRead"] == 1) #if >1 would need a multiply here
        module.addInst("s_cmp_eq_u32", sgpr("OrigLoopCounter"), 0, "completely skipped unroll loop?")
        module.addInst("s_cselect_b32", sgpr(tmpSgpr+0), 0, sgpr("GlobalReadIncsA"), "force to 0?")
        module.addInst("s_cselect_b32", sgpr(tmpSgpr+1), 0, sgpr("GlobalReadIncsB"), "force to 0?")
        module.addCode(self.setTailSrd(kernel, self.tPA, sgpr(tmpSgpr+0)))
        module.addSpaceLine()
        module.addCode(self.setTailSrd(kernel, self.tPB, sgpr(tmpSgpr+1)))
        module.addSpaceLine()
        #module.addCode(self.getBomb())

      module.addComment("numIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)" \
          % (self.unrollChar, self.unrollChar))
      # size % DepthU
      module.addCode(scalarStaticDivideAndRemainder(tmpSgpr, loopCounterName, "SizesSum+%u"%loopIdx, kernel["DepthU"], tmpSgpr+2, 2))
      loopCounter = sgpr(loopCounterName)

      if kernel["LocalSplitU"] > 1:
        # (size % DepthU) + LSU - 1
        module.addInst("s_add_u32", loopCounter, hex(kernel["LocalSplitU"]-1), loopCounter, "(size % DepthU) + LSU - 1" )
        dividend = tmpSgpr+2
        module.addInst("s_mov_b32", sgpr(dividend), loopCounter, "copy for divide LSU" )
        module.addCode(scalarStaticDivideAndRemainder( loopCounterName, None, dividend, kernel["LocalSplitU"], tmpSgpr, 0))

      # if GSU numIter=0 if gsuSumIdx != remainder
      if kernel["GlobalSplitU"] > 1:
        module.addInst("s_cmp_lg_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
            "gsuSumIdx == numIterPerWgRemainder" )
        module.addInst("s_cmov_b32", loopCounter, hex(0), "numIter=0 if gsuSimIdx!=remainder")

      # if tail numIter == 0 skip altogether
      skipTailLoopLabel = Code.Label.getFormatting("SkipTailLoop%s"%(loopChar) )
      module.addInst("s_cmp_eq_u32", loopCounter, \
          hex(0), "numIter%s == 0"%loopChar )
      module.addInst("s_mov_b32", sgpr("OrigLoopCounter"), 0, \
          "repurpose to count each localRead increment")
      module.addInst("s_cbranch_scc1 %s"\
          % skipTailLoopLabel, \
          "skip to end of tail loop b/c numIter==0")

    ########################################
    # Unrolled Loop
    elif loopIdx == self.unrollIdx:
      loopCounterName = self.loopCounterName(kernel, loopIdx)
      loopCounter = sgpr(loopCounterName)
      if not self.do["PreLoop"]: module.addInst(".endif")

      sumSize = "SizesSum+%u"%loopIdx
      #sumSize = self.sumSize(kernel, loopIdx)
      if self.unrollIncIsDepthU:
        module.addInst("s_mov_b32", loopCounter, 0,\
                  "init loop counter, unrollIncIsDepthU mode")

      else:
        # TODO - use named arguments
        tmpSgpr = self.getTmpSgpr(3).idx()
        quotient = loopCounterName
        dividend = sumSize
        divisor = kernel["DepthU"]
        if self.noTailLoop and kernel["AssertSummationElementMultiple"] % kernel["DepthU"] != 0:
          # round up SizesSum/DepthU for noTailLoop case
          module.addInst("s_add_i32", sgpr(quotient), (divisor - 1), sgpr(dividend), \
              "round up SizeSum / DepthU" )
          module.addCode(scalarStaticDivideAndRemainder(quotient, None, quotient, divisor, tmpSgpr, 0))
        else:
          module.addCode(scalarStaticDivideAndRemainder(quotient, None, dividend, divisor, tmpSgpr, 0))
        # if GSU numIter++ if gsuSumIdx < remainder
        if kernel["GlobalSplitU"] > 1:
          module.addCode(self.calculateLoopNumIterGsu(kernel, loopCounterName, tmpSgpr))

      module.addInst("s_mov_b32", sgpr("OrigLoopCounter"), \
                loopCounter, \
                "copy loop counter")
    else:
      # other summation, not unroll loop
      #printExit("no assembly support for 2+ dimensional summation")
      module.addComment1("%sother summation, numIter%s = size%s" \
          % (self.indent, loopChar, loopChar))
      loopCounter = self.loopCounter(kernel, loopIdx)
      module.addInst("s_mov_b32", loopCounter, \
                sgpr("SizesSum+%u"%loopIdx), \
                "init loop counter")

    return module

  ##############################################################################
  # Open Loop
  # uDu: 'None' means not generating branching label which decides which part of G2L
  #      buffer to write to LDS
  ##############################################################################
  def openLoop(self, kernel, loopIdx, uDu=None, noLabelGen=False, beginLabelOnly=False):
    module = Code.Module("openLoop")
    # TODO - rewrite this function to simplify control-flow between tail-loop / unroll loop
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
      self.inTailLoop = True
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if not tailLoop and not noLabelGen:
      module.addCode(Code.Label("openLoop%s"%loopChar, ""))
    loopLabelBegin = Code.Label("%sLoopBegin%s%s"%("Tail" if tailLoop else "", loopChar, "_G2L%s"%uDu if uDu is not None else "" ), "" )
    loopLabelEnd = Code.Label("%sLoopEnd%s%s"%("Tail" if tailLoop else "", loopChar, "_G2L%s"%uDu if uDu is not None else ""), "" )

    if beginLabelOnly:
      # generate only beginLabel, then, return
      module.addCode(loopLabelBegin)
      return module

    # is numIter at least 1? otherwise skip to end
    # PGL needs a skip-check here if not bufferload
    # If kernel["SuppressNoLoadLoop"] we don't have a special loop for the 'last iter'
    loopCounter = self.loopCounter(kernel, loopIdx)
    if tailLoop:
      endCounter = 0
    elif kernel["PrefetchGlobalRead"] == 1:
      if kernel["SuppressNoLoadLoop"]:
        endCounter =  0
      else:
        endCounter = 1
    elif kernel["PrefetchGlobalRead"] == 2:
      if kernel["SuppressNoLoadLoop"]:
        endCounter =  1
      else:
        endCounter = 2
    else:
      endCounter =  0

    if tailLoop:
      # comment out since redundant
      """
      module.addInst("s_cmp_le_u32", \
          loopCounter, \
          hex(endCounter), \
          "LoopCounter%s < EndCounter"%(loopChar) )
      module.addInst("s_cbranch_scc1 %s"%loopLabelEnd.getLabelName(), \
          "do not enter Loop%s"%loopChar )

      module.addInst("s_mov_b32", sgpr("OrigLoopCounter"), 0, \
          "repurpose to count each localRead increment")
      """

      # LSU not all threads will do summation
      if kernel["LocalSplitU"] > 1:
        tmpSgpr = self.getTmpSgpr(1).idx()
        module.addComment1("apply exec mask for LSU")
        tmpVgpr = self.vgprPool.checkOutAligned(2, 2, "tmpVgpr")
        dummy = self.vgprPool.checkOut(1,"dummy")
        sgId = self.vgprPool.checkOut(1,"sgId")
        divisor = kernel["SubGroup0"]*kernel["SubGroup1"]
        module.addCode(vectorStaticDivide(sgId, "Serial", divisor, tmpVgpr, tmpSgpr))
        numIter = self.vgprPool.checkOut(1,"numIter")
        module.addInst("v_mov_b32", vgpr(numIter), sgpr("SizesSum+0"), "sizeU to vgpr")
        divisor = kernel["DepthU"]
        module.addCode(vectorStaticDivideAndRemainder(dummy, numIter, numIter, divisor, tmpVgpr, tmpSgpr))
        self.vgprPool.checkIn(dummy)
        #module.addCode() dump(vgpr(sgId)) )
        #module.addCode() dump(vgpr(numIter)) )
        module.addInst("_v_cmpx_lt_u32", self.vcc, \
            vgpr(sgId), vgpr(numIter), "sgId < numIter")
        self.vgprPool.checkIn(tmpVgpr)
        #self.tailNumIter = numIter
        #self.vgprPool.checkIn(numIter)
        # thread is active is sgId < numIter % LocalSplitU

      # begin loop
      if not noLabelGen:
        module.addCode(loopLabelBegin)

      # LSU mask for this iteration
      if kernel["LocalSplitU"] > 1:
        module.addInst("_v_cmpx_lt_u32", self.vcc, \
            vgpr(sgId), vgpr(numIter), "sgId < numIter")
        module.addInst("_v_add_co_u32", vgpr(sgId), self.vcc, hex(kernel["LocalSplitU"]), \
            vgpr(sgId), "sgId+=LSU")
        self.vgprPool.checkIn(sgId)
        self.vgprPool.checkIn(numIter)
        #module.addCode() dump(vgpr(sgId)) )

    else: # not tailloop:

      if loopIdx == self.unrollIdx:
        # 1 loop check is necessary only when AssertSummationElementMultiple % (DepthU * 2) != 0
        if kernel["PrefetchGlobalRead"] == 2 and kernel["AssertSummationElementMultiple"] % (kernel["DepthU"] * 2) != 0:
          if not self.unrollIncIsDepthU:
            module.addInst("s_cmp_eq_u32", \
                loopCounter, \
                hex(endCounter-1), \
                "LoopCounter%s < EndCounter"%(loopChar) )
          else:
            module.addInst("s_cmp_ge_u32", \
                loopCounter, \
                sgpr("UnrollLoopLastIter"), \
                "LoopCounter%s > EndCounter"%(loopChar) )
          toPGR1 = Code.Label.getFormatting(self.labels.getName("toPGR1"))
          module.addInst("s_cbranch_scc1 %s"%toPGR1, "PGR=2 but only 1 loop, toPGR1")

        if self.unrollIncIsDepthU:
          if kernel["PrefetchGlobalRead"] == 2:
            tmpSgpr = self.getTmpSgpr(1).idx()
            module.addInst("s_add_u32", sgpr(tmpSgpr),\
                loopCounter, \
                 "DepthU", "")
            loopCounter = sgpr(tmpSgpr)
          module.addInst("s_cmp_ge_u32", \
              loopCounter, \
              sgpr("UnrollLoopLastIter"), \
              "LoopCounter%s > EndCounter"%(loopChar) )
        else:
          module.addInst("s_cmp_le_u32", \
              loopCounter, \
              hex(endCounter), \
              "LoopCounter%s < EndCounter"%(loopChar) )
        jumpLabel = loopLabelEnd
        if kernel["PrefetchGlobalRead"]==2 and (not kernel["SuppressNoLoadLoop"]) and kernel["ExpandPointerSwap"]:
          # PGR=2 and EPS and no SuppressNoLoadLoop case, need to jump to EvenExit
          jumpLabel = Code.Label("LoopEnd%s_evenexit"%(loopChar), "" )
        module.addInst("s_cbranch_scc1 %s"%jumpLabel.getLabelName(), \
            "do not enter Loop%s"%loopChar )

      if not noLabelGen:
        module.addCode(loopLabelBegin)

      if loopIdx != self.unrollIdx:
        # reset LRO since these may have changed due to odd-iter exit ?
        if kernel["PrefetchGlobalRead"]:
          module.addComment0("openLoop - reset LRO for possible odd-iter exit")
          module.addCode(self.localReadResetOffsets(kernel, self.tPA))
          module.addCode(self.localReadResetOffsets(kernel, self.tPB))

    return module

  ##############################################################################
  # Close Loop
  # finalLoop : final unroll loop
  # uDu: 'None' means not generating branching label which decides which part of G2L
  #      buffer to write to LDS
  ##############################################################################
  def closeLoop(self, kernel, loopIdx, finalLoop, uDu=None, emitEndLabelOnly=False, oddLabel=False):
    module = Code.Module("closeLoop")
    if emitEndLabelOnly:
      loopIdx = self.unrollIdx
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      module.addCode(Code.Label("SkipTailLoop%s"%(loopChar), ""))
      return module

    finalJump = "s_cbranch_scc0"
    nonFinalJumpNeeded = True

    #module.addCode(self.syncStr, "")
    #module.addCode("s_endpgm")
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
      loopChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      loopLabelBegin = Code.Label("TailLoopBegin%s%s"%(loopChar, "_G2L%s"%uDu if uDu is not None else ""), "" )
      loopLabelEnd = Code.Label("TailLoopEnd%s%s"%(loopChar, "_G2L%s"%uDu if uDu is not None else ""), "" )
      loopLabelEndOddExit = Code.Label("TailLoopEnd%s_oddexit"%(loopChar), "unroll loop odditer exit" )
      loopCounter = self.loopCounter(kernel, loopIdx)

      unrollInc      = 1
      KinInnerUnroll = kernel["InnerUnroll"]
      if kernel["EnableMatrixInstruction"]:
        unrollInc      *= kernel["MatrixInstK"]
        KinInnerUnroll *= kernel["MatrixInstK"]
      if kernel["AssertSummationElementMultiple"] % KinInnerUnroll == 0:
        unrollInc *= kernel["InnerUnroll"]

      module.addComment1("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))

      module.addInst("s_sub_i32", \
          loopCounter, \
          loopCounter, \
          hex(unrollInc), \
          "dec counter%s (tailLoop)"%(loopChar) )

      # Track # LDS reads?
      module.addInst("s_add_u32", \
        sgpr("OrigLoopCounter"), \
        sgpr("OrigLoopCounter"), \
        hex(unrollInc),
        "inc counter%s"%(loopChar) )

      endCounter = 0
      module.addInst("s_cmp_le_i32", \
          loopCounter, \
          hex(endCounter), \
        "counter%s<=%d"%(loopChar,endCounter) )
    else: # not tailloop
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      loopLabelBegin = Code.Label("LoopBegin%s"%(loopChar), "" )
      loopLabelEnd = Code.Label("LoopEnd%s"%(loopChar), "" )
      loopLabelEndOddExit = Code.Label("LoopEnd%s_oddexit"%(loopChar), "unroll loop odditer exit" )
      loopLabelEndEvenExit = Code.Label("LoopEnd%s_evenexit"%(loopChar), "unroll loop eveniter exit" )
      loopCounter = self.loopCounter(kernel, loopIdx)
      module.addComment1("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))

      if self.unrollIncIsDepthU and loopIdx==self.unrollIdx:
        assert (not kernel["SuppressNoLoadLoop"]) # not accounting for end-of-loop iteration change here in deprecated mode

        if kernel["PrefetchGlobalRead"] == 2:
          tmpSgpr = self.getTmpSgpr(1).idx()
          module.addInst("s_add_u32", sgpr(tmpSgpr),\
              loopCounter, \
               "DepthU", "")
          module.addInst("s_cmp_ge_u32", \
              sgpr(tmpSgpr), \
              sgpr("UnrollLoopLastIter"), \
              "LoopCounter%s + DU < EndCounter. Go to PGR1"%(loopChar) )
        else:
          module.addInst("s_cmp_ge_u32", \
              loopCounter, \
              sgpr("UnrollLoopLastIter"), \
            "counter%s==0"%(loopChar) )
      else:
        # If PrefetchGlobalRead=1 the loads in the loop prefetch next macro-tile
        # For the final trip through the unroll loop we need to ensure those loads stay in bounds.

        # One technique is to create a copy of the unroll loop with all loads removed.
        # However buffer load doesn't need this loop copy since we OOB loads can be suppressed by buffer limit hardware
        # So can do one more iteration (endCounter==0) in the main unroll loop, and adjust the pointer
        # increments appropriately.
        # Also sum idx other than unroll always compare against 0 (there is no PGR to account for)
        if kernel["PrefetchGlobalRead"] == 1 and not kernel["SuppressNoLoadLoop"] and loopIdx == self.unrollIdx:
          endCounter = 1
        elif kernel["PrefetchGlobalRead"] == 2 and not kernel["SuppressNoLoadLoop"] and loopIdx == self.unrollIdx:
          endCounter = 2
        else:
          endCounter = 0

        if kernel["AssertSummationElementMultiple"] % (kernel["DepthU"] * 2) == 0 and endCounter > 0:
          # if AssertSummationElementMultiple is multiple of DepthU*2, loop exit is necessary only once in 2 Loop iterations
          #  In endCounter % 2 == 1 case, exit at lc % 2 == 0 (= oddLabel). It means no exit if not oddLabel
          #  In endCounter % 2 == 0 case, exit at lc % 2 == 1 (= not oddLabel). It means no exit if oddLabel
          # No exit case, no code is necessary except for final Loop

          # decrement by 2 if PGR=2 and StaggerU is 0, else 1
          decValue = 2 if kernel["PrefetchGlobalRead"]==2 and kernel["StaggerU"] == 0 else 1
          decCode = Code.Inst("s_sub_u32", \
              loopCounter, loopCounter, \
              decValue, \
              "dec counter%s"%(loopChar) )
          condCode = Code.Inst("s_cmp_eq_i32", \
              loopCounter, \
              hex(endCounter), \
            "counter%s==%d"%(loopChar,endCounter) )

          noExit = False

          if endCounter%2 != 0:
            if not oddLabel:
              noExit = True
          else:
            if oddLabel:
              noExit = True

          if noExit:
            # No exit. No dec code if decValue is 2
            if decValue == 2:
              decCode = ""
            condCode = ""
            nonFinalJumpNeeded = False
            if finalLoop:
              # No exit and finalLoop case, use s_branch (no condition)
              finalJump = "s_branch"

          if decCode: module.addCode(decCode)
          if condCode: module.addCode(condCode)
        else:
          module.addInst("s_sub_u32", \
              loopCounter, loopCounter, \
              1, \
              "dec counter%s"%(loopChar) )

          module.addInst("s_cmp_eq_i32", \
              loopCounter, \
              hex(endCounter), \
            "counter%s==%d"%(loopChar,endCounter) )

    jumpLabel = loopLabelEnd
    if not tailLoop and not kernel["SuppressNoLoadLoop"] and kernel["ExpandPointerSwap"]:
      # in this case, odd or/and even code is generated and use odd/even exit to avoid skipping odd/even code
      # (end label is generated after odd/even code)
      jumpLabel = loopLabelEndOddExit if oddLabel else loopLabelEndEvenExit
    if not finalLoop:
      if nonFinalJumpNeeded:
        # just an exit check, else fall through to the next loop copy
        module.addInst("s_cbranch_scc1 %s"%(jumpLabel.getLabelName()), "exit Loop%s"%loopChar )
    else: #finalLoop:

      if tailLoop and kernel.enabledSplitLDS:
        tailLoopLabelEnd = Code.Label.getFormatting(
          "TailLoopEnd%s%s"%(loopChar, "_G2L%s"%(kernel["DepthULdsDivisor"]-1) if kernel.enabledSplitLDS else "") )
        module.addInst("s_cbranch_scc1", tailLoopLabelEnd, "break Loop%s"%loopChar)
        thresForNextSubLoop = (uDu+1)*(kernel["_DepthULds"])
        module.addInst("s_cmp_ge_u32", sgpr("OrigLoopCounter"), thresForNextSubLoop,
          "OrigLoopCounter >= %u (G2L buffer %u/%u)"%(thresForNextSubLoop, uDu, kernel["DepthULdsDivisor"]) )

      module.addInst("%s %s"%(finalJump, loopLabelBegin.getLabelName()), \
          "restart Loop%s"%(loopChar ))

      if not tailLoop and loopIdx == self.unrollIdx:
        oddIterPreCode = Code.Module()
        oddIterCode = Code.Module()
        evenIterPreCode = Code.Module()
        evenIterCode = Code.Module()
        if not kernel["SuppressNoLoadLoop"] and kernel["ExpandPointerSwap"]:
          oddIterPreCode.addCode(loopLabelEndOddExit)
          # In this case we kept the 'no-load' loop which has LDS offsets assuming first bank of LDS
          # if we exit the main loop at an odd iter - need to swap LDS read pointers
          # so the ds_reads read from the 'high' buffer of LDS
          oddIterPreCode.addComment1("Select high bank of LDS")
          # Generate local read address code only if DirectToVgpr is not enabled
          if not kernel["DirectToVgprA"]:
            oddIterCode.addCode(self.localReadSwapOffsets(kernel, False, self.tPA))
          # Generate local read address code only if DirectToVgpr is not enabled
          if not kernel["DirectToVgprB"]:
            oddIterCode.addCode(self.localReadSwapOffsets(kernel, False, self.tPB))

          evenIterPreCode.addCode(loopLabelEndEvenExit)
          # generate even code here (so far, for PrefetchGlobalRead=2 only)
          if kernel["PrefetchGlobalRead"]==2:
            # Generate local write address code only for PrefetchGlobalRead==2 (localWriteSwapOffsets does nothing if DirectToVgpr is enabled)
            # Code is unnecessary if DirectToLds is enabled, but internal SwapOffset is necessary if useInitAccVgprOpt is True
            if kernel["DirectToLdsA"]:
              if self.useInitAccVgprOpt:
                self.localWriteSwapOffsets(kernel, True, self.tPA)
            else:
              evenIterCode.addCode(self.localWriteSwapOffsets(kernel, False, self.tPA))
            if kernel["DirectToLdsB"]:
              if self.useInitAccVgprOpt:
                self.localWriteSwapOffsets(kernel, True, self.tPB)
            else:
              evenIterCode.addCode(self.localWriteSwapOffsets(kernel, False, self.tPB))
            # swap internal write pointer as well (except for useInitAccVgprOpt case)
            if not self.useInitAccVgprOpt:
              evenIterCode.addCode(self.localWriteSwapOffsets(kernel, True, self.tPA))
              evenIterCode.addCode(self.localWriteSwapOffsets(kernel, True, self.tPB))

        # generate even, odd exit code
        # not oddLabel case, order is even -> odd
        firstPreCode = evenIterPreCode
        firstCode = evenIterCode
        secondPreCode = oddIterPreCode
        secondCode = oddIterCode
        if oddLabel:
          # oddLabel case, swap the order (odd -> even)
          firstPreCode, secondPreCode = secondPreCode, firstPreCode
          firstCode, secondCode = secondCode, firstCode

        module.addCode(firstPreCode)
        module.addCode(firstCode)

        # if secondCode exist, add jump to skip secondCode
        if secondCode.count():
          module.addInst("s_branch %s"%loopLabelEnd.getLabelName(), \
              "exit unroll loop%s (and skip second exit code)"%(loopChar ))
        module.addCode(secondPreCode)
        module.addCode(secondCode)

      module.addCode(loopLabelEnd)

      if tailLoop:
        if len(kernel["ProblemType"]["IndicesSummation"]) > 1:
          # recover the 'damage' done to LRO:
          stmp = self.getTmpSgpr(1).idx()

          # if LRA is backed-up before (wlr case), we simply restore the addr (sub inc*loop doesn't work)
          tPList = []
          if self.oriLraA != None:
            if not kernel["DirectToVgprA"]: # no local read code if DirectToVgpr is enabled
              module.addInst("v_mov_b32", vgpr("LocalReadAddrA"), vgpr(self.oriLraA), "restore LRA")
            self.vgprPool.checkIn(self.oriLraA)
            self.oriLraA = None
          else:
            tPList.append(self.tPA)
          if self.oriLraB != None:
            if not kernel["DirectToVgprB"]: # no local read code if DirectToVgpr is enabled
              module.addInst("v_mov_b32", vgpr("LocalReadAddrB"), vgpr(self.oriLraB), "restore LRA")
            self.vgprPool.checkIn(self.oriLraB)
            self.oriLraB = None
          else:
            tPList.append(self.tPB)
          for tP in tPList:
            tc     = tP["tensorChar"]
            LdsPad = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
            inc    = kernel["LocalSplitU"]*(kernel["MacroTile%s"%tc]+LdsPad)*tP["bpe"]

            # aligned with localReadInc
            if kernel["EnableMatrixInstruction"]:
              if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
                inc = kernel["LocalSplitU"] * tP["bpe"]
              # No need to *= K, because LoopCounter is increased by K each time
              # inc *= kernel["MatrixInstK"]

            if not kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
              module.addInst("s_mov_b32", sgpr(stmp), inc, "tailloop lds offset")
              module.addInst("s_mul_i32", sgpr(stmp), sgpr("OrigLoopCounter"), sgpr(stmp), "scale by mul")
              module.addInst("_v_sub_u32", vgpr("LocalReadAddr%s"%tc), vgpr("LocalReadAddr%s"%tc), sgpr(stmp), "remove lro damage")
          # if LWA is backed-up before, we simply restore the addr
          if self.oriLwaA != None:
            if not kernel["DirectToVgprA"]: # no local write code if DirectToVgpr is enabled
              module.addInst("v_mov_b32", vgpr("LocalWriteAddrA"), vgpr(self.oriLwaA), "restore LWA")
            if not kernel["DirectToVgprB"]: # no local write code if DirectToVgpr is enabled
              module.addInst("v_mov_b32", vgpr("LocalWriteAddrB"), vgpr(self.oriLwaB), "restore LWA")
            self.vgprPool.checkIn(self.oriLwaA)
            self.vgprPool.checkIn(self.oriLwaB)
            self.oriLwaA = None
            self.oriLwaB = None

    # restore all threads
    if tailLoop and kernel["LocalSplitU"] > 1:
      sgprCnt = self.laneSGPRCount
      waveSize = kernel["WavefrontSize"]
      module.addComment1("restore full exec mask")
      fullExec = self.getTmpSgpr(sgprCnt).idx()
      activeMask = "0xFFFFFFFF" if (waveSize == 32) else "0xFFFFFFFFFFFFFFFF"
      module.addInst("s_mov_b{}".format(waveSize), sgpr(fullExec,sgprCnt), activeMask, "restore all threads active")
      module.addInst("s_or_saveexec_b{}".format(waveSize),  sgpr(fullExec,sgprCnt), sgpr(fullExec,sgprCnt), "full mask -> exec" )
    return module

  ##############################################################################
  def openLoopCopy(self, kernel, lc):
    return Code.Label("LoopCopy%u"%(lc+1), "" )

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self, kernel, label = None, isOptNLL = False):
    module = Code.Module("endSummation")

    module.addCode(Code.Label((self.labels.getUniqueNamePrefix("Summation_End") if label is None else label), ""))

    if kernel["StorePriorityOpt"]:
      module.addInst("s_setprio", "0", "optimization store")

    vbegin = self.startVgprValuA
    vsize = self.lastVgprForReads - self.startVgprValuA

    self.vgprPool.add(vbegin, vsize, "endSummation")
    module.addComment0("endSummation: add vgpr [%u...%u) to pool" % \
            (vbegin, vbegin+vsize))

    lastRegTag=None
    for i in range(self.lastPostLoopSgpr, self.sgprPool.size()):
      regTag = self.sgprPool.pool[i].tag
      if regTag != lastRegTag:
        lastRegTag = regTag
        if self.sgprPool.pool[i].status == RegisterPool.Status.InUse:
          module.addCode(self.undefineSgpr(regTag))

    if self.db["InitVgpr"] & 0x2:
      module.addCode(self.vgprPool.initTmps(self.initVgprValue,start=0, stop=100))
    if 0: # FIXME: Can remove?
      for i in range(0,16+1):
         #module.addInst("v_mov_b32", vgpr(21), hex(self.initVgprValue), "hack tmp in pool")
         module.addInst("v_mov_b32", vgpr(21), vgpr(21), "hack tmp in pool")

    # this doesn't seem to do anything - not being aggressive with lastPostLoopSgpr
    if self.db["InitSgpr"] & 0x2:
      module.addCode(self.sgprPool.initTmps(self.initSgprValue))

    if self.db["ConservativeWaitCnt"] & 0x10:
      module.addInst("s_barrier", "debug")
      module.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "")
      if self.archCaps["SeparateVscnt"]:
        module.addInst("s_waitcnt_vscnt", "null", "0", "")

    if kernel["SuppressNoLoadLoop"]:
      module.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "wait for all summation activity")
      if self.archCaps["SeparateVscnt"]:
        module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

    # copy accumulated C from agpr to vgpr
    if kernel["EnableMatrixInstruction"]:
      #TODO avoid s_nop if its possible
      #instCycles = kernel["MatrixInstM"] // 2 # 32x32 is 64 cycles, 16x16 is 32 cycles, 4x4 is 8 cycles
      #module.addInst("s_nop", "%u" % instCycles, "")
      module.addCode(self.MapAcctoArchRegs(kernel,option=0, isOptNLL=isOptNLL))
      if kernel["MIArchVgpr"]:
        module.addCode(self.MulMIoutAlphaToArch(kernel))

    return module

  ##############################################################################
  # MFMA Iteration
  ##############################################################################
  def mfmaIter(self, kernel, u, innerUnroll, vregSetIdx, lastKinloop=False, tail=False, firstIter=False):
    imod = Code.Module("mi")
    shiftK = Code.Module("shiftK")
    m = (u) % (self.numVgprBuffer+1) # local to use for MACs

    # calculate constant
    numRegistersIn   = kernel["ProblemType"]["DataType"].numRegisters()
    numRegistersOut  = kernel["MIRegPerOut"]
    loopCounterName  = self.loopCounterName(kernel, self.unrollIdx)
    accs_per_wave    = kernel["MatrixInstM"] * kernel["MatrixInstN"] * kernel["MatrixInstB"] \
                       / self.kernel["WavefrontSize"] * numRegistersOut
    dividerFortidInK = kernel["MatrixInstN"] * kernel["MatrixInstB"]
    numMIInput       = kernel["MIInputPerThread"]
    miInTypeName     = kernel["ProblemType"]["DataType"].toNameAbbrev() # v_mfma_[...xK]<InType>
    miOutTypeName    = kernel["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() # v_mfma_<OutType>..
    vgprPerInput     = int(numMIInput * numRegistersIn)
    shiftPerElement  = int(numRegistersIn * 32)
    s_nop            = 0
    accumRegType     = "a" if not kernel["MIArchVgpr"] else "v"
    mfma_1k          = "_1k" if kernel["MFMA_BF16_1K"] else ""
    accStoreCIdx     = 0

    # alloc vgpr
    kReg    = None
    abReg   = None
    tmpVgpr = None
    dummy   = None

    if (numRegistersIn < 1) and ((kernel["UnrollMajorLDSA"] == False) or (kernel["UnrollMajorLDSB"] == False)):
      s_nop = 2

    # here we remap index to where it read for wider local read
    # ex. if we read 2 iteration at a time,
    #   original   : _ds_load_b64  valuA_X0_I0
    #   read 2 iter: _ds_load_b128 valuA_X0_I0 (we read valuA_X0_I0 and valuA_X1_I0)
    # instead of using valuA_X1_I0, we use valuA_X0_I0+2 as mfma input

    vgprBufferA_new = (m//self.numIterPerCoalescedReadA)*self.numIterPerCoalescedReadA
    vgprBufferA_new_offset = m%self.numIterPerCoalescedReadA*kernel["InnerUnroll"]*vgprPerInput

    vgprBufferB_new = (m//self.numIterPerCoalescedReadB)*self.numIterPerCoalescedReadB
    vgprBufferB_new_offset = m%self.numIterPerCoalescedReadB*kernel["InnerUnroll"]*vgprPerInput

    numVgprPerBlockA = self.numVgprG2LA // 2
    numVgprPerBlockB = self.numVgprG2LB // 2

    # handle multiple K element in MFMA instruction
    if tail and kernel["MatrixInstK"] > 1:
      kReg    = self.vgprPool.checkOut(1,"kReg") # remainder
      tmpSgpr = self.getTmpSgpr(3).idx()
      shiftK.addCode(vectorStaticRemainder(dummy, kReg, "Serial", self.kernel["WavefrontSize"], tmpVgpr, tmpSgpr))
      shiftK.addCode(vectorStaticDivide(kReg, kReg, dividerFortidInK, tmpVgpr, tmpSgpr))
      shiftK.addCode(staticMultiply(vgpr(kReg), vgpr(kReg), numMIInput, sgpr(tmpSgpr)))

      # replace 0 for differnet thread
      shiftK.addInst("v_cmp_ge_i32", sgpr(tmpSgpr, 2), vgpr(kReg), sgpr(loopCounterName), "check K index >= Size L")
      for bk in range(0, vgprPerInput):
        for a in range(0, kernel["MIWaveTileA"]):
          for iui in range(0, innerUnroll):
            aStr = vgpr("ValuA_X%u_I%u+%u+%u" % (m, iui, a*vgprPerInput, bk), 1)
            shiftK.addInst("v_cndmask_b32", aStr, aStr, hex(0), sgpr(tmpSgpr, 2), "set 0 if K_idx >= sizeL")
        for b in range(0, kernel["MIWaveTileB"]):
          for iui in range(0, innerUnroll):
            bStr = vgpr("ValuB_X%u_I%u+%u+%u" % (m, iui, b*vgprPerInput, bk), 1)
            shiftK.addInst("v_cndmask_b32", bStr, bStr, hex(0), sgpr(tmpSgpr, 2), "set 0 if K_idx >= sizeL")

      # replace 0 for same thread
      if numMIInput > 1:
        abReg   = self.vgprPool.checkOutAligned(vgprPerInput, 2 if vgprPerInput>1 else 1, "abReg")
        tmpVgpr = self.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        dummy   = self.vgprPool.checkOut(1,"dummy")
        shiftK.addInst("_v_sub_u32",    vgpr(kReg), sgpr(loopCounterName), vgpr(kReg), "get distance between size and k index")
        shiftK.addInst("v_cmp_lt_i32", sgpr(tmpSgpr,2), vgpr(kReg), numMIInput, "set partial 0 if distance less than input per thread")
        shiftK.addInst("s_and_b32",    sgpr(tmpSgpr+2), sgpr(loopCounterName), numMIInput-1, "get inputs for edge thread")
        shiftK.addInst("s_sub_u32",    sgpr(tmpSgpr+2), numMIInput, sgpr(tmpSgpr+2), "use shift to fill 0 for outside element")
        shiftK.addInst("s_lshl_b32",   sgpr(tmpSgpr+2), sgpr(tmpSgpr+2), log2(shiftPerElement), "use shift to fill 0 for outside element")
        for a in range(0, kernel["MIWaveTileA"]):
          for iui in range(0, innerUnroll):
            iuiA_new = (iui//self.numReadsIterCoalescedA)*self.numReadsIterCoalescedA
            iuiA_new_offset = iui%self.numReadsIterCoalescedA*vgprPerInput
            a_new = a*vgprPerInput*self.numReadsIterCoalescedA
            aStr = vgpr("ValuA_X%u_I%u+%u+%u+%u" % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset), vgprPerInput)
            tmpVregIdx = 0
            shiftK.addInst("v_lshlrev_b%u" % (vgprPerInput*32), vgpr(abReg, vgprPerInput), sgpr(tmpSgpr+2), aStr, "")
            for bk in range(0, vgprPerInput):
              aStr  = vgpr("ValuA_X%u_I%u+%u+%u+%u+%u" % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset, bk), 1)
              if kernel["DirectToVgprA"]:
                # overwrite aStr for DirectToVgprA
                tmp   = tmpVregIdx + bk
                aStr  = vgpr("G2LA+%u" % (tmp), vgprPerInput)
              shiftK.addInst("v_cndmask_b32", aStr, aStr, vgpr(abReg+bk), sgpr(tmpSgpr, 2), "")
        for b in range(0, kernel["MIWaveTileB"]):
          for iui in range(0, innerUnroll):
            iuiB_new = (iui//self.numReadsIterCoalescedB)*self.numReadsIterCoalescedB
            iuiB_new_offset = iui%self.numReadsIterCoalescedB*vgprPerInput
            b_new = b*vgprPerInput*self.numReadsIterCoalescedB
            bStr = vgpr("ValuB_X%u_I%u+%u+%u+%u" % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset), vgprPerInput)
            tmpVregIdx = 0
            shiftK.addInst("v_lshlrev_b%u" % (vgprPerInput*32), vgpr(abReg, vgprPerInput), sgpr(tmpSgpr+2), bStr, "")
            for bk in range(0, vgprPerInput):
              bStr = vgpr("ValuB_X%u_I%u+%u+%u+%u+%u" % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset, bk), 1)
              if kernel["DirectToVgprB"]:
                # overwrite bStr for DirectToVgprB
                tmp   = tmpVregIdx + bk
                bStr  = vgpr("G2LB+%u" % (tmp), 1)
              shiftK.addInst("v_cndmask_b32", bStr, bStr, vgpr(abReg+bk), sgpr(tmpSgpr, 2), "")

      s_nop = 2

    if s_nop != 0:
      imod.addInst("s_nop", "%u" % (s_nop - 1), "")

    for iui in range(0, innerUnroll):
      iuiA_new = (iui//self.numReadsIterCoalescedA)*self.numReadsIterCoalescedA
      iuiA_new_offset = iui%self.numReadsIterCoalescedA*vgprPerInput
      iuiB_new = (iui//self.numReadsIterCoalescedB)*self.numReadsIterCoalescedB
      iuiB_new_offset = iui%self.numReadsIterCoalescedB*vgprPerInput
      zgemmVaddSrcCheck = [[], [], []] # to avoid generating redundant v_add
      outer = 1
      loopSwap = False
      # complex case, swap inner loop and outer loop so that idxA comes outer
      # this is to re-use same tmp vgpr to nagate ai or ar
      if kernel["ProblemType"]["DataType"].isComplex() and self.tPB["tile01Idx"]:
        outer = 0
        loopSwap = True
      inner = 1 - outer # inner is the opposite of outer
      for idxOuter in range(0, kernel["MIWaveTile"][outer]):
        for idxInner in range(0, kernel["MIWaveTile"][inner]):
          idx0 = idxInner
          idx1 = idxOuter
          if loopSwap:
            idx0, idx1 = idx1, idx0
          accIdx   = idx1 * kernel["MIWaveTile"][0] + idx0
          accStart = accIdx * accs_per_wave
          accEnd   = accStart + accs_per_wave - 1
          accStartSrc1 = accStart
          accEndSrc1   = accEnd
          accStartSrc2 = accStart
          accEndSrc2   = accEnd
          if firstIter:
            # use the last accs_per_wave as src (assuming only these are initialized to 0)
            numAccvgprs = self.numVgprValuC if kernel["MIArchVgpr"] else self.totalAgprs
            accStartSrc1 = numAccvgprs - accs_per_wave
            accEndSrc1   = accStartSrc1 + accs_per_wave - 1
          idxA     = idx0 if self.tPB["tile01Idx"] else idx1
          idxB     = idx1 if self.tPB["tile01Idx"] else idx0
          a_new    = idxA*vgprPerInput*self.numReadsIterCoalescedA
          b_new    = idxB*vgprPerInput*self.numReadsIterCoalescedB
          aStr     = "ValuA_X%u_I%u+%u+%u+%u" % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset)
          bStr     = "ValuB_X%u_I%u+%u+%u+%u" % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset)
          if kernel["DirectToVgprA"]:
              # overwrite aStr for DirectToVgprA
              numVgprValuAPerBlock = kernel["MIWaveTileA"] * kernel["MIInputPerThread"] * self.tPA["bpe"] // self.bpr
              # re-calculate vgprBufferA_new and offset for DirectToVgpr. Use u instead of m (number of local prefetch buffer does not matter)
              vgprBufferA_new = (u//self.numIterPerCoalescedReadA)*self.numIterPerCoalescedReadA
              vgprBufferA_new_offset = u%self.numIterPerCoalescedReadA*kernel["InnerUnroll"]*vgprPerInput
              a_new += vregSetIdx * numVgprPerBlockA + (iuiA_new + vgprBufferA_new * kernel["InnerUnroll"]) * numVgprValuAPerBlock
              aStr  = "G2LA+%u+%u+%u" % (a_new, vgprBufferA_new_offset, iuiA_new_offset)
              # self.vgprValuDouble case, need to change valuB to toggle double buffer
              if self.vgprValuDouble and vregSetIdx > 0:
                numOneSet = self.numVgprValuB//2
                bStr += "+%u"%(vregSetIdx * numOneSet)
          if kernel["DirectToVgprB"]:
              # overwrite bStr for DirectToVgprB
              numVgprValuBPerBlock = kernel["MIWaveTileB"] * kernel["MIInputPerThread"] * self.tPB["bpe"] // self.bpr
              # re-calculate vgprBufferB_new and offset for DirectToVgpr. Use u instead of m (number of local prefetch buffer does not matter)
              vgprBufferB_new = (u//self.numIterPerCoalescedReadB)*self.numIterPerCoalescedReadB
              vgprBufferB_new_offset = u%self.numIterPerCoalescedReadB*kernel["InnerUnroll"]*vgprPerInput
              b_new += vregSetIdx * numVgprPerBlockB + (iuiB_new + vgprBufferB_new * kernel["InnerUnroll"]) * numVgprValuBPerBlock
              bStr  = "G2LB+%u+%u+%u" % (b_new, vgprBufferB_new_offset, iuiB_new_offset)
              # self.vgprValuDouble case, need to change valuA to toggle double buffer
              if self.vgprValuDouble and vregSetIdx > 0:
                numOneSet = self.numVgprValuA//2
                aStr += "+%u"%(vregSetIdx * numOneSet)
          aStr     = vgpr(aStr, vgprPerInput)
          bStr     = vgpr(bStr, vgprPerInput)
          Str0     = aStr if self.tPB["tile01Idx"] else bStr
          Str1     = bStr if self.tPB["tile01Idx"] else aStr

          if kernel["ProblemType"]["DataType"].isComplex():
            # override because complex mul is emulated by 4 mfma insts
            # TODO: adopt component system
            miInTypeName = miOutTypeName #"f32" for SingleComplex, "f64" for DoubleComplex
            ccA = kernel["ProblemType"]["ComplexConjugateA"]
            ccB = kernel["ProblemType"]["ComplexConjugateB"]
            ccVgprs = [None]*3 # three terms that can be negated: [real1, imag0, imag1]
            ccInsts = [None]*3
            accImOffset = self.AccVgprImagNumOffset(kernel)
            # for firstIter, need to use accStartSrc for img instead of adding accImOffset
            accStartSrcImg1 = accStartSrc1 if firstIter else accStartSrc1+accImOffset
            accEndSrcImg1 = accStartSrcImg1 + accs_per_wave - 1
            accStartSrcImg2 = accStartSrc2+accImOffset
            accEndSrcImg2 = accStartSrcImg2 + accs_per_wave - 1

            # vgpr A,B setting. In complex case, numRegistersIn does not match. Use numRegistersOut instead
            ar = vgpr("ValuA_X%u_I%u+%u+%u+%u"   % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset), numRegistersOut)
            ai = vgpr("ValuA_X%u_I%u+%u+%u+%u+%u" % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset, numRegistersOut), numRegistersOut)
            br = vgpr("ValuB_X%u_I%u+%u+%u+%u"   % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset), numRegistersOut)
            bi = vgpr("ValuB_X%u_I%u+%u+%u+%u+%u" % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset, numRegistersOut), numRegistersOut)
            if kernel["DirectToVgprA"]:
              ## overwrite aStr for DirectToVgprA
              ar  = vgpr("G2LA+%u+%u+%u" % (a_new, vgprBufferA_new_offset, iuiA_new_offset), numRegistersOut)
              ai  = vgpr("G2LA+%u+%u+%u+%u" % (a_new, vgprBufferA_new_offset, iuiA_new_offset, numRegistersOut), numRegistersOut)
            if kernel["DirectToVgprB"]:
              # overwrite bStr for DirectToVgprB
              br  = vgpr("G2LB+%u+%u+%u" % (b_new, vgprBufferB_new_offset, iuiB_new_offset), numRegistersOut)
              bi  = vgpr("G2LB+%u+%u+%u+%u" % (b_new, vgprBufferB_new_offset, iuiB_new_offset, numRegistersOut), numRegistersOut)
            v_mfma = "v_mfma_%s_%ux%ux%u%s "%(miOutTypeName, kernel["MatrixInstM"], kernel["MatrixInstN"], kernel["MatrixInstK"], miInTypeName)
            v_add = "v_add_" + miOutTypeName
            offsetVgpr = [0,0,0]
            forceGenerate = ccA and ccB # so far, v_add is always necessary for ccA and ccB case
            if ccA == ccB:
              arrayIndex = 0
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate r1")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again
              if forceGenerate or (ai not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = Code.Inst(v_add, vgpr(ccVgprs[arrayIndex] + offsetVgpr[arrayIndex], numRegistersOut), "-"+ai, "0", "Ai=-Ai")
                zgemmVaddSrcCheck[arrayIndex].append(ai)
            if ccA:
              arrayIndex = 1
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate i0")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again
              if forceGenerate or (ai not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = Code.Inst(v_add, vgpr(ccVgprs[arrayIndex] + offsetVgpr[arrayIndex], numRegistersOut), "-"+ai, "0", "Ai=-Ai")
                zgemmVaddSrcCheck[arrayIndex].append(ai)
            if ccB:
              arrayIndex = 2
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate i1")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again
              if forceGenerate or (ar not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = Code.Inst(v_add, vgpr(ccVgprs[arrayIndex] + offsetVgpr[arrayIndex], numRegistersOut), "-"+ar, "0", "Ar=-Ar")
                zgemmVaddSrcCheck[arrayIndex].append(ar)
            (src0, src1) = (br, ar) if kernel["SourceSwap"] else (ar, br)
            for inst in ccInsts:
              if inst is not None:
                imod.addCode(inst)
            imod.addInst(v_mfma, "%s[%u:%u]" % (accumRegType, accStart, accEnd), src0, src1, "%s[%u:%u]"%(accumRegType, accStartSrc1, accEndSrc1), "Cr += Ar*Br")
            (src0, src1) = (bi, (vgpr(ccVgprs[0] + offsetVgpr[0], numRegistersOut) if ccVgprs[0] else ai)) if kernel["SourceSwap"] else ((vgpr(ccVgprs[0] + offsetVgpr[0], numRegistersOut) if ccVgprs[0] else ai), bi)
            imod.addInst(v_mfma + "%s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx), src0, src1, "%s[%u:%u]" %(accumRegType, accStartSrc2, accEndSrc2), "Cr += %sAi*Bi"%("-" if ccVgprs[0] else ""))
            (src0, src1) = (br, (vgpr(ccVgprs[1] + offsetVgpr[1], numRegistersOut) if ccVgprs[1] else ai)) if kernel["SourceSwap"] else ((vgpr(ccVgprs[1] + offsetVgpr[1], numRegistersOut) if ccVgprs[1] else ai), br)
            imod.addInst(v_mfma + "%s[%u:%u]" % (accumRegType, accStart+accImOffset, accEnd+accImOffset), src0, src1, "%s[%u:%u]"%(accumRegType, accStartSrcImg1, accEndSrcImg1), "Ci += %sAi*Br"%("-" if ccVgprs[1] else ""))
            (src0, src1) = (bi, (vgpr(ccVgprs[2] + offsetVgpr[2], numRegistersOut) if ccVgprs[2] else ar)) if kernel["SourceSwap"] else ((vgpr(ccVgprs[2] + offsetVgpr[2], numRegistersOut) if ccVgprs[2] else ar), bi)
            imod.addInst(v_mfma + "%s[%u+%u:%u+%u]" % (accumRegType, accStart+accImOffset, accStoreCIdx, accEnd+accImOffset, accStoreCIdx), src0, src1, "%s[%u:%u]"%(accumRegType, accStartSrcImg2, accEndSrcImg2), "Ci += %sAr*Bi"%("-" if ccVgprs[2] else ""))

            for v in ccVgprs:
              if v is not None: self.vgprPool.checkIn(v)
          else:
            v_mfma = "v_mfma_%s_%ux%ux%u%s%s" % (miOutTypeName, kernel["MatrixInstM"], kernel["MatrixInstN"], kernel["MatrixInstK"], miInTypeName, mfma_1k)
            if kernel["SourceSwap"]:
              src0 = Str1
              src1 = Str0
            else:
              src0 = Str0
              src1 = Str1
            imod.addInst(v_mfma, "%s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx), \
                         src0, src1, "%s[%u:%u]" % (accumRegType, accStartSrc1, accEndSrc1), "")

    # release register
    if kReg is not None: self.vgprPool.checkIn(kReg)
    if abReg is not None: self.vgprPool.checkIn(abReg)
    if tmpVgpr is not None: self.vgprPool.checkIn(tmpVgpr)
    if dummy is not None: self.vgprPool.checkIn(dummy)

    mfmaMod = Code.Module("mfmaCode")
    mfmaMod.addCode(shiftK)
    mfmaMod.addCode(imod)

    return mfmaMod

  ##############################################################################
  # At Least 1 Unroll
  # prefetch means this is in the prefetch code, either before unroll loop
  # or in the PAP code.
  # isOptNLL : this is for the store-interleaved NLL optimization
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL):
    module = Code.Module("openSumAtLeastUnroll")
    if prefetch:
      if not isOptNLL:
        module.addCode(self.checkLastIter(kernel))
        if kernel["StorePriorityOpt"]:
          module.addInst("s_setprio 0", "optimization store")
        if self.doShadowInit:
          shadowName = Code.Label.getFormatting("ShadowInitStart")
          module.addInst("s_cbranch_scc1 %s"\
              % shadowName, \
              "skip to ShadowInitStart iter b/c numIter==0")
        else:
          loopChar = self.indexChars[ \
              kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
          labelName = Code.Label.getFormatting("LoopEnd%s"%loopChar)
          module.addInst("s_cbranch_scc1 %s" % labelName,
              "skip to unrollLoop end loop%s iter b/c numIter==0" % loopChar)
    elif isOptNLL:
      skipOptNLL = Code.Label("OptNLL_End", "")
      tmpSgpr = self.getTmpSgpr(2).idx()

      module.addCode(self.checkIsBetaZero(kernel, tmpSgpr, skipOptNLL))

      # check alpha
      if self.do["ApplyAlpha"]:
        # (The new hgemm (h,h,h,h,s,s) is included in ComputeType=Single)
        if kernel["ProblemType"]["ComputeDataType"].isHalf():
          # for (h,h,h,h,h,h) no HPA,
          module.addInst("s_mov_b32", sgpr(tmpSgpr), "0x3c003c00", "Packed alpha==1.0")
          module.addInst("s_cmp_eq_u32", sgpr("Alpha"), sgpr(tmpSgpr), "alpha == 1.0?")

        # Shouldn't go here. Currently, DataType=B->ComputeDataType=S
        # (bf-gemm is included in ComputeType=Single)
        elif kernel["ProblemType"]["ComputeDataType"].isBFloat16():
          module.addInst("s_mov_b32", sgpr(tmpSgpr), "0x3f803f80", "Packed alpha==1.0")
          module.addInst("s_cmp_eq_u32", sgpr("Alpha"), sgpr(tmpSgpr), "alpha == 1.0?")

        elif kernel["ProblemType"]["ComputeDataType"].isInt32():
          module.addInst("s_cmp_eq_u32", sgpr("Alpha"), "1", "Alpha == 1.0 ?")

        # This covers sgemm, bfgemm + HPA (b,b,b,b,s,s), and also hgemm (h,h,h,h,s,s)
        elif kernel["ProblemType"]["ComputeDataType"].isSingle():
          #module.addInst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
          module.addInst("s_cmp_eq_u32", sgpr("Alpha"), "1.0", "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["ComputeDataType"].isDouble():
          module.addInst("s_mov_b32", sgpr(tmpSgpr+0), 0x00000000, "Low part of double 1.0")
          module.addInst("s_mov_b32", sgpr(tmpSgpr+1), "0x3ff00000", "High part of double 1.0")
          module.addInst("s_cmp_eq_u64", sgpr("Alpha",2), sgpr(tmpSgpr,2), "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
          module.addInst("s_mov_b32", sgpr(tmpSgpr+0), "1.0", "Real part of 1.0")
          module.addInst("s_mov_b32", sgpr(tmpSgpr+1), "0.0", "Imaginary part of 1.0")
          module.addInst("s_cmp_eq_u64", sgpr("Alpha",2), sgpr(tmpSgpr,2), "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
          module.addInst("s_mov_b32", sgpr(tmpSgpr+0), "0x00000000", "lsb of real part of 1.0")
          module.addInst("s_mov_b32", sgpr(tmpSgpr+1), "0x3ff00000", "msb of real part of 1.0")
          module.addInst("s_cmp_eq_u64", sgpr("Alpha",2), sgpr(tmpSgpr,2), "Alpha.real == 1.0 ?")
          module.addInst("s_cbranch_scc0", skipOptNLL.getLabelName(), "branch if alpha.real != 1")
          module.addInst("s_mov_b32", sgpr(tmpSgpr+0), "0x00000000", "lsb of imag part of 0.0")
          module.addInst("s_mov_b32", sgpr(tmpSgpr+1), "0x00000000", "msb of imag part of 0.0")
          module.addInst("s_cmp_eq_u64", sgpr("Alpha+2",2), sgpr(tmpSgpr,2), "Alpha.imag == 0.0 ?")

        module.addInst("s_cbranch_scc0", skipOptNLL.getLabelName(), "branch if alpha != 1")
        module.addSpaceLine()

      module.addCode(self.checkIsEdge(kernel, tmpSgpr, skipOptNLL))
      module.addSpaceLine()

      # Check tail loop required:
      # Skip tail loop check if noTailLoop is true
      if not self.noTailLoop:
        loopChar = self.indexChars[ \
            kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
        module.addCode(scalarStaticDivideAndRemainder(tmpSgpr, tmpSgpr+1, "SizesSum+%u"%self.unrollIdx, \
                  kernel["DepthU"], tmpSgpr+2, 2))
        module.addInst("s_cmp_eq_u32", sgpr(tmpSgpr+1), \
            hex(0), "numIter%s == 0"%loopChar )
        module.addInst("s_cbranch_scc0", skipOptNLL.getLabelName(), "skip if tail loop required")

      # save the vgprPool for generating the normal path.
      # dump the 'dirty' pool upon s_endpgm and swap back the 'clean' pool
      # so we can avoid explicit vgpr check-in/out
      self.savedVgprPool = deepcopy(self.vgprPool)
      self.savedSgprPool = deepcopy(self.sgprPool)

      # comment out the following codes that attempt to reduce vgpr consumption
      # however, the kernel vgpr count is governed by peak vgpr consumption so saving
      # a few here shouldn't affect kernel's overall vgpr consumption.
      # the following code is for reference and will be removed in the future
      """
      added = [] # track registers added to pool
      if kernel["PrefetchGlobalRead"]:
        if not kernel["DirectToLdsA"]:
          added.append(self.vgprPool.addRange(self.startVgprG2LA, \
              self.startVgprG2LA+self.numVgprG2LA-1, "startOptNLL"))
          added.append(self.vgprPool.addRange(self.startVgprLocalWriteAddressesA, \
                       self.startVgprLocalWriteAddressesA, "startOptNLL"))
        if not kernel["DirectToLdsB"]:
          added.append(self.vgprPool.addRange(self.startVgprG2LB, \
              self.startVgprG2LB+self.numVgprG2LB-1, "startOptNLL"))
          added.append(self.vgprPool.addRange(self.startVgprLocalWriteAddressesB, \
                       self.startVgprLocalWriteAddressesB, "startOptNLL"))

      if kernel["BufferLoad"]:
        added.append(self.vgprPool.addRange(self.startVgprGlobalReadOffsetA, \
            self.startVgprGlobalReadOffsetB, "startOptNLL"))
      else:
        added.append(self.vgprPool.addRange(self.startVgprGlobalReadAddressesA, \
            self.startVgprGlobalReadAddressesB, "startOptNLL"))
      module.addComment1("reclaim VGPRS: " + ", ".join(added))
      """

    return module

  ##############################################################################
  def closeSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isNGLL):
    module = Code.Module("closeSumAtLeastUnroll")
    if not prefetch:
      if isNGLL:
        toPGR1 = Code.Label(self.labels.getName("toPGR1"), "")
        module.addCode(toPGR1)
      else:
        if isOptNLL:
            endSumLabel = "Summation_End_OptNLL"

            module.addComment0("Stores for OptNLL")
            module.addCode(self.endSummation(kernel, endSumLabel, isOptNLL))

            # perhaps could work with LSU>1 by adding other indices here, but not tested
            assert (kernel["LocalSplitU"] == 1)
            module.addCode(self.notLocalSplitUGlobalWriteIndices(kernel))

            # add stores for opt NLL
            (fullVw, elements) = self.notLocalFullTileElements(kernel, False)
            alpha = False
            beta = False
            module.addCode(self.globalWriteElements(kernel, [fullVw], [elements], applyAlpha=alpha, betas=[beta], edges=[False]))

            self.cleanupGlobalWrite(kernel)
            module.addSpaceLine()
            module.addCode(self.functionEnd(kernel, False))
            module.addCode(Code.Label("OptNLL_End", ""))

        else:
          module.addCode(Code.Label("PrefetchGlobalLastIterEnd", ""))

    # swap back vgpr pool if any
    if self.savedVgprPool != None:
      # in case pool size in current path is larger than pool size in main path
      # and it will miss allocate vgpr since allocating vgpr is based on pool size in main path
      oldSize = self.savedVgprPool.size()
      newSize = self.vgprPool.size()
      if newSize > self.savedVgprPool.size():
        for i in range(oldSize,newSize):
          self.savedVgprPool.pool.append(self.savedVgprPool.Register(RegisterPool.Status.Available,"restore vgprPool"))
      self.vgprPool = self.savedVgprPool # restore vgprPool before alternate path
      self.savedVgprPool = None
    # swap back sgpr pool if any
    if self.savedSgprPool != None:
      # in case pool size in current path is larger than pool size in main path
      # and it will miss allocate vgpr since allocating vgpr is based on pool size in main path
      oldSize = self.savedSgprPool.size()
      newSize = self.sgprPool.size()
      if newSize > self.savedSgprPool.size():
        for i in range(oldSize-1,newSize):
          self.savedSgprPool.pool.append(self.savedSgprPool.Register(RegisterPool.Status.Available,"restore sgprPool"))
      self.sgprPool = self.savedSgprPool # restore vgprPool before alternate path
      self.savedSgprPool = None
    return module

  ##############################################################################
  # incLower must be constant or SGPR unsigned value
  def incrementSrd(self, kernel, tP, incLower, incUpper, checkShadowLimitCopy=True):
    imod = Code.Module("incrementSrd")
    tc = tP["tensorChar"]

    imod.addInst("s_add_u32", \
         sgpr("Srd%s+0"%(tc)), \
         sgpr("Srd%s+0"%(tc)), \
         incLower, \
        "gra SRD += inc(lower)" )
    imod.addInst("s_addc_u32 ", \
         sgpr("Srd%s+1"%(tc)), \
         sgpr("Srd%s+1"%(tc)), \
         incUpper, \
         "gra SRD += inc(upper)" )

    # also have to move the boundary since we change the base
    # so less buffers to the edge:
    if self.use64bShadowLimit:
      imod.addInst("s_sub_u32", \
          sgpr("ShadowLimit%s+0"%tc), \
          sgpr("ShadowLimit%s+0"%tc), \
          incLower, \
            "limit -= inc)")
      imod.addInst("s_subb_u32", \
          sgpr("ShadowLimit%s+1"%tc), \
          sgpr("ShadowLimit%s+1"%tc), \
          incUpper, \
            "limit -= inc)" )
      if checkShadowLimitCopy:
        imod.addInst("s_cmp_eq_u32", sgpr("ShadowLimit%s+1"%tc), 0, "are we within 2^32?")
        if self.staggerU:
          # staggerU case, need to restore BufferLimit when ShadowLimit goes to negative value
          imod.addInst("s_cselect_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "BufferLimit", "Move shadow to real if we are within 2^32")
        else:
          imod.addInst("s_cmov_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "Move shadow to real if we are within 2^32")
    else:
      imod.addInst("s_sub_u32", \
           sgpr("Srd%s+2"%(tc)), \
           sgpr("Srd%s+2"%(tc)), \
           incLower, \
            "limit -= inc)" )
    return imod

  ##############################################################################
  # incLower must be constant or SGPR unsigned value
  def setTailSrd(self, kernel, tP, incLower):
    # In SuppressNoLoadLoop, the final loop iteration moves the SRD base forward
    # and the ShadowLimit backwards by one extra 'click' of GlobalReadIncs[AB].
    # Note the ShadowLimit may become negative - for example edge tiles where the
    # increment is > tile width.
    # The SuppressNoLoadLoop mode also forces the SRD limit to 0 on the final iteration.
    # The code here undoes the final click step by moving the base backwards and the
    # limit forwards (reading from the ShadowLimit).
    # It only works if use64bShadowLimit is enabled (since this enables use of the ShadowLimit)

    tc = tP["tensorChar"]
    module = Code.Module("setTailSrd")
    incUpper = 0

    module.addInst("s_sub_u32 ", \
         sgpr("Srd%s+0"%(tc)), \
         sgpr("Srd%s+0"%(tc)), \
         incLower, \
        "gra SRD -= inc(lower)" )
    module.addInst("s_subb_u32 ", \
         sgpr("Srd%s+1"%(tc)), \
         sgpr("Srd%s+1"%(tc)), \
         incUpper, \
        "gra SRD -= inc(upper)" )

    # using Shadow limit here which only works with 64-bit PBC:
    assert(self.use64bShadowLimit)

    module.addInst("s_add_u32", \
        sgpr("ShadowLimit%s+0"%tc), \
        sgpr("ShadowLimit%s+0"%tc), \
         incLower, \
          "limit -= inc)")
    module.addInst("s_addc_u32", \
        sgpr("ShadowLimit%s+1"%tc), \
        sgpr("ShadowLimit%s+1"%tc), \
         incUpper, \
          "limit -= inc)" )
    module.addInst("s_cmp_eq_u32", sgpr("ShadowLimit%s+1"%tc), 0, "are we within 2^32?")
    module.addInst("s_cmov_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "Move shadow to real if we are within 2^32")

    return module

  ##############################################################################
  # Global Read: Increment A/B
  # loopIdx is summation idx:
  #   self.unrollIdx, or an idx from 0..NumIndicesSummation
  # prefetchIndex is >0 (1...PrefetchGlobalRead) if this increment follows a
  #   global prefetch or 0 otherwise
  # incs is number of increments to perform
  ##############################################################################
  def globalReadIncrement(self, kernel, imod, loopIdx, tP, prefetchIndex, incs=1):
    if not self.do["GlobalInc"]: return ""
    tc = tP["tensorChar"]
    loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]

    imod.addComment1("global read inc %s loop%s"%(tc,loopChar))

    if kernel["BufferLoad"]:
      # TODO - does this handle N-dim tensors correctly?
      #if tP["isB"]:
      #  module.addInst("s_mov_b32", sgpr("OffsetB"), sgpr("SrdB+0"), "hack to save")
      if self.staggerU and loopIdx == self.unrollIdx:
        # add a wrap increment, if needed:
        incLower = self.getTmpSgpr(3).idx()
        incUpper = incLower + 1
        tmpS =    incLower + 2
        if prefetchIndex:
          imod.addInst("s_add_u32", sgpr(tmpS), self.loopCounter(kernel, self.unrollIdx), prefetchIndex, "remove pf(%u)"%prefetchIndex)
          imod.addInst("s_cmp_eq_u32",  sgpr("StaggerUIter"), sgpr(tmpS), "Is this wrapIter? (pf)")
        else:
          imod.addInst("s_cmp_eq_u32",  self.loopCounter(kernel, self.unrollIdx), \
                    sgpr("StaggerUIter"), "Is this the wrapIter?")
        imod.addInst("s_cselect_b32", sgpr(incLower), sgpr("WrapU%s+0"%tc), sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                    "incLower <- ?")
        imod.addInst("s_cselect_b32", sgpr(incUpper), sgpr("WrapU%s+1"%tc), 0,
                    "incUpper <- ?")
        imod.addCode(self.incrementSrd(kernel, tP, sgpr(incLower), sgpr(incUpper), checkShadowLimitCopy=True))
      else:
        if loopIdx != self.unrollIdx or (tc in ('A', 'B') and kernel["ProblemType"]["IndicesSummation"][self.unrollIdx] in kernel["ProblemType"]["MirrorDims%s"%tc]):
          incUpper = sgpr(self.getTmpSgpr(1).idx())
          # GRO may be negative for other summation if stride-other < stride-unroll or if mirror dim.
          imod.addInst("s_ashr_i32", incUpper, sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), 31, "sign-extend")
        else:
          incUpper = 0 # GRO is positive for loop unroll
        imod.addCode( self.incrementSrd(kernel, tP, sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), incUpper))
    else:
      graIdx = 0
      #for perp in range(0, tP["nrp"]):
      #  for para in range(0, tP["nrc"]):
      #    for s in range(0, tP["nrcv"]):
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
              if self.globalReadIncsUseVgpr:
                imod.addInst("_v_add_co_u32 ", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    vgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], 2*loopIdx)), \
                    "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
                imod.addInst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    vgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], 2*loopIdx)), \
                    self.vcc, \
                    "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
              else:
                imod.addInst("_v_add_co_u32 ", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    sgpr("GlobalReadIncs%s+%u"%(tP["tensorChar"], loopIdx)), \
                    "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
                imod.addInst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    0,
                    self.vcc, \
                    "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
              graIdx += self.rpga
      #module.addCode(dump(vgpr("GlobalReadAddrA+0")))
      #module.addCode(dump(vgpr("GlobalReadAddrA+1")))
      #module.addInst("s_endpgm")

  def globalReadIncrementAB(self, kernel, loopIdx, prefetchIndex, incs=1):
    imod = Code.Module("globalReadIncrementAB%s")
    problemType = self.kernel["ProblemType"]
    unrollLoopCounter = self.loopCounter(kernel, self.unrollIdx)

    incCodeA = imod.addCode(Code.Module("globalReadIncrementA"))
    incCodeB = imod.addCode(Code.Module("globalReadIncrementB"))

    if self.unrollIncIsDepthU and loopIdx==self.unrollIdx:
      loopCounter = self.loopCounter(kernel, self.unrollIdx)
      incCodeA.addInst("s_add_u32",
                   loopCounter, loopCounter,
                   "DepthU",  "increment psdIter")

    self.globalReadIncrement(kernel, incCodeA, loopIdx, self.tPA, prefetchIndex, incs)
    self.globalReadIncrement(kernel, incCodeB, loopIdx, self.tPB, prefetchIndex, incs)
    return imod

  ##############################################################################
  # Global Read:
  # globalReadGuardK is called for loads in the tail loop
  # Must ensure each load is in bounds - either using buffer bounds
  # or exec-mask checks.
  ##############################################################################
  def globalReadGuardK(self, kernel, tP, vregSetIdx):
    module = Code.Module("globalReadGuardK")
    tc = tP["tensorChar"]
    problemType = self.kernel["ProblemType"]
    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth

    ########################################
    # Calculate Max Addr
    ########################################

    tmpSgpr = self.getTmpSgpr(2).idx()
    maxAddrSgpr = tmpSgpr

    if not kernel["BufferLoad"]:
      module.addComment0("flat addressing - max read address = size[n] * stride[n-1]")
      dim = len(tP["ia"])-1 # dim
      sizeIdx = tP["ia"][dim]
      sizeIdxIsSum = sizeIdx in kernel["ProblemType"]["IndicesSummation"]
      if sizeIdxIsSum:
        sizeIdx -= kernel["ProblemType"]["NumIndicesC"]
      # TODO-multiply by largest stride
      module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(maxAddrSgpr+0), sgpr(maxAddrSgpr+1),  \
                  sgpr("Sizes%s+%u"%("Sum" if sizeIdxIsSum else "Free", sizeIdx)),  \
                  sgpr("Stride%s%s"%(tc, self.indexChars[tP['ia'][-1]])), \
                  "64b tensor%s size in elements"%tc))
      module.addInst("s_lshl_b64", \
        sgpr(maxAddrSgpr,2), \
        sgpr(maxAddrSgpr,2), \
        hex(log2(tP["bpe"])), "<- tensor%s size in bytes"%tc)

      module.addInst("s_add_u32", \
          sgpr(maxAddrSgpr+0), \
          sgpr(self.sgprs["AddressA"] if tP["isA"] else self.sgprs["AddressB"]), \
          sgpr(maxAddrSgpr+0), \
          "prepend address lower")
      module.addInst("s_addc_u32", \
          sgpr(maxAddrSgpr+1), \
          sgpr((self.sgprs["AddressA"] if tP["isA"] else self.sgprs["AddressB"])+1), \
          sgpr(maxAddrSgpr+1), \
          "prepend address upper")
      # sgpr->vgpr
      maxAddrVgpr = self.vgprPool.checkOutAligned(2, 2, "maxAddrVgpr")
      module.addInst("v_mov_b32", vgpr(maxAddrVgpr+0), sgpr(maxAddrSgpr+0), "sgpr->vgpr")
      module.addInst("v_mov_b32", vgpr(maxAddrVgpr+1), sgpr(maxAddrSgpr+1), "sgpr->vgpr")

      # full exec mask
      fullExec = tmpSgpr
      sgprCnt = self.laneSGPRCount
      waveSize = kernel["WavefrontSize"]
      activeMask = "0xFFFFFFFF" if (waveSize == 32) else "0xFFFFFFFFFFFFFFFF"
      module.addInst("s_mov_b{}".format(waveSize), sgpr(fullExec,sgprCnt), activeMask, "to restore all threads active")
      bpeVgpr = self.vgprPool.checkOut(1, "bpeVgpr")
      module.addInst("v_mov_b32", vgpr(bpeVgpr), hex(tP["bpe"]), "bpe")

      # can remove this?
      zeroVgpr = self.vgprPool.checkOut(1,"zeroVgpr")
      module.addInst("v_mov_b32", vgpr(zeroVgpr), hex(0), "zero")

    extraFields = ""
    if tP["NonTemporal"]%2==1:
      extraFields += " glc"
    if tP["NonTemporal"]//2==1:
      extraFields += " slc"
    if kernel["DirectToLds%s"%tc]:
      extraFields += " lds"

    directToLdsLoads = 0
    prevLdsOffset    = 0
    # print("tc={}, nrp={}, nrpv={}, nrc={}, nrcv/nrcvpi={}, sgprforGRO={}".format(tc, tP["nrp"], tP["nrpv"], tP["nrc"], tP["nrcv"]//tP["nrcvpi"], problemType["ZeroPad%s"%tc], kernel["UseSgprForGRO"]))

    instOffset = 0
    loopCnt = -1

    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
            i = sPara + (tP["nrcv"] // tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            loopCnt += 1
            graIdx = i * self.rpgo if kernel["BufferLoad"] else i * self.rpga
            g2lIdx = i * loadWidth

            destVgprHi = None
            dataIsI8 = False
            packInt8Code = None

            instOffsetInc = 0 # increment value for instOffset. Need to apply after r loop

            r = 0
            numLoadVectorComp = loadWidth*self.bpr//tP["bpe"]
            if kernel["ProblemType"]["DataType"].isDouble() and kernel["BufferLoad"]:
              # adjustment for dgemm + BufferLoad
              # use same buffer_load instruction for tail loop as out of tail loop
              # this is mandatory for DirectToLds case. Also, it improves tail loop performance.
              # so far, limit to double only
              numLoadVectorComp = numLoadVectorComp // kernel["GlobalLoadVectorWidth%c"%tc]

            int8TempVgpr = numLoadVectorComp - 1
            # for each component in vector
            while r < numLoadVectorComp:
              numElementsPerLoad = 1
              if kernel["ProblemType"]["DataType"].isInt8():
                # TODO-Int8, Check this:
                # if tP["glvw"]>1 and kernel["AssertSummationElementMultiple"] % 2 == 0:
                # # Pack two FP16 values into a single load dword x2
                #   numElementsPerLoad = 2
                # elif self.archCaps["HasEccHalf"]:
                #   destVgprHi = self.vgprPool.checkOut(1, 'destVgprHi')

                # Check out 3 regs once , for component 1,2,3 (r = 1,2,3)
                if r == 1:
                  packInt8Code = Code.Module()
                  destVgprHi = self.vgprPool.checkOut( int8TempVgpr , 'destVgprHi')
                dataIsI8 = True
                regIdx = r // 4
              elif kernel["ProblemType"]["DataType"].isHalf() or \
                 kernel["ProblemType"]["DataType"].isBFloat16():
                if tP["glvw"]>1 and kernel["AssertSummationElementMultiple"] % 2 == 0:
                # Pack two FP16 values into a single load dword x2
                  numElementsPerLoad = 2
                elif self.archCaps["HasEccHalf"]:
                  # In some cards, loading half types into register will zero out
                  # the other half. Therefore we need to load into a separate register
                  # then pack 2 registers into one
                  destVgprHi = self.vgprPool.checkOut(1, 'destVgprHi')
                regIdx = r // 2
              elif kernel["ProblemType"]["DataType"].isInt8x4() or \
                   kernel["ProblemType"]["DataType"].isSingle():
                regIdx = r
              elif kernel["ProblemType"]["DataType"].isDouble():
                numElementsPerLoad = kernel["GlobalLoadVectorWidth%c"%tc] # adjust numElementsPerLoad for DGEMM
                regIdx = r*2
              elif kernel["ProblemType"]["DataType"].isSingleComplex():
                regIdx = r*2
              elif kernel["ProblemType"]["DataType"].isDoubleComplex() :
                regIdx = r*4
              else:
                printWarning("DataType unsupported")
              module.addComment0("g2l=%u, load component %u"%(g2lIdx, r))

              offset = 0

              if kernel["BufferLoad"]:
                # Use buffer limit to stay in-bounds - the limit was set to edge when SRD initialized
                # and each increment of SRD base in the unroll loop does a corresponding decrement
                # of the srd limit - so base+limit stays constant and also points at maximum
                # element that should be accessed.
                if kernel["_UseSgprForGRO"]:
                  offsetVgpr = "GlobalReadOffset%s+0"%(tc)
                else:
                  offsetVgpr = "GlobalReadOffset%s+%u"%(tc, graIdx)

                # Vgpr for GRO
                if not kernel["_UseSgprForGRO"]:
                  soffset = "0"
                # instruction offset with Sgpr for GRO
                elif kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx))
                # Sgpr for GRO
                else:
                  soffset = "0" if graIdx == 0 else sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))

                unrollMirrorWithSoffset = kernel["ProblemType"]["IndicesSummation"][self.unrollIdx] in problemType["MirrorDims%s"%tc] and soffset != "0"
                # ScalarGlobalReadOffset should be negative value with unroll mirroring.
                # However, buffer_load uses soffset as uint value, so GRO - SGRO, SGRO = 0
                if unrollMirrorWithSoffset:
                  codeMod = Code.Module("mirrorIdx%u"%loopCnt)
                  codeMod.addInst("_v_sub_u32", vgpr(offsetVgpr), vgpr(offsetVgpr), soffset, "mirror unroll: GRO=GRO-SGRO, soffset=0")
                  module.addCode(codeMod)
                  soffset_prev = soffset
                  soffset = "0"

                if kernel["DirectToLds%s"%tc]:
                  # need to increment ldsInc only once per each loopCnt
                  # this is pre count up, so increment it at r == 0
                  if r == 0:
                    ldsInc = (self.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]
                  else:
                    ldsInc = 0
                  if kernel["LdsBlockSizePerPad%s"%tc] != 0:
                    ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
                  else:
                    padInterval = (self.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.bpr
                    ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpe"]
                  #print("ldsInc", ldsInc)
                  #print("GlobalLoadVectorWidth", kernel["GlobalLoadVectorWidth%c"%tc])
                  #print("bpr", self.bpr)
                  if kernel["UseInstOffsetForGRO"]:
                    # buffer_load only support 12 bit instruction offset
                    # we have to increase m0 if offset is larger thant 12 bits
                    if instOffset >= self.buff_load_inst_offset_max:
                      inc = (instOffset // self.buff_load_inst_offset_max) * self.buff_load_inst_offset_max
                      module.addInst("s_add_u32", mgpr(0), mgpr(0), inc, "Move LDS write address to next base" )
                      instOffset -= inc
                  elif directToLdsLoads != 0 and ldsInc > 0:
                      if tP["nrc"] > 1:
                        # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                        divisorName = tP["lvc"]
                        divisor = kernel[divisorName]
                        # DirectToLds + NumLoadsCoalesced>1 case, need to adjust m0 increment value to store values to correct location in LDS
                        wSize = max(self.kernel["WavefrontSize"], divisor)
                        lscaOffset = para * wSize * tP["bpe"] * tP["glvw"]
                        ldsOffset = ldsInc * tP["nrc"] * (sPerp + tP["nrpv"] * perp) + lscaOffset
                        ldsInc = ldsOffset - prevLdsOffset
                        prevLdsOffset = ldsOffset
                      module.addInst("s_add_u32", mgpr(0), mgpr(0), ldsInc, "Move LDS write address to next line" )

                  destVgpr=0
                elif kernel["DirectToVgpr%s"%tc]:
                  numVgprG2L = self.numVgprG2LA if tP["isA"] else self.numVgprG2LB
                  numVgprPerBlock = numVgprG2L // 2 # numVgprG2L is doubled for DirectToVgpr
                  idx = g2lIdx + vregSetIdx * numVgprPerBlock
                  destVgpr="G2L%s+%u+%u"%(tc, idx, regIdx)
                else:
                  destVgpr="G2L%s+%u+%u"%(tc, g2lIdx, regIdx)

                offset = r * tP["bpe"] + instOffset
                hi8 = 0
                hi16 = 0
                comment = "load one buffer value"
                if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
                  if numElementsPerLoad==2:
                    # Pack two FP16 values into a single load dword x2
                    r += 1 # skip next element since we loaded 2X here
                    comment = "load packed 2X half buffer value"
                  elif not kernel["DirectToLds%s"%tc]:
                    hi16=loopCnt%2 if tP["glvw"]==1 else r%2
                    comment="load one buffer value"

                if kernel["ProblemType"]["DataType"].isInt8():
                  # TODO-Int8, Check this:
                  # if numElementsPerLoad==2:
                  #   # Pack two FP16 values into a single load dword x2
                  #   r += 1 # skip next element since we loaded 2X here
                  #   comment = "load packed 2X half buffer value"
                  if not kernel["DirectToLds%s"%tc]:
                    hi8  = (loopCnt%4) %2 if tP["glvw"]==1 else (r%4) %2
                    hi16 = (loopCnt%4)//2 if tP["glvw"]==1 else (r%4)//2
                    comment="load one buffer value"

                bpl = numElementsPerLoad*self.bpeAB # bytesPerLoad

                # if hi8=1 or hi16=1 (component 1,2,3 for int8) or (component 1 for half), use the temp destVgprHi
                # but only when hi16=1 we use the _d16_hi version instruction, see the below visualized int8 comment
                loadVgpr = destVgprHi if ((hi16 or hi8) and destVgprHi != None) else destVgpr
                if kernel["ProblemType"]["DataType"].isInt8() and (not self.archCaps["HasEccHalf"]):
                  module.addInst("v_mov_b32", vgpr(loadVgpr), 0, "set to zero to avoid unexpected value")
                module.addCode(self.chooseGlobalRead(True, \
                          bpl, destVgpr=loadVgpr, \
                          addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 4), \
                          soffset=soffset, offset=offset, \
                          extraFields=extraFields, \
                          hi16=hi16, \
                          comment=comment))

                if unrollMirrorWithSoffset:
                  codeMod = Code.Module("mirrorIdx%u"%loopCnt)
                  codeMod.addInst("_v_add_u32", vgpr(offsetVgpr), vgpr(offsetVgpr), soffset_prev, "mirror unroll: restore GRO=GRO+SGRO")
                  module.addCode(codeMod)

                if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  instOffsetInc += ldsInc
                # print("  bpl={}, destVgpr={}, soffset={}, offset={}, hi16={}".format(bpl, destVgpr, soffset, offset, hi16))

              else: # Not buffer load, ie 'flat' load
                # mask if current address if in bounds
                module.addInst("_v_cmpx_lt_u64", self.vcc, \
                    vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), \
                    vgpr(maxAddrVgpr,2), \
                    "addr < maxAddr")
                hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and r%2==1
                destVgpr="G2L%s+%u+%u"%(tc, g2lIdx, regIdx)
                # load one element from address
                module.addCode(self.chooseGlobalRead(False, \
                          self.bpeAB, destVgpr=destVgprHi if (hi16 and destVgprHi != None) else destVgpr, \
                          addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                          soffset=0, offset=0, \
                          extraFields=extraFields, \
                          hi16=hi16, \
                          comment="load one flat value"))

                # restore full exec mask
                module.addInst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), self.vcc, sgpr(fullExec,self.laneSGPRCount), \
                    "all threads active")

                # increment address by 1 element (BPE)
                module.addInst("_v_add_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    vgpr(bpeVgpr), "gra += 1 (lower)")
                module.addInst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    vgpr(zeroVgpr), \
                    self.vcc, \
                    "gra += 1 (upper)")

              # int8 byte:
              # |--------|--------|--------|---V0---|, r = 0, hi8=0, hi16=0, load d16
              # |--------|--------|--------|---V1---|, r = 1, hi8=1, hi16=0, load d16
              # |--------|---V2---|--------|--------|, r = 2, hi8=0, hi16=1, load d16_hi
              # |--------|---V3---|--------|--------|, r = 3, hi8=1, hi16=1, load d16_hi
              # V1, V3 -> shift left 8 bits, or 4 regs (pack)
              # DestV0|=(V1 << 8), DestV0|= V2, DestV0|=(V3 << 8)
              # Int8 (byte)
              if dataIsI8 and (destVgprHi != None):
                # hi8  -> r = 1,3
                # hi16 -> r = 2,3
                if hi8 or hi16:
                  # r = 1,2,3, vmcnt needed for one packing
                  packInt8Code.addInst("s_waitcnt", "vmcnt(%u)"%(int8TempVgpr-r), "" )
                if hi8:
                  # r = 1,3,   shift needed
                  packInt8Code.addInst("v_lshlrev_b32", vgpr(destVgprHi), "0x8", vgpr(destVgprHi), "shift left to higher 8 bits")
                if hi8 or hi16:
                  # r = 1,2,3, packing
                  packInt8Code.addInst("v_or_b32", vgpr(destVgpr), vgpr(destVgpr), vgpr(destVgprHi), "pack a sub 8-bit with dest")
                destVgprHi += 1

              # Half
              elif destVgprHi != None and r % 2 == 1:
                module.addInst("s_waitcnt", "vmcnt(0)", "")
                module.addInst("v_or_b32", vgpr(destVgpr), vgpr(destVgpr), vgpr(destVgprHi), "HasEccHalf: pack")

              # For half (bf16). Note: for int8, we will checkin after loading all components
              if (destVgprHi != None) and (not dataIsI8):
                self.vgprPool.checkIn(destVgprHi)
                destVgprHi = None

              r += 1 # next component (for half, byte)

            # end R loop

            instOffset += instOffsetInc # add increment value for instOffset. Need to apply after r loop
            # increment once per r loop (at the end)
            directToLdsLoads+=1

            # for int8:
            # we do the 3 packs, and checking the 3 extra vgprs after loading all components
            if dataIsI8:
              assert packInt8Code != None and destVgprHi != None
              module.addCode(packInt8Code)
              self.vgprPool.checkIn(destVgprHi - int8TempVgpr)
              destVgprHi = None

    if self.db["ConservativeWaitCnt"] & 0x1:
        module.addInst("s_barrier",  "debug")
        module.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "")
        if self.archCaps["SeparateVscnt"]:
          module.addInst("s_waitcnt_vscnt", "null", "0", "")
        module.addInst("s_barrier",  "debug")
        #module.addCode(self.getCmpAssert(self.asmAssert.lt, vgpr("Serial"), 64)) # examine second wavefront

    # TODO - can remove one of these m0 restores if A and B both TLU
    if kernel["DirectToLds%s"%tP["tensorChar"]]:
      module.addInst("s_mov_b32", mgpr(0), \
          hex(kernel["LdsNumElements"] * tP["bpe"]), \
          "Restore LDS clamp at %u bytes"%(kernel["LdsNumElements"] * tP["bpe"]))

    if not kernel["BufferLoad"]:
      self.vgprPool.checkIn(maxAddrVgpr)
      self.vgprPool.checkIn(bpeVgpr)
      self.vgprPool.checkIn(zeroVgpr)

    return module

  ##############################################################################
  # DirectToLds M0 update: Do It A/B
  ##############################################################################
  def directToLdsM0Update(self, kernel, mode, tP, usePlaceHolder=False):
    tc = tP["tensorChar"]
    imod = Code.Module("directToLdsM0Update%s_%u"%(tc,mode))
    DtldsModule = imod.addCode(Code.Module("dtls_offset%s"%tP["tensorChar"]))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return imod
    if kernel["DirectToLds%s"%tc]:
      # DirectToLds only enabled for TLU=1 cases, where the registers are directly copied into LDS
      # for cases both A&B are DTLS, updating m0 for each GlobalRead requires instruction schedule
      # along with global reads
      assert (kernel["LocalWriteUseSgpr%s"%tc])
      if kernel["ExpandPointerSwap"]:
        DtldsModule.addInst("s_add_u32", mgpr(0), sgpr("LocalWriteAddr%s"%tc), \
                      tP["localWriteSwapByteOffset"], "m0 <- LDS write address")
      else:
        DtldsModule.addInst("s_mov_b32", mgpr(0), sgpr("LocalWriteAddr%s"%tc), "m0 <- LDS write address")

      # PrefetchGlobalRead=2 case, generate local read wait for DirectToLds
      if kernel["PrefetchGlobalRead"]==2:
        # do not generate local read wait for PGR=2
        DtldsModule.addComment0("before DirectToLds load, ensure prior ds_reads have finished")
        DtldsModule.addInst("s_waitcnt", "lgkmcnt(0)", "")
        if not kernel["NoLdsWriteCode"]:
          if usePlaceHolder:
            waitStr = "__placeholder__"
          else:
            waitStr = "0"
          DtldsModule.addInst("s_waitcnt", "vmcnt(%s)"%waitStr, "")
        DtldsModule.addInst("s_barrier", "")

    return imod

  ##############################################################################
  # Global Read: Do It A/B
  ##############################################################################
  def globalReadDo(self, kernel, mode, tP, vregSetIdx=0):
    tc = tP["tensorChar"]
    problemType = self.kernel["ProblemType"]
    imod = Code.StructuredModule("globalReadDo%s_%u"%(tc,mode))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return imod

    # sizeK % LOCAL_DEPTHU
    guardK = (mode==2)

    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth # load width in elements?
    bpl = self.bpeAB * tP["glvw"] # bytes per load
    instOffset = 0

    loopIdx = self.unrollIdx # TODO - does this handle multiple summation indices?
    if kernel["SuppressNoLoadLoop"]:
      if mode==1 and tP["isA"]:
        imod.header.addInst("s_cmp_eq_i32", \
              self.loopCounter(kernel, loopIdx), \
              "%u"% 1, \
              "%s"%"is this the last iteration")
        imod.header.addInst("s_cmov_b32", \
              sgpr("SrdA+2"), \
              0,
              "Set limit to 0 for last iteration")
        imod.header.addInst("s_cmov_b32", \
              sgpr("SrdB+2"), \
              0,
              "Set limit to 0 for last iteration")

    # set the first tc for below wait code for DirectToLds
    # if DirectToVgprA is enabled, change the first to B
    tc1st = 'A'
    if kernel["DirectToVgprA"]:
      tc1st = 'B'

    if tc == tc1st and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) and not kernel["PrefetchGlobalRead"]==2:
      # generate local read wait for DirectToLds except for PrefetchGlobalRead=2 (for PGR=2, generate wait after m0 value setting)
      imod.header.addComment0("before DirectToLds load, ensure prior ds_reads have finished")
      if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]): # do not generate sync here if DirectToVgpr is enabled
        imod.header.addCode("s_waitcnt", "lgkmcnt(0)", "")
      else:
        imod.header.addCode(self.syncThreads(kernel))


    if guardK:
      imod.middle.addCode(self.globalReadGuardK(kernel, tP, vregSetIdx))
      return imod

    # else not-guardK below:

    extraFields = ""
    if tP["NonTemporal"]%2==1:
      extraFields += " glc"
    if tP["NonTemporal"]//2==1:
      extraFields += " slc"
    if kernel["DirectToLds%s"%tc]:
      extraFields += " lds"

    directToLdsLoads = 0
    instOffset       = 0
    prevLdsOffset    = 0

    loopCnt = -1
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
            i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            loopCnt += 1
            graIdx = i * self.rpgo if kernel["BufferLoad"] else i * self.rpga
            g2lIdx = i * loadWidth
            # Each load may contains a small bundle of instructions, package them together in loadModule:
            loadModule = Code.Module("load%u"%loopCnt)
            imod.middle.addCode(loadModule)

            if kernel["BufferLoad"]:
              if kernel["_UseSgprForGRO"]:
                offsetVgpr= "GlobalReadOffset%s+0"%(tc)
              else:
                offsetVgpr= "GlobalReadOffset%s+%u"%(tc, graIdx)

              # vgpr for GRO
              if not kernel["_UseSgprForGRO"]:
                soffset = "0"
              # instruction offset with Sgpr for GRO
              elif kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx))
              # Sgpr for GRO
              else:
                soffset = "0" if graIdx == 0 else sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))

              unrollMirrorWithSoffset = kernel["ProblemType"]["IndicesSummation"][self.unrollIdx] in problemType["MirrorDims%s"%tc] and soffset != "0"
              # ScalarGlobalReadOffset should be negative value with unroll mirroring.
              # However, buffer_load uses soffset as uint value, so GRO - SGRO, SGRO = 0
              if unrollMirrorWithSoffset:
                codeMod = Code.Module("mirrorIdx%u"%loopCnt)
                codeMod.addInst("_v_sub_u32", vgpr(offsetVgpr), vgpr(offsetVgpr), soffset, "mirror unroll: GRO=GRO-SGRO, soffset=0")
                loadModule.addCode(codeMod)
                soffset_prev = soffset
                soffset = "0"

              if kernel["DirectToLds%s"%tc]:
                # use bpe with GlobalLoadVectorWidth
                ldsInc = (self.kernel["WavefrontSize"] * kernel["GlobalLoadVectorWidth%c"%tc] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"] * kernel["GlobalLoadVectorWidth%c"%tc]) * tP["bpe"]
                if kernel["LdsBlockSizePerPad%s"%tc] != 0:
                  ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
                else:
                  padInterval = (self.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.bpr
                  ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpe"]

                if kernel["UseInstOffsetForGRO"]:
                  # buffer_load only support 12 bit instruction offset
                  # we have to increase m0 if offset is larger thant 12 bits
                  if instOffset >= self.buff_load_inst_offset_max:
                    inc = (instOffset // self.buff_load_inst_offset_max) * self.buff_load_inst_offset_max
                    loadModule.addInst("s_add_u32", mgpr(0), mgpr(0), inc, "Move LDS write address to next base" )
                    instOffset -= inc
                elif directToLdsLoads != 0:
                  # m0 offset conversion (only for UseInstOffsetForGRO == 0)
                  # in tP["glvw"] == 1 and tP["nrc"] > 1 case, only m0 offset conversion is necessary. row and column index conversion is not necessary.
                  if tP["nrc"] > 1:
                    # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                    divisorName = tP["lvc"]
                    divisor = kernel[divisorName]
                    # DirectToLds + NumLoadsCoalesced>1 case, need to adjust m0 increment value to store values to correct location in LDS
                    wSize = max(self.kernel["WavefrontSize"], divisor)
                    lscaOffset = para * wSize * tP["bpe"] * tP["glvw"]
                    ldsOffset = ldsInc * tP["nrc"] * (sPerp + tP["nrpv"] * perp) + lscaOffset
                    ldsInc = ldsOffset - prevLdsOffset
                    prevLdsOffset = ldsOffset
                  loadModule.addInst("s_add_u32", mgpr(0), mgpr(0), ldsInc, "Move LDS write address to next line" )
                directToLdsLoads+=1
                destVgpr=0
              elif kernel["DirectToVgpr%s"%tc]:
                # DirectToVgpr case. Need to toggle destination vreg set and adjust instOffset
                destVgpr="G2L%s%u+%u"%(tc, vregSetIdx, g2lIdx)
              else:
                destVgpr="G2L%s+%u"%(tc, g2lIdx)

              # TODO: is it possible to load only hi16 when no in tail? (need to check INT8 too)
              loadModule.addCode( self.chooseGlobalRead(kernel["BufferLoad"], \
                        bpl, destVgpr=destVgpr, \
                        addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 4), \
                        soffset=soffset, offset=instOffset, \
                        extraFields=extraFields, \
                        hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and loopCnt%2==1, \
                        comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp)))

              if unrollMirrorWithSoffset:
                codeMod = Code.Module("mirrorIdx%u"%loopCnt)
                codeMod.addInst("_v_add_u32", vgpr(offsetVgpr), vgpr(offsetVgpr), soffset_prev, "mirror unroll: restore GRO=GRO+SGRO")
                loadModule.addCode(codeMod)

              if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  instOffset += ldsInc

              #print "IM=", type(imod.instList[-1]), imod.instList[-1],
            else: # not buffer load
              # load one element from address
              if kernel["DirectToVgpr%s"%tc]:
                # DirectToVgpr case. Need to toggle destination vreg set and adjust instOffset
                destVgpr="G2L%s%u+%u"%(tc, vregSetIdx, g2lIdx)
              else:
                destVgpr="G2L%s+%u"%(tc, g2lIdx)
              loadModule.addCode( self.chooseGlobalRead(False, \
                        bpl, \
                        destVgpr=destVgpr, \
                        addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                        soffset=0, offset=0, \
                        extraFields=extraFields, \
                        hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and loopCnt%2==1, \
                        comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp )))

    if self.db["ConservativeWaitCnt"] & 0x1:
        imod.footer.addInst( "s_barrier", "debug")
        imod.footer.addInst( "s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "conservative wait")
        if self.archCaps["SeparateVscnt"]:
          imod.footer.addInst( "s_waitcnt_vscnt", "null", "0", "stores")
        imod.footer.addInst( "s_barrier", "debug")
        #module.addCode(self.getCmpAssert(self.asmAssert.lt, vgpr("Serial"), 64)) # examine second wavefront

    # TODO - can remove one of these m0 restores if A and B both TLU
    if kernel["DirectToLds%s"%tP["tensorChar"]] and not (mode == 1 and kernel["PrefetchGlobalRead"]==2):
      inst = "s_mov_b32"
      dst = mgpr(0)
      src = hex(kernel["LdsNumElements"] * tP["bpe"])
      comment = "Restore LDS clamp at %u bytes"%(kernel["LdsNumElements"] * tP["bpe"])
      # PGR=2 case, footer is located before global read. To avoid setting clamp before global read, store lds clamp code in middle
      if kernel["PrefetchGlobalRead"] == 2:
        imod.middle.addInst(inst, dst, src, comment)
      else:
        imod.footer.addInst(inst, dst, src, comment)

    return imod

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP):
    if not self.do["LocalWrite"]: return Code.Module("localWriteSwapOffsets (No local write)")
    if kernel["1LDSBuffer"]: return Code.Module("localWriteSwapOffsets (Empty)")
    module = Code.Module("localWriteSwapOffsets")
    tc = tP["tensorChar"]
    #fixme-iui  need to use wrapping increment for double or triple buffering:
    if internalPointerSwap:
      tP["localWriteSwapByteOffset"] = 0 if tP["localWriteSwapByteOffset"] else kernel["LdsOffsetA_Blk"]*tP["bpe"]
      module.addComment1("(EPS=1) local write swap internal offset -> %u" % tP["localWriteSwapByteOffset"])
    else:
      if kernel["LocalWriteUseSgpr%s"%tc]:
        module.addInst("s_xor_b32", \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "swap Red Blk SGPR")
      elif not kernel["DirectToVgpr%s"%tc]: # no local write code if DirectToVgpr is enabled
        numLwa = self.numVgprLocalWriteAddressesA if tP["isA"] else self.numVgprLocalWriteAddressesB
        for i in range(0,numLwa):
          module.addInst("v_xor_b32", \
              vgpr("LocalWriteAddr%s+%u"%(tc,i)), \
              hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
              vgpr("LocalWriteAddr%s+%u"%(tc,i)), \
              "swap Red Blk")
    return module

  ##############################################################################
  # Local Write: Reset Offsets A/B
  # used for global-read + tail-loop to reset to writing in red
  ##############################################################################
  def localWriteResetOffsets(self, kernel, internalPointerSwap, tP):
    tc = tP["tensorChar"]
    if not self.do["LocalWrite"]: return Code.Module("localWriteResetOffsets (no local write)")
    if kernel["1LDSBuffer"] or kernel["DirectToVgpr%s"%tc]: # no local write code if DirectToVgpr is enabled
      return Code.Module("localWriteResetOffsets (Empty)")
    module = Code.Module("localWriteResetOffsets")
    resetMask = hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]-1 | self.LdsOOB)
    if internalPointerSwap:
      tP["localWriteSwapByteOffset"] = 0
    else:
      if kernel["LocalWriteUseSgpr%s"%tc]:
        module.addInst("s_and_b32", \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            resetMask, \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "reset to Red")
      else:
        module.addInst("v_and_b32", \
            vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            resetMask, \
            vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "reset to Red")
    return module

  ##############################################################################
  # Calculate offset to use for LDS write
  # Intro:
  #   Each WI has a 2D tile index (coal, perp).
  #     - Code above computes global mem address by scaling one dim by the
  #       lda and adding the other.
  #     - Here we compute a linear LDS offset by scaling one dim by the MT
  #       dim and adding the other.
  #   Result is we map a tile from global memory into LDS.  Consecutive LDS
  #   locations contain elements from different summation 'rows' - therefore
  #   loading a row of LDS will feed computations for different C tile indices.
  # Notes:
  #   Total load insts is nrc * nrp which load the macro-tile.
  #   Par and coalesced are ~synonyms referring to same dimension
  #   Either nrpv or nrvc must be 1 - can't have vectors in both dimensions.
  #     Thus either sPerp or sPara is 0.
  # Inputs:
  #   perp : index of the load in perp dimension (0...nrp)
  #   par  : index of the load in the para dim (0...nrc)
  #   sPerp : component index of the perp vector (0...nrpv)
  #   sPara : component index of the par vector (0...nrcv)
  # Outputs:
  #   offsetBytes : Offset in bytes for the _ds_store instruction
  #   i : i-th instruction
  #   comment : Comment with the text version of the formula
  #############################################################################
  def calculateLdsWriteOffset(self, perp, para, sPerp, sPara, kernel, tP, localWriteCnt):
    tc = tP["tensorChar"]
    mask = 0
    #print "tc ", tc, " perp ", perp, " para ", para, " sPerp ", sPerp, " sPara ", sPara
    lscaOffset = para * kernel[tP["lsc"]]
    perp_masked = perp
    perp_rem = 0
    lspaOffset = perp_masked * kernel[tP["lsp"]]
    rem = 0

    # Add component offset to interleave from different regs
    # and compute mysterious "i"
    assert(sPerp==0 or sPara==0)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset += sPerp & mask
      lscaOffset += sPara
      rem = (sPerp & ~mask)
      i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp_masked))
      #print "nrcv ", tP["nrcv"], " nrcvpi ", tP["nrcvpi"], " nrc ", tP["nrc"], " nrpv ", tP["nrpv"]
    else:
      lscaOffset += sPara
      lspaOffset += sPerp
      rem = 0
      i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para * tP["glvw"] + tP["nrc"] * (sPerp + tP["glvw"] * tP["nrpv"] * perp ))

    #if not tP["tlu"]:
    #  tmp = sPara
    #  sPara = sPerp
    #  sPerp = tmp
    # print("0lspaOffset", lspaOffset)
    # print("0lscaOffset", lscaOffset)

    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    lds_stride = (kernel["_DepthULds"] + LdsPad) if kernel["UnrollMajorLDS%s" % tP["tensorChar"]] \
            else (kernel[tP["mt"]] + LdsPad)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset *= lds_stride
      lspaOffset += rem + perp_rem
    else:
      lscaOffset *= lds_stride
      lscaOffset += rem

    # print("1lspaOffset", lspaOffset)
    # print("1lscaOffset", lscaOffset)
    #if tP["tlu"]:
    #  lspaOffset *= tP["glvw"]
    #  lscaOffset *= tP["glvw"]

    # print("2lspaOffset", lspaOffset)
    # print("2lscaOffset", lscaOffset)
    offsetElements = (lspaOffset + lscaOffset)
    # print("offsetElements", offsetElements)
    offsetBytes = offsetElements*tP["bpe"]

    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
      offsetBytes = offsetBytes + (offsetBytes // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]

    offsetBytes += tP["localWriteSwapByteOffset"]

    #print("offsetBytes", offsetBytes)
    #print "offset", offset

    comment = "lwo%s_%u_%u_%u_%u = (%s%d*%s)" \
        % (tP["tensorChar"], \
        para, sPara, perp, sPerp, \
        (("%u + "%sPara) if tP["wtc"] else ""), \
        para, tP["lsc"] )
    if not tP["tlu"]:
      comment += "*(MT%s+PAD)" % (tP["tileChar"])
    comment += " + (%s%d*%s)" % (
        (("%u + "%sPerp) if tP["wuc"] else ""), perp, \
        tP["lsp"])
    if tP["tlu"]:
      comment += "(*MT%s+PAD)" % (tP["tileChar"])
    comment += " = %u" % (offsetBytes)

    return (offsetBytes, i, comment)

  def recalcLocalWriteAddresses(self, kernel, tP, uDu):

    tc = tP["tensorChar"]

    module = Code.Module("recalcLocalWriteAddresses")
    module.addComment1("recalculate LocalWriteAddr{}".format(tc))

    lwvw = getattr(self, "localWriteWidth{}".format(tc))
    newInstIdx = self.selectMemoryInstruction("LocalWrite", lwvw*kernel["DepthULdsDivisor"], \
        False, \
        self.localWrite2CoalescedA, self.localWrite2PerpendicularA,
        [self.localWriteStrideTileA, self.localWriteStrideUnrollA] )
    tP["localWriteInstruction"] = self.memoryInstructions["LocalWrite"][newInstIdx]

    # global read tile assignment
    module.addCode(self.graTileAssignment(kernel, tP))
    # global read tile offsets
    module.addCode(self.graTileOffsets(kernel, tP))
    # global read unroll offsets
    module.addCode(self.graUnrollOffsets(kernel, tP))
    # still needed for vgpr resource management
    # intentionally not emitting code
    self.graFinalOffsets(kernel, tP)

    # local write tile assignments
    module.addCode(self.lwaTileAssignment(kernel, tP))
    # local write unroll assignments
    module.addCode(self.lwaUnrollAssignment(kernel, tP))
    # local write local write first offsets
    module.addCode(self.lwaFirstOffset(kernel, tP, uDu))

    return module

  def recalcLocalReadAddressesAB(self, kernel):
    imod = Code.Module()

    if self.inTailLoop:
      # it do 1 iteration each loop in tail loop, and is no use to wider local read next iteration.
      # In 1 block MI, it remap localReadAddr in order to let each thread wider local read continuous k
      # this decrease performance since it require more loop to handle continuous k in each thread.
      # recalculate localReadAddr to cancel wider local read in tail loop
      # TODO: If DepthULdsDivisor>1, local read addr is incremented for each K the loop iterates, which
      # upon second sub-loop needs to be reset to its original value. Backing up local read address would
      # be nicer than recomputing them
      if kernel.enabledSplitLDS or ((self.numReadsIterCoalescedA > 1 or self.numReadsIterCoalescedB > 1) and kernel["MatrixInstB"] == 1): #and tP["isB"]:
        self.numReadsIterCoalescedA = 1
        self.numReadsIterCoalescedB = 1
        self.lrvwA = kernel["MIInputPerThread"]
        self.lrvwB = kernel["MIInputPerThread"]

        imod.addCode(self.lraTileAssignment(kernel, self.tPA, self.tPB))
        imod.addCode(self.lraFinalOffset(kernel, self.tPA))
        imod.addCode(self.lraDeclareAddresses(kernel, self.tPA))
        imod.addCode(self.lraFinalOffset(kernel, self.tPB))
        imod.addCode(self.lraDeclareAddresses(kernel, self.tPB))
        localRead2Perpendicular = False
        instructions = self.memoryInstructions

        localReadWidth = self.tPA["bpe"] / self.bpr
        if kernel["UnrollMajorLDSA"]:
          localReadWidth = (kernel["MIInputPerThread"] * self.tPA["bpe"]) // self.bpr
        self.localReadInstructionIdxA = \
          self.selectMemoryInstruction("LocalRead", localReadWidth, \
          False, \
          self.localRead2CoalescedA, localRead2Perpendicular,
          [self.localReadStrideCoalescedA] )
        self.localReadInstructionA = instructions["LocalRead"][self.localReadInstructionIdxA]

        localReadWidth = self.tPB["bpe"] / self.bpr
        if kernel["UnrollMajorLDSB"]:
          localReadWidth = (kernel["MIInputPerThread"] * self.tPB["bpe"]) // self.bpr
        self.localReadInstructionIdxB = \
          self.selectMemoryInstruction("LocalRead", localReadWidth, \
          False, \
          self.localRead2CoalescedB, localRead2Perpendicular,
          [self.localReadStrideCoalescedB] )
        self.localReadInstructionB = instructions["LocalRead"][ \
          self.localReadInstructionIdxB]

        self.tPA["localReadInstruction"] = self.localReadInstructionA
        self.tPB["localReadInstruction"] = self.localReadInstructionB
    return imod

  ##############################################################################
  # Local Write in Prefetch Pass (PreLoop): Do It A/B
  ##############################################################################
  def preLoopLocalWriteDo(self, kernel, tPA, tPB):
    imod = Code.Module()

    LWDoMod = imod.addCode(Code.Module())
    LWDoA = self.localWriteDo(kernel, tPA)
    LWDoB = self.localWriteDo(kernel, tPB)
    LWDoMod.addComment1("local write a")
    LWDoMod.addCode(LWDoA)
    LWDoMod.addComment1("local write b")
    LWDoMod.addCode(LWDoB)
    return imod

  ##############################################################################
  # Local Write: Do It A/B
  # uDu: 'None' means to use fractional local write (where not all threads are active)
  #      when DepthULdsDivisor > 1
  ##############################################################################
  def localWriteDo(self, kernel, tP, uDu=0):
    if not self.do["LocalWrite"]: return "", -1

    tc = tP["tensorChar"]
    self.localWriteDoCnt += 1
    imod = Code.Module()

    if (not kernel["DirectToLds%s"%tc]) and (not kernel["DirectToVgpr%s"%tc]):
      instruction = tP["localWriteInstruction"]
      numBlocks = instruction.numBlocks
      numOffsets = instruction.numOffsets
      blockWidth = instruction.blockWidth
      #offsetMultiplier = instruction.offsetMultiplier
      g2lIdx = 0
      #module.addCode(dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"])))
      if 0:
        print("\nLocalWrite", tP["tensorChar"])
        print("tlu", tP["tlu"])
        print("lsc", kernel[tP["lsc"]])
        print("lsp", kernel[tP["lsp"]])
        print("wtc", tP["wtc"])
        print("wuc", tP["wuc"])
        print("nrc", tP["nrc"])
        print("nrp", tP["nrp"])
        print("nwcv", tP["nwcv"])
        print("nwpv", tP["nwpv"])
        print("nrcvpi", tP["nrcvpi"])
        print("nwcvpi", tP["nwcvpi"])

      tmpLocalWriteAddr = -1

      # using _ds_store_b8: need one more vgpr space to do lshr
      tmpVgprOffset = ((self.numVgprG2LA if (tP['tensorChar'] == 'A') else self.numVgprG2LB) / 2) if (blockWidth == 0.25) else 0

      loopCnt = 0
      # if transposing, positions of sPerp and sPara are transposed
      instructionCnt = -1
      for perp in range(0, tP["nrp"]):
        instructionCnt += 1
        localWriteCode = imod.addCode(Code.Module("LocalWrite%u perp=%d"%(instructionCnt,perp)))
        lwa = "LocalWriteAddr%s"%tc  # default

        for para in range(0, tP["nrc"]):
          if para>=1:
            localWriteCode = imod.addCode(Code.Module("LocalWrite%u perp=%d para=%d"%(instructionCnt,perp,para)))

          for s in range(0, max(tP["nwcv"],tP["nwpv"])//tP["nwcvpi"]):
            sPerp = 0
            sPara = 0
            if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
              if tP["wtc"]:
                sPerp = s
              elif tP["wuc"]:
                sPara = s
            else:
              if tP["wtc"]:
                sPara = s
              elif tP["wuc"]:
                sPerp = s

            #print("perp:{}/{} para:{}/{} sPerp:{} sPara:{} loopCnt:{}".format(perp,tP["nrp"],para,tP["nrc"],sPerp,sPara,loopCnt))
            (offset, i, comment) = self.calculateLdsWriteOffset(perp, para, sPerp, sPara, kernel, tP, loopCnt)

            if uDu is None:
              g2lIdx = int(i * blockWidth)
            else:
              # Example: DepthULdsDivisor=2
              # v0, v1, v2, v3 | v0, v1, v2, v3 | ... ----> unroll dim
              # -----Thd 0----- -----Thd 1-----   ...
              # 1st subloop writes v0,v1 to LDS
              # 2nd subloop writes v2,v3 to LDS
              g2lIdx = int((i * kernel["DepthULdsDivisor"] + uDu) * blockWidth)
              #print("uDu=%u, g2lIdx = %u, offset: %u"%(uDu, g2lIdx, offset))

            # TODO- INT8: check uDu
            if (blockWidth == 0.25) and ((s % 4) == 0):
                src = "G2L%s+%u" % (tc, g2lIdx)
                dst = "G2L%s+%u+%u" % (tc, tmpVgprOffset, g2lIdx)
                localWriteCode.addInst("v_mov_b32", vgpr(dst), vgpr(src), "another VGPR storing lshr 8-bit value")
                localWriteCode.addInst("v_lshrrev_b32", vgpr(dst), "0x8", vgpr(dst), "G2L Vpgr >> 8")

            paramList = []
            paramList.append(vgpr(lwa))
            for blockIdx in range(0, numBlocks):
              if blockWidth == 1:
                paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx)))
              elif blockWidth == 0.25 and ((s % 2) == 1): # Int8, s = 1 or 3 (high8Bits)
                paramList.append(vgpr("G2L%s+%u+%u"%(tc, tmpVgprOffset, g2lIdx)))
              else:
                paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), blockWidth))
              if self.db["ForceInputValue%s"%tc]:
                localWriteCode.addInst("v_mov_b32", vgpr("G2L%s+%u"%(tc, g2lIdx)), self.db["ForceValue%s"%tc], "ForceInputValue")

            for oIdx in range(0, numOffsets):
              paramList.append(offset)

            #print "offset", offset

            paramTuple = tuple(paramList)
            #comment = "Reg -> L %u_%u_%u_%u"%(para, sPara, perp, sPerp)
            #comment += " #%u"%self.localWriteDoCnt
            nonTemporal = 0
            isHigh16Bits = False
            if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
              if s%2==1:
                isHigh16Bits = True
              if tP["glvw"]==1 and instructionCnt%2==1:
                isHigh16Bits = True

            #       |  hi16  |  hi16  |        |        |
            #       |  hi8   |        |   hi8  |        |
            #############################################
            # VGPR: |---w4---|---w3---|---w2---|---w1---| -> b8_d16: get w1 / _b8_d16_hi: get w3
            # LSHR: |--------|---w4---|--------|---w2---| -> b8_d16: get w2 / _b8_d16_hi: get w4
            elif kernel["ProblemType"]["DataType"].isInt8():
              isHigh16Bits = (s % 4) > 1 # 2,3
              # TODO
              # if tP["glvw"]==1 and instructionCnt%2==1:
              #   isHigh16Bits = True
            localWriteCode.addCode(Code.LocalWriteInst( \
                instruction.IssueLatency, \
                tP["localWriteInstruction"].toCodeInst(paramTuple, \
                nonTemporal, isHigh16Bits),comment))

            loopCnt+=1
      if tmpLocalWriteAddr != -1:
        self.vgprPool.checkIn(tmpLocalWriteAddr)

    # localWriteDoCnt<=2 is prefetch if PrefetchGlobalRead:
    if 0 and tP["isB"]: # post-lds-write
    #if 0 and self.localWriteDoCnt >= 0:
      localWriteCode.addInst( "s_waitcnt lgkmcnt(0) & vmcnt(0)", "")
      if self.archCaps["SeparateVscnt"]:
        localWriteCode.addInst( "s_waitcnt_vscnt", "null", "0", "")
      localWriteCode.addInst("s_barrier", "dump LDS" )
      localWriteCode.addCode(self.getCmpAssert(self.asmAssert.ne, sgpr("WorkGroup0"),1))
      #localWriteCode.addCode(self.getBomb())

    return imod

  ##############################################################################
  # Local Read: Swap Offsets A/B
  # internalPointerSwap: swap internally tracked offsets - rather than
  #    emit specific instructions to do the pointer swap
  ##############################################################################
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    tc=tP["tensorChar"]
    if (not self.do["LocalRead%s"%tc]) or kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
      return Code.Module("localReadSwapOffsets (no local read)")
    if kernel["1LDSBuffer"]:
      return Code.Module("localReadSwapOffsets (Empty)")
    module = Code.Module("localReadSwapOffsets")
    if internalPointerSwap:
      tP["localReadSwapByteOffset"] = 0 if tP["localReadSwapByteOffset"] else kernel["LdsOffsetA_Blk"]*tP["bpe"]
      module.addComment1("local read swap internal offset -> %u" % tP["localReadSwapByteOffset"])
    else:
      module.addInst("v_xor_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "swap Red Blk")
    return module

  ##############################################################################
  # Local Read: Reset Offsets A/B
  # x % n == n & (n-1) for n power of 2
  # tP[localReadOffset] maintains running count of offsets
  # This is called from the tail loop to reset read offsets?
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return Code.Module("localReadResetOffsets (no local read)")
    if kernel["1LDSBuffer"] or kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
      return Code.Module("localReadResetOffsets (Empty)")
    module = Code.Module("localReadResetOffsets")
    if tP["localReadInstruction"].numOffsets == 1:
      tP["localReadSwapByteOffset"] = 0
      module.addComment1("localReadResetOffsets")
      tP["localReadOffset"] = 0
      module.addComment0("handled internally")
    module.addInst("v_and_b32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]-1), \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "reset Red,Blk -> Red")
    return module

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  def localReadInitPointers(self, kernel, tP):
    tc=tP["tensorChar"]
    if (not self.do["LocalRead%s"%tc]) or kernel["DirectToVgpr%s"%tc]:# no local read code if DirectToVgpr is enabled
      return Code.Module("localReadInitPointers (Empty)")
    module = Code.Module("localReadInitPointers")
    if self.localReadInstructionA.numOffsets == 1:
      module.addComment1("localReadInitPointers")
      tP["localReadOffset"] = 0
    else:
      module.addInst("v_and_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffset%s_Blk"%tP["tensorChar"]]*tP["bpe"]-1), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "init Red,Blk -> Red")
    return module

  ##############################################################################
  # Local Read offset conversion for DirectToLds
  ##############################################################################
  def localReadOffsetConvForDTL(self, kernel, tP, offset_val):
    tc = tP["tensorChar"]
    bit2 = offset_val & 4
    bit3 = offset_val & 8
    bit4 = offset_val & 16
    bit5 = offset_val & 32
    if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 8):
      # dword_x2 case
      # (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
      newVal = (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
    else:  #if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 16):  # most preferred case
      # dword_x4 case
      # (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
      newVal = (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
    offset_val = offset_val & (~0x3c)
    offset_val = offset_val | newVal
    return offset_val

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  def localReadInc(self, kernel, iui, tP):
    tc = tP["tensorChar"]
    if not self.do["LocalRead%s" % tc] or kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
      return Code.Module("localReadInc (Empty)")

    module = Code.Module("localReadInc")

    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0

    if self.inTailLoop:
      inc = kernel["LocalSplitU"] * (kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad) * tP["bpe"]
      comment = " (LSU*(MT+PAD)*bpe)"
      if kernel["EnableMatrixInstruction"]:
        matrixInstK = kernel["MatrixInstK"]
        if kernel["UnrollMajorLDS%s" % tc]:
          if kernel["DirectToLds%s" % tc] and kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"] > 4:
            # DirectToLds special case. Need special address coonversion
            localReadOffset = kernel["LocalSplitU"] * kernel["MatrixInstK"] * max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)
            localReadOffset *= tP["bpe"]
            prev_offset_val = 0 if iui == 0 else localReadOffset * iui
            offset_val = localReadOffset * (iui + 1)
            # offset conversion or DirectToLds
            prev_offset_val= self.localReadOffsetConvForDTL(kernel, tP, prev_offset_val)
            offset_val= self.localReadOffsetConvForDTL(kernel, tP, offset_val)
            inc = offset_val - prev_offset_val
            matrixInstK = 1 # multiplying matrixInstK is not necessary
            comment = ""
          else:
            inc = kernel["LocalSplitU"] * tP["bpe"]
            comment = " (LSU*bpe)"
        inc *= matrixInstK
      tmpSgpr = self.getTmpSgpr(1).idx()
      module.addInst("s_mov_b32", sgpr(tmpSgpr), hex(inc), "inc")
      module.addInst("_v_add_co_u32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          self.vcc, \
          sgpr(tmpSgpr), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "lr%s += %u%s"%(tP["tensorChar"], inc, comment) )
    else:
      if tP["localReadInstruction"].numOffsets == 1:
        if kernel["EnableMatrixInstruction"]:
          if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
            tP["localReadOffset"] += kernel["LocalSplitU"] * kernel["MatrixInstK"] * max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)
          else:
            if tc == "A":
              if kernel["MatrixInstB"] != 1 or self.lrvwA == self.lrvwB:
                tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * kernel["MatrixInstK"] * self.numReadsIterCoalescedA
              else:
                if (self.localReadDoCntA)%(kernel["LocalReadVectorWidth"]//self.lrvwA):
                  tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * self.lrvwA
                else:
                  tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*kernel["LocalReadVectorWidth"]//self.lrvwA-self.lrvwA*(kernel["LocalReadVectorWidth"]//self.lrvwA-1))
            else:
              if kernel["MatrixInstB"] != 1 or self.lrvwA == self.lrvwB:
                tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * kernel["MatrixInstK"] * self.numReadsIterCoalescedB
              else:
                if (self.localReadDoCntB)%(kernel["LocalReadVectorWidth"]//self.lrvwB):
                  tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * self.lrvwB
                else:
                  tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*kernel["LocalReadVectorWidth"]//self.lrvwB-self.lrvwB*(kernel["LocalReadVectorWidth"]//self.lrvwB-1))
        else:
          tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad)
        module.addComment0("N/A, lro->%d" % tP["localReadOffset"])
        module.addComment0("self.localReadDoCntA %d self.localReadDoCntB %d" % (self.localReadDoCntA,self.localReadDoCntB))
      else:
        inc = kernel["LocalSplitU"] * (kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad)
        module.addInst("_v_add_co_u32", \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            self.vcc, \
            hex(inc), \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            "lr%s += %u (LSU+(MT+Pad)*bpe"%(tP["tensorChar"], inc) )

    return module

  ##############################################################################
  # Local Read: Do It A/B
  # iui = Inner Unroll Idx
  # uIdx - Unroll Idx
  # epsi = expand pointer swap index. Only used for PAP
  ##############################################################################
  def localReadDo(self, kernel, bufferIdx, iui, epsi, tP):

    if not self.do["LocalRead%s" % tP["tensorChar"]]:
      imod = Code.Module("LocalReadDo%s_I%s" % (tP["tensorChar"], iui))
      pack = Code.Module("pack%s_I%s" % (tP["tensorChar"], iui))
      return imod, pack

    component = Component.LocalRead.find(self)
    if component:
      return component(self, bufferIdx, iui, epsi, tP)

  ##############################################################################
  # Save the local read pointers, for example when creating a duplicated
  # optimized path (like optNLL)
  ##############################################################################
  def saveLocalPointers(self, kernel):
    self.tPA["savedLocalReadOffset"] = self.tPA["localReadOffset"]
    self.tPB["savedLocalReadOffset"] = self.tPB["localReadOffset"]
    self.savedLocalReadDoCntA = self.localReadDoCntA
    self.savedLocalReadDoCntB = self.localReadDoCntB
    if kernel["ExpandPointerSwap"]:
      self.tPA["savedLocalWriteSwapByteOffset"] = self.tPA["localWriteSwapByteOffset"]
      self.tPB["savedLocalWriteSwapByteOffset"] = self.tPB["localWriteSwapByteOffset"]

  ##############################################################################
  # Restore the saved local read pointers
  # Must be paired with an earlier call to savePointers
  ##############################################################################
  def restoreLocalPointers(self, kernel):
    self.tPA["localReadOffset"] = self.tPA["savedLocalReadOffset"]
    self.tPB["localReadOffset"] = self.tPB["savedLocalReadOffset"]
    self.localReadDoCntA = self.savedLocalReadDoCntA
    self.localReadDoCntB = self.savedLocalReadDoCntB
    if kernel["ExpandPointerSwap"]:
      self.tPA["localWriteSwapByteOffset"] = self.tPA["savedLocalWriteSwapByteOffset"]
      self.tPB["localWriteSwapByteOffset"] = self.tPB["savedLocalWriteSwapByteOffset"]

  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponents(self, kernel, tP):
    component = Component.ShiftVectorComponents.find(self)
    if component:
      return component(self, kernel, tP)

  ##############################################################################
  # Complex Declare Tmp Registers - SKIP
  ##############################################################################
  def complexDeclareTmpRegisters(self, kernel):
    module = Code.Module("complexDeclareTmpRegisters")
    return module

  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  def localSplitULocalWrite(self, kernel):
    module = Code.Module("localSplitULocalWrite")
    # wait for summation to be done with lds before writing reduction values
    module.addCode(self.syncThreads(kernel, "pre-lsu local write"))

    tmpVgpr = self.vgprPool.checkOutAligned(2, 2, "tmpVgpr")
    lr0 = self.vgprPool.checkOut(1,"lr0")
    lr1 = self.vgprPool.checkOut(1,"lr1")
    sg = self.vgprPool.checkOut(1,"sg")
    copy = self.vgprPool.checkOut(1,"copy")
    tmpSgpr = self.getTmpSgpr(1).idx()

    # lr0 = serial % SG0
    module.addCode(vectorStaticDivideAndRemainder(lr1, lr0, "Serial", \
        kernel["SubGroup0"], tmpVgpr, tmpSgpr))

    # lr1 = (serial / SG0) % SG1
    # sg  = (serial / SG0) / SG1
    module.addInst("v_mov_b32", vgpr(copy), vgpr(lr1), "copy for divide")
    module.addCode(vectorStaticDivideAndRemainder(sg, lr1, copy, \
        kernel["SubGroup1"], tmpVgpr, tmpSgpr))

    # lr0 *= VW
    module.addInst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["VectorWidth"]*self.bpeCinternal), "VW")
    module.addInst("v_mul_lo_u32", vgpr(lr0), sgpr(tmpSgpr), vgpr(lr0), \
        "lr0 *= VW")
    # lr1 *= VW*MT0
    module.addInst("s_mov_b32", sgpr(tmpSgpr), \
        hex(kernel["VectorWidth"]*kernel["MacroTile0"]*self.bpeCinternal), "VW*MT0")
    module.addInst("v_mul_lo_u32", vgpr(lr1), sgpr(tmpSgpr), vgpr(lr1), \
        "lr1 *= VW*MT0")
    # sg  *= MT0*MT1
    module.addInst("s_mov_b32", sgpr(tmpSgpr), \
        hex(kernel["MacroTile0"]*kernel["MacroTile1"]*self.bpeCinternal), "MT0*MT1")
    module.addInst("v_mul_lo_u32", vgpr(sg), sgpr(tmpSgpr), vgpr(sg), \
        "sg *= MT0*MT1")

    # thread offset
    addr = lr0
    module.addInst("_v_add_co_u32", vgpr(addr), self.vcc, vgpr(lr1), vgpr(addr),  "")
    module.addInst("_v_add_co_u32", vgpr(addr), self.vcc, vgpr(sg), vgpr(addr),  "threadOffset")
    self.vgprPool.checkIn(lr0)
    self.vgprPool.checkIn(lr1)
    self.vgprPool.checkIn(sg)
    self.vgprPool.checkIn(copy)
    self.vgprPool.checkIn(tmpVgpr)

    # dump addr
    # module.addInst(dump(vgpr(addr)))

    # do writes
    # LDS Layout example (for Sgemm, LSU=4, TT=8x8, WG=[8,4,4]), 128 WI/WG
    # VectorWidth = GlobalWriteVectorWidth = 4
    # SubGroup0 (WI:00-32)  : LDS 0x0000-
    # SubGroup1 (WI:33-64)  : LDS 0x2000-
    # SubGroup2 (WI:65-95)  : LDS 0x4000-
    # SubGroup3 (WI:96-127) : LDS 0x6000-

    # Interleave within a subgroup is interesting...
    #       Start LDS Addr
    # WI00 - 0x000
    # WI01 - 0x010
    # ...
    # WI07 - 0x070
    # WI08 - 0x400
    # WI09 - 0x410
    # ...
    # WI0F - 0x470
    # WI10 - 0x800
    # ...
    # ...
    # WI1f - 0xc70
    # WI20 - 0x1000  (start SubGroup1)

    # so a zoom-in on the pattern at beginning of LDS, for the case above:
    #   WI (hex) |x00-|x01-|...   |x07-|0x0-|0x1-|...|0x7-|0x0-| ... ... ||0x8-|
    # ValuC      |0123|0123|...   |0123|4567|4567|...|4567|89AB| ... ... ||0123
    #            |                     |                  |               |
    # LDS Addr  0x0                  0x80               0x100           0x400

    bytesPerElem = kernel["ProblemType"]["ComputeDataType"].numBytes()
    regsPerElem  = kernel["ProblemType"]["ComputeDataType"].numRegisters()
    bytesPerVector = kernel["VectorWidth"] * bytesPerElem
    bytesPerStep = min(bytesPerVector, 16) # max length of ds inst is 16 bytes(128bits)
    regsPerStep  = int((bytesPerStep+3)//4)
    elementStep = bytesPerStep // bytesPerElem

    for j in range(0, kernel["ThreadTile1"]//kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]//kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          for vc in range(0, kernel["VectorWidth"], elementStep):
            # for half, write 2 elements (4 bytes)
            # for single, write 1 element (4 bytes)
            # double doesn't work yet
            writeOffset = vc \
                + i*kernel["SubGroup0"]*kernel["VectorWidth"] \
                + s*kernel["MacroTile0"] \
                + j*kernel["MacroTile0"]*kernel["SubGroup1"]*kernel["VectorWidth"]
            regIdx = vc \
                + i*kernel["VectorWidth"] \
                + s*kernel["ThreadTile0"] \
                + j*kernel["ThreadTile0"]*kernel["VectorWidth"]
            regIdx = int(regIdx * regsPerElem)

            module.addInst(f"_ds_store_b{bytesPerStep*8}", vgpr(addr), vgpr("ValuC+%u"%regIdx, regsPerStep), \
                "offset:%u"%(writeOffset*self.bpeCinternal), "j=%u i=%u s=%u vc=%u"%(j,i,s,vc))

    module.addInst("s_waitcnt", "lgkmcnt(0)", "wait for all writes")
    if self.archCaps["SeparateVscnt"]:
      module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
    module.addCode(self.syncThreads(kernel, "post-lsu local write"))
    # module.addCode(self.dumpLds(kernel, 0, 16))
    #module.addCode(self.getBomb(5))
    return module

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    # alloc resource
    tmpSgpr  = self.getTmpSgpr(1).idx()
    baseAddr = self.vgprPool.checkOut(1,"baseAddr")

    # calculate parameters
    bytesPerElem = kernel["ProblemType"]["ComputeDataType"].numBytes()
    regsPerElem  = kernel["ProblemType"]["ComputeDataType"].numRegisters()
    bytesPerVector = kernel["GlobalWriteVectorWidth"] * bytesPerElem
    bytesPerStep = 16
    while (bytesPerVector % bytesPerStep) != 0:
      bytesPerStep //= 2
    regsPerStep  = int((bytesPerStep+3)//4)
    elementStep = bytesPerStep // bytesPerElem

    # generate source
    module = Code.Module("localSplitULocalRead")
    module.addCode(staticMultiply(vgpr(baseAddr), vgpr("Serial"), kernel["GlobalWriteVectorWidth"]*self.bpeAB, sgpr(tmpSgpr)))
    # Load values for each subgroup
    for r in range(0, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"], elementStep):
          offset = s + i*kernel["NumThreads"]*kernel["GlobalWriteVectorWidth"] + r * kernel["MacroTile0"]*kernel["MacroTile1"]
          regIdx = int((s + i*kernel["GlobalWriteVectorWidth"] + r*kernel["GlobalWriteVectorWidth"]*kernel["NumGlobalWriteVectorsPerThread"]) * regsPerElem)
          module.addInst(f"_ds_load_b{bytesPerStep*8}", vgpr("ValuC+%u"%regIdx,regsPerStep), vgpr(baseAddr), \
              "offset:%u"%(offset*self.bpeCinternal), "r=%u i=%u s=%u"%(r,i,s))
    module.addInst("s_waitcnt", "lgkmcnt(0)", "wait for all reads")

    if self.archCaps["SeparateVscnt"]:
      module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

    # free resources
    self.vgprPool.checkIn(baseAddr)

    return module

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  def localSplitUReduction(self, kernel):
    module = Code.Module("localSplitUReduction")

    is_non_hpa_fp16 = kernel["ProblemType"]["DataType"].isHalf() and (not kernel["ProblemType"]["HighPrecisionAccumulate"])
    elementStep = 2 if is_non_hpa_fp16 else 1
    regsPerElem = kernel["ProblemType"]["DataType"].numRegisters()

    for r in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"], elementStep):
          cIdx = int((s + i * kernel["GlobalWriteVectorWidth"]) * regsPerElem)
          regIdx = int((s + i * kernel["GlobalWriteVectorWidth"] + r * kernel["GlobalWriteVectorWidth"] * kernel["NumGlobalWriteVectorsPerThread"]) * regsPerElem)

          if is_non_hpa_fp16:
            module.addInst("v_pk_add_f16", vgpr("ValuC+%u"%cIdx), vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), \
                         "c[%u] += c[%u]"%(cIdx, regIdx) )
          elif kernel["ProblemType"]["DataType"].isInt8x4():
            module.addInst("_v_add_i32", vgpr("ValuC+%u"%cIdx), vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), \
                         "c[%u] += c[%u]"%(cIdx, regIdx))

          elif kernel["ProblemType"]["DataType"].isSingle():
            module.addInst("v_add_f32", vgpr("ValuC+%u"%cIdx), vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), \
                         "c[%u] += c[%u]"%(cIdx, regIdx))
          elif kernel["ProblemType"]["DataType"].isDouble():
            module.addInst("v_add_f64", vgpr("ValuC+%u"%cIdx,2), vgpr("ValuC+%u" % regIdx,2), vgpr("ValuC+%u"%cIdx,2), \
                         "c[%u] += c[%u]"%(cIdx, regIdx))
          elif kernel["ProblemType"]["DataType"].isSingleComplex():
            module.addInst("v_add_f32", vgpr("ValuC+%u"%(cIdx+0)), vgpr("ValuC+%u" % (regIdx+0)), vgpr("ValuC+%u"%(cIdx+0)), \
                         "c[%u] += c[%u], real part"%(cIdx, regIdx) )
            module.addInst("v_add_f32", vgpr("ValuC+%u"%(cIdx+1)), vgpr("ValuC+%u" % (regIdx+1)), vgpr("ValuC+%u"%(cIdx+1)), \
                         "c[%u] += c[%u], imaginary part"%(cIdx+1, regIdx+1) )
          elif kernel["ProblemType"]["DataType"].isDoubleComplex():
            module.addInst("v_add_f64", vgpr("ValuC+%u"%(cIdx+0),2), vgpr("ValuC+%u" % (regIdx+0),2), vgpr("ValuC+%u"%(cIdx+0),2), \
                         "c[%u] += c[%u], real part"%(cIdx, regIdx) )
            module.addInst("v_add_f64", vgpr("ValuC+%u"%(cIdx+2),2), vgpr("ValuC+%u" % (regIdx+2),2), vgpr("ValuC+%u"%(cIdx+2),2), \
                         "c[%u] += c[%u], imaginary part"%(cIdx+2, regIdx+2) )
          else:
            # TODO: hpa_half, int8
            assert(0) # unsupported data type, need to modify here and LSU write/read code
    return module

  ##############################################################################
  # computeStoreSrd
  # Add tile assignment fields to store srd
  # This is based on WG not the WI/TT assignment
  ##############################################################################
  def computeStoreSrdStart(self, kernel):
    module = Code.Module("computeStoreSrdStart")

    tmpS0 = self.getTmpSgpr(3).idx()
    tmpS1 = tmpS0+1
    wgMT1 = tmpS0+2

    # Compute and save wg1*MT1 - the element offset that is top of the macro-tile in output space
    assert kernel["BufferStore"]
    module.addSpaceLine()
    module.addInst("s_mul_i32", \
        sgpr(wgMT1), \
        "MT1", \
        sgpr("WorkGroup1"), \
        "<- wg1*MT1")

    # Overall strategy is to set the SRD to the top-left of the macro-tile.
    # TT offsets are from this base (and include the column)

    # In non-packed mode:
    # higher-order tensor dims are static since this kernel operates within
    # the 2D Tensor formed by Index0 and Indexa.
    # Index0 and Index1 vary for each work-item (aka 'dynamic') so roll these into the VGPR

    # In packed mode:
    # Higher-order dimensions may be packed into coord0 / coord1 - see rowstart calculation below

    # Walk through addressing components (each tensor index) in C
    # For static dims add to SrdC / SrdD to compute a new base.
    # For dynamic (based on TT assignment) - save in coutRowPtr in computeStoreVgprs,
    # which saves the TT assignment for each WI scaled by StrideC0
    # TODO - future opportunities for store vgpr and other optimization
    #  - coutRowPtr and tid1 are strongly related - can we merge or remove one of these?
    # Packed follows same philosophy but may have more vector components
    indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
    numDim = len(indices)
    addrSrcSgpr = "Address" # use "Address" only for the first iteration
    for i in range(1, numDim):
      if i == kernel["ProblemType"]["Index0"]:
        # Used if the output is transposed?
        addToSrd = False
      elif i == kernel["ProblemType"]["Index1"] and len(kernel["PackedC1IndicesX"]) == 1:
        coord = sgpr(wgMT1)
        addToSrd = True
      elif i != kernel["ProblemType"]["Index0"] and i != kernel["ProblemType"]["Index1"] and not isPackedIndex(kernel, i):
        # group index, this is higher-order Tensor dimension, just add to SRD base:
        isStridedBuffer = kernel["ProblemType"]["StridedBatched"] or kernel["_GlobalAccumulation"]
        coord = sgpr("WorkGroup2") if isStridedBuffer else None
        addToSrd = True if isStridedBuffer else False
      else:
        # could be packed higher-order index, just ignore
        coord = None
        addToSrd = False

      if addToSrd:
        # These are constant across all workitems, just add to the SRD:
        strideC = "StrideC%s"%self.indexChars[i]
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(strideC), "CScale %s by Stride"%coord))
        module.addInst("s_lshl_b64", sgpr(tmpS0,2), sgpr(tmpS0,2), log2(self.bpeCexternal), "scale by bpe")

        module.addInst("s_add_u32",  sgpr("SrdC+0"), sgpr("%sC+0"%addrSrcSgpr), sgpr(tmpS0), "add lo to SRD")
        module.addInst("s_addc_u32", sgpr("SrdC+1"), sgpr("%sC+1"%addrSrcSgpr), sgpr(tmpS1), "add hi to SRD")

        # These are constant across all workitems, just add to the SRD:
        stride = "StrideD%s" % (self.indexChars[i])
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(stride), "Scale %s by Stride"%coord))
        module.addInst("s_lshl_b64", sgpr(tmpS0,2), sgpr(tmpS0,2), log2(self.bpeCexternal), "scale by bpe")

        module.addInst("s_add_u32",  sgpr("SrdD+0"), sgpr("%sD+0"%addrSrcSgpr), sgpr(tmpS0), "add lo to SRD")
        module.addInst("s_addc_u32", sgpr("SrdD+1"), sgpr("%sD+1"%addrSrcSgpr), sgpr(tmpS1), "add hi to SRD")

        module.addSpaceLine()

        addrSrcSgpr = "Srd" # update src Sgpr for the second or later iterations

    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      # GSU algorithm 2: adjust output buffer address to per GSU buffer
      tmpSgpr = self.getTmpSgpr(5).idx()
      module.addComment("GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s")
      module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpSgpr+0), sgpr(tmpSgpr+1), sgpr("SizesFree+0"), sgpr("GSUSumIdx"), "Free0"))
      for i in range(1, numDim):
        module.addInst("s_sub_u32",  sgpr(tmpSgpr+4), sgpr("SizesFree+%u"%i), 1, "Free%u" % i)
        module.addInst("s_mul_i32",  sgpr(tmpSgpr+4), sgpr(tmpSgpr+4), sgpr("GSUSumIdx"), "Free%u" % i)
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpSgpr+2), sgpr(tmpSgpr+3), sgpr(tmpSgpr+4), sgpr("StrideC%s"%self.indexChars[i]), "Free%u" % i))
        module.addInst("s_add_u32",  sgpr(tmpSgpr+0), sgpr(tmpSgpr+0), sgpr(tmpSgpr+2), "Free%u" % i)
        module.addInst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), sgpr(tmpSgpr+3), "Free%u" % i)
      module.addInst("s_lshl_b64", sgpr(tmpSgpr+0,2), sgpr(tmpSgpr+0,2), log2(self.bpeCexternal), "scale by bpe")
      module.addInst("s_add_u32",  sgpr("SrdD+0"), sgpr("SrdD+0"), sgpr(tmpSgpr+0), "add lo GSU offset to SRD")
      module.addInst("s_addc_u32", sgpr("SrdD+1"), sgpr("SrdD+1"), sgpr(tmpSgpr+1), "add hi GSU offset to SRD")

    for cdir in (0,1):
      indices = kernel["PackedC%uIndicesX"%cdir]
      packedSizes = "PackedSize%u"%cdir
      if len(indices) > 1:
        for i,idx in enumerate(indices[1:]):
          if i==0:
            module.addInst("s_mul_i32", sgpr(packedSizes), self.sizeRef(indices[0]), \
                      self.sizeRef(idx), "first packed size")
          else:
            module.addInst("s_mul_i32", sgpr(packedSizes), sgpr(packedSizes), \
                      self.sizeRef (idx), "first packed size")

    return module

  ##############################################################################
  # computeStoreVgprs
  # Compute workitem/TT offsets in VGPRS
  # and coord0/coord1
  # tid0Scale specifies the number of output elements in 0/coalesced dim
  # that should be written by each work-item in each batch element.
  ##############################################################################
  def computeStoreVgprs(self, kernel, divisor, tid0Scale, tid1Scale):
    module = Code.Module("computeStoreVgprs")
    module.addComment0("computeStoreVgprs")
    component = Component.ComputeStoreVgprs.find(self)
    if component:
      module.addCode(component(self, kernel, divisor, tid0Scale, tid1Scale)) #FIXME
    return module

  ##############################################################################
  # globalWriteWorkGroupInitBeforePersistentLoop:
  ##############################################################################
  def globalWriteWorkGroupInitBeforePersistentLoop(self, kernel):
    module = Code.Module("globalWriteWorkGroupInitBeforePersistentLoop")
    if kernel["BufferStore"]:
      module.addCode(self.allocPostLoopSrd(kernel, "D"))
      module.addCode(self.allocPostLoopSrd(kernel, "C"))
    return module

  ##############################################################################
  # globalWriteWorkGroupInit:
  ##############################################################################
  def globalWriteWorkGroupInit(self, kernel):
    module = Code.Module("globalWriteWorkGroupInit")
    if kernel["BufferStore"]:
      module.addCode(self.allocPostLoopSrd(kernel, "D"))
      module.addCode(self.allocPostLoopSrd(kernel, "C"))
      module.addCode(self.computeStoreSrdStart(kernel))
    return module

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    module = Code.Module("localSplitUGlobalWriteIndices")

    # lr0 = serial % SG0
    module.addCode(self.computeStoreVgprs(kernel, \
              divisor = kernel["MacroTile0"] // kernel["GlobalWriteVectorWidth"], \
              tid0Scale=kernel["GlobalWriteVectorWidth"], \
              tid1Scale=1))

    if kernel["BufferStore"]:
      #print "----AddressC-LocalSplitU"
      #print self.vgprPool.state()
      self.addrD = -1
      self.addrC = -1
    else:
      self.addrD = self.vgprPool.checkOut(2)
      module.addInst("v_mov_b32", \
          vgpr(self.addrD+0), \
          sgpr("AddressD+0"), \
          "sgpr -> vgpr")
      module.addInst("v_mov_b32", \
          vgpr(self.addrD+1), \
          sgpr("AddressD+1"), \
          "sgpr -> vgpr")
      self.addrC = self.vgprPool.checkOut(2)
      module.addInst("v_mov_b32", \
          vgpr(self.addrC+0), \
          sgpr("AddressC+0"), \
          "sgpr -> vgpr")
      module.addInst("v_mov_b32", \
          vgpr(self.addrC+1), \
          sgpr("AddressC+1"), \
          "sgpr -> vgpr")

    return module

  ##############################################################################
  def allocPostLoopSrd(self, kernel, ch):
    module = Code.Module("allocPostLoopSrd")
    # Buffer-load uses one base read pointer stored in the SRD - set it here:
    module.addInst("s_mov_b32", sgpr("Srd%s+0"%ch), sgpr("Address%s+0"%ch), "init SRD base address (lower)" )
    module.addInst("s_mov_b32", sgpr("Srd%s+1"%ch), sgpr("Address%s+1"%ch), "init SRD base address (upper) + other fields" )
    module.addInst("s_mov_b32", sgpr("Srd%s+2"%ch), hex(0x80000000), "")
    module.addInst("s_mov_b32", sgpr("Srd%s+3"%ch), "Srd127_96", "Set bits 127_96 in post-loop SRD")
    module.addSpaceLine()
    return module

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    #print "GlobalWriteIndices"
    if not self.do["PostLoop"]: return ""
    module = Code.Module("notLocalSplitUGlobalWriteIndices")

    module.addCode(self.computeStoreVgprs(kernel,
              divisor = kernel["SubGroup0"],\
              tid0Scale=kernel["VectorWidth"], \
              tid1Scale=kernel["VectorWidth"]))

    if kernel["BufferStore"]:
      #print "----AddressC-nonLSU-----"
      #print self.vgprPool.state()
      self.addrD = -1
      self.addrC = -1
    else:
      self.addrD = self.vgprPool.checkOut(2, 'addrD')
      module.addInst("v_mov_b32", \
          vgpr(self.addrD+0), \
          sgpr("AddressD+0"), \
          "sgpr -> vgpr")
      module.addInst("v_mov_b32", \
          vgpr(self.addrD+1), \
          sgpr("AddressD+1"), \
          "sgpr -> vgpr")
      self.addrC = self.vgprPool.checkOut(2, 'addrC')
      module.addInst("v_mov_b32", \
          vgpr(self.addrC+0), \
          sgpr("AddressC+0"), \
          "sgpr -> vgpr")
      module.addInst("v_mov_b32", \
          vgpr(self.addrC+1), \
          sgpr("AddressC+1"), \
          "sgpr -> vgpr")
    return module

  ##############################################################################
  # Release any resources used by the global write
  def cleanupGlobalWrite(self, kernel):
    self.vgprPool.checkIn(self.coord0)
    self.vgprPool.checkIn(self.coord1)

    if kernel["StoreRemapVectorWidth"]:
      self.vgprPool.checkIn(self.storeRemapLW)
      self.vgprPool.checkIn(self.storeRemapLR)
      self.vgprPool.checkIn(self.storeRemapCoord0)
      self.vgprPool.checkIn(self.storeRemapCoord1)
      self.vgprPool.checkIn(self.storeRemapOffsetCoord1)
    if kernel["BufferStore"]:
      self.vgprPool.checkIn(self.cinRowPtr)
      self.vgprPool.checkIn(self.coutRowPtr)
    if not kernel["BufferStore"]:
      self.vgprPool.checkIn(self.addrD)
      self.vgprPool.checkIn(self.addrC)

    if self.betaVgpr != None:
      self.vgprPool.checkIn(self.betaVgpr)

  ##############################################################################
  # Return max global write vector width, in elements
  def maxGwvw(self, kernel):
    atomic = (kernel["GlobalSplitU"] > 1) and (kernel["_GlobalAccumulation"] != 'MultipleBuffer')

    if kernel["BufferStore"]:
      if atomic:
        return kernel["VectorAtomicWidth"]
      else:
        return 1000  # no limit
    else:
      if atomic:
        return 1  # flat vector atomic is not tested
      else:
        return 1000  # no limit

  ##############################################################################
  # Partition thread-tile into writeElements for store code
  # This function creates the writeElement mapping for full tiles
  # (ie non-edge cases)
  ##############################################################################
  def notLocalFullTileElements(self, kernel, edge):
    component = Component.NotLocalFullTileElements.find(self)
    if component:
      return component(self, kernel, edge)

  ##############################################################################
  # Store Remap: Local Write
  ##############################################################################
  def storeRemapAddLocalWrite(self, kernel, ss, addrCalc, srcVgpr):
    """
    Add localWrite for the element with addrCalc and srcVgpr.
    """

    module = Code.Module("storeRemapAddLocalWrite srcVgpr %s"%str(srcVgpr))

    bps = self.bpeCexternal * ss.cfg.gwvw
    rpv = self.bpeCexternal * ss.cfg.gwvw / self.bpr

    addr0 = vgpr(self.storeRemapLW)
    offset =  addrCalc.coordOffset0 * self.bpeCexternal

    if bps==2:
      module.addInst("_ds_store_b16", addr0, vgpr(srcVgpr, rpv*2), \
                     "offset:%u"%offset, "storeRemap lw")
    elif bps==4:
      module.addInst("_ds_store_b32", addr0, vgpr(srcVgpr, rpv), \
                     "offset:%u"%offset, "storeRemap lw")
    elif bps==8:
      module.addInst("_ds_store_b64", addr0, vgpr(srcVgpr, rpv), \
                     "offset:%u"%offset, "storeRemap lw")
    elif bps==16:
      module.addInst("_ds_store_b128", addr0, vgpr(srcVgpr, rpv), \
                     "offset:%u"%offset, "storeRemap lw")
    else:
      assert 0, "StoreRemap: bad bps!"

    return module

  ##############################################################################
  # Store Remap: Local Read and Global Write
  ##############################################################################
  def storeRemapAddStore(self, kernel, ss, addrCalc, tmpVgpr, tmpS01, edge):
    module = Code.Module("storeRemapAddStore")

    module.addInst("s_waitcnt", "lgkmcnt(0)", "wait for LDS write" )

    numStoreInst = 0

    #Data exchange between different waves
    #Make sure LDS writes are finished of all waves
    if kernel["MIWaveGroup"][0] > 1:
      # FIXME: Indent?
      module.addInst("%s" %(self.indent + self.syncStr), "wait all lds write finished")
    module.addSpaceLine()

    gwvw = kernel["StoreRemapVectorWidth"]
    nElements = kernel["MacroTile0"]*kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]//self.kernel["WavefrontSize"]

    bpe = self.bpeCexternal
    bps = bpe * gwvw
    rpe = self.bpeCexternal / self.bpr
    rpv = rpe * gwvw

    # num registers to check out
    storeRegs = []
    for i in range(0, nElements, gwvw):
      storeRegs.append(self.vgprPool.checkOutAligned(int(rpv), int(rpv), "store element d"))
    src = vgpr(self.storeRemapLR)
    for rIdx, i in enumerate(range(0, nElements, gwvw)):
      offset = self.storeRemapLrOffset * bpe * (i//gwvw)
      dst = vgpr(storeRegs[rIdx], rpv)
      if bps==4:
        module.addInst("_ds_load_b32", dst, src, "offset:%u"%offset, "storeRemap lr")
      elif bps==8:
        module.addInst("_ds_load_b64", dst, src, "offset:%u"%offset, "storeRemap lr")
      elif bps==16:
        module.addInst("_ds_load_b128", dst, src, "offset:%u"%offset, "storeRemap lr")
      else:
        assert 0, "StoreRemap: bad bps!"

    module.addSpaceLine()

    # Global Write
    ntStr = ""
    if kernel.enabledSetPrioSplitLDS:
      module.addInst("s_setprio", "1", "")
    if kernel["NonTemporalD"]%2==1:
      ntStr += " glc"
    if kernel["NonTemporalD"]//2==1:
      ntStr += " slc"

    addr1 = sgpr("SrdD", 4)
    packedD1 = kernel["PackedC1IndicesX"]
    strideD1 = "StrideD%s" % (self.indexChars[packedD1[0]])

    vTmp = self.vgprPool.checkOut(1, "SR Store temp addr0")
    addr0 = vgpr(vTmp)

    if not edge:
      for rIdx, i in enumerate(range(0, nElements, gwvw)):
        if i == 0:
          module.addInst("v_mov_b32", addr0, vgpr(self.storeRemapOffsetCoord1), "coord1")
        else:
          currentStep = i//gwvw
          module.addInst("_v_add_u32", addr0, vgpr(self.storeRemapOffsetCoord1), self.storeRemapNCPL * currentStep , "coord1 += nColPerLoad")

        module.addInst("v_mul_lo_u32", addr0, addr0, sgpr(strideD1), "coord1 offset =  coord1 * StrideD")
        module.addInst("_v_add_lshl_u32", addr0, addr0,  vgpr(self.storeRemapCoord0), hex(log2(bpe)), "global write D address")

        lgkmcnt = min((nElements-i)//gwvw - 1, 15)
        module.addInst("s_waitcnt", "lgkmcnt(%u)"% lgkmcnt, "wait for LDS read" )

        numStoreInst += 1
        module.addInst(self.chooseGlobalWrite(True, bps, storeRegs[rIdx], rpv, addr0, addr1, 0, ntStr))
    else:
      tmpS23 = tmpS01+self.laneSGPRCount
      coord0 = tmpVgpr
      coord1 = coord0+1
      lrVw = kernel["StoreRemapVectorWidth"]
      edgeVw = min(kernel["AssertFree0ElementMultiple"],kernel["StoreRemapVectorWidth"])
      bps = self.bpeCexternal * edgeVw
      rpv = self.bpeCexternal / self.bpr * edgeVw
      for rIdx, i in enumerate(range(0, nElements, lrVw)):
        for vi in range (0, lrVw, edgeVw):

          if vi == 0:
            lgkmcnt = min((nElements-i)//lrVw - 1, 15)
            module.addInst("s_waitcnt", "lgkmcnt(%u)"% lgkmcnt, "wait for LDS read" )

          sizeBoundary = [0,0]
          sizeBoundary[0] = \
              sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
              else self.sizeRef(kernel["ProblemType"]["Index0"])
          sizeBoundary[1] = \
              sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
              else self.sizeRef(kernel["ProblemType"]["Index1"])

          currentStep = i//lrVw

          # calculate global coordination
          module.addInst("_v_add_u32", vgpr(coord1), vgpr(self.storeRemapCoord1), self.storeRemapNCPL * currentStep , "coord1 += nColPerLoad")
          module.addInst("_v_add_u32",vgpr(coord0), vgpr(self.storeRemapCoord0), vi , "coord0 += element index of load vector")
          module.addInst("_v_add_u32", addr0, vgpr(self.storeRemapOffsetCoord1), self.storeRemapNCPL * currentStep , \
                         "offset coord1 += nColPerLoad")

          module.addInst("v_cmp_lt_u32",  sgpr(tmpS01,self.laneSGPRCount), vgpr(coord0), sizeBoundary[0], "coord0 < size0" )
          module.addInst("v_cmp_lt_u32",  sgpr(tmpS23,self.laneSGPRCount), vgpr(coord1), sizeBoundary[1], "coord1 < size1" )
          module.addInst("s_and_b{}".format(self.kernel["WavefrontSize"]),
                         sgpr(tmpS23,self.laneSGPRCount),
                         sgpr(tmpS01,self.laneSGPRCount),
                         sgpr(tmpS23,self.laneSGPRCount), "in0 && in1" )

          module.addInst("v_mul_lo_u32", addr0, addr0, sgpr(strideD1), "coord1 element offset =  coord1 * StrideD")
          module.addInst("_v_add_lshl_u32", addr0, addr0,  vgpr(coord0), hex(log2(bpe)), "scale to BPE")
          module.addInst("v_cndmask_b32", addr0, -1, addr0, sgpr(tmpS23,self.laneSGPRCount), "clip if OOB. offset" )

          sumIdx = storeRegs[rIdx] + int(vi*rpe)
          numStoreInst += 1
          if bps == 2:
            module.addInst(self.chooseGlobalWrite(True, bpe, sumIdx, rpe, addr0, addr1, 0, ntStr, hi16=vi%2))
          else:
            module.addInst(self.chooseGlobalWrite(True, bps, sumIdx, rpv, addr0, addr1, 0, ntStr))

    module.addSpaceLine()
    self.vgprPool.checkIn(vTmp)
    for v in storeRegs:
      self.vgprPool.checkIn(v)

    #Data exchange between different waves
    #Make sure LDS reads are finished of all waves
    if kernel["MIWaveGroup"][0] > 1:
      module.addInst("%s" % (self.indent + self.syncStr), "wait all lds read finished")

    return module, numStoreInst

  ##############################################################################
  # Store remap compute vgprs:
  ##############################################################################
  def storeRemapComputeStoreVgprs(self, kernel):
    module = Code.Module("storeRemapComputeStoreVgprs")
    module.addComment0("Store Remap Local Write address")

    tmpS0 = self.getTmpSgpr(2).idx()
    wgMT1 = tmpS0+1

    wg0="WorkGroup0"
    wg1="WorkGroup1"

    tid0 = self.vgprPool.checkOut(1, "SR coord0")
    tid1 = self.vgprPool.checkOut(1, "SR coord1")
    coord1Offset = self.vgprPool.checkOut(1, "SR coord1 offset")
    storeRemapLW = self.vgprPool.checkOut(1, "SR local write")
    storeRemapLR = self.vgprPool.checkOut(1, "SR local read")

    tmpV0 = self.vgprPool.checkOut(5, "tmpV0")
    waveCoord0 = tmpV1 = tmpV0+1
    ldsStride = tmpV0+2
    coord0 = tmpV0+3
    waveCoord1 = tmpV0+4

    gwvw = kernel["StoreRemapVectorWidth"]
    ldsPad = max(kernel["StoreRemapVectorWidth"],kernel["MIOutputVectorWidth"])

    #calculate local write Address: v[vgprLocalWriteAddrC]
    module.addCode(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.kernel["WavefrontSize"]*kernel["MIWaveGroup"][0], \
      tmpV0, tmpS0))

    module.addInst("v_mul_lo_u32", vgpr(waveCoord1),
                   hex(kernel["MatrixInstN"]), vgpr(tid1), "coord1 offset of LDS for each Wave")
    module.addInst("v_and_b32", vgpr(tid1),
                   hex(kernel["MatrixInstN"]-1), vgpr("Serial"), "coord1 offset of LDS for each thread")
    module.addInst("_v_add_u32", vgpr(tid1), vgpr(waveCoord1),vgpr(tid1),"coord1 offset in MacroTile")
    module.addInst("v_mov_b32", vgpr(ldsStride), hex(kernel["MacroTile0"]+ldsPad), \
                   "lds stride = MT0 + PAD")
    module.addInst("v_mul_lo_u32", vgpr(tmpV0), vgpr(tid1), vgpr(ldsStride), \
                  "lds coord1 offset = Col-id* lds stride")

    module.addCode(vectorStaticDivideAndRemainder(waveCoord0, tid0, tid0, self.kernel["WavefrontSize"],tmpV0, tmpS0))
    module.addInst("v_lshrrev_b32", vgpr(coord0),
                   hex(log2(kernel["MatrixInstN"])), vgpr(tid0), \
                   "tid / matrixInstN")

    if kernel["MIOutputVectorWidth"] > 1:
      module.addInst("v_lshlrev_b32", vgpr(coord0), hex(log2(kernel["MIOutputVectorWidth"])), vgpr(coord0), \
                     "lds coord0 offset *= 4 (each thread hold 4 element)")

    module.addInst("v_mad_u32_u24", vgpr(coord0), kernel["MatrixInstM"]*kernel["MatrixInstBM"], vgpr(waveCoord0), vgpr(coord0), \
                   "coord0 += waveCoord0 * wave M shape(blockM*MiM)")

    module.addInst("_v_add_lshl_u32", \
      vgpr(storeRemapLW), \
      vgpr(tmpV0), \
      vgpr(coord0), \
      hex(log2(self.bpeCexternal)), \
      "local write C address")

    module.addSpaceLine()
    # calculate local read address : v[vgprLocalReadAddrC]

    module.addComment0("Store Remap Local Read address")

    module.addCode(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.kernel["WavefrontSize"], \
      tmpV0, tmpS0))
    module.addInst("v_mul_lo_u32", vgpr(waveCoord1),
                   hex(kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]), vgpr(tid1), "coord1 offset of LDS for each Wave")

    nThreadPerCol = kernel["MacroTile0"] // gwvw
    nColPerLoad = self.kernel["WavefrontSize"] // nThreadPerCol
    self.storeRemapLrOffset = (kernel["MacroTile0"]+ldsPad) * nColPerLoad
    self.storeRemapNCPL = nColPerLoad

    module.addInst("v_lshrrev_b32", vgpr(tmpV1),\
                   hex(log2(nThreadPerCol)), vgpr(tid0), \
                   "tid / nThreadPerCol")
    module.addInst("_v_add_u32", vgpr(coord1Offset), vgpr(waveCoord1),vgpr(tmpV1),"coord1 offset in MacroTile")
    module.addInst("v_mul_lo_u32", vgpr(tmpV0), vgpr(coord1Offset), vgpr(ldsStride), \
                   "lds coord1 offset = Col-id* lds stride")

    module.addInst("v_and_b32", vgpr(coord0),
                   hex(nThreadPerCol-1), vgpr(tid0), "coord0 offset of LDS for each thread")
    module.addInst("v_lshlrev_b32", vgpr(coord0), hex(log2(gwvw)), vgpr(coord0), \
                   "lds coord0 offset *= gwvw (each thread hold gwvw element)")

    module.addInst("_v_add_lshl_u32", \
      vgpr(storeRemapLR), \
      vgpr(tmpV0), \
      vgpr(coord0), \
      hex(log2(self.bpeCexternal)), \
      "local read C address")
    module.addSpaceLine()

    # calculate global write coord0 and coord1
    module.addComment0("Store Remap global write coord0 and coord1")
    module.addCode(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.kernel["WavefrontSize"]*kernel["MIWaveGroup"][0], \
      tmpV0, tmpS0))

    ColsPerBlockShape = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

    module.addInst("v_mul_lo_u32", vgpr(waveCoord1),
                  hex(ColsPerBlockShape), vgpr(tid1), "coord1 offset of global memory for each Wave")

    module.addCode(vectorStaticDivideAndRemainder(tid1, tid0, tid0, self.kernel["WavefrontSize"], \
      tmpV0, tmpS0))
    module.addInst("v_mad_u32_u24", vgpr(waveCoord1), kernel["MatrixInstN"]//kernel["MIWaveGroup"][0], vgpr(tid1), vgpr(waveCoord1), \
                   "waveCoord1 += waveCoord0 * MiN / WaveGroupM")

    module.addInst("v_lshrrev_b32", vgpr(tmpV1),\
                   hex(log2(nThreadPerCol)), vgpr(tid0), \
                   "tid / nThreadPerCol")

    module.addInst("_v_add_u32", vgpr(coord1Offset), vgpr(waveCoord1),vgpr(tmpV1),"coord1 offset in MacroTile")

    module.addInst("s_mul_i32", \
        sgpr(tmpS0), \
        hex(kernel["MacroTile0"]), \
        sgpr(wg0), \
        "%s = wg0*MT0"%sgpr(tmpS0))

    module.addInst("_v_add_co_u32", vgpr(tid0), self.vcc, sgpr(tmpS0), vgpr(coord0), "coord0 = coord0 + wg0 * MT0")

    module.addInst("s_mul_i32", \
        sgpr(wgMT1), \
        "MT1", \
        sgpr(wg1), \
        "<- wg1*MT1")
    module.addInst("_v_add_co_u32", \
        vgpr(tid1), \
        self.vcc, \
        sgpr(wgMT1), \
        vgpr(coord1Offset), \
        "coord1 = tid1*VW + wg1*MT1")

    module.addSpaceLine()

    module.addCode(self.syncThreads(kernel, "StoreRemap Start"))

    self.storeRemapLW = storeRemapLW  #local write
    self.storeRemapLR = storeRemapLR  #local read
    self.storeRemapCoord0 = tid0      #global coord0
    self.storeRemapCoord1 = tid1      #global coord1
    self.storeRemapOffsetCoord1 = coord1Offset #offset coord1

    self.vgprPool.checkIn(tmpV0)

    return module

  ##############################################################################
  # Not LocalSplitU: Global Write
  # Determine write batching pattern
  # element() specifies TT 'coordinate' to write
  # vectorWidths specifies width of vector to store
  # TODO - why does this use VectorWidth to control store width?  Could be GlobalWriteVectorWidth?
  #
  # Function creates one mapping for full tiles and one for edge tiles,
  # then calls globalWriteElements to generate the code for the new tiles.
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""
    elements = [[] for y in range(2)] # 2D array for Full, Edge

    (fullVw, elements[False]) = self.notLocalFullTileElements(kernel, False)
    (edgeVw, elements[True])  = self.notLocalFullTileElements(kernel, True)

    # print("len(elements[False])= ", len(elements[False]))
    # print("len(elements[True])= ", len(elements[True]))
    vectorWidths = [fullVw, edgeVw]

    module = Code.Module("notLocalSplitUGlobalWrite")
    module.addCode(self.globalWriteElements(kernel, vectorWidths, elements))

    self.cleanupGlobalWrite(kernel)

    return module

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""

    fullVw = kernel["GlobalWriteVectorWidth"] if kernel["_VectorStore"] else 1
    fullVw = min(fullVw, self.maxGwvw(kernel))
    elements = [[] for y in range(2)] # 2D array for Full, Edge
    # Full tile loop:
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for vc1 in range(0, 1):
        for tt0 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], fullVw): # note step by fullVw
            element = (tt1, tt0, vc1, vc0)
            elements[False].append(element)

    # Edge tile loop - note if we know AF0EM we can can use a larger vector
    # and reduce the boundary checks accordingly.  But if no AF0EM guarantee
    # then use a conservative 1
    edgeVw = kernel["GlobalWriteVectorWidth"] if kernel["_VectorStore"] else 1
    edgeVw = min(edgeVw, self.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
    assert(kernel["GlobalWriteVectorWidth"]%edgeVw == 0)
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for vc1 in range(0, 1):
        for tt0 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], edgeVw):
            element = (tt1, tt0, vc1, vc0)
            elements[True].append(element)

    vectorWidths = [fullVw, edgeVw]
    module = Code.Module("localSplitUGlobalWrite")
    module.addCode(self.globalWriteElements(kernel, vectorWidths, elements))
    self.cleanupGlobalWrite(kernel)
    return module

  ##############################################################################
  # checkIsBetaZero
  # tmpSgpr is one temp sgpr
  # betaLabel is label to branch to if beta != 0
  ##############################################################################
  def checkIsBetaZero(self, kernel, tmpSgpr, betaLabel):
    module = Code.Module("checkIsBetaZero label %s"%betaLabel)
    assert(isinstance(betaLabel, Code.Label))
    betaLabelName = betaLabel.getLabelName()
    if kernel["ProblemType"]["UseBeta"]:
      if self.bpeCinternal <= self.bpr: # 1 register to check for Beta==0
        module.addInst("s_cmpk_eq_u32", sgpr("Beta"), hex(0), "Beta == 0")
      else: # multiple registers to check for Beta==0
        module.addInst("s_mov_b32", sgpr(tmpSgpr), sgpr("Beta+0"), "tmp = Beta[0]")
        for i in range(1, self.bpeCinternal//self.bpr):
          module.addInst("s_or_b32", sgpr(tmpSgpr), sgpr("Beta+%u"%i), sgpr(tmpSgpr), "tmp |= Beta[%u] " % i)
        module.addInst("s_cmpk_eq_u32", sgpr(tmpSgpr), hex(0), "Beta == 0")
      module.addInst("s_cbranch_scc0", betaLabelName, "Branch if Beta is not zero")
    module.addSpaceLine()
    return module

  ##############################################################################
  # checkIsEdge
  # tmpSgpr must have at least 6 free SGPR
  # isEdgeTarget is the branch target if edges are required
  ##############################################################################
  def checkIsEdge(self, kernel, tmpSgpr, isEdgeTarget):
    assert(isinstance(isEdgeTarget, Code.Label))
    isEdgeTargetLabel = isEdgeTarget.getLabelName()
    module = Code.Module("checkIsEdge")
    tmpS0  = tmpSgpr
    tmpS1  = tmpS0 + 1
    tmpS23 = tmpS1 + 1

    wg0="WorkGroup0"
    wg1="WorkGroup1"

    # check edge0 ###
    # s23 = rMT0 = Size0 % MT0
    #--
    sizeBoundary = [0,0]
    sizeBoundary[0] = \
        sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
        else self.sizeRef(kernel["ProblemType"]["Index0"])
    sizeBoundary[1] = \
        sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
        else self.sizeRef(kernel["ProblemType"]["Index1"])

    module.addCode(scalarStaticDivideAndRemainder(tmpS1, tmpS0, sizeBoundary[0], kernel["MacroTile0"], tmpS23, 2))
    # s23 = nwg0-1
    module.addInst("s_add_u32", sgpr(tmpS1), hex(-1), sgpr("NumWorkGroups0"), "" )
    module.addInst("s_cmp_ge_u32", sgpr(wg0), sgpr(tmpS1), "wg0 >= nwg0-1 ?")
    module.addInst("s_cselect_b32", sgpr(tmpS0), sgpr(tmpS0), 0, "set rMT0")
    # s01 now = myMT0 = wg0 < nwg0-1 ? MT0 : rMT0

    # if rMT0 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      module.addInst("s_cmpk_gt_u32", sgpr(tmpS0), hex(0), "rMT0 > 0")
      if self.db["ForceEdgeStores"]:
        module.addInst("s_cmp_eq_u32", sgpr(tmpS0), sgpr(tmpS0), "ForceEdgeStores!")
      module.addInst("s_cbranch_scc1", isEdgeTargetLabel, "jump if edges required")

    # check edge1 ###
    # TODO-packed - this only needs to change to handle packing into C1 index
    # change would be similar to above - multiply by product of packed sizes in C1
    # --

    # s23 = rMT1 = Size1 % MT1
    module.addCode(scalarStaticDivideAndRemainder(tmpS1, tmpS0, sizeBoundary[1], kernel["MacroTile1"], tmpS23, 2))
    # s01 now = myMT1 = wg1 < nwg1-1 ? MT1 : rMT1

    # s23 = nwg1-1
    module.addInst("s_add_u32", sgpr(tmpS1), hex(-1), sgpr("NumWorkGroups1"), "" )
    module.addInst("s_cmp_ge_u32", sgpr(wg1), sgpr(tmpS1), "wg1 >= nwg1-1")
    module.addInst("s_cselect_b32", sgpr(tmpS0), sgpr(tmpS0), 0, "set rMT1")

    # if rMT1 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      module.addInst("s_cmpk_gt_u32", sgpr(tmpS0), hex(0), "rMT1 > 0")
      module.addInst("s_cbranch_scc1", isEdgeTargetLabel, "jump if edges required")

    return module

  ##############################################################################
  # Global Write Elements
  ##############################################################################

  class BF16CVTVgprStruct(NamedTuple): # class for bf16 vgprs
    vgprBf16Temp: int = -1
    vgprBf16Mask: int = -1
    vgprFp32Nan: int = -1
    vgprBf16Inc: int = -1

  def globalWriteElements(self, kernel, vectorWidths, elements,
                          applyAlpha=True, # defaults to generating *=alpha codes
                          betas=None, # if left unspecified, then let global parameter decide
                          edges=None):
    if not self.do["PostLoop"]: return Code.Module("GlobalWriteElements (Empty)")
    module = Code.Module("GlobalWriteElements")
    atomic = (kernel["GlobalSplitU"] > 1) and (kernel["_GlobalAccumulation"] != 'MultipleBuffer')
    activation = ActivationModule(self.vcc)

    # write possibilities and labels
    # if beta/edge combo not specified fall back to global param definition
    if betas is None:
      hasBeta = kernel["ProblemType"]["UseBeta"] and (kernel["_GlobalAccumulation"] != 'MultipleBuffer')
      betas = [False, True] if hasBeta else [False]
    if edges is None:
      edges = [False, True] if self.do["EdgeWrite"] else [False]
    writeLabels = {}
    for beta in betas:
      writeLabels[beta] = {}
      for edge in edges:
        writeLabels[beta]["EdgeCheck0"] = Code.Label(self.labels.getNameInc("GW_B%u_E%u_EdgeCheck0" % ( 1 if beta else 0, 1 if edge else 0) ), "")
        writeLabels[beta]["EdgeCheck1"] = Code.Label(self.labels.getNameInc("GW_B%u_E%u_EdgeCheck1" % ( 1 if beta else 0, 1 if edge else 0) ), "")
        writeLabels[beta][edge] = Code.Label(self.labels.getNameInc("GW_B%u_E%u" % ( 1 if beta else 0, 1 if edge else 0) ), "")
      if not beta:
        betaLabel = Code.Label(self.labels.getNameInc("GW_Beta"), "")
    endLabel = Code.Label(self.labels.getNameInc("GW_End"), "")

    # Layout
    """
    if B1 goto label_B1
    if E1 goto label_B0_E1
    label_B0_E0:
    writes
    goto label_End
    label_B0_E1:
    writes
    goto label_End
    label_B1:
    if E1 goto label_B1_E1
    label_B1_E0:
    writes
    goto label_End
    label_B1_E1:
    writes
    goto label_End
    label_End
    """
    self.betaVgpr = None

    ########################################
    # Vgprs
    if kernel["BufferStore"]:
      numTmpVgpr = 2
      if len(kernel["PackedC0IndicesX"]) > 1:
        numTmpVgpr += 1
    else:
      numTmpVgpr = 2 + 3 # GLOBAL_OFFSET_C needs 3, plus 2 tmps?
    tmpVgpr = self.vgprPool.checkOutAligned(numTmpVgpr, 2, "store tmps")

    isHpaBF16 = kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]
    bf16CVTVgpr = self.vgprPool.checkOut(4) if isHpaBF16 else None
    bf16CVTVgprStruct = self.BF16CVTVgprStruct(vgprBf16Temp=bf16CVTVgpr, vgprBf16Mask=(bf16CVTVgpr+1),
                                               vgprFp32Nan=(bf16CVTVgpr+2), vgprBf16Inc=(bf16CVTVgpr+3)) if bf16CVTVgpr else None

    ########################################
    # Sgprs

    # allocate tmps for the store header (before the batch implementations)
    tmpSgpr = self.getTmpSgpr(4).idx()

    # branch B1 or B0
    betaLabel = Code.Label(self.labels.getNameInc("GW_Beta"), "")

    if False in betas and True in betas:
      module.addCode(self.checkIsBetaZero(kernel, tmpSgpr, betaLabel))

    for beta in betas:
      # start B1
      if beta:
        module.addCode(betaLabel)

      ########################################
      # branch if Edge0 or Edge1
      if False in edges and True in edges:
        module.addCode(self.checkIsEdge(kernel, tmpSgpr, writeLabels[beta][True]))

      # by now we either jumped to E1 or stayed at E0
      for edge in edges:
        module.addCode(writeLabels[beta][edge])

        # for storeRemap edge case, non-beta still can enable vector stores
        if kernel["StoreRemapVectorWidth"] and not beta:
          edgeI = False
        else:
          edgeI = edge
        #edgeI = True  # set to True to disable vector stores
        gwvw = vectorWidths[edgeI]
        #print "globalWriteElements: edge=", edge, "beta=", beta, "atomic=", atomic

        ########################################
        # Calculate Vgprs for Write Batching
        ########################################

        ss = StoreState(self, kernel, gwvw, edge, beta, atomic, elements[edgeI])

        # how many vgprs are needed for zero elements
        # 2 for addressC in vgpr for addition - already checked out
        # 2 for coord0,1 of thread - already checked out
        # 2 for tmp - already checked out

        # 5 = how many vgprs are needed per element (flat)
        #  - 2 for addr
        #  - 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
        #  - if beta gwvw*rpe for new value
        #  - if atomic 2*rpe for old and cmp values

        # print("numVgprsPerAddr=%u, numVgprsPerDataPerVI=%u, numVgprPerValuC=%u"%(ss.cfg.numVgprsPerAddr, ss.cfg.numVgprsPerDataPerVI, ss.cfg.numVgprPerValuC))
        numVgprsPerElement = ss.cfg.numVgprPerValuC*gwvw + ss.cfg.numVgprsPerAddr + int(ceil(ss.cfg.numVgprsPerDataPerVI * gwvw))

        if kernel["GroupLoadStore"] and kernel["ProblemType"]["UseBeta"]:
          numVgprsPerElement += ss.cfg.numVgprsPerAddr

        #print self.vgprPool.state()
        # Use VGPR up to next occupancy threshold:
        maxVgprs = self.getMaxRegsForOccupancy(kernel["NumThreads"], self.vgprPool.size(), \
                                               self.getLdsSize(kernel), self.agprPool.size(), self.unifiedVgprRegs)
        if self.serializedStore: # get aggressive when serializedStore is on; not necessarily exclusive to this parameter
          len(elements[edgeI])
          tl = []
          for i in range(self.vgprPool.size()-self.vgprPool.available(), maxVgprs):
            tl.append(self.vgprPool.checkOut(1, "grow-pool up to next occupancy for GlobalWrite"))
          for t in tl:
            self.vgprPool.checkIn(t)
        align = 1
        # align adjustment
        if ss.cfg.numVgprsPerAddr > 1:
          align = max(align, ss.cfg.numVgprsPerAddr)
        if ss.cfg.numVgprPerValuC*gwvw > 1:
          align = max(align, ss.cfg.numVgprPerValuC*gwvw)
        if int(ceil(ss.cfg.numVgprsPerDataPerVI * gwvw)) > 1:
          align = max(align, int(ceil(ss.cfg.numVgprsPerDataPerVI * gwvw)))
        numVgprAvailable = self.vgprPool.availableBlock(numVgprsPerElement, align)

        # Grow the register pool if needed - we need enough regs for at least one element
        # Unfortunate since this means the write logic is setting the VGPR requirement
        # for the entire kernel but at least we have a functional kernel.
        # Before growing the pool, see if we can shrink the write vector width instead?
        # TODO : the vgprSerial is needed for-ever and if we grow here will split the
        # range of the tmps.  Maybe want to move vgprSerial to first vgpr?

        # TODO: Minimum elems for StoreRemap
        # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
        minElements = 2 if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) else 1
        minNeeded = minElements * numVgprsPerElement
        shrinkDb = 0
        if shrinkDb:
          print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)
        if numVgprAvailable < minNeeded:
          gwvwOrig = gwvw
          currentOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
              self.vgprPool.size(), self.agprPool.size(), self.unifiedVgprRegs)
          futureOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
              self.vgprPool.size() - numVgprAvailable + minNeeded, self.agprPool.size(), self.unifiedVgprRegs)

          if shrinkDb:
            print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
                % (currentOccupancy, futureOccupancy, self.vgprPool.size(), \
                   numVgprAvailable, minElements*numVgprsPerElement))
          if futureOccupancy > currentOccupancy:
            if shrinkDb:
              print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                    (self.kernelName))
              print("   numVgprAvailable=", numVgprAvailable, \
                    "numVgprsPerElement=", numVgprsPerElement, "atomic=", atomic, \
                    "beta=", beta, "gwvw=", gwvw)
          elif gwvw != gwvwOrig:
            ss.gwvw = gwvw # make both representations consistent
            if shrinkDb:
              print2("info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                  % (self.kernelName, gwvwOrig, gwvw, currentOccupancy))

          if numVgprAvailable < minElements*numVgprsPerElement:
            print2("info: growing pool += %d * %d for GlobalWrite\n" \
                % (minElements,numVgprsPerElement))
            print2(self.vgprPool.state())
            tl = []
            for i in range(0,minElements):
              tl.append(self.vgprPool.checkOut(numVgprsPerElement, "grow-pool for GlobalWrite"))
            for t in tl:
              self.vgprPool.checkIn(t)
            numVgprAvailable = self.vgprPool.available()
            print2(self.vgprPool.state())

        # set atomicW after we potentially resize GWVW
        atomicW = min(gwvw, kernel["VectorAtomicWidth"])

        # print("NumVgprAvailable", numVgprAvailable)
        if numVgprsPerElement:
          numElementsPerBatch = numVgprAvailable // numVgprsPerElement
        else:
          numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

        assert(self.numVgprValuC % gwvw == 0) # sanity check

        numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

        if shrinkDb:
          print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", ss.cfg.numElementsPerBatchLimitedBySgprs, \
              "WARNING" if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
        if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
          numElementsPerBatch = ss.cfg.numElementsPerBatchLimitedBySgprs

        # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
        if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
          # only do an even number of halves - since these share hi/lo pieces of some registers?
          if numElementsPerBatch > 1:
            numElementsPerBatch = int(numElementsPerBatch/2)*2
          elif not kernel["EnableMatrixInstruction"]:
            # The globalWriteBatch routine below can't handle odd elements per batch
            # and 0 elements per batch is illegal.
            # so if we don't have *GPR resources to handle a larger batch then need
            # to mark overflowedResources rather than generate a kernel that won't work.
            # It might be possible to fix globalWriteBatch to handle this case but these
            # are likely to be low-performing so likely not worth optimizing.
            if shrinkDb:
              print("WARNING: half requires at least two elements per batch")
            self.overflowedResources = 3

        assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%self.kernelName

        #numElementsPerBatch=min(2,numElementsPerBatch) # hack to control number of batches
        if atomic and (ss.optSingleColVgpr or ss.optSharedColVgpr):
          # hack to avoid re-using address vgpr across rows
          # atomics need to perform several memory operations
          # if the batch spans multiple rows, need multiple address vgpr
          # which is not currently supported in the two opt*ColVgpr modes
          firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0]
          numElementsPerBatch=min(len(firstRow),numElementsPerBatch)

        # check best numElementsPerBatch to handle a column block
        # elements of column block must be multiple size of numElementsPerBatch
        if kernel["StoreRemapVectorWidth"]:
          firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0] # format for element = (tt1, tt0, vc1, vc0)
          # find the largest factor and smaller than numElementPerBatch
          nBatchesPerRow = 1
          for d in range(1, len(firstRow)+1):
            largestFactor = len(firstRow)//d
            if len(firstRow)%d == 0 and largestFactor <= numElementsPerBatch:
              numElementsPerBatch = largestFactor
              nBatchesPerRow = d
              break

        # if no atomics and no edge, then write whole vectors
        #if not atomic and not edge:
        #  numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
        #  #print "  NumVectorsPerBatch", numVectorsPerBatch
        #  numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
        numBatches = max(1, ceil_divide(len(elements[edgeI]),numElementsPerBatch))

        numSgprs = ss.cfg.numTempSgprPerBatch + ss.cfg.numMaskSgprPerBatch + ss.cfg.numMaskSgprPerElement * numElementsPerBatch

        if self.db["PrintStoreRegisterDb"]:
          print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
          print ("numSgprs=", numSgprs, "sgprPool.size()=", self.sgprPool.size(), "numTempSgprPerBatch=", ss.cfg.numTempSgprPerBatch,
                 "numMaskSgprPerBatch=", ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", ss.cfg.numMaskSgprPerElement)
          print(self.sgprPool.state())
        module.addComment1("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
            (edgeI, numSgprs, ss.cfg.numTempSgprPerBatch, ss.cfg.numMaskSgprPerBatch, ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
        #module.addComment("storeStats, %d, %d, %d"% (edgeI, numSgprs, numElementsPerBatch))
        # so if we don't have *GPR resources to handle a larger batch then need
        # to mark overflowedResources rather than generate a kernel that won't work.
        # Temporary solution for getTmpSgpr. getTmpSgpr().idx() does not return the class thus it may be garbage collected anytime.
        getTmpSgprClass = self.getTmpSgpr(numSgprs, 2)
        tmpSgpr = getTmpSgprClass.idx()

        elementSgprs = tmpSgpr + ss.cfg.numTempSgprPerBatch

        codeAccVgprRead = deepcopy(self.codeAccVgprRead) if self.serializedStore else None
        codeMulAlpha    = deepcopy(self.codeMulAlpha) if self.serializedStore else None

        self.alphaBeforeLoadC = False
        if kernel["MIArchVgpr"] and applyAlpha:
          codeAccVgprRead = None

          #Only apply when 2 wave optimization features are enabled
          if (kernel["StorePriorityOpt"] or kernel["StoreSyncOpt"]) and beta:
            self.alphaBeforeLoadC = True
        else:
          codeMulAlpha = None

        for batchIdx in range(0, numBatches):
          elementStartIdx = batchIdx * numElementsPerBatch
          elementStopIdx = min( elementStartIdx + numElementsPerBatch, len(elements[edgeI]) )
          elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
          #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,numVgprsPerElement ))
          # elementVgprs can be large and should be perfectly tuned to the number of available
          # VGPRS.  We do not want to accidentally overflow and grow the pool here:

          if kernel["StoreRemapVectorWidth"]:
            #Indication if this batch is last batch for this column block shape
            self.StoreRemapLastBatch = 1 if (batchIdx+1) % nBatchesPerRow == 0 else 0

          module.addCode(self.globalWriteBatch(kernel, activation, ss, batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
              elementsThisBatch, self.addrD, self.addrC, \
              tmpVgpr, bf16CVTVgprStruct, \
              elementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha))
        # Delete tmp class
        del getTmpSgprClass

        # TODO - if this is the last tile, don't need to jump to next instruction
        module.addInst("s_branch", endLabel.getLabelName(), "jump to end")
        del ss

    # End label
    module.addCode(endLabel)
    self.vgprPool.checkIn(tmpVgpr)
    if bf16CVTVgpr is not None:
      self.vgprPool.checkIn(bf16CVTVgpr)
    return module

  ##############################################################################
  # chooseGlobalRead :
  # create the load instruction for requested vector width and other parms
  # return an Inst class
  #
  # bpl = bytes per load op
  ##############################################################################
  def chooseGlobalRead(self, useBuffer, bpl, destVgpr, \
                       addr0, addr1, soffset, offset, extraFields, hi16=0, comment="load C"):
  # rpv = regs per vector
    rpv = bpl/4.0

    if useBuffer:
      rv = Code.Module("Global Read")
      tailFields = "offen offset:%u"%offset
      # buffer_load offset field is 12-bit.
      # if offset >= 4096, use soffset instead
      if offset >= 4096:
        if soffset == 0 or soffset == "0":
          tailFields = "offen offset:0"
          soffset = sgpr(self.getTmpSgpr(1).idx())
          rv.addInst("s_mov_b32", soffset, offset, "large offset")
        else:
          assert 0, "offset too large and soffset set"
      if extraFields != "":
        tailFields += ", %s"% extraFields
      if bpl==1 and hi16:
        rv.addCode(Code.GlobalReadInst("_buffer_load_d16_hi_u8", vgpr(destVgpr, rpv*4), addr0, \
                  addr1, soffset, tailFields, comment))
        return rv
      elif bpl==1 and not hi16:
        rv.addCode(Code.GlobalReadInst("_buffer_load_d16_u8", vgpr(destVgpr, rpv*4), addr0, \
                  addr1, soffset, tailFields, comment))
        return rv
      elif bpl==2 and hi16:
        rv.addCode(Code.GlobalReadInst("_buffer_load_d16_hi_b16", vgpr(destVgpr, rpv*2), addr0, \
                  addr1, soffset, tailFields, comment))
        return rv
      elif bpl==2 and not hi16:
        rv.addCode(Code.GlobalReadInst("_buffer_load_d16_b16", vgpr(destVgpr, rpv*2), addr0, \
                  addr1, soffset, tailFields, comment))
        return rv
      elif bpl==4:
        rv.addCode(Code.GlobalReadInst("_buffer_load_b32", vgpr(destVgpr, rpv), addr0, \
                  addr1, soffset, tailFields, comment))
        return rv
      elif bpl==8:
        rv.addCode(Code.GlobalReadInst("_buffer_load_b64", vgpr(destVgpr, rpv), addr0, \
                  addr1, soffset, tailFields, comment))
        return rv
      elif bpl==16:
        rv.addCode(Code.GlobalReadInst("_buffer_load_b128", vgpr(destVgpr, rpv), addr0, \
                  addr1, soffset, tailFields, comment))
        return rv
      elif bpl==32:
        # split into two dwordx4 loads. Second load offset is +0.5 bpl
        tailFields1 = "offen offset:%u"%(offset + bpl/2)
        if extraFields != "":
          tailFields1 += ", %s"% extraFields

        rv = Code.Module("emulated _buffer_load_b256")
        rv.addCode(Code.GlobalReadInst("_buffer_load_b128", vgpr(destVgpr, rpv/2), addr0, \
                  addr1, soffset, tailFields, comment))
        rv.addCode(Code.GlobalReadInst("_buffer_load_b128", vgpr(int(destVgpr + rpv/2), rpv/2), addr0, \
                  addr1, soffset, tailFields1, comment))
      else:
        assert 0, "chooseGlobalRead: bad bpl"

      return rv

    else:
      if bpl==2 and hi16:
        return Code.GlobalReadInst("_flat_load_d16_hi_b16", vgpr(destVgpr, rpv*2), addr0, extraFields, comment )
      elif bpl==2 and not hi16:
        return Code.GlobalReadInst("_flat_load_d16_b16", vgpr(destVgpr, rpv*2), addr0, extraFields, comment )
      elif bpl==4:
        return Code.GlobalReadInst("_flat_load_b32", vgpr(destVgpr, rpv), addr0, extraFields, comment )
      elif bpl==8:
        return Code.GlobalReadInst("_flat_load_b64", vgpr(destVgpr, rpv), addr0, extraFields, comment )
      elif bpl==16:
        return Code.GlobalReadInst("_flat_load_b128", vgpr(destVgpr, rpv), addr0, extraFields, comment )
      else:
        assert 0, "chooseGlobalRead: bad bpl"

  ##############################################################################
  def chooseGlobalWrite(self, useBuffer, bps, srcVgpr, rpv, \
                        addr0, addr1, offset, extraFields, hi16=0):
    """
    create the store instruction for requested vector width and other parms
    rpv = regs per vector
    """

    module = Code.Module("chooseGlobalWrite %s -> %s (%s)"%(srcVgpr, addr0, addr1))

    if useBuffer:
      tmpSgpr = 0
      # buffer_load offset field is 12-bit.
      # if offset >= 4096, use soffset instead
      if offset >= 4096:
        tmpSgpr = sgpr(self.getTmpSgpr(1).idx())
        module.addInst("s_mov_b32", tmpSgpr, offset, "large offset")
        offset = 0

      if bps==2 and hi16:
        module.addInst("_buffer_store_d16_hi_b16", vgpr(srcVgpr, rpv*2), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==2 and not hi16:
        module.addInst("_buffer_store_b16", vgpr(srcVgpr, rpv*2), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==4:
        module.addInst("_buffer_store_b32", vgpr(srcVgpr, rpv), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==8:
        module.addInst("_buffer_store_b64", vgpr(srcVgpr, rpv), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==16:
        module.addInst("_buffer_store_b128", vgpr(srcVgpr, rpv), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps == 32:
        # split into two dwordx4 loads. Offset the second by +0.5 bps
        module.addInst("_buffer_store_b128", vgpr(srcVgpr, rpv/2), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")

        module.addInst("_buffer_store_b128", vgpr(int(srcVgpr +rpv/2), rpv/2), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%(int(offset+bps/2)), extraFields, "store D")
      else:
        assert 0, "bad bps"
    else:
      if bps==2 and hi16:
        module.addInst("_flat_store_d16_hi_b16", addr0, vgpr(srcVgpr*2), extraFields, "store D" )
      elif bps==2 and not hi16:
        module.addInst("_flat_store_d16_b16", addr0, vgpr(srcVgpr, rpv*2), extraFields, "store D" )
      elif bps==4:
        module.addInst("_flat_store_b32", addr0, vgpr(srcVgpr, rpv), extraFields, "store D" )
      elif bps==8:
        module.addInst("_flat_store_b64", addr0, vgpr(srcVgpr, rpv), extraFields, "store D" )
      elif bps==16:
        module.addInst("_flat_store_b128", addr0, vgpr(srcVgpr, rpv), extraFields, "store D" )
      else:
         assert 0, "bad bps"

    return module

  ##############################################################################
  def addStore(self, kernel, ss, addrCalc, sumIdx, tmpS01, edge):
    """
    Add stores for the element with addrCalc and sumIdx.
    tmpS01 is a single :temp sGPR
    """
    module = Code.Module("addStore sumIdx %s"%(str(sumIdx)))
    if self.do["GlobalWrite"]:
      # perform vector stores here, so no VI indexing.
      # if GWVW > Vw, might need to support loops to
      # implement wider stores
      ntStr = ""
      if kernel["NonTemporalD"]%2==1:
        ntStr += " glc"
      if kernel["NonTemporalD"]//2==1:
        ntStr += " slc"

      bps = self.bpeCexternal * ss.cfg.gwvw
      rpv = self.bpeCexternal * ss.cfg.gwvw / self.bpr

      if kernel["BufferStore"]:
        addr0 = vgpr(addrCalc.addrDVgpr)
        addr1 = sgpr("SrdD", 4)
      else:
        addr0 = vgpr(addrCalc.addrDVgpr,2)
        addr1 = ""

      useBuffer = kernel["BufferStore"]
      if ss.optSrdIncForRow and addrCalc.rowInc:
        module.addCode(addrCalc.incrementToNextRow(kernel, "D", ss, tmpS01))
      if kernel["ProblemType"]["DestDataType"].isHalf() or kernel["ProblemType"]["DestDataType"].isBFloat16():

        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # (H,H,H,H,H,H), internal H
          module.addCode(self.chooseGlobalWrite(useBuffer, bps, sumIdx//2, rpv, \
                           addr0, addr1, addrCalc.globalOffset, ntStr, hi16=sumIdx%2))
        else:
          # (B,B,B,B,S,S), internal S
          # (H,H,H,H,H,H), internal S
          # (H,H,H,H,S,S), internal S
          module.addCode(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                         addr0, addr1, addrCalc.globalOffset, ntStr, hi16=0))
      elif kernel["ProblemType"]["DestDataType"].isInt32() or kernel["ProblemType"]["DestDataType"].isSingle():
        module.addCode(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                       addr0, addr1, addrCalc.globalOffset, ntStr))
      elif kernel["ProblemType"]["DestDataType"].isDouble() or kernel["ProblemType"]["DestDataType"].isSingleComplex():
        if kernel["AtomicAddC"] and not edge:
          module.addInst("buffer_atomic_add_f64", vgpr(sumIdx*2, 2), vgpr(addrCalc.addrDVgpr), sgpr("SrdD", 4), "0", "offen offset:{}".format(addrCalc.globalOffset), "AtomicAddC")
        else:
          module.addCode(self.chooseGlobalWrite(useBuffer, bps, sumIdx*2, rpv, \
                         addr0, addr1, addrCalc.globalOffset, ntStr))
      elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
        rps = kernel["ProblemType"]["DestDataType"].numRegisters()
        module.addCode(self.chooseGlobalWrite(useBuffer, bps, sumIdx*rps, rpv, \
                       addr0, addr1, addrCalc.globalOffset, ntStr))
      elif kernel["ProblemType"]["DestDataType"].isInt8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          module.addCode(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                         addr0, addr1, addrCalc.globalOffset, ntStr))
    return module

  ##############################################################################
  # choose the ADD instruction for combining external C with internal C
  # used in atomic=1 case to compute expected external data
  ##############################################################################
  def chooseAddForAtomic(self, kernel, dst, src0, src1, comment):
    module = Code.Module("chooseAddForAtomic")
    if kernel["ProblemType"]["DataType"].isBFloat16():
      if kernel["_GlobalAccumulation"]:
        module.addInst("v_add_f32", dst, src0, src1, comment)
    elif kernel["ProblemType"]["DataType"].isHalf():
      if kernel["_GlobalAccumulation"]:
        module.addInst("v_add_f32", dst, src0, src1, comment)
      elif kernel["ProblemType"]["HighPrecisionAccumulate"]:
        module.addInst("v_mad_mix need madmix bozo", \
                       dst, src0, src1, \
                       comment)
      else:
        module.addInst("v_pk_add_f16", \
                       dst, src0, src1, \
                       comment)
    elif kernel["ProblemType"]["DataType"].isInt8x4() or kernel["ProblemType"]["DataType"].isInt8():
      # assume v_add_i32 can be used in place of v_add_f32
      # need to add saturation directive to v_add_i32 instruction to clamp integer arithmetic
      module.addInst("_v_add_i32", \
                     dst, src0, src1, \
                     comment)
    elif kernel["ProblemType"]["DataType"].isSingle():
      module.addInst("v_add_f32", \
                     dst, src0, src1, \
                     comment)
    else:
       #support for double
      module.addInst("v_add_f64", \
                     dst, src0, src1, \
                     comment)

    return module

  ##############################################################################
  def applyAlpha(self, kernel, gwvw, elementSumIdx, elementIdx, tmpS01):
    module = Code.Module("applyAlpha")

    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      return module

    if self.do["ApplyAlpha"]:
      for vi in range(0, gwvw):
        sumIdxV = elementSumIdx[elementIdx] + vi

        if kernel["ProblemType"]["ComputeDataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # (h,h,h,h,h,h), internal alpha is f16 (2-16bits)
          if sumIdxV%2:
            module.addInst("v_pk_mul_f16", vgpr("ValuC+%u"%(sumIdxV//2)), sgpr("Alpha"), vgpr("ValuC+%u"%(sumIdxV//2)), "*= alpha sumIdx=%u vi=%u"%(elementSumIdx[elementIdx], vi))

        # Int8 (TODO- Int8x4 not checked, but should be OK)
        elif kernel["ProblemType"]["ComputeDataType"].isInt32():
          # below assume we use v_mul_lo_u32. Could also use v_mul_i32_i24.
          # module.addInst("v_mul_i32_i24", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha" )
          module.addInst("v_mul_lo_u32", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha" )
          if self.db["ForceExpectedValue"]:
            module.addInst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.db["ValueCExpectedValue"], "force expected value" )
          if self.db["CheckValueC"]:
            module.addInst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
            module.addCode(self.getCmpAssert(self.asmAssert.eq, vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01)))

        # sgemm, HPA-bfgemm(b,b,b,b,s,s), and HPA-hgemm(h,h,h,h,s,s)
        # (h,h,h,h,h,h) + HPA (will be converted to (h,h,h,h,s,s)), internal alpha is single
        elif kernel["ProblemType"]["ComputeDataType"].isSingle() or (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):
          module.addInst("v_mul_f32", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha" )
          if self.db["ForceExpectedValue"]:
            module.addInst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.db["ValueCExpectedValue"], "force expected value" )
          if self.db["ForceVSerial"]:
            module.addInst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), vgpr("Serial"), "force expected value to serial" )
          if self.db["CheckValueC"]:
            module.addInst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
            module.addCode(self.getCmpAssert(self.asmAssert.eq, vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01)))

        # dgemm
        elif kernel["ProblemType"]["ComputeDataType"].isDouble():
          module.addInst("v_mul_f64", vgpr("ValuC+%u"%(sumIdxV*2),2), sgpr("Alpha",2), vgpr("ValuC+%u"%(sumIdxV*2),2), "*= alpha")

        # single precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
          tmpVgpr = self.vgprPool.checkOut(1)
          module.addInst("v_mov_b32", vgpr(tmpVgpr), vgpr("ValuC+%u"%(sumIdxV*2)), "store Cr")
          module.addInst("v_mul_f32", vgpr("ValuC+%u"%(sumIdxV*2)), sgpr("Alpha"), vgpr("ValuC+%u"%(sumIdxV*2)), "*= alpha ( Cr = Ar * Cr)")
          module.addInst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2)), "-" + sgpr("Alpha+1"), vgpr("ValuC+%u"%(sumIdxV*2+1)), "*= alpha ( Cr += -Ai * Ci )")
          module.addInst("v_mul_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), sgpr("Alpha"), vgpr("ValuC+%u"%(sumIdxV*2+1)), "*= alpha ( Ci = Ar * Ci)")
          module.addInst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), sgpr("Alpha+1"), vgpr(tmpVgpr), "*= alpha ( Ci += Ai * Cr_backup )")
          self.vgprPool.checkIn(tmpVgpr)

        # double precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
          vtmp1 = self.vgprPool.checkOutAligned(2, 2)
          vtmp2 = self.vgprPool.checkOutAligned(2, 2)
          # tmp1 = a.real * b.real
          module.addInst("v_mul_f64", vgpr(vtmp1,2), sgpr("Alpha+0",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), "")
          # tmp2 = a.imag * b.real
          module.addInst("v_mul_f64", vgpr(vtmp2,2), sgpr("Alpha+2",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), "")
          # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
          module.addInst("v_fma_f64", vgpr("ValuC+%u"%(sumIdxV*4+0),2), sgpr("Alpha+2",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(vtmp1,2), "")
          # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
          module.addInst("v_fma_f64", vgpr("ValuC+%u"%(sumIdxV*4+2),2), sgpr("Alpha+0",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(vtmp2,2), "")
          self.vgprPool.checkIn(vtmp1)
          self.vgprPool.checkIn(vtmp2)

    return module

  ##############################################################################
  # Global Read C Input
  ##############################################################################
  def readCInput(self, kernel, ss, addrCalc, vc0, data, gwvw, addr, tmpS01):
    module = Code.Module("readCInput")
    bps = kernel["ProblemType"]["DestDataType"].numBytes() * gwvw
    useBuffer = kernel["BufferStore"]

    if kernel["BufferStore"]:
      addr0 = vgpr(addr)
      addr1 = sgpr("SrdC", 4)
    else:
      addr0 = vgpr(addr,2)
      addr1 = ""

    extraStr = ""
    if kernel["NonTemporalC"]%2==1:
      extraStr += " glc"
    if kernel["NonTemporalC"]//2==1:
      extraStr += " slc"

    if ss.optSrdIncForRow and addrCalc.rowInc:
      module.addCode(addrCalc.incrementToNextRow(kernel, "C", ss, tmpS01))

    if kernel["ProblemType"]["DestDataType"].isHalf():
      module.addCode(self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=0, offset=addrCalc.globalOffset, \
                extraFields=extraStr, hi16=vc0 % 2,
                comment="load C for beta calc"))
    elif kernel["ProblemType"]["DestDataType"].isInt8():
     module.addCode(self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=0, offset=addrCalc.globalOffset, \
                extraFields=extraStr, hi16=vc0 % 4,
                comment="load C for beta calc"))
    elif kernel["ProblemType"]["DestDataType"].isBFloat16() or \
         kernel["ProblemType"]["DestDataType"].isInt32() or \
         kernel["ProblemType"]["DestDataType"].isSingle() or \
         kernel["ProblemType"]["DestDataType"].isDouble() or \
         kernel["ProblemType"]["DestDataType"].isSingleComplex() or \
         kernel["ProblemType"]["DestDataType"].isDoubleComplex():
      module.addCode(self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=0, offset=addrCalc.globalOffset, \
                extraFields=extraStr, \
                comment="load C for beta calc"))

    return module

  ##############################################################################
  # Global Write Batch
  ##############################################################################
  def globalWriteBatch(self, kernel, activation, ss, batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
      batchElements, addrD, addrC, \
      tmpVgpr, bf16CVTVgprStruct, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha):
    module = Code.Module("globalWriteBatch (Atomic)") if atomic else Code.Module("globalWriteBatch (Non atomic)")

    module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
              (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

    if kernel["StoreSyncOpt"]:
      module.addInst("s_sleep", "%d" % (kernel["StoreSyncOpt"]-1), "optimization: sync and wait")
      module.addInst("s_barrier", "")

    if atomic:
      # all kinds of code relies on this assumption:
      assert(atomicW <= gwvw)
      if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) \
         and not kernel["_GlobalAccumulation"]:
        assert(atomicW >= 2)

    # comment tt1, tt0, vc1, vc0
    # tt = thread tile, vc=vector component
    commentStr = "Global Write%s%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
        % (" Beta" if beta else "", " Edge" if edge else "", batchIdx)
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      commentStr += "(%u,%u,%u,%u:vw%u%s)" % \
        (element[0], element[1], element[2], element[3], gwvw,
         ":vaw:%u"%atomicW if atomic else "")
      if elementIdx < len(batchElements)-1:
        commentStr += "; "
    module.addComment2(commentStr)

    ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=False, \
                                  allowLRVWforTLUandMI=self.allowLRVWforTLUandMI, lrvwB=self.lrvwB)

    loadsIssued = 0
    storesIssued = 0
    tmpS01 = tmpSgpr # scratch sgprs
    tmpS23 = tmpS01+self.laneSGPRCount

    wavelen = self.kernel["WavefrontSize"]
    laneSGPRC = self.laneSGPRCount

    ########################################
    # calculate addr and masks
    module.addComment1("calc coords, apply mask, and issue loads (if necessary)")
    # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
    # on the thread and tid number.  These are ELEMENT offsets from start of tensor C
    # for the top-left corner this thread will write.  These are not changed
    # across all the store loop iters.
    if self.db["ConservativeWaitCnt"] & 0x10:
      module.addInst("s_barrier", "debug")
      module.addInst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
      if self.archCaps["SeparateVscnt"]:
        module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
      module.addInst("s_barrier", "debug")
    if not edge and self.db["ForceEdgeStores"]>=2:
      module.addInst(self.getBomb()) # should not get here
    if edge and self.db["AssertNoEdge"]:
      module.addInst(self.getBomb()) # should not get here

    ########################################
    # rC *= alpha
    if not kernel["InterleaveAlpha"] and applyAlpha and self.alphaBeforeLoadC:
      module.addComment1("rC *= alpha batchElements=%s"%batchElements)
      if codeMulAlpha is None:
        for elementIdx in range(0, len(batchElements)):
          module.addCode(self.applyAlpha(kernel, gwvw, ss.elementSumIdx, elementIdx, tmpS01))
      else:
          regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
          for elementIdx in range(0, len(batchElements)):
            for vi in range(0, gwvw):
              module.addCode(replacePlaceHolder(codeMulAlpha.items().pop(0), "__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi)))

    atomicAddC = kernel["AtomicAddC"] and not edge

    loadCInputCode = Code.Module("loadCInputCode")

    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
      addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
      addrCalc = ss.elementAddr[elementIdx]
      data = ss.elementData[elementIdx]
      mask = ss.elementMask[elementIdx]
      sumIdx = ss.elementSumIdx[elementIdx]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]

      module.addCode(addrCalc.emitAddressSetupCode(kernel, ss, tmpVgpr, tmpS01, edge, beta, atomic, elementIdx, addrDVgpr))

      if edge:
        module.addCode(addrCalc.edgeProtectCode(kernel, edge, beta, atomic, mask, tmpSgpr))

      # create code Module to push mov vgpr,acc instructions

      if beta and not atomicAddC:
        module.addCode(addrCalc.emitLdChange(kernel, ss, 'C', edge, beta, mask, (elementIdx == 0), tmpVgpr, addrCVgpr, addrC))
        if kernel["GroupLoadStore"]:
          loadCInputCode.addCode(self.readCInput(kernel, ss, addrCalc, vc0, data, gwvw, addrCVgpr, tmpS01))
        else:
          module.addCode(self.readCInput(kernel, ss, addrCalc, vc0, data, gwvw, addrCVgpr, tmpS01))
        loadsIssued += 1

      module.addCode(addrCalc.emitLdChange(kernel, ss, 'D', edge, beta, mask, (elementIdx == len(batchElements)-1), tmpVgpr, addrDVgpr, addrD))

      if atomic and (not self.useAtomicAdd):
        # load c into data+1 because of CAS structure
        # TODO - Fix for double here, would need bigger load
        # FIME
        bps = kernel["ProblemType"]["DestDataType"].numBytes()
        # gwvw is the number of elements in the batch
        # iterate over number of atomic operations to perform, each of width atomicW
        for avi in range(0, gwvw//atomicW):
          dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
          bpm = self.bpeCexternal * atomicW
          useBuffer = kernel["BufferStore"]
          if kernel["BufferStore"]: # yes, BufferStore here - use same addressing regs for this load
            addr0 = vgpr(addrDVgpr)
            addr1 = sgpr("SrdD", 4)
          else:
            addr0 = vgpr(addrDVgpr,2)
            addr1 = ""
          # Calculate vgpr Index for 32-bit/64-bit instruction
          # DGEMM use SRCS[2] register
          vgprIdx = 1*(bpm//4)
          module.addCode(self.chooseGlobalRead(useBuffer, bpm, dataV+vgprIdx, \
                    addr0, addr1, soffset=0, offset=addrCalc.globalOffset, extraFields="",
                    comment="load D (atomic) bpm=%u vaw=%u"%(bpm,atomicW)))

      if kernel["InterleaveAlpha"] and applyAlpha:
        module.addCode(self.applyAlpha(kernel, gwvw, ss.elementSumIdx, elementIdx, tmpS01))

      if not kernel["BufferStore"]:
        offsetSrc = (tmpVgpr+2) if beta else addrDVgpr

        module.addInst("_v_add_co_u32",  vgpr(addrDVgpr+0), self.vcc, vgpr(addrD+0), \
            vgpr(offsetSrc+0), "addrDVgpr = D + index*bytes (lo)" )
        module.addInst("_v_addc_co_u32", vgpr(addrDVgpr+1), self.vcc, vgpr(addrD+1), \
            vgpr(offsetSrc+1), self.vcc, "addrDVgpr = D + index*bytes (hi)")

        # restore full exec mask for calculating addr of next element
        if edge and (beta or atomic):
          module.addInst("s_mov_b{}".format(kernel["WavefrontSize"]), self.exec, -1, "full mask -1 -> exec" )

    module.addCode(loadCInputCode)

    if beta and kernel["StoreSyncOpt"]:
      module.addInst("s_sleep", "%d" %(kernel["StoreSyncOpt"]-1), "optimization: sync and wait")
      module.addInst("s_barrier", "")

    ########################################
    # AccVgpr read
    if kernel.enabledSetPrioSplitLDS:
      module.addInst("s_setprio", "0", "")
    if codeAccVgprRead is not None:
      regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
      # loop over store instructions within one batch
      for elementIdx in range(0, len(batchElements)):
        # loop over scalars within one store instruction
        for vi in range(0, gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            module.addCode(replacePlaceHolder(codeAccVgprRead.items().pop(0), "__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx)))

      if not kernel["MIArchVgpr"]:
        module.addInst("s_nop", "1", "2 wait states required before reading vgpr")

    ########################################
    # rC *= alpha
    if not kernel["InterleaveAlpha"] and applyAlpha and not self.alphaBeforeLoadC:
      module.addComment1("rC *= alpha batchElements=%s"%batchElements)
      if codeMulAlpha is None:
        for elementIdx in range(0, len(batchElements)):
          module.addCode(self.applyAlpha(kernel, gwvw, ss.elementSumIdx, elementIdx, tmpS01))
      else:
          regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
          for elementIdx in range(0, len(batchElements)):
            for vi in range(0, gwvw):
              module.addCode(replacePlaceHolder(codeMulAlpha.items().pop(0), "__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi )))

    ########################################
    # Atomic
    ########################################
    # flat_atomic_cmpswap tmp addr data:
    #   tmp = mem[addr]
    #   src = data[vi*numVgprsPerDataPerVI][0] new C
    #   cmp = data[vi*numVgprsPerDataPerVI][1] original C
    #   mem[addr] = (tmp==cmp) ? src : tmp
    #   addr = vgpr(addr,2)
    #   data = vgpr(tmpVgpr,2)
    #   tmp = vgpr(tmpVgpr+4)

    # buffer_atomic_cmpswap:
    #   dest is 64 bits, two consec VGPR:
    #     - lower is desired swap value (computed new value) "src"
    #       src = data[vi*numVgprsPerDataPerVI][0] new C
    #     - upper is expected value in memory (from prev load).  "cmp".
    #       cmp = data[vi*numVgprsPerDataPerVI][1] original C
    #   src0 is address offset from SRD
    #
    # After buffer_atomic_cmpswap:
    #   dest =
    #       - data[vi*numVgprsPerDataPerVI][0] C loaded from memory, overwrites src
    if atomic:
      del tmpVgpr # catch bugs
      # TODO for atomic GWVW:
      #  - Use vi to compute addresses, sumIdx.
      #  - Need a solution for the mask.  Can move to all buffer or can fix?

      element = batchElements[0]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]
      labelString = "Global_Write%s%s_%u_%u_%u_%u" % ("_Beta" if beta else "", "_Edge" if edge else "", vc0, vc1, d0, d1 )
      labelComment = "Global_Write (Beta) (Edge) vc0 vc1 d0 d1"
      label = Code.Label(self.labels.getName(labelString), labelComment)
      labelString += "_EarlyExit"
      labelAfterAtomicLoop = Code.Label(self.labels.getName(labelString), labelComment)

      if self.useAtomicAdd:
        ########################################
        # first attempt write
        module.addComment1("issue first atomic writes")
        for elementIdx in range(0, len(batchElements)):
          element  = batchElements[elementIdx]
          addrCalc = ss.elementAddr[elementIdx]
          mask     = ss.elementMask[elementIdx]
          d1       = element[0]
          d0       = element[1]
          vc1      = element[2]
          vc0      = element[3]

          # apply in-bounds exec mask
          if edge:
            module.addInst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec (before atomic)" )

          for avi in range(0, gwvw//atomicW):
            dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
            sumIdxV = ss.elementSumIdx[elementIdx] + avi
            if self.do["GlobalWrite"]:
              if kernel["BufferStore"]:
                module.addInst("buffer_atomic_add_f32", \
                               vgpr("ValuC+%u"%sumIdxV), \
                               vgpr(addrCalc.addrDVgpr,1), \
                               sgpr("SrdD", 4), \
                               "0", "offen", "offset:%u" % addrCalc.globalOffset, \
                               "attempt write avi=%u" % (avi))
              else:
                pass # TODO:

        if edge:
          module.addInst("s_mov_b{}".format(wavelen), self.exec, -1, "full mask -> exec" )
      else:
        ########################################
        # wait for batched load
        # TODO - we are always atomic here?
        module.addInst("s_waitcnt", "vmcnt(0)", "wait C (atomic)" )
        if self.archCaps["SeparateVscnt"]:
          module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

        ########################################
        # first attempt write
        module.addComment1("issue first atomic writes")
        for elementIdx in range(0, len(batchElements)):
          element = batchElements[elementIdx]
          addrCalc = ss.elementAddr[elementIdx]
          mask = ss.elementMask[elementIdx]
          d1 = element[0]
          d0 = element[1]
          vc1 = element[2]
          vc0 = element[3]

          # apply in-bounds exec mask
          if edge:
            module.addInst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec (before atomic)" )

          for avi in range(0, gwvw//atomicW):
            dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
            sumIdxV = ss.elementSumIdx[elementIdx] + avi
            ## number of src[s]/dst[s] register for DGEMM / SGEMM HGEMM
            vgprCnt = 2 if kernel["ProblemType"]["DestDataType"].isDouble() else 1
            if kernel["ProblemType"]["DestDataType"].numRegisters() < 1 and not kernel["_GlobalAccumulation"]:
              sumIdxV //= 2
            if kernel["ProblemType"]["DestDataType"].isDouble(): sumIdxV = sumIdxV * 2
            bpm = self.bpeCexternal * atomicW
            # Calculate vgpr Index for 32-bit/64-bit instruction
            # DGEMM use SRCS[2] register
            vgprIdx = 1*(bpm//4)
            # for atomic, data[1] = original c, data[0] = new c
            module.addCode(self.chooseAddForAtomic(kernel, \
                      vgpr(dataV+0,vgprCnt), vgpr(dataV+1*vgprIdx,vgprCnt), vgpr("ValuC+%u"%sumIdxV,vgprCnt), \
                      "desired value avi=%u"%avi))

            # attempt write
            atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
            if self.do["GlobalWrite"]:
              if kernel["BufferStore"]:
                # use cmpswap_x2 for DGEMM in CAS loop
                if kernel["ProblemType"]["DestDataType"].isDouble():
                  module.addInst("_buffer_atomic_cmpswap_b64", \
                                 vgpr(dataV,4), \
                                 vgpr(addrCalc.addrDVgpr,1), \
                                 sgpr("SrdD", 4),  \
                                 "0", "offen", "offset:%u" % addrCalc.globalOffset, "glc", \
                                 "attempt write avi=%u"%(avi))
                else:
                # use cmpswap for SGEMM in CAS loop
                  module.addInst("_buffer_atomic_cmpswap_b32", \
                                 vgpr(dataV,2), \
                                 vgpr(addrCalc.addrDVgpr,1), \
                                 sgpr("SrdD", 4),  \
                                 "0", "offen", "offset:%u" % addrCalc.globalOffset, "glc", \
                                 "attempt write avi=%u"%(avi))
              else:
                module.addInst("_flat_atomic_cmpswap_b32", \
                               vgpr(atomicDestVgpr), vgpr(addrCalc.addrDVgpr,2), \
                               vgpr(dataV,2), "glc", "attempt write" )
            else:
               module.addInst("v_mov_b32", vgpr(atomicDestVgpr), vgpr(dataV+1), "Fake successful CAS" )
               # Fake successful CAS swap:

        ########################################
        # wait for first attempt write
        module.addInst("s_waitcnt vmcnt(0)", "wait for atomic writes" )
        if self.archCaps["SeparateVscnt"]:
          module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

        ########################################
        # check first attempt
        module.addComment1("check success of writes, update masks")
        for elementIdx in range(0, len(batchElements)):
          element = batchElements[elementIdx]
          mask = ss.elementMask[elementIdx]
          d1 = element[0]
          d0 = element[1]
          vc1 = element[2]
          vc0 = element[3]

          # calculate new masks
          if edge:
            module.addInst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec" )
            for avi in range(0, gwvw//atomicW):
              dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
              atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
              # need to apply element mask before comparison
              # so that all valid lanes are doing the cmp
              if avi == 0:
                # use u64 for DGEMM
                if kernel["ProblemType"]["DestDataType"].isDouble():
                  module.addInst("v_cmp_ne_u64", sgpr(tmpS01,laneSGPRC), vgpr(atomicDestVgpr,2), \
                      vgpr(dataV+2,2), "c read during atomic == c read during prior load (avi=%u, first)"%avi )
                else:
                  module.addInst("v_cmp_ne_u32", sgpr(tmpS01,laneSGPRC), vgpr(atomicDestVgpr), \
                      vgpr(dataV+1), "c read during atomic == c read during prior load (avi=%u, first)"%avi )
              else:
                if kernel["ProblemType"]["DestDataType"].isDouble():
                  module.addInst("v_cmp_ne_u64", sgpr(tmpS23,laneSGPRC), vgpr(atomicDestVgpr,2), \
                      vgpr(dataV+2,2), "c read during atomic != c read during prior load" )
                else:
                  module.addInst("v_cmp_ne_u32", sgpr(tmpS23,laneSGPRC), vgpr(atomicDestVgpr), \
                      vgpr(dataV+1), "c read during atomic == c read during prior load (avi=%u)"%avi )
                module.addInst("s_or_b{}".format(wavelen), sgpr(tmpS01,laneSGPRC), \
                      sgpr(tmpS01,laneSGPRC), sgpr(tmpS23,laneSGPRC), "combine with tmp mask")

            module.addInst("s_and_b{}".format(wavelen),  sgpr(mask,laneSGPRC), sgpr(tmpS01,laneSGPRC), sgpr(mask,laneSGPRC), "inBounds & must try again" )

          else:
            for avi in range(0, gwvw//atomicW):
              dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
              atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
              if kernel["ProblemType"]["DestDataType"].isDouble():
                module.addInst("v_cmp_ne_u64", sgpr(mask,laneSGPRC), vgpr(atomicDestVgpr,2), \
                    vgpr(dataV+2,2), "c read during atomic != c read during prior load" )
              else:
                module.addInst("v_cmp_ne_u32", sgpr(mask,laneSGPRC), vgpr(atomicDestVgpr), \
                    vgpr(dataV+1), "c read during atomic != c read during prior load" )

        # or masks together to check early exit
        module.addComment1("or masks to check for exit")
        module.addInst("s_mov_b{}".format(wavelen), sgpr(tmpS01,laneSGPRC), hex(0), "empty mask" )
        for elementIdx in range(0, len(batchElements)):
          mask = ss.elementMask[elementIdx]
          module.addInst("s_or_b{}".format(wavelen), sgpr(tmpS01,laneSGPRC), sgpr(mask,laneSGPRC), sgpr(tmpS01,laneSGPRC), "or to add threads" )
        module.addInst("s_or_saveexec_b{}".format(wavelen), sgpr(tmpS23,laneSGPRC), sgpr(tmpS01,laneSGPRC), "apply combined mask" )
        module.addInst("s_cbranch_execz", labelAfterAtomicLoop.getLabelName(), "if exec is zero skip loop" )

        # begin atomic loop
        module.addComment1("atomic CAS loop")
        module.addCode(label)

        module.addComment1("apply updated masks and issue writes again")
        for elementIdx in range(0, len(batchElements)):
          element = batchElements[elementIdx]
          addrCalc = ss.elementAddr[elementIdx]
          addr = ss.elementAddr[elementIdx].addrDVgpr
          mask = ss.elementMask[elementIdx]
          vgprCnt = 2 if kernel["ProblemType"]["DestDataType"].isDouble() else 1   # number of registers for f32/f64
          bpm = self.bpeCexternal * atomicW
          vgprIdx = 1*(bpm//4)   # index register

          for avi in range(0, gwvw//atomicW):
            dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
            atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
            sumIdxV = ss.elementSumIdx[elementIdx] + avi
            if kernel["ProblemType"]["DestDataType"].numRegisters() < 1 and not kernel["_GlobalAccumulation"]:
              sumIdxV //= 2
            if kernel["ProblemType"]["DestDataType"].isDouble():  sumIdxV =  sumIdxV * 2

            # apply mask for element
            module.addInst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "must try again" )
            if kernel["ProblemType"]["DestDataType"].isDouble():
              #64-bit C val move by 2 32-bit instructions
              module.addInst("v_mov_b32", vgpr(dataV+2), vgpr(atomicDestVgpr), "dataV+2 = tmp (new original C)" )
              module.addInst("v_mov_b32", vgpr(dataV+3), vgpr(atomicDestVgpr+1), "dataV+3 = tmp (new original C)" )
            else:
              module.addInst("v_mov_b32", vgpr(dataV+1), vgpr(atomicDestVgpr), "dataV+1 = tmp (new original C)" )
            module.addCode(self.chooseAddForAtomic(kernel, \
                           vgpr(dataV+0,vgprCnt), vgpr(dataV+1*vgprIdx,vgprCnt), vgpr("ValuC+%u"%sumIdxV,vgprCnt), \
                           "newC = rC + originalC"))
            if self.do["GlobalWrite"]:
              if kernel["BufferStore"]:
                # Using no-ret version here?
                # cmpswap_x2 for DGEMM
                if kernel["ProblemType"]["DestDataType"].isDouble():
                  module.addInst("_buffer_atomic_cmpswap_b64", \
                                 vgpr(dataV,4), \
                                 vgpr(addr,1), \
                                 sgpr("SrdD", 4), \
                                 "0", "offen", "offset:%u" % (addrCalc.globalOffset), "glc", \
                                 "try again")
                else:
                  module.addInst("_buffer_atomic_cmpswap_b32", \
                                 vgpr(dataV,2), \
                                 vgpr(addr,1), \
                                 sgpr("SrdD", 4), \
                                 "0", "offen", "offset:%u" % (addrCalc.globalOffset), "glc", \
                                 "try again")
              else:
                module.addInst("_flat_atomic_cmpswap_b32", vgpr(atomicDestVgpr), \
                    vgpr(addr,2), vgpr(dataV,2), "glc", "try again")

        # wait for batched write
        module.addInst("s_waitcnt vmcnt(0)", "wait for atomic writes" )
        if self.archCaps["SeparateVscnt"]:
          module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

        # check batched write success
        module.addComment1("apply masks and check for success")
        for elementIdx in range(0, len(batchElements)):
          element = batchElements[elementIdx]
          data = ss.elementData[elementIdx]
          mask = ss.elementMask[elementIdx]
          for avi in range(0, gwvw//atomicW):
            dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
            atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2

            # apply mask for element
            module.addInst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "must try again" )

            # compare success
            if kernel["ProblemType"]["DestDataType"].isDouble():
              module.addInst("v_cmp_ne_u64", sgpr(tmpS01,laneSGPRC), vgpr(data+2,2), vgpr(atomicDestVgpr,2), \
                  "c read during atomic != c read during prior load" )
            else:
              module.addInst("v_cmp_ne_u32", sgpr(tmpS01,laneSGPRC), vgpr(data+1), vgpr(atomicDestVgpr), \
                  "c read during atomic == c read during prior load" )
            # update element mask
            module.addInst("s_and_b{}".format(wavelen),  sgpr(mask,laneSGPRC), sgpr(tmpS01,laneSGPRC), sgpr(mask,laneSGPRC), "inBounds & must try again" )

        # or masks together
        module.addComment1("or masks to check for exit")
        module.addInst("s_mov_b{}".format(wavelen), sgpr(tmpS01,laneSGPRC), hex(0), "empty mask" )
        for elementIdx in range(0, len(batchElements)):
          mask = ss.elementMask[elementIdx]
          module.addInst("s_or_b{}".format(wavelen), sgpr(tmpS01,laneSGPRC), sgpr(mask,laneSGPRC), sgpr(tmpS01,laneSGPRC), "or to add threads" )

        # apply combined masks and exit
        module.addInst("s_or_saveexec_b{}".format(wavelen), sgpr(tmpS23,laneSGPRC), sgpr(tmpS01,laneSGPRC), "apply combined mask" )
        module.addInst("s_cbranch_execnz", label.getLabelName(), "try again if not complete" )
        module.addCode(labelAfterAtomicLoop)
        module.addInst("s_mov_b{}".format(wavelen), self.exec, -1, "full mask -> exec" )

    ########################################
    # Not Atomic
    ########################################
    else:
      # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
      interleaveStoreVmcnt = self.interleaveStoreVmcnt and not edge
      for elementIdx in range(0, len(batchElements)):
        for vi in range(0, gwvw):
          sumIdxV = ss.elementSumIdx[elementIdx] + vi
          # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
          if kernel["ProblemType"]["ComputeDataType"].isInt32() or \
             kernel["ProblemType"]["ComputeDataType"].isSingle(): # covers sgemm/gemm_ex(HHS/HSS/BBS/BSS)
              if self.db["ForceExpectedValue"]:
                module.addInst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.db["ValueCExpectedValue"], "force expected value" )
              if self.db["ForceVSerial"]:
                module.addInst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), vgpr("Serial"), "force expected value to serial" )
              if self.db["CheckValueC"]:
                module.addInst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
                module.addCode(self.getCmpAssert(self.asmAssert.eq, vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01)))

      ########################################
      # wait for batched load
      if beta and not interleaveStoreVmcnt:
        module.addInst("s_waitcnt", "vmcnt(0)", "wait C")
        if self.archCaps["SeparateVscnt"]:
          module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

      module.addComment1("apply mask, calc new C and issue writes")
      # module.addCode(self.getBomb()) # can see store addresses just before the store inst

      # Create a suffix and check if the string exists
      activationLabelSuffix = "%s%s%u"%("Beta_" if beta else "", "Edge_" if edge else "", batchIdx)
      if activationLabelSuffix in self.globalWriteIfStateLabelSuffixDict:
        self.globalWriteIfStateLabelSuffixDict[activationLabelSuffix] += 1
        activationLabelSuffix = activationLabelSuffix + "_%u"%self.globalWriteIfStateLabelSuffixDict[activationLabelSuffix]
      else:
        self.globalWriteIfStateLabelSuffixDict[activationLabelSuffix] = 0
      activationCDataType = kernel["ProblemType"]["ComputeDataType"] if kernel["ProblemType"]["ActivationHPA"] else \
                                                                        kernel["ProblemType"]["DestDataType"]
      activationLabelEndModule = Code.Label("Activation_End_%s"%activationLabelSuffix, "")
      activationLabelModules = []
      activationEnumStrList = []
      if ((kernel["_GlobalAccumulation"] != 'MultipleBuffer') and kernel["ActivationFused"]) and \
        (kernel["ProblemType"]["ActivationType"] != 'none'):
        if kernel["ProblemType"]["ActivationType"] == 'all':
          activationEnumStrList = ActivationType.getEnumStrList(activationCDataType)
          for index, enumStr in enumerate(activationEnumStrList):
            activationLabelModule = Code.Label("Activation_%s_%s"% (enumStr.capitalize(), activationLabelSuffix), "")
            activationLabelModules.append(activationLabelModule)
          for index, activationLabelModule in enumerate(activationLabelModules):
            if index != 0:
              enumIndex = ActivationType.getEnumIndex(activationEnumStrList[index])
              module.addInst("s_cmpk_eq_u32", sgpr("ActivationType"), enumIndex, "activationType == %u"%enumIndex)
              module.addInst("s_cbranch_scc1", activationLabelModule.getLabelName(), "Branch if true")
        else:
          activationEnumStrList.append(str(kernel["ProblemType"]["ActivationType"]).lower())
      else:
        activationLabelModules.append("")
        activationEnumStrList.append("none")
      loadsIssuedRestore = loadsIssued
      storesIssuedRestore = storesIssued
      for index, activationLabelModule in enumerate(activationLabelModules):
        loadsIssued = loadsIssuedRestore
        storesIssued = storesIssuedRestore
        activationTypeStr = activationEnumStrList[index]
        if activationLabelModule:
          module.addCode(activationLabelModule)

        if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
          module.addInst("v_mov_b32", vgpr(bf16CVTVgprStruct.vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" )
          module.addInst("v_mov_b32", vgpr(bf16CVTVgprStruct.vgprFp32Nan), "0x7fff0000", "fp32 Nan" )
          module.addInst("v_mov_b32", vgpr(bf16CVTVgprStruct.vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" )

        storeCode = Code.Module("GroupLoadStore")
        for elementIdx in range(0, len(batchElements)):
          element = batchElements[elementIdx]
          addr = ss.elementAddr[elementIdx].addrDVgpr
          mask = ss.elementMask[elementIdx]
          addrCalc = ss.elementAddr[elementIdx]
          d1 = element[0]
          d0 = element[1]
          vc1 = element[2]
          vc0 = element[3]
          sumIdx = ss.elementSumIdx[elementIdx]

          # print(str(element)+" rowInc="+str(addrCalc.rowInc))
          # Already write wave column block into LDS
          # Now read lds data back to registers and write to global memroy
          if ss.optSrdIncForRow and addrCalc.rowInc and kernel["StoreRemapVectorWidth"] > 0:
            module.addComment1("StoreRemap: shift coord1 address")
            module.addCode(addrCalc.incrementToNextRow(kernel, "D", ss, tmpS01))
            module.addInst("v_mov_b32", vgpr(tmpVgpr), addrCalc.rowInc, "set shift rows")
            module.addInst("_v_add_u32", vgpr(self.storeRemapCoord1), vgpr(self.storeRemapCoord1), vgpr(tmpVgpr), "shift storeRemap coord1")

          # apply in-bounds exec mask
          if edge and not kernel["BufferStore"]:
            module.addInst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec" )

          if beta:
            # if GWVW=1 the half path still assumes we have
            # at least two stores so does some combining across VI -
            # for example assuming we can have two elements and can use pk_mul
            # here:
            if beta and interleaveStoreVmcnt:
              if self.archCaps["SeparateVscnt"]:
                vmcnt = loadsIssued - elementIdx - 1
                vmComment = "{} = {} - {} - 1".format(vmcnt, loadsIssued, elementIdx)
              else:
                waitStoreCnt = storesIssued if not kernel["GroupLoadStore"] else 0
                vmcnt = loadsIssued - elementIdx + waitStoreCnt - 1
                vmComment = "{} = {} - {} + {} - 1".format(vmcnt, loadsIssued, elementIdx, waitStoreCnt)

              maxVmcnt = globalParameters["AsmCaps"][self.version]["MaxVmcnt"]
              vmcnt = min(vmcnt, maxVmcnt)
              #print "wmvcnt=", vmcnt
              module.addSpaceLine()
              if not atomicAddC:
                module.addInst("s_waitcnt", "vmcnt(%u)"%vmcnt, "wait C (interleaved) " + vmComment)

            module.addCode(self.addSumAlphaWithCBeta(kernel, ss, gwvw, elementIdx, vc0, atomicAddC, tmpVgpr, bf16CVTVgprStruct))

          SaturateTypeInt8 = SaturateCastType.NORMAL
          # Activation
          activationModule = ""
          isActivationInsertAfter = False
          if self.insertActivationAfterPacked(kernel, activationTypeStr):
            isActivationInsertAfter = True
            activationModule = Code.Module("ActivationAfterPack")
            for vi in range(0, gwvw):
              sumIdxV = ss.elementSumIdx[elementIdx] + vi
              if kernel["ProblemType"]["DestDataType"].isHalf() or \
                 kernel["ProblemType"]["DestDataType"].isBFloat16():
                if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                  # Generate single f16 code if edge is detected.
                  if ((vi + 1) == gwvw) and ((gwvw % 2) == 1):
                    activation.setUsePK(False)
                  # Original packed route
                  elif vi%2 == 1:
                    assert (gwvw % 2 == 0)
                  else:
                    continue
                  vgprIdx = ss.elementSumIdx[elementIdx] + vi//2
                else:
                  if (sumIdxV % 2 != 0):
                    continue
                  vgprIdx = sumIdxV // 2
              elif kernel["ProblemType"]["DestDataType"].isSingle():
                vgprIdx = sumIdxV
              elif kernel["ProblemType"]["DestDataType"].isDouble():
                vgprIdx = sumIdxV * 2
              elif kernel["ProblemType"]["DestDataType"].isInt32():
                vgprIdx = sumIdxV
              else:
                raise RuntimeError("Unsupported data type %s for activation vgpr index."%str(kernel["ProblemType"]["DestDataType"]))
              # Here we still use DestDataType cause the data is ready to be written to global
              actModule = activation.getModule(kernel["ProblemType"]["DestDataType"], activationTypeStr, vgprIdx)
              activationModule.addCode(activation.assignGpr(actModule, tmpVgpr, tmpSgpr))
              activation.setUsePK(True)
          else:
            activationModule = Code.Module("ActivationBeforePack")
            if kernel["ProblemType"]["DestDataType"].isInt8():
              if (activationTypeStr == 'abs') or (activationTypeStr == 'relu'):
                SaturateTypeInt8 = SaturateCastType.DO_NOTHING
                activation.setSaturationForInt8(True)
            activation.setVgprPrefixFormat("ValuC+%u")
            for vi in range(0, gwvw):
              vgprIdx = ss.elementSumIdx[elementIdx] + vi
              actModule = activation.getModule(activationCDataType, activationTypeStr, vgprIdx)
              activationModule.addCode(activation.assignGpr(actModule, tmpVgpr, tmpSgpr))
            activation.setSaturationForInt8(False)
            activation.setVgprPrefixFormat("")

          # pack stores, beta and non-beta reach here:
          packModule = Code.Module("Empty pack module")
          if kernel["ProblemType"]["HighPrecisionAccumulate"] and (kernel["_GlobalAccumulation"] != 'MultipleBuffer'):
            packdata = Component.PackData.find(self)
            if kernel["ProblemType"]["DestDataType"].isHalf():
              packModule = packdata(gwvw, ss.elementSumIdx[elementIdx], inputPrefix="ValuC+")
            elif kernel["ProblemType"]["DestDataType"].isBFloat16():
              packModule = packdata(gwvw, ss.elementSumIdx[elementIdx], bf16CVTVgprStruct=bf16CVTVgprStruct,
                                    tmpS01=tmpS01, laneSGPRC=laneSGPRC, inputPrefix="ValuC+")
            elif kernel["ProblemType"]["DestDataType"].isInt8():
              packModule = packdata(gwvw, ss.elementSumIdx[elementIdx], tmpVgpr, tmpS01,
                                    SaturateTypeInt8=SaturateTypeInt8, inputPrefix="ValuC+")

          if isActivationInsertAfter:
            module.addCode(packModule)
            module.addCode(activationModule)
          else:
            module.addCode(activationModule)
            module.addCode(packModule)

          if not kernel["StoreRemapVectorWidth"]:
            tmpStoreCode = self.addStore(kernel, ss, addrCalc, sumIdx, tmpS01, edge)
            if kernel["GroupLoadStore"]:
              storeCode.addCode(tmpStoreCode)
            else:
              module.addCode(tmpStoreCode)
            storesIssued += 1

          else:
            rpe = self.bpeCinternal//self.bpr
            module.addCode(self.storeRemapAddLocalWrite(kernel, ss, addrCalc, sumIdx*rpe))
            # Column Block Shape has been written to LDS
            # Now read back and write out to global memory

        module.addCode(storeCode)

        if self.db["CheckStoreC"]>=0:
          useBuffer = kernel["BufferStore"]
          # Note - CheckStoreC won't work for EDGE store cases since they load 0 for OOB, would need more sophisticated check
          # Note - TODO- CheckStoreC also won't work for StoreRemap
          module.addInst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
          if self.archCaps["SeparateVscnt"]:
            module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
          for elementIdx in range(0, len(batchElements)):
            addr = ss.elementAddr[elementIdx].addrDVgpr
            sumIdx = ss.elementSumIdx[elementIdx]

            bps = kernel["ProblemType"]["DestDataType"].numBytes() * gwvw
            if kernel["BufferStore"]:
              addr0 = vgpr(addr)
              addr1 = sgpr("SrdC", 4)
            else:
              addr0 = vgpr(addr,2)
              addr1 = ""

            if kernel["ProblemType"]["DestDataType"].isHalf() or kernel["ProblemType"]["DestDataType"].isBFloat16():
              if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
                module.addCode(self.chooseGlobalRead(useBuffer, bps, sumIdx//2, \
                                      addr0, addr1, soffset=0, offset=0, extraFields="", hi16=sumIdx%2))
              else:
                module.addCode(self.chooseGlobalRead(useBuffer, bps, sumIdx, \
                                      addr0, addr1, soffset=0, offset=0, extraFields="", hi16=0))
            elif kernel["ProblemType"]["DestDataType"].isInt32() or kernel["ProblemType"]["DestDataType"].isSingle():
              module.addCode(self.chooseGlobalRead(useBuffer, bps, sumIdx, \
                                    addr0, addr1, soffset=0, offset=0, extraFields=""))
            elif kernel["ProblemType"]["DestDataType"].isDouble() or kernel["ProblemType"]["DestDataType"].isSingleComplex() :
              module.addCode(self.chooseGlobalRead(useBuffer, bps, sumIdx*2, \
                                    addr0, addr1, soffset=0, offset=0, extraFields=""))
            elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
              module.addCode(self.chooseGlobalRead(useBuffer, bps, sumIdx*4, \
                                    addr0, addr1, soffset=0, offset=0, extraFields=""))
          module.addInst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
          if self.archCaps["SeparateVscnt"]:
            module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

          # Add checks for expected values:
          module.addInst("s_mov_b32", sgpr(tmpS01), self.db["CheckStoreC"], "expected value")
          for elementIdx in range(0, len(batchElements)):
            sumIdx = ss.elementSumIdx[elementIdx]
            # Need to fix for other types:
            assert (kernel["ProblemType"]["DestDataType"].isSingle() or kernel["ProblemType"]["DestDataType"].isInt32())
            module.addCode(self.getCmpAssert(self.asmAssert.eq, vgpr(sumIdx), sgpr(tmpS01)))


        if edge and (atomic or not kernel["BufferStore"]):
          # subsequent batch must start with full exec mask
          # BufferStore doesn't need exec since it used buffer range checking when
          # possible
          module.addInst("s_mov_b{}".format(wavelen), self.exec, -1, "full mask -> exec" )

        if self.db["ConservativeWaitCnt"] & 0x40:
          module.addInst("s_barrier", "debug")
          module.addInst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
          if self.archCaps["SeparateVscnt"]:
            module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
          module.addInst("s_barrier", "debug")

        if (index < (len(activationLabelModules) - 1)):
          module.addInst("s_branch", activationLabelEndModule.getLabelName(), "")
      if len(activationLabelModules) > 1:
        module.addCode(activationLabelEndModule)

    # return registers to pool:
    lastData = -1
    for elementIdx in range(0, len(batchElements)):
      if not ss.sharedColDVgprs:
        addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
        addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
        self.vgprPool.checkIn(addrDVgpr)
        if addrCVgpr != addrDVgpr:
          self.vgprPool.checkIn(addrCVgpr)

      data = ss.elementData[elementIdx]
      if data != 0:
        if data != lastData:
          self.vgprPool.checkIn(data)
        lastData = data

    ss.firstBatch = False
    ss.checkInTempVgprC()
    if kernel["StoreRemapVectorWidth"]:
      if self.StoreRemapLastBatch == 1:
        module.addComment1("Handle local read and global write")
        # this seems buggy? it's possible to issue more than one stores for SR
        # module.addCode(self.storeRemapAddStore(kernel, ss, addrCalc, tmpVgpr, tmpS01, edge))
        # storesIssued += 1
        storeModule, numNewStores = self.storeRemapAddStore(kernel, ss, addrCalc, tmpVgpr, tmpS01, edge)
        module.addCode(storeModule)
        storesIssued += numNewStores

    if self.serializedStore:
      module.addInst("s_nop", "0", "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst")

    return module

  ##############################################################################
  def openPrefetchGlobalRead2(self, kernel):
    imod = Code.Module()
    loopCounter = self.loopCounter(kernel, self.unrollIdx)
    imod.addInst("s_cmp_eq_u32 %s %s" %(loopCounter, hex(1)),"PGR=2 but only 1 loop")
    skipPGR2 = Code.Label(self.labels.getName("skipPGR2"), "")
    imod.addInst("s_cbranch_scc1", skipPGR2.getLabelName(),"PGR=2 but only 1 loop")
    return imod

  def closePrefetchGlobalRead2(self, kernel):
    imod = Code.Module()
    skipPGR2 = Code.Label(self.labels.getName("skipPGR2"), "")
    imod.addCode(skipPGR2)
    return imod

  def addSumAlphaWithCBeta(self, kernel, ss, gwvw, elementIdx, vc0, atomicAddC, tmpVgpr, bf16CVTVgprStruct):
    module = Code.Module("addSumAlphaWithCBeta #elementIdx%u, vc0 %u"%(elementIdx, vc0))
    for vi in range(0, gwvw):
      dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
      sumIdxV = ss.elementSumIdx[elementIdx] + vi
      if kernel["ProblemType"]["DestDataType"].isHalf():
        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          if sumIdxV%2==0 or (not ss.cfg.halfDataRegPerVI and gwvw==1):
            # dataV+0 = new c = old c*beta
            module.addInst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
                "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
            # dataV+0 = new c = old c*beta + rC
            module.addInst("v_pk_add_f16", vgpr("ValuC+%u"%(sumIdxV//2)), vgpr(dataV), vgpr("ValuC+%u"%(sumIdxV//2)), \
                "sum*alpha + C*beta")
          else:
            pass # add will have been done previously
        else: # HPA
          # dataV+0 = new c = old c*beta + rC
          # src0 = beta = f32 = opsel 00
          # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
          # src2 = sumIdxV = f32 = opsel 00
          dataCExternal = ss.elementData[elementIdx] + vi//2
          hi16 = (vi + gwvw*vc0) % 2
          module.addInst(self.mixinst, vgpr("ValuC+%u"%sumIdxV), sgpr("Beta"), \
              vgpr(dataCExternal), vgpr("ValuC+%u"%sumIdxV), \
              "op_sel:[0,%u,0] op_sel_hi:[0,1,0]" % (hi16), \
              "//C*=beta")

      elif kernel["ProblemType"]["DestDataType"].isBFloat16():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # dataV+0 = new c = old c*beta + rC
          # src0 = beta = f32 = opsel 00
          # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
          # src2 = sumIdxV = f32 = opsel 00
          dataCExternal = ss.elementData[elementIdx] + vi//2
          if (vi%2) == 1:
            module.addInst("v_and_b32", vgpr(tmpVgpr), vgpr(dataCExternal), vgpr(bf16CVTVgprStruct.vgprBf16Mask), "convert bf16 to fp32")
          else:
            module.addInst("v_lshlrev_b32", vgpr(tmpVgpr), "16", vgpr(dataCExternal), "convert bf16 to fp32" )
          module.addInst("_v_mac_f32", vgpr("ValuC+%u"%sumIdxV), vgpr(tmpVgpr), sgpr("Beta"), \
              "finalSum = sum*alpha + C*beta")
      elif kernel["ProblemType"]["DestDataType"].isSingle():
        module.addInst("_v_mac_f32", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), sgpr("Beta"), \
            "finalSum = sum*alpha + C*beta")

      elif kernel["ProblemType"]["DestDataType"].isInt8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          assert (gwvw % 4 == 0)
          if (vi%4) != 3:
            tmpC = tmpVgpr
            module.addInst("v_bfe_i32", vgpr(tmpC), vgpr(dataV+0), (vi * 8), 8, "int8 to int32")
          else:
            tmpC = dataV+0
            module.addInst("v_ashrrev_i32_e32", vgpr(dataV+0), 24, vgpr(dataV+0), "int8 to int32")
          module.addInst("v_mul_lo_u32", vgpr(tmpC), sgpr("Beta"), vgpr(tmpC), \
              "C = C*beta")
          module.addInst("_v_add_u32", vgpr("ValuC+%u"%sumIdxV), vgpr(tmpC), vgpr("ValuC+%u"%sumIdxV), \
              "finalSum = sum*alpha + C*beta")

      elif kernel["ProblemType"]["DestDataType"].isInt32():
        # assume we will need to replace v_mac_f32 with v_add_u32 and s_mul_lo_i32
        # v_mad_i32_i24
        # module.addInst("v_mad_i32_i24", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), sgpr("Beta"), vgpr("ValuC+%u"%sumIdxV), \
        #     "finalSum = sum*alpha + C*beta")
        module.addInst("v_mul_lo_u32", vgpr(dataV+0), sgpr("Beta"), vgpr(dataV+0), \
            "C = C*beta")
        module.addInst("_v_add_u32", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), vgpr("ValuC+%u"%sumIdxV), \
            "finalSum = sum*alpha + C*beta")

      elif kernel["ProblemType"]["DestDataType"].isDouble():
        # dataV+0 = new c = old c*beta
        if not atomicAddC:
          module.addInst("v_fma_f64", vgpr("ValuC+%u"%(sumIdxV*2),2), vgpr(dataV+0,2), sgpr("Beta",2), vgpr("ValuC+%u"%(sumIdxV*2),2), \
              "finalSum = sum*alpha + C*beta")

      # single precision complex
      elif kernel["ProblemType"]["DestDataType"].isSingleComplex():
        module.addInst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2)), vgpr(dataV+0), sgpr("Beta"), "finalSum Cr += old Cr * Br")
        module.addInst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2)), vgpr(dataV+1), "-"+sgpr("Beta+1"), "finalSum Cr += old Ci * -Bi")
        module.addInst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), vgpr(dataV+1), sgpr("Beta"), "finalSum Ci += old Ci * Br")
        module.addInst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), vgpr(dataV+0), sgpr("Beta+1"), "finalSum Ci += old Cr * Bi")

      # double precision complex
      elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
        module.addInst("v_fma_f64", vgpr("ValuC+%u"%(sumIdxV*4+0),2), vgpr(dataV+0,2), sgpr("Beta+0",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), "c.real += a.real * b.real")
        module.addInst("v_fma_f64", vgpr("ValuC+%u"%(sumIdxV*4+0),2), vgpr(dataV+2,2), sgpr("Beta+2",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), "c.real -= a.imag * b.imag")
        module.addInst("v_fma_f64", vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(dataV+0,2), sgpr("Beta+2",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), "c.imag += a.real * b.imag")
        module.addInst("v_fma_f64", vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(dataV+2,2), sgpr("Beta+0",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), "c.imag += a.imag * b.real")
    return module

  ########################################
  # Activation related
  ########################################

  def insertActivationAfterPacked(self, kernel, activationTypeStr):
    result = False
    if ((kernel["ProblemType"]["ActivationType"] != 'none') and \
      (kernel["_GlobalAccumulation"] != 'MultipleBuffer') and kernel["ActivationFused"]):
      if kernel["ProblemType"]["ActivationHPA"]:
        # Still use BFloat16 for abs.
        if kernel["ProblemType"]["DestDataType"].isBFloat16() and (activationTypeStr == 'abs'):
          result = True
        elif kernel["ProblemType"]["DestDataType"].isHalf() and \
           ((activationTypeStr == 'abs') or (activationTypeStr == 'relu')):
          result = True
      else:
        result = True
    return result

  ##############################################################################
  # Function End
  ##############################################################################
  def functionEnd(self, kernel, addLabel=True):
    imod = Code.Module()
    if addLabel:
      imod.addCode(Code.Label("KernelEnd", ""))
    imod.addInst("s_endpgm", "Kernel End")
    return imod

  ##############################################################################
  # Function Suffix
  ##############################################################################
  def functionSuffix(self, kernel):
    if self.vgprPool.size() > self.maxVgprs:
      self.overflowedResources = 1
    elif self.sgprPool.size() > self.maxSgprs:
      self.overflowedResources = 2

    if kernel["ScheduleIterAlg"] == 2 and \
        self.getOccupancy(kernel["NumThreads"], self.vgprPool.size(), \
        self.getLdsSize(kernel), self.agprPool.size(), self.unifiedVgprRegs) < 2:
      self.overflowedResources = 6

    vgprPerCU = 65536
    vgprPerThreadPerOccupancy = vgprPerCU // kernel["NumThreads"]
    numWorkGroupsPerCU = vgprPerThreadPerOccupancy // max(self.vgprPool.size(), self.agprPool.size())
    if numWorkGroupsPerCU < 1:
      self.overflowedResources = 4

    module = Code.Module("functionSuffix")
    if self.overflowedResources:
      module.addInst(".endif", "overflowed resources")

    self.vgprPool.checkFinalState()
    return module

  ##############################################################################
  # openOddNoLoadLoopForDTV
  # generate open code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  def openOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    module = Code.Module("openOddNoLoadLoopForDTV")
    evenStartLabelName = Code.Label.getFormatting("EvenStart" + name)
    # odd exit check code
    # use OrigLoopCounter & 1
    tmpSgpr = self.getTmpSgpr(1).idx()
    #scc0or1 = 0 if isNGLL else 1
    #oddOrEven = "Even" if isNGLL else "Odd"
    module.addInst("s_and_b32",sgpr(tmpSgpr), sgpr("OrigLoopCounter"), 1, "test if OrigLoopCounter is Odd ?")
    module.addInst("s_cbranch_scc0", evenStartLabelName, "Skip odd code if OrigLoopCounter is Even")

    return module

  ##############################################################################
  # closeOddNoLoadLoopForDTV
  # generate close code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  def closeOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    module = Code.Module("closeOddNoLoadLoopForDTV")
    evenStartLabelName = Code.Label.getFormatting("EvenStart" + name)
    evenEndLabelName = Code.Label.getFormatting("EvenEnd" + name)
    # odd exit code
    module.addInst("s_branch", evenEndLabelName, "Skip even code")
    # generate even start label
    module.addCode(Code.Label(evenStartLabelName, ""))
    return module

  ##############################################################################
  # generateEvenEndLabeNoLoadLoopForDTV
  # generate even end label for DirectToVgpr
  ##############################################################################
  def generateEvenEndLabeNoLoadLoopForDTV(self, kernel, isNGLL, name):
    module = Code.Module("generateEvenEndLabeNoLoadLoopForDTV")
    evenEndLabelName = "EvenEnd" + name
    # generate even end label
    module.addCode(self.getLabelDef(evenEndLabelName))
    return module

  ##############################################################################
  # generateOddEndVgprCopyForDTV
  # generate odd end vgpr copy for DirectToVgpr
  ##############################################################################
  def generateOddEndVgprCopyForDTV(self, kernel):
    module = Code.Module("generateOddEndVgprCopyForDTV")
    vregNameBase = "G2LA" if kernel["DirectToVgprA"] else "G2LB"
    numVreg = self.numVgprG2LA//2 if kernel["DirectToVgprA"] else self.numVgprG2LB//2
    vregSet0 = vregNameBase + "0+"
    vregSet1 = vregNameBase + "1+"
    self.comment("copy Vreg set1 to Vreg set0 for DirectToVgpr + PrefetchAcrossPersistent")
    for index in range(numVreg):
      module.addInst("v_mov_b32", vgpr(vregSet0+str(index)), vgpr(vregSet1+str(index)), "")
    return module

  ##############################################################################
  # WaitCnt- DONE
  # 3 components can contribute to the waitcnt:
  #   - Pending global reads.  (skipGlobalRead)
  #   - Pending local write.  (skipLocalWrite)
  #   - Pending local reads (skipLocalRead)
  # If a skip* arg is -1, the associated component does not contribute to
  # the expected lgkmcnt or vmcnt
  ##############################################################################
  def wait(self, kernel, tPA, tPB, skipGlobalRead, skipLocalWrite, \
      skipLocalRead, comment):
    if not self.do["Wait"]: return Code.Module("noWait")
    # skip = -1 -> ignore
    # skip =  n -> waitcnt(n*num)

    lgkmcnt = 0 if skipLocalWrite > -1 or skipLocalRead > -1 else -1

    if skipLocalWrite > -1 or skipLocalRead > -1:
      if skipLocalWrite > -1:
        numA = 0 if kernel["DirectToLdsA"] \
               else tPA["nrp"]*tPA["nrc"]*max(tPA["nwcv"],tPA["nwpv"])//tPA["nwcvpi"]
        numB = 0 if kernel["DirectToLdsB"] \
               else tPB["nrp"]*tPB["nrc"]*max(tPB["nwcv"],tPB["nwpv"])//tPB["nwcvpi"]
        lgkmcnt += skipLocalWrite * (numA + numB)
      if skipLocalRead > -1:
        readsPerIter = self.numReadsPerIterA + self.numReadsPerIterB
        lgkmcnt += skipLocalRead * readsPerIter

    vmcnt = 0 if skipGlobalRead > -1 else -1
    if skipGlobalRead > -1:
      numA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"] \
          * self.numReadVectorComponentsA
      numB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"] \
          * self.numReadVectorComponentsB
      vmcnt += skipGlobalRead * (numA + numB)

      # Unlike flat loads, BufferLoad do not increment the outstanding
      # lgkmcnt
      if lgkmcnt > -1 and not kernel["BufferLoad"]:
        lgkmcnt += skipGlobalRead * (numA + numB)

    if (self.db["ConservativeWaitCnt"] & 0x2) and skipGlobalRead != -1 or \
       (self.db["ConservativeWaitCnt"] & 0x4) and skipLocalWrite != -1 or \
       (self.db["ConservativeWaitCnt"] & 0x8) and skipLocalRead  != -1:
        imod = Code.Module("ConservativeWaitCnt")
        imod.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "debug %s"%comment )
        if self.archCaps["SeparateVscnt"]:
          imod.addInst("s_waitcnt_vscnt", "null", "0", "writes")
        imod.addInst("s_barrier", "debug" )
        return imod

    maxLgkmcnt = globalParameters["AsmCaps"][self.version]["MaxLgkmcnt"]
    lgkmcnt = min(lgkmcnt, maxLgkmcnt)
    if lgkmcnt >= 0 and vmcnt >= 0:
      vmcnt = -1 # preserve prior behavior of removing vmcnt here?
    maxVmcnt = globalParameters["AsmCaps"][self.version]["MaxVmcnt"]
    vmcnt = min(vmcnt, maxVmcnt)

    waitcnt = Code.WaitCnt(self.version, lgkmcnt,vmcnt,comment)
    if 0 and lgkmcnt == 0:
      imod = Code.Module("DebugWait")
      imod.addCode(waitcnt)
      imod.addCode(self.getBomb())
      return imod
    return waitcnt

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def syncThreads(self, kernel, comment=""):
    module = Code.Module("syncThreads")
    if kernel["NumThreads"] > self.kernel["WavefrontSize"] and self.do["Sync"]:
      if self.archCaps["SeparateVscnt"]:
        module.addInst("s_waitcnt_lgkmcnt", "null", "0", "extra navi wait")
      elif kernel.enabledSplitLDS or kernel["ScheduleIterAlg"] == 2 \
        or kernel["PrefetchGlobalRead"] == 2:
        module.addComment("Skip force waitcnt0")
      elif self.archCaps["Waitcnt0Disabled"]:
        module.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "force waitcnt0" )

      module.addInst(self.syncStr, comment)
    else:
      module.addComment("Skip barrier: NumThreads=%s"%(kernel["NumThreads"]) + \
              comment)
    return module

  ########################################
  # dump lds state
  ########################################
  def dumpLds(self, kernel, startU, numU):
    module = Code.Module("dumpLds")
    if globalParameters["DebugKernel"]:
      module.addComment1("dump lds state")
      module.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
      if self.archCaps["SeparateVscnt"]:
        module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
      module.addInst("s_barrier", "dump LDS" )
      tmp = self.vgprPool.checkOut(1)
      tmpAddr = self.vgprPool.checkOut(1)
      module.addInst("v_lshlrev_b32", \
          vgpr(tmpAddr), \
          hex(log2(self.bpeAB)), \
          vgpr("Serial"), \
          "dump lds")
      for i in range(startU, startU+numU):
        module.addInst("_ds_load_b32", vgpr(tmp), \
            vgpr(tmpAddr) + " offset:%u"%(i*kernel["NumThreads"]*4), "dump lds")
        module.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "dump" )
        if self.archCaps["SeparateVscnt"]:
          module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
        module.addCode(self.dump(vgpr(tmp)))
      self.vgprPool.checkIn(tmp)
      self.vgprPool.checkIn(tmpAddr)
    return module

  ########################################
  # init lds state
  ########################################
  def initLds(self, kernel, value):
    module = Code.Module("initLds")
    module.addComment1("init lds state")
    module.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
    if self.archCaps["SeparateVscnt"]:
      module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
    module.addInst("s_barrier", "init LDS" )
    tmp = self.vgprPool.checkOut(1)
    tmpAddr = self.vgprPool.checkOut(1)
    module.addInst("v_mov_b32", vgpr(tmp), hex(value), "Init value")
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    writesPerThread = ((kernel["LdsNumElements"]*numBytesPerElement-1)//kernel["NumThreads"]//4) + 1
    module.addInst("v_lshlrev_b32", \
        vgpr(tmpAddr), \
        2,
        vgpr("Serial"), \
        "set per-thread address to init LDS")
    for i in range(0, writesPerThread):
      module.addInst("_ds_store_b32", vgpr(tmpAddr), vgpr(tmp),
                     "offset:%u" % (i*kernel["NumThreads"]*4), "init lds")


    module.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "wait for LDS init to complete" )
    if self.archCaps["SeparateVscnt"]:
      module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
    module.addInst("s_barrier", "init LDS exit" )
    self.vgprPool.checkIn(tmp)
    self.vgprPool.checkIn(tmpAddr)
    return module

  def AccVgprImagNumOffset(self, kernel):
    acc2arch, _ = self.AccToArchMapper(kernel)
    return len(acc2arch) * kernel["MIRegPerOut"]

  ##############################################################################
  # AccToArchMapper
  # Provides forward (acc2arch) and backward (arch2acc) index transformation
  #  - Forward transformation is currently used for acc->vgpr copying
  #  - Backward transformation is used in ShiftVectorComponent() to map logical
  #    C-tile index back to original acc index
  ##############################################################################
  def AccToArchMapper(self, kernel):
    acc2arch = dict()
    arch2acc = dict()

    matrixInstM  = (kernel["MatrixInstM"] * kernel["MatrixInstBM"]) if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
    matrixInstN  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
    matrixInstBM = 1                                                if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
    matrixInstBN = 1                                                if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

    OutputsPerMFMA1B = matrixInstM * matrixInstN // self.kernel["WavefrontSize"]
    VectorWidth0     = kernel["VectorWidth"] if kernel["SourceSwap"] else 1
    outerTT0         = kernel["MIWaveTile"][0] // VectorWidth0
    lrvwB            = self.lrvwB if self.allowLRVWforTLUandMI else 1
    VectorWidth1     = lrvwB
    outerTT1         = kernel["MIWaveTile"][1] // VectorWidth1

    for wgIdx1 in range(0, outerTT1):
      for lb in range(0, lrvwB):
        for wgIdx0 in range(0, outerTT0):
          for bIdx1 in range(0, matrixInstBN):
            for bIdx0 in range(0, matrixInstBM):
              for tIdx in range(0, OutputsPerMFMA1B):
                for vw0 in range(0, VectorWidth0):
                  src, dst = 0, 0
                  if kernel["SourceSwap"]:
                    src = tIdx + OutputsPerMFMA1B * (bIdx0 + matrixInstBM * (bIdx1 + matrixInstBN * (vw0 + VectorWidth0 * (wgIdx0 + outerTT0 * (wgIdx1 * lrvwB + lb)))))
                    dst = vw0 + VectorWidth0 * ((bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * ((tIdx + OutputsPerMFMA1B * (bIdx1 + matrixInstBN * wgIdx1)) * lrvwB + lb))))
                  else:
                    src = tIdx + OutputsPerMFMA1B * (bIdx1 + matrixInstBN * (bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * wgIdx1)))
                    dst = tIdx + OutputsPerMFMA1B * (bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * (bIdx1 + matrixInstBN * wgIdx1)))
                  acc2arch[src] = dst
                  arch2acc[dst] = src

    return acc2arch, arch2acc

  ##############################################################################
  # MapAcctoArch
  # function to map MFMA Acc  Registers to Arch VGPR register
  # option :
  #         0 - one-to-one mapping of ACC -> VGPR  using VW
  #         1 - using ds swizzle map strided lanes output of MFMA to  coalescing
  #             lanes of v_mac
  ##############################################################################
  def MapAcctoArchRegs(self, kernel, option, isOptNLL=False):
    module = Code.Module("MapAcctoArchRegs")
    module.addComment1("Mapping of Acc register -> C Vgpr register")

    acc2arch, _ = self.AccToArchMapper(kernel)

    complexMultiplier = 2 if kernel["ProblemType"]["DataType"].isComplex() else 1
    self.codeAccVgprRead = Code.Module("AccVgprRead")
    self.codeAccVgprRead.itemList = [None] * kernel["MIRegPerOut"] * complexMultiplier * len(acc2arch)
    accImOffset = self.AccVgprImagNumOffset(kernel)
    for i in range(len(acc2arch)):
      for cm in range(complexMultiplier):
        for r in range(kernel["MIRegPerOut"]):
          destIdx = (acc2arch[i]*complexMultiplier + cm) * kernel["MIRegPerOut"] + r
          srcIdx = ((i * kernel["MIRegPerOut"] + r) + (cm*accImOffset))
          if not kernel["MIArchVgpr"]:
            accStr = accvgpr(srcIdx)
            self.codeAccVgprRead.itemList[destIdx] = Code.Inst("v_accvgpr_read_b32",
                                                            vgpr("ValuC+__placeholder__"),
                                                            accStr, "copy acc to vreg[%u]" % destIdx)
          else:
            self.codeAccVgprRead.itemList[destIdx] = Code.Inst("v_mov_b32",
                                                              vgpr("ValuC+__placeholder__"),
                                                              vgpr("ValuC+%u"%srcIdx), "copy MI out reg to vreg[%u]" % destIdx)

    return module

  ##############################################################################
  # MulMIoutAlphaToArch
  # function to handle MFMA alpha*MIout to Arch VGPR register
  ##############################################################################
  def MulMIoutAlphaToArch(self, kernel):
    module = Code.Module("MulMIoutAlphaToArch")
    module.addComment1("Multiply MI out register with Alpha -> C Vgpr register")

    acc2arch, _ = self.AccToArchMapper(kernel)

    self.codeMulAlpha = Code.Module("MulAlpha")
    self.codeMulAlpha.itemList = [None] * len(acc2arch)
    for i in range(len(acc2arch)):
      destIdx = acc2arch[i]
      srcIdx  = i * kernel["MIRegPerOut"]
      if kernel["ProblemType"]["ComputeDataType"].isDouble():
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_mul_f64", vgpr("ValuC+__placeholder__",2),
                                                       sgpr("Alpha",2),
                                                       vgpr("ValuC+%u"%srcIdx,2), "Multiply MI out reg with alpha")
      elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
          (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_mul_f32", vgpr("ValuC+__placeholder__"),
                                                       sgpr("Alpha"),
                                                       vgpr("ValuC+%u"%srcIdx), "Multiply MI out reg with alpha")
      elif kernel["ProblemType"]["ComputeDataType"].isInt32():
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_mul_lo_u32", vgpr("ValuC+__placeholder__"),
                                                       sgpr("Alpha"),
                                                       vgpr("ValuC+%u"%srcIdx), "Multiply MI out reg with alpha")
      elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
        accImOffset = self.AccVgprImagNumOffset(kernel)
        imod = Code.Module()
        # cannot use tmp vgpr for write batch, use allocated vgpr instead
        vtmp1 = self.startVgprAlphaTmp
        vtmp2 = vtmp1 + 2
        # tmp1 = a.real * b.real
        imod.addInst("v_mul_f64", vgpr(vtmp1,2), sgpr("Alpha+0",2), vgpr("ValuC+%u"%srcIdx,2), "")
        # tmp2 = a.imag * b.real
        imod.addInst("v_mul_f64", vgpr(vtmp2,2), sgpr("Alpha+2",2), vgpr("ValuC+%u"%srcIdx,2), "")
        # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
        imod.addInst("v_fma_f64", vgpr("ValuC+__placeholder__",2), sgpr("Alpha+2",2), vgpr("ValuC+%u"%(srcIdx+accImOffset),2), vgpr(vtmp1,2), "")
        # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
        imod.addInst("v_fma_f64", vgpr("ValuC+__placeholder__ +2",2), sgpr("Alpha+0",2), vgpr("ValuC+%u"%(srcIdx+accImOffset),2), vgpr(vtmp2,2), "")
        self.codeMulAlpha.itemList[destIdx] = imod

    return module

  def s_mul_u64_u32 (self, dst0, dst1,  src0, src1, comment):
    vtmp0 = self.vgprPool.checkOut(2)
    module = s_mul_int_64_32(globalParameters["AsmCaps"][self.version]["HasSMulHi"], \
                             dst0, dst1, src0, src1, False, vtmp0, comment)
    self.vgprPool.checkIn(vtmp0)
    return module

  def s_mul_i64_i32 (self, dst0, dst1,  src0, src1, comment):
    vtmp0 = self.vgprPool.checkOut(2)
    module = s_mul_int_64_32(globalParameters["AsmCaps"][self.version]["HasSMulHi"], \
                             dst0, dst1, src0, src1, True, vtmp0, comment)
    self.vgprPool.checkIn(vtmp0)
    return module

  def getBomb(self, cookie=None):
    scratchVgpr = self.vgprPool.checkOut(2)
    bomb(scratchVgpr, cookie)
    self.vgprPool.checkIn(scratchVgpr)

  def getCmpAssert(self, function, val0, val1, cookie=-1):
    scratchVgpr = self.vgprPool.checkOut(2)
    function(val0, val1, scratchVgpr, cookie)
    self.vgprPool.checkIn(scratchVgpr)

  def getMultipleB32Assert(self, sval, multiple2, cookie=-1):
    scratchVgpr = self.vgprPool.checkOut(2)
    self.asmAssert.multiple_b32(sval, multiple2, scratchVgpr, cookie)
    self.vgprPool.checkIn(scratchVgpr)

  def getVectorDiffAssert(self, v0, v1, expectedScalarDiff, cookie=-1):
    cmpvtmp = self.vgprPool.checkOut(1)
    vtmp = self.vgprPool.checkOut(2)
    self.asmAssert.assert_vector_diff(v0, v1, expectedScalarDiff, cmpvtmp, vtmp, cookie)
    self.vgprPool.checkIn(vtmp)
    self.vgprPool.checkIn(cmpvtmp)

  ########################################
  # Store to Debug Buffer
  ########################################
  def dump(self, vgprStore):
    module = Code.Module("dump vgpr[%s]"%str(vgprStore))
    if globalParameters["DebugKernel"]:
      afterDump = -1
      if self.db["DebugKernelMaxItems"] != -1:
        afterDump = self.labels.getUniqueName()
        afterDump = Code.Label(afterDump, "skip debug target")
        module.addInst("s_cmp_lt_u32", sgpr("DebugKernelItems"), 16,  "")
        module.addInst("s_cbranch_scc0", afterDump.getLabelName(), \
                       "skip if already wrote enough work-items" )
        module.addInst("s_add_u32", sgpr("DebugKernelItems"), \
                       sgpr("DebugKernelItems"), \
                       hex(1), "inc items written" )

      module.addInst("_flat_store_b32", vgpr("AddressDbg", 2), \
          vgprStore, "debug dump store" )
      module.addInst("_v_add_co_u32", vgpr("AddressDbg"), self.vcc, vgpr("AddressDbg"), \
          hex(4), "debug dump inc" )

      if self.db["DebugKernelMaxItems"] != -1:
        assert(isinstance(afterDump, Code.Label)) # Dummy guard in case someone remove the if above
        module.addCode(afterDump)

    return module

  def dumpSgpr(self, sgprStore):
    module = Code.Module("dumpSgpr %s"%sgprStore)
    if globalParameters["DebugKernel"]:
      tmp = self.vgprPool.checkOut(1,"tmp")
      module.addInst("v_mov_b32", vgpr(tmp), sgprStore, "debug dump sgpr store")
      module.addCode(self.dump(tmp))
      self.vgprPool.checkIn(vgpr(tmp))
    return module
