from .. import Code
from ..Common import globalParameters
from ..DataType import DataType
from ..Component import Component, GlobalWriteComponents
from ..SolutionStructs import Solution
from ..Activation import ActivationModule, ActivationType
from ..AsmStoreState import StoreState
from ..AsmUtils import vgpr, sgpr, replaceHolder, SaturateCastType
from ..AsmAddressCalculation import AddrCalculation

class GlobalWriteBatchComponent(GlobalWriteComponents):
  def __call__(self, kernel: Solution, activation: ActivationModule, ss: StoreState, batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrD, addrC, \
    tmpVgpr, bf16CVTVgprStruct, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
    parentWriter) -> Code.Module:
    return GlobalWriteBatchWriter(kernel, activation, ss, batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
      batchElements, addrD, addrC, \
      tmpVgpr, bf16CVTVgprStruct, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
      parentWriter).emit()

class GlobalWriteBatchWriter:
  def __init__(self, kernel: Solution, activation: ActivationModule, ss: StoreState, batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrD, addrC, \
    tmpVgpr, bf16CVTVgprStruct, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
      parentWriter):
    self.kernel = kernel
    self.activation = activation
    self.ss = ss
    self.batchIdx = batchIdx
    self.applyAlpha = applyAlpha
    self.beta = beta
    self.edge = edge
    self.atomic = atomic
    self.gwvw = gwvw
    self.atomicW = atomicW
    self.batchElements = batchElements
    self.addrD = addrD
    self.addrC = addrC
    self.tmpVgpr = tmpVgpr
    self.bf16CVTVgprStruct = bf16CVTVgprStruct
    self.batchElementSgprs = batchElementSgprs
    self.tmpSgpr = tmpSgpr
    self.codeAccVgprRead = codeAccVgprRead
    self.codeMulAlpha = codeMulAlpha
    self.parentWriter = parentWriter
    self.loadsIssued = 0
    self.storesIssued = 0

  @property
  def useAtomicAdd(self) -> bool:
    return self.parentWriter.asmCaps["HasAtomicAdd"] and self.kernel["ProblemType"]["ComputeDataType"].isSingle() and \
           self.kernel["_GlobalAccumulation"] == "SingleBuffer"

  @property
  def wavelen(self) -> int:
    return self.kernel["WavefrontSize"]

  @property
  def laneSGPRC(self) -> int:
    return self.parentWriter.laneSGPRCount

  @property
  def tmpS01(self):
    return self.tmpSgpr

  @property
  def tmpS23(self):
    return self.tmpS01 + self.laneSGPRC

  @property
  def debugConfig(self):
    return self.parentWriter.db

  @property
  def archCaps(self):
    return self.parentWriter.archCaps

  @property
  def computeDataType(self) -> DataType:
    return self.kernel["ProblemType"]["ComputeDataType"]

  @property
  def destDataType(self) -> DataType:
    return self.kernel["ProblemType"]["DestDataType"]

  @property
  def moduleName(self):
    return "globalWriteBatch (Atomic)" if self.atomic else "globalWriteBatch (Non atomic)"

  def emit(self) -> Code.Module:
    assert self._checkAtomicPreconditions()
    module = Code.Module(self.moduleName)
    self._prolog(module)
    self._emitAdd(module)
    self._epilog(module)
    return module

  ##############################################################################
  # choose the ADD instruction for combining external C with internal C
  # used in atomic=1 case to compute expected external data
  ##############################################################################
  def _chooseAddForAtomic(self, kernel, dst, src0, src1, comment):
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

  def _prolog(self, module: Code.Module):
    module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
              (self.ss.optSingleColVgpr, self.ss.optSharedColVgpr, self.ss.optSGPRUsage, self.ss.optSrdIncForRow))

    if self.kernel["StoreSyncOpt"]:
      self._storeSyncOpt()

    # comment tt1, tt0, vc1, vc0
    # tt = thread tile, vc=vector component
    commentStr = "Global Write%s%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
        % (" Beta" if self.beta else "", " Edge" if self.edge else "", self.batchIdx)

    commentStr = '; '.join([commentStr] \
                            + ["(%u,%u,%u,%u:vw%u%s)" % \
                               (element[0], element[1], element[2], element[3], self.gwvw,
                               ":vaw:%u"%self.atomicW if self.atomic else "") for element in self.batchElements])
    module.addComment2(commentStr)

    self.ss.setupStoreElementsForBatch(self.kernel, self.gwvw, self.batchElements, self.batchElementSgprs, isOptNLL=False, \
                                  allowLRVWforTLUandMI=self.parentWriter.allowLRVWforTLUandMI, lrvwB=self.parentWriter.lrvwB)

    self.loadsIssued = 0
    self.storesIssued = 0

    ########################################
    # calculate addr and masks
    module.addComment1("calc coords, apply mask, and issue loads (if necessary)")
    # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
    # on the thread and tid number.  These are ELEMENT offsets from start of tensor C
    # for the top-left corner this thread will write.  These are not changed
    # across all the store loop iters.
    if self.debugConfig["ConservativeWaitCnt"] & 0x10:
      module.addInst("s_barrier", "debug")
      module.addInst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
      if self.archCaps["SeparateVscnt"]:
        module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
      module.addInst("s_barrier", "debug")
    if not self.edge and self.debugConfig["ForceEdgeStores"] >= 2:
      module.addInst(self.parentWriter.getBomb()) # should not get here
    if self.edge and self.debugConfig["AssertNoEdge"]:
      module.addInst(self.parentWriter.getBomb()) # should not get here

    ########################################
    # rC *= alpha
    if not self.kernel["InterleaveAlpha"] and self.applyAlpha and self.parentWriter.alphaBeforeLoadC:
      module.addComment1("rC *= alpha batchElements=%s"%self.batchElements)
      if self.codeMulAlpha is None:
        for elementIdx in range(len(self.batchElements)):
          module.addCode(self.applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01))
      else:
          regsPerScalar = self.parentWriter.bpeCinternal // self.parentWriter.bpr # register per scalar
          for elementIdx in range(len(self.batchElements)):
            for vi in range(self.gwvw):
              module.addCode(replaceHolder(self.codeMulAlpha.items().pop(0), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi))

    loadCInputCode = Code.Module("loadCInputCode")

    for elementIdx, element in enumerate(self.batchElements):
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addrCVgpr = addrCalc.addrCVgpr
      addrDVgpr = addrCalc.addrDVgpr
      data = self.ss.elementData[elementIdx]
      mask = self.ss.elementMask[elementIdx]
      vc0 = element[3]

      module.addCode(addrCalc.emitAddressSetupCode(self.kernel, self.ss, self.tmpVgpr, self.tmpS01, self.edge, self.beta, self.atomic, elementIdx, addrDVgpr))

      if self.edge:
        module.addCode(addrCalc.edgeProtectCode(self.kernel, self.edge, self.beta, self.atomic, mask, self.tmpSgpr))

      # create code Module to push mov vgpr,acc instructions
      if self.beta:
        module.addCode(addrCalc.emitLdChange(self.kernel, self.ss, 'C', self.edge, self.beta, mask, (elementIdx == 0), self.tmpVgpr, addrCVgpr, self.addrC))
        if self.kernel["GroupLoadStore"]:
          loadCInputCode.addCode(self.parentWriter.readCInput(self.kernel, self.ss, addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01))
        else:
          module.addCode(self.parentWriter.readCInput(self.kernel, self.ss, addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01))
        self.loadsIssued += 1

      module.addCode(addrCalc.emitLdChange(self.kernel, self.ss, 'D', self.edge, self.beta, mask, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, addrDVgpr, self.addrD))

      if self.atomic and (not self.useAtomicAdd):
        # load c into data+1 because of CAS structure
        # TODO - Fix for double here, would need bigger load
        # FIXME
        # gwvw is the number of elements in the batch
        # iterate over number of atomic operations to perform, each of width atomicW
        for avi in range(self.gwvw // self.atomicW):
          dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
          bpm = self.parentWriter.bpeCexternal * self.atomicW
          useBuffer = self.kernel["BufferStore"]
          if self.kernel["BufferStore"]: # yes, BufferStore here - use same addressing regs for this load
            addr0 = vgpr(addrDVgpr)
            addr1 = sgpr("SrdD", 4)
          else:
            addr0 = vgpr(addrDVgpr, 2)
            addr1 = ""
          # Calculate vgpr Index for 32-bit/64-bit instruction
          # DGEMM use SRCS[2] register
          vgprIdx = bpm // 4
          module.addCode(self.parentWriter.chooseGlobalRead(useBuffer, bpm, dataV + vgprIdx, \
                    addr0, addr1, soffset=0, offset=addrCalc.globalOffset, extraFields="",
                    comment="load D (atomic) bpm=%u vaw=%u"%(bpm,self.atomicW)))

      if self.kernel["InterleaveAlpha"] and self.applyAlpha:
        module.addCode(self.applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01))

      if not self.kernel["BufferStore"]:
        offsetSrc = (self.tmpVgpr + 2) if self.beta else addrDVgpr

        module.addInst("_v_add_co_u32",  vgpr(addrDVgpr+0), self.parentWriter.vcc, vgpr(self.addrD+0), \
            vgpr(offsetSrc+0), "addrDVgpr = D + index*bytes (lo)" )
        module.addInst("_v_addc_co_u32", vgpr(addrDVgpr+1), self.vcc, vgpr(self.addrD+1), \
            vgpr(offsetSrc+1), self.vcc, "addrDVgpr = D + index*bytes (hi)")

        # restore full exec mask for calculating addr of next element
        if self.edge and (self.beta or self.atomic):
          module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, -1, "full mask -1 -> exec" )

    module.addCode(loadCInputCode)

    if self.beta and self.kernel["StoreSyncOpt"]:
      self._storeSyncOpt()

    ########################################
    # AccVgpr read
    if self.kernel.enabledSetPrioSplitLDS:
      module.addInst("s_setprio", "0", "")
    if self.codeAccVgprRead is not None:
      regsPerScalar = self.parentWriter.bpeCinternal // self.parentWriter.bpr # register per scalar
      # loop over store instructions within one batch
      for elementIdx in range(len(self.batchElements)):
        # loop over scalars within one store instruction
        for vi in range(self.gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            module.addCode(replaceHolder(self.codeAccVgprRead.items().pop(0), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))

      if not self.kernel["MIArchVgpr"]:
        module.addInst("s_nop", "1", "2 wait states required before reading vgpr")

    ########################################
    # rC *= alpha
    if not self.kernel["InterleaveAlpha"] and self.applyAlpha and not self.parentWriter.alphaBeforeLoadC:
      module.addComment1("rC *= alpha batchElements=%s"%self.batchElements)
      if self.codeMulAlpha is None:
        for elementIdx in range(len(self.batchElements)):
          module.addCode(self.parentWriter.applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01))
      else:
          regsPerScalar = self.parentWriter.bpeCinternal // self.parentWriter.bpr # register per scalar
          for elementIdx in range(len(self.batchElements)):
            for vi in range(self.gwvw):
              module.addCode(replaceHolder(self.codeMulAlpha.items().pop(0), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi ))

  def _epilog(self, module: Code.Module):
    # return registers to pool:
    lastData = -1
    for elementIdx in range(len(self.batchElements)):
      if not self.ss.sharedColDVgprs:
        addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
        addrDVgpr = addrCalc.addrDVgpr
        addrCVgpr = addrCalc.addrCVgpr
        self.parentWriter.vgprPool.checkIn(addrDVgpr)
        if addrCVgpr != addrDVgpr:
          self.parentWriter.vgprPool.checkIn(addrCVgpr)

      data = self.ss.elementData[elementIdx]
      if data != 0:
        if data != lastData:
          self.parentWriter.vgprPool.checkIn(data)
        lastData = data

    self.ss.firstBatch = False
    self.ss.checkInTempVgprC()
    if self.kernel["StoreRemapVectorWidth"]:
      if self.parentWriter.StoreRemapLastBatch == 1:
        module.addComment1("Handle local read and global write")
        # this seems buggy? it's possible to issue more than one stores for SR
        # module.addCode(self.storeRemapAddStore(kernel, ss, addrCalc, tmpVgpr, tmpS01, edge))
        # storesIssued += 1
        addrCalc = self.ss.elementAddr[-1]
        storeModule, numNewStores = self.parentWriter.storeRemapAddStore(self.kernel, self.ss, addrCalc, self.tmpVgpr, self.tmpS01, self.edge)
        module.addCode(storeModule)
        self.storesIssued += numNewStores

    if self.parentWriter.serializedStore:
      module.addInst("s_nop", "0", "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst")

  def _emitAdd(self, module: Code.Module):
    if self.atomic:
      del self.tmpVgpr # catch bugs
      if self.useAtomicAdd:
        self._emitAtomicAdd(module)
      else:
        self._emitCasAdd(module)
    else:
      self._emitNonatomicAdd(module)

  def _emitNonatomicAdd(self, module: Code.Module):
    ########################################
    # Not Atomic
    ########################################
    # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
    interleaveStoreVmcnt = self.parentWriter.interleaveStoreVmcnt and not self.edge

    for elementIdx in range(len(self.batchElements)):
      for vi in range(self.gwvw):
        sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
        # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
        if self.kernel["ProblemType"]["ComputeDataType"].isInt32() or \
            self.kernel["ProblemType"]["ComputeDataType"].isSingle(): # covers sgemm/gemm_ex(HHS/HSS/BBS/BSS)
            if self.debugConfig["ForceExpectedValue"]:
              module.addInst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.debugConfig["ValueCExpectedValue"], "force expected value" )
            if self.parentWriter.db["ForceVSerial"]:
              module.addInst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), vgpr("Serial"), "force expected value to serial" )
            if self.parentWriter.db["CheckValueC"]:
              module.addInst("s_mov_b32", sgpr(self.tmpS01), self.debugConfig["ValueCExpectedValue"], "Move expected value")
              module.addCode(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr("ValuC+%u"%sumIdxV), sgpr(self.tmpS01)))

    ########################################
    # wait for batched load
    if self.beta and not interleaveStoreVmcnt:
      module.addInst("s_waitcnt", "vmcnt(0)", "wait C")
      if self.archCaps["SeparateVscnt"]:
        module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

    module.addComment1("apply mask, calc new C and issue writes")
    # module.addCode(self.getBomb()) # can see store addresses just before the store inst

    # Create a suffix and check if the string exists
    activationLabelSuffix = "%s%s%u"%("Beta_" if self.beta else "", "Edge_" if self.edge else "", self.batchIdx)
    if activationLabelSuffix in self.parentWriter.globalWriteIfStateLabelSuffixDict:
      self.parentWriter.globalWriteIfStateLabelSuffixDict[activationLabelSuffix] += 1
      activationLabelSuffix = activationLabelSuffix + "_%u"%self.parentWriter.globalWriteIfStateLabelSuffixDict[activationLabelSuffix]
    else:
      self.parentWriter.globalWriteIfStateLabelSuffixDict[activationLabelSuffix] = 0
    activationCDataType = self.kernel["ProblemType"]["ComputeDataType"] if self.kernel["ProblemType"]["ActivationHPA"] else \
                                                                      self.kernel["ProblemType"]["DestDataType"]
    activationLabelEndModule = Code.Label("Activation_End_%s"%activationLabelSuffix, "")
    activationLabelModules = []
    activationEnumStrList = []
    if ((self.kernel["_GlobalAccumulation"] != 'MultipleBuffer') and self.kernel["ActivationFused"]) and \
      (self.kernel["ProblemType"]["ActivationType"] != 'none'):
      if self.kernel["ProblemType"]["ActivationType"] == 'all':
        activationEnumStrList = ActivationType.getEnumStrList(activationCDataType)
        for index, enumStr in enumerate(activationEnumStrList):
          activationLabelModule = Code.Label("Activation_%s_%s"% (enumStr.capitalize(), activationLabelSuffix), "")
          activationLabelModules.append(activationLabelModule)
        for index, activationLabelModule in enumerate(activationLabelModules):
          if index != 0:
            enumIndex = ActivationType.getEnumIndex(activationEnumStrList[index])
            module.addInst("s_cmpk_eq_u32", sgpr("ActivationType"), enumIndex, "activationType == %u"%enumIndex)
            module.addCode(Code.BranchInst("s_cbranch_scc1", activationLabelModule.getLabelName(), "Branch if true"))
      else:
        activationEnumStrList.append(str(self.kernel["ProblemType"]["ActivationType"]).lower())
    else:
      activationLabelModules.append("")
      activationEnumStrList.append("none")
    loadsIssuedRestore = self.loadsIssued
    storesIssuedRestore = self.storesIssued
    for index, activationLabelModule in enumerate(activationLabelModules):
      self.loadsIssued = loadsIssuedRestore
      self.storesIssued = storesIssuedRestore
      activationTypeStr = activationEnumStrList[index]
      if activationLabelModule:
        module.addCode(activationLabelModule)

      if self.kernel["ProblemType"]["DestDataType"].isBFloat16() and self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
        module.addInst("v_mov_b32", vgpr(self.bf16CVTVgprStruct.vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" )
        module.addInst("v_mov_b32", vgpr(self.bf16CVTVgprStruct.vgprFp32Nan), "0x7fff0000", "fp32 Nan" )
        module.addInst("v_mov_b32", vgpr(self.bf16CVTVgprStruct.vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" )

      storeCode = Code.Module("GroupLoadStore")
      for elementIdx in range(0, len(self.batchElements)):
        element = self.batchElements[elementIdx]
        addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
        addr = addrCalc.addrDVgpr
        mask = self.ss.elementMask[elementIdx]
        vc0 = element[3]
        sumIdx = self.ss.elementSumIdx[elementIdx]

        # print(str(element)+" rowInc="+str(addrCalc.rowInc))
        # Already write wave column block into LDS
        # Now read lds data back to registers and write to global memroy
        if self.ss.optSrdIncForRow and addrCalc.rowInc and self.kernel["StoreRemapVectorWidth"] > 0:
          module.addComment1("StoreRemap: shift coord1 address")
          module.addCode(addrCalc.incrementToNextRow(self.kernel, "D", self.ss, self.tmpS01))
          module.addInst("v_mov_b32", vgpr(self.tmpVgpr), addrCalc.rowInc, "set shift rows")
          module.addInst("_v_add_u32", vgpr(self.parentWriter.storeRemapCoord1), vgpr(self.parentWriter.storeRemapCoord1), vgpr(self.tmpVgpr), "shift storeRemap coord1")

        # apply in-bounds exec mask
        if self.edge and not self.kernel["BufferStore"]:
          module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, sgpr(mask, self.laneSGPRC), "sgprs -> exec" )

        if self.beta:
          # if GWVW=1 the half path still assumes we have
          # at least two stores so does some combining across VI -
          # for example assuming we can have two elements and can use pk_mul
          # here:
          if self.beta and interleaveStoreVmcnt:
            if self.parentWriter.archCaps["SeparateVscnt"]:
              vmcnt = self.loadsIssued - elementIdx - 1
              vmComment = "{} = {} - {} - 1".format(vmcnt, self.loadsIssued, elementIdx)
            else:
              waitStoreCnt = self.storesIssued if not self.kernel["GroupLoadStore"] else 0
              vmcnt = self.loadsIssued - elementIdx + waitStoreCnt - 1
              vmComment = "{} = {} - {} + {} - 1".format(vmcnt, self.loadsIssued, elementIdx, waitStoreCnt)

            maxVmcnt = globalParameters["AsmCaps"][self.parentWriter.version]["MaxVmcnt"]
            vmcnt = min(vmcnt, maxVmcnt)
            #print "wmvcnt=", vmcnt
            module.addSpaceLine()
            module.addInst("s_waitcnt", "vmcnt(%u)"%vmcnt, "wait C (interleaved) " + vmComment)

          module.addCode(self.parentWriter.addSumAlphaWithCBeta(self.kernel, self.ss, self.gwvw, elementIdx, vc0, self.tmpVgpr, self.bf16CVTVgprStruct))

        SaturateTypeInt8 = SaturateCastType.NORMAL
        # Activation
        activationModule = None
        isActivationInsertAfter = False
        if self.parentWriter.insertActivationAfterPacked(self.kernel, activationTypeStr):
          isActivationInsertAfter = True
          activationModule = Code.Module("ActivationAfterPack")
          for vi in range(0, self.gwvw):
            sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
            if self.kernel["ProblemType"]["DestDataType"].isHalf() or \
                self.kernel["ProblemType"]["DestDataType"].isBFloat16():
              if self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
                # Generate single f16 code if edge is detected.
                if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
                  self.activation.setUsePK(False)
                # Original packed route
                elif vi%2 == 1:
                  assert (self.gwvw % 2 == 0)
                else:
                  continue
                vgprIdx = self.ss.elementSumIdx[elementIdx] + vi//2
              else:
                if (sumIdxV % 2 != 0):
                  continue
                vgprIdx = sumIdxV // 2
            elif self.kernel["ProblemType"]["DestDataType"].isSingle():
              vgprIdx = sumIdxV
            elif self.kernel["ProblemType"]["DestDataType"].isDouble():
              vgprIdx = sumIdxV * 2
            elif self.kernel["ProblemType"]["DestDataType"].isInt32():
              vgprIdx = sumIdxV
            else:
              raise RuntimeError("Unsupported data type %s for activation vgpr index."%str(self.kernel["ProblemType"]["DestDataType"]))
            # Here we still use DestDataType cause the data is ready to be written to global
            actModule = self.activation.getModule(self.kernel["ProblemType"]["DestDataType"], activationTypeStr, vgprIdx)
            activationModule.addCode(self.activation.assignGpr(actModule, self.tmpVgpr, self.tmpSgpr))
            self.activation.setUsePK(True)
        else:
          activationModule = Code.Module("ActivationBeforePack")
          if self.kernel["ProblemType"]["DestDataType"].isInt8():
            if (activationTypeStr == 'abs') or (activationTypeStr == 'relu'):
              SaturateTypeInt8 = SaturateCastType.DO_NOTHING
              self.activation.setSaturationForInt8(True)
          self.activation.setVgprPrefixFormat("ValuC+%u")
          for vi in range(0, self.gwvw):
            vgprIdx = self.ss.elementSumIdx[elementIdx] + vi
            actModule = self.activation.getModule(activationCDataType, activationTypeStr, vgprIdx)
            activationModule.addCode(self.activation.assignGpr(actModule, self.tmpVgpr, self.tmpSgpr))
          self.activation.setSaturationForInt8(False)
          self.activation.setVgprPrefixFormat("")

        # pack stores, beta and non-beta reach here:
        packModule = Code.Module("Empty pack module")
        if self.kernel["ProblemType"]["HighPrecisionAccumulate"] and (self.kernel["_GlobalAccumulation"] != 'MultipleBuffer'):
          packdata = Component.find(self.parentWriter)
          if self.kernel["ProblemType"]["DestDataType"].isHalf():
            packModule = packdata(self.gwvw, self.ss.elementSumIdx[elementIdx], inputPrefix="ValuC+")
          elif self.kernel["ProblemType"]["DestDataType"].isBFloat16():
            packModule = packdata(self.gwvw, self.ss.elementSumIdx[elementIdx], bf16CVTVgprStruct=self.bf16CVTVgprStruct,
                                  tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+")
          elif self.kernel["ProblemType"]["DestDataType"].isInt8():
            packModule = packdata(self.gwvw, self.ss.elementSumIdx[elementIdx], self.tmpVgpr, self.tmpS01,
                                  SaturateTypeInt8=SaturateTypeInt8, inputPrefix="ValuC+")

        if isActivationInsertAfter:
          module.addCode(packModule)
          module.addCode(activationModule)
        else:
          module.addCode(activationModule)
          module.addCode(packModule)

        if not self.kernel["StoreRemapVectorWidth"]:
          tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, addrCalc, sumIdx, self.tmpS01, self.edge)
          if self.kernel["GroupLoadStore"]:
            storeCode.addCode(tmpStoreCode)
          else:
            module.addCode(tmpStoreCode)
          self.storesIssued += 1

        else:
          rpe = self.parentWriter.bpeCinternal // self.parentWriter.bpr
          module.addCode(self.parentWriter.storeRemapAddLocalWrite(self.kernel, self.ss, addrCalc, sumIdx*rpe))
          # Column Block Shape has been written to LDS
          # Now read back and write out to global memory

      module.addCode(storeCode)

      if self.parentWriter.db["CheckStoreC"]>=0:
        useBuffer = self.kernel["BufferStore"]
        # Note - CheckStoreC won't work for EDGE store cases since they load 0 for OOB, would need more sophisticated check
        # Note - TODO- CheckStoreC also won't work for StoreRemap
        module.addInst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        if self.parentWriter.archCaps["SeparateVscnt"]:
          module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
        for elementIdx in range(0, len(self.batchElements)):
          addr = self.ss.elementAddr[elementIdx].addrDVgpr
          sumIdx = self.ss.elementSumIdx[elementIdx]

          bps = self.kernel["ProblemType"]["DestDataType"].numBytes() * self.gwvw
          if self.kernel["BufferStore"]:
            addr0 = vgpr(addr)
            addr1 = sgpr("SrdC", 4)
          else:
            addr0 = vgpr(addr,2)
            addr1 = ""

          if self.kernel["ProblemType"]["DestDataType"].isHalf() or self.kernel["ProblemType"]["DestDataType"].isBFloat16():
            if not self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
              module.addCode(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx//2, \
                                    addr0, addr1, soffset=0, offset=0, extraFields="", hi16=sumIdx%2))
            else:
              module.addCode(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx, \
                                    addr0, addr1, soffset=0, offset=0, extraFields="", hi16=0))
          elif self.kernel["ProblemType"]["DestDataType"].isInt32() or self.kernel["ProblemType"]["DestDataType"].isSingle():
            module.addCode(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx, \
                                  addr0, addr1, soffset=0, offset=0, extraFields=""))
          elif self.kernel["ProblemType"]["DestDataType"].isDouble() or self.kernel["ProblemType"]["DestDataType"].isSingleComplex() :
            module.addCode(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx*2, \
                                  addr0, addr1, soffset=0, offset=0, extraFields=""))
          elif self.kernel["ProblemType"]["DestDataType"].isDoubleComplex():
            module.addCode(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx*4, \
                                  addr0, addr1, soffset=0, offset=0, extraFields=""))
        module.addInst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        if self.archCaps["SeparateVscnt"]:
          module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

        # Add checks for expected values:
        module.addInst("s_mov_b32", sgpr(self.tmpS01), self.parentWriter.db["CheckStoreC"], "expected value")
        for elementIdx in range(0, len(self.batchElements)):
          sumIdx = self.ss.elementSumIdx[elementIdx]
          # Need to fix for other types:
          assert (self.kernel["ProblemType"]["DestDataType"].isSingle() or self.kernel["ProblemType"]["DestDataType"].isInt32())
          module.addCode(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr(sumIdx), sgpr(self.tmpS01)))


      if self.edge and (self.atomic or not self.kernel["BufferStore"]):
        # subsequent batch must start with full exec mask
        # BufferStore doesn't need exec since it used buffer range checking when
        # possible
        module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, -1, "full mask -> exec" )

      if self.parentWriter.db["ConservativeWaitCnt"] & 0x40:
        module.addInst("s_barrier", "debug")
        module.addInst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
        if self.parentWriter.archCaps["SeparateVscnt"]:
          module.addInst("s_waitcnt_vscnt", "null", "0", "writes")
        module.addInst("s_barrier", "debug")

      if (index < (len(activationLabelModules) - 1)):
        module.addCode(Code.BranchInst("s_branch", activationLabelEndModule.getLabelName(), ""))
    if len(activationLabelModules) > 1:
      module.addCode(activationLabelEndModule)

  def _emitAtomicAdd(self, module: Code.Module):
    ########################################
    # first attempt write
    module.addComment1("issue first atomic writes")
    for elementIdx in range(len(self.batchElements)):
      addrCalc = self.ss.elementAddr[elementIdx]
      mask     = self.ss.elementMask[elementIdx]

      # apply in-bounds exec mask
      if self.edge:
        module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, sgpr(mask, self.laneSGPRC), "sgprs -> exec (before atomic)" )

      for avi in range(0, self.gwvw // self.atomicW):
        sumIdxV = self.ss.elementSumIdx[elementIdx] + avi
        if self.parentWriter.do["GlobalWrite"]:
          if self.kernel["BufferStore"]:
            module.addInst("buffer_atomic_add_f32", \
                            vgpr("ValuC+%u"%sumIdxV), \
                            vgpr(addrCalc.addrDVgpr,1), \
                            sgpr("SrdD", 4), \
                            "0", "offen", "offset:%u" % addrCalc.globalOffset, \
                            "attempt write avi=%u" % (avi))
          else:
            pass # TODO:

    if self.edge:
      module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, -1, "full mask -> exec" )

  def _emitCasAdd(self, module: Code.Module):
    # TODO for atomic GWVW:
    #  - Use vi to compute addresses, sumIdx.
    #  - Need a solution for the mask.  Can move to all buffer or can fix?
    element = self.batchElements[0]
    d1 = element[0]
    d0 = element[1]
    vc1 = element[2]
    vc0 = element[3]
    labels = self.parentWriter.labels
    labelString = "Global_Write%s%s_%u_%u_%u_%u" % ("_Beta" if self.beta else "", "_Edge" if self.edge else "", vc0, vc1, d0, d1 )
    labelComment = "Global_Write (Beta) (Edge) vc0 vc1 d0 d1"
    label = Code.Label(labels.getName(labelString), labelComment)
    labelString += "_EarlyExit"
    labelAfterAtomicLoop = Code.Label(labels.getName(labelString), labelComment)

    ########################################
    # wait for batched load
    # TODO - we are always atomic here?
    module.addInst("s_waitcnt", "vmcnt(0)", "wait C (atomic)" )
    if self.archCaps["SeparateVscnt"]:
      module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

    ########################################
    # first attempt write
    module.addComment1("issue first atomic writes")
    for elementIdx, element in enumerate(self.batchElements):
      addrCalc = self.ss.elementAddr[elementIdx]
      mask = self.ss.elementMask[elementIdx]

      # apply in-bounds exec mask
      if self.edge:
        module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, sgpr(mask, self.parentWriter.laneSGPRCount), "sgprs -> exec (before atomic)" )

      for avi in range(0, self.gwvw//self.atomicW):
        dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
        sumIdxV = self.ss.elementSumIdx[elementIdx] + avi
        ## number of src[s]/dst[s] register for DGEMM / SGEMM HGEMM
        vgprCnt = 2 if self.kernel["ProblemType"]["DestDataType"].isDouble() else 1
        if self.kernel["ProblemType"]["DestDataType"].numRegisters() < 1 and not self.kernel["_GlobalAccumulation"]:
          sumIdxV //= 2
        if self.kernel["ProblemType"]["DestDataType"].isDouble(): sumIdxV = sumIdxV * 2
        bpm = self.parentWriter.bpeCexternal * self.atomicW
        # Calculate vgpr Index for 32-bit/64-bit instruction
        # DGEMM use SRCS[2] register
        vgprIdx = 1*(bpm//4)
        # for atomic, data[1] = original c, data[0] = new c
        module.addCode(self._chooseAddForAtomic(self.kernel, \
                  vgpr(dataV+0,vgprCnt), vgpr(dataV+1*vgprIdx,vgprCnt), vgpr("ValuC+%u"%sumIdxV,vgprCnt), \
                  "desired value avi=%u"%avi))

        # attempt write
        atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
        if self.parentWriter.do["GlobalWrite"]:
          if self.kernel["BufferStore"]:
            # use cmpswap_x2 for DGEMM in CAS loop
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
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
            # Fake successful CAS swap
            module.addInst("v_mov_b32", vgpr(atomicDestVgpr), vgpr(dataV+1), "Fake successful CAS" )

    ########################################
    # wait for first attempt write
    module.addInst("s_waitcnt vmcnt(0)", "wait for atomic writes" )
    if self.parentWriter.archCaps["SeparateVscnt"]:
      module.addInst("s_waitcnt_vscnt", "null", "0", "writes")

    ########################################
    # check first attempt
    module.addComment1("check success of writes, update masks")
    for elementIdx, element in enumerate(self.batchElements):
      mask = self.ss.elementMask[elementIdx]

      # calculate new masks
      if self.edge:
        module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, sgpr(mask, self.laneSGPRC), "sgprs -> exec" )
        for avi in range(0, self.gwvw // self.atomicW):
          dataV = self.ss.elementData[elementIdx] + int(avi * self.ss.cfg.numVgprsPerDataPerVI)
          atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
          # need to apply element mask before comparison
          # so that all valid lanes are doing the cmp
          if avi == 0:
            # use u64 for DGEMM
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.addInst("v_cmp_ne_u64", sgpr(self.tmpS01, self.laneSGPRC), vgpr(atomicDestVgpr,2), \
                  vgpr(dataV+2,2), "c read during atomic == c read during prior load (avi=%u, first)"%avi )
            else:
              module.addInst("v_cmp_ne_u32", sgpr(self.tmpS01, self.laneSGPRC), vgpr(atomicDestVgpr), \
                  vgpr(dataV+1), "c read during atomic == c read during prior load (avi=%u, first)"%avi )
          else:
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.addInst("v_cmp_ne_u64", sgpr(self.tmpS23, self.laneSGPRC), vgpr(atomicDestVgpr,2), \
                  vgpr(dataV+2,2), "c read during atomic != c read during prior load" )
            else:
              module.addInst("v_cmp_ne_u32", sgpr(self.tmpS23, self.laneSGPRC), vgpr(atomicDestVgpr), \
                  vgpr(dataV+1), "c read during atomic == c read during prior load (avi=%u)"%avi )
            module.addInst("s_or_b{}".format(self.wavelen), sgpr(self.tmpS01, self.laneSGPRC), \
                  sgpr(self.tmpS01, self.laneSGPRC), sgpr(self.tmpS23, self.laneSGPRC), "combine with tmp mask")

        module.addInst("s_and_b{}".format(self.wavelen), sgpr(mask, self.laneSGPRC), sgpr(self.tmpS01, self.laneSGPRC), sgpr(mask,self.laneSGPRC), "inBounds & must try again" )

      else:
        for avi in range(0, self.gwvw//self.atomicW):
          dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
          atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
          if self.kernel["ProblemType"]["DestDataType"].isDouble():
            module.addInst("v_cmp_ne_u64", sgpr(mask, self.laneSGPRC), vgpr(atomicDestVgpr,2), \
                vgpr(dataV+2,2), "c read during atomic != c read during prior load" )
          else:
            module.addInst("v_cmp_ne_u32", sgpr(mask, self.laneSGPRC), vgpr(atomicDestVgpr), \
                vgpr(dataV+1), "c read during atomic != c read during prior load" )

    # or masks together to check early exit
    module.addComment1("or masks to check for exit")
    module.addInst("s_mov_b{}".format(self.wavelen), sgpr(self.tmpS01, self.laneSGPRC), hex(0), "empty mask" )
    for elementIdx in range(0, len(self.batchElements)):
      mask = self.ss.elementMask[elementIdx]
      module.addInst("s_or_b{}".format(self.wavelen), sgpr(self.tmpS01, self.laneSGPRC), sgpr(mask, self.laneSGPRC), sgpr(self.tmpS01, self.laneSGPRC), "or to add threads" )
    module.addInst("s_or_saveexec_b{}".format(self.wavelen), sgpr(self.tmpS23,self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), "apply combined mask" )
    module.addInst("s_cbranch_execz", labelAfterAtomicLoop.getLabelName(), "if exec is zero skip loop" )

    # begin atomic loop
    module.addComment1("atomic CAS loop")
    module.addCode(label)

    module.addComment1("apply updated masks and issue writes again")
    for elementIdx in range(0, len(self.batchElements)):
      addrCalc = self.ss.elementAddr[elementIdx]
      addr = addrCalc.addrDVgpr
      mask = self.ss.elementMask[elementIdx]
      vgprCnt = 2 if self.kernel["ProblemType"]["DestDataType"].isDouble() else 1   # number of registers for f32/f64
      bpm = self.parentWriter.bpeCexternal * self.atomicW
      vgprIdx = 1*(bpm//4)   # index register

      for avi in range(0, self.gwvw//self.atomicW):
        dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
        atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
        sumIdxV = self.ss.elementSumIdx[elementIdx] + avi
        if self.kernel["ProblemType"]["DestDataType"].numRegisters() < 1 and not self.kernel["_GlobalAccumulation"]:
          sumIdxV //= 2
        if self.kernel["ProblemType"]["DestDataType"].isDouble():
          sumIdxV =  sumIdxV * 2

        # apply mask for element
        module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, sgpr(mask,self.laneSGPRC), "must try again" )
        if self.kernel["ProblemType"]["DestDataType"].isDouble():
          #64-bit C val move by 2 32-bit instructions
          module.addInst("v_mov_b32", vgpr(dataV+2), vgpr(atomicDestVgpr), "dataV+2 = tmp (new original C)" )
          module.addInst("v_mov_b32", vgpr(dataV+3), vgpr(atomicDestVgpr+1), "dataV+3 = tmp (new original C)" )
        else:
          module.addInst("v_mov_b32", vgpr(dataV+1), vgpr(atomicDestVgpr), "dataV+1 = tmp (new original C)" )
        module.addCode(self._chooseAddForAtomic(self.kernel, \
                        vgpr(dataV+0,vgprCnt), vgpr(dataV+1*vgprIdx,vgprCnt), vgpr("ValuC+%u"%sumIdxV,vgprCnt), \
                        "newC = rC + originalC"))
        if self.parentWriter.do["GlobalWrite"]:
          if self.kernel["BufferStore"]:
            # Using no-ret version here?
            # cmpswap_x2 for DGEMM
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
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
    for elementIdx in range(0, len(self.batchElements)):
      data = self.ss.elementData[elementIdx]
      mask = self.ss.elementMask[elementIdx]
      for avi in range(0, self.gwvw//self.atomicW):
        dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
        atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2

        # apply mask for element
        module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, sgpr(mask,self.laneSGPRC), "must try again" )

        # compare success
        if self.kernel["ProblemType"]["DestDataType"].isDouble():
          module.addInst("v_cmp_ne_u64", sgpr(self.tmpS01,self.laneSGPRC), vgpr(data+2,2), vgpr(atomicDestVgpr,2), \
              "c read during atomic != c read during prior load" )
        else:
          module.addInst("v_cmp_ne_u32", sgpr(self.tmpS01,self.laneSGPRC), vgpr(data+1), vgpr(atomicDestVgpr), \
              "c read during atomic == c read during prior load" )
        # update element mask
        module.addInst("s_and_b{}".format(self.wavelen),  sgpr(mask,self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), sgpr(mask,self.laneSGPRC), "inBounds & must try again" )

    # or masks together
    module.addComment1("or masks to check for exit")
    module.addInst("s_mov_b{}".format(self.wavelen), sgpr(self.tmpS01,self.laneSGPRC), hex(0), "empty mask" )
    for elementIdx in range(0, len(self.batchElements)):
      mask = self.ss.elementMask[elementIdx]
      module.addInst("s_or_b{}".format(self.wavelen), sgpr(self.tmpS01,self.laneSGPRC), sgpr(mask,self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), "or to add threads" )

    # apply combined masks and exit
    module.addInst("s_or_saveexec_b{}".format(self.wavelen), sgpr(self.tmpS23, self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), "apply combined mask" )
    module.addInst("s_cbranch_execnz", label.getLabelName(), "try again if not complete" )
    module.addCode(labelAfterAtomicLoop)
    module.addInst("s_mov_b{}".format(self.wavelen), self.parentWriter.exec, -1, "full mask -> exec" )

  def _checkAtomicPreconditions(self) -> bool:
    if self.atomic:
      # all kinds of code relies on this assumption:
      if self.atomicW > self.gwvw:
        return False

      if (self.kernel["ProblemType"]["DataType"].isHalf() or self.kernel["ProblemType"]["DataType"].isBFloat16()) \
        and not self.kernel["_GlobalAccumulation"]:
        return self.atomicW >= 2
    return True

  def _storeSyncOpt(self, module: Code.Module):
    module.addInst("s_sleep", "%d" % (self.kernel["StoreSyncOpt"] - 1), "optimization: sync and wait")
    module.addInst("s_barrier", "")
