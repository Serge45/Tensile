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

from __future__ import print_function
from .Common import globalParameters, printExit
from copy import deepcopy
import ctypes
# Global to print module names around strings
printModuleNames = 0

def slash(comment):
  """
  This comment is a single line // MYCOMMENT
  """
  return "// %s\n"%comment

def block(comment):
  """
  This comment is a single line /* MYCOMMENT  */
  """
  return "/* %s */\n"%comment

def blockNewLine(comment):
  """
  This comment is a blank line followed by /* MYCOMMENT  */
  """
  return "\n/* %s */\n"%comment

def block3Line(comment):
  kStr = "\n/******************************************/\n"
  for line in comment.split("\n"):
    kStr += "/*"
    kStr += " %-38s " % line
    kStr += "*/\n"
  kStr += "/******************************************/\n"
  return kStr

def printItemList(listOfItems, tag="__unnamed__"):
  header = "="*40
  print("%s\nbegin list %s\n%s"%(header, tag, header))
  for i, item in enumerate(listOfItems):
    item = list(item) if isinstance(item, tuple) else [item]
    print("list[%s] %s"%(i, "-"*30))
    for j, t in enumerate(item):
      ostream = t.prettyPrint()
      ostream = ostream[:-1] if len(ostream)>0 and ostream[-1:] == '\n' else ostream
      print(ostream)
  print("%s\nend list %s\n%s"%(header, tag, header))

class Item:
  """
  Base class for Modules, Instructions, etc
  Item is a atomic collection of or more instructions and commentsA
  """
  def __init__(self):
    self.parent = ""

  def toStr(self):
    return str(self)

  def countType(self,ttype):
    return int(isinstance(self, ttype))

  def prettyPrint(self, indent=""):
    ostream = ""
    ostream += "%s%s "%(indent, type(self).__name__)
    ostream += str(self)
    return ostream

class SignatureArgument(Item):
  def __init__(self, name, size, valueKind, valueType, isAddrSpaceQual = None):
    self.name      = name
    self.size      = size
    self.valueKind = valueKind
    self.valueType = valueType

    self.isAddrSpaceQual = isAddrSpaceQual

class SignatureArgumentV2(SignatureArgument):
  def __init__(self, size, align, name, valueKind, valueType, isAddrSpaceQual=None):
    super().__init__(name, size, valueKind, valueType, isAddrSpaceQual)
    self.align = align

  def __str__(self):
    signatureIndent = " " * 8
    kStr = ""
    kStr += signatureIndent[2:] + "- Name:            %s\n" % self.name
    kStr += signatureIndent + "Size:            %s\n" % self.size
    kStr += signatureIndent + "Align:          %s\n" % self.align
    kStr += signatureIndent + "ValueKind:      %s\n" % self.valueKind
    kStr += signatureIndent + "ValueType:      %s\n" % self.valueType
    if self.isAddrSpaceQual != None:
        kStr += signatureIndent + "AddrSpaceQual:   %s\n" % self.isAddrSpaceQual
    return kStr

class SignatureArgumentV3(SignatureArgument):
  def __init__(self, size, offset, name, valueKind, valueType, isAddrSpaceQual=None):
    super().__init__(name, size, valueKind, valueType, isAddrSpaceQual)
    self.offset = offset

  def __str__(self):
    signatureIndent = " " * 8
    kStr = ""
    kStr += signatureIndent[2:] + "- .name:            %s\n" % self.name
    kStr += signatureIndent + ".size:            %s\n" % self.size
    kStr += signatureIndent + ".offset:          %s\n" % self.offset
    kStr += signatureIndent + ".value_kind:      %s\n" % self.valueKind
    kStr += signatureIndent + ".value_type:      %s\n" % self.valueType
    if self.isAddrSpaceQual != None:
        kStr += signatureIndent + ".address_space:   %s\n" % self.isAddrSpaceQual
    return kStr

class SignatureKernelDescriptorV3(Item):
  def __init__(self, target, accumOffset, totalVgprs, totalSgprs, groupSegSize, hasWave32, waveFrontSize, reserved, sgprWg, vgprWi, name):
    self.name  = name  # kernel name
    self.target = target
    self.accumOffset = accumOffset
    self.totalVgprs = totalVgprs
    self.totalSgprs = totalSgprs
    self.groupSegSize = groupSegSize
    self.hasWave32 = hasWave32
    self.waveFrontSize = waveFrontSize
    self.sgprWg = sgprWg
    self.vgprWi = vgprWi

  def __str__(self):
    kdIndent = " " * 2
    kStr = ""
    kStr += ".amdgcn_target \"amdgcn-amd-amdhsa--%s\"\n" % self.target
    kStr += ".text\n"
    kStr += ".protected %s\n" % self.name
    kStr += ".globl %s\n" % self.name
    kStr += ".p2align 8\n"
    kStr += ".type %s,@function\n" % self.name
    kStr += ".section .rodata,#alloc\n"
    kStr += ".p2align 6\n"
    kStr += ".amdhsa_kernel %s\n" % self.name
    kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_segment_ptr 1\n"
    if self.accumOffset != None:
      kStr += kdIndent + ".amdhsa_accum_offset %u // accvgpr offset\n" % self.accumOffset
    kStr += kdIndent + ".amdhsa_next_free_vgpr %u // vgprs\n" % self.totalVgprs
    kStr += kdIndent + ".amdhsa_next_free_sgpr %u // sgprs\n" % self.totalSgprs
    kStr += kdIndent + ".amdhsa_group_segment_fixed_size %u // lds bytes\n" % self.groupSegSize
    if self.hasWave32:
      if self.waveFrontSize == 32:
        kStr += kdIndent + ".amdhsa_wavefront_size32 1 // 32-thread wavefronts\n"
      else:
        kStr += kdIndent + ".amdhsa_wavefront_size32 0 // 64-thread wavefronts\n"
    kStr += kdIndent + ".amdhsa_private_segment_fixed_size 0\n"
    kStr += kdIndent + ".amdhsa_system_sgpr_workgroup_id_x %u\n" % self.sgprWg[0]
    kStr += kdIndent + ".amdhsa_system_sgpr_workgroup_id_y %u\n" % self.sgprWg[1]
    kStr += kdIndent + ".amdhsa_system_sgpr_workgroup_id_z %u\n" % self.sgprWg[2]
    kStr += kdIndent + ".amdhsa_system_vgpr_workitem_id %u\n" % self.vgprWi
    kStr += kdIndent + ".amdhsa_float_denorm_mode_32 3\n"
    kStr += kdIndent + ".amdhsa_float_denorm_mode_16_64 3\n"
    kStr += ".end_amdhsa_kernel\n"
    kStr += ".text\n"
    return kStr

  def prettyPrint(self, indent=""):
    ostream = ""
    ostream += "%s%s "%(indent, type(self).__name__)
    return ostream

class SignatureCodeMetaV3(Item):
  def __init__(self, groupSegSize, totalVgprs, totalSgprs, flatWgSize, waveFrontSize, name):
    self.name = name
    self.groupSegSize = groupSegSize
    self.totalVgprs = totalVgprs
    self.totalSgprs = totalSgprs
    self.flatWgSize = flatWgSize
    self.waveFrontSize = waveFrontSize
    self.offset = 0
    self.argList = []

  def __str__(self):
    kStr = ""
    kStr += ".amdgpu_metadata\n"
    kStr += "---\n"
    kStr += "amdhsa.version:\n"
    kStr += "  - 1\n"
    kStr += "  - 0\n"
    kStr += "amdhsa.kernels:\n"
    kStr += "  - .name: %s\n" % self.name
    kStr += "    .symbol: '%s.kd'\n" % self.name
    kStr += "    .language:                   OpenCL C\n"
    kStr += "    .language_version:\n"
    kStr += "      - 2\n"
    kStr += "      - 0\n"
    kStr += "    .args:\n"
    for i in self.argList:
      kStr += str(i)
    kStr += "    .group_segment_fixed_size:   %u\n" % self.groupSegSize
    kStr += "    .kernarg_segment_align:      %u\n" % 8
    kStr += "    .kernarg_segment_size:       %u\n" % (((self.offset+7)//8)*8) # round up to .kernarg_segment_align
    kStr += "    .max_flat_workgroup_size:    %u\n" % self.flatWgSize
    kStr += "    .private_segment_fixed_size: %u\n" % 0
    kStr += "    .sgpr_count:                 %u\n" % self.totalSgprs
    kStr += "    .sgpr_spill_count:           %u\n" % 0
    kStr += "    .vgpr_count:                 %u\n" % self.totalVgprs
    kStr += "    .vgpr_spill_count:           %u\n" % 0
    kStr += "    .wavefront_size:             %u\n" % self.waveFrontSize

    kStr += "...\n"
    kStr += ".end_amdgpu_metadata\n"
    kStr += "%s:\n" % self.name
    return kStr

  def addArg(self, dSize, *args):
    self.argList.append(SignatureArgumentV3(dSize, self.offset, *args))
    self.offset += dSize

  def prettyPrint(self, indent=""):
    ostream = ""
    ostream += "%s%s "%(indent, type(self).__name__)
    return ostream

class Signature(Item):
  def __init__(self, codeObjectVersion, commentHeader, name=""):
    self.name    = name
    self.codeObjectVersion = codeObjectVersion
    self.commentHeader = TextBlock(slash(commentHeader))
    self.kernelDescriptor = None
    self.optCommentList = []
    self.codeMeta = None

  def addKernelDescriptor(self, kd):
    if self.codeObjectVersion == "v2":
      pass
    elif self.codeObjectVersion == "v3":
      assert(isinstance(kd, SignatureKernelDescriptorV3))
      self.kernelDescriptor = kd

  def addOptConfigComment(self, tt, sg, vw, glvwA, glvwB, d2lA, d2lB, useSgprForGRO):
    self.optCommentList.append(TextBlock(block3Line( "Optimizations and Config:" )))
    self.optCommentList.append(TextBlock(block( "ThreadTile= %u x %u" % (tt[0], tt[1]) )))
    self.optCommentList.append(TextBlock(block( "SubGroup= %u x %u" % (sg[0], sg[1]) )))
    self.optCommentList.append(TextBlock(block( "VectorWidth=%u" % vw )))
    self.optCommentList.append(TextBlock(block( "GlobalLoadVectorWidthA=%u, GlobalLoadVectorWidthB=%u" % (glvwA, glvwB) )))
    self.optCommentList.append(TextBlock(block( "DirectToLdsA=%s" % d2lA )))
    self.optCommentList.append(TextBlock(block( "DirectToLdsB=%s" % d2lB )))
    self.optCommentList.append(TextBlock(block( "UseSgprForGRO=%s" % useSgprForGRO )))

  def addCodeMeta(self, cm):
    if self.codeObjectVersion == "v2":
      pass
    elif self.codeObjectVersion == "v3":
      assert(isinstance(cm, SignatureCodeMetaV3))
      self.codeMeta = cm

  def __str__(self):
    kStr = ""
    kStr += str(self.commentHeader)
    kStr += str(self.kernelDescriptor)
    for i in self.optCommentList:
      kStr += str(i)
    kStr += str(self.codeMeta)
    return kStr

  def prettyPrint(self, indent=""):
    ostream = ""
    ostream += "%s%s "%(indent, type(self).__name__)
    return ostream

class Module(Item):
  """
  Modules contain lists of text instructions, Inst objects, or additional modules
  They can be easily converted to string that represents all items in the list
  and can be mixed with standard text.
  The intent is to allow the kernel writer to express the structure of the
  code (ie which instructions are a related module) so the scheduler can later
  make intelligent and legal transformations.
  """
  def __init__(self, name=""):
    self.name     = name
    self.itemList = []
    self.tempVgpr = None

  def findNamedItem(self, targetName):
    return next((item for item in self.itemList if item.name==targetName), None)

  def setInlineAsmPrintMode(self, mode):
    for item in self.itemList:
      if isinstance(item, Inst):
        item.setInlineAsmPrintMode(mode)
      elif isinstance(item, Module):
        item.setInlineAsmPrintMode(mode)

  def __str__(self):
    s = ""
    if printModuleNames:
      s += "// %s { \n" % self.name
    s += "".join([str(x) for x in self.itemList])
    if printModuleNames:
      s += "// } %s\n" % self.name
    return s

  def addSpaceLine(self):
    self.itemList.append(TextBlock("\n"))

  def addCode(self, item):
    """
    Add specified item to the list of items in the module.
    Item MUST be a Item (not a string) - can use
    addText(...)) to add a string.
    All additions to itemList should use this function.

    Returns item to facilitate one-line create/add patterns
    """
    #assert (isinstance(item, Item)) # for debug
    if isinstance(item,Item):
      item.parent = self
      self.itemList.append(item)
    else:
      assert 0, "unknown item type (%s) for Module.addCode. item=%s"%(type(item), item)
    return item

  def addCodeList(self, items):
    """
    Add items to module.

    Returns items to facilitate one-line create/add patterns
    """
    assert(isinstance(items, list))
    for i in items:
      self.addCode(i)
    return items

  def addModuleAsFlatItems(self, module):
    """
    Add items to module.

    Returns items to facilitate one-line create/add patterns
    """
    assert(isinstance(module, Module))
    for i in module.flatitems():
      self.addCode(i)
    return module

  def addCodeBeforeItem(self, item, newItem):
    """
    Add specified item to the list of items in the module.
    Item MUST be a Item (not a string) - can use
    addText(...)) to add a string.
    All additions to itemList should use this function.

    Returns item to facilitate one-line create/add patterns
    """
    #assert (isinstance(item, Item)) # for debug
    if isinstance(newItem,Item):
      item.parent = self
      itemIdx = -1
      for idx, selfItem in enumerate(self.itemList):
        if item == selfItem:
          itemIdx = idx
          break
      if itemIdx != -1:
        self.itemList.insert(idx, newItem)
    elif isinstance(newItem,str):
      self.addCodeBeforeItem(item, TextBlock(newItem))
    else:
      assert 0, "unknown item type (%s) for Module.addCodeByIndex. item=%s"%(type(item), item)
    return item

  def findIndex(self, targetItem):
    if isinstance(targetItem, Item):
      return self.itemList.index(targetItem)
    return -1

  def addComment(self, comment):
    """
    Convenience function to format arg as a comment and add TextBlock item
    This comment is a single line // MYCOMMENT
    """
    self.addCode(TextBlock(slash(comment)))

  def addComment0(self, comment):
    """
    Convenience function to format arg as a comment and add TextBlock item
    This comment is a single line /* MYCOMMENT  */
    """
    self.addCode(TextBlock(block(comment)))

  def addComment1(self, comment):
    """
    Convenience function to format arg as a comment and add TextBlock item
    This comment is a blank line followed by /* MYCOMMENT  */
    """
    self.addCode(TextBlock(blockNewLine(comment)))

  def addComment2(self, comment):
    self.addCode(TextBlock(block3Line(comment)))

  def addInst(self, *args):
    """
    Convenience function to construct a single Inst and add to items
    """
    self.addCode(Inst(*args))

  def prettyPrint(self,indent=""):
    ostream = ""
    ostream += '%s%s "%s"\n'%(indent, type(self).__name__, self.name)
    for i in self.itemList:
      ostream += i.prettyPrint(indent.replace("|--", "| ") + "|--")
    return ostream
    """
    Test code:
      mod1 = Code.Module("TopModule")
      mod2 = Code.Module("Module-lvl2")
      mod2.addCode(Code.Inst("bogusInst", "comments"))
      mod3 = Code.Module("Module-lvl3")
      mod3.addCode(Code.TextBlock("bogusTextBlock\nbogusTextBlock2\nbogusTextBlock3"))
      mod3.addCode(Code.GlobalReadInst("bogusGlobalReadInst", "comments"))
      mod2.addCode(mod3)
      mod1.addCode(mod2)
      mod1.addCode(Code.Inst("bogusInst", "comments"))
      mod1.addCode(mod2)

      print(mod1.prettyPrint())
    Output:
      Module "TopModule"
      |--Module "Module-lvl2"
      | |--Inst bogusInst                                          // comments
      | |--Module "Module-lvl3"
      | | |--TextBlock
      | | | |--bogusTextBlock
      | | | |--bogusTextBlock2
      | | | |--bogusTextBlock3
      | | |--GlobalReadInst bogusGlobalReadInst                                // comments
      |--Inst bogusInst                                          // comments
      |--Module "Module-lvl2"
      | |--Inst bogusInst                                          // comments
      | |--Module "Module-lvl3"
      | | |--TextBlock
      | | | |--bogusTextBlock
      | | | |--bogusTextBlock2
      | | | |--bogusTextBlock3
      | | |--GlobalReadInst bogusGlobalReadInst                                // comments
    """

  def countTypeList(self, ttypeList):
    count = 0
    # add "Module" type to type list filter, where we want to count recursively
    # the types under "Module"
    if Module not in ttypeList:
      ttypeList.append(Module)
    for ttype in ttypeList:
      count += self.countType(ttype)
    return count

  def countType(self,ttype):
    """
    Count number of items with specified type in this Module
    Will recursively count occurrences in submodules
    (Overrides Item.countType)
    """
    count=0
    for i in self.itemList:
      if isinstance(i, Module):
        count += i.countType(ttype)
      else:
        count += int(isinstance(i, ttype))
    return count

  def count(self):
    count=0
    for i in self.itemList:
      if isinstance(i, Module):
        count += i.count()
      else:
        count += 1
    return count

  def items(self):
    """
    Return list of items in the Module
    Items may be other Modules, TexBlock, or Inst
    """
    return self.itemList

  def removeItemByIndex(self, index):
    """
    Remove item from itemList, remove the last element if
    exceed length of the itemList
    Items may be other Modules, TexBlock, or Inst
    """
    if index >= len(self.itemList):
      index = -1
    del self.itemList[index]

  def removeItemsByName(self, name):
    """
    Remove items from itemList
    Items may be other Modules, TexBlock, or Inst
    """
    self.itemList = [ x for x in self.itemList if x.name != name ]

  def flatitems(self):
    """
    Return flattened list of items in the Module
    Items in sub-modules will be flattened into single list
    Items may be TexBlock or Inst
    """
    flatitems = []
    for i in self.itemList:
      if isinstance(i, Module):
        flatitems += i.flatitems()
      else:
        flatitems.append(i)
    return flatitems

  def addTempVgpr(self, vgpr):
    self.tempVgpr = vgpr

def moduleSetNoComma(module, noComma=False):
  for item in module.items():
    if isinstance(item, Module):
      moduleSetNoComma(item)
    elif isinstance(item, Inst):
      item.setNoComma(noComma)

class Macro(Item):
  def __init__(self):
    self.name = ""
    self.itemList = []
    self.macro = ""

  def __init__(self, *args):
    self.addTitle(*args)
    self.itemList = []

  def addTitle(self, *args):
    self.name = args[0]
    self.macro = Inst(*args, "")
    self.macro.setNoComma(True)

  def addCode(self, item):
    """
    Add specified item to the list of items in the Macro.
    Item MUST be a Item (not a string) - can use
    addText(...)) to add a string.
    All additions to itemList should use this function.
    Currently only accepts Inst.

    Returns item to facilitate one-line create/add patterns
    """

    if isinstance(item, Inst):
      item.parent = self
      item.setNoComma(True)
      self.itemList.append(item)
    elif isinstance(item, TextBlock):
      item.parent = self
      self.itemList.append(item)
    elif isinstance(item, Module):
      item.parent = self
      moduleSetNoComma(item, True)
      self.itemList.append(item)
    else:
      assert 0, "unknown item type (%s) for Macro.addCode. item=%s"%(type(item), item)
    return item

  def addInst(self, *args):
    """
    Convenience function to construct a single Inst and add to items
    """
    self.addCode(Inst(*args))

  def addComment0(self, comment):
    """
    Convenience function to format arg as a comment and add TextBlock item
    This comment is a single line /* MYCOMMENT  */
    """
    self.addCode(TextBlock("/* %s */\n"%comment))

  def prettyPrint(self,indent=""):
    ostream = ""
    ostream += '%s%s "%s"\n'%(indent, type(self).__name__, self.name)
    for i in self.itemList:
      ostream += i.prettyPrint(indent.replace("|--", "| ") + "|--")
    return ostream

  def items(self):
    """
    Return list of items in the Macro
    Items may be other Inst
    """
    return self.itemList

  def __str__(self):
    assert(self.macro)
    s = ""
    if printModuleNames:
      s += "// %s { \n" % self.name
    s += ".macro " + str(self.macro)
    s += "".join([("    " + str(x)) for x in self.itemList])
    s += ".endm\n"
    if printModuleNames:
      s += "// } %s\n" % self.name
    return s

class StructuredModule(Module):
  def __init__(self, name=None):
    Module.__init__(self,name)
    self.header = Module("header")
    self.middle = Module("middle")
    self.footer =  Module("footer")

    self.addCode(self.header)
    self.addCode(self.middle)
    self.addCode(self.footer)


class Label (Item):
  """
  Label that can be the target of a jump.
  """
  def __init__(self, label, comment):
    assert(isinstance(label, int) or isinstance(label, str))
    self.label = label
    self.comment = comment

  @staticmethod
  def getFormatting(label):
    if isinstance(label, int):
      return "label_%04u" % (label)
    else:
      return "label_%s" % (label)

  def getLabelName(self):
    return Label.getFormatting(self.label)

  def __str__(self):
    t = self.getLabelName() + ":"
    if self.comment:
      t += "  /// %s" % self.comment
    t += "\n"
    return t

class TextBlock(Item):
  """
  An unstructured block of text that can contain comments and instructions
  """
  def __init__(self,text):
    assert(isinstance(text, str))
    self.text = text
    self.name = text

  def __str__(self):
    return self.text

  def prettyPrint(self, indent=""):
    ostream = ""
    ostream += "%s%s "%(indent, type(self).__name__)
    l = [_i for _i in str(self).split("\n") if len(_i)>0]
    l.insert(0, "")
    ostream += "%s"%(("\n"+indent.replace("|-", "| |-")).join(l))
    ostream += "\n"
    return ostream

class Inst(Item):
  """
  Inst is a single instruction and is base class for other instructions.
  Currently just stores text+comment but over time may grow
  """
  def __init__(self, *args):
    params = args[0:len(args)-1]
    self.comment = args[len(args)-1]
    assert(isinstance(self.comment, str))
    self.inst   = params[0]
    self.name   = self.inst
    self.params = ""
    if len(params) > 1:
      self.params = list(param for param in params[1:] if param != "")
    self.outputInlineAsm = False
    self.noComma = False

  def setNoComma(self, noComma=False):
    self.noComma = noComma

  def setInlineAsmPrintMode(self, mode):
    isinstance(mode, bool)
    self.outputInlineAsm = mode

  def formatWithComment(self, formatting, comment, *args):
    instStr     = formatting %(args)
    if comment:
      return "%-50s // %s\n" % (instStr, comment)
    else:
      return "%s\n" % (instStr)

  def copy(self, inst):
    assert(isinstance(inst, Inst))
    self.name    = deepcopy(inst.name)
    self.inst    = deepcopy(inst.inst)
    self.comment = deepcopy(inst.comment)
    self.params  = deepcopy(inst.params)
    self.outputInlineAsm = inst.outputInlineAsm

  def __str__(self):
    params = [self.inst]
    params.extend(self.params)
    formatting = "\"%s" if self.outputInlineAsm else "%s"
    if len(params) > 1:
      formatting += " %s"
    for i in range(0, len(params)-2):
      if not self.noComma:
        formatting += ","
      formatting += " %s"
    if self.outputInlineAsm:
      formatting += "\\n\\t\""
    return self.formatWithComment(formatting, self.comment, *params)

class CompoundInst(Item):  # FIXME: Workaround for WaitCnt
  def __init__(self, name=""):
    self.name = name
    self.instList = []

  def addInst(self, *args):
    self.instList.append(Inst(*args))

  def addComment0(self, comment):
    """
    Convenience function to format arg as a comment and add TextBlock item
    This comment is a single line /* MYCOMMENT  */
    """
    self.instList.append(TextBlock(block(comment)))

  def prettyPrint(self,indent=""):
    ostream = ""
    ostream += '%s%s "%s"\n'%(indent, type(self).__name__, self.name)
    for i in self.instList:
      ostream += i.prettyPrint(indent.replace("|--", "| ") + "|--")
    return ostream

  def __str__(self):
    kStr = ""
    for i in self.instList:
      kStr += str(i)
    return kStr

class WaitCnt(CompoundInst):
  """
  Construct a waitcnt from specified lgkmcnt and vmcnt:
  lgkmcnt, vmcnt:
    if -1 then will not be added to the wait term.

  If lgkmcnt=vmcnt= -1 then the waitcnt is a nop and
  an instruction with a comment is returned.
  """
  def __init__(self, version,lgkmcnt=-1,vmcnt=-1,comment=""):
    super().__init__("wait")

    self.version = version
    self.lgkmcnt = lgkmcnt
    self.vmcnt   = vmcnt
    self.comment = "lgkmcnt={} vmcnt={}".format(lgkmcnt, vmcnt) + comment

  def instructions(self):
    self.instList = []
    main_args = []
    wait_store = False
    if self.lgkmcnt != -1:
      currentIsa = globalParameters["CurrentISA"]
      maxLgkmcnt = globalParameters["AsmCaps"][currentIsa]["MaxLgkmcnt"]
      main_args += ["lgkmcnt(%u)" % (min(self.lgkmcnt,maxLgkmcnt))]
      wait_store = True

    if self.vmcnt != -1:
      main_args += ["vmcnt(%u)" % self.vmcnt]

    if len(main_args) > 0:
      self.addInst("s_waitcnt", *main_args, self.comment)
      if wait_store and self.version[0] == 10 and self.vmcnt != -1:
        self.addInst("s_waitcnt_vscnt", "null", self.vmcnt, "writes")
    else:
      self.addComment0(self.comment)

  def prettyPrint(self,indent=""):
    self.instructions()
    return super().prettyPrint(indent)

  def __str__(self):
    self.instructions()
    return super().__str__()

# uniq type that can be used in Module.countType
class GlobalReadInst (Inst):
  def __init__(self,*args):
    Inst.__init__(self,*args)

# uniq type that can be used in Module.countType
class LocalWriteInst (Inst):
  def __init__(self,issuelatency,*args):
    self.IssueLatency = issuelatency
    Inst.__init__(self,*args)

# uniq type that can be used in Module.countType
class LocalReadInst (Inst):
  def __init__(self,issuelatency,readToTempVgpr,*args):
    self.IssueLatency = issuelatency
    self.readToTempVgpr = readToTempVgpr
    Inst.__init__(self,*args)

class BitfieldStructure(ctypes.Structure):
  def field_desc(self, field):
    fname = field[0]
    bits = " ({}b)".format(field[2]) if len(field) > 2 else ""
    value = getattr(self, fname)
    return "{0}{1}: {2}".format(fname, bits, value)

  def desc(self):
    return '\n'.join([self.field_desc(field) for field in self._fields_])

class BitfieldUnion(ctypes.Union):
  def __str__(self):
    return "0x{0:08x}".format(self.value)

  def desc(self):
    return "hex: {}\n".format(self) + self.fields.desc()

class SrdUpperFields9XX(BitfieldStructure):
  _fields_ = [("dst_sel_x",      ctypes.c_uint, 3),
              ("dst_sel_y",      ctypes.c_uint, 3),
              ("dst_sel_z",      ctypes.c_uint, 3),
              ("dst_sel_w",      ctypes.c_uint, 3),
              ("num_format",     ctypes.c_uint, 3),
              ("data_format",    ctypes.c_uint, 4),
              ("user_vm_enable", ctypes.c_uint, 1),
              ("user_vm_mode",   ctypes.c_uint, 1),
              ("index_stride",   ctypes.c_uint, 2),
              ("add_tid_enable", ctypes.c_uint, 1),
              ("_unusedA",       ctypes.c_uint, 3),
              ("nv",             ctypes.c_uint, 1),
              ("_unusedB",       ctypes.c_uint, 2),
              ("type",           ctypes.c_uint, 2)]

  @classmethod
  def default(cls):
    return cls(data_format = 4)

class SrdUpperValue9XX(BitfieldUnion):
  _fields_ = [("fields", SrdUpperFields9XX), ("value", ctypes.c_uint32)]

  @classmethod
  def default(cls):
    return cls(fields=SrdUpperFields9XX.default())

class SrdUpperFields10XX(BitfieldStructure):
  _fields_ = [("dst_sel_x",      ctypes.c_uint, 3),
              ("dst_sel_y",      ctypes.c_uint, 3),
              ("dst_sel_z",      ctypes.c_uint, 3),
              ("dst_sel_w",      ctypes.c_uint, 3),
              ("format",         ctypes.c_uint, 7),
              ("_unusedA",       ctypes.c_uint, 2),
              ("index_stride",   ctypes.c_uint, 2),
              ("add_tid_enable", ctypes.c_uint, 1),
              ("resource_level", ctypes.c_uint, 1),
              ("_unusedB",       ctypes.c_uint, 1),
              ("LLC_noalloc",    ctypes.c_uint, 2),
              ("oob_select",     ctypes.c_uint, 2),
              ("type",           ctypes.c_uint, 2)]


  @classmethod
  def default(cls):
    return cls(format         = 4,
               resource_level = 1,
               oob_select     = 3)


class SrdUpperValue10XX(BitfieldUnion):
  _fields_ = [("fields", SrdUpperFields10XX), ("value", ctypes.c_uint32)]

  @classmethod
  def default(cls):
    return cls(fields=SrdUpperFields10XX.default())


class SrdUpperFields11XX(BitfieldStructure):
  _fields_ = [("dst_sel_x",      ctypes.c_uint, 3),
              ("dst_sel_y",      ctypes.c_uint, 3),
              ("dst_sel_z",      ctypes.c_uint, 3),
              ("dst_sel_w",      ctypes.c_uint, 3),
              ("format",         ctypes.c_uint, 7),
              ("_unusedA",       ctypes.c_uint, 2),
              ("index_stride",   ctypes.c_uint, 2),
              ("add_tid_enable", ctypes.c_uint, 1),
              ("resource_level", ctypes.c_uint, 1),
              ("_unusedB",       ctypes.c_uint, 1),
              ("LLC_noalloc",    ctypes.c_uint, 2),
              ("oob_select",     ctypes.c_uint, 2),
              ("type",           ctypes.c_uint, 2)]


  @classmethod
  def default(cls):
    return cls(format         = 4,
               resource_level = 1,
               oob_select     = 3)


class SrdUpperValue11XX(BitfieldUnion):
  _fields_ = [("fields", SrdUpperFields11XX), ("value", ctypes.c_uint32)]

  @classmethod
  def default(cls):
    return cls(fields=SrdUpperFields11XX.default())


def SrdUpperValue(isa):
  if isa[0] == 11:
    return SrdUpperValue11XX.default()
  elif isa[0] == 10:
    return SrdUpperValue10XX.default()
  else:
    return SrdUpperValue9XX.default()
