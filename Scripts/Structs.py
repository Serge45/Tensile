################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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

import sys

from Common import *

################################################################################
# Data Type
################################################################################
class DataType:
  single        = 0
  double        = 1
  complexSingle = 2
  complexDouble = 3
  half          = 4
  num           = 5
  none          = 6

  # data type properties
  idxChar    = 0
  idxReg     = 1
  idxOpenCL  = 2
  idxHIP     = 3
  idxLibType = 4
  idxLibEnum = 5
  #    char, reg, ocl,       hip,        libType,                libEnum
  properties = [
      [ "S", 1,   "float",   "float",   "float",                 "tensileDataTypeFloat"         ],
      [ "D", 2,   "double",  "double",  "double",                "tensileDataTypeDouble"        ],
      [ "C", 2,   "float2",  "float_2", "TensileComplexFloat",   "tensileDataTypeComplexFloat"  ],
      [ "Z", 4,   "double2", "double_2", "TensileComplexDouble", "tensileDataTypeComplexDouble" ],
      [ "H", 0.5, "ERROR",   "fp16",     "TensileHalf",          "tensileDataTypeHalf"          ]
  ]

  ########################################
  def __init__( self, value ):
    if isinstance(value, int):
      self.value = value
    if isinstance(value, str):
      for propertiesIdx in range(0,6):
        for dataTypeIdx in range(0,self.num):
          if value.lower() == self.properties[dataTypeIdx][propertiesIdx].lower():
            self.value = dataTypeIdx
            return


  ########################################
  def toChar(self):
    return self.properties[self.value][self.idxChar]
  def toOpenCL(self):
    return self.properties[self.value][self.idxOpenCL]
  def toHIP(self):
    return self.properties[self.value][self.idxOpenCL]
  def toDevice(self, backend):
    if backend.isOpenCL():
      return self.toOpenCL()
    else:
      return self.toHIP()
  def toCpp(self):
    return self.properties[self.value][self.idxLibType]
  def getLibString(self):
    return self.properties[self.value][self.idxLibEnum]

  ########################################
  def zeroString(self, backend):
    zeroString = "("
    zeroString += self.toDevice(backend)
    zeroString += ")("
    if self.isReal():
      zeroString += "0.0"
    else:
      zeroString += "0.0, 0.0"
    zeroString += ")"
    return zeroString

  ########################################
  def isReal(self):
    if self.value == self.half or self.value == self.single or self.value == self.double:
      return True
    else:
      return False
  def isComplex(self):
    return not self.isReal()
  def isDouble(self):
    if self.value == self.double or self.value == self.complexDouble:
      return True
    else:
      return False

  ########################################
  def numRegisters( self ):
    return properties[self.value][self.toLibEnum]
  def numBytes( self ):
    return self.numRegisters() * 4

  def __str__(self):
    return self.toChar()

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return (self.value)
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, DataType) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result



################################################################################
# Device
################################################################################
class Device:

  ########################################
  def __init__( self, name, numComputeUnits, clockFrequency, flopsPerClock):
    self.name = name
    self.numComputeUnits = numComputeUnits
    self.clockFrequency = clockFrequency
    self.flopsPerClock = flopsPerClock

  ########################################
  def __str__(self):
    state = "[Device"
    state += "; " + self.name
    state += "; " + str(self.numComputeUnits)
    state += "; " + str(self.clockFrequency)
    state += "; " + str(self.flopsPerClock)
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        self.name, \
        self.numComputeUnits, \
        self.clockFrequency, \
        self.flopsPerClock, \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Device) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

# ProblemSize
#  GEMM: M, N, K, [lda, ldb, ldc]
#  TensorContraction: sizeI, sizeJ, ...; [ stridesC, A, B ]


################################################################################
# ProblemType
class ProblemType:
  operationTypes = ["GEMM", "TensorContraction"]
  state = {}

  ########################################
  def __init__(self, config):
    for key in defaultProblemType:
      self.assignWithDefault(key, defaultProblemType[key], config)

    if "DataType" in config:
      self["DataType"] = DataType(config["DataType"])
    else:
      self["DataType"] = DataType(0)

    if self["OperationType"] == "GEMM":
      self.initGEMM(config)
    elif self["OperationType"] == "TensorContraction":
      self.initTensorContraction(config)


  ########################################
  def initGEMM(self, config):
    sumIdx = 3 if self["Batched"] else 2
    self["IndexAssignmentsA"] = [0, sumIdx] # N
    self["IndexAssignmentsB"] = [sumIdx, 1] # N
    if self["TransposeA"]:
      self["IndexAssignmentsA"] = [sumIdx, 0] # T
    if self["TransposeB"]:
      self["IndexAssignmentsB"] = [1, sumIdx] # T
    if self["Batched"]:
      self["IndexAssignmentsA"].append(2)
      self["IndexAssignmentsB"].append(2)
      self["NumDimensionsC"] = 3
    else:
      self["NumDimensionsC"] = 2

  ########################################
  def initTensorContraction(self, config):
    self.assign("NumDimensionsC", config)
    self.assign("IndexAssignmentsA", config)
    self.assign("IndexAssignmentsB", config)

  ########################################
  def isGEMM(self):
    return self.operationType == 0

  ########################################
  def isTensorContraction(self):
    return self.operationType == 1

  ########################################
  def __str__(self):
    # C dimensions
    name = "C"
    name += indexChars[:self["NumDimensionsC"]].lower()
    # A dimensions
    name += "_A"
    for i in self["IndexAssignmentsA"]:
      name += indexChars[i].lower()
    # B dimensions
    name += "_B"
    for i in self["IndexAssignmentsB"]:
      name += indexChars[i].lower()

    # precision and other
    name += "_"
    name += self["DataType"].toChar()
    if self["HighPrecisionAccumulate"]: name += "A"
    if self["UseBeta"]: name += "B"
    if self["UseOffsets"]: name += "O"
    if self["UseInitialStrides"]: name += "I"
    return name

  def assignWithDefault(self, parameter, default, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      self[parameter] = default
  def assign(self, parameter, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      sys.exit("Tensile::ProblemType::init ERROR - parameter \"%s\" must be defined" % parameter)
  def __getitem__(self, key):
    return self.state[key]
  def __setitem__(self, key, value):
    self.state[key] = value
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return self.state
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, ExactMatch) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# ProblemBenchmarkSizes
################################################################################
class ProblemBenchmarkSizeRange:
  dimensionSizes = []

  ########################################
  def __init__(self, problemType, config):
    totalDimensions = max(problemType[IndexAssignmentsA])
    if len(config) < self.totalDimensions:
      printExit("SizeRange config (%s) has %u < %u elements required by ProblemType (%s)"
          % ( str(config), len(config), self.totalDimensions, problemType ))
    if len(config) < self.totalDimensions:
      printWarning("SizeRange config (%s) has %u > %u elements than are required by ProblemType (%s)"
          % ( str(config), len(config), self.totalDimensions, problemType ))
    for dim in self.dimensionSizes:
      if len(dim) == 1:
        self.dimensionSizes.append([dim[0], 16, 0, dim[0]])
      elif len(dim) == 2:
        self.dimensionSizes.append([dim[0], 16, 0, dim[1]])
      elif len(dim) == 3:
        self.dimensionSizes.append([dim[0], dim[1], 0, dim[2]])
      elif len(dim) == 4:
        self.dimensionSizes.append([dim[0], dim[1], dim[2], dim[3]])
      else:
        sys.exit("Tensile::%s::%s: ERROR - ProblemBenchmarkSizeRange(%s) has %u descriptors rather than 1-4."
          % ( __file__, __line__, dim, len(dim) ))

  ########################################
  def maxNumElements(self):
    return [ 1, 1, 1 ] # TODO [maxC, maxA, maxB]

  def __str__(self):
    return str(self.dimensionSizes)




# this will have a list of index size assignments
#order of assignments: i, j, k, l, m, ...


################################################################################
# Solution
################################################################################
class Solution:
  state = {}

  ########################################
  def __init__(self, config):
    # problem type
    if "ProblemType" in config:
      self["ProblemType"] = ProblemType(config["ProblemType"])
    else:
      self["ProblemType"] = ProblemType(defaultProblemType)
      #sys.exit("Tensile::%s::%s: ERROR - No ProblemType in config: %s" % ( __file__, __line__, str(config) ))

    for key in defaultSolution:
      self.assignWithDefault(key, defaultSolution[key], config)


  ########################################
  # create a dictionary with booleans on whether to include parameter in name
  @staticmethod
  def getMinNaming(kernels):
    requiredParameters = {}
    for key in kernels[0]:
      required = False
      for i in range(1, len(kernels)):
        if kernels[0][key] != kernels[i][key]:
          required = True
          break
      if required:
        requiredParameters[key] = True
    return requiredParameters

  ########################################
  def getNameFull(self):
    requiredParameters = {}
    for key in self.state:
      requiredParameters[key] = True
    return self.getNameMin(requiredParameters)

  ########################################
  def getNameMin(self, requiredParameters):
    name = ""
    first = True
    for key in self.state:
      if requiredParameters[key]:
        if not first:
          name += "_"
        else:
          first = False
        name += self.getParameterNameAbbreviation(key)
        name += self.getParameterValueAbbreviation(self[key])
    return name

  ########################################
  @ staticmethod
  def getParameterNameAbbreviation( name ):
    return ''.join([c for c in name if c.isupper()])

  ########################################
  @ staticmethod
  def getParameterValueAbbreviation( value ):
    if isinstance(value, str):
      return ''.join([c for c in value if c.isupper()])
    elif isinstance(value, bool):
      return "1" if value else "0"
    elif isinstance(value, int):
      return str(value)
    elif isinstance(value, ProblemType):
      return str(value)
    elif isinstance(value, list):
      abbrev = ""
      for i in range(0, len(value)):
        element = value[i]
        abbrev += Solution.getParameterValueAbbreviation(element)
        if i < len(value)-1:
          abbrev += "_"
      return abbrev
    else:
      printExit("Parameter \"%s\" is new object type" % value)
      return str(value)

  def assignWithDefault(self, parameter, default, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      self[parameter] = default
  def assign(self, parameter, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      sys.exit("Tensile::Solution::init: ERROR - parameter \"%s\" must be defined" % parameter)
  def __getitem__(self, key):
    return self.state[key]
  def __setitem__(self, key, value):
    self.state[key] = value
  def __str__(self):
    return self.getNameFull()
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return state
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

