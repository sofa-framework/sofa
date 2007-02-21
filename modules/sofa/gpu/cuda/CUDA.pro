# Target is a library: sofagpucuda

SOFA_DIR = ../../../..
TEMPLATE = lib
include($$SOFA_DIR/sofa.cfg)

TARGET = sofagpucuda$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES

HEADERS += mycuda.h \
           CudaTypes.h \
           CudaMechanicalObject.h \
           CudaMechanicalObject.inl \
           CudaUniformMass.h \
           CudaUniformMass.inl \
           CudaFixedConstraint.h \
           CudaFixedConstraint.inl \
           CudaSpringForceField.h \
           CudaSpringForceField.inl

SOURCES += mycuda.cpp \
           CudaMechanicalObject.cpp \
           CudaUniformMass.cpp \
           CudaFixedConstraint.cpp \
           CudaSpringForceField.cpp

CUDA_SOURCES += mycuda.cu \
           CudaMechanicalObject.cu \
           CudaUniformMass.cu \
           CudaFixedConstraint.cu \
           CudaSpringForceField.cu

########################################################################
#  CUDA
########################################################################

INCLUDEPATH += $(CUDA_INC_DIR)
QMAKE_LIBDIR += $(CUDA_LIB_DIR)
LIBS += -lcudart

win32 {
  cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
  cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
}
unix {
  cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
  cuda.commands = ${CUDA_BIN_DIR}/nvcc -c ${QMAKE_CXXFLAGS} ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
  #cuda.depends = g++ -E -M -I $$CILK/include/cilk ${QMAKE_CXXFLAGS} ${QMAKE_FILE_NAME} | sed "s,^.*: ,,"
}
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_UNIX_COMPILERS += cuda

########################################################################
