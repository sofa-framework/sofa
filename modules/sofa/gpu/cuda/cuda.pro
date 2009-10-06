# Target is a library: sofagpucuda

SOFA_DIR = ../../../..
TEMPLATE = lib
DEFINES += SOFA_BUILD_GPU_CUDA
include($${SOFA_DIR}/sofa.cfg)

TARGET = sofagpucuda$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES

CONFIG -= staticlib
CONFIG += dll

LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += -lsofasimulation$$LIBSUFFIX
LIBS += -lsofatree$$LIBSUFFIX
contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV
LIBS += -lsofaautomatescheduler$$LIBSUFFIX
} # END SOFA_DEV
LIBS += -lsofacomponentbase$$LIBSUFFIX
LIBS += -lsofacomponentmastersolver$$LIBSUFFIX
LIBS += -lsofacomponentfem$$LIBSUFFIX
LIBS += -lsofacomponentinteractionforcefield$$LIBSUFFIX
LIBS += -lsofacomponentcontextobject$$LIBSUFFIX
LIBS += -lsofacomponentbehaviormodel$$LIBSUFFIX
LIBS += -lsofacomponentlinearsolver$$LIBSUFFIX
LIBS += -lsofacomponentodesolver$$LIBSUFFIX
LIBS += -lsofacomponentbase$$LIBSUFFIX
LIBS += -lsofacomponentcontroller$$LIBSUFFIX
LIBS += -lsofacomponentengine$$LIBSUFFIX
LIBS += -lsofacomponentvisualmodel$$LIBSUFFIX
LIBS += -lsofacomponentmass$$LIBSUFFIX
LIBS += -lsofacomponentforcefield$$LIBSUFFIX
LIBS += -lsofacomponentmapping$$LIBSUFFIX
LIBS += -lsofacomponentconstraint$$LIBSUFFIX
LIBS += -lsofacomponentcollision$$LIBSUFFIX
LIBS += -lsofacomponentmisc$$LIBSUFFIX
LIBS += -lsofacomponent$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS

HEADERS += mycuda.h \
           gpucuda.h \
           CudaTypes.h \
	   CudaTypesBase.h \
           CudaCommon.h \
           CudaMath.h \
           CudaMechanicalObject.h \
           CudaMechanicalObject.inl \
           CudaUniformMass.h \
           CudaUniformMass.inl \
           CudaFixedConstraint.h \
           CudaFixedConstraint.inl \
           CudaSpringForceField.h \
           CudaSpringForceField.inl \
           CudaTetrahedronFEMForceField.h \
           CudaTetrahedronFEMForceField.inl \
           CudaPlaneForceField.h \
           CudaPlaneForceField.inl \
           CudaSphereForceField.h \
           CudaSphereForceField.inl \
           CudaEllipsoidForceField.h \
           CudaEllipsoidForceField.inl \
           CudaIdentityMapping.h \
           CudaIdentityMapping.inl \
           CudaBarycentricMapping.h \
           CudaBarycentricMapping.inl \
           CudaRigidMapping.h \
           CudaRigidMapping.inl \
           CudaSubsetMapping.h \
           CudaSubsetMapping.inl \
           CudaDistanceGridCollisionModel.h \
           CudaContactMapper.h \
           CudaCollisionDetection.h \
           CudaPointModel.h \
           CudaSphereModel.h \
           CudaPenalityContactForceField.h \
           CudaPenalityContactForceField.inl \
           CudaSpatialGridContainer.h \
           CudaSpatialGridContainer.inl \
           CudaVisualModel.h \
           CudaVisualModel.inl \
           CudaTetrahedralVisualModel.h \
           CudaTetrahedralVisualModel.inl \
           CudaParticleSource.h \
           CudaParticleSource.inl

SOURCES += mycuda.cpp \
           CudaBoxROI.cpp  \
           CudaMechanicalObject.cpp \
           CudaUniformMass.cpp \
           CudaFixedConstraint.cpp \
           CudaSpringForceField.cpp \
           CudaTetrahedronFEMForceField.cpp \
           CudaPlaneForceField.cpp \
           CudaSphereForceField.cpp \
           CudaEllipsoidForceField.cpp \
           CudaIdentityMapping.cpp \
           CudaBarycentricMapping.cpp \
           CudaRigidMapping.cpp \
           CudaSubsetMapping.cpp \
           CudaDistanceGridCollisionModel.cpp \
           CudaCollision.cpp \
           CudaCollisionDetection.cpp \
		   CudaSphereModel.cpp \
           CudaPointModel.cpp \
           CudaPenalityContactForceField.cpp \
           CudaVisualModel.cpp \
           CudaTetrahedralVisualModel.cpp \
           CudaSetTopology.cpp \
           CudaParticleSource.cpp

CUDA_SOURCES += mycuda.cu \
           CudaMechanicalObject.cu \
           CudaUniformMass.cu \
           CudaFixedConstraint.cu \
           CudaSpringForceField.cu \
           CudaTetrahedronFEMForceField.cu \
           CudaPlaneForceField.cu \
           CudaSphereForceField.cu \
           CudaEllipsoidForceField.cu \
           CudaBarycentricMapping.cu \
           CudaRigidMapping.cu \
           CudaSubsetMapping.cu \
           CudaCollisionDetection.cu \
           CudaContactMapper.cu \
           CudaPenalityContactForceField.cu \
           CudaVisualModel.cu \
           CudaParticleSource.cu

contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV

HEADERS += \
	   	CudaLCP.h \
	   	CudaMatrixUtils.h \
       	CudaMasterContactSolver.h \
	   	CudaBTDLinearSolver.h \
	   	CudaUnilateralInteractionConstraint.h \
	   	CudaPrecomputedWarpPreconditioner.h \
        CudaTetrahedronTLEDForceField.h \
       	CudaHexahedronFEMForceField.h \
       	CudaHexahedronFEMForceField.inl \
       	CudaHexahedronTLEDForceField.h \
		CudaTetrahedronSuperTLEDForceField.h \
       	CudaUncoupledConstraintCorrection.h \
		CudaRasterizer.h \
		CudaLDIPenalityContactForceField.h
		
SOURCES += \
        CudaBTDLinearSolver.cpp  \
	   	CudaLinearSolverConstraintCorrection.cpp \
	   	CudaRotationFinder.cpp \
	   	CudaLCP.cpp \
       	CudaMasterContactSolver.cpp \
       	CudaSpatialGridContainer.cpp \
	   	CudaUnilateralInteractionConstraint.cpp \
	   	CudaPrecomputedWarpPreconditioner.cpp \  	
       	CudaHexahedronFEMForceField.cpp \
     	CudaTetrahedronTLEDForceField.cpp \
       	CudaHexahedronTLEDForceField.cpp \
		CudaTetrahedronSuperTLEDForceField.cpp \
       	CudaUncoupledConstraintCorrection.cpp \
       	CudaRasterizer.cpp \
		CudaLDIPenalityContactForceField.cpp

CUDA_SOURCES += \
	   	CudaLCP.cu \
	   	CudaMatrixUtils.cu \
       	CudaSpatialGridContainer.cu \
        CudaHexahedronFEMForceField.cu \
       	CudaTetrahedronTLEDForceField.cu \
       	CudaHexahedronTLEDForceField.cu \
		CudaTetrahedronSuperTLEDForceField.cu \
		CudaRasterizer.cu \
		CudaLDIPenalityContactForceField.cu
		
contains(DEFINES,SOFA_HAVE_BOOST){
contains(DEFINES,SOFA_HAVE_CSPARSE){
HEADERS += \
	   	CudaUpdatePrecomputedPreconditioner.h
	   	
SOURCES += \	
		CudaUpdatePrecomputedPreconditioner.cpp
}
}

HEADERS += \
	   	CudaParticlesRepulsionForceField.h \
	   	CudaParticlesRepulsionForceField.inl
SOURCES += CudaParticlesRepulsionForceField.cpp
CUDA_SOURCES += CudaParticlesRepulsionForceField.cu

HEADERS += \
	   	CudaSPHFluidForceField.h \
	   	CudaSPHFluidForceField.inl
SOURCES += CudaSPHFluidForceField.cpp
CUDA_SOURCES += CudaSPHFluidForceField.cu

HEADERS += scan.h
CUDA_SOURCES += scan.cu

#HEADERS += radixsort.cuh radixsort_kernel.cu
#CUDA_SOURCES += radixsort.cu

HEADERS += radixsort.h
CUDA_SOURCES += radixsort.cu

} # END SOFA_DEV
