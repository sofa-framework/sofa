# Target is a library: sofagpucuda

SOFA_DIR = ../../../..
TEMPLATE = lib
TARGET = sofagpucuda

DEFINES += SOFA_BUILD_GPU_CUDA

include($${SOFA_DIR}/sofa.cfg)

CONFIG += $$CONFIGLIBRARIES
DEFINES += CUDPP_STATIC_LIB

!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

win32 {
QMAKE_CXXFLAGS += /bigobj
}

LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += -lsofasimulation$$LIBSUFFIX
LIBS += -lsofatree$$LIBSUFFIX
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
LIBS += -lsofacomponentprojectiveconstraintset$$LIBSUFFIX
LIBS += -lsofacomponentconstraintset$$LIBSUFFIX
LIBS += -lsofacomponentcollision$$LIBSUFFIX
LIBS += -lsofacomponentmisc$$LIBSUFFIX
LIBS += -lsofacomponent$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS

HEADERS += mycuda.h \
           gpucuda.h \
           BaseCudaLDIContactResponse.h \
           CudaTypes.h \
	   CudaTypesBase.h \
           CudaCommon.h \
           CudaMath.h \
           CudaMath.inl \
           CudaMechanicalObject.h \
           CudaMechanicalObject.inl \
           CudaUniformMass.h \
           CudaUniformMass.inl \
           CudaFixedConstraint.h \
           CudaFixedConstraint.inl \
	   CudaLinearMovementConstraint.h \
	   CudaLinearMovementConstraint.inl \
	   CudaLinearForceField.h \
	   CudaLinearForceField.inl \
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
           CudaTriangleModel.h \
           CudaTriangleModel.inl \
           CudaPenalityContactForceField.h \
           CudaPenalityContactForceField.inl \
           CudaSpatialGridContainer.h \
           CudaSpatialGridContainer.inl \
           CudaVisualModel.h \
           CudaVisualModel.inl \
           CudaTetrahedralVisualModel.h \
           CudaTetrahedralVisualModel.inl \
           CudaParticleSource.h \
           CudaParticleSource.inl \
           VolumetricContact.h \
           VolumetricContact.inl \
	   CudaMemoryManager.h 

SOURCES += mycuda.cpp \
           CudaBoxROI.cpp  \
	   CudaSphereROI.cpp  \
	   CudaBeamLinearMapping.cpp \
	   CudaRestShapeSpringsForceField.cpp  \
	   CudaIndexValueMapper.cpp \
           CudaMechanicalObject.cpp \
           CudaUniformMass.cpp \
	   CudaExtraMonitor.cpp \
           CudaFixedConstraint.cpp \
           CudaFixedTranslationConstraint.cpp \
           CudaLinearMovementConstraint.cpp \
           CudaLinearVelocityConstraint.cpp \
	   CudaLinearForceField.cpp \
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
           CudaTriangleModel.cpp \
           CudaPenalityContactForceField.cpp \
           CudaVisualModel.cpp \
           CudaTetrahedralVisualModel.cpp \
           CudaSetTopology.cpp \
           CudaParticleSource.cpp \
           VolumetricContact.cpp 

CUDA_SOURCES += mycuda.cu \
           CudaMechanicalObject.cu \
           CudaUniformMass.cu \
	   CudaTypesBase.cu \
           CudaFixedConstraint.cu \
           CudaLinearMovementConstraint.cu \
	   CudaLinearForceField.cu \
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
           CudaFixedPlaneConstraint.h \
           CudaFixedPlaneConstraint.inl \
	   	CudaLCP.h \
			CudaMathRigid.h \
			CudaMathRigid.inl \
	   	CudaMatrixUtils.h \
	   	CudaDiagonalMatrix.h \
        CudaRotationMatrix.h \
       	#CudaMasterContactSolver.h \
	   	CudaBTDLinearSolver.h \
	   	CudaUnilateralInteractionConstraint.h \
		CudaBlockJacobiPreconditioner.h \
	   	CudaPrecomputedWarpPreconditioner.h \
		CudaPrecomputedWarpPreconditioner.inl \
	        CudaParallelMatrixLinearSolver.h \
		CudaWarpPreconditioner.h \
		CudaConstantForceField.inl \
		CudaJacobiPreconditioner.h \
        CudaTetrahedronTLEDForceField.h \
       	CudaHexahedronFEMForceField.h \
       	CudaHexahedronFEMForceField.inl \
		#CudaHexahedronGeodesicalDistance.h \
		#CudaHexahedronGeodesicalDistance.inl \
       	CudaHexahedronTLEDForceField.h \
               CudaJointSpringForceField.h \
               CudaJointSpringForceField.inl \
		CudaTetrahedronSuperTLEDForceField.h \
       	CudaUncoupledConstraintCorrection.h \
		CudaRasterizer.h \
		CudaRasterizer.inl \
		CudaLDIPenalityContactForceField.h \
		CudaLDISimpleContactConstraint.h \
		CudaComplianceMatrixUpdateManager.h \
		CudaDiagonalMass.h \
		CudaDiagonalMass.inl \
           PairwiseCudaRasterizer.h \
           PairwiseCudaRasterizer.inl \
           ProximityRasterizer.h \
           ProximityRasterizer.inl 
		
		
SOURCES += \
        CudaBTDLinearSolver.cpp  \
           CudaFixedPlaneConstraint.cpp \
	   	CudaLinearSolverConstraintCorrection.cpp \
	   	CudaRotationFinder.cpp \
	   	CudaLCP.cpp \
       	#CudaMasterContactSolver.cpp \
       	CudaSpatialGridContainer.cpp \
	   	CudaUnilateralInteractionConstraint.cpp \
	   	CudaPrecomputedWarpPreconditioner.cpp \
		CudaWarpPreconditioner.cpp \
		CudaConstantForceField.cpp \
		CudaAspirationForceField.cpp \
		CudaJacobiPreconditioner.cpp \
		CudaBlockJacobiPreconditioner.cpp \
		CudaHexahedronFEMForceField.cpp \
		#CudaHexahedronGeodesicalDistance.cpp \
     	CudaTetrahedronTLEDForceField.cpp \
       	CudaHexahedronTLEDForceField.cpp \
               CudaJointSpringForceField.cpp \
		CudaPairInteractionForceField.cpp \
		CudaTetrahedronSuperTLEDForceField.cpp \
       	CudaUncoupledConstraintCorrection.cpp \
		CudaRasterizer.cpp \
		CudaLDIPenalityContactForceField.cpp \
		CudaLDISimpleContactConstraint.cpp \
	CudaComplianceMatrixUpdateManager.cpp \
	CudaTetrahedronCuttingManager.cpp \
	CudaDiagonalMass.cpp \
	CudaBilateralInteractionConstraint.cpp \
           PairwiseCudaRasterizer.cpp \
           ProximityRasterizer.cpp

CUDA_SOURCES += \
           CudaFixedPlaneConstraint.cu \
	   	CudaLCP.cu \
	   	CudaMatrixUtils.cu \
       	CudaSpatialGridContainer.cu \
        CudaHexahedronFEMForceField.cu \
	CudaLinearSolverConstraintCorrection.cu \
		#CudaHexahedronGeodesicalDistance.cu \
               CudaJointSpringForceField.cu \
       	CudaTetrahedronTLEDForceField.cu \
       	CudaHexahedronTLEDForceField.cu \
		CudaTetrahedronSuperTLEDForceField.cu \
		CudaRasterizer.cu \
		CudaLDIPenalityContactForceField.cu \
	CudaComplianceMatrixUpdateManager.cu \
	CudaDiagonalMass.cu 
		

contains(DEFINES,SOFA_HAVE_CSPARSE){
	HEADERS += \
		  CudaPrecomputedLinearSolver.h	\
		  CudaSparseLDLSolver.h \
		  CudaSparseLDLSolver.inl \

	SOURCES += \	
		  CudaPrecomputedLinearSolver.cpp \
		  CudaSparseLDLSolver.cpp

	CUDA_SOURCES += \
		CudaSparseLDLSolver.cu
}

contains(DEFINES,SOFA_HAVE_TAUCS_MT){ 
	HEADERS += \
		CudaSparseTaucsLLtSolver.h \
		CudaSparseTaucsLLtSolver.inl

	SOURCES += \	
		CudaSparseTaucsLLtSolver.cpp

	CUDA_SOURCES += \
		CudaSparseTaucsLLtSolver.cu
}

contains(DEFINES,SOFA_HAVE_EIGEN2){
HEADERS += \
                CudaLDIContactLMConstraint.h \
                VolumetricFrameContact.h \
                VolumetricFrameContact.inl
           
	   	
SOURCES += \	
                CudaLDIContactLMConstraint.cpp \
                VolumetricFrameContact.cpp
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

!contains(DEFINES,SOFA_GPU_CUDPP){
	HEADERS += radixsort.cuh radixsort_kernel.cu
	CUDA_SOURCES += radixsort.cu
}


} # END SOFA_DEV
