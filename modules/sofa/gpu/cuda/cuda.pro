# Target is a library: sofagpucuda

load(sofa/pre)

TEMPLATE = lib
TARGET = sofagpucuda
INCLUDEPATH += /usr/local/cuda/include/

DEFINES += SOFA_BUILD_GPU_CUDA

DEFINES += CUDPP_STATIC_LIB

win32 {
	CONFIG(debug, debug|release) {
		QMAKE_LFLAGS += /NODEFAULTLIB:libcmtd
	} else {
		QMAKE_LFLAGS += /NODEFAULTLIB:libcmt
	}
	QMAKE_CXXFLAGS += /bigobj
}

HEADERS += mycuda.h \
           gpucuda.h \
           BaseCudaLDIContactResponse.h \
           CudaTypes.h \
	   CudaTypesBase.h \
           CudaCommon.h \
           CudaMath.h \
           CudaMath.inl \
	   CudaMathRigid.h \
	   CudaMathRigid.inl \
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
           CudaTetrahedronTLEDForceField.h \
           CudaTetrahedronTLEDForceField.inl \
		   CudaHexahedronTLEDForceField.h \
		   CudaHexahedronTLEDForceField.inl \
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
	   CudaSimpleTesselatedTetraMechanicalMapping.cpp \
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
           CudaTetrahedronTLEDForceField.cpp \
	       CudaHexahedronTLEDForceField.cpp \
           CudaPlaneForceField.cpp \
           CudaSphereForceField.cpp \
           CudaEllipsoidForceField.cpp \
           CudaIdentityMapping.cpp \
           CudaBarycentricMapping.cpp \
           CudaBarycentricMappingRigid.cpp \
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
    	   CudaMappedBeamToTetraForceField.cpp \
	   CudaRespirationController.cpp \
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
           CudaTetrahedronTLEDForceField.cu \
	       CudaHexahedronTLEDForceField.cu \
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
       	CudaHexahedronFEMForceField.h \
       	CudaHexahedronFEMForceField.inl \
		#CudaHexahedronGeodesicalDistance.h \
		#CudaHexahedronGeodesicalDistance.inl \
               CudaJointSpringForceField.h \
               CudaJointSpringForceField.inl \
		CudaRasterizer.h \
		CudaRasterizer.inl \
		CudaLDIPenalityContactForceField.h \
		CudaLDISimpleContactConstraint.h \
		CudaComplianceMatrixUpdateManager.h \
		CudaComplianceMatrixUpdateManager.inl \
		CudaDiagonalMass.h \
		CudaDiagonalMass.inl \
           PairwiseCudaRasterizer.h \
           PairwiseCudaRasterizer.inl \
           ProximityRasterizer.h \
           ProximityRasterizer.inl \ 
                VolumetricFrameContact.h \
                VolumetricFrameContact.inl
		
		
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
               CudaJointSpringForceField.cpp \
		CudaPairInteractionForceField.cpp \
		CudaRasterizer.cpp \
		CudaLDIPenalityContactForceField.cpp \
		CudaLDISimpleContactConstraint.cpp \
	CudaComplianceMatrixUpdateManager.cpp \
	CudaTetrahedronCuttingManager.cpp \
	CudaDiagonalMass.cpp \
	CudaBilateralInteractionConstraint.cpp \
           PairwiseCudaRasterizer.cpp \
           ProximityRasterizer.cpp \
           VolumetricFrameContact.cpp

CUDA_SOURCES += \
           CudaFixedPlaneConstraint.cu \
	   	CudaLCP.cu \
	   	CudaMatrixUtils.cu \
       	CudaSpatialGridContainer.cu \
        CudaHexahedronFEMForceField.cu \
	CudaLinearSolverConstraintCorrection.cu \
		#CudaHexahedronGeodesicalDistance.cu \
               CudaJointSpringForceField.cu \
		CudaRasterizer.cu \
		CudaLDIPenalityContactForceField.cu \
	CudaComplianceMatrixUpdateManager.cu \
	CudaDiagonalMass.cu 
		

contains(DEFINES,SOFA_HAVE_CSPARSE){
	HEADERS += \
		  CudaPrecomputedLinearSolver.h	\
		  CudaPrecomputedLinearSolver.inl \
		  CudaSparseLDLSolver.h \
		  CudaSparseLDLSolver.inl \
		  CudaSparseXXTSolver.h \
		  CudaSparseXXTSolver.inl \

	SOURCES += \	
		  CudaPrecomputedLinearSolver.cpp \
		  CudaSparseLDLSolver.cpp \
		  CudaSparseXXTSolver.cpp

	CUDA_SOURCES += \
		CudaSparseLDLSolver.cu \
		CudaSparseXXTSolver.cu
}

contains(DEFINES,SOFA_EXTLIBS_TAUCS_MT){ 
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
                CudaLDIContactLMConstraint.h
           
	   	
SOURCES += \	
                CudaLDIContactLMConstraint.cpp
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
	HEADERS += radixsort.h
	CUDA_SOURCES += radixsort.cu
}


} # END SOFA_DEV

load(sofa/post)
