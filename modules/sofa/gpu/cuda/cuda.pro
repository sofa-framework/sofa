# Target is a library: sofagpucuda

load(sofa/pre)

TEMPLATE = lib
TARGET = sofagpucuda
INCLUDEPATH += /usr/local/cuda/include/
DEPENDPATH += /usr/local/cuda/include/

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
           CudaTypes.h \
	   CudaTypesBase.h \
           CudaCommon.h \
           CudaMath.h \
           CudaMath.inl \
	   CudaMathRigid.h \
	   CudaMathRigid.inl \
       CudaScan.h \
       CudaSort.h \
           CudaMechanicalObject.h \
           CudaMechanicalObject.inl \
           CudaUniformMass.h \
           CudaUniformMass.inl \
           CudaDiagonalMass.h \
           CudaDiagonalMass.inl \
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
           CudaBarycentricMappingRigid.h \
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
	   CudaMemoryManager.h 

SOURCES += mycuda.cpp \
           CudaBoxROI.cpp  \
	   CudaSphereROI.cpp  \
	   CudaBeamLinearMapping.cpp \
	   CudaRestShapeSpringsForceField.cpp  \
	   CudaIndexValueMapper.cpp \
           CudaMechanicalObject.cpp \
           CudaUniformMass.cpp \
           CudaDiagonalMass.cpp \
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
           CudaCollision.cpp \
           CudaCollisionDistanceGrid.cpp \
           CudaDistanceGridCollisionModel.cpp \
           CudaCollisionDetection.cpp \
	   CudaSphereModel.cpp \
           CudaPointModel.cpp \
           CudaTriangleModel.cpp \
           CudaPenalityContactForceField.cpp \
           CudaVisualModel.cpp \
           CudaTetrahedralVisualModel.cpp \
           CudaSetTopology.cpp \
           CudaParticleSource.cpp

CUDA_SOURCES += mycuda.cu \
           CudaScan.cu \
           CudaSort.cu \
           CudaMechanicalObject.cu \
           CudaUniformMass.cu \
           CudaDiagonalMass.cu \
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

load(sofa/post)
