HEADERS +=     OpenCLPlaneForceField.h \
	OpenCLPlaneForceField.inl

SOURCES +=  OpenCLPlaneForceField.cpp

OTHER_FILES += kernels/OpenCLGenericParticleForceField.cl \
		kernels/OpenCLGenericParticleForceField_Plane.macrocl

#obsolete file
#OTHER_FILES += kernels/OpenCLPlaneForceField.cl \
