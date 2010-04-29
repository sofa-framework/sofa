HEADERS += OpenCLSphereForceField.h \
	OpenCLSphereForceField.inl

SOURCES += OpenCLSphereForceField.cpp

OTHER_FILES += kernels/OpenCLGenericParticleForceField.cl \
		kernels/OpenCLGenericParticleForceField_Sphere.macrocl \

#obsolete
#		kernels/OpenCLSphereForceField.cl
