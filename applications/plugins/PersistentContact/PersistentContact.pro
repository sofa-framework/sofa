load(sofa/pre)
defineAsPlugin(PersistentContact)

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

TEMPLATE = lib
TARGET = PersistentContact$$LIBSUFFIX

DEFINES += SOFA_BUILD_PERSISTENTCONTACT


SOURCES = FreeMotionVelocityMasterSolver.cpp \
		  PersistentContactMapping.cpp \
		  PersistentFrictionContact.cpp \
		  PersistentUnilateralInteractionConstraint.cpp \
		  PersistentContactBarycentricMapping.cpp \
		  PersistentContactRigidMapping.cpp \
	      initPersistentContact.cpp

HEADERS = FreeMotionVelocityMasterSolver.h \
		  PersistentContact.h \
		  PersistentContactMapping.h \
		  PersistentFrictionContact.h \
		  PersistentFrictionContact.inl \
		  PersistentUnilateralInteractionConstraint.h \
		  PersistentUnilateralInteractionConstraint.inl \
		  PersistentContactBarycentricMapping.h \
		  PersistentContactBarycentricMapping.inl \
		  PersistentContactRigidMapping.h \
		  PersistentContactRigidMapping.inl
		  
README_FILE = PersistentContact.txt

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"


load(sofa/post)
