load(sofa/pre)
defineAsPlugin(PersistentContact)

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

TARGET = PersistentContact

DEFINES += SOFA_BUILD_PERSISTENTCONTACT


SOURCES = PersistentContactMapping.cpp \
		  PersistentFrictionContact.cpp \
		  PersistentUnilateralInteractionConstraint.cpp \
		  PersistentContactBarycentricMapping.cpp \
		  PersistentContactRigidMapping.cpp \
	      initPersistentContact.cpp

HEADERS = PersistentContact.h \
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

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR"


load(sofa/post)
