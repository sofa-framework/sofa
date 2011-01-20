######  PLUGIN TARGET
TARGET = PersistentContact

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

DEFINES += SOFA_BUILD_PERSISTENTCONTACT

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS

INCLUDEPATH += $$SOFA_DIR/extlibs

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

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"


