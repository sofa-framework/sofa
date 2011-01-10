######  PLUGIN TARGET
TARGET = ContinuousContact

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

DEFINES += SOFA_BUILD_CONTINUOUSCONTACT

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS

INCLUDEPATH += $$SOFA_DIR/extlibs

SOURCES = ContinuousFrictionContact.cpp \
		  ContinuousUnilateralInteractionConstraint.cpp \
		  ContinuousContactBarycentricMapping.cpp \
		  ContinuousContactRigidMapping.cpp \
	      initContinuousContact.cpp

HEADERS = ContinuousContact.h \
		  ContinuousContactMapping.h \
		  ContinuousFrictionContact.h \
		  ContinuousFrictionContact.inl \
		  ContinuousUnilateralInteractionConstraint.h \
		  ContinuousUnilateralInteractionConstraint.inl \
		  ContinuousContactBarycentricMapping.h \
		  ContinuousContactBarycentricMapping.inl \
		  ContinuousContactRigidMapping.h \
		  ContinuousContactRigidMapping.inl
		  
README_FILE = ContinuousContact.txt

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"


