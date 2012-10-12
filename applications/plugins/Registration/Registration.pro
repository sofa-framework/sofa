load(sofa/pre)
defineAsPlugin(Registration)

TEMPLATE = lib
TARGET = sofaRegistration

DEFINES += SOFA_BUILD_REGISTRATION

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications


HEADERS += \
        initRegistration.h \
	RegistrationContact.h \
	RegistrationContact.inl \
	RegistrationContactForceField.h \
	RegistrationContactForceField.inl \
	ClosestPointRegistrationForceField.h \
	ClosestPointRegistrationForceField.inl \
	RegistrationExporter.h

SOURCES += \
	initRegistration.cpp \
	RegistrationContact.cpp \
	RegistrationContactForceField.cpp \
	ClosestPointRegistrationForceField.cpp \
	RegistrationExporter.cpp

contains(DEFINES, SOFA_HAVE_IMAGE) {

    contains(DEFINES, SOFA_IMAGE_HAVE_OPENCV) { # should be "SOFA_HAVE_OPENCV" -> use "SOFA_IMAGE_HAVE_OPENCV" until the opencv plugin is fixed..
            INCLUDEPATH += $$SOFA_OPENCV_PATH
            LIBS += -lml  -lcvaux -lhighgui -lcv -lcxcore
            }

    INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs/CImg \
                   $$SOFA_INSTALL_INC_DIR/applications/plugins/image

    HEADERS +=  IntensityProfileRegistrationForceField.h \
                IntensityProfileRegistrationForceField.inl

    SOURCES += IntensityProfileRegistrationForceField.cpp
    }

load(sofa/post)
	
