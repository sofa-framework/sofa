load(sofa/pre)
defineAsPlugin(Sensable)
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

TEMPLATE = lib

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = SensablePlugin
DEFINES += SOFA_BUILD_SENSABLEPLUGIN

SOURCES = \
initSensable.cpp \
NewOmniDriver.cpp \
OmniDriver.cpp \

HEADERS = \
NewOmniDriver.h \
OmniDriver.h

README_FILE = PluginSensable.txt

win32 {
	INCLUDEPATH += '"'$$quote($$system(echo %OH_SDK_BASE%/include))'"'
	TARGET_MACHINE = $$system(echo %TARGET_MACHINE%)
	contains(TARGET_MACHINE, x64) {
		CONFIG(debug, debug|release) : QMAKE_LIBDIR += '"'$$quote($$system(echo %OH_SDK_BASE%/lib/x64/DebugAcademicEdition))'"'
		else :                         QMAKE_LIBDIR += '"'$$quote($$system(echo %OH_SDK_BASE%/lib/x64/ReleaseAcademicEdition))'"'
	} else {
		CONFIG(debug, debug|release) : QMAKE_LIBDIR += '"'$$quote($$system(echo %OH_SDK_BASE%/lib/Win32/DebugAcademicEdition))'"'
		else :                         QMAKE_LIBDIR += '"'$$quote($$system(echo %OH_SDK_BASE%/lib/Win32/ReleaseAcademicEdition))'"'
	}
	INCLUDEPATH += '"'$$quote($$system(echo %OH_SDK_BASE%/utilities/include))'"'
	TARGET_MACHINE = $$system(echo %TARGET_MACHINE%)
	contains(TARGET_MACHINE, x64) {
		CONFIG(debug, debug|release) : QMAKE_LIBDIR += '"'$$quote($$system(echo %OH_SDK_BASE%/utilities/lib/x64/DebugAcademicEdition))'"'
		else :                         QMAKE_LIBDIR += '"'$$quote($$system(echo %OH_SDK_BASE%/utilities/lib/x64/ReleaseAcademicEdition))'"'
	} else {
		CONFIG(debug, debug|release) : QMAKE_LIBDIR += '"'$$quote($$system(echo %OH_SDK_BASE%/utilities/lib/Win32/DebugAcademicEdition))'"'
		else :                         QMAKE_LIBDIR += '"'$$quote($$system(echo %OH_SDK_BASE%/utilities/lib/Win32/ReleaseAcademicEdition))'"'
	}
}

unix {
        LIBS *= -lHD -lHL -lHDU -lHLU
}

#unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
#win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
