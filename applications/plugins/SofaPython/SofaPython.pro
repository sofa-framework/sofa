load(sofa/pre)
defineAsPlugin(SofaPython)

TARGET = SofaPython

DEFINES += SOFA_BUILD_SOFAPYTHON

unix:macx {
    #QMAKE_LFLAGS_SHLIB *= -Wl,--no-undefined
    #python
    LIBS += -F/opt/local/Library/Frameworks/ -framework Python
    INCLUDEPATH += /opt/local/Library/Frameworks/Python.framework/Headers

    #SofaGUI
    LIBS += -lsofaguimain
}

unix:!macx {
    #python
    INCLUDEPATH *= $$system(python-config --includes | sed -e s/\\ -I/\\ /g -e s/^-I//g)
    LIBS *= $$system(python-config --libs)
}

win32 {
	#python
    INCLUDEPATH *= "C:\Python27\include"
    LIBS *= -L"C:\Python27\libs" -lpython27
}

SOURCES = initSofaPython.cpp \
    ScriptController.cpp \
    PythonScriptController.cpp \
    PythonEnvironment.cpp \
    Binding.cpp \
    Binding_SofaModule.cpp \
    Binding_Base.cpp \
    Binding_Context.cpp \
    Binding_BaseContext.cpp \
    Binding_Node.cpp \
    Binding_BaseObjectDescription.cpp \
    Binding_Data.cpp \
    Binding_BaseObject.cpp \
    Binding_BaseState.cpp \
    PythonMacros.cpp \
    PythonVisitor.cpp \
    Binding_DisplayFlagsData.cpp \
    ScriptEvent.cpp \
    PythonScriptEvent.cpp \
    Binding_BaseLoader.cpp \
    Binding_MeshLoader.cpp \
    Binding_Vector.cpp \
    Binding_Topology.cpp \
    Binding_MechanicalObject.cpp

HEADERS = initSofaPython.h \
    ScriptController.h \
    PythonScriptController.h \
    PythonMacros.h \
    PythonEnvironment.h \
    Binding.h \
    Binding_Base.h \
    Binding_SofaModule.h \
    Binding_Node.h \
    Binding_Context.h \
    Binding_BaseContext.h \
    Binding_BaseObjectDescription.h \
    Binding_Data.h \
    Binding_BaseObject.h \
    Binding_BaseState.h \
    PythonVisitor.h \
    Binding_DisplayFlagsData.h \
    ScriptEvent.h \
    PythonScriptEvent.h \
    Binding_BaseLoader.h \
    Binding_MeshLoader.h \
    Binding_Vector.h \
    Binding_Topology.h \
    Binding_MechanicalObject.h

README_FILE = SofaPython.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
