load(sofa/pre)
defineAsPlugin(Python)

TARGET = Python

DEFINES += SOFA_BUILD_PYTHON

unix:macx {
    #QMAKE_LFLAGS_SHLIB *= -Wl,--no-undefined
    #python
    LIBS += -F/opt/local/Library/Frameworks/ -framework Python
    INCLUDEPATH += /opt/local/Library/Frameworks/Python.framework/Headers

    #boost.python
    INCLUDEPATH += /opt/local/include
    LIBS += -L/opt/local/lib/ -lboost_python$$BOOST_SUFFIX
}

unix:!macx {
    #python
    INCLUDEPATH *= $$system(python-config --includes | sed -e s/\\ -I/\\ /g -e s/^-I//g)
    LIBS *= $$system(python-config --libs)

    #boost.python
    LIBS *= -lboost_python$$BOOST_SUFFIX
}

SOURCES = initPython.cpp \
    ScriptController.cpp \
    PythonScriptController.cpp \
    PythonEnvironment.cpp \
    PythonBindings.cpp

HEADERS = initPython.h \
    ScriptController.h \
    PythonScriptController.h \
    PythonEnvironment.h \
    PythonBindings.h

README_FILE = Python.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
