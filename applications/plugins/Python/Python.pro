load(sofa/pre)
defineAsPlugin(Python)

TARGET = Python

DEFINES += SOFA_BUILD_PYTHON

#python
mac: LIBS += -F/opt/local/Library/Frameworks/ -framework Python
INCLUDEPATH += /opt/local/Library/Frameworks/Python.framework/Headers

#boost.python
LIBS += -L/opt/local/lib/ -lboost_python
INCLUDEPATH += /opt/local/include

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
