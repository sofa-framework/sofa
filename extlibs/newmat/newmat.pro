# Target is a library:  newmat
load(sofa/pre)

TEMPLATE = lib
TARGET = newmat

CONFIGSTATIC = static

DEFINES += use_namespace 

SOURCES = \    
        newmat/newmat1.cpp \
        newmat/newmat2.cpp \
        newmat/newmat3.cpp \
        newmat/newmat4.cpp \
        newmat/newmat5.cpp \
        newmat/newmat6.cpp \
        newmat/newmat7.cpp \
        newmat/newmat8.cpp \
        newmat/newmat9.cpp \
        newmat/newmatex.cpp \
        newmat/newmatrm.cpp \
        newmat/bandmat.cpp \
        newmat/submat.cpp \
        newmat/myexcept.cpp \
        newmat/cholesky.cpp \
        newmat/evalue.cpp \
        newmat/fft.cpp \
        newmat/hholder.cpp \
        newmat/jacobi.cpp \
        newmat/newfft.cpp \
        newmat/sort.cpp \
        newmat/svd.cpp 

HEADERS = \
        newmat/include.h \
        newmat/newmat.h \
        newmat/newmatrc.h \
        newmat/boolean.h \
        newmat/myexcept.h \
        newmat/controlw.h \
        newmat/newmatap.h \
        newmat/newmatrm.h \
        newmat/precisio.h 

load(sofa/post)
