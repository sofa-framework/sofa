#!/bin/bash
rm -f ../sofa-framework-${1:-VERSION}.zip
find readme.txt Authors.txt *.bat *.sh sofa.cfg sofa-default.cfg *.pro *.kdevelop Doxyfile extlibs framework -name .svn  -prune -o -name OBJ -prune -o \( -type f -a ! -name '*.bak' -a ! -name '*~' -a ! -name Makefile \) -print | zip -9 ../sofa-framework-${1:-VERSION}.zip -@
rm -f ../sofa-modules-${1:-VERSION}.zip
find modules -name .svn  -prune -o -name OBJ -prune -o \( -type f -a ! -name '*~' -a ! -name '*.bak' -a ! -name Makefile \) -print | zip -9 ../sofa-modules-${1:-VERSION}.zip -@
rm -f ../sofa-applications-${1:-VERSION}.zip
find applications examples share -name .svn  -prune -o -name OBJ -prune -o \( -type f -a ! -name '*~' -a ! -name '*.bak' -a ! -name Makefile \) -print | zip -9 ../sofa-applications-${1:-VERSION}.zip -@
