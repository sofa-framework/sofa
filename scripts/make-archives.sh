#!/bin/bash
echo FILES
rm -f ../sofa-framework-${1:-VERSION}.files
find readme.txt LICENCE.txt Authors.txt *.bat config*.sh Project*.sh sofa-dependencies.prf sofa-default.prf *.pro extlibs framework features .qmake.cache -name .svn  -prune -o -name OBJ -prune -o \( -type f -a ! -name '*.bak' -a ! -name '*~' -a ! -name 'Makefile*' \) -print > ../sofa-framework-${1:-VERSION}.files
rm -f ../sofa-modules-${1:-VERSION}.files
find modules -name .svn  -prune -o -name OBJ -prune -o \( -type f -a ! -name '*~' -a ! -name '*.bak' -a ! -name 'Makefile*' \) -print > ../sofa-modules-${1:-VERSION}.files
rm -f ../sofa-applications-${1:-VERSION}.files
find applications tests examples share scripts -name .svn  -prune -o -name OBJ -prune -o \( -type f -a ! -name '*~' -a ! -name '*.bak' -a ! -name 'Makefile*' \) -print > ../sofa-applications-${1:-VERSION}.files
rm -f ../sofa-documentation-${1:-VERSION}.files
find doc doxygen.sh Doxyfile.in -name .svn  -prune -o -name OBJ -prune -o \( -type f -a ! -name '*~' -a ! -name '*.bak' -a ! -name 'Makefile*' \) -print > ../sofa-documentation-${1:-VERSION}.files

if which zip >/dev/null 2>/dev/null; then
    echo ZIP
    rm -f ../sofa-framework-${1:-VERSION}.zip
    zip -9 ../sofa-framework-${1:-VERSION}.zip -@ < ../sofa-framework-${1:-VERSION}.files
    rm -f ../sofa-modules-${1:-VERSION}.zip
    zip -9 ../sofa-modules-${1:-VERSION}.zip -@ < ../sofa-modules-${1:-VERSION}.files
    rm -f ../sofa-applications-${1:-VERSION}.zip
    zip -9 ../sofa-applications-${1:-VERSION}.zip -@ < ../sofa-applications-${1:-VERSION}.files
    rm -f ../sofa-documentation-${1:-VERSION}.zip
    zip -9 ../sofa-documentation-${1:-VERSION}.zip -@ < ../sofa-documentation-${1:-VERSION}.files
fi
if which 7z >/dev/null 2>/dev/null; then
    echo 7Z
    Z="7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=128m"
    rm -f ../sofa-framework-${1:-VERSION}.7z
    echo $Z -i@../sofa-framework-${1:-VERSION}.files ../sofa-framework-${1:-VERSION}.7z
    $Z -i@../sofa-framework-${1:-VERSION}.files ../sofa-framework-${1:-VERSION}.7z
    rm -f ../sofa-modules-${1:-VERSION}.7z
    $Z -i@../sofa-modules-${1:-VERSION}.files ../sofa-modules-${1:-VERSION}.7z
    rm -f ../sofa-applications-${1:-VERSION}.7z
    $Z -i@../sofa-applications-${1:-VERSION}.files ../sofa-applications-${1:-VERSION}.7z
    rm -f ../sofa-documentation-${1:-VERSION}.7z
    $Z -i@../sofa-documentation-${1:-VERSION}.files ../sofa-documentation-${1:-VERSION}.7z
fi
rm -f ../sofa-framework-${1:-VERSION}.files
rm -f ../sofa-modules-${1:-VERSION}.files
rm -f ../sofa-applications-${1:-VERSION}.files
rm -f ../sofa-documentation-${1:-VERSION}.files
