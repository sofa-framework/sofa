#!/bin/bash

if test $# -lt 1; then
	echo $0 application_directory
	exit 2
fi

echo "Warning do not forget to check the CGoGN_ROOT_DIR variable (ccmake)"

if test -d $PWD/$1; then
	echo Directory $1 already exist
	exit 3
fi

apps=$1

mkdir $apps
cd $apps

mkdir bin
mkdir build

echo "cmake_minimum_required(VERSION 2.8)" > CMakeLists.txt
echo project\( $apps \)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo SET\(CGoGN_ROOT_DIR \$\{CMAKE_SOURCE_DIR\}/../../CGoGN CACHE STRING \"CGoGN root dir\"\)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo include\(\$\{CGoGN_ROOT_DIR\}/apps_cmake.txt\)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo SET\(CMAKE_BUILD_TYPE Debug\)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo link_directories\( \$\{CGoGN_ROOT_DIR\}/lib/Debug/ \$\{CGoGN_ROOT_DIR\}/lib/Release \$\{Boost_LIBRARY_DIRS\}\)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo include_directories\(   >> CMakeLists.txt
echo "    " \$\{CGoGN_ROOT_DIR\}/include	   >> CMakeLists.txt	
echo "    " \$\{COMMON_INCLUDES\}				   >> CMakeLists.txt
echo "    " \$\{CMAKE_CURRENT_SOURCE_DIR\}		   >> CMakeLists.txt
echo "    " \$\{CMAKE_CURRENT_BINARY_DIR\}		   >> CMakeLists.txt
echo \)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo add_executable\( $apps ${apps}.cpp \)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo target_link_libraries\( $apps \$\{CGoGN_LIBS_D\} \$\{COMMON_LIBS\}\)   >> CMakeLists.txt

echo "/*******************************************************************************" > ${apps}.cpp
echo "* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *" >> ${apps}.cpp
echo "* version 0.1                                                                  *" >> ${apps}.cpp
echo "* Copyright (C) 2009-2011, IGG Team, LSIIT, University of Strasbourg           *" >> ${apps}.cpp
echo "*                                                                              *" >> ${apps}.cpp
echo "* This library is free software; you can redistribute it and/or modify it      *" >> ${apps}.cpp
echo "* under the terms of the GNU Lesser General Public License as published by the *" >> ${apps}.cpp
echo "* Free Software Foundation; either version 2.1 of the License, or (at your     *" >> ${apps}.cpp
echo "* option) any later version.                                                   *" >> ${apps}.cpp
echo "*                                                                              *" >> ${apps}.cpp
echo "* This library is distributed in the hope that it will be useful, but WITHOUT  *" >> ${apps}.cpp
echo "* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *" >> ${apps}.cpp
echo "* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *" >> ${apps}.cpp
echo "* for more details.                                                            *" >> ${apps}.cpp
echo "*                                                                              *" >> ${apps}.cpp
echo "* You should have received a copy of the GNU Lesser General Public License     *" >> ${apps}.cpp
echo "* along with this library; if not, write to the Free Software Foundation,      *" >> ${apps}.cpp
echo "* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *" >> ${apps}.cpp
echo "*                                                                              *" >> ${apps}.cpp
echo "* Web site: http://cgogn.unistra.fr/                                           *" >> ${apps}.cpp
echo "* Contact information: cgogn@unistra.fr                                        *" >> ${apps}.cpp
echo "*                                                                              *" >> ${apps}.cpp
echo "*******************************************************************************/" >> ${apps}.cpp
echo "" >> ${apps}.cpp
echo \#include \"Utils/cgognStream.h\" >> ${apps}.cpp
echo "" >> ${apps}.cpp
echo "using namespace CGoGN;" >> ${apps}.cpp
echo "" >> ${apps}.cpp

echo "int main(int argc, char **argv)" >> ${apps}.cpp
echo "{" >> ${apps}.cpp
echo "	CGoGNout << \"Hello CGoGN\"<<CGoGNendl;" >> ${apps}.cpp
echo "}" >> ${apps}.cpp

