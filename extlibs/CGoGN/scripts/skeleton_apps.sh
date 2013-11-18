#!/bin/bash

if test $# -lt 1; then
	echo $0 application_directory [src_files]
	exit 2
fi

if test -d $PWD/$1; then
	echo Directory $1 already exist
	exit 3
fi

echo "Warning do not forget to check the CGoGN_ROOT_DIR variable (ccmake)"

apps=$1

#create a string with first letter capitalize (${apps^} does not work on osX !
apps_maj=`echo $1 | awk '
BEGIN { upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lower = "abcdefghijklmnopqrstuvwxyz"
}
{
	FIRSTCHAR = substr($1, 1, 1)
	if (CHAR = index(lower, FIRSTCHAR))
		$1 = substr(upper, CHAR, 1) substr($1, 2)
	print $0
}' `



mkdir $apps
cd $apps


sources=\$\{CMAKE_SOURCE_DIR\}/$1.cpp
touch $1.cpp
shift
while test $# -ge 1; do
	touch $1
	sources=`echo $sources " " \$\{CMAKE_SOURCE_DIR\}/$1`
	shift
done



mkdir bin
mkdir build
mkdir Release
mkdir Debug

echo "cmake_minimum_required(VERSION 2.8)" > CMakeLists.txt
echo ""  >> CMakeLists.txt
echo project\( $apps \)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo SET\(CGoGN_ROOT_DIR \$\{CMAKE_SOURCE_DIR\}/../../CGoGN CACHE STRING \"CGoGN root dir\"\)   >> CMakeLists.txt
echo include\(\$\{CGoGN_ROOT_DIR\}/apps_cmake.txt\)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo add_subdirectory\(\$\{CMAKE_SOURCE_DIR\}/Release Release\)  >> CMakeLists.txt
echo IF \(NOT WIN32\)  >> CMakeLists.txt
echo "	" add_subdirectory\(\$\{CMAKE_SOURCE_DIR\}/Debug Debug\)  >> CMakeLists.txt
echo ENDIF \(NOT WIN32\)  >> CMakeLists.txt



cd Debug
echo "cmake_minimum_required(VERSION 2.8)" > CMakeLists.txt
echo ""  >> CMakeLists.txt
echo SET\(CMAKE_BUILD_TYPE Debug\)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo include_directories\(   >> CMakeLists.txt
echo "	" \$\{CGoGN_ROOT_DIR\}/include	   >> CMakeLists.txt	
echo "	" \$\{COMMON_INCLUDES\}				   >> CMakeLists.txt
echo "	" \$\{CMAKE_CURRENT_SOURCE_DIR\}		   >> CMakeLists.txt
echo "	" \$\{CMAKE_CURRENT_BINARY_DIR\}		   >> CMakeLists.txt
echo \)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo link_directories\( \$\{CGoGN_ROOT_DIR\}/lib/Debug/ \$\{CGoGN_ROOT_DIR\}/lib/Release \$\{Boost_LIBRARY_DIRS\}\)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo QT4_WRAP_UI\( ${apps}_ui \$\{CMAKE_SOURCE_DIR\}/${apps}.ui \)  >> CMakeLists.txt
echo QT4_WRAP_CPP\( ${apps}_moc \$\{CMAKE_SOURCE_DIR\}/${apps}.h \)  >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo add_executable\( ${apps}D $sources >> CMakeLists.txt
echo "	" \$\{${apps}_moc\}  >> CMakeLists.txt
echo "	" \$\{${apps}_ui\} \)  >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo target_link_libraries\( ${apps}D \$\{CGoGN_LIBS_D\} \$\{COMMON_LIBS\}\)   >> CMakeLists.txt


cd ../Release
echo "cmake_minimum_required(VERSION 2.8)" > CMakeLists.txt
echo ""  >> CMakeLists.txt
echo SET\(CMAKE_BUILD_TYPE Release\)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo include_directories\(   >> CMakeLists.txt
echo "	" \$\{CGoGN_ROOT_DIR\}/include	   >> CMakeLists.txt	
echo "	" \$\{COMMON_INCLUDES\}				   >> CMakeLists.txt
echo "	" \$\{CMAKE_CURRENT_SOURCE_DIR\}		   >> CMakeLists.txt
echo "	" \$\{CMAKE_CURRENT_BINARY_DIR\}		   >> CMakeLists.txt
echo \)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo link_directories\( \$\{CGoGN_ROOT_DIR\}/lib/Release \$\{Boost_LIBRARY_DIRS\}\)   >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo QT4_WRAP_UI\( ${apps}_ui \$\{CMAKE_SOURCE_DIR\}/${apps}.ui \)  >> CMakeLists.txt
echo QT4_WRAP_CPP\( ${apps}_moc \$\{CMAKE_SOURCE_DIR\}/${apps}.h \)  >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo add_executable\( ${apps} $sources >> CMakeLists.txt
echo "	" \$\{${apps}_moc\}  >> CMakeLists.txt
echo "	" \$\{${apps}_ui\} \)  >> CMakeLists.txt
echo ""  >> CMakeLists.txt
echo target_link_libraries\( $apps \$\{CGoGN_LIBS_R\} \$\{COMMON_LIBS\}\)   >> CMakeLists.txt

cd ..

echo "/*******************************************************************************" > ${apps}.h
echo "* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *" >> ${apps}.h
echo "* version 0.1                                                                  *" >> ${apps}.h
echo "* Copyright (C) 2009-2011, IGG Team, LSIIT, University of Strasbourg           *" >> ${apps}.h
echo "*                                                                              *" >> ${apps}.h
echo "* This library is free software; you can redistribute it and/or modify it      *" >> ${apps}.h
echo "* under the terms of the GNU Lesser General Public License as published by the *" >> ${apps}.h
echo "* Free Software Foundation; either version 2.1 of the License, or (at your     *" >> ${apps}.h
echo "* option) any later version.                                                   *" >> ${apps}.h
echo "*                                                                              *" >> ${apps}.h
echo "* This library is distributed in the hope that it will be useful, but WITHOUT  *" >> ${apps}.h
echo "* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *" >> ${apps}.h
echo "* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *" >> ${apps}.h
echo "* for more details.                                                            *" >> ${apps}.h
echo "*                                                                              *" >> ${apps}.h
echo "* You should have received a copy of the GNU Lesser General Public License     *" >> ${apps}.h
echo "* along with this library; if not, write to the Free Software Foundation,      *" >> ${apps}.h
echo "* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *" >> ${apps}.h
echo "*                                                                              *" >> ${apps}.h
echo "* Web site: http://cgogn.unistra.fr/                                           *" >> ${apps}.h
echo "* Contact information: cgogn@unistra.fr                                        *" >> ${apps}.h
echo "*                                                                              *" >> ${apps}.h
echo "*******************************************************************************/" >> ${apps}.h
echo \#ifndef _${apps}_ >> ${apps}.h
echo \#define _${apps}_ >> ${apps}.h
echo \#include \"Utils/Qt/qtSimple.h\" >> ${apps}.h
echo \#include \"ui_${apps}.h\" >> ${apps}.h
echo \#include \"Utils/Qt/qtui.h\" >> ${apps}.h
echo "" >> ${apps}.h
echo "using namespace CGoGN;" >> ${apps}.h
echo "" >> ${apps}.h
echo "class $apps_maj: public Utils::QT::SimpleQT" >> ${apps}.h
echo "{" >> ${apps}.h
echo "	Q_OBJECT" >> ${apps}.h
echo "public:" >> ${apps}.h
echo "" >> ${apps}.h
echo "	$apps_maj() {}" >> ${apps}.h
echo "" >> ${apps}.h
echo "	~$apps_maj() {}" >> ${apps}.h
echo "" >> ${apps}.h
echo "	void cb_redraw();" >> ${apps}.h
echo "" >> ${apps}.h
echo "	void cb_initGL();" >> ${apps}.h
echo "" >> ${apps}.h
echo "	//void cb_mousePress(int button, int x, int y);" >> ${apps}.h
echo "" >> ${apps}.h
echo "	//void cb_mouseRelease(int button, int x, int y);" >> ${apps}.h
echo "" >> ${apps}.h
echo "	//void cb_mouseClick(int button, int x, int y);" >> ${apps}.h
echo "" >> ${apps}.h
echo "	//void cb_mouseMove(int buttons, int x, int y);" >> ${apps}.h
echo "" >> ${apps}.h
echo "	//void cb_wheelEvent(int delta, int x, int y);" >> ${apps}.h
echo "" >> ${apps}.h
echo "	//void cb_keyPress(int code);" >> ${apps}.h
echo "" >> ${apps}.h
echo "	//void cb_keyRelease(int code);" >> ${apps}.h
echo "" >> ${apps}.h
echo "public slots:" >> ${apps}.h
echo "};" >> ${apps}.h
echo "" >> ${apps}.h
echo "#endif" >> ${apps}.h



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
echo "* Web site: http://cgogn.u-strasbg.fr/                                         *" >> ${apps}.cpp
echo "* Contact information: cgogn@unistra.fr                                        *" >> ${apps}.cpp
echo "*                                                                              *" >> ${apps}.cpp
echo "*******************************************************************************/" >> ${apps}.cpp
echo "" >> ${apps}.cpp
echo \#include \"${apps}.h\" >> ${apps}.cpp
echo "" >> ${apps}.cpp
echo "void $apps_maj::cb_initGL()" >> ${apps}.cpp
echo "{}" >> ${apps}.cpp
echo "" >> ${apps}.cpp
echo "void $apps_maj::cb_redraw()" >> ${apps}.cpp
echo "{}" >> ${apps}.cpp
echo "" >> ${apps}.cpp

echo "int main(int argc, char **argv)" >> ${apps}.cpp
echo "{" >> ${apps}.cpp
echo "" >> ${apps}.cpp
echo "	QApplication app(argc, argv);" >> ${apps}.cpp
echo "	$apps_maj sqt;" >> ${apps}.cpp
echo "	"sqt.setWindowTitle\(\"$apps_maj\"\)\; >> ${apps}.cpp
echo "" >> ${apps}.cpp
echo "	Utils::QT::uiDockInterface dock;" >> ${apps}.cpp
echo "	sqt.setDock(&dock);" >> ${apps}.cpp
echo "" >> ${apps}.cpp

echo "	sqt.show();" >> ${apps}.cpp
echo "" >> ${apps}.cpp
echo "	return app.exec();" >> ${apps}.cpp
echo "}" >> ${apps}.cpp



echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" > ${apps}.ui
echo "<ui version=\"4.0\">" >> ${apps}.ui
echo " <class>DockWidget</class>" >> ${apps}.ui
echo " <widget class=\"QDockWidget\" name=\"DockWidget\">" >> ${apps}.ui
echo "  <property name=\"geometry\">" >> ${apps}.ui
echo "   <rect>" >> ${apps}.ui
echo "	<x>0</x>" >> ${apps}.ui
echo "	<y>0</y>" >> ${apps}.ui
echo "	<width>144</width>" >> ${apps}.ui
echo "	<height>258</height>" >> ${apps}.ui
echo "   </rect>" >> ${apps}.ui
echo "  </property>" >> ${apps}.ui
echo "  <property name=\"sizePolicy\">" >> ${apps}.ui
echo "   <sizepolicy hsizetype=\"Expanding\" vsizetype=\"Expanding\">" >> ${apps}.ui
echo "	<horstretch>0</horstretch>" >> ${apps}.ui
echo "	<verstretch>0</verstretch>" >> ${apps}.ui
echo "   </sizepolicy>" >> ${apps}.ui
echo "  </property>" >> ${apps}.ui
echo "  <property name=\"cursor\">" >> ${apps}.ui
echo "   <cursorShape>PointingHandCursor</cursorShape>" >> ${apps}.ui
echo "  </property>" >> ${apps}.ui
echo "  <property name=\"floating\">" >> ${apps}.ui
echo "   <bool>false</bool>" >> ${apps}.ui
echo "  </property>" >> ${apps}.ui
echo "  <property name=\"features\">" >> ${apps}.ui
echo "   <set>QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetMovable</set>" >> ${apps}.ui
echo "  </property>" >> ${apps}.ui
echo "  <property name=\"allowedAreas\">" >> ${apps}.ui
echo "   <set>Qt::LeftDockWidgetArea|Qt::RightDockWidgetArea</set>" >> ${apps}.ui
echo "  </property>" >> ${apps}.ui
echo "  <property name=\"windowTitle\">" >> ${apps}.ui
echo "   <string>Control</string>" >> ${apps}.ui
echo "  </property>" >> ${apps}.ui
echo "  <widget class=\"QWidget\" name=\"dockWidgetContents\">" >> ${apps}.ui
echo "   <property name=\"sizePolicy\">" >> ${apps}.ui
echo "	<sizepolicy hsizetype=\"Expanding\" vsizetype=\"Expanding\">" >> ${apps}.ui
echo "	 <horstretch>0</horstretch>" >> ${apps}.ui
echo "	 <verstretch>0</verstretch>" >> ${apps}.ui
echo "	</sizepolicy>" >> ${apps}.ui
echo "   </property>" >> ${apps}.ui
echo "   <property name=\"minimumSize\">" >> ${apps}.ui
echo "	<size>" >> ${apps}.ui
echo "	 <width>122</width>" >> ${apps}.ui
echo "	 <height>0</height>" >> ${apps}.ui
echo "	</size>" >> ${apps}.ui
echo "   </property>" >> ${apps}.ui
echo "  </widget>" >> ${apps}.ui
echo " </widget>" >> ${apps}.ui
echo " <resources/>" >> ${apps}.ui
echo " <connections/>" >> ${apps}.ui
echo "</ui>" >> ${apps}.ui



