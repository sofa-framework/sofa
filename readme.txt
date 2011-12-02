
   SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1
           (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH

  Authors: See Authors.txt


SOFA is an Open Source framework primarily targeted at real-time simulation, 
with an emphasis on medical simulation. It is mainly intended for the 
research community to help foster newer algorithms, but can also be used as 
an efficient prototyping tool. SOFA's advanced software architecture allows:
(1) the creation of complex and evolving simulations by combining new algorithms with 
    existing algorithms; 
(2) the modification of key parameters of the simulation  such as deformable
    behavior, surface representation, solvers, constraints, collision algorithm, etc. by 
    simply editing an XML file; 
(3) the synthesis of complex models from simpler ones using a scene-graph description; 
(4) the efficient simulation of the dynamics of interacting objects using abstract 
    equation solvers; and 
(5) the comparison of various algorithms available in SOFA. 


LICENSES: 

The SOFA distribution is composed of three parts:
 - SOFA FRAMEWORK: this is essential the core of SOFA
 - SOFA MODULES: represent the functionalities available in SOFA
 - SOFA APPLICATIONS: they are built using the SOFA framework and modules.

SOFA is free software; you can redistribute it and/or modify it under the 
terms of the following licenses.

The SOFA FRAMEWORK can be redistributed and/or modified under the terms of 
the GNU Lesser General Public License as published by the Free Software 
Foundation; either version 2.1 of the License, or (at your option) any later 
version.

The SOFA MODULES can be redistributed and/or modified under the terms of 
the GNU Lesser General Public License as published by the Free Software 
Foundation; either version 2.1 of the License, or (at your option) any later 
version.

This SOFA APPLICATIONS can be redistributed and/or modified under the terms of
the GNU General Public License as published by the Free Software Foundation;
either version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public 
License and GNU General Public License for more details.

Contact information: contact@sofa-framework.org

-----------------------------------------------------------------------------

INSTALLATION

NOTE: for up-to-date instructions, please refer to the web site:
http://www.sofa-framework.org/installation

Before building Sofa, copy sofa-default.prf to sofa-local.prf and
customize it to your local configuration.

Building Sofa under Windows :
-----------------------------
 - First of all, get the  correct version of windows dependencies from
 SOFA web page, and unzip it in Sofa directory. Then...

 - MSVC 8 (Visual 2005) :
     Launch Project VC8.bat.	 

 - MSVC 9 (Visual 2008) and MSVC 10 (Visual 2010) :
     Launch Project VC9.bat.	 

 - console :
     Open a command prompt with environment variables pointing to Visual
     Studio.
     If you already have qmake installed and configured simply launch qmake.
     If not launch project.bat.
     Then launch nmake to build all projects.

After creating the projects, verify that project runSofa is the activated
project, and that Release mode is used instead of Debug.


Building Sofa under Linux :
---------------------------

- QtCreator :
     Open the Sofa.pro project

- console :
     Run qmake, then make

Building Sofa under Mac OS :
----------------------------
     Download the mac dependencies package from SOFA web page,
     and unzip it in Sofa directory.
     Run Project MacOS.sh
     Then run make to build all projects.
     The generated Xcode projects can  be used to easily edit the code
     but with current version of qmake then can not be used to compile
     it correctly.
     It is also possible (and recommended) to use QtCreator instead.
