  
   SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1
           (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS

  Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette, 
  F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann, and F. Poyer


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

Before building Sofa, look in sofa.cfg to configure it. Most importantly
specify the GUI to use (FLTK, QT, or both).

Building Sofa under Windows :
-----------------------------
	
 - MSVC 6 :
     Launch Project VC6.bat. 

 - MSVC 7 : (Visual .net)
     Launch Project VC7.bat.	 

 - MSVC 8 : (Visual 2005)
     Launch Project VC8.bat.	 

 - console :
     Open a command prompt with environment variables pointing to Visual
     Studio.
     If you already have qmake installed and configured simply launch qmake.
     If not launch project.bat.
     Then launch nmake to build all projects.

After creating the projects, verify that project runSOFA is the activated
project.


Building Sofa under Linux :
---------------------------

- KDevelop :
     Open the Sofa.kdevelop project

- console :
     Run qmake, then make


