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

After creating the projects, verify that project example1 is the activated
project.


Building Sofa under Linux :
---------------------------

- KDevelop :
     Open the Sofa.kdevelop project

- console :
     Run qmake, then make

TEST MAIL Automatique
