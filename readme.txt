Building Sofa under Windows :
-----------------------------
	
 - MSVC 6 :
     Launch Project VC6.bat. 

 - MSVC 7 : (Visual .net)
     Launch Project VC7.bat.	 

 - MSVC 8 : (Visual 2005)
     Launch Project VC8.bat.	 

 - console :
     Change the templates in sofa.cfg from vcapp/vclib/vcsubdirs to app/lib/subdirs.
     Then Launch Project VC6/7/8.bat depending on the version you use.
     Finally, build all project with nmake. 

After creating the project, verify that project example1 is the activated project.


Building Sofa under Linux :
---------------------------

- KDevelop :
     Open the Sofa.kdevelop project

- console :
     Run qmake, then make
