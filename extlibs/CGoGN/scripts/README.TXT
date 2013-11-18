
Easy application creation:
-------------------------
skeleton_apps.sh:
Usage: skeleton_apps directory [list of additional source names]
This must be done in a CGoGN_Apps directory. It create an empty application
with source, header, ui, CMakeLists.txt, ...

skeleton_test.sh:
Usage: skeleton_apps directory 
This must be done in a CGoGN_Apps directory. It create an empty debug application
with only one source (no header, no ui, no Debug/Release)


duplicate_apps.sh:
Usage duplicate_apps src_dir dest_dir
Duplicate an application:
	- files that have the name of directory src are changed to dst 
	- includes are modified to follow these changes
	- Classes and vars in .h & .cpp file that have the name (with first 
 	  letter in uppercase) are changed with dst name

