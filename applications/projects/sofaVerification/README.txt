How to use sofaVerification:

the goal of this project is to create a dashboard for Sofa, but not testing the result of the compilation, but the behavior of Sofa through the different commits.
In a file (by default, it will be "verification.ini"), we will specify a list of scenes to test. We first need to store a reference state for these scenes. Typically, we should do that using the latest stable version of Sofa, and don't touch it anymore. This will run the simulation on a given number of iterations, and at each time step save the position of the dofs (even through the mappings). Now, we have in memory the correct behavior of these scenes.
Each time someone commits something, we should run the verification that consists in running the same set of scenes, and comparing at each time step the state simulated with the reference state. We accumulate the error, and as a result present the total error by dof, and the time spent. Like this, we can quickly, and automatically know if we added error, if we improved or not the computation time...
We can add files to the set of files too.


This script is done using Bash, and use the debug version of sofaVerification. We have seen errors due to optimization when compiling with -O2. 

The command line: [param] means the option is optional, the order doesn't import

./runVerification.sh [-i number_of_iterations] [-r] [-f file.ini]  [-a set_of_files_to_add]


-i : to specify the number of iterations. By default, we use "100"
-r : to save new references for the scenes: just record the positions of the dofs
-f : to specify the reference set of files. By default, we use "verification.ini"
-a : to add a set of files to the reference set of files, and record them. Just like "verification.ini", it merges the reference set specified by the option -f (or verification.ini by default) with the file specified by the option -a. It will initialize the added files using the number of iterations specified by the option -i. 


