This directory gathers examples used in the SOFA documentation.
Each subdirectory contains an example and a .pro project file.
The project files include a common configuration file, examples.cfg, located in this directory.
You might have to edit this file to set the value of variable SOFAMAIN appropriately. This value defines the location of the main Sofa directory with respect to the examples (not with respect to this directory).

Remark for linux users: 
- your LD_LIBRARY_PATH environment variable should contain the directory where the shared libraries are: Sofa/lib/linux, where Sofa must be replaced by the complete path to the main Sofa directory
- each example comes with a kdevelop project file

