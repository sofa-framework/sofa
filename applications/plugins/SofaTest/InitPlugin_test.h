/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef INITPlugin_Test_H
#define INITPlugin_Test_H


#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_TEST
#define SOFA_SOFATEST_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_SOFATEST_API  SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

/** \mainpage
This is the test suite of Sofa. It contains base classes to ease the development of tests, such as sofa::Sofa_test , sofa::Mapping_test, etc.
The actual tests are implemented in sub-directories of the modules and plugins. For instance, SofaRigid/SofaRigid_test/RigidMapping_test.cpp.

The tests are optional. They are activated using the SOFA-MISC_TESTS flag of the cmake configuration.
This generates a set of executables, each of them running the test suite of the module. For instance, from the debug build directory, run:
 \code{.sh} bin/SofaConstraint_testd \endcode
To run all the tests, run command \code{.sh} ctest --verbose \endcode

<h3> Motivation </h3>

A test suite for SOFA  is being developed using the googletest framework : http://code.google.com/p/googletest/wiki/Documentation
Tests serve two purposes:
- Automatically detect regressions. They are automatically run after each commit and their results are displayed on the dashboard http://www.sofa-framework.org/dash/trunk/. This way, changes which break existing features are detected as soon as possible.
- Help developing. Creating the specific test at the same time as your new feature (test-oriented development) has significant advantages:
    -  it helps you specifying your code: what it does is what is tested
    - focusing on your contribution, without being distracted by other stuff
    - being sure that your contribution will not be accidentally broken by anyone.
In summary, test-oriented development generates better code and is easier.  Therefore, we strongly urge you to apply it. Feel free to ask us for advice.

<h3> Activation </h3>

When the SOFA-MISC_TESTS option is checked in CMake, all the applications/plugins/PluginName/PluginName_test projects are automatically included by Cmake in the Sofa project/solution.
Each test project generates an executable, which outputs its results on the standard output. The final output is the number of successful tests (PASSED) and the number of fails (FAILED) if any.

Plugin SofaTest is the basis of all tests. It includes:
- base classes for creating tests in Sofa. As such, the other tests include it in their cmake LinkerDependencies.
- subdirectory SofaTest_test is used to test the components of the sofa/modules directory. It is far from complete.

Other plugins provide tests, such as Compliant, Flexible and Image. Note that the tests are generally not extensive, so they do not guaranty that the code is bug-free.

<h3> Running the tests </h3>

Once you build every tests you want, simply go in your build directory and execute the following command in order to launch the whole test suite:
- ctest --verbose

<h3> How to create tests in your plugin </h3>
Say you are creating YourPlugin in applications/plugins/YourPlugin. The steps to create a test suite are:
- create directory called applications/plugins/YourPlugin/YourPlugin_test or some other name ending up with _test, so that it is automatically included in the test suite.
- in this directory, create a cmake project file for an executable, and set up dependencies on YourPlugin and on SofaTest. See e.g. applications/plugins/Compliant/Compliant_test/CMakeLists.txt
- create a number of .cpp files to test your classes. Each test or test suite typically derives from class Sofa_test or one of the generic test classes derived from it: Solver_test, Mapping_test or ProjectionConstraintSet_test. The test code typically includes checkings, such as ASSERT_TRUE(bool). It is run by macros such as TEST_F at the end of the file.
See e.g. Compliant_test.


<h3>  How to test components </h3>

- Force field: Force field tests should derive from the base class ForceField_test.h available in  plugin SofaTest.This base class creates a minimal scene with a mechanical object and a forcefield. Then call  the function run_test with positions, velocities and the corresponding expected forces. This function automatically checks not only the forces (function addForce), but also the stiffness (methods addDForce and addKToMatrix), using finite differences.

For example, see StiffSpringForceField_test or QuadPressureForceField_test.

- Mapping: Mapping tests should derive from the base class Mapping_test.h available in plugin SofaTest.This base class creates a scene with two mechanical objects (parent and children nodes) and a mapping between them. Then it compares the actual output positions with the expected ones and automatically tests the methods related to Jacobian (applyJ, applyJT, applyDJT and getJs).

For example, RigidMapping_test tests the mapping from local to world coordinates.

- Solvers: To test a solver, one tests its convergence to a static solution. For example, EulerImplicit_test tests the convergence of euler implicit solver with a mass-spring system. This system is composed of 2 particles in gravity with one fixed particle. The other particle should move to a balance point. Then one checks two criteria:
    -if it has converged
    -if it has converged to the expected position

Other solver tests are available in Compliant_test: AssembledSolver_test and DampedOscillator_test.

- Projective constraint: To test projective constraint, one creates a minimal scene with a mechanical object, a topology and the projective constraint. One defines the  constraint parameters (points to project, normal of the projection...). Then one inits the scene and call the projectPosition() function. Finaly one checks two criteria:
    -if constrained particle have the expected position.
    -if unconstrained particle have not changed.

Some projective constraint tests are available in SofaTest_test: ProjectToLineConstraint and ProjectToPlaneConstraint.

- Engine test: To test engine you set input values and check if the ouput values correspond to the expected ones. The test Engine_test tests if the update method is called only if necessary. To test this a minimal engine TestEngine was created with a counter in its update method.


<h3> Test entirely written in python </h3>
- Testing a Sofa scene
The SofaTest plugin has a python API giving a Controller. You can write a Sofa scene in python (with the regular SofaPython API and the createScene function), and add a SofaTest.Controller to your scene. From the SofaTest.Controller you can return the test result (functions sendSuccess / sendFailure). A message can be passed in case of failure. Warning: do not forget to call the base function SofaTest.Controller.onLoaded if your surcharge this function in your controller.
- Test a pure python function (independent from SOFA)
You simply need to create a python script with a function "run()" return the test result as a boolean.

Your python scripts must be added to the gtest framework with the SofaTest/Python_test.h API to be executed automatically
Note that arguments can be given, so the same script can be called with several parameters (accessible as argc/argv on the python side).
Have a look to SofaTest_test for an example.

<h3> Investigating failures </h3>
Regressions typically break a couple of tests, but not all of them. To investigate, you generally want to run these tests only. Moreover, you typically need to modify these, by adding some debug prints or changing parameters. To avoid damaging the test suite, it is a good idea to clone it and work on the cloned version. Assuming that you are investigating test failures in SomePlugin/SomePlugin_test, you can apply the following steps:
- copy SomePlugin/SomePlugin_test to SomePlugin/SomePluginTMP_test or any other name ending up with _test.
- move to this directory and edit CMakeLists.txt to remove all the test files you do not need
- update you Sofa project/solution by running cmake as you usually do; the new test directory will automatically be included in your project/solution if its name ends up with _test
- modify the test as needed, and fix the problems
- update the original tests if necessary
- check that the original tests are successful

Feel free to add new tests to the original test suite, but think twice before modifying an existing test: this might destroy its ability to detect other problems.

<h3> Different test levels </h3>

<h4> Low-level tests </h4>

Low-level tests test only one single component or feature.
For example, the matrix test (Matrix_test.cpp in SofaTest_test) checks the filling of EigenMatrix.

<h4> Middle-level tests </h4>

Middle-level tests test one component needed other components.
For example:
- to test force field you need to have a mechanical object
- to test mapping you need to have two mechanical objects and a mapping between them.
For mapping and force field, two base classes are available in SofaTest.
Then to test a force field/mapping, you just have to create a class inherited from the base class ForceField_test/Mapping_test (see StiffSpringForceField_test.cpp/RigidMapping_test.cpp in SofaTest_test).

To do a middle-level test, one needs to create the minimal scene necessary to test the component. Then one can directly test the function (for example AddForce() for force, apply() for mapping).

<h4> High-level tests </h4>

High-level tests test a whole simulation scene until convergence.
For example, patch test (AffinePatch_test.cpp in SofaTest_test) is a high-level test because it tests the result after the convergence of a whole simulation scene. The patch test applies indeed an affine movement on regular grid border points and then after simulation one checks if the points within the grid have had the same affine movement.

To do a high-level test, one needs to create the scene (in xml, in C++ or in python), to init it and then animate until convergence.



<h3> Data files</h3>
Some tests require to open data files. These are typically located in the same directory as the test code.
The path to the current directory can be defined in the CMakeLists.txt, and passed by the compiler as a predefined symbol.
For instance, if you set the following line in CMakeLists.txt:

AddCompilerDefinitions("THIS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\\"")

then the following instruction creates a complete, absolute path to fileName:

std::string fileName = std::string(THIS_DIR) + "/" + fileName;

See e.g. SofaTest_test/CMakeLists.txt and SofaTest_test/LoadScene_test.cpp




<h3> Regression Tests</h3>

They are high-level tests checking that the result of a simulation is always giving the same results.
The states (position/velocity) of the indendent dofs are compared to a reference.
Have a look to SofaTest_test/Regression_test documentation for more details (how to add scene files to be tested).




@author François Faure, Aurélie Dégletagne, Matthieu Nesme and hopefully lots of others!
@date Started in 2013

This is a the starting page of the plugin documentation, defined in file InitPlugin_test.h
  */

#endif // INITPlugin_Test_H
