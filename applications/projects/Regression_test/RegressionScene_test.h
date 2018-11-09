/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_RegressionScene_test_H
#define SOFA_RegressionScene_test_H

#include "RegressionScene_list.h"
#include <sofa/helper/testing/BaseTest.h>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>

#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <SofaExporter/WriteState.h>
#include <SofaGeneralLoader/ReadState.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <SofaValidation/CompareState.h>

#include <SofaTest/Sofa_test.h>
#include <sofa/helper/system/FileRepository.h>

using sofa::helper::testing::BaseTest;

namespace sofa 
{

/// To Perform a Regression Test on scenes
///
/// A scene is run for a given number of steps and the state (position/velocity) of every independent dofs is stored in files. These files must be added to the repository.
/// At each commit, a test runs the scenes again for the same given number of steps. Then the independent states are compared to the references stored in the files.
///
/// The reference files are generated when running the test for the first time on a scene.
/// @warning These newly created reference files must be added to the repository.
/// If the result of the simulation changed voluntarily, these files must be manually deleted (locally) so they can be created again (by running the test).
/// Their modifications must be pushed to the repository.
///
/// Scene tested for regression must be listed in a file "list.txt" located in a "regression" directory in the test directory ( e.g. myplugin/myplugin_test/regression/list.txt)
/// Each line of the "list.txt" file must contain: a local path to the scene, the number of simulation steps to run, and a numerical epsilon for comparison.
/// e.g. "gravity.scn 5 1e-10" to run the scene "regression/gravity.scn" for 5 time steps, and the state difference must be smaller than 1e-10
///
/// As an example, have a look to SofaTest_test/regression
///
/// @author Matthieu Nesme
/// @date 2015
class RegressionScene_test: public BaseSimulationTest, public ::testing::WithParamInterface<RegressionSceneTest_Data>
{
public:
    /// Method that given the RegressionSceneTest_Data will return the name of the file tested without the path neither the extension
    static std::string getTestName(const testing::TestParamInfo<RegressionSceneTest_Data>& p);

    /// Method to really perfom the test and compare the states vector between current simulation and reference file.
    void runRegressionStateTest(RegressionSceneTest_Data data);
};


/// Structure creating and storing the RegressionSceneTest_Data from Sofa src paths as a list for gtest 
static struct RegressionStateScenes_list : public RegressionScene_list
{
    RegressionStateScenes_list()
    {
        collectScenesFromPaths("list.txt");
    }
} regressionState_tests;




} // namespace sofa

#endif // SOFA_RegressionScene_test_H
