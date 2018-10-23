/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/******************************************************************************
 * Contributors:                                                              *
 *    - damien.marchal@univ-lille1.fr                                         *
 *****************************************************************************/

#include <vector>

#include <SofaPython3/PythonTest.h>
using sofapython3::PythonTest ;
using sofapython3::PythonTestList ;
using sofapython3::PrintTo ;
using std::string;

namespace
{

class Sofa : public PythonTest {};

/// static build of the test list
static struct PythonModule_Sofa_tests : public PythonTestList
{
    PythonModule_Sofa_tests()
    {
        addTestDir(std::string(PYTHON_TESTFILES_DIR)+"/Core", "Sofa_Core_");       
        addTestDir(std::string(PYTHON_TESTFILES_DIR)+"/Helper", "Sofa_Helper_");
        addTestDir(std::string(PYTHON_TESTFILES_DIR)+"/Simulation", "Sofa_Simulation_");
        addTestDir(std::string(PYTHON_TESTFILES_DIR)+"/Types", "Sofa_Types_");
    }
} python_tests;

/// run test list using the custom name function getTestName.
/// this allows to do gtest_filter=*FileName*
INSTANTIATE_TEST_CASE_P(SofaPython3,
                        Sofa,
                        ::testing::ValuesIn(python_tests.list),
                        Sofa::getTestName);

TEST_P(Sofa, all_tests) { run(GetParam()); }

}
