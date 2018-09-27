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

  class PythonModule_SofaTypes_test : public PythonTestList
  {
  public:
    PythonModule_SofaTypes_test()
    {
      addTestDir(std::string(PYTHON_TESTFILES_DIR));
    }
  } python_tests;

  /// run test list using the custom name function getTestName.
  /// this allows to do gtest_filter=*FileName*
  INSTANTIATE_TEST_CASE_P(Batch,
			  PythonTest,
              ::testing::ValuesIn(python_tests.list),
              PythonTest::getTestName);

  TEST_P(PythonTest, all_tests) { run(GetParam()); }

}
