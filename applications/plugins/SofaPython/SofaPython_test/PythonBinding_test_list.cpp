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
/******************************************************************************
 * Contributors:
 *      - damien.marchal@univ-lille1.fr
 ******************************************************************************/
#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;

class SetOfPythonScenes
{
public:
    std::vector<std::string> m_scenes ;

    SetOfPythonScenes(const std::initializer_list<std::string>& s)
    {
        for(auto& i : s)
        {
            addATestScene(i) ;
        }
    }

    void addATestScene(const std::string s)
    {
        static const std::string rootPath = std::string(SOFAPYTHON_TEST_PYTHON_DIR);
        m_scenes.push_back(rootPath+"/"+s);
    }
} ;

class PythonBinding_tests :  public BaseSimulationTest
                            ,public ::testing::WithParamInterface<std::string>
{
public:

    PythonBinding_tests()
    {
        importPlugin("SofaPython") ;
    }

    void runTest(const std::string& filename){
        SceneInstance c=SceneInstance::LoadFromFile(filename) ;
    }
} ;

SetOfPythonScenes scenes = {"test_BindingBase.py",
                            "test_BindingData.py",
                            "test_BindingLink.py",
                            "test_BindingSofa.py"} ;

TEST_P(PythonBinding_tests, scene)
{
   this->runTest(GetParam());
}

INSTANTIATE_TEST_CASE_P(PythonBinding,
						PythonBinding_tests,
                        ::testing::ValuesIn(scenes.m_scenes));



