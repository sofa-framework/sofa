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
#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

#include <SofaTest/Python_test.h>
using sofa::Python_test ;

#include <SofaTest/TestMessageHandler.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::core::objectmodel::Data ;
using sofa::simulation::graph::DAGSimulation;

#include <SofaGeneralEngine/SphereROI.h>
using sofa::component::engine::SphereROI ;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include "../SceneLoaderPSL.h"
using sofa::simulation::SceneLoaderPSL ;
using sofa::core::objectmodel::BaseObject ;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;
using sofa::core::RegisterObject ;

#include <PSL/components/TestResult.h>
using sofa::component::TestResult ;

#include <SofaPython/PythonEnvironment.h>
using sofa::simulation::PythonEnvironment ;


using std::vector;
using std::string;

namespace
{

void anInit(){
    static bool _inited_ = false;
    if(!_inited_){
        PluginManager::getInstance().loadPlugin("PSL") ;
        PluginManager::getInstance().loadPlugin("SofaPython") ;
    }
}


class PSL_test : public BaseTest,
                 public ::testing::WithParamInterface<std::vector<std::string>>
{
public:
    Node::SPtr m_root ;

    void SetUp(){
        anInit() ;
        if( !sofa::simulation::getSimulation() )
            sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());
    }

    void TearDown(){
         sofa::simulation::getSimulation()->unload( m_root ) ;
    }

    void checkTestFilesMsg(const std::vector<std::string>& params)
    {
        ASSERT_EQ(params.size(), (unsigned int)3) ;
        if(params[2] == "Error"){
            EXPECT_MSG_EMIT(Error) ;
            checkTestFiles(params) ;
        }else{
            EXPECT_MSG_NOEMIT(Error) ;
            checkTestFiles(params) ;
        }
    }

    void checkTestFiles(const std::vector<std::string>& params)
    {
        std::string sresult = params[1];
        std::string scenePath = std::string(PSL_TESTFILES_DIR)+params[0];

        m_root = sofa::simulation::getSimulation()->load(scenePath.c_str());
        ASSERT_NE(m_root.get(), nullptr) << "Missing root node";

        m_root->init(sofa::core::ExecParams::defaultInstance()) ;
        TestResult* result = nullptr ;

        if(params[2] == "NoError" ){
            /// The scene must contains a TestResult object initialized to Success to indicate
            /// a failure
            m_root->getTreeObject(result) ;
            ASSERT_NE(result, nullptr) << "Missing component TestResult";

            ASSERT_EQ(result->m_result.getValueString(), sresult) ;
        }
    }
};

std::vector<std::vector<std::string>> testvalues = {
    {"test_emptyfile.psl", "Fail", "Error"},
    {"test_syntax_object.psl", "Success", "Error"},
    {"test_syntax_using.psl", "Success", "Error"},
    {"test_node.psl", "Success", "NoError"},
    {"test_node.pslx", "Success", "NoError"},
    {"test_node_fail.psl", "Fail", "NoError"},
    {"test_node_fail2.psl", "Fail", "NoError"},
    {"test_object.psl", "Success", "NoError"},
    {"test_object_datafield.psl", "Success", "NoError"},
    {"test_python.psl", "Success", "NoError"},
    {"test_pythondsl.psl", "Success", "NoError"},
    {"test_pythonlocals.psl", "Success", "NoError"},
    {"test_pythonglobals.psl", "Success", "NoError"},
    {"test_pythonobject.psl", "Success", "NoError"},
    {"test_pythonexcept.psl", "Success", "Error"},
    {"test_template.psl", "Success", "NoError"},
    {"test_template_frame.psl", "Success", "NoError"},
    {"test_template_kwargs.psl", "Success", "NoError"},
    {"test_template_empty.psl", "Success", "NoError"},
    {"test_import.psl", "Success", "NoError"},
    {"test_importpython.psl", "Success", "NoError"},
    {"test_properties.psl", "Success", "NoError"},
    {"test_properties_raw.psl", "Success", "NoError"},
    {"test_pslversion.psl", "Success", "NoError"},
    {"test_pslversion_invalid.psl", "Fail", "Error"}
};


TEST_P(PSL_test, checkTestFiles)
{
    checkTestFilesMsg(GetParam()) ;
}

INSTANTIATE_TEST_CASE_P(BaseTestSet,
                        PSL_test,
                        ::testing::ValuesIn(testvalues));


}
