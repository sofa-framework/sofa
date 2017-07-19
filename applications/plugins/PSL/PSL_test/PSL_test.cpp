/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

/******************************************************************************
 * Contributors:                                                              *
 *    - damien.marchal@univ-lille1.fr                                         *
 *****************************************************************************/

#include <vector>
#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test ;

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


class PSL_test : public Sofa_test<>,
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

    void checkTestFiles(const std::vector<std::string>& params)
    {
        EXPECT_MSG_NOEMIT(Error) ;
        std::string sresult = params[1];
        std::string scenePath = std::string(PSL_TESTFILES_DIR)+params[0];

        m_root = sofa::simulation::getSimulation()->load(scenePath.c_str());
        m_root->init(sofa::core::ExecParams::defaultInstance()) ;

        ASSERT_NE(m_root.get(), nullptr) << "Missing root node";
        TestResult* result = nullptr ;

        /// The scene must contains a TestResult object initialized to Success to indicate
        /// a failure
        m_root->getTreeObject(result) ;
        ASSERT_NE(result, nullptr) << "Missing component TestResult";

        ASSERT_EQ(result->m_result.getValueString(), sresult) ;
    }
};


std::vector<std::vector<std::string>> testvalues = {
    {"test_node.psl", "Success", "NoError"},
    {"test_node_fail.psl", "Fail", "NoError"},
    {"test_node_fail2.psl", "Fail", "NoError"},
    {"test_object.psl", "Success", "NoError"},
    {"test_python.psl", "Success", "NoError"},
    {"test_pythonlocals.psl", "Success", "NoError"},
    {"test_template.psl", "Success", "NoError"}
};

TEST_P(PSL_test, checkTestFiles)
{
    checkTestFiles(GetParam()) ;
}

INSTANTIATE_TEST_CASE_P(BaseTestSet,
                        PSL_test,
                        ::testing::ValuesIn(testvalues));


}
