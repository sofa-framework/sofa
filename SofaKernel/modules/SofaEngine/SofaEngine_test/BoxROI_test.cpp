/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaBaseMechanics/UniformMass.h>

#include <vector>
using std::vector ;

#include <string>
using std::string ;

#include <gtest/gtest.h>
using testing::Types;

#include <sofa/helper/BackTrace.h>
#include <SofaBaseMechanics/MechanicalObject.h>
using namespace sofa::defaulttype ;

#include <SofaEngine/BoxROI.h>
using sofa::component::engine::BoxROI ;

#include <SofaBaseMechanics/initBaseMechanics.h>
using sofa::component::initBaseMechanics ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::core::ExecParams ;
using sofa::component::container::MechanicalObject ;
using sofa::defaulttype::Vec3dTypes ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/helper/logging/Message.h>
using sofa::helper::logging::MessageDispatcher ;

#include <sofa/helper/logging/ClangMessageHandler.h>
using sofa::helper::logging::ClangMessageHandler ;

int initMessage(){
    MessageDispatcher::clearHandlers() ;
    MessageDispatcher::addHandler(new ClangMessageHandler()) ;
    return 0;
}
int messageInited = initMessage();


template <typename TDataType>
struct BoxROITest :  public ::testing::Test
{
    typedef BoxROI<TDataType> TheBoxROI;
    Simulation* m_simu  {nullptr} ;
    Node::SPtr m_root ;
    Node::SPtr m_node ;
    typename TheBoxROI::SPtr m_boxroi;

    virtual void SetUp()
    {
        initBaseMechanics();
        setSimulation( m_simu = new DAGSimulation() );
        m_root = m_simu->createNewGraph("root");
    }

    void TearDown()
    {
        if (m_root != NULL){
            m_simu->unload(m_root);
        }
    }

    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some one removed.
    void attributesTests(){
        m_node = m_root->createChild("node") ;
        m_boxroi = New< TheBoxROI >() ;
        m_node->addObject(m_boxroi) ;

        /// List of the supported attributes the user expect to find
        /// This list needs to be updated if you add an attribute.
        vector<string> attrnames = {
            "box",
            "position", "edges",  "triangles", "tetrahedra", "hexahedra", "quad",
            "computeEdges", "computeTriangles", "computeTetrahedra", "computeHexahedra", "computeQuad",
            "indices", "edgeIndices", "triangleIndices", "tetrahedronIndices", "hexahedronIndices",
            "quadIndices",
            "pointsInROI", "edgesInROI", "trianglesInROI", "tetrahedraInROI", "hexahedraInROI", "quadInROI",
            "nbIndices",
            "drawBoxes", "drawPoints", "drawEdges", "drawTriangles", "drawTetrahedra", "drawHexahedra", "drawQuads",
            "drawSize",
            "doUpdate"
        };

        for(auto& attrname : attrnames)
            EXPECT_NE( m_boxroi->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

        /// List of the attributes that are deprecated.
        vector<string> deprecatednames = {
              "pointsInBox", "edgesInBox", "f_trianglesInBox", "f_tetrahedraInBox", "f_tetrahedraInBox", "f_quadInBOX",
              "rest_position", "isVisible"
        };

        for(auto& attrname : deprecatednames)
            EXPECT_NE( m_boxroi->findData(attrname), nullptr ) << "Missing deprecated attribute with name '" << attrname << "'." ;

        return ;
    }
};


typedef Types<
    Vec3Types
> DataTypes;

TYPED_TEST_CASE(BoxROITest, DataTypes);


TYPED_TEST(BoxROITest, attributesTests) {
    ASSERT_NO_THROW(this->attributesTests()) ;
}

