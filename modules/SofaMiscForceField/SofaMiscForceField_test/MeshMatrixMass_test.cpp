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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaMiscForceField/MeshMatrixMass.h>

#include <SofaMiscForceField/initMiscForcefield.h>
using sofa::core::ExecParams ;

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetTopologyContainer.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/helper/testing/BaseTest.h>

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <string>
using std::string ;

#include <gtest/gtest.h>

using namespace sofa::defaulttype;
using namespace sofa::component::topology;

using sofa::core::objectmodel::New;
using sofa::core::objectmodel::BaseObject;
using sofa::component::mass::MeshMatrixMass;
using sofa::component::container::MechanicalObject;


namespace sofa {

// Define a test for MeshMatrixMass that is somewhat generic.
//
// It creates a single-Node scene graph with a MechanicalObject, a MeshMatrixMass,
// and a GeometryAlgorithms as well as a TopologyContainer (both needed by
// MeshMatrixMass).
//
// Given the positions and the topology, it then checks the expected values for
// the mass.
template <class TDataTypes, class TMassType>
class MeshMatrixMass_test : public ::testing::Test
{
public:
    typedef TDataTypes DataTypes;
    typedef TMassType MassType;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename helper::vector<MassType> VecMass;
    typedef MeshMatrixMass<TDataTypes, TMassType> TheMeshMatrixMass ;

    simulation::Simulation* simulation;
    simulation::Node::SPtr root;
    simulation::Node::SPtr node;
    typename MechanicalObject<DataTypes>::SPtr mstate;
    typename MeshMatrixMass<DataTypes, MassType>::SPtr mass;

    virtual void SetUp()
    {
        component::initMiscForcefield();
        simulation::setSimulation(simulation = new simulation::graph::DAGSimulation());
        root = simulation::getSimulation()->createNewGraph("root");
    }

    void TearDown()
    {
        if (root!=NULL)
            simulation::getSimulation()->unload(root);
    }

    void createSceneGraph(VecCoord positions, BaseObject::SPtr topologyContainer, BaseObject::SPtr geometryAlgorithms)
    {
        node = root->createChild("node");
        mstate = New<MechanicalObject<DataTypes> >();
        mstate->x = positions;
        node->addObject(mstate);
        node->addObject(topologyContainer);
        node->addObject(geometryAlgorithms);
        mass = New<MeshMatrixMass<DataTypes, MassType> >();
        node->addObject(mass);
    }

    void check(MassType expectedTotalMass, const VecMass& expectedMass)
    {
        // Check that the mass vector has the right size.
        ASSERT_EQ(mstate->x.getValue().size(), mass->d_vertexMassInfo.getValue().size());
        // Safety check...
        ASSERT_EQ(mstate->x.getValue().size(), expectedMass.size());

        // Check the total mass.
        EXPECT_FLOAT_EQ(expectedTotalMass, mass->d_totalMass.getValue());

        // Check the mass at each index.
        for (size_t i = 0 ; i < mstate->x.getValue().size() ; i++)
            EXPECT_FLOAT_EQ(expectedMass[i], mass->d_vertexMassInfo.getValue()[i]);
    }

    void runTest(VecCoord positions, BaseObject::SPtr topologyContainer, BaseObject::SPtr geometryAlgorithms,
                 MassType expectedTotalMass, const VecMass& expectedMass)
    {
        createSceneGraph(positions, topologyContainer, geometryAlgorithms);
        simulation::getSimulation()->init(root.get());
        check(expectedTotalMass, expectedMass);
    }


    //---------------------------------------------------------------
    // HEXA topology
    //---------------------------------------------------------------
    void check_DefaultAttributes_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' />                                                               "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        EXPECT_TRUE( mass->findData("vertexMass") != nullptr ) ;
        EXPECT_TRUE( mass->findData("totalMass") != nullptr ) ;
        EXPECT_TRUE( mass->findData("massDensity") != nullptr ) ;

        EXPECT_TRUE( mass->findData("showGravityCenter") != nullptr ) ;
        EXPECT_TRUE( mass->findData("showAxisSizeFactor") != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.05 ) ;
            EXPECT_EQ( mass->getVertexMass()[7], 0.05 ) ;
        }
        return ;
    }


    void check_TotalMass_Initialization_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' totalMass='2.0' />                                               "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 2.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.25 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.0125 ) ;
            EXPECT_EQ( mass->getVertexMass()[1], 0.025 ) ;
        }

        return ;
    }


    void check_MassDensity_Initialization_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0' />                                             "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 8.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 1.0 ) ;
            EXPECT_EQ( (float)mass->getVertexMass()[0], (float)0.05 ) ;
        }

        return ;
    }


    void check_VertexMass_Lumping_Initialization_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1'                                                      "
                "               vertexMass='1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1' />               "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true ) ;
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 27.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 3.375 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 1.0 ) ;
        }

        return ;
    }


    /// Check for the definition of two concurrent input data
    /// first, the totalMass info is used if existing (the most generic one)
    /// second, the massDensity
    /// at last the vertexMass info

    void check_DoubleDeclaration_TotalMassAndMassDensity_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0' totalMass='2.0' />                             "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 2.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.25 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.0125 ) ;
        }

        return ;
    }


    void check_DoubleDeclaration_TotalMassAndVertexMass_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1' totalMass='2.0'                                      "
                "               vertexMass='1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1' />               "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true ) ;
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 2.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.25 ) ;
            EXPECT_EQ( (float)mass->getVertexMass()[0], (float)0.0125 ) ;
        }

        return ;
    }


    void check_DoubleDeclaration_MassDensityAndVertexMass_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1' massDensity='1.0'                                    "
                "               vertexMass='1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1' />               "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true ) ;
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 8.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 1.0 ) ;
            EXPECT_EQ( (float)mass->getVertexMass()[0], (float)0.05 ) ;
        }

        return ;
    }


    /// Check wrong input values, in all cases the Mass is initialized
    /// using the default totalMass value = 1.0

    void check_TotalMass_WrongValue_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' totalMass='-2.0' />                                              "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.00625 ) ;
        }

        return ;
    }


    void check_MassDensity_WrongValue_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='-1.0' />                                            "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.00625 ) ;
        }

        return ;
    }


    void check_MassDensity_WrongSize_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0 4.0' />                                         "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.00625 ) ;
        }

        return ;
    }


    void check_VertexMass_WrongValue_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1'                                                      "
                "               vertexMass='1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1' />              "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.00625 ) ;
        }

        return ;
    }


    void check_VertexMass_WrongSize_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1' vertexMass='1 2' />                                  "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.00625 ) ;
        }

        return ;
    }


    /// Check coupling of wrong data values/size and concurrent data

    void check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0' totalMass='-2.0' />                            "
                "</Node>                                                                                            " ;

        /// Here : default totalMass value will be used since negative value is given

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.00625 ) ;
        }

        return ;
    }


    void check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0 1.0' totalMass='2.0' />                         "
                "</Node>                                                                                            " ;

        /// Here : totalMass value will be used due to concurrent data

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 2.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.25 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 0.0125 ) ;
        }

        return ;
    }





    //---------------------------------------------------------------
    // TETRA topology
    //---------------------------------------------------------------
    void check_DefaultAttributes_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' />                                                           "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        EXPECT_TRUE( mass->findData("vertexMass") != nullptr ) ;
        EXPECT_TRUE( mass->findData("totalMass") != nullptr ) ;
        EXPECT_TRUE( mass->findData("massDensity") != nullptr ) ;

        EXPECT_TRUE( mass->findData("showGravityCenter") != nullptr ) ;
        EXPECT_TRUE( mass->findData("showAxisSizeFactor") != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.25/3.0) ) ;
            EXPECT_EQ( mass->getVertexMass()[7], (0.25/3.0) ) ;
        }
        return ;
    }


    void check_TotalMass_Initialization_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' totalMass='2.0'/>                                            "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 2.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.25 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.5/3.0) ) ;
            EXPECT_EQ( mass->getVertexMass()[1], 0.1 ) ;
        }

        return ;
    }


    void check_MassDensity_Initialization_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0' />                                         "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 8.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 1.0 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (2.0/3.0) ) ;
        }

        return ;
    }


    void check_VertexMass_Lumping_Initialization_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' lumping='1' vertexMass='1 1 1 1 1 1 1 1' />                  "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true ) ;
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 8.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 1.0 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], 1.0 ) ;
        }

        return ;
    }


    /// Check for the definition of two concurrent input data
    /// first, the totalMass info is used if existing (the most generic one)
    /// second, the massDensity
    /// at last the vertexMass info

    void check_DoubleDeclaration_TotalMassAndMassDensity_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0' totalMass='2.0' />                         "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 2.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.25 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.5/3.0) ) ;
        }

        return ;
    }


    void check_DoubleDeclaration_TotalMassAndVertexMass_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' lumping='1' vertexMass='1 1 1 1 1 1 1 1' totalMass='2.0'/>   "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true ) ;
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 2.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.25 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.5/3.0) ) ;
        }

        return ;
    }


    void check_DoubleDeclaration_MassDensityAndVertexMass_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' vertexMass='1 1 1 1 1 1 1 1' lumping='1' massDensity='1.0' />"
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true ) ;
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 8.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 1.0 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (2.0/3.0) ) ;
        }

        return ;
    }


    /// Check wrong input values, in all cases the Mass is initialized
    /// using the default totalMass value = 1.0

    void check_TotalMass_WrongValue_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' totalMass='-2.0'/>                                           "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.25/3.0) ) ;
        }

        return ;
    }


    void check_MassDensity_WrongValue_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='-1.0'/>                                         "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.25/3.0) ) ;
        }

        return ;
    }


    void check_MassDensity_WrongSize_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0 4.0'/>                                      "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.25/3.0) ) ;
        }

        return ;
    }


    void check_VertexMass_WrongValue_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' vertexMass='1 -1 1 1 1 1 1 1' lumping='1' />                 "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.25/3.0) ) ;
        }

        return ;
    }


    void check_VertexMass_WrongSize_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' vertexMass='1.0 2.0' lumping='1' />                          "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";
        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.25/3.0) ) ;
        }

        return ;
    }


    /// Check coupling of wrong data values/size and concurrent data

    void check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0' totalMass='-2.0' />                        "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        /// Here : default totalMass value will be used since negative value is given

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.125 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.25/3.0) ) ;
        }

        return ;
    }


    void check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                      "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0 1.0' totalMass='2.0' />                     "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        /// Here : totalMass value will be used due to concurrent data

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 2.0 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
            EXPECT_EQ( (float)mass->getMassDensity()[0], 0.25 ) ;
            EXPECT_EQ( mass->getVertexMass()[0], (0.5/3.0) ) ;
        }

        return ;
    }

}
;


typedef MeshMatrixMass_test<Vec3Types, Vec3Types::Real> MeshMatrixMass3_test;

TEST_F(MeshMatrixMass3_test, singleTriangle)
{
    VecCoord positions;
    positions.push_back(Coord(0.0f, 0.0f, 0.0f));
    positions.push_back(Coord(1.0f, 0.0f, 0.0f));
    positions.push_back(Coord(0.0f, 1.0f, 0.0f));

    TriangleSetTopologyContainer::SPtr topologyContainer = New<TriangleSetTopologyContainer>();
    topologyContainer->addTriangle(0, 1, 2);

    TriangleSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<TriangleSetGeometryAlgorithms<Vec3Types> >();

    const MassType expectedTotalMass = 1.0f;
    const VecMass expectedMass(3, (MassType)(expectedTotalMass/(3.0*2.0)));

    runTest(positions,
            topologyContainer,
            geometryAlgorithms,
            expectedTotalMass,
            expectedMass);
}

TEST_F(MeshMatrixMass3_test, singleQuad)
{
    VecCoord positions;
    positions.push_back(Coord(0.0f, 0.0f, 0.0f));
    positions.push_back(Coord(0.0f, 1.0f, 0.0f));
    positions.push_back(Coord(1.0f, 1.0f, 0.0f));
    positions.push_back(Coord(1.0f, 0.0f, 0.0f));

    QuadSetTopologyContainer::SPtr topologyContainer = New<QuadSetTopologyContainer>();
    topologyContainer->addQuad(0, 1, 2, 3);

    QuadSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<QuadSetGeometryAlgorithms<Vec3Types> >();

    const MassType expectedTotalMass = 1.0f;
    const VecMass expectedMass(4, (MassType)(expectedTotalMass/(4.0*2.0)));

    runTest(positions,
            topologyContainer,
            geometryAlgorithms,
            expectedTotalMass,
            expectedMass);
}

TEST_F(MeshMatrixMass3_test, singleTetrahedron)
{
    VecCoord positions;
    positions.push_back(Coord(0.0f, 0.0f, 0.0f));
    positions.push_back(Coord(0.0f, 1.0f, 0.0f));
    positions.push_back(Coord(1.0f, 0.0f, 0.0f));
    positions.push_back(Coord(0.0f, 0.0f, 1.0f));

    TetrahedronSetTopologyContainer::SPtr topologyContainer = New<TetrahedronSetTopologyContainer>();
    topologyContainer->addTetra(0, 1, 2, 3);

    TetrahedronSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<TetrahedronSetGeometryAlgorithms<Vec3Types> >();

    const MassType expectedTotalMass = 1.0f;
    const VecMass expectedMass(4, (MassType)(expectedTotalMass/(4.0*2.5)));

    runTest(positions,
            topologyContainer,
            geometryAlgorithms,
            expectedTotalMass,
            expectedMass);
}

TEST_F(MeshMatrixMass3_test, singleHexahedron)
{
    VecCoord positions;
    positions.push_back(Coord(0.0f, 0.0f, 0.0f));
    positions.push_back(Coord(1.0f, 0.0f, 0.0f));
    positions.push_back(Coord(1.0f, 1.0f, 0.0f));
    positions.push_back(Coord(0.0f, 1.0f, 0.0f));
    positions.push_back(Coord(0.0f, 0.0f, 1.0f));
    positions.push_back(Coord(1.0f, 0.0f, 1.0f));
    positions.push_back(Coord(1.0f, 1.0f, 1.0f));
    positions.push_back(Coord(0.0f, 1.0f, 1.0f));

    HexahedronSetTopologyContainer::SPtr topologyContainer = New<HexahedronSetTopologyContainer>();
    topologyContainer->addHexa(0, 1, 2, 3, 4, 5, 6, 7);

    HexahedronSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<HexahedronSetGeometryAlgorithms<Vec3Types> >();

    const MassType expectedTotalMass = 1.0f;
    const VecMass expectedMass(8, (MassType)(expectedTotalMass/(8.0*2.5)));

    runTest(positions,
            topologyContainer,
            geometryAlgorithms,
            expectedTotalMass,
            expectedMass);
}

TEST_F(MeshMatrixMass3_test, check_DefaultAttributes_Hexa){
    check_DefaultAttributes_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_TotalMass_Initialization_Hexa){
    check_TotalMass_Initialization_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_Initialization_Hexa){
    check_MassDensity_Initialization_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_Lumping_Initialization_Hexa){
    check_VertexMass_Lumping_Initialization_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_Hexa){
    check_DoubleDeclaration_TotalMassAndMassDensity_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndVertexMass_Hexa){
    check_DoubleDeclaration_TotalMassAndVertexMass_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_MassDensityAndVertexMass_Hexa){
    check_DoubleDeclaration_MassDensityAndVertexMass_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_TotalMass_WrongValue_Hexa){
    check_TotalMass_WrongValue_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_WrongValue_Hexa){
    check_MassDensity_WrongValue_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_WrongSize_Hexa){
    check_MassDensity_WrongSize_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_WrongValue_Hexa){
    check_VertexMass_WrongValue_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_WrongSize_Hexa){
    check_VertexMass_WrongSize_Hexa() ;
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Hexa){
    check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Hexa() ;
}

TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Hexa){
    check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Hexa() ;
}

TEST_F(MeshMatrixMass3_test, check_DefaultAttributes_Tetra){
    check_DefaultAttributes_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_TotalMass_Initialization_Tetra){
    check_TotalMass_Initialization_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_Initialization_Tetra){
    check_MassDensity_Initialization_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_Lumping_Initialization_Tetra){
    check_VertexMass_Lumping_Initialization_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_Tetra){
    check_DoubleDeclaration_TotalMassAndMassDensity_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndVertexMass_Tetra){
    check_DoubleDeclaration_TotalMassAndVertexMass_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_MassDensityAndVertexMass_Tetra){
    check_DoubleDeclaration_MassDensityAndVertexMass_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_TotalMass_WrongValue_Tetra){
    check_TotalMass_WrongValue_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_WrongValue_Tetra){
    check_MassDensity_WrongValue_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_WrongSize_Tetra){
    check_MassDensity_WrongSize_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_WrongValue_Tetra){
    check_VertexMass_WrongValue_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_WrongSize_Tetra){
    check_VertexMass_WrongSize_Tetra() ;
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Tetra){
    check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Tetra() ;
}

TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Tetra){
    check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Tetra() ;
}




} // namespace sofa
