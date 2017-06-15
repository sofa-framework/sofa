/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaBaseMechanics/DiagonalMass.h>

#include <SofaBaseMechanics/initBaseMechanics.h>
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
using sofa::component::mass::DiagonalMass;
using sofa::component::container::MechanicalObject;


namespace sofa {

// Define a test for DiagonalMass that is somewhat generic.
//
// It creates a single-Node scene graph with a MechanicalObject, a DiagonalMass,
// and a GeometryAlgorithms as well as a TopologyContainer (both needed by
// DiagonalMass).
//
// Given the positions and the topology, it then checks the expected values for
// the mass.
template <class TDataTypes, class TMassType>
class DiagonalMass_test : public ::testing::Test
{
public:
    typedef TDataTypes DataTypes;
    typedef TMassType MassType;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename helper::vector<MassType> VecMass;
    typedef DiagonalMass<TDataTypes, TMassType> TheDiagonalMass ;

    simulation::Simulation* simulation;
    simulation::Node::SPtr root;
    simulation::Node::SPtr node;
    typename MechanicalObject<DataTypes>::SPtr mstate;
    typename DiagonalMass<DataTypes, MassType>::SPtr mass;

    virtual void SetUp()
    {
        component::initBaseMechanics();
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
        mass = New<DiagonalMass<DataTypes, MassType> >();
        node->addObject(mass);
    }

    void check(MassType expectedTotalMass, const VecMass& expectedMass)
    {
        // Check that the mass vector has the right size.
        ASSERT_EQ(mstate->x.getValue().size(), mass->d_mass.getValue().size());
        // Safety check...
        ASSERT_EQ(mstate->x.getValue().size(), expectedMass.size());

        // Check the total mass.
        EXPECT_FLOAT_EQ(expectedTotalMass, mass->d_totalMass.getValue());

        // Check the mass at each index.
        for (size_t i = 0 ; i < mstate->x.getValue().size() ; i++)
            EXPECT_FLOAT_EQ(expectedMass[i], mass->d_mass.getValue()[i]);
    }

    void runTest(VecCoord positions, BaseObject::SPtr topologyContainer, BaseObject::SPtr geometryAlgorithms,
                 MassType expectedTotalMass, const VecMass& expectedMass)
    {
        createSceneGraph(positions, topologyContainer, geometryAlgorithms);
        simulation::getSimulation()->init(root.get());
        check(expectedTotalMass, expectedMass);
    }

    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some are removed.
    void checkAttributes(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <DiagonalMass name='m_mass'/>                            "
                "</Node>                                                     " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        EXPECT_TRUE( mass->findData("mass") != nullptr ) ;
        EXPECT_TRUE( mass->findData("totalMass") != nullptr ) ;
        EXPECT_TRUE( mass->findData("massDensity") != nullptr ) ;
        EXPECT_TRUE( mass->findData("computeMassOnRest") != nullptr ) ;

        EXPECT_TRUE( mass->findData("showGravityCenter") != nullptr ) ;
        EXPECT_TRUE( mass->findData("showAxisSizeFactor") != nullptr ) ;

        EXPECT_TRUE( mass->findData("fileMass") != nullptr ) ;

        // This one is an alias...
        EXPECT_TRUE( mass->findData("filename") != nullptr ) ;

        return ;
    }


    void checkTotalMassFromMassDensity_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                          "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                    "
                "    <MechanicalObject />                                                                       "
                "    <RegularGrid nx='2' ny='2' nz='2' xmin='0' xmax='2' ymin='0' ymax='2' zmin='0' zmax='2' /> "
                "    <HexahedronSetGeometryAlgorithms />                                                        "
                "    <DiagonalMass name='m_mass' massDensity='1.0' />                                           "
                "</Node>                                                                                        " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 8 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
        }

        return ;
    }

    void checkMassDensityFromTotalMass_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                          "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                    "
                "    <MechanicalObject />                                                                       "
                "    <RegularGrid nx='2' ny='2' nz='2' xmin='0' xmax='2' ymin='0' ymax='2' zmin='0' zmax='2' /> "
                "    <HexahedronSetGeometryAlgorithms/>                                                         "
                "    <DiagonalMass name='m_mass' totalMass='10'/>                                               "
                "</Node>                                                                                        " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        root->init(ExecParams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getMassDensity(), 1.25 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
        }

        return ;
    }

    void checkTotalMassOverwritesMassDensity_Hexa(){
        string scene =
                "<?xml version='1.0'?>                                                                          "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                    "
                "    <MechanicalObject />                                                                       "
                "    <RegularGrid nx='2' ny='2' nz='2' xmin='0' xmax='2' ymin='0' ymax='2' zmin='0' zmax='2' /> "
                "    <HexahedronSetGeometryAlgorithms />                                                        "
                "    <DiagonalMass name='m_mass' massDensity='1.0' totalMass='10'/>                             "
                "</Node>                                                                                        " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;
        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getMassDensity(), 1.25 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
        }

        return ;
    }

    void checkTotalMassFromMassDensity_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                  "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' massDensity='1.0'/>                                        "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;
        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 8 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
        }

        return ;
    }

    void checkMassDensityFromTotalMass_Tetra(){
        string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetTopologyAlgorithms template='Vec3d' name='TopoAlgo' />                  "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' totalMass='10.0'/>                                        "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getMassDensity(), 1.25 ) ; //casting in float seems due to HexahedronSetGeometryAlgorithms
        }

        return ;
    }

    void checkAttributeLoadFromFile(const std::string& filename, int masscount, double totalMass){
        std::stringstream scene;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                 "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                 "   <DiagonalMass name='m_mass' filename='"<< filename <<"'/>      "
                 "</Node>                                                     " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;
        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            // The number of mass in card.rigid is one so this should be
            // returned from the getMassCount()
            EXPECT_EQ( mass->getMassCount(), masscount ) ;

            // it indicate it has not been properly initialized.
            // the source code should be fixed.
            EXPECT_NE( mass->getTotalMass(), totalMass ) ;
        }

        return ;
    }
};


typedef DiagonalMass_test<Vec3Types, Vec3Types::Real> DiagonalMass3_test;

TEST_F(DiagonalMass3_test, singleEdge)
{
    VecCoord positions;
    positions.push_back(Coord(0.0f, 0.0f, 0.0f));
    positions.push_back(Coord(0.0f, 1.0f, 0.0f));

    EdgeSetTopologyContainer::SPtr topologyContainer = New<EdgeSetTopologyContainer>();
    topologyContainer->addEdge(0, 1);

    EdgeSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<EdgeSetGeometryAlgorithms<Vec3Types> >();

    const MassType expectedTotalMass = 1.0f;
    const VecMass expectedMass(2, (MassType)(expectedTotalMass/2));

    runTest(positions,
            topologyContainer,
            geometryAlgorithms,
            expectedTotalMass,
            expectedMass);
}

TEST_F(DiagonalMass3_test, singleTriangle)
{
    VecCoord positions;
    positions.push_back(Coord(0.0f, 0.0f, 0.0f));
    positions.push_back(Coord(1.0f, 0.0f, 0.0f));
    positions.push_back(Coord(0.0f, 1.0f, 0.0f));

    TriangleSetTopologyContainer::SPtr topologyContainer = New<TriangleSetTopologyContainer>();
    topologyContainer->addTriangle(0, 1, 2);

    TriangleSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<TriangleSetGeometryAlgorithms<Vec3Types> >();

    const MassType expectedTotalMass = 0.5f;
    const VecMass expectedMass(3, (MassType)(expectedTotalMass/3));

    runTest(positions,
            topologyContainer,
            geometryAlgorithms,
            expectedTotalMass,
            expectedMass);
}

TEST_F(DiagonalMass3_test, singleQuad)
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
    const VecMass expectedMass(4, (MassType)(expectedTotalMass/4));

    runTest(positions,
            topologyContainer,
            geometryAlgorithms,
            expectedTotalMass,
            expectedMass);
}

TEST_F(DiagonalMass3_test, singleTetrahedron)
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

    const MassType expectedTotalMass = 1.0f/6.0f;
    const VecMass expectedMass(4, (MassType)(expectedTotalMass/4));

    runTest(positions,
            topologyContainer,
            geometryAlgorithms,
            expectedTotalMass,
            expectedMass);
}

TEST_F(DiagonalMass3_test, singleHexahedron)
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
    const VecMass expectedMass(8, (MassType)(expectedTotalMass/8));

    runTest(positions,
            topologyContainer,
            geometryAlgorithms,
            expectedTotalMass,
            expectedMass);
}

TEST_F(DiagonalMass3_test, checkAttributes){
    checkAttributes() ;
}

TEST_F(DiagonalMass3_test, checkTotalMassFromMassDensity_Hexa){
    checkTotalMassFromMassDensity_Hexa();
}

TEST_F(DiagonalMass3_test, checkMassDensityFromTotalMass_Hexa){
    checkMassDensityFromTotalMass_Hexa();
}

TEST_F(DiagonalMass3_test, checkTotalMassOverwritesMassDensity_Hexa){
    checkTotalMassOverwritesMassDensity_Hexa();
}

TEST_F(DiagonalMass3_test, checkTotalMassFromMassDensity_Tetra){
    checkTotalMassFromMassDensity_Tetra();
}

TEST_F(DiagonalMass3_test, checkMassDensityFromTotalMass_Tetra){
    checkMassDensityFromTotalMass_Tetra();
}

/// Rigid file are not handled only xs3....
TEST_F(DiagonalMass3_test, checkAttributeLoadFromXps){
    checkAttributeLoadFromFile("BehaviorModels/card.rigid", 0, 0);
}

TEST_F(DiagonalMass3_test, checkAttributeLoadFromFile){
    checkAttributeLoadFromFile("BehaviorModels/chain.xs3", 6, 0.6);
}


} // namespace sofa
