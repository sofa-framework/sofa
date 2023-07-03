/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/mass/DiagonalMass.h>

using sofa::core::ExecParams ;

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/EdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/QuadSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.h>

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/simulation/graph/SimpleApi.h>

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <string>
using std::string ;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;


using namespace sofa::defaulttype;
using namespace sofa::component::topology::container::dynamic;

using sofa::core::objectmodel::New;
using sofa::core::objectmodel::BaseObject;
using sofa::component::mass::DiagonalMass;
using sofa::component::statecontainer::MechanicalObject;


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
class DiagonalMass_test : public BaseTest
{
public:
    typedef TDataTypes DataTypes;
    typedef TMassType MassType;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef typename DataTypes::Real Real;
    typedef typename type::vector<MassType> VecMass;
    typedef DiagonalMass<TDataTypes> TheDiagonalMass ;

    simulation::Simulation* simulation = nullptr;
    simulation::Node::SPtr root;
    simulation::Node::SPtr node;
    typename MechanicalObject<DataTypes>::SPtr mstate;
    typename DiagonalMass<DataTypes>::SPtr mass;

    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Grid");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");

        simulation = simulation::getSimulation();
        root = simulation::getSimulation()->createNewGraph("root");
    }

    void TearDown() override
    {
        if (root!=nullptr)
            sofa::simulation::node::unload(root);
    }

    void createSceneGraph(VecCoord positions, BaseObject::SPtr topologyContainer, BaseObject::SPtr geometryAlgorithms)
    {
        node = root->createChild("node");
        mstate = New<MechanicalObject<DataTypes> >();
        mstate->x = positions;
        node->addObject(mstate);
        node->addObject(topologyContainer);
        node->addObject(geometryAlgorithms);
        mass = New<DiagonalMass<DataTypes> >();
        mass->f_printLog.setValue(1.0);
        node->addObject(mass);
    }

    void check(MassType expectedTotalMass, const VecMass& expectedMass)
    {
        // Check that the mass vector has the right size.
        ASSERT_EQ(mstate->x.getValue().size(), mass->d_vertexMass.getValue().size());
        // Safety check...
        ASSERT_EQ(mstate->x.getValue().size(), expectedMass.size());

        // Check the total mass.
        EXPECT_NEAR(expectedTotalMass, mass->d_totalMass.getValue(), 1e-4);

        // Check the mass at each index.
        for (size_t i = 0 ; i < mstate->x.getValue().size() ; i++)
            EXPECT_NEAR(expectedMass[i], mass->d_vertexMass.getValue()[i], 1e-4);
    }

    void runTest(VecCoord positions, BaseObject::SPtr topologyContainer, BaseObject::SPtr geometryAlgorithms,
                 MassType expectedTotalMass, const VecMass& expectedMass)
    {
        createSceneGraph(positions, topologyContainer, geometryAlgorithms);
        sofa::simulation::node::initRoot(root.get());
        check(expectedTotalMass, expectedMass);
    }

    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some are removed.
    void checkAttributes(){
        static const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "    <MechanicalObject />                                                                       "
                "    <RegularGrid nx='2' ny='2' nz='2' xmin='0' xmax='2' ymin='0' ymax='2' zmin='0' zmax='2' /> "
                "    <HexahedronSetGeometryAlgorithms />                                                        "
                "   <DiagonalMass name='m_mass'/>                            "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        EXPECT_TRUE( mass->findData("vertexMass") != nullptr ) ;
        EXPECT_TRUE( mass->findData("totalMass") != nullptr ) ;
        EXPECT_TRUE( mass->findData("massDensity") != nullptr ) ;
        EXPECT_TRUE( mass->findData("computeMassOnRest") != nullptr ) ;

        EXPECT_TRUE( mass->findData("showGravityCenter") != nullptr ) ;
        EXPECT_TRUE( mass->findData("showAxisSizeFactor") != nullptr ) ;

        EXPECT_TRUE( mass->findData("fileMass") != nullptr ) ;

        // This one is an alias...
        EXPECT_TRUE( mass->findData("filename") != nullptr ) ;
    }


    void checkTotalMassFromMassDensity_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                          "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                    "
                "    <MechanicalObject />                                                                       "
                "    <RegularGrid nx='2' ny='2' nz='2' xmin='0' xmax='2' ymin='0' ymax='2' zmin='0' zmax='2' /> "
                "    <HexahedronSetGeometryAlgorithms />                                                        "
                "    <DiagonalMass name='m_mass' massDensity='1.0' />                                           "
                "</Node>                                                                                        " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_EQ( float(mass->getTotalMass()), 8 );
            EXPECT_EQ(float(mass->getMassDensity()), 1);

            const VecMass& vMasses = mass->d_vertexMass.getValue();
            EXPECT_EQ(float(vMasses[0]), 1.0);
            EXPECT_EQ(float(vMasses[1]), 1.0);
            EXPECT_EQ(float(vMasses[2]), 1.0);
            EXPECT_EQ(float(vMasses[3]), 1.0);
        }
    }

    void checkMassDensityFromTotalMass_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                          "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                    "
                "    <MechanicalObject />                                                                       "
                "    <RegularGrid nx='2' ny='2' nz='2' xmin='0' xmax='2' ymin='0' ymax='2' zmin='0' zmax='2' /> "
                "    <HexahedronSetGeometryAlgorithms/>                                                         "
                "    <DiagonalMass name='m_mass' totalMass='10'/>                                               "
                "</Node>                                                                                        " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 10);
            EXPECT_EQ(float(mass->getMassDensity()), 1.25);

            const VecMass& vMasses = mass->d_vertexMass.getValue();
            EXPECT_EQ(float(vMasses[0]), 1.25);
            EXPECT_EQ(float(vMasses[1]), 1.25);
            EXPECT_EQ(float(vMasses[2]), 1.25);
            EXPECT_EQ(float(vMasses[3]), 1.25);
        }
    }

    void checkTotalMassOverwritesMassDensity_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                          "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                    "
                "    <MechanicalObject />                                                                       "
                "    <RegularGrid nx='2' ny='2' nz='2' xmin='0' xmax='2' ymin='0' ymax='2' zmin='0' zmax='2' /> "
                "    <HexahedronSetGeometryAlgorithms />                                                        "
                "    <DiagonalMass name='m_mass' massDensity='1.0' totalMass='10'/>                             "
                "</Node>                                                                                        " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 10);
            EXPECT_EQ(float(mass->getMassDensity()), 1.25);
        }
    }

    void checkTotalMassFromMassDensity_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' massDensity='1.0'/>                                        "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 8);
            EXPECT_EQ(float(mass->getMassDensity()), 1);

            const VecMass& vMasses = mass->d_vertexMass.getValue();
            EXPECT_NEAR(float(vMasses[0]), 1.66667, 1e-4);
            EXPECT_EQ(float(vMasses[1]), 1.0);
            EXPECT_EQ(float(vMasses[2]), 1.0);
            EXPECT_NEAR(float(vMasses[3]), 0.333333, 1e-4);
        }
    }

    void checkTotalMassFromNegativeMassDensity_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' massDensity='-1.0'/>                                       "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 1);
            EXPECT_EQ(float(mass->getMassDensity()), 0.125);
        }
    }

    void checkMassDensityFromTotalMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' totalMass='10.0'/>                                         "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 10);
            EXPECT_EQ(float(mass->getMassDensity()), 1.25);

            const VecMass& vMasses = mass->d_vertexMass.getValue();
            EXPECT_NEAR(float(vMasses[0]), 2.08333, 1e-4);
            EXPECT_EQ(float(vMasses[1]), 1.25);
            EXPECT_EQ(float(vMasses[2]), 1.25);
            EXPECT_NEAR(float(vMasses[3]), 0.416667, 1e-4);
        }
    }

    void checkMassDensityFromNegativeTotalMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' totalMass='-10.0'/>                                        "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 1);
            EXPECT_EQ(float(mass->getMassDensity()), 0.125);
        }
    }

    void checkDoubleDeclaration_MassDensityTotalMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' massDensity='10.0' totalMass='10.0'/>                      "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 10);
            EXPECT_EQ(float(mass->getMassDensity()), 1.25);
        }
    }

    void checkDoubleDeclaration_NegativeMassDensityTotalMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' massDensity='-10.0' totalMass='10.0'/>                     "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 10);
            EXPECT_EQ(float(mass->getMassDensity()), 1.25);
        }
    }

    void checkDoubleDeclaration_MassDensityNegativeTotalMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' massDensity='10.0' totalMass='-10.0'/>                     "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 1);
            EXPECT_EQ(float(mass->getMassDensity()), 0.125);
        }
    }

    void checkMassDensityTotalMassFromVertexMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' vertexMass='2 2 2 2 2 2 2 2'/>                             "
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 16.0);
            EXPECT_EQ(float(mass->getMassDensity()), 2.0);

            const VecMass& vMasses = mass->d_vertexMass.getValue();
            EXPECT_EQ(float(vMasses[0]), 2);
            EXPECT_EQ(float(vMasses[1]), 2);
            EXPECT_EQ(float(vMasses[2]), 2);
            EXPECT_EQ(float(vMasses[3]), 2);
        }
    }

    void checkTotalMassFromNegativeMassDensityVertexMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' massDensity = '-1.0' vertexMass='2.08334 1.25 1.25 0.416667 0.416667 1.25 1.25 2.08333'/>"
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ(mass->getMassCount(), 8);
            EXPECT_EQ(float(mass->getTotalMass()), 1.0);
            EXPECT_EQ(float(mass->getMassDensity()), 0.125);
        }
    }

    void checkWrongSizeVertexMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' /> "
                "            <DiagonalMass name='m_mass' vertexMass='10 2.08334 1.25 1.25 0.416667 0.416667 1.25 1.25 2.08333'/>"
                "    </Node>                                                                                        "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 ) ;
            EXPECT_EQ( (float)mass->getMassDensity(), 0.125 ) ;
            EXPECT_EQ( (float)mass->getTotalMass(), 1.0 ) ;
        }
    }

    void checkAttributeLoadFromFile(const std::string& filename, int masscount, double totalMass, bool shouldFail)
    {

        std::stringstream scene;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                 "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                 "   <DiagonalMass name='m_mass' filename='"<< filename <<"'/>      "
                 "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.str().c_str());
        ASSERT_NE(root.get(), nullptr) ;

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(shouldFail)
        {
            EXPECT_MSG_EMIT(Error);
            root->init(sofa::core::execparams::defaultInstance());
            EXPECT_FALSE( mass->isComponentStateValid() );
        }else
        {
            EXPECT_MSG_NOEMIT(Error);
            root->init(sofa::core::execparams::defaultInstance()) ;
            EXPECT_TRUE( mass->isComponentStateValid() );
        }

        if(mass!=nullptr){
            // The number of mass in card.rigid is one so this should be
            // returned from the getMassCount()
            EXPECT_EQ( mass->getMassCount(), masscount ) ;

            // it indicate it has not been properly initialized.
            // the source code should be fixed.
            EXPECT_NE( mass->getTotalMass(), totalMass ) ;
        }
    }


    void checkTopologicalChanges_Hexa()
    {
        static const string scene =
            "<?xml version='1.0'?>                                                                              "
            "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
            "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
            "    <Node name='Hexa' >                                                                            "
            "            <MechanicalObject position = '@../grid.position' />                                    "
            "            <HexahedronSetTopologyContainer name='Container' src='@../grid' />                     "
            "            <HexahedronSetTopologyModifier name='Modifier' />                                      "
            "            <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                   "
            "            <DiagonalMass name='m_mass' massDensity='1.0'/>                                        "
            "    </Node>                                                                                        "
            "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>();
        ASSERT_NE(mass, nullptr);

        HexahedronSetTopologyModifier* modifier = root->getTreeObject<HexahedronSetTopologyModifier>();
        ASSERT_NE(modifier, nullptr);

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        static const Real refValue = Real(1.0 / 8.0);  // 0.125        

        // check value at init
        EXPECT_EQ(vMasses.size(), 27);
        EXPECT_NEAR(vMasses[0], refValue, 1e-4);
        EXPECT_NEAR(vMasses[1], refValue * 2, 1e-4);
        EXPECT_NEAR(mass->getTotalMass(), 8, 1e-4);
        
        sofa::type::vector<sofa::Index> hexaIds = { 0 };        
        // remove hexahedron id: 0
        modifier->removeHexahedra(hexaIds);
        EXPECT_EQ(vMasses.size(), 26);
        EXPECT_NEAR(vMasses[0], refValue, 1e-4); // check update of Mass when removing tetra
        EXPECT_NEAR(vMasses[1], refValue, 1e-4);

        EXPECT_NEAR(mass->getTotalMass(), 7.0, 1e-4);

        // remove hexahedron id: 0
        modifier->removeHexahedra(hexaIds);
        EXPECT_EQ(vMasses.size(), 25);
        EXPECT_NEAR(vMasses[0], refValue, 1e-4); // check update of Mass when removing tetra
        EXPECT_NEAR(vMasses[1], refValue, 1e-4);
        EXPECT_NEAR(mass->getTotalMass(), 6.0, 1e-4);

        hexaIds.push_back(1);
        // remove hexahedron id: 0, 1
        modifier->removeHexahedra(hexaIds);
        EXPECT_EQ(vMasses.size(), 21);
        EXPECT_NEAR(vMasses[0], refValue, 1e-4);
        EXPECT_NEAR(vMasses[1], refValue * 2, 1e-4);
        EXPECT_NEAR(mass->getTotalMass(), 4.0, 1e-4);

        hexaIds.push_back(2);
        hexaIds.push_back(3);
        // remove hexahedron id: 0, 1, 2, 3
        modifier->removeHexahedra(hexaIds);
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }


    void checkTopologicalChanges_Tetra()
    {
        static const string scene =
            "<?xml version='1.0'?>                                                                              "
            "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
            "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
            "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
            "    <Node name='Tetra' >                                                                           "
            "            <MechanicalObject position='@../grid.position' />                                      "
            "            <TetrahedronSetTopologyContainer name='Container' />                                   "
            "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
            "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
            "            <Hexa2TetraTopologicalMapping input='@../grid' output='@Container' />                  "
            "            <DiagonalMass name='m_mass' massDensity='1.0'/>                                        "
            "    </Node>                                                                                        "
            "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);
        
        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>();
        ASSERT_NE(mass, nullptr);

        TetrahedronSetTopologyModifier* modifier = root->getTreeObject<TetrahedronSetTopologyModifier>();        
        ASSERT_NE(modifier, nullptr);

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        static const Real refValue = Real(1.0/3.0);  //0.3333
        static const Real refValue2 = 2 - refValue; // 1.6667
       
        // check value at init
        EXPECT_EQ(vMasses.size(), 8);
        EXPECT_NEAR(vMasses[0], refValue2, 1e-4);
        EXPECT_NEAR(mass->getTotalMass(), 8.0, 1e-4);

        sofa::type::vector<sofa::Index> tetraIds = { 0 };
        // remove tetrahedron id: 0
        modifier->removeTetrahedra(tetraIds); 
        EXPECT_EQ(vMasses.size(), 8);
        EXPECT_NEAR(vMasses[0], refValue2 - refValue, 1e-4); // check update of Mass when removing tetra
        EXPECT_NEAR(mass->getTotalMass(), 8.0 - (4 * refValue), 1e-4);
        Real lastV = vMasses[7];
        
        // remove tetrahedron id: 0
        modifier->removeTetrahedra(tetraIds);
        EXPECT_EQ(vMasses.size(), 7);
        EXPECT_NEAR(vMasses[0], refValue2 - 2 *refValue, 1e-4); // check update of Mass when removing tetra
        EXPECT_NEAR(vMasses[4], lastV, 1e-4); // vertex 4 has been removed because isolated, check swap value
        EXPECT_NEAR(mass->getTotalMass(), 8.0 - (8 * refValue), 1e-4);

        tetraIds.push_back(1);
        // remove tetrahedron id: 0, 1
        modifier->removeTetrahedra(tetraIds);
        EXPECT_EQ(vMasses.size(), 6);
        EXPECT_NEAR(vMasses[0], refValue, 1e-4);
        EXPECT_NEAR(mass->getTotalMass(), 8.0 - (16 * refValue), 1e-4);

        // remove tetrahedron id: 0, 1
        modifier->removeTetrahedra(tetraIds); // remove tetra 0, 1
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }

    void checkTopologicalChanges_Quad()
    {
        static const string scene =
            "<?xml version='1.0'?>                                                                              "
            "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
            "    <RegularGridTopology name='grid' n='3 3 1' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
            "    <Node name='Quad' >                                                                            "
            "            <MechanicalObject position = '@../grid.position' />                                    "
            "            <QuadSetTopologyContainer name='Container' src='@../grid' />                     "
            "            <QuadSetTopologyModifier name='Modifier' />                                      "
            "            <QuadSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                   "
            "            <DiagonalMass name='m_mass' massDensity='1.0'/>                                        "
            "    </Node>                                                                                        "
            "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>();
        ASSERT_NE(mass, nullptr);

        QuadSetTopologyModifier* modifier = root->getTreeObject<QuadSetTopologyModifier>();
        ASSERT_NE(modifier, nullptr);

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        static const Real refValue = Real(1.0 / 4.0);  // 0.25
        static const Real initMass = mass->getTotalMass();

        // check value at init. Grid of 4 quads, should be first and third line: 0.25, 0.5, 0.25, second line: 0.5, 1.0, 0.5
        // 1/4 ---- 1/2 ---- 1/4
        //  |   m:1  |   m:1  |
        //  |  id:2  |  id:3  |
        // 1/2 ----  1  ---- 1/2
        //  |   m:1  |   m:1  |
        //  |  id:0  |  id:1  |
        // 1/4 ---- 1/2 ---- 1/4
        EXPECT_EQ(vMasses.size(), 9);  
        EXPECT_NEAR(vMasses[0], refValue, 1e-4);
        EXPECT_NEAR(vMasses[1], refValue * 2, 1e-4);
        EXPECT_NEAR(vMasses[4], refValue * 4, 1e-4);
        EXPECT_NEAR(initMass, 4, 1e-4);
        Real lastV = vMasses[8];
       
        
        // remove quad id: 0
        // 1/4 ---- 1/2 ---- 1/4
        //  |   m:1  |   m:1  |
        //  |  id:2  |  id:0  |
        // 1/4 ---- 3/4 ---- 1/2
        //           |   m:1  |
        //           |  id:1  |
        //          1/4 ---- 1/4
        sofa::type::vector<sofa::Index> ids = { 0 };
        modifier->removeQuads(ids, true, true);
        EXPECT_EQ(vMasses.size(), 8);
        EXPECT_NEAR(vMasses[0], lastV, 1e-4); // check update of Mass when removing quad, vMasses[0] is now mapped to point position 8.
        EXPECT_NEAR(vMasses[1], refValue, 1e-4); // one neighboord quad removed. 
        EXPECT_NEAR(vMasses[4], refValue * 3, 1e-4); // one neighboord quad removed. 

        EXPECT_NEAR(mass->getTotalMass(), initMass - 1, 1e-4);
        lastV = vMasses[7];

        // remove quad id: 0
        // 1/4 ---- 1/4
        //  |   m:1  |
        //  |  id:0  |
        // 1/4 ---- 2/4 ---- 1/4
        //           |   m:1  |
        //           |  id:1  |
        //          1/4 ---- 1/4
        modifier->removeQuads(ids, true, true);
        EXPECT_EQ(vMasses.size(), 7);
        EXPECT_NEAR(vMasses[0], lastV - refValue, 1e-4); // check swap value
        EXPECT_NEAR(vMasses[1], refValue, 1e-4);
        EXPECT_NEAR(vMasses[4], refValue * 2, 1e-4); // one neighboord quad removed. 
        EXPECT_NEAR(mass->getTotalMass(), initMass - 2, 1e-4);

        ids.push_back(1);
        // remove quad id: 0, 1
        modifier->removeQuads(ids, true, true);
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }


    void checkTopologicalChanges_Triangle()
    {
        static const string scene =
            "<?xml version='1.0'?>                                                                              "
            "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
            "    <RegularGridTopology name='grid' n='3 3 1' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
            "    <Node name='Triangle' >                                                                            "
            "            <MechanicalObject position = '@../grid.position' />                                    "
            "            <TriangleSetTopologyContainer name='Container' src='@../grid' />                     "
            "            <TriangleSetTopologyModifier name='Modifier' />                                      "
            "            <TriangleSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                   "
            "            <DiagonalMass name='m_mass' massDensity='1.0'/>                                        "
            "    </Node>                                                                                        "
            "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>();
        ASSERT_NE(mass, nullptr);

        TriangleSetTopologyModifier* modifier = root->getTreeObject<TriangleSetTopologyModifier>();
        ASSERT_NE(modifier, nullptr);

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        static const Real refValue = Real(1.0 / 3.0);  // 0.3333
        static const Real refValue2 = Real(1.0 / 2.0);  // 0.5
        const Real initMass = mass->getTotalMass();

        // check value at init
        EXPECT_EQ(vMasses.size(), 9);
        EXPECT_NEAR(vMasses[0], refValue, 1e-4);
        EXPECT_NEAR(vMasses[1], refValue2, 1e-4);
        EXPECT_NEAR(initMass, 4, 1e-4);

        sofa::type::vector<sofa::Index> ids = { 0 };
        // remove Triangle id: 0
        modifier->removeTriangles(ids, true, true);
        EXPECT_EQ(vMasses.size(), 9);
        EXPECT_NEAR(vMasses[0], refValue * 0.5, 1e-4); // check update of Mass when removing tetra
        EXPECT_NEAR(vMasses[1], refValue, 1e-4);
        EXPECT_NEAR(mass->getTotalMass(), initMass - refValue2, 1e-4);

        // remove Triangle id: 0
        modifier->removeTriangles(ids, true, true);
        EXPECT_EQ(vMasses.size(), 9);
        EXPECT_NEAR(vMasses[0], refValue * 0.5, 1e-4); // check swap value
        EXPECT_NEAR(vMasses[1], refValue, 1e-4);
        EXPECT_NEAR(mass->getTotalMass(), initMass - 2 * refValue2, 1e-4);

        ids.push_back(1);
        // remove Triangle id: 0, 1
        modifier->removeTriangles(ids, true, true);
        EXPECT_EQ(vMasses.size(), 7);
        EXPECT_NEAR(vMasses[0], refValue, 1e-4);
        EXPECT_NEAR(vMasses[1], refValue, 1e-4);
        EXPECT_NEAR(mass->getTotalMass(), initMass - 4 * refValue2, 1e-4);

        ids.push_back(2);
        ids.push_back(3);
        // remove Triangle id: 0, 1, 2, 3
        modifier->removeTriangles(ids, true, true);
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }

    void checkTopologicalChanges_Edge()
    {
        static const string scene =
            "<?xml version='1.0'?>                                                                              "
            "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
            "    <RegularGridTopology name='grid' n='4 1 1' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
            "    <Node name='Edge' >                                                                            "
            "            <MechanicalObject position = '@../grid.position' />                                    "
            "            <EdgeSetTopologyContainer name='Container' src='@../grid' />                     "
            "            <EdgeSetTopologyModifier name='Modifier' />                                      "
            "            <EdgeSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                   "
            "            <DiagonalMass name='m_mass' massDensity='1.0'/>                                        "
            "    </Node>                                                                                        "
            "</Node>                                                                                            ";

        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheDiagonalMass* mass = root->getTreeObject<TheDiagonalMass>();
        ASSERT_NE(mass, nullptr);

        EdgeSetTopologyModifier* modifier = root->getTreeObject<EdgeSetTopologyModifier>();
        ASSERT_NE(modifier, nullptr);

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        static const Real refValue = Real(2.0 / 3.0);  // Medge (length/(n-1)): 2/3
        static const Real refValue2 = Real(1.0 / 3.0);  // Mpoint = Medge/2
        const Real initMass = mass->getTotalMass();

        // check value at init
        EXPECT_EQ(vMasses.size(), 4);
        EXPECT_NEAR(vMasses[0], refValue2, 1e-4);
        EXPECT_NEAR(vMasses[1], refValue, 1e-4);
        EXPECT_NEAR(initMass, 2, 1e-4);

        sofa::type::vector<sofa::Index> ids = { 0 };
        // remove Edge id: 0
        modifier->removeEdges(ids, true);
        EXPECT_EQ(vMasses.size(), 3);
        EXPECT_NEAR(vMasses[0], refValue2, 1e-4); // check swap point
        EXPECT_NEAR(vMasses[1], refValue2, 1e-4); // check edge remove update
        EXPECT_NEAR(mass->getTotalMass(), initMass - refValue, 1e-4);

        // remove Edge id: 0
        modifier->removeEdges(ids, true);
        EXPECT_EQ(vMasses.size(), 2);
        EXPECT_NEAR(vMasses[0], refValue2, 1e-4); 
        EXPECT_NEAR(vMasses[1], refValue2, 1e-4);
        EXPECT_NEAR(mass->getTotalMass(), initMass - 2 * refValue, 1e-4);

        // remove Edge id: 0
        modifier->removeEdges(ids, true);
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }
};


typedef DiagonalMass_test<Vec3Types, Vec3Types::Real> DiagonalMass3_test;

TEST_F(DiagonalMass3_test, singleEdge)
{
    VecCoord positions;
    positions.push_back(Coord(0.0f, 0.0f, 0.0f));
    positions.push_back(Coord(0.0f, 1.0f, 0.0f));

    const EdgeSetTopologyContainer::SPtr topologyContainer = New<EdgeSetTopologyContainer>();
    topologyContainer->addEdge(0, 1);

    const EdgeSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
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

    const TriangleSetTopologyContainer::SPtr topologyContainer = New<TriangleSetTopologyContainer>();
    topologyContainer->addTriangle(0, 1, 2);

    const TriangleSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<TriangleSetGeometryAlgorithms<Vec3Types> >();

    const MassType expectedTotalMass = 1.0f;
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

    const QuadSetTopologyContainer::SPtr topologyContainer = New<QuadSetTopologyContainer>();
    topologyContainer->addQuad(0, 1, 2, 3);

    const QuadSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
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

    const TetrahedronSetTopologyContainer::SPtr topologyContainer = New<TetrahedronSetTopologyContainer>();
    topologyContainer->addTetra(0, 1, 2, 3);

    const TetrahedronSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<TetrahedronSetGeometryAlgorithms<Vec3Types> >();

    const MassType expectedTotalMass = 1.0f;
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

    const HexahedronSetTopologyContainer::SPtr topologyContainer = New<HexahedronSetTopologyContainer>();
    topologyContainer->addHexa(0, 1, 2, 3, 4, 5, 6, 7);

    const HexahedronSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
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

TEST_F(DiagonalMass3_test, checkTotalMassFromNegativeMassDensity_Tetra){
    checkTotalMassFromNegativeMassDensity_Tetra();
}

TEST_F(DiagonalMass3_test, checkMassDensityFromTotalMass_Tetra){
    checkMassDensityFromTotalMass_Tetra();
}

TEST_F(DiagonalMass3_test, checkMassDensityFromNegativeTotalMass_Tetra){
    checkMassDensityFromNegativeTotalMass_Tetra();
}

TEST_F(DiagonalMass3_test, checkDoubleDeclaration_MassDensityTotalMass_Tetra){
    checkDoubleDeclaration_MassDensityTotalMass_Tetra();
}

TEST_F(DiagonalMass3_test, checkDoubleDeclaration_NegativeMassDensityTotalMass_Tetra){
    checkDoubleDeclaration_NegativeMassDensityTotalMass_Tetra();
}

TEST_F(DiagonalMass3_test, checkDoubleDeclaration_MassDensityNegativeTotalMass_Tetra){
    checkDoubleDeclaration_MassDensityNegativeTotalMass_Tetra();
}

TEST_F(DiagonalMass3_test, checkMassDensityTotalMassFromVertexMass_Tetra){
    checkMassDensityTotalMassFromVertexMass_Tetra();
}

TEST_F(DiagonalMass3_test, checkTotalMassFromNegativeMassDensityVertexMass_Tetra){
    checkTotalMassFromNegativeMassDensityVertexMass_Tetra();
}

TEST_F(DiagonalMass3_test, checkWrongSizeVertexMass_Tetra){
    checkWrongSizeVertexMass_Tetra();
}


TEST_F(DiagonalMass3_test, checkTopologicalChanges_Hexa) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Hexa();
}

TEST_F(DiagonalMass3_test, checkTopologicalChanges_Tetra) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Tetra();
}

TEST_F(DiagonalMass3_test, checkTopologicalChanges_Quad) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Quad();
}

TEST_F(DiagonalMass3_test, checkTopologicalChanges_Triangle) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Triangle();
}

TEST_F(DiagonalMass3_test, checkTopologicalChanges_Edge) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Edge();
}


/// Rigid file are not handled only xs3....
TEST_F(DiagonalMass3_test, checkAttributeLoadFromXpsRigid){
    checkAttributeLoadFromFile("BehaviorModels/card.rigid", 0, 0, true);
}

TEST_F(DiagonalMass3_test, checkAttributeLoadFromXpsMassSpring){
    checkAttributeLoadFromFile("BehaviorModels/chain.xs3", 6, 0.6, false);
}


} // namespace sofa
