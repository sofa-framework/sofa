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
#include <sofa/component/mass/MeshMatrixMass.h>
using sofa::core::execparams::defaultInstance; 

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/testing/NumericTest.h>

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
using sofa::component::mass::MeshMatrixMass;
using sofa::component::statecontainer::MechanicalObject;


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
class MeshMatrixMass_test : public BaseTest

{
public:
    typedef TDataTypes DataTypes;
    typedef TMassType MassType;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename type::vector<MassType> VecMass;
    typedef MeshMatrixMass<TDataTypes> TheMeshMatrixMass ;

    simulation::Simulation* simulation = nullptr;
    simulation::Node::SPtr root;
    simulation::Node::SPtr node;
    typename MechanicalObject<DataTypes>::SPtr mstate;
    typename MeshMatrixMass<DataTypes>::SPtr mass;

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
        mass = New<MeshMatrixMass<DataTypes> >();
        node->addObject(mass);
    }

    void check(MassType expectedTotalMass, MassType expectedMassDensity, const VecMass& expectedVMass, const VecMass& expectedEMass)
    {
        // Check that the mass vector has the right size.
        if (mstate != nullptr)
        {
            ASSERT_EQ(mstate->x.getValue().size(), mass->d_vertexMass.getValue().size());
        }

        // Check the total mass.
        EXPECT_FLOATINGPOINT_EQ(expectedTotalMass, mass->d_totalMass.getValue());

        // Check mass density
        EXPECT_FLOATINGPOINT_EQ(expectedMassDensity, mass->getMassDensity()[0]);

        // Check the mass at each index.
        auto vertexMass = mass->d_vertexMass.getValue();
        ASSERT_EQ(expectedVMass.size(), vertexMass.size());

        for (size_t i = 0 ; i < vertexMass.size(); i++)
            EXPECT_FLOATINGPOINT_EQ(expectedVMass[i], vertexMass[i]);

        // Check edge mass 
        auto edgeMass = mass->d_edgeMass.getValue();
        ASSERT_EQ(expectedEMass.size(), edgeMass.size());

        for (size_t i = 0; i < edgeMass.size(); i++) {
            if (edgeMass[i] != 0.0) // == 0 is possible if edge is not part of the element structure (for example in grid)
                EXPECT_FLOATINGPOINT_EQ(expectedEMass[i], edgeMass[i]);
        }
    }

    void runTest(VecCoord positions, BaseObject::SPtr topologyContainer, BaseObject::SPtr geometryAlgorithms,
                 MassType expectedTotalMass, MassType expectedMassDensity, const VecMass& expectedVMass, const VecMass& expectedEMass)
    {
        createSceneGraph(positions, topologyContainer, geometryAlgorithms);
        sofa::simulation::node::initRoot(root.get());
        check(expectedTotalMass, expectedMassDensity, expectedVMass, expectedEMass);
    }


    //---------------------------------------------------------------
    // HEXA topology
    //---------------------------------------------------------------
    void check_DefaultAttributes_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' />                                                               "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
    
        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        EXPECT_TRUE( mass->findData("vertexMass") != nullptr );
        EXPECT_TRUE( mass->findData("totalMass") != nullptr );
        EXPECT_TRUE( mass->findData("massDensity") != nullptr );

        EXPECT_TRUE( mass->findData("showGravityCenter") != nullptr );
        EXPECT_TRUE( mass->findData("showAxisSizeFactor") != nullptr );

        if(mass!=nullptr){
            static const MassType volume = 8.0;
            static const MassType expectedTotalMass = 1.0f;
            static const MassType expectedDensity = expectedTotalMass / volume;
            static const VecMass expectedVMass(8, (MassType)(expectedDensity * volume * 1 / 20));
            static const VecMass expectedEMass(18, (MassType)(expectedDensity * volume * 1 / 40));

            check(expectedTotalMass, expectedDensity, expectedVMass, expectedEMass);
        }
        return ;
    }


    void check_TotalMass_Initialization_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' totalMass='2.0' />                                               "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
    
        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            static const MassType volume = 8.0;
            static const MassType volumeElem = volume / 8.0; // 8 hexa in the grid
            static const MassType expectedTotalMass = 2.0f;
            static const MassType expectedDensity = expectedTotalMass / volume;

            EXPECT_EQ( mass->d_vertexMass.getValue().size(), 27 );
            EXPECT_EQ( mass->d_edgeMass.getValue().size(), 90);

            EXPECT_FLOATINGPOINT_EQ( mass->getTotalMass(), expectedTotalMass);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], expectedDensity);

            EXPECT_FLOATINGPOINT_EQ(mass->d_vertexMass.getValue()[0], (MassType)(expectedDensity * volumeElem * 1 / 20));
            EXPECT_FLOATINGPOINT_EQ(mass->d_vertexMass.getValue()[1], (MassType)(expectedDensity * volumeElem * 1 / 20 * 2)); // vertex shared by 2 hexa
            EXPECT_FLOATINGPOINT_EQ(mass->d_edgeMass.getValue()[0], (MassType)(expectedDensity * volumeElem * 1 / 40 * 2)); // Edge shared by 2 hexa
            EXPECT_FLOATINGPOINT_EQ(mass->d_edgeMass.getValue()[2], (MassType)(expectedDensity * volumeElem * 1 / 40)); 
        }

        return ;
    }


    void check_MassDensity_Initialization_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0' />                                             "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            static const MassType volume = 8.0;
            static const MassType volumeElem = volume / 8.0; // 8 hexa in the grid            
            static const MassType expectedDensity = 1.0;
            static const MassType expectedTotalMass = expectedDensity * volume;

            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_EQ( mass->d_edgeMass.getValue().size(), 90);

            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], expectedDensity);

            EXPECT_FLOATINGPOINT_EQ(mass->d_vertexMass.getValue()[0], (MassType)(expectedDensity * volumeElem * 1 / 20));
            EXPECT_FLOATINGPOINT_EQ(mass->d_vertexMass.getValue()[1], (MassType)(expectedDensity * volumeElem * 1 / 20 * 2)); // vertex shared by 2 hexa
            EXPECT_FLOATINGPOINT_EQ(mass->d_edgeMass.getValue()[0], (MassType)(expectedDensity * volumeElem * 1 / 40 * 2)); // Edge shared by 2 hexa
            EXPECT_FLOATINGPOINT_EQ(mass->d_edgeMass.getValue()[2], (MassType)(expectedDensity * volumeElem * 1 / 40));
        }

        return ;
    }


    void check_VertexMass_Lumping_Initialization_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1'                                                      "
                "               vertexMass='1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1' />               "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
    
        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){

            static const MassType volume = 8.0;
            static const MassType expectedTotalMass = 27.0;
            static const MassType expectedDensity = expectedTotalMass / volume;

            EXPECT_EQ(mass->getMassCount(), 27);
            EXPECT_EQ(mass->d_edgeMass.getValue().size(), 90);

            EXPECT_EQ(mass->isLumped(), true);
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], expectedDensity);
            EXPECT_EQ(mass->getVertexMass()[0], 1.0);

            EXPECT_FLOATINGPOINT_EQ(mass->d_vertexMass.getValue()[0], 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->d_vertexMass.getValue()[1], 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->d_edgeMass.getValue()[0], 0.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->d_edgeMass.getValue()[2], 0.0_sreal);
        }

        return ;
    }


    /// Check for the definition of two concurrent input data
    /// first, the totalMass info is used if existing (the most generic one)
    /// second, the massDensity
    /// at last the vertexMass info

    void check_DoubleDeclaration_TotalMassAndMassDensity_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0' totalMass='2.0' />                             "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
    
        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ( mass->getTotalMass(), 2.0_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getMassDensity()[0], SReal(mass->getTotalMass() / 8.0)); // 8 hexa
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(mass->getMassDensity()[0] / 20));
        }

        return ;
    }


    void check_DoubleDeclaration_TotalMassAndVertexMass_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1' totalMass='2.0'                                      "
                "               vertexMass='1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1' />               "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true );
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ( mass->getTotalMass(), 2.0_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getMassDensity()[0], SReal(mass->getTotalMass() / 8.0));
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(mass->getMassDensity()[0] / 20));
        }

        return ;
    }


    void check_DoubleDeclaration_MassDensityAndVertexMass_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1' massDensity='1.0'                                    "
                "               vertexMass='1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1' />               "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true );
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ( mass->getTotalMass(), 8.0_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getMassDensity()[0], 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], 0.05_sreal);
        }

        return ;
    }


    /// Check wrong input values, in all cases the Mass is initialized
    /// using the default totalMass value = 1.0

    void check_TotalMass_WrongValue_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' totalMass='-2.0' />                                              "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ( mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], 0.00625_sreal);
        }

        return ;
    }


    void check_MassDensity_WrongValue_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='-1.0' />                                            "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], 0.00625_sreal);
        }

        return ;
    }


    void check_MassDensity_WrongSize_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0 4.0' />                                         "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], 0.00625_sreal);
        }

        return ;
    }


    void check_VertexMass_WrongValue_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1'                                                      "
                "               vertexMass='1 -1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1' />              "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], 0.00625_sreal);
        }

        return ;
    }


    void check_VertexMass_WrongSize_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' lumping='1' vertexMass='1 2' />                                  "
                "</Node>                                                                                            " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], 0.00625_sreal);
        }

        return ;
    }


    /// Check coupling of wrong data values/size and concurrent data

    void check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0' totalMass='-2.0' />                            "
                "</Node>                                                                                            " ;

        /// Here : default totalMass value will be used since negative value is given

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], 0.00625_sreal);
        }

        return ;
    }


    void check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Hexa(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <HexahedronSetTopologyContainer name='Container' src='@grid' />                                "
                "    <MechanicalObject />                                                                           "
                "    <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                           "
                "    <MeshMatrixMass name='m_mass' massDensity='1.0 1.0' totalMass='2.0' />                         "
                "</Node>                                                                                            " ;

        /// Here : totalMass value will be used due to concurrent data

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 27 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 2.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.25_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], 0.0125_sreal);
        }

        return ;
    }





    //---------------------------------------------------------------
    // TETRA topology
    //---------------------------------------------------------------
    void check_DefaultAttributes_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' />                                                           "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        EXPECT_TRUE( mass->findData("vertexMass") != nullptr );
        EXPECT_TRUE( mass->findData("totalMass") != nullptr );
        EXPECT_TRUE( mass->findData("massDensity") != nullptr );

        EXPECT_TRUE( mass->findData("showGravityCenter") != nullptr );
        EXPECT_TRUE( mass->findData("showAxisSizeFactor") != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.25/3.0) );
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[7], SReal(0.25/3.0) );
        }
        return ;
    }


    void check_TotalMass_Initialization_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' totalMass='2.0'/>                                            "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 2.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.25_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.5/3.0) );
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[1], 0.1_sreal);
        }

        return ;
    }


    void check_MassDensity_Initialization_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0' />                                         "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 8.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(2.0/3.0) );
        }

        return ;
    }


    void check_VertexMass_Lumping_Initialization_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' lumping='1' vertexMass='1 1 1 1 1 1 1 1' />                  "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true );
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 8.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], 1.0_sreal);
        }

        return ;
    }


    /// Check for the definition of two concurrent input data
    /// first, the totalMass info is used if existing (the most generic one)
    /// second, the massDensity
    /// at last the vertexMass info

    void check_DoubleDeclaration_TotalMassAndMassDensity_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0' totalMass='2.0' />                         "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 2.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.25_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.5/3.0) );
        }

        return ;
    }


    void check_DoubleDeclaration_TotalMassAndVertexMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' lumping='1' vertexMass='1 1 1 1 1 1 1 1' totalMass='2.0'/>   "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true );
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 2.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.25_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.5/3.0) );
        }

        return ;
    }


    void check_DoubleDeclaration_MassDensityAndVertexMass_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' vertexMass='1 1 1 1 1 1 1 1' lumping='1' massDensity='1.0' />"
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->isLumped(), true );
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 8.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(2.0/3.0) );
        }

        return ;
    }


    /// Check wrong input values, in all cases the Mass is initialized
    /// using the default totalMass value = 1.0

    void check_TotalMass_WrongValue_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' totalMass='-2.0'/>                                           "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());   

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.25/3.0) );
        }

        return ;
    }


    void check_MassDensity_WrongValue_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='-1.0'/>                                         "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.25/3.0) );
        }

        return ;
    }


    void check_MassDensity_WrongSize_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0 4.0'/>                                      "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.25/3.0) );
        }

        return ;
    }


    void check_VertexMass_WrongValue_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' vertexMass='1 -1 1 1 1 1 1 1' lumping='1' />                 "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.25/3.0) );
        }

        return ;
    }


    void check_VertexMass_WrongSize_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' vertexMass='1.0 2.0' lumping='1' />                          "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";
        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.25/3.0) );
        }

        return ;
    }


    /// Check coupling of wrong data values/size and concurrent data

    void check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0' totalMass='-2.0' />                        "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        /// Here : default totalMass value will be used since negative value is given

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 1.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.125_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.25/3.0) );
        }

        return ;
    }


    void check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Tetra(){
        static const string scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <MechanicalObject />                                                                           "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "        <MechanicalObject src='@../grid'/>                                                         "
                "        <TetrahedronSetTopologyContainer name='Container' />                                       "
                "        <TetrahedronSetTopologyModifier name='Modifier' />                                         "
                "        <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                      "
                "        <Hexa2TetraTopologicalMapping name='default28' input='@../grid' output='@Container' />     "
                "        <MeshMatrixMass name='m_mass' massDensity='1.0 1.0' totalMass='2.0' />                     "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";

        /// Here : totalMass value will be used due to concurrent data

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        EXPECT_TRUE( mass != nullptr );

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMassCount(), 8 );
            EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 2.0_sreal);
            EXPECT_FLOATINGPOINT_EQ(mass->getMassDensity()[0], 0.25_sreal);
            EXPECT_FLOATINGPOINT_EQ( mass->getVertexMass()[0], SReal(0.5/3.0) );
        }

        return ;
    }


    void checkTopologicalChanges_Hexa(bool lumped)
    {
        string scene;
        if(!lumped)
        {
            scene =
                    "<?xml version='1.0'?>                                                                              "
                    "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                    "    <DefaultAnimationLoop />                                                                       "
                    "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                    "    <Node name='Hexa' >                                                                            "
                    "            <MechanicalObject position = '@../grid.position' />                                    "
                    "            <HexahedronSetTopologyContainer name='Container' src='@../grid' />                     "
                    "            <HexahedronSetTopologyModifier name='Modifier' />                                      "
                    "            <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                   "
                    "            <MeshMatrixMass name='m_mass' massDensity='1.0'/>                                      "
                    "    </Node>                                                                                        "
                    "</Node>                                                                                            ";
        }
        else
        {
            scene =
                    "<?xml version='1.0'?>                                                                              "
                    "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                    "    <DefaultAnimationLoop />                                                                       "
                    "    <RegularGridTopology name='grid' n='3 3 3' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                    "    <Node name='Hexa' >                                                                            "
                    "            <MechanicalObject position = '@../grid.position' />                                    "
                    "            <HexahedronSetTopologyContainer name='Container' src='@../grid' />                     "
                    "            <HexahedronSetTopologyModifier name='Modifier' />                                      "
                    "            <HexahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                   "
                    "            <MeshMatrixMass name='m_mass' massDensity='1.0' lumping='true'/>                       "
                    "    </Node>                                                                                        "
                    "</Node>                                                                                            ";
        }
        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        ASSERT_NE(mass, nullptr);

        HexahedronSetTopologyModifier* modifier = root->getTreeObject<HexahedronSetTopologyModifier>();
        ASSERT_NE(modifier, nullptr);

        static const MassType volume = 8.0;
        static const MassType volumeElem = volume / 8.0; // 8 hexa in the grid            
        static const MassType expectedDensity = 1.0;
        static const MassType expectedTotalMass = expectedDensity * volume;

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        const VecMass& eMasses = mass->d_edgeMass.getValue();
        static const MassType refValueV = (MassType)(expectedDensity * volumeElem * 1 / 20);
        static const MassType refValueE = (MassType)(expectedDensity * volumeElem * 1 / 40);

        // check value at init
        EXPECT_EQ(vMasses.size(), 27);
        EXPECT_EQ(eMasses.size(), 90);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass);
        
        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 2);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE * 2);
            EXPECT_FLOATINGPOINT_EQ(eMasses[2], refValueE); // eMasses[1] == 0 because not taken into account from grid to hexahedron Topology
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[2], 0._sreal);
        }
        
        // -- remove hexahedron id: 0 -- 
        sofa::type::vector<sofa::Index> hexaIds = { 0 };
        modifier->removeHexahedra(hexaIds);
        EXPECT_EQ(vMasses.size(), 26);
        EXPECT_EQ(eMasses.size(), 87);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 7.0_sreal);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV); // check update of Mass when removing tetra
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], refValueE);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], 0._sreal);
        }

        // -- remove hexahedron id: 0 --
        modifier->removeHexahedra(hexaIds);
        EXPECT_EQ(vMasses.size(), 25);
        EXPECT_EQ(eMasses.size(), 84);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), 6.0_sreal);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV); // check update of Mass when removing tetra
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], refValueE);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], 0._sreal);
        }
        
        // -- remove hexahedron id: 0, 1 --
        hexaIds.push_back(1);
        modifier->removeHexahedra(hexaIds);
        EXPECT_EQ(vMasses.size(), 21);
        EXPECT_EQ(eMasses.size(), 74);
        EXPECT_NEAR(mass->getTotalMass(), 4.0, 1e-14);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 2);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[20], refValueE);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[20], 0._sreal);
        }

        // -- remove hexahedron id: 0, 1, 2, 3 --
        hexaIds.push_back(2);
        hexaIds.push_back(3);
        modifier->removeHexahedra(hexaIds);
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_EQ(eMasses.size(), 36); // TODO epernod 2021-06-29: not empty, because not all edges are linked to hexahedron. Potential bug here.
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }


    void checkTopologicalChanges_Tetra(bool lumped)
    {
        string scene;
        if(!lumped)
        {
             scene =
                    "<?xml version='1.0'?>                                                                              "
                    "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                    "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                    "    <DefaultAnimationLoop />                                                                       "
                    "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                    "    <Node name='Tetra' >                                                                           "
                    "            <MechanicalObject position='@../grid.position' />                                      "
                    "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                    "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                    "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                    "            <Hexa2TetraTopologicalMapping input='@../grid' output='@Container' />                  "
                    "            <MeshMatrixMass name='m_mass' massDensity='1.0'/>                                      "
                    "    </Node>                                                                                        "
                    "</Node>                                                                                            ";
        }
        else
        {
            scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <RequiredPlugin name='Sofa.Component.Topology.Mapping'/>                                       "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='2 2 2' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Tetra' >                                                                           "
                "            <MechanicalObject position='@../grid.position' />                                      "
                "            <TetrahedronSetTopologyContainer name='Container' />                                   "
                "            <TetrahedronSetTopologyModifier name='Modifier' />                                     "
                "            <TetrahedronSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                  "
                "            <Hexa2TetraTopologicalMapping input='@../grid' output='@Container' />                  "
                "            <MeshMatrixMass name='m_mass' massDensity='1.0' lumping='true'/>                       "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";
        }

        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        ASSERT_NE(mass, nullptr);

        TetrahedronSetTopologyModifier* modifier = root->getTreeObject<TetrahedronSetTopologyModifier>();
        ASSERT_NE(modifier, nullptr);

        static const MassType volume = 8.0;
        static const MassType volumeElem = volume / 6.0; // 6 tetra in the grid            
        static const MassType expectedDensity = 1.0;
        static const MassType expectedTotalMass = expectedDensity * volume;
        static const MassType massElem = expectedDensity * volumeElem;

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        const VecMass& eMasses = mass->d_edgeMass.getValue();
        static const MassType refValueV = (MassType)(expectedDensity * volumeElem * 1 / 10);
        static const MassType refValueE = (MassType)(expectedDensity * volumeElem * 1 / 20);

        // check value at init
        EXPECT_EQ(vMasses.size(), 8);
        EXPECT_EQ(eMasses.size(), 19);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV * 5);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 3);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE * 3);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], refValueE * 2);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], 0._sreal);
        }

        // -- remove tetrahedron id: 0 -- 
        sofa::type::vector<sofa::Index> elemIds = { 0 };
        modifier->removeTetrahedra(elemIds);
        EXPECT_EQ(vMasses.size(), 8);
        EXPECT_EQ(eMasses.size(), 18);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass - massElem);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV * 4); // check update of Mass when removing tetra
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 2);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE * 2);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], refValueE);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], 0._sreal);
        }


        // -- remove tetrahedron id: 0 --
        modifier->removeTetrahedra(elemIds);
        EXPECT_EQ(vMasses.size(), 7);
        EXPECT_EQ(eMasses.size(), 15);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass - 2 * massElem);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV * 3); // check update of Mass when removing tetra
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 2);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], refValueE);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], 0._sreal);
        }


        // -- remove tetrahedron id: 0, 1 --
        elemIds.push_back(1);
        modifier->removeTetrahedra(elemIds);
        EXPECT_EQ(vMasses.size(), 6);
        EXPECT_EQ(eMasses.size(), 11);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass - 4 * massElem);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], refValueE);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], 0._sreal);
        }

        // -- remove tetrahedron id: 0, 1 --
        modifier->removeTetrahedra(elemIds);
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_EQ(eMasses.size(), 0);
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }

    void checkTopologicalChanges_Quad(bool lumped)
    {
        string scene;
        if(!lumped)
        {
            scene =
                    "<?xml version='1.0'?>                                                                              "
                    "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                    "    <DefaultAnimationLoop />                                                                       "
                    "    <RegularGridTopology name='grid' n='3 3 1' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                    "    <Node name='Quad' >                                                                            "
                    "            <MechanicalObject position = '@../grid.position' />                                    "
                    "            <QuadSetTopologyContainer name='Container' src='@../grid' />                           "
                    "            <QuadSetTopologyModifier name='Modifier' />                                            "
                    "            <QuadSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                         "
                    "            <MeshMatrixMass name='m_mass' massDensity='1.0'/>                                      "
                    "    </Node>                                                                                        "
                    "</Node>                                                                                            ";
        }
        else
        {
            scene =
                    "<?xml version='1.0'?>                                                                              "
                    "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                    "    <DefaultAnimationLoop />                                                                       "
                    "    <RegularGridTopology name='grid' n='3 3 1' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                    "    <Node name='Quad' >                                                                            "
                    "            <MechanicalObject position = '@../grid.position' />                                    "
                    "            <QuadSetTopologyContainer name='Container' src='@../grid' />                           "
                    "            <QuadSetTopologyModifier name='Modifier' />                                            "
                    "            <QuadSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                         "
                    "            <MeshMatrixMass name='m_mass' massDensity='1.0' lumping='true'/>                       "
                    "    </Node>                                                                                        "
                    "</Node>                                                                                            ";
        }

        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        ASSERT_NE(mass, nullptr);

        QuadSetTopologyModifier* modifier = root->getTreeObject<QuadSetTopologyModifier>();
        ASSERT_NE(modifier, nullptr);

        static const MassType volume = 4.0;
        static const MassType volumeElem = volume / 4.0; // 4 quads in the grid            
        static const MassType expectedDensity = 1.0;
        static const MassType expectedTotalMass = expectedDensity * volume;
        static const MassType massElem = expectedDensity * volumeElem;

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        const VecMass& eMasses = mass->d_edgeMass.getValue();
        static const MassType refValueV = (MassType)(expectedDensity * volumeElem * 1 / 8);
        static const MassType refValueE = (MassType)(expectedDensity * volumeElem * 1 / 16);

        // check value at init
        EXPECT_EQ(vMasses.size(), 9);
        EXPECT_EQ(eMasses.size(), 16);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 2);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE * 2);
            EXPECT_FLOATINGPOINT_EQ(eMasses[2], refValueE); // eMasses[1] == 0 because not taken into account from grid to quad Topology
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[2], 0._sreal);
        }

        // -- remove quad id: 0 -- 
        sofa::type::vector<sofa::Index> elemIds = { 0 };
        modifier->removeQuads(elemIds, true, true);
        EXPECT_EQ(vMasses.size(), 8);
        EXPECT_EQ(eMasses.size(), 14);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass - massElem);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV); // check update of Mass when removing tetra
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], refValueE);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], 0._sreal);
        }

        // -- remove quad id: 0 --
        modifier->removeQuads(elemIds, true, true);
        EXPECT_EQ(vMasses.size(), 7);
        EXPECT_EQ(eMasses.size(), 12);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass - 2 * massElem);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], refValueE);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], 0._sreal);
        }

        // -- remove quad id: 0, 1 --
        elemIds.push_back(1);
        modifier->removeQuads(elemIds, true, true);        
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_EQ(eMasses.size(), 4); // TODO epernod 2021-06-29: not empty, because not all edges are linked to quads. Potential bug here.
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }


    void checkTopologicalChanges_Triangle(bool lumped)
    {
        string scene;
        if(!lumped)
        {
            scene =
                    "<?xml version='1.0'?>                                                                              "
                    "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                    "    <DefaultAnimationLoop />                                                                       "
                    "    <RegularGridTopology name='grid' n='3 3 1' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                    "    <Node name='Triangle' >                                                                        "
                    "            <MechanicalObject position = '@../grid.position' />                                    "
                    "            <TriangleSetTopologyContainer name='Container' src='@../grid' />                       "
                    "            <TriangleSetTopologyModifier name='Modifier' />                                        "
                    "            <TriangleSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                     "
                    "            <MeshMatrixMass name='m_mass' massDensity='1.0'/>                                      "
                    "    </Node>                                                                                        "
                    "</Node>                                                                                            ";
        }
        else
        {
            scene =
                    "<?xml version='1.0'?>                                                                              "
                    "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                    "    <DefaultAnimationLoop />                                                                       "
                    "    <RegularGridTopology name='grid' n='3 3 1' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                    "    <Node name='Triangle' >                                                                        "
                    "            <MechanicalObject position = '@../grid.position' />                                    "
                    "            <TriangleSetTopologyContainer name='Container' src='@../grid' />                       "
                    "            <TriangleSetTopologyModifier name='Modifier' />                                        "
                    "            <TriangleSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                     "
                    "            <MeshMatrixMass name='m_mass' massDensity='1.0' lumping='true'/>                       "
                    "    </Node>                                                                                        "
                    "</Node>                                                                                            ";
        }

        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        ASSERT_NE(mass, nullptr);

        TriangleSetTopologyModifier* modifier = root->getTreeObject<TriangleSetTopologyModifier>();
        ASSERT_NE(modifier, nullptr);

        static const MassType volume = 4.0;
        static const MassType volumeElem = volume / 8.0; // 8 triangles in the grid            
        static const MassType expectedDensity = 1.0;
        static const MassType expectedTotalMass = expectedDensity * volume;
        static const MassType massElem = expectedDensity * volumeElem;

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        const VecMass& eMasses = mass->d_edgeMass.getValue();
        static const MassType refValueV = (MassType)(expectedDensity * volumeElem * 1 / 6);
        static const MassType refValueE = (MassType)(expectedDensity * volumeElem * 1 / 12);

        // check value at init
        EXPECT_EQ(vMasses.size(), 9);
        EXPECT_EQ(eMasses.size(), 16);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV * 2);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 3);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE * 2);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], refValueE * 2);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[1], 0._sreal);
        }


        // -- remove triangle id: 0 -- 
        sofa::type::vector<sofa::Index> elemIds = { 0 };
        modifier->removeTriangles(elemIds, true, true);
        EXPECT_EQ(vMasses.size(), 9);
        EXPECT_EQ(eMasses.size(), 15);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass - massElem);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV); // check update of Mass when removing tetra
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 2);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], refValueE * 2);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], 0._sreal);
        }

        // -- remove triangle id: 0 --
        modifier->removeTriangles(elemIds, true, true);
        EXPECT_EQ(vMasses.size(), 9);
        EXPECT_EQ(eMasses.size(), 14);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass - 2 * massElem);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 2);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], refValueE * 2);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], 0._sreal);
        }

        // -- remove triangle id: 0, 1 --
        elemIds.push_back(1);
        modifier->removeTriangles(elemIds, true, true);
        EXPECT_EQ(vMasses.size(), 7);
        EXPECT_EQ(eMasses.size(), 10);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), expectedTotalMass - 4 * massElem);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], refValueV * 2);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], refValueV * 2);
        // check edge mass (only if not lumped else eMass is not computed)
        if(!lumped)
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], refValueE);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], refValueE);
        }
        else
        {
            EXPECT_FLOATINGPOINT_EQ(eMasses[0], 0._sreal);
            EXPECT_FLOATINGPOINT_EQ(eMasses[3], 0._sreal);
        }

        // -- remove triangle id: 0, 1, 2, 3 --
        elemIds.push_back(2);
        elemIds.push_back(3);
        modifier->removeTriangles(elemIds, true, true);
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_EQ(eMasses.size(), 0);
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }

    void checkTopologicalChanges_Edge(bool lumped)
    {
        string scene;
        if(!lumped)
        {
            scene =
                    "<?xml version='1.0'?>                                                                              "
                    "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                    "    <DefaultAnimationLoop />                                                                       "
                    "    <RegularGridTopology name='grid' n='4 1 1' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                    "    <Node name='Edge' >                                                                            "
                    "            <MechanicalObject position = '@../grid.position' />                                    "
                    "            <EdgeSetTopologyContainer name='Container' src='@../grid' />                           "
                    "            <EdgeSetTopologyModifier name='Modifier' />                                            "
                    "            <EdgeSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                         "
                    "            <MeshMatrixMass name='m_mass' massDensity='1.0'/>                                      "
                    "    </Node>                                                                                        "
                    "</Node>                                                                                            ";
        }
        else
        {
            scene =
                "<?xml version='1.0'?>                                                                              "
                "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                                        "
                "    <DefaultAnimationLoop />                                                                       "
                "    <RegularGridTopology name='grid' n='4 1 1' min='0 0 0' max='2 2 2' p0='0 0 0' />               "
                "    <Node name='Edge' >                                                                            "
                "            <MechanicalObject position = '@../grid.position' />                                    "
                "            <EdgeSetTopologyContainer name='Container' src='@../grid' />                           "
                "            <EdgeSetTopologyModifier name='Modifier' />                                            "
                "            <EdgeSetGeometryAlgorithms template='Vec3d' name='GeomAlgo' />                         "
                "            <MeshMatrixMass name='m_mass' massDensity='1.0' lumping='true'/>                       "
                "    </Node>                                                                                        "
                "</Node>                                                                                            ";
        }

        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        ASSERT_NE(root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        TheMeshMatrixMass* mass = root->getTreeObject<TheMeshMatrixMass>();
        ASSERT_NE(mass, nullptr);

        EdgeSetTopologyModifier* modifier = root->getTreeObject<EdgeSetTopologyModifier>();
        ASSERT_NE(modifier, nullptr);

        const VecMass& vMasses = mass->d_vertexMass.getValue();
        const VecMass& eMasses = mass->d_edgeMass.getValue();
        static const MassType wrongValue = 0;
       // TODO epernod 2021-06-29: MeshMatrixMass based on edge topology doesn't support topological changes
       // Keeping expected values for record
       /*
       static const MassType volume = 2.0;
       static const MassType volumeElem = volume / 3.0; // 3 edges in the grid            
       static const MassType expectedDensity = 1.0;
       static const MassType expectedTotalMass = expectedDensity * volume;
       static const MassType massElem = expectedDensity * volumeElem;
       static const MassType refValueV = (MassType)(expectedDensity * volumeElem * 1 / 6);
       static const MassType refValueE = (MassType)(expectedDensity * volumeElem * 1 / 12);
       */

        // check value at init
        EXPECT_EQ(vMasses.size(), 4);
        EXPECT_EQ(eMasses.size(), 3);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), /*expectedTotalMass*/ wrongValue);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], wrongValue);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], wrongValue);
        // check edge mass
        EXPECT_FLOATINGPOINT_EQ(eMasses[0], wrongValue);
        EXPECT_FLOATINGPOINT_EQ(eMasses[1], wrongValue);

        // -- remove edge id: 0 -- 
        sofa::type::vector<sofa::Index> elemIds = { 0 };
        modifier->removeEdges(elemIds, true);
        EXPECT_EQ(vMasses.size(), 3);
        EXPECT_EQ(eMasses.size(), 2);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), /*expectedTotalMass - massElem*/ wrongValue);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], wrongValue); // check update of Mass when removing tetra
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], wrongValue);
        // check edge mass
        EXPECT_FLOATINGPOINT_EQ(eMasses[0], wrongValue);
        EXPECT_FLOATINGPOINT_EQ(eMasses[1], wrongValue);


        // -- remove edge id: 0 --
        modifier->removeEdges(elemIds, true);
        EXPECT_EQ(vMasses.size(), 2);
        EXPECT_EQ(eMasses.size(), 1);
        EXPECT_FLOATINGPOINT_EQ(mass->getTotalMass(), /*expectedTotalMass - 2 * massElem*/ wrongValue);

        // check vertex mass
        EXPECT_FLOATINGPOINT_EQ(vMasses[0], wrongValue);
        EXPECT_FLOATINGPOINT_EQ(vMasses[1], wrongValue);
        // check edge mass
        EXPECT_FLOATINGPOINT_EQ(eMasses[0], wrongValue);


        // -- remove edge id: 0 --
        modifier->removeEdges(elemIds, true);
        EXPECT_EQ(vMasses.size(), 0);
        EXPECT_EQ(eMasses.size(), 0);
        EXPECT_NEAR(mass->getTotalMass(), 0, 1e-4);
    }
};


typedef MeshMatrixMass_test<Vec3Types, Vec3Types::Real> MeshMatrixMass3_test;

TEST_F(MeshMatrixMass3_test, singleTriangle)
{
    VecCoord positions;
    positions.push_back(Coord(0.0f, 0.0f, 0.0f));
    positions.push_back(Coord(1.0f, 0.0f, 0.0f));
    positions.push_back(Coord(0.0f, 1.0f, 0.0f));

    const TriangleSetTopologyContainer::SPtr topologyContainer = New<TriangleSetTopologyContainer>();
    topologyContainer->addTriangle(0, 1, 2);

    const TriangleSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<TriangleSetGeometryAlgorithms<Vec3Types> >();

    static const MassType surface = 0.5;
    static const MassType expectedTotalMass = 1.0f;
    static const MassType density = expectedTotalMass / surface;

    static const VecMass expectedVMass(3, (MassType)(density * surface * 1 / 6));
    static const VecMass expectedEMass(3, (MassType)(density * surface * 1 / 12));

    runTest(positions, topologyContainer, geometryAlgorithms, expectedTotalMass, density, expectedVMass, expectedEMass);
}

TEST_F(MeshMatrixMass3_test, singleQuad)
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

    static const MassType surface = 1.0;
    static const MassType expectedTotalMass = 1.0f;
    static const MassType density = expectedTotalMass / surface;
    static const VecMass expectedVMass(4, (MassType)(density * surface * 1 / 8));
    static const VecMass expectedEMass(4, (MassType)(density * surface * 1 / 16));

    runTest(positions, topologyContainer, geometryAlgorithms, expectedTotalMass, density, expectedVMass, expectedEMass);
}

TEST_F(MeshMatrixMass3_test, singleTetrahedron)
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

    static const MassType volume = MassType(1.0/3.0) * 0.5; // V = 1/3 * B * h
    static const MassType expectedTotalMass = 1.0f;
    static const MassType density = expectedTotalMass / volume;
    static const VecMass expectedVMass(4, (MassType)(density * volume * 1 / 10));
    static const VecMass expectedEMass(6, (MassType)(density * volume * 1 / 20));

    runTest(positions, topologyContainer, geometryAlgorithms, expectedTotalMass, density, expectedVMass, expectedEMass);
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

    const HexahedronSetTopologyContainer::SPtr topologyContainer = New<HexahedronSetTopologyContainer>();
    topologyContainer->addHexa(0, 1, 2, 3, 4, 5, 6, 7);

    const HexahedronSetGeometryAlgorithms<Vec3Types>::SPtr geometryAlgorithms
        = New<HexahedronSetGeometryAlgorithms<Vec3Types> >();

    static const MassType volume = 1.0;
    static const MassType expectedTotalMass = 1.0f;
    static const MassType density = expectedTotalMass / volume;
    static const VecMass expectedVMass(8, (MassType)(density * volume * 1 / 20));
    static const VecMass expectedEMass(12, (MassType)(density * volume * 1 / 40));

    runTest(positions, topologyContainer, geometryAlgorithms, expectedTotalMass, density, expectedVMass, expectedEMass);
}



TEST_F(MeshMatrixMass3_test, check_DefaultAttributes_Hexa){
    check_DefaultAttributes_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_TotalMass_Initialization_Hexa){
    check_TotalMass_Initialization_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_Initialization_Hexa){
    check_MassDensity_Initialization_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_Lumping_Initialization_Hexa){
    check_VertexMass_Lumping_Initialization_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_Hexa){
    check_DoubleDeclaration_TotalMassAndMassDensity_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndVertexMass_Hexa){
    check_DoubleDeclaration_TotalMassAndVertexMass_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_MassDensityAndVertexMass_Hexa){
    check_DoubleDeclaration_MassDensityAndVertexMass_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_TotalMass_WrongValue_Hexa){
    check_TotalMass_WrongValue_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_WrongValue_Hexa){
    check_MassDensity_WrongValue_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_WrongSize_Hexa){
    check_MassDensity_WrongSize_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_WrongValue_Hexa){
    check_VertexMass_WrongValue_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_WrongSize_Hexa){
    check_VertexMass_WrongSize_Hexa();
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Hexa){
    check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Hexa();
}

TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Hexa){
    check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Hexa();
}

TEST_F(MeshMatrixMass3_test, check_DefaultAttributes_Tetra){
    check_DefaultAttributes_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_TotalMass_Initialization_Tetra){
    check_TotalMass_Initialization_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_Initialization_Tetra){
    check_MassDensity_Initialization_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_Lumping_Initialization_Tetra){
    check_VertexMass_Lumping_Initialization_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_Tetra){
    check_DoubleDeclaration_TotalMassAndMassDensity_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndVertexMass_Tetra){
    check_DoubleDeclaration_TotalMassAndVertexMass_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_MassDensityAndVertexMass_Tetra){
    check_DoubleDeclaration_MassDensityAndVertexMass_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_TotalMass_WrongValue_Tetra){
    check_TotalMass_WrongValue_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_WrongValue_Tetra){
    check_MassDensity_WrongValue_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_MassDensity_WrongSize_Tetra){
    check_MassDensity_WrongSize_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_WrongValue_Tetra){
    check_VertexMass_WrongValue_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_VertexMass_WrongSize_Tetra){
    check_VertexMass_WrongSize_Tetra();
}


TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Tetra){
    check_DoubleDeclaration_TotalMassAndMassDensity_WrongValue_Tetra();
}

TEST_F(MeshMatrixMass3_test, check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Tetra){
    check_DoubleDeclaration_TotalMassAndMassDensity_WrongSize_Tetra();
}


TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Hexa_sparse) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Hexa(false);
}

TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Tetra_sparse) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Tetra(false);
}

TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Quad_sparse) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Quad(false);
}

TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Triangle_sparse) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Triangle(false);
}

TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Edge_sparse) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Edge(false);
}

TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Hexa_lumped) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Hexa(true);
}

TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Tetra_lumped) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Tetra(true);
}

TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Quad_lumped) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Quad(true);
}

TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Triangle_lumped) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Triangle(true);
}

TEST_F(MeshMatrixMass3_test, checkTopologicalChanges_Edge_lumped) {
    EXPECT_MSG_NOEMIT(Error);
    checkTopologicalChanges_Edge(true);
}


} // namespace sofa
