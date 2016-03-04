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
#include <SofaBaseMechanics/DiagonalMass.h>

#include <SofaBaseMechanics/initBaseMechanics.h>
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

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>

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
        ASSERT_EQ(mstate->x.getValue().size(), mass->f_mass.getValue().size());
        // Safety check...
        ASSERT_EQ(mstate->x.getValue().size(), expectedMass.size());

        // Check the total mass.
        EXPECT_FLOAT_EQ(expectedTotalMass, mass->m_totalMass.getValue());

        // Check the mass at each index.
        for (size_t i = 0 ; i < mstate->x.getValue().size() ; i++)
            EXPECT_FLOAT_EQ(expectedMass[i], mass->f_mass.getValue()[i]);
    }

    void runTest(VecCoord positions, BaseObject::SPtr topologyContainer, BaseObject::SPtr geometryAlgorithms,
                 MassType expectedTotalMass, const VecMass& expectedMass)
    {
        createSceneGraph(positions, topologyContainer, geometryAlgorithms);
        simulation::getSimulation()->init(root.get());
        check(expectedTotalMass, expectedMass);
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


} // namespace sofa
