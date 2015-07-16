/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
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
using sofa::component::mass::DiagonalMass;
using sofa::component::container::MechanicalObject;

namespace sofa {

template <class T>
class DiagonalMass_test : public ::testing::Test
{
public:
    simulation::Node::SPtr root;
    simulation::Simulation* simulation;

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
};


// Define the list of DataTypes to instanciate the tests with.
using testing::Types;
typedef Types<
#if defined(SOFA_FLOAT)
    Vec3fTypes
#elif defined(SOFA_DOUBLE)
    Vec3dTypes
#else
    Vec3fTypes,
    Vec3dTypes
#endif
> DataTypes;

// Instanciate the test suite for each type in 'DataTypes'.
TYPED_TEST_CASE(DiagonalMass_test, DataTypes);


TYPED_TEST(DiagonalMass_test, singleHexahedron)
{
    typedef TypeParam DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef MechanicalObject<DataTypes> MechanicalObject;
    typedef DiagonalMass<DataTypes, typename DataTypes::Real> DiagonalMass;

    // Create the scene graph.
    simulation::Node::SPtr node = this->root->createChild("oneHexa");
    HexahedronSetTopologyContainer::SPtr container = New<HexahedronSetTopologyContainer>();
    node->addObject(container);
    typename MechanicalObject::SPtr mstate = New<MechanicalObject>();
    node->addObject(mstate);
    node->addObject(New<HexahedronSetGeometryAlgorithms<DataTypes> >());
    typename DiagonalMass::SPtr mass = New<DiagonalMass>();
    node->addObject(mass);

    // Handcraft a cubic hexahedron.
    VecCoord pos;
    pos.push_back(Coord(0.0, 0.0, 0.0));
    pos.push_back(Coord(1.0, 0.0, 0.0));
    pos.push_back(Coord(1.0, 1.0, 0.0));
    pos.push_back(Coord(0.0, 1.0, 0.0));
    pos.push_back(Coord(0.0, 0.0, 1.0));
    pos.push_back(Coord(1.0, 0.0, 1.0));
    pos.push_back(Coord(1.0, 1.0, 1.0));
    pos.push_back(Coord(0.0, 1.0, 1.0));
    mstate->x = pos;
    container->addHexa(0,1,2,3,4,5,6,7);

    // DiagonalMass computes the mass in reinit(), so init() the graph.
    simulation::getSimulation()->init(this->root.get());

    // Check that the mass vector has the right size.
    ASSERT_EQ(mstate->x.getValue().size(), mass->f_mass.getValue().size());

    // Check the mass at each index.
    EXPECT_FLOAT_EQ(0.125, mass->f_mass.getValue()[0]);
    EXPECT_FLOAT_EQ(0.125, mass->f_mass.getValue()[1]);
    EXPECT_FLOAT_EQ(0.125, mass->f_mass.getValue()[2]);
    EXPECT_FLOAT_EQ(0.125, mass->f_mass.getValue()[3]);
    EXPECT_FLOAT_EQ(0.125, mass->f_mass.getValue()[4]);
    EXPECT_FLOAT_EQ(0.125, mass->f_mass.getValue()[5]);
    EXPECT_FLOAT_EQ(0.125, mass->f_mass.getValue()[6]);
    EXPECT_FLOAT_EQ(0.125, mass->f_mass.getValue()[7]);

    // Check the total mass.
    EXPECT_FLOAT_EQ(1.0f, mass->m_totalMass.getValue());
}

} // namespace sofa
