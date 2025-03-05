﻿/******************************************************************************
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
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/component/constraint/projective/FixedPlaneProjectiveConstraint.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/mass/UniformMass.h>
#include <SceneCreator/SceneCreator.h>
#include <sofa/component/mechanicalload/ConstantForceField.h>


namespace sofa{
namespace {
using namespace modeling;
using namespace core::objectmodel;

template<typename DataTypes>
void createUniformMass(simulation::Node::SPtr node, component::statecontainer::MechanicalObject<DataTypes>& /*dofs*/)
{
    typename component::mass::UniformMass<DataTypes>::SPtr uniformMass = New<component::mass::UniformMass<DataTypes> >();
    uniformMass->d_totalMass.setValue(1.0);
    node->addObject(uniformMass);
}

template <typename _DataTypes>
struct FixedPlaneProjectiveConstraint_test : public BaseSimulationTest
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Real  Real;

    typedef component::constraint::projective::FixedPlaneProjectiveConstraint<DataTypes> FixedPlaneProjectiveConstraint;
    typedef component::mechanicalload::ConstantForceField<DataTypes> ForceField;
    typedef component::statecontainer::MechanicalObject<DataTypes> MechanicalObject;

    typedef typename MechanicalObject::VecCoord  VecCoord;
    typedef typename MechanicalObject::Coord  Coord;
    typedef typename MechanicalObject::VecDeriv  VecDeriv;
    typedef typename MechanicalObject::Deriv  Deriv;
    typedef sofa::type::fixed_array<bool,Deriv::total_size> VecBool;

    bool test(double epsilon, const std::string &integrationScheme )
    {
        typename sofa::component::statecontainer::MechanicalObject<DataTypes>::SPtr  mstate;

        /// Scene initialization
        sofa::simulation::Simulation* simulation;
        simulation = sofa::simulation::getSimulation();
        simulation::Node::SPtr root = simulation->createNewGraph("root");
        root->setGravity( type::Vec3(0,0,0) );

        simulation::Node::SPtr node = createEulerSolverNode(root,"EulerSolver", integrationScheme);

        mstate = New<sofa::component::statecontainer::MechanicalObject<DataTypes> >();
        mstate->resize(1);
        node->addObject(mstate);
        createUniformMass<DataTypes>(node, *mstate.get());

        Deriv force;
        const size_t sizeD = force.size();
        for(unsigned i=0; i<sizeD; i++)
        {
            force[i]=10;
        }

        Coord fixed;

        typename ForceField::SPtr forceField = addNew<ForceField>(node);
        forceField->setForce( 0, force );
        typename FixedPlaneProjectiveConstraint::SPtr constraint = addNew<FixedPlaneProjectiveConstraint>(node);
        constraint->d_indices.setValue({0});
        if(constraint->d_indices.getValue().empty())
        {
            ADD_FAILURE() << "Empty indices" << std::endl;
            return false;
        }

        // Init simulation
        sofa::simulation::node::initRoot(root.get());

        for(unsigned i=0; i < _DataTypes::spatial_dimensions; i++)
        {
            fixed[i] = 1.;
            constraint->d_direction.setValue(fixed);

            // Perform one time step
            sofa::simulation::node::animate(root.get(),0.5);

            // Check if the particle moved in a fixed direction
            typename MechanicalObject::ReadVecDeriv readV = mstate->readVelocities();
            if( readV[0][i] > epsilon )
            {
                ADD_FAILURE() << "position (index " << i << ") changed, fixed direction did not work " << readV[0] << std::endl;
                return false;
            }

            sofa::simulation::node::reset(root.get());
            fixed[i] = false;
        }

        sofa::simulation::node::unload(root);
        return true;
    }
};

// Define the list of DataTypes to instantiate
using ::testing::Types;
typedef Types<
    defaulttype::Vec3Types,
    defaulttype::Vec6Types,
    defaulttype::Rigid3Types
> DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_SUITE(FixedPlaneProjectiveConstraint_test, DataTypes);

// test cases
TYPED_TEST( FixedPlaneProjectiveConstraint_test , testContraintExplicit )
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_TRUE(  this->test(1e-8, std::string("Explicit")) );
}

TYPED_TEST( FixedPlaneProjectiveConstraint_test , testConstraintImplicitWithCG )
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_TRUE(  this->test(1e-8, std::string("Implicit")) );
}

TYPED_TEST( FixedPlaneProjectiveConstraint_test , testConstraintImplicitWithSparseLDL )
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_TRUE(  this->test(1e-8, std::string("Implicit_SparseLDL")) );
}

}// namespace
}// namespace sofa
