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

#include <SofaTest/Sofa_test.h>
#include <SofaBoundaryCondition/PartialFixedConstraint.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SceneCreator/SceneCreator.h>
#include <SofaBoundaryCondition/ConstantForceField.h>

#include <SofaTest/TestMessageHandler.h>


namespace sofa{
namespace {
using namespace modeling;
using namespace core::objectmodel;

template<typename DataTypes>
void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<DataTypes>& /*dofs*/)
{
    node->addObject(New<component::mass::UniformMass<DataTypes, typename DataTypes::Real> >());
}

template<>
void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<defaulttype::Rigid3Types>& /*dofs*/)
{
    node->addObject(New<component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass> >());
}

template<>
void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<defaulttype::Rigid2Types>& /*dofs*/)
{
    node->addObject(New<component::mass::UniformMass<defaulttype::Rigid2Types, defaulttype::Rigid2Mass> >());
}

template <typename _DataTypes>
struct PartialFixedConstraint_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Real  Real;

    typedef component::projectiveconstraintset::PartialFixedConstraint<DataTypes> PartialFixedConstraint;
    typedef component::forcefield::ConstantForceField<DataTypes> ForceField;
    typedef component::container::MechanicalObject<DataTypes> MechanicalObject;

    typedef typename MechanicalObject::VecCoord  VecCoord;
    typedef typename MechanicalObject::Coord  Coord;
    typedef typename MechanicalObject::VecDeriv  VecDeriv;
    typedef typename MechanicalObject::Deriv  Deriv;
    typedef sofa::helper::fixed_array<bool,Deriv::total_size> VecBool;

    bool test(double epsilon, const std::string &integrationScheme )
    {
        typename sofa::component::container::MechanicalObject<DataTypes>::SPtr  mstate;

        /// Scene initialization
        sofa::simulation::Simulation* simulation;
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
        simulation::Node::SPtr root = simulation->createNewGraph("root");
        root->setGravity( defaulttype::Vector3(0,0,0) );
        simulation::Node::SPtr node = createEulerSolverNode(root,"EulerSolver", integrationScheme);

        mstate = New<sofa::component::container::MechanicalObject<DataTypes> >();
        mstate->resize(1);
        node->addObject(mstate);
        createUniformMass<DataTypes>(node, *mstate.get());

        std::cout<<"1"<<std::endl;

        Deriv force;
        size_t sizeD = force.size();
        for(unsigned i=0; i<sizeD; i++)
        {
            force[i]=10;
        }

        std::cout<<"2"<<std::endl;

        VecBool fixed;
        for(unsigned i=0; i<sizeD; i++)
        {
            fixed[i]=false;
        }

        std::cout<<"fixed = "<<fixed<<std::endl;

        typename ForceField::SPtr forceField = addNew<ForceField>(node);
        forceField->setForce( 0, force );
        typename PartialFixedConstraint::SPtr constraint = addNew<PartialFixedConstraint>(node);

        std::cout<<"3"<<std::endl;

        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());

        for(unsigned i=0; i<sizeD; i++)
        {
            fixed[i] = true;
            constraint->d_fixedDirections.setValue(fixed);

            std::cout<<"4"<<std::endl;

            // Perform one time step
            sofa::simulation::getSimulation()->animate(root.get(),0.5);

            std::cout<<"5"<<std::endl;

            // Check if the particle moved in a fixed direction
            typename MechanicalObject::ReadVecDeriv readV = mstate->readVelocities();
            if( readV[0][i] > epsilon )
            {
                ADD_FAILURE() << "position (index " << i << ") changed, fixed direction did not work" << std::endl;
                return false;
            }

            std::cout<<"6"<<std::endl;

            sofa::simulation::getSimulation()->reset(root.get());
            fixed[i] = false;

            std::cout<<"7"<<std::endl;
        }

        simulation::getSimulation()->unload(root);

        std::cout<<"7"<<std::endl;
        return true;
    }
};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    defaulttype::Vec1Types,
    defaulttype::Vec2Types,
    defaulttype::Vec3Types,
    defaulttype::Vec6Types,
    defaulttype::Rigid2Types,
    defaulttype::Rigid3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(PartialFixedConstraint_test, DataTypes);

// test cases
TYPED_TEST( PartialFixedConstraint_test , testContraintExplicit )
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_TRUE(  this->test(1e-8, std::string("Explicit")) );
}

TYPED_TEST( PartialFixedConstraint_test , testContraintImplicit )
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_TRUE(  this->test(1e-8, std::string("Implicit")) );
}

}// namespace
}// namespace sofa
