
/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <SofaTest/Sofa_test.h>
#include <SofaBoundaryCondition/PointConstraint.h>
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

template<typename DataTypes>
void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<DataTypes>& /*dofs*/)
{
    node->addObject(sofa::core::objectmodel::New<component::mass::UniformMass<DataTypes, typename DataTypes::Real> >());
}

//template<>
//void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<defaulttype::Rigid3Types>& /*dofs*/)
//{
//    node->addObject(sofa::core::objectmodel::New<component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass> >());
//}

//template<>
//void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<defaulttype::Rigid2Types>& /*dofs*/)
//{
//    node->addObject(sofa::core::objectmodel::New<component::mass::UniformMass<defaulttype::Rigid2Types, defaulttype::Rigid2Mass> >());
//}




template <typename _DataTypes>
struct PointConstraint_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef component::projectiveconstraintset::PointConstraint<DataTypes> PointConstraint;
    typedef component::forcefield::ConstantForceField<DataTypes> ForceField;
    typedef component::container::MechanicalObject<DataTypes> MechanicalObject;

    typedef typename MechanicalObject::VecCoord  VecCoord;
    typedef typename MechanicalObject::Coord  Coord;
    typedef typename MechanicalObject::VecDeriv  VecDeriv;
    typedef typename MechanicalObject::Deriv  Deriv;
    typedef typename DataTypes::Real  Real;
    typedef sofa::helper::fixed_array<bool,Deriv::total_size> VecBool;

    bool test(double epsilon)
    {
        //Init

        simulation::Simulation* simulation;
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
        Deriv force;
        for(unsigned i=0; i<force.size(); i++)
            force[i]=50;

        /// Scene creation
        simulation::Node::SPtr root = simulation->createNewGraph("root");
        root->setGravity( defaulttype::Vector3(0,0,0) );

        simulation::Node::SPtr node = createEulerSolverNode(root,"test");

        typename MechanicalObject::SPtr dofs = addNew<MechanicalObject>(node);
        dofs->resize(2);

        createUniformMass<DataTypes>(node, *dofs.get());

        typename ForceField::SPtr forceField = addNew<ForceField>(node);
        forceField->setForce( 0, force );

        typename PointConstraint::SPtr constraint = addNew<PointConstraint>(node);


        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());

        unsigned int dofsNbr = dofs->getSize();
        helper::vector<unsigned int> fixed_indices;

        for(unsigned i=0; i<dofsNbr; i++)
        {
            fixed_indices.push_back(i);
        }

        constraint->f_indices.setValue(fixed_indices);
        for(unsigned i=0; i<dofsNbr; i++)
        {
            // Perform one time step
            sofa::simulation::getSimulation()->animate(root.get(),0.5);

            // Check if the particle moved in a fixed direction
            typename MechanicalObject::ReadVecDeriv readV = dofs->readVelocities();
            for (unsigned int j = 0; j < readV[i].size(); ++j) {
                if( readV[i][j]>epsilon )
                {
                    ADD_FAILURE() << "Error: non null velocity in direction " << j << std::endl;
                    return false;
                }
            }

            sofa::simulation::getSimulation()->reset(root.get());
        }

        return true;
    }
};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    defaulttype::Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(PointConstraint_test, DataTypes);
// first test case
TYPED_TEST( PointConstraint_test , testValue )
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_TRUE(  this->test(1e-8) );
}

}// namespace
}// namespace sofa







