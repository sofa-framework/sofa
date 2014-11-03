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

#include "Sofa_test.h"
#include <SofaBoundaryCondition/FixedConstraint.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <plugins/SceneCreator/SceneCreator.h>
#include <SofaBoundaryCondition/ConstantForceField.h>

namespace sofa{
namespace {
using namespace modeling;
using core::objectmodel::New;

template<typename DataTypes>
void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<DataTypes>& dofs)
{
    node->addObject(New<component::mass::UniformMass<DataTypes, typename DataTypes::Real> >());
}

template<>
void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<defaulttype::Rigid3Types>& dofs)
{
    node->addObject(New<component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass> >());
}

template<>
void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<defaulttype::Rigid2Types>& dofs)
{
    node->addObject(New<component::mass::UniformMass<defaulttype::Rigid2Types, defaulttype::Rigid2Mass> >());
}

template <typename _DataTypes>
struct FixedConstraint_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef component::projectiveconstraintset::FixedConstraint<DataTypes> FixedConstraint;
    typedef component::forcefield::ConstantForceField<DataTypes> ForceField;
    typedef component::container::MechanicalObject<DataTypes> MechanicalObject;
    
    
    typedef typename MechanicalObject::VecCoord  VecCoord;
    typedef typename MechanicalObject::Coord  Coord;
    typedef typename MechanicalObject::VecDeriv  VecDeriv;
    typedef typename MechanicalObject::Deriv  Deriv;
    typedef typename DataTypes::Real  Real;

   


	bool test(double epsilon)
	{
		//Init

        simulation::Simulation* simulation;  
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
        Coord initCoord;
        Deriv force;
        for(unsigned i=0; i<force.size(); i++)
            force[i]=10;

        /// Scene creation 
        simulation::Node::SPtr root = simulation->createNewGraph("root");
        root->setGravity( defaulttype::Vector3(0,0,0) );

        simulation::Node::SPtr node = createEulerSolverNode(root,"test");
        
        typename MechanicalObject::SPtr dofs = addNew<MechanicalObject>(node);
        dofs->resize(1);

        createUniformMass<DataTypes>(node, *dofs.get());

        typename ForceField::SPtr forceField = addNew<ForceField>(node);
        forceField->setForce( 0, force );

        node->addObject(sofa::core::objectmodel::New<FixedConstraint>());

        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());

        // Perform one time step
        sofa::simulation::getSimulation()->animate(root.get(),0.5);

        // Check if the particle moved
        typename MechanicalObject::ReadVecCoord readX = dofs->readPositions();
        if( (readX[0]-initCoord).norm2()>epsilon )
        {
            ADD_FAILURE() << "Error: unmatching position between " << readX[0] << " and " << initCoord<< endl;
            return false;
        }
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
TYPED_TEST_CASE(FixedConstraint_test, DataTypes);
// first test case
TYPED_TEST( FixedConstraint_test , testValue )
{
    EXPECT_TRUE(  this->test(1e-8) );
}

}// namespace 
}// namespace sofa







