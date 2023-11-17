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

#include <sofa/component/constraint/projective/FixedProjectiveConstraint.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/mass/UniformMass.h>
#include <SceneCreator/SceneCreator.h>
#include <sofa/component/mechanicalload/ConstantForceField.h>
#include <sofa/component/topology/container/dynamic/PointSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/PointSetTopologyModifier.h>

#include <sofa/simulation/graph/SimpleApi.h>

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

namespace sofa{
namespace {
using namespace modeling;
using core::objectmodel::New;

template<typename DataTypes>
void createUniformMass(simulation::Node::SPtr node, component::statecontainer::MechanicalObject<DataTypes>& /*dofs*/)
{
    node->addObject(New<component::mass::UniformMass<DataTypes> >());
}

template <typename _DataTypes>
struct FixedProjectiveConstraint_test : public BaseTest
{
    typedef _DataTypes DataTypes;
    typedef component::constraint::projective::FixedProjectiveConstraint<DataTypes> FixedProjectiveConstraint;
    typedef component::mechanicalload::ConstantForceField<DataTypes> ForceField;
    typedef component::statecontainer::MechanicalObject<DataTypes> MechanicalObject;


    typedef typename MechanicalObject::VecCoord  VecCoord;
    typedef typename MechanicalObject::Coord  Coord;
    typedef typename MechanicalObject::VecDeriv  VecDeriv;
    typedef typename MechanicalObject::Deriv  Deriv;
    typedef typename DataTypes::Real  Real;

    bool test(double epsilon, const std::string &integrationScheme )
    {
        //Init

        simulation::Simulation* simulation = sofa::simulation::getSimulation();
        assert(simulation);

        Coord initCoord1, initCoord2;
        Deriv force;
        for(unsigned i=0; i<force.size(); i++)
            force[i]=10;

        /// Scene creation
        const simulation::Node::SPtr root = simulation->createNewGraph("root");
        root->setGravity( type::Vec3(0,0,0) );

        simpleapi::createObject(root , "RequiredPlugin", {{"name", "Sofa.Component.LinearSolver.Direct"}}) ;
        simpleapi::createObject(root , "RequiredPlugin", {{"name", "Sofa.Component.ODESolver.Forward"}}) ;
        simpleapi::createObject(root , "RequiredPlugin", {{"name", "Sofa.Component.ODESolver.Backward"}}) ;

        simulation::Node::SPtr node = createEulerSolverNode(root,"test", integrationScheme);

        typename MechanicalObject::SPtr dofs = addNew<MechanicalObject>(node);
        dofs->resize(2);
        typename MechanicalObject::WriteVecCoord writeX = dofs->writePositions();
        for(unsigned int i=0;i<writeX[0].size();i++)
        {
            writeX[0][i] = 1.0+0.1*i ;
            writeX[1][i] = 2.0+0.1*i ;
        }

        typename MechanicalObject::ReadVecCoord readX = dofs->readPositions();
        initCoord1 = readX[0];
        initCoord2 = readX[1];


        createUniformMass<DataTypes>(node, *dofs.get());

        typename ForceField::SPtr forceField = addNew<ForceField>(node);
        forceField->setForce( 0, force );
        forceField->setForce( 1, force );

        /// Let's fix the particle's movement for particle number 1 (the second one).
        typename FixedProjectiveConstraint::SPtr cst = sofa::core::objectmodel::New<FixedProjectiveConstraint>();
        cst->findData("indices")->read("1");

        node->addObject(cst);

        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        /// Perform one time step
        sofa::simulation::node::animate(root.get(), 0.5);

        /// Check if the first particle moved...this one should because it is not fixed
        /// so it is a failure if the particle is not moving at all.
        if( (readX[0]-initCoord1).norm2() < epsilon )
        {
            ADD_FAILURE() << "similar position between ["
                          << "-> " << readX[0] << ", " << readX[1]
                          << "] and ["
                          << "-> " << initCoord1 << ", " << initCoord2 << "] while they shouldn't." << std::endl;
            return false;
        }

        /// Check if the second particle have not moved. This one shouldn't move so it is
        /// a failure if it does.
        if( (readX[1]-initCoord2).norm2() >= epsilon )
        {
            ADD_FAILURE() << "different position between [" << readX[0] << ", ->" << readX[1]
                                                            << "] and ["
                                                            << initCoord1 << ", ->" << initCoord2 << "] while they shouldn't." << std::endl;
            return false;
        }

        std::cout << " position [" << readX[0] << ", ->" << readX[1]
                                                                    << "] and ["
                                                                    << initCoord1 << ", ->" << initCoord2 << "] " << std::endl;


        return true;

    }

    bool testTopologicalChanges()
    {
        simulation::Simulation* simulation = sofa::simulation::getSimulation();
        assert(simulation);
        
        /// Scene creation
        const simulation::Node::SPtr root = simulation->createNewGraph("root");
        root->setGravity(type::Vec3(0, 0, 0));

        /// Create euler solver
        simulation::Node::SPtr node = createEulerSolverNode(root, "test");

        /// Add mechanicalObject
        sofa::Size nbrDofs = 5;
        typename MechanicalObject::SPtr dofs = addNew<MechanicalObject>(node);
        dofs->resize(nbrDofs);
        
        /// Add PointSetTopology
        const auto tCon = sofa::core::objectmodel::New<sofa::component::topology::container::dynamic::PointSetTopologyContainer>();
        const auto tMod = sofa::core::objectmodel::New<sofa::component::topology::container::dynamic::PointSetTopologyModifier>();
        tCon->setNbPoints(nbrDofs);
        node->addObject(tCon);
        node->addObject(tMod);

        /// Add Mass
        createUniformMass<DataTypes>(node, *dofs.get());

        /// Add force
        typename ForceField::SPtr forceField = addNew<ForceField>(node);
        
        // create a force vector
        Deriv force;
        auto typeSize = force.size();
        
        for (unsigned i = 0; i < typeSize; i++)
            force[i] = 10;

        /// Fill position and force
        typename MechanicalObject::WriteVecCoord writeX = dofs->writePositions();
        for (sofa::Index id = 0; id < nbrDofs; id++)
        {
            for (unsigned int i = 0; i < typeSize; i++)
            {
                writeX[id][i] = id + 0.1 * i;  // create position filled as a grid:  0; 0.1; 0.2 ...  \n  1; 1.1; 1.2 ... \n 2; 2.1; 2.2 ...
            }

            forceField->setForce(id, force);
        }

        /// Add fixconstraint
        typename FixedProjectiveConstraint::SPtr cst = sofa::core::objectmodel::New<FixedProjectiveConstraint>();
        type::vector<Index> indices = { 0, 1, 2 };
        cst->d_indices.setValue(indices);
        node->addObject(cst);


        /// Init simulation
        sofa::simulation::node::initRoot(root.get());

        /// Perform two time steps
        sofa::simulation::node::animate(root.get(), 0.1);
        sofa::simulation::node::animate(root.get(), 0.1);

        typename MechanicalObject::ReadVecCoord readX = dofs->readPositions();

        /// check info before topological changes
        EXPECT_EQ(tCon->getNbPoints(), nbrDofs);
        EXPECT_EQ(readX[0][0], 0);
        EXPECT_EQ(readX[1][0], 1);
        EXPECT_EQ(readX[2][0], 2);
        EXPECT_NEAR(readX[3][0], 4.32231, 1e-4); // compare to computed values
        EXPECT_NEAR(readX[4][0], 5.32231, 1e-4);

        /// remove some points from topological mechanism
        sofa::type::vector< sofa::Index > indicesRemove = {0, 2, 3};
        tMod->removePoints(indicesRemove, true);
        nbrDofs -= sofa::Size(indicesRemove.size());

        /// new positions are now: {id[4], id[1]}  because remove use swap + pop_back
        EXPECT_EQ(tCon->getNbPoints(), nbrDofs);
        EXPECT_NEAR(readX[0][0], 5.32231, 1e-4);
        EXPECT_NEAR(readX[1][0], 1.0, 1e-4);

        return true;
    }

};

// Define the list of DataTypes to instanciate
using ::testing::Types;
typedef Types<
    defaulttype::Vec1Types,
    defaulttype::Vec2Types,
    defaulttype::Vec3Types,
    defaulttype::Vec6Types,
    defaulttype::Rigid2Types,
    defaulttype::Rigid3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE(FixedProjectiveConstraint_test, DataTypes);
// first test case
TYPED_TEST( FixedProjectiveConstraint_test , testValueImplicitWithCG )
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_TRUE(  this->test(1e-8,std::string("Implicit")) );
}

TYPED_TEST( FixedProjectiveConstraint_test , testValueExplicit )
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_TRUE(  this->test(1e-8, std::string("Explicit")) );
}

TYPED_TEST( FixedProjectiveConstraint_test , testValueImplicitWithSparseLDL )
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_TRUE(  this->test(1e-8, std::string("Implicit_SparseLDL")) );
}

TYPED_TEST(FixedProjectiveConstraint_test, testTopologicalChanges)
{
    EXPECT_MSG_NOEMIT(Error);
    EXPECT_TRUE(this->testTopologicalChanges());
}

}// namespace
}// namespace sofa






