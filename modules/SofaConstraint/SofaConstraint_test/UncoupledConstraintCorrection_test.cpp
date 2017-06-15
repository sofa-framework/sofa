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
#include <SofaTest/Sofa_test.h>
#include <SofaTest/TestMessageHandler.h>


#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/CleanupVisitor.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaConstraint/FreeMotionAnimationLoop.h>
#include <SofaConstraint/UncoupledConstraintCorrection.h>

namespace sofa {

/** Test the UncoupledConstraintCorrection class
*/
struct UncoupledConstraintCorrection_test: public Sofa_test<SReal>
{
    // root
    simulation::Simulation* simulation;
    simulation::Node::SPtr root;

    UncoupledConstraintCorrection_test()
    {
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
    }

    /// create a component and replace it with an other one
    void objectRemovalThenStep()
    {
        typedef component::constraintset::LCPConstraintSolver LCPConstraintSolver;
        typedef component::animationloop::FreeMotionAnimationLoop FreeMotionAnimationLoop;
        typedef component::container::MechanicalObject<defaulttype::Vec3Types> MechanicalObject3;
        typedef component::constraintset::UncoupledConstraintCorrection<defaulttype::Vec3Types> UncoupledConstraintCorrection;

        root = simulation::getSimulation()->createNewGraph("root");

        LCPConstraintSolver::SPtr lcpConstraintSolver = core::objectmodel::New<LCPConstraintSolver>();
        lcpConstraintSolver->tol.setValue(0.001);
        lcpConstraintSolver->maxIt.setValue(1000);
        root->addObject(lcpConstraintSolver);
        root->addObject(core::objectmodel::New<FreeMotionAnimationLoop>(root.get()));

        simulation::Node::SPtr child = root->createChild("collision");
        child->addObject(core::objectmodel::New<MechanicalObject3>());
        child->addObject(core::objectmodel::New<UncoupledConstraintCorrection>());

        simulation->init(root.get());

        // removal
        {
            simulation::Node::SPtr nodeToRemove = child;
            nodeToRemove->detachFromGraph();
            nodeToRemove->execute<simulation::CleanupVisitor>(sofa::core::ExecParams::defaultInstance());
            nodeToRemove->execute<simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
        }

        simulation->animate(root.get(), 0.04);

        simulation->unload(root);
    }
};

// run the tests
TEST_F( UncoupledConstraintCorrection_test,objectRemovalThenStep) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->objectRemovalThenStep();
}


}// namespace sofa







