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
#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest;

#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/CleanupVisitor.h>
using namespace sofa::simulation;

namespace
{

/** Test the UncoupledConstraintCorrection class */
struct UncoupledConstraintCorrection_test: public BaseSimulationTest
{
    /// create a component and replace it with an other one
    void objectRemovalThenStep()
    {
        SceneInstance sceneinstance("xml"
                    "<Node>\n"
                    "   <LCPConstraintSolver maxIt='1000' tol='0.001' />\n"
                    "   <FreeMotionAnimationLoop />\n"
                    "   <UncoupledConstraintCorrection />\n"
                    "   <Node name='collision'>\n"
                    "         <MechanicalObject />\n"
                    "         <UncoupledConstraintCorrection />\n"
                    "   </Node>\n"
                    "</Node>\n"
                    );

        sceneinstance.initScene();

        /// removal
        sofa::core::sptr<sofa::simulation::Node> nodeToRemove = sceneinstance.root->getTreeNode("collision");
        nodeToRemove->detachFromGraph();
        nodeToRemove->execute<sofa::simulation::CleanupVisitor>(sofa::core::ExecParams::defaultInstance());
        nodeToRemove->execute<sofa::simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
        sceneinstance.simulate(0.04);
    }
};

/// run the tests
TEST_F( UncoupledConstraintCorrection_test,objectRemovalThenStep) {
    EXPECT_MSG_NOEMIT(Error) ;
    objectRemovalThenStep();
}

}/// namespace sofa







