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

#include <SofaSimulationGraph/SimpleApi.h>
using namespace sofa::simpleapi;

namespace
{

/** Test the UncoupledConstraintCorrection class
*/
struct GenericConstraintSolver_test : BaseSimulationTest
{
    void SetUp()
    {
        sofa::simpleapi::importPlugin("SofaAllCommonComponents");
        sofa::simpleapi::importPlugin("SofaMiscCollision");
    }

    void enableConstraintForce()
    {
        SceneInstance sceneinstance("xml",
                    "<Node>\n"
                    "   <RequiredPlugin name='SofaAllCommonComponents'/>"
                    "   <RequiredPlugin name='SofaMiscCollision'/>"
                    "   <FreeMotionAnimationLoop />\n"
                    "   <GenericConstraintSolver name='solver' constraintForces='-1 -1 -1' computeConstraintForces='True' maxIt='1000' tolerance='0.001' />\n"
                    "   <Node name='collision'>\n"
                    "         <MechanicalObject />\n"
                    "         <UncoupledConstraintCorrection />\n"
                    "   </Node>\n"
                    "</Node>\n"
                    );

        sceneinstance.initScene();
        sceneinstance.simulate(0.01);
        auto solver = sceneinstance.root->getObject("solver");
        ASSERT_NE(solver, nullptr);
        ASSERT_STREQ(solver->findData("constraintForces")->getValueString().c_str(), "");
    }
};

/// run the tests
TEST_F(GenericConstraintSolver_test, checkConstraintForce)
{
    EXPECT_MSG_NOEMIT(Error);
    enableConstraintForce();
}


} /// namespace sofa







