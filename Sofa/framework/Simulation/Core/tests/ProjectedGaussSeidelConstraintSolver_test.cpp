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
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <SofaSimulationGraph/SimpleApi.h>
using namespace sofa::simpleapi;

namespace
{

/** Test the UncoupledConstraintCorrection class
*/
struct ProjectedGaussSeidelConstraintSolver_test : BaseSimulationTest
{
    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component");
        sofa::simpleapi::importPlugin("Sofa.Component.Collision.Geometry");
        sofa::simpleapi::importPlugin("Sofa.Component.Collision.Detection.Intersection");
        sofa::simpleapi::importPlugin("Sofa.Component.Collision.Response.Contact");
    }

    void enableConstraintForce()
    {
        SceneInstance sceneinstance("xml",
                    "<Node>\n"
                    "   <RequiredPlugin name='Sofa.Component'/>"
                    "   <RequiredPlugin name='Sofa.Component.Collision.Geometry'/>"
                    "   <RequiredPlugin name='Sofa.Component.Collision.Detection.Intersection'/>"
                    "   <RequiredPlugin name='Sofa.Component.Collision.Response.Contact'/>"
                    "   <FreeMotionAnimationLoop />\n"
                    "   <ProjectedGaussSeidelConstraintSolver name='solver' constraintForces='-1 -1 -1' computeConstraintForces='True' maxIt='1000' tolerance='0.001' />\n"
                    "   <Node name='collision'>\n"
                    "         <MechanicalObject />\n"
                    "         <UncoupledConstraintCorrection useOdeSolverIntegrationFactors='0' />\n"
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
TEST_F(ProjectedGaussSeidelConstraintSolver_test, checkConstraintForce)
{
    EXPECT_MSG_NOEMIT(Error);
    enableConstraintForce();
}


} /// namespace sofa







