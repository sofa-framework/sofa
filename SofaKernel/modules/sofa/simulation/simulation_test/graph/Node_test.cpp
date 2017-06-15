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
#include <SceneCreator/SceneCreator.h>

#include <sofa/simulation/Node.h>

namespace sofa {

using namespace modeling;

struct Node_test : public Sofa_test<>
{
    Node_test()
    {
        /* create trivial DAG :
         *
         * R
         * |
         * A
         * |
         * B
         *
         */
        sofa::simulation::setSimulation(new simulation::graph::DAGSimulation());
        root = clearScene();
        root->setName("R");
        A = root->createChild("A");
        B = A->createChild("B");

    }

    simulation::Node::SPtr root;
    simulation::Node::SPtr A;
    simulation::Node::SPtr B;

};

TEST_F( Node_test, getPathName)
{
    EXPECT_MSG_NOEMIT(Error, Warning);

    EXPECT_EQ("", root->getPathName());
    EXPECT_EQ("/A/B", B->getPathName());
}

}// namespace sofa







