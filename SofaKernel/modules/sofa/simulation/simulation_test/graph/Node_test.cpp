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
#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;

#include <SofaSimulationGraph/SimpleApi.h>
using namespace sofa::simpleapi ;

namespace sofa {

struct Node_test : public BaseSimulationTest
{
    void test1()
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
        EXPECT_MSG_NOEMIT(Error, Warning);

        SceneInstance si("R") ;
        Node::SPtr A = createChild(si.root, "A");
        Node::SPtr B = createChild(A, "B");

        EXPECT_EQ("", si.root->getPathName());
        EXPECT_EQ("/A/B", B->getPathName());
    }
};

TEST_F( Node_test, getPathName)
{
    this->test1() ;
}

}// namespace sofa







