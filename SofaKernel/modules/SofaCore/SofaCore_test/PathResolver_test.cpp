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
#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

#include <sofa/core/PathResolver.h>
using sofa::core::PathResolver;

class PathResolver_test: public BaseTest{};

TEST_F(PathResolver_test, checkSyntaxValidPath)
{
    ASSERT_TRUE(PathResolver::PathHasValidSyntax("/root/node1/object.position"));
    ASSERT_TRUE(PathResolver::PathHasValidSyntax("/root/node1/object"));
    ASSERT_TRUE(PathResolver::PathHasValidSyntax("/"));
    ASSERT_TRUE(PathResolver::PathHasValidSyntax(""));
}

TEST_F(PathResolver_test, checkSyntaxInvalidPath)
{
    ASSERT_FALSE(PathResolver::PathHasValidSyntax("@/a/path/is/not/and/address"));
    ASSERT_FALSE(PathResolver::PathHasValidSyntax("/a/t\\dq/p"));
    ASSERT_FALSE(PathResolver::PathHasValidSyntax("/a/t\\no space/p"));
}
