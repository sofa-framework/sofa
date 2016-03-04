/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "../Node_test.h"
#include <gtest/gtest.h>
#include <sofa/simulation/tree/GNode.h>

using sofa::simulation::tree::GNode;

TEST(GNodeTest, objectDestruction_singleObject)
{
    Node_test_objectDestruction_singleObject<GNode>();
}

TEST(GNodeTest, objectDestruction_multipleObjects)
{
    Node_test_objectDestruction_multipleObjects<GNode>();
}

TEST(GNodeTest, objectDestruction_childNode_singleObject)
{
    Node_test_objectDestruction_childNode_singleObject<GNode>();
}

TEST(GNodeTest, objectDestruction_childNode_complexChild)
{
    Node_test_objectDestruction_childNode_complexChild<GNode>();
}
