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
#include <sofa/core/topology/Topology.h>

#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa::core::topology
{

bool Topology::insertInNode(objectmodel::BaseNode* node)
{
    node->addTopology(this);
    Inherit1::insertInNode(node);
    return true;
}

bool Topology::removeInNode(objectmodel::BaseNode* node)
{
    node->removeTopology(this);
    Inherit1::removeInNode(node);
    return true;
}

// Tetrahedron
const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
const unsigned int trianglesOrientationInTetrahedronArray[4][3] = {{1,2,3}, {0,3,2}, {1,3,0}, {0,2,1}};

// Hexahedron
const unsigned int edgesInHexahedronArray[12][2] = {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};
const unsigned int quadsOrientationInHexahedronArray[6][4] = {{0,3,2,1}, {4,5,6,7}, {0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {3,0,4,7}};
const unsigned int verticesInHexahedronArray[2][2][2] = { {{0,4}, {3,7}}, {{1,5}, {2,6}} };

} // namespace sofa::core::topology
