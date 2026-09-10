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


// Quadratic triangle subdivision
const unsigned int trianglesInQuadraticTriangles[4][3] = { {0,3,4}, {3,1,5}, {3,5,4}, {4,5,2}};

// Quadratic Quad subdivision into 4 linear quads
const unsigned int quadsInQuadraticQuads[4][4] = {
    {0, 4, 8, 7}, // Sub-quad 0
    {4, 1, 5, 8}, // Sub-quad 1
    {8, 5, 2, 6}, // Sub-quad 2
    {7, 8, 6, 3}  // Sub-quad 3
};

// Tetrahedron
const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
const unsigned int trianglesOrientationInTetrahedronArray[4][3] = {{1,2,3}, {0,3,2}, {1,3,0}, {0,2,1}};

// Hexahedron
const unsigned int edgesInHexahedronArray[12][2] = {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};
const unsigned int quadsOrientationInHexahedronArray[6][4] = {{0,3,2,1}, {4,5,6,7}, {0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {3,0,4,7}};
const unsigned int verticesInHexahedronArray[2][2][2] = { {{0,4}, {3,7}}, {{1,5}, {2,6}} };

// Quadratic tetrahedron
const unsigned int quadraticTrianglesInQuadraticTetrahedronArray[4][6] = { {0,1,3,4,6,8}, {2,0,3,5,9,6}, {1,2,3,7,8,9}, {2,1,0,7,5,4} };

// Quadratic hexahedron (6 quadratic quads of 9 nodes each)
const unsigned int quadraticQuadsInQuadraticHexahedronArray[6][9] = {
    {0,4,7,3,10,17,15,9,25},    // Face 0: -X
    {1,2,6,5,11,14,18,12,23},  // Face 1: +X
    {0,1,5,4,8,12,16,10,22},   // Face 2: -Y
    {2,3,7,6,13,15,19,14,24},  // Face 3: +Y
    {1,0,3,2,8,9,13,11,20},  // Face 4: -Z
    {6,7,4,5,19,17,16,18,21}    // Face 5: +Z
};



} // namespace sofa::core::topology
