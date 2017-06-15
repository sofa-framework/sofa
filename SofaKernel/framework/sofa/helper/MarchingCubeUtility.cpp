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
/**** Geometry Shader Marching Cubes
  * Copyright Cyril Crassin, tt 2007.
  * This code is partially based on the example of
  * Paul Bourke "Polygonising a scalar field" located at :
  * http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
****/

#include <sofa/helper/MarchingCubeUtility.h>
#include <stack>

#define PRECISION 16384.0

#include <string.h>
#include <set>


namespace sofa
{

namespace helper
{
using std::stack;

const int MarchingCubeEdgeTable[256] =
{
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

const int MarchingCubeFaceTable[256] =
{
    0x0 , 0x19, 0x15, 0x1d, 0x25, 0x3d, 0x35, 0x3d, 0x29, 0x39,
    0x3d, 0x3d, 0x2d, 0x3d, 0x3d, 0x3c, 0x1a, 0x1b, 0x1f, 0x1f,
    0x3f, 0x3f, 0x3f, 0x3f, 0x3b, 0x3b, 0x3f, 0x3f, 0x3f, 0x3f,
    0x3f, 0x3e, 0x16, 0x1f, 0x17, 0x1f, 0x37, 0x3f, 0x37, 0x3f,
    0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3e, 0x1e, 0x1f,
    0x1f, 0xf , 0x3f, 0x3f, 0x3f, 0x2f, 0x3f, 0x3f, 0x3f, 0x2f,
    0x3f, 0x3f, 0x3f, 0x2e, 0x26, 0x3f, 0x37, 0x3f, 0x27, 0x3f,
    0x37, 0x3f, 0x2f, 0x3f, 0x3f, 0x3f, 0x2f, 0x3f, 0x3f, 0x3e,
    0x3e, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
    0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3e, 0x36, 0x3f, 0x37, 0x3f,
    0x37, 0x3f, 0x33, 0x3b, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
    0x3b, 0x3a, 0x3e, 0x3f, 0x3f, 0x2f, 0x3f, 0x3f, 0x3b, 0x2b,
    0x3f, 0x3f, 0x3f, 0x2f, 0x3f, 0x3f, 0x3b, 0x2a, 0x2a, 0x3b,
    0x3f, 0x3f, 0x2f, 0x3f, 0x3f, 0x3f, 0x2b, 0x3b, 0x3f, 0x3f,
    0x2f, 0x3f, 0x3f, 0x3e, 0x3a, 0x3b, 0x3f, 0x3f, 0x3f, 0x3f,
    0x3f, 0x3f, 0x3b, 0x33, 0x3f, 0x37, 0x3f, 0x37, 0x3f, 0x36,
    0x3e, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
    0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3e, 0x3e, 0x3f, 0x3f, 0x2f,
    0x3f, 0x3f, 0x3f, 0x2f, 0x3f, 0x37, 0x3f, 0x27, 0x3f, 0x37,
    0x3f, 0x26, 0x2e, 0x3f, 0x3f, 0x3f, 0x2f, 0x3f, 0x3f, 0x3f,
    0x2f, 0x3f, 0x3f, 0x3f, 0xf , 0x1f, 0x1f, 0x1e, 0x3e, 0x3f,
    0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x37, 0x3f, 0x37,
    0x1f, 0x17, 0x1f, 0x16, 0x3e, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
    0x3b, 0x3b, 0x3f, 0x3f, 0x3f, 0x3f, 0x1f, 0x1f, 0x1b, 0x1a,
    0x3c, 0x3d, 0x3d, 0x2d, 0x3d, 0x3d, 0x39, 0x29, 0x3d, 0x35,
    0x3d, 0x25, 0x1d, 0x15, 0x19, 0x0
};

const int MarchingCubeTriTable[256][16] =
{
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};


MarchingCubeUtility::MarchingCubeUtility()
    : cubeStep ( 1 ), convolutionSize ( 1 ),
      dataResolution ( 0,0,0 ), dataVoxelSize ( 1.0f,1.0f,1.0f ),
      verticesIndexOffset( 0), verticesTranslation( 0,0,0)
{
    // // Computes non trivial faces.
    // int nonTrivialFaces[256];
    //    for( int i = 0; i < 256; i++)
    // {
    // nonTrivialFaces[i] = 0;
    // }
    //
    // for( int i = 0; i < 256; i++)
    // {
    //    if ( MarchingCubeEdgeTable[i] &    1 ) {nonTrivialFaces[i] |= (1 << 0); nonTrivialFaces[i] |= (1 << 4);}
    //    if ( MarchingCubeEdgeTable[i] &    2 ) {nonTrivialFaces[i] |= (1 << 0); nonTrivialFaces[i] |= (1 << 2);}
    //    if ( MarchingCubeEdgeTable[i] &    4 ) {nonTrivialFaces[i] |= (1 << 0); nonTrivialFaces[i] |= (1 << 5);}
    //    if ( MarchingCubeEdgeTable[i] &    8 ) {nonTrivialFaces[i] |= (1 << 0); nonTrivialFaces[i] |= (1 << 3);}
    //    if ( MarchingCubeEdgeTable[i] &   16 ) {nonTrivialFaces[i] |= (1 << 1); nonTrivialFaces[i] |= (1 << 4);}
    //    if ( MarchingCubeEdgeTable[i] &   32 ) {nonTrivialFaces[i] |= (1 << 1); nonTrivialFaces[i] |= (1 << 2);}
    //    if ( MarchingCubeEdgeTable[i] &   64 ) {nonTrivialFaces[i] |= (1 << 1); nonTrivialFaces[i] |= (1 << 5);}
    //    if ( MarchingCubeEdgeTable[i] &  128 ) {nonTrivialFaces[i] |= (1 << 1); nonTrivialFaces[i] |= (1 << 3);}
    //    if ( MarchingCubeEdgeTable[i] &  256 ) {nonTrivialFaces[i] |= (1 << 3); nonTrivialFaces[i] |= (1 << 4);}
    //    if ( MarchingCubeEdgeTable[i] &  512 ) {nonTrivialFaces[i] |= (1 << 4); nonTrivialFaces[i] |= (1 << 2);}
    //    if ( MarchingCubeEdgeTable[i] & 1024 ) {nonTrivialFaces[i] |= (1 << 2); nonTrivialFaces[i] |= (1 << 5);}
    //    if ( MarchingCubeEdgeTable[i] & 2048 ) {nonTrivialFaces[i] |= (1 << 5); nonTrivialFaces[i] |= (1 << 3);}
    // }
    //
    //    for( int i = 0; i < 256; i++)
    //      std::printf( "0x%2x, ", nonTrivialFaces[i]);
}


/*
    Linearly interpolate the position where an isosurface cuts
    an edge between two vertices, each with their own scalar value
    */
void MarchingCubeUtility::vertexInterp ( Vector3 &p, const float isolevel,
        const Vector3 &p1, const Vector3 &p2,
        const float valp1, const float valp2 ) const
{
    float mu = ( isolevel - valp1 ) / ( valp2 - valp1 );
    p = p1 + ( p2 - p1 ) * mu;
    p = ( ( p + Vector3 ( 1.0f, 1.0f, 1.0f ) ) *0.5f ).linearProduct ( dataVoxelSize.linearProduct ( dataResolution ) ) + dataVoxelSize/2.0;
    p += verticesTranslation;
    p[0] = ( int ) helper::round( p[0] * (SReal)PRECISION ) / (SReal)PRECISION;
    p[1] = ( int ) helper::round( p[1] * (SReal)PRECISION ) / (SReal)PRECISION;
    p[2] = ( int ) helper::round( p[2] * (SReal)PRECISION ) / (SReal)PRECISION;
}


bool MarchingCubeUtility::testGrid ( const float v, const float isolevel ) const
{
    return ( v<isolevel );
}


void MarchingCubeUtility::initCell ( GridCell& cell, const Vec3i& coord, const unsigned char* data, const Vector3& gridStep, const Vec3i& dataGridStep ) const
{
    Vector3 vcurf ( ( float ) coord[0], ( float ) coord[1], ( float ) coord[2] );

    cell.pos[0]=vcurf.linearProduct ( gridStep )-Vector3 ( 1.0f,1.0f,1.0f );
    Vec3i valPos0=coord.linearProduct ( dataGridStep );
    cell.val[0]=(float)((( valPos0[0] >= roi.min[0]) && ( valPos0[1] >= roi.min[1]) && ( valPos0[2] >= roi.min[2]) && ( valPos0[0] < roi.max[0]) && ( valPos0[1] < roi.max[1]) && ( valPos0[2] < roi.max[2]))?data[valPos0[0] + valPos0[1]*dataResolution[0] + valPos0[2]*dataResolution[0]*dataResolution[1]]:0);

    Vec3i valPos;

    cell.pos[1]=cell.pos[0]+Vector3 ( gridStep[0], 0, 0 );
    valPos=valPos0+Vec3i ( dataGridStep[0], 0, 0 );
    cell.val[1]=(float)((( valPos[0] >= roi.min[0]) && ( valPos[1] >= roi.min[1]) && ( valPos[2] >= roi.min[2]) && ( valPos[0] < roi.max[0]) && ( valPos[1] < roi.max[1]) && ( valPos[2] < roi.max[2]))?data[valPos[0] + valPos[1]*dataResolution[0] + valPos[2]*dataResolution[0]*dataResolution[1]]:0);

    cell.pos[2]=cell.pos[0]+Vector3 ( gridStep[0], gridStep[1], 0 );
    valPos=valPos0+Vec3i ( dataGridStep[0], dataGridStep[1], 0 );
    cell.val[2]=(float)((( valPos[0] >= roi.min[0]) && ( valPos[1] >= roi.min[1]) && ( valPos[2] >= roi.min[2]) && ( valPos[0] < roi.max[0]) && ( valPos[1] < roi.max[1]) && ( valPos[2] < roi.max[2]))?data[valPos[0] + valPos[1]*dataResolution[0] + valPos[2]*dataResolution[0]*dataResolution[1]]:0);

    cell.pos[3]=cell.pos[0]+Vector3 ( 0, gridStep[1], 0 );
    valPos=valPos0+Vec3i ( 0, dataGridStep[1], 0 );
    cell.val[3]=(float)((( valPos[0] >= roi.min[0]) && ( valPos[1] >= roi.min[1]) && ( valPos[2] >= roi.min[2]) && ( valPos[0] < roi.max[0]) && ( valPos[1] < roi.max[1]) && ( valPos[2] < roi.max[2]))?data[valPos[0] + valPos[1]*dataResolution[0] + valPos[2]*dataResolution[0]*dataResolution[1]]:0);

    cell.pos[4]=cell.pos[0]+Vector3 ( 0, 0, gridStep[2] );
    valPos=valPos0+Vec3i ( 0, 0, dataGridStep[2] );
    cell.val[4]=(float)((( valPos[0] >= roi.min[0]) && ( valPos[1] >= roi.min[1]) && ( valPos[2] >= roi.min[2]) && ( valPos[0] < roi.max[0]) && ( valPos[1] < roi.max[1]) && ( valPos[2] < roi.max[2]))?data[valPos[0] + valPos[1]*dataResolution[0] + valPos[2]*dataResolution[0]*dataResolution[1]]:0);

    cell.pos[5]=cell.pos[0]+Vector3 ( gridStep[0], 0, gridStep[2] );
    valPos=valPos0+Vec3i ( dataGridStep[0], 0, dataGridStep[2] );
    cell.val[5]=(float)((( valPos[0] >= roi.min[0]) && ( valPos[1] >= roi.min[1]) && ( valPos[2] >= roi.min[2]) && ( valPos[0] < roi.max[0]) && ( valPos[1] < roi.max[1]) && ( valPos[2] < roi.max[2]))?data[valPos[0] + valPos[1]*dataResolution[0] + valPos[2]*dataResolution[0]*dataResolution[1]]:0);

    cell.pos[6]=cell.pos[0]+Vector3 ( gridStep[0], gridStep[1], gridStep[2] );
    valPos=valPos0+Vec3i ( dataGridStep[0], dataGridStep[1], dataGridStep[2] );
    cell.val[6]=(float)((( valPos[0] >= roi.min[0]) && ( valPos[1] >= roi.min[1]) && ( valPos[2] >= roi.min[2]) && ( valPos[0] < roi.max[0]) && ( valPos[1] < roi.max[1]) && ( valPos[2] < roi.max[2]))?data[valPos[0] + valPos[1]*dataResolution[0] + valPos[2]*dataResolution[0]*dataResolution[1]]:0);

    cell.pos[7]=cell.pos[0]+Vector3 ( 0, gridStep[1], gridStep[2] );
    valPos=valPos0+Vec3i ( 0, dataGridStep[1], dataGridStep[2] );
    cell.val[7]=(float)((( valPos[0] >= roi.min[0]) && ( valPos[1] >= roi.min[1]) && ( valPos[2] >= roi.min[2]) && ( valPos[0] < roi.max[0]) && ( valPos[1] < roi.max[1]) && ( valPos[2] < roi.max[2]))?data[valPos[0] + valPos[1]*dataResolution[0] + valPos[2]*dataResolution[0]*dataResolution[1]]:0);
}

/*
    Given a grid cell and an isolevel, calculate the triangular
    facets required to represent the isosurface through the cell.
    Return the number of triangular facets, the array "triangles"
    will be loaded up with the vertices at most 5 triangular facets.
    0 will be returned if the grid cell is either totally above
    of totally below the isolevel.
    */
int MarchingCubeUtility::polygonise ( const GridCell &grid, int& cubeConf, float isolevel, sofa::helper::vector< PointID > &triangles,
        std::map< Vector3, PointID> &map_vertices, sofa::helper::vector< Vector3 > &map_indices ) const
{

    int i,ntriang;
    Vector3 vertindex[12];

    /*
      Determine the index into the edge table which
      tells us which vertices are inside of the surface
      */
    cubeConf = 0;
    if ( testGrid ( grid.val[0], isolevel ) ) cubeConf |= 1;
    if ( testGrid ( grid.val[1], isolevel ) ) cubeConf |= 2;
    if ( testGrid ( grid.val[2], isolevel ) ) cubeConf |= 4;
    if ( testGrid ( grid.val[3], isolevel ) ) cubeConf |= 8;
    if ( testGrid ( grid.val[4], isolevel ) ) cubeConf |= 16;
    if ( testGrid ( grid.val[5], isolevel ) ) cubeConf |= 32;
    if ( testGrid ( grid.val[6], isolevel ) ) cubeConf |= 64;
    if ( testGrid ( grid.val[7], isolevel ) ) cubeConf |= 128;

    /* Cube is entirely in/out of the surface */
    if ( MarchingCubeEdgeTable[cubeConf] == 0 ) return 0;

    /* Find the vertices where the surface intersects the cube */
    if ( MarchingCubeEdgeTable[cubeConf] & 1 )
        vertexInterp ( vertindex[0], isolevel,grid.pos[0],grid.pos[1],grid.val[0],grid.val[1] );
    if ( MarchingCubeEdgeTable[cubeConf] & 2 )
        vertexInterp ( vertindex[1], isolevel,grid.pos[1],grid.pos[2],grid.val[1],grid.val[2] );
    if ( MarchingCubeEdgeTable[cubeConf] & 4 )
        vertexInterp ( vertindex[2], isolevel,grid.pos[2],grid.pos[3],grid.val[2],grid.val[3] );
    if ( MarchingCubeEdgeTable[cubeConf] & 8 )
        vertexInterp ( vertindex[3], isolevel,grid.pos[3],grid.pos[0],grid.val[3],grid.val[0] );
    if ( MarchingCubeEdgeTable[cubeConf] & 16 )
        vertexInterp ( vertindex[4], isolevel,grid.pos[4],grid.pos[5],grid.val[4],grid.val[5] );
    if ( MarchingCubeEdgeTable[cubeConf] & 32 )
        vertexInterp ( vertindex[5], isolevel,grid.pos[5],grid.pos[6],grid.val[5],grid.val[6] );
    if ( MarchingCubeEdgeTable[cubeConf] & 64 )
        vertexInterp ( vertindex[6], isolevel,grid.pos[6],grid.pos[7],grid.val[6],grid.val[7] );
    if ( MarchingCubeEdgeTable[cubeConf] & 128 )
        vertexInterp ( vertindex[7], isolevel,grid.pos[7],grid.pos[4],grid.val[7],grid.val[4] );
    if ( MarchingCubeEdgeTable[cubeConf] & 256 )
        vertexInterp ( vertindex[8], isolevel,grid.pos[0],grid.pos[4],grid.val[0],grid.val[4] );
    if ( MarchingCubeEdgeTable[cubeConf] & 512 )
        vertexInterp ( vertindex[9], isolevel,grid.pos[1],grid.pos[5],grid.val[1],grid.val[5] );
    if ( MarchingCubeEdgeTable[cubeConf] & 1024 )
        vertexInterp ( vertindex[10], isolevel,grid.pos[2],grid.pos[6],grid.val[2],grid.val[6] );
    if ( MarchingCubeEdgeTable[cubeConf] & 2048 )
        vertexInterp ( vertindex[11], isolevel,grid.pos[3],grid.pos[7],grid.val[3],grid.val[7] );

    /* Create the triangle */
    ntriang = 0;
    std::map< Vector3, PointID>::iterator iter;
    Vector3 current_P;
    PointID current_ID;
    for ( i=0; MarchingCubeTriTable[cubeConf][i]!=-1; i+=3 )
    {
        Vec<3,unsigned int> current_triangle;
        for ( PointID j=0; j<3; ++j )
        {
            current_P = vertindex[MarchingCubeTriTable[cubeConf][i+j]];
            //Search if the current Vertex P is already stored with an ID
            iter = map_vertices.find ( current_P );
            if ( iter != map_vertices.end() ) current_ID = iter->second;
            else
            {
                //Add new Vertex in map
                current_ID = static_cast<PointID>(map_indices.size()) + verticesIndexOffset;
                map_indices.push_back ( current_P );
                map_vertices.insert ( std::make_pair ( current_P, current_ID ) );
            }
            current_triangle[j]=current_ID;
        }

        if ( current_triangle[0] == current_triangle[1] || current_triangle[0] == current_triangle[2] || current_triangle[2] == current_triangle[1] ) continue;
        triangles.push_back ( current_triangle[0] );
        triangles.push_back ( current_triangle[1] );
        triangles.push_back ( current_triangle[2] );

        ntriang+=3;
    }
    return ( ntriang );
}




void MarchingCubeUtility::propagateFrom ( const sofa::helper::vector<Vec3i>& coord,
        unsigned char* data,
        const float isolevel,
        sofa::helper::vector< PointID >& mesh,
        sofa::helper::vector< Vector3 >& vertices,
        std::set<Vec3i>& generatedCubes,
        std::map< Vector3, PointID>& map_vertices,
        helper::vector< helper::vector<unsigned int> >* triangleIndexInRegularGrid,
        bool propagate
                                        ) const
{
    Vec3i bboxMin = Vec3i ( bbox.min / cubeStep );
    Vec3i bboxMax = Vec3i ( bbox.max / cubeStep );
    Vec3i gridSize = Vec3i ( dataResolution /cubeStep );

    Vector3 gridStep = Vector3 ( 2.0f/ ( ( float ) gridSize[0] ), 2.0f/ ( ( float ) gridSize[1] ), 2.0f/ ( ( float ) gridSize[2] ) );

    Vec3i dataGridStep ( dataResolution[0]/gridSize[0],dataResolution[1]/gridSize[1],dataResolution[2]/gridSize[2] );

    Vec3i cubeCoord, nextCube;
    stack<Vec3i> cubesToGenerate; // Stack of cubes to generate.
    for( sofa::helper::vector<Vec3i>::const_iterator it = coord.begin(); it != coord.end(); ++it)
    {
        const Vec3i& voxel = *it;
        if ( ( voxel[0] >= bbox.min[0]-1 ) && ( voxel[1] >= bbox.min[1]-1 ) && ( voxel[2] >= bbox.min[2]-1 ) &&
                ( voxel[0] <= bbox.max[0] ) && ( voxel[1] <= bbox.max[1] ) && ( voxel[2] <= bbox.max[2] ) )
            cubesToGenerate.push ( *it ); // Adds the first non-trivial cube.
    }

    int cubeConf;
    while ( !cubesToGenerate.empty() )
    {
        cubeCoord = cubesToGenerate.top(); // Get the last cube on the stack.
        cubesToGenerate.pop();             // Remove it from the stack.

        if ( generatedCubes.find ( cubeCoord ) != generatedCubes.end() ) continue;

        GridCell cell;
        initCell ( cell, cubeCoord, data, gridStep, dataGridStep );

        int numvert = polygonise ( cell, cubeConf, isolevel, mesh, map_vertices, vertices );

        if ( triangleIndexInRegularGrid ) updateTriangleInRegularGridVector ( *triangleIndexInRegularGrid, cubeCoord, cell, numvert / 3 );

        if( propagate)
        {
            // Propagate
            generatedCubes.insert ( cubeCoord ); // spaceIndex cube has been polygonized

            if ( ( MarchingCubeFaceTable[cubeConf] &  1 ) && ( cubeCoord[2] >= bboxMin[2] ) ) { cubesToGenerate.push ( cubeCoord + Vec3i ( 0, 0,-1 ) );}
            if ( ( MarchingCubeFaceTable[cubeConf] &  2 ) && ( cubeCoord[2] <= bboxMax[2]-2 ) ) { cubesToGenerate.push ( cubeCoord + Vec3i ( 0, 0, 1 ) );}
            if ( ( MarchingCubeFaceTable[cubeConf] &  4 ) && ( cubeCoord[0] <= bboxMax[0]-2 ) ) { cubesToGenerate.push ( cubeCoord + Vec3i ( 1, 0, 0 ) );}
            if ( ( MarchingCubeFaceTable[cubeConf] &  8 ) && ( cubeCoord[0] >= bboxMin[0] ) ) { cubesToGenerate.push ( cubeCoord + Vec3i ( -1, 0, 0 ) );}
            if ( ( MarchingCubeFaceTable[cubeConf] & 16 ) && ( cubeCoord[1] >= bboxMin[1] ) ) { cubesToGenerate.push ( cubeCoord + Vec3i ( 0,-1, 0 ) );}
            if ( ( MarchingCubeFaceTable[cubeConf] & 32 ) && ( cubeCoord[1] <= bboxMax[1]-2 ) ) { cubesToGenerate.push ( cubeCoord + Vec3i ( 0, 1, 0 ) );}
        }
    }
}



void MarchingCubeUtility::run ( unsigned char *_data, const sofa::helper::vector< Vec3i > & seeds,
        const float isolevel,
        sofa::helper::vector< PointID >& mesh,
        sofa::helper::vector< Vector3>& vertices,
        std::map< Vector3, PointID>& map_vertices,
        helper::vector< helper::vector<unsigned int> >*triangleIndexInRegularGrid,
        bool propagate ) const
{
//    Vec3i gridSize = Vec3i ( dataResolution[0]/cubeStep, dataResolution[1]/cubeStep, dataResolution[2]/cubeStep );
    std::set<Vec3i> generatedCubes;

    size_t datasize = dataResolution[0]*dataResolution[1]*dataResolution[2];
    if ( datasize == 0 )
        return;

    unsigned char* data;
    bool smooth = false;
    if ( convolutionSize != 0 )
    {
        data = new unsigned char[datasize];
        memcpy ( data, _data, datasize*sizeof ( unsigned char ) );
        smoothData ( data );
        smooth = true;
    }
    else
    {
        data = _data;
    }

    propagateFrom ( seeds, data, isolevel, mesh, vertices, generatedCubes, map_vertices, triangleIndexInRegularGrid, propagate );
    if ( smooth )
        delete [] data;
}



void MarchingCubeUtility::run ( unsigned char *_data, const sofa::helper::vector< Vec3i > & seeds,
        const float isolevel,
        sofa::helper::vector< PointID >& mesh,
        sofa::helper::vector< Vector3>& vertices,
        helper::vector< helper::vector<unsigned int> >*triangleIndexInRegularGrid,
        bool propagate ) const
{
    std::map< Vector3, PointID> map_vertices;
    for ( size_t i = map_vertices.size(); i < vertices.size(); i++ )
        map_vertices.insert ( std::make_pair ( vertices[i], i ) );

    run( _data, seeds, isolevel, mesh, vertices, map_vertices, triangleIndexInRegularGrid, propagate);
}



void MarchingCubeUtility::run ( unsigned char *_data, const float isolevel,
        sofa::helper::vector< PointID >& mesh,
        sofa::helper::vector< Vector3 >& vertices,
        helper::vector< helper::vector<unsigned int> >* triangleIndexInRegularGrid ) const
{
    size_t datasize = dataResolution[0]*dataResolution[1]*dataResolution[2];

    if ( datasize == 0 )
        return;

    unsigned char* data;
    bool smooth = false;
    if ( convolutionSize != 0 )
    {
        data = new unsigned char[datasize];
        memcpy(data, _data, datasize*sizeof(unsigned char));
        smoothData ( data );
        smooth = true;
    }
    else
    {
        data = _data;
    }

    std::map< Vector3, PointID> map_vertices;
    for ( size_t i = 0; i < vertices.size(); i++ )
        map_vertices.insert ( std::make_pair ( vertices[i], i ) );

    Vec3i bboxMin = Vec3i ( bbox.min / cubeStep );
    Vec3i bboxMax = Vec3i ( bbox.max / cubeStep );
    Vec3i gridSize = Vec3i ( dataResolution /cubeStep );

    Vector3 gridStep = Vector3 ( 2.0f/ ( ( float ) gridSize[0] ), 2.0f/ ( ( float ) gridSize[1] ), 2.0f/ ( ( float ) gridSize[2] ) );

    Vec3i dataGridStep ( dataResolution[0]/gridSize[0],dataResolution[1]/gridSize[1],dataResolution[2]/gridSize[2] );

    int cubeConf;
    for ( int k=bboxMin[2]; k<bboxMax[2]-1; k++ )
        for ( int j=bboxMin[1]; j<bboxMax[1]-1; j++ )
            for ( int i=bboxMin[0]; i<bboxMax[0]-1; i++ )
            {
                GridCell cell;
                initCell ( cell, Vec3i ( i, j, k ), data, gridStep, dataGridStep );

                int numvert = polygonise ( cell, cubeConf, isolevel, mesh, map_vertices, vertices );

                if ( triangleIndexInRegularGrid ) updateTriangleInRegularGridVector ( *triangleIndexInRegularGrid, Vec3i ( i, j, k ), cell, numvert / 3 );
            }

    if (smooth)
        delete [] data;

}



void MarchingCubeUtility::run ( unsigned char *data, const float isolevel,
        sofa::helper::io::Mesh &m ) const
{
    using sofa::helper::vector;
    using sofa::defaulttype::Vector3;

    std::cout << "Creating Mesh using Marching Cubes\n";
    vector<Vector3> &vertices                 = m.getVertices();
    vector< vector < vector <int> > > &facets = m.getFacets();

    vector< PointID >       triangles;

    //Do the Marching Cube
    run ( data, isolevel, triangles, vertices );

    const size_t numTriangles = triangles.size() /3;
    facets.resize ( numTriangles, vector< vector < int > > ( 3, vector<int> ( 3, 0 ) ) );
    for ( size_t i=0; i<triangles.size(); /*i+=3*/ )
    {
        vector< vector< int > > &vertNormTexIndices = facets[i/3];
        vector<int> &vIndices = vertNormTexIndices[0];

        vIndices[0] = triangles[i++];
        vIndices[1] = triangles[i++];
        vIndices[2] = triangles[i++];
    }
}



// A priori, il n'y a pas de données sur les bords (tout du moins sur le premier voxel)
void MarchingCubeUtility::findSeeds ( vector<Vec3i>& seeds, const float isoValue, unsigned char *_data )
{
    std::cout << "MarchingCubeUtility::findSeeds(). Begining." << std::endl;
    //vector< unsigned char > data ( dataResolution[0]*dataResolution[1]*dataResolution[2] );
    std::set<unsigned int> parsedVoxels;
    size_t datasize = dataResolution[0]*dataResolution[1]*dataResolution[2];
    if ( datasize == 0 )
        return;
    uint8_t* data;
    bool smooth = false;
    if ( convolutionSize != 0 )
    {
        data = new uint8_t[datasize];
        memcpy(data, _data, datasize*sizeof(uint8_t));
        smoothData ( data );
        smooth = true;
    }
    else
    {
        data = _data;
    }

    Vec3i bboxMin = Vec3i ( bbox.min / cubeStep );
    Vec3i bboxMax = Vec3i ( bbox.max / cubeStep );
    Vec3i gridSize = Vec3i ( dataResolution /cubeStep );

    int index;
    for ( int k=bboxMin[2]; k<bboxMax[2]-1; k++ )
        for ( int j=bboxMin[1]; j<bboxMax[1]-1; j++ )
            for ( int i=bboxMin[0]; i<bboxMax[0]-1; i++ )
            {
                index = i + j*gridSize[0] + k*gridSize[0]*gridSize[1];

                if ( data[index] >= isoValue)
                {
                    Vec3i currentCube ( i, j , k );
                    if ( parsedVoxels.find ( index ) == parsedVoxels.end() )
                    {
                        seeds.push_back ( currentCube - Vector3 ( 1,0,0 ) );
                        // propager sur les autres voxels et les incrire ds parsedVoxels.
                        findConnectedVoxels ( parsedVoxels, isoValue, currentCube, data );
                    }
                }
            }
    if (smooth)
        delete [] data;
    std::cout << "MarchingCubeUtility::findSeeds(). Ending. Seeds: " << seeds << std::endl;
}



void MarchingCubeUtility::findSeedsFromRealCoords ( vector<Vec3i>& mCubeCoords, const vector<Vector3>& realCoords ) const
{
    mCubeCoords.clear();
    Vector3 gridSize = Vector3 ( (SReal) 1.0 / dataVoxelSize[0]*cubeStep, (SReal) 1.0 / dataVoxelSize[1]*cubeStep, (SReal) 1.0 / dataVoxelSize[2]*cubeStep );

    for ( vector<Vector3>::const_iterator it = realCoords.begin(); it != realCoords.end(); ++it )
    {
        Vec3i seed = ( ( *it ) - verticesTranslation - ( dataVoxelSize/2.0 ) ).linearProduct ( gridSize );
        mCubeCoords.push_back ( seed );
        assert ( seed[0] >= 0 );
        assert ( seed[1] >= 0 );
        assert ( seed[2] >= 0 );
        assert ( seed[0] < gridSize[0] );
        assert ( seed[1] < gridSize[1] );
        assert ( seed[2] < gridSize[2] );
    }
}



void MarchingCubeUtility::setVerticesIndexOffset( unsigned int verticesIndexOffset)
{
    this->verticesIndexOffset = verticesIndexOffset;
}



void MarchingCubeUtility::setVerticesTranslation( Vector3 verticesTranslation)
{
    this->verticesTranslation = verticesTranslation;
}



void MarchingCubeUtility::updateTriangleInRegularGridVector ( helper::vector< helper::vector<unsigned int /*regular grid space index*/> >& triangleIndexInRegularGrid, const Vec3i& coord, const GridCell& cell, unsigned int nbTriangles ) const
{
    vector<unsigned int> voxels;
    if ( cell.val[0] ) voxels.push_back ( ( coord[0]+0 ) + ( coord[1]+0 ) *dataResolution[0] + ( coord[2]+0 ) *dataResolution[0]*dataResolution[1] ); //les voxels occupes ds ce cube
    if ( cell.val[1] ) voxels.push_back ( ( coord[0]+1 ) + ( coord[1]+0 ) *dataResolution[0] + ( coord[2]+0 ) *dataResolution[0]*dataResolution[1] ); //les voxels occupes ds ce cube
    if ( cell.val[3] ) voxels.push_back ( ( coord[0]+0 ) + ( coord[1]+1 ) *dataResolution[0] + ( coord[2]+0 ) *dataResolution[0]*dataResolution[1] ); //les voxels occupes ds ce cube
    if ( cell.val[2] ) voxels.push_back ( ( coord[0]+1 ) + ( coord[1]+1 ) *dataResolution[0] + ( coord[2]+0 ) *dataResolution[0]*dataResolution[1] ); //les voxels occupes ds ce cube
    if ( cell.val[4] ) voxels.push_back ( ( coord[0]+0 ) + ( coord[1]+0 ) *dataResolution[0] + ( coord[2]+1 ) *dataResolution[0]*dataResolution[1] ); //les voxels occupes ds ce cube
    if ( cell.val[5] ) voxels.push_back ( ( coord[0]+1 ) + ( coord[1]+0 ) *dataResolution[0] + ( coord[2]+1 ) *dataResolution[0]*dataResolution[1] ); //les voxels occupes ds ce cube
    if ( cell.val[7] ) voxels.push_back ( ( coord[0]+0 ) + ( coord[1]+1 ) *dataResolution[0] + ( coord[2]+1 ) *dataResolution[0]*dataResolution[1] ); //les voxels occupes ds ce cube
    if ( cell.val[6] ) voxels.push_back ( ( coord[0]+1 ) + ( coord[1]+1 ) *dataResolution[0] + ( coord[2]+1 ) *dataResolution[0]*dataResolution[1] ); //les voxels occupes ds ce cube

    for ( size_t i = 0; i < nbTriangles; i++ )
    {
        triangleIndexInRegularGrid.push_back ( voxels );
    }
}



void MarchingCubeUtility::findConnectedVoxels ( std::set<unsigned int>& connectedVoxels, const float isoValue, const Vec3i& from, unsigned char* data )
{
    Vec3i bboxMin = Vec3i ( bbox.min / cubeStep );
    Vec3i bboxMax = Vec3i ( bbox.max / cubeStep );
    int minX = bboxMin[0];
    int minY = bboxMin[1];
    int minZ = bboxMin[2];
    int maxX = bboxMax[0]-1;
    int maxY = bboxMax[1]-1;
    int maxZ = bboxMax[2]-1;

    std::stack<Vec3i> voxelsToTest;
    voxelsToTest.push( from);

    while (! voxelsToTest.empty())
    {
        Vec3i coord = voxelsToTest.top();
        voxelsToTest.pop();

        int index = coord[0] + coord[1]*dataResolution[0] + coord[2]*dataResolution[0]*dataResolution[1];

        if ( connectedVoxels.find ( index ) != connectedVoxels.end() ) continue;

        if ( data[index] < isoValue ) continue;

        connectedVoxels.insert ( index );

        if ( coord[0] > minX ) voxelsToTest.push( coord + Vec3i (-1, 0, 0 ));
        if ( coord[0] < maxX ) voxelsToTest.push( coord + Vec3i ( 1, 0, 0 ));
        if ( coord[1] > minY ) voxelsToTest.push( coord + Vec3i ( 0,-1, 0 ));
        if ( coord[1] < maxY ) voxelsToTest.push( coord + Vec3i ( 0, 1, 0 ));
        if ( coord[2] > minZ ) voxelsToTest.push( coord + Vec3i ( 0, 0,-1 ));
        if ( coord[2] < maxZ ) voxelsToTest.push( coord + Vec3i ( 0, 0, 1 ));
    }
}



void MarchingCubeUtility::smoothData ( unsigned char *data ) const
{
    std::cout << "Smoothing Data using " << convolutionSize << "x"<< convolutionSize << "x"<< convolutionSize << " as gaussian convolution kernel\n";
    vector< float > convolutionKernel;
    createGaussianConvolutionKernel ( convolutionKernel );

    vector<unsigned char> input_data ( ( int ) ( ( dataResolution[0]+convolutionSize )
            * ( dataResolution[1]+convolutionSize )
            * ( dataResolution[2]+convolutionSize ) ),
            0 );

    for ( int k=0; k<dataResolution[2]; ++k )
        for ( int j=0; j<dataResolution[1]; ++j )
        {
            memcpy ( &input_data[0] + convolutionSize/2
                    + ( j + convolutionSize/2 ) * ( dataResolution[0]+convolutionSize )
                    + ( k + convolutionSize/2 ) * ( dataResolution[0]+convolutionSize ) * ( dataResolution[1]+convolutionSize ),
                    data + j*dataResolution[0] + k*dataResolution[0]*dataResolution[1],
                    sizeof ( unsigned char ) *dataResolution[0] );
        }

    for ( int k=0; k<dataResolution[2]; ++k )
        for ( int j=0; j<dataResolution[1]; ++j )
            for ( int i=0; i<dataResolution[0]; ++i )
            {
                applyConvolution ( &convolutionKernel[0], i,j,k, &input_data[0], data );
            }
}

void  MarchingCubeUtility::applyConvolution ( const float* convolutionKernel,
        unsigned int x, unsigned int y, unsigned int z,
        const unsigned char* input_data,
        unsigned char* output_data ) const
{
    const size_t index = x + dataResolution[0] * ( y + dataResolution[1] * z );
    output_data[index] = 0;
    size_t idx=0;
    for ( unsigned int k=0; k<convolutionSize; ++k )
        for ( unsigned int j=0; j<convolutionSize; ++j )
            for ( unsigned int i=0; i<convolutionSize; ++i )
            {
                output_data[index] += (unsigned char)(convolutionKernel[idx++]
                        * input_data[ ( x+i ) + ( dataResolution[0]+convolutionSize ) * ( ( y+j ) + ( z+k ) * ( dataResolution[1]+convolutionSize ) ) ]);
            }
}

void MarchingCubeUtility::createGaussianConvolutionKernel ( vector< float >  &convolutionKernel ) const
{
    if ( convolutionSize<=1 )
    {
        convolutionKernel.resize ( 1, 1.0f );
        return;
    }
    else
    {
        convolutionKernel.resize ( convolutionSize*convolutionSize*convolutionSize );
    }

    const float step = 4.0f / ( float ) ( convolutionSize-1 );

    float total = 0.0;
    size_t idx=0;
    for ( unsigned int k=0; k<convolutionSize; ++k )
        for ( unsigned int j=0; j<convolutionSize; ++j )
            for ( unsigned int i=0; i<convolutionSize; ++i )
            {
                const float x = -2.0f + i * step;
                const float y = -2.0f + j * step;
                const float z = -2.0f + k * step;
                convolutionKernel[idx] = ( float ) ( exp ( - 0.5f * ( x*x+y*y+z*z ) ) );
                total += convolutionKernel[idx++];
            }

    total = 1.0f/total;
    for ( size_t i=0; i<convolutionKernel.size(); ++i )
        convolutionKernel[i] *= total;
}

}

}
