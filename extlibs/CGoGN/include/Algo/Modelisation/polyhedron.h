/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef POLYHEDRON_H
#define POLYHEDRON_H

#include <vector>
#include "Algo/Modelisation/subdivision.h"
#include "Geometry/transfo.h"
#include "Topology/generic/cellmarker.h"

#include "Utils/os_spec.h"


namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Modelisation
{

//enum { NONE, GRID, CUBE, CYLINDER, CONE, SPHERE, TORE, COMPOSED };

//template <typename PFP>
//void sewFaceEmb(typename PFP::MAP& map, Dart d, Dart e);
//
//template <typename PFP>
//Dart newFaceEmb(typename PFP::MAP& map, unsigned int n);

/**
* sudivide the all quads of a CC into 2 triangles
*/
 template <typename PFP>
 void quads2TrianglesCC(typename PFP::MAP& the_map, Dart primd);

/**
* Create a triangle fans (to close cylinders)
* simple topo creation  no modification of Polyhedron
* @param n nb of triangles in the fan in n
* @return the dart
*/
// template <typename PFP>
// Dart triangleFan_topo(typename PFP::MAP& the_map, int n);


/**
 * Unsex the Umbrella around a vertex, close the hole and then
 * create a symetric to construct a polyedron
 * @param d a dart from the vertex
 */
template <typename PFP>
void explodPolyhedron(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);




/**
 * create a n-sided pyramid
 */
template <typename PFP>
Dart createPyramid(typename PFP::MAP& map, unsigned int nbSides, bool withBoundary = true);

/**
 * create a n-sided prism
 */
template <typename PFP>
Dart createPrism(typename PFP::MAP& map, unsigned int nbSides, bool withBoundary = true);

/**
 * create a n-sided diamond
 */
template <typename PFP>
Dart createDiamond(typename PFP::MAP& map, unsigned int nbSides, bool withBoundary = true);

/**
 * create a tetrahedron
 */
template <typename PFP>
Dart createTetrahedron(typename PFP::MAP& map, bool withBoundary = true);

/**
 * create a hexahedron
 */
template <typename PFP>
Dart createHexahedron(typename PFP::MAP& map, bool withBoundary = true);

/**
 * create a 3-sided prism
 */
template <typename PFP>
Dart createTriangularPrism(typename PFP::MAP& map, bool withBoundary = true);

/**
 * create a 4-sided pyramid
 */
template <typename PFP>
Dart createQuadrangularPyramid(typename PFP::MAP& map, bool withBoundary = true);

/**
 * create 4-sided diamond (i.e. an octahedron)
 */
template <typename PFP>
Dart createOctahedron(typename PFP::MAP& map, bool withBoundary = true);

//TODO optimize
template <typename PFP>
bool isPyra(typename PFP::MAP& map, Dart d, unsigned int thread = 0);

//TODO optimize
template <typename PFP>
bool isPrism(typename PFP::MAP& map, Dart d, unsigned int thread = 0);


template <typename PFP>
Dart embedPrism(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, unsigned int n, bool withBoundary, float bottom_radius, float top_radius, float height);


template <typename PFP>
Dart embedPyramid(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, unsigned int n, bool withBoundary, float radius, float height);

} // namespace Modelisation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Modelisation/polyhedron.hpp"

#endif
