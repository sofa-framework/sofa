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

#ifndef __DRAWER_CELLS__
#define __DRAWER_CELLS__

#include "Utils/drawer.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

/**
 * add a cell to a drawer
 * @param the cell (VERTEX,EDGE,...)
 *  * @param dr the drawer to use
 * @param map the map
 * @param d the dart
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerCells(unsigned int cell, Utils::Drawer& dr, typename PFP::MAP& map, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions);

/**
 * add a set of volumes to a drawer
 * @param the cell (VERTEX,EDGE,...)
 * @param dr the drawer to use
 * @param map the map
 * @param vd the darts
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerCell(unsigned int cell, Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions);

/**
 * add a set of vertices to a drawer
 * @param dr the drawer to use
 * @param map the map
 * @param vd the darts
 * @param positions attribute of positions
 */
template<typename PFP>
void drawerVertices(Utils::Drawer& dr, typename PFP::MAP& map, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions);

/**
 * add a set of edges to a drawer
 * @param dr the drawer to use
 * @param map the map
 * @param vd the darts
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerEdges(Utils::Drawer& dr, typename PFP::MAP& map, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k);

/**
 * add a set of faces to a drawer
 * @param dr the drawer to use
 * @param map the map
 * @param vd the darts
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerFaces(Utils::Drawer& dr, typename PFP::MAP& map, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k);

/**
 * add a set of volumes to a drawer
 * @param dr the drawer to use
 * @param map the map
 * @param vd the darts
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerVolumes(Utils::Drawer& dr, typename PFP::MAP& map, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k);

/**
 * add a vertex to a drawer
 * @param dr the drawer to use
 * @param map the map
 * @param d the dart
 * @param positions attribute of positions
 */
template<typename PFP>
void drawerVertex(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions);

/**
 * add an edge to a drawer
 * @param dr the drawer to use
 * @param map the map
 * @param d the dart
 * @param positions attribute of positions
 */
template<typename PFP>
void drawerEdge(Utils::Drawer& dr, typename PFP::MAP& map,  Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k);

/**
 * add a face to a drawer
 * @param dr the drawer to use
 * @param map the map
 * @param d the dart
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerFace(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k);

/**
 * add a volume to a drawer
 * @param dr the drawer to use
 * @param map the map
 * @param d the dart
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerVolume(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k);

/**
 * add an edge to a drawer, use between begin / end
 * @param dr the drawer to use
 * @param map the map
 * @param d the dart
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerAddEdge(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k);

/**
 * add an shrinked edge from center to a drawer, use between begin / end
 * @param dr the drawer to use
 * @param map the map
 * @param d the dart
 * @param positions attribute of positions
 * @param C center of cell to draw
 * @param k shrinking factor
 */
template<typename PFP>
void drawerAddEdgeShrink(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3& C, float k);

/**
 * add an face to a drawer, use between begin / end
 * @param dr the drawer to use
 * @param map the map
 * @param d the dart
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerAddFace(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k);

/**
 * add a volume to a drawer, use between begin / end
 * @param dr the drawer to use
 * @param map the map
 * @param d the dart
 * @param positions attribute of positions
 * @param k shrinking factor
 */
template<typename PFP>
void drawerAddVolume(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k);

} // namespace Render

} // namespace Algo

} // namespace CGoGN

#include "Algo/Render/GL2/drawerCells.hpp"

#endif
