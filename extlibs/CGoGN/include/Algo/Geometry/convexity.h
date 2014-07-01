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

#ifndef __ALGO_GEOMETRY_CONVEXITY_H__
#define __ALGO_GEOMETRY_CONVEXITY_H__

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

/**
 * Test if an edge bounded by 2 faces is convex or concave
 */
template <typename PFP>
bool isEdgeConvex(typename PFP::MAP& map, Edge e, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position) ;

} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Geometry/convexity.hpp"

#endif
