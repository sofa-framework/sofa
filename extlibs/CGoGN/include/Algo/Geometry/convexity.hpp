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

#include "Geometry/basic.h"
#include "Algo/Geometry/normal.h"

#include <limits>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP>
bool isEdgeConvexe(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& position)
{
	typedef typename PFP::VEC3 VEC3 ;

	const VEC3 n = faceNormal<PFP>(map, d, position);
	const VEC3 e = vectorOutOfDart<PFP>(map, map.phi1(map.phi2(d)), position) ;

	if((e * n) > 0)
		return true;
	else
		return false;
}

} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
