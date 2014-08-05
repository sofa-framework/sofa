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

#ifndef __ALGO_GEOMETRY_ORIENTATION_H__
#define __ALGO_GEOMETRY_ORIENTATION_H__

#include "Geometry/basic.h"
#include "Geometry/orientation.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP>
bool isTetrahedronWellOriented(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, bool CCW)
{
	typedef typename PFP::VEC3 VEC3 ;

	VEC3 points[4] ;
	points[0] = positions[d] ;
	points[1] = positions[map.phi1(d)] ;
	points[2] = positions[map.phi_1(d)] ;
	points[3] = positions[map.phi_1(map.phi2(d))] ;

	return Geom::isTetrahedronWellOriented<VEC3>(points, CCW) ;
}

} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif
