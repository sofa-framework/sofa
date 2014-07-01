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


#ifndef __SIMPLEX__
#define __SIMPLEX__

#include "Topology/generic/cells.h"

namespace CGoGN
{

namespace Algo
{

namespace Topo
{


template <typename MAP, unsigned int ORBIT>
bool isSimplex(const MAP& map, Dart d)
{
	if (ORBIT==VOLUME)
	{
		Dart d1 = map.phi2(d);
		Dart e = map.phi1(d);
		Dart d2 = map.phi2(e);
		e = map.phi1(e);
		Dart d3 = map.phi2(d);

		if (map.phi1(e) != d)  // check that face of d is a triangle
			return false;

		if (map.phi_1(d1) != map.template phi<12>(d2))
			return false;
		if (map.phi_1(d2) != map.template phi<12>(d3))
			return false;
		if (map.phi_1(d3) != map.template phi<12>(d1))
			return false;

		if (! map.isCycleTriangle(d1))
			return false;
		if (! map.isCycleTriangle(d2))
			return false;
		if (! map.isCycleTriangle(d3))
			return false;

		return true;
	}
	if (ORBIT==FACE)
	{
		return map.isCycleTriangle(d);
	}

	return true;
}


} // namespace Topo

} // namespace Algo

} // namespace CGoGN


#endif
