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

#ifndef _CGOGN_TRIANGULATION_H_
#define _CGOGN_TRIANGULATION_H_

#include <math.h>
#include <vector>
#include <list>
#include <set>
#include <utility>

#include "Algo/Geometry/normal.h"
#include "Topology/generic/autoAttributeHandler.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Modelisation
{

template <typename PFP>
class EarTriangulation
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;

protected:
	// forward declaration
	class VertexPoly;

	// multiset typedef for simple writing
	typedef std::multiset<VertexPoly, VertexPoly> VPMS;
	typedef typename VPMS::iterator VMPSITER;
	typedef NoTypeNameAttribute<VMPSITER> EarAttr ;

	class VertexPoly
	{
	public:
		Dart dart;
		float angle;
		float length;

		VertexPoly()
		{}

		VertexPoly(Dart d, float v, float l) : dart(d), angle(v), length(l)
		{}

		bool operator()(const VertexPoly& vp1, const VertexPoly& vp2)
		{
			if (fabs(vp1.angle - vp2.angle) < 0.2f)
				return vp1.length < vp2.length;
			return vp1.angle < vp2.angle;
		}
	};

protected:
	typename PFP::MAP& m_map;

	VertexAutoAttribute<EarAttr, MAP> m_dartEars;

	VertexAttribute<VEC3, MAP> m_position;

	VPMS m_ears;

	bool inTriangle(const VEC3& P, const VEC3& normal, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc);

	void recompute2Ears(Dart d, const VEC3& normalPoly, bool convex);

	float computeEarInit(Dart d, const VEC3& normalPoly, float& val);

public:

	EarTriangulation(MAP& map) : m_map(map), m_dartEars(map)
	{
		m_position = map.template getAttribute<VEC3, VERTEX, MAP>("position");
	}

//	void trianguleFace(Dart d, DartMarker& mark);
	void trianguleFace(Dart d);

	void triangule(unsigned int thread = 0);
};

} // namespace Modelisation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Modelisation/triangulation.hpp"

#endif
