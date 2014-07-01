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

#ifndef __2MR_SQRT2_FILTER__
#define __2MR_SQRT2_FILTER__

#include <cmath>
#include "Algo/Multiresolution/filter.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MR
{

namespace Primal
{

namespace Filters
{

/*********************************************************************************
 *                           SYNTHESIS FILTERS
 *********************************************************************************/

template <typename PFP>
class Sqrt2FaceSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	Sqrt2FaceSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorF<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			VEC3 p = Geometry::faceCentroid<PFP>(m_map, d, m_position);

			m_map.incCurrentLevel() ;

			Dart midF = m_map.phi2(d);
			if(m_map.isBoundaryEdge(d))
			{
				midF = m_map.phi1(m_map.phi2(m_map.phi_1(d)));
			}

			m_position[midF] = p ;

			m_map.decCurrentLevel() ;
		}
	}
} ;

} // namespace Filters

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif

