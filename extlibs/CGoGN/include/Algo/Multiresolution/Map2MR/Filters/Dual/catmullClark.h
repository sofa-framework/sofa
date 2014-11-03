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

#ifndef __MR_CC_MASK__
#define __MR_CC_MASK__

#include <cmath>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MR
{

namespace Dual
{

namespace Filters
{

template <typename PFP>
class CCVertexSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	CCVertexSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<MAP> trav(m_map) ;
		for (Dart dV = trav.begin(); dV != trav.end(); dV = trav.next())
		{
			m_map.incCurrentLevel() ;

			Dart start = m_map.phi2(m_map.phi1(m_map.phi2(dV)));
			Dart dit = start;
			do
			{
				Dart d = m_map.phi1(m_map.phi2(m_map.phi1(m_map.phi2(dit))));
				unsigned int degree = m_map.faceDegree(d);
				VEC3 p(0.0);

				REAL a0 = 1.0 / 2.0 + 1.0 / (4.0 * REAL(degree));
				REAL a1 = 1.0 / 8.0 + 1.0 / (4.0 * REAL(degree));
				REAL ak_1 = a1;

				if(degree == 3)
				{
					p += a0 * m_position[d] + a1 * m_position[m_map.phi1(d)] + ak_1 * m_position[m_map.phi_1(d)];

					m_map.incCurrentLevel();
					m_position[d] = p;
					m_map.decCurrentLevel();

				}
				else if(degree == 4)
				{
					REAL a2 = 1.0 / (4.0 * REAL(degree));

					p += a0 * m_position[d] + a1 * m_position[m_map.phi1(d)] + a2 * m_position[m_map.phi1(m_map.phi1(d))] + ak_1 * m_position[m_map.phi_1(d)];

					m_map.incCurrentLevel();
					m_position[d] = p;

				}
				else
				{
					p += a0 * m_position[d] + a1 * m_position[m_map.phi1(d)];

					Dart end = m_map.phi_1(m_map.phi_1(d));
					Dart dit = m_map.phi1(m_map.phi1(d));
					do
					{
						REAL ai = 1.0 / (4.0 * REAL(degree));
						p += ai * m_position[dit];

						dit = m_map.phi1(dit);
					}
					while(dit != end);

					p += ak_1 * m_position[m_map.phi_1(d)];

					m_map.incCurrentLevel();
					m_position[d] = p;
				}


				dit = m_map.phi1(dit);
			}
			while(dit != start);
		}

		m_map.decCurrentLevel();
	}
} ;

} // namespace Masks

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif
