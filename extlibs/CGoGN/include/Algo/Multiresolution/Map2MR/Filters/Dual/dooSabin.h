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

#ifndef __MR_DOOSABIN_MASK__
#define __MR_DOOSABIN_MASK__

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
class DooSabinVertexSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	DooSabinVertexSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		for (Dart dV = m_map.begin(); dV != m_map.end();  m_map.next(dV))
		{
			VEC3 p(0.0);

			unsigned int N = m_map.faceDegree(dV);
			REAL K0 = float(N+5)/float(4*N);//(1.0 / 4.0) + (5.0 / 4.0) * double(N);
			p += K0 * m_position[dV];
			unsigned int j = 1;
			Dart tmp = m_map.phi1(dV);
			do
			{
				REAL Kj = (3.0 + 2.0 * cos(2.0 * double(j) * M_PI / double(N))) / (4.0 * N);
				p += Kj * m_position[tmp];
				tmp = m_map.phi1(tmp);
				++j;
			}while(tmp != dV);

			m_map.incCurrentLevel();

			m_position[dV] = p;

			m_map.decCurrentLevel();
		}
	}
} ;

} // namespace Masks

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif
