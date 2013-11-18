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

namespace Volume
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
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	DooSabinVertexSynthesisFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
//		TraversorV<typename PFP::MAP> tv(m_map);
//		for(Dart d = tv.begin() ; d != tv.end() ; d = tv.next())
//		{
//			if(m_map.isBoundaryVertex(d))
//			{
//				Dart db = m_map.findBoundaryFaceOfVertex(d);
//
//				typename PFP::VEC3 p(0.0);
//
//				unsigned int N = m_map.faceDegree(db);
//				typename PFP::REAL K0 = float(N+5)/float(4*N);//(1.0 / 4.0) + (5.0 / 4.0) * double(N);
//				p += K0 * m_position[db];
//				unsigned int j = 1;
//				Dart tmp = m_map.phi1(db);
//				do
//				{
//					typename PFP::REAL Kj = (3.0 + 2.0 * cos(2.0 * double(j) * M_PI / double(N))) / (4.0 * N);
//					p += Kj * m_position[tmp];
//					tmp = m_map.phi1(tmp);
//					++j;
//				}while(tmp != db);
//
//				m_map.incCurrentLevel();
//
//				m_position[db] = p;
//
//				m_map.decCurrentLevel();
//
//			}
//			else
//			{
//				Traversor3VW<typename PFP::MAP> tvw(m_map,d);
//				for(Dart dit = tvw.begin() ; dit != tvw.end() ; dit = tvw.next())
//				{
//					typename PFP::VEC3 p(0.0);
//					unsigned int count = 0;
//
//					Dart ditface = dit;
//					do
//					{
//						typename PFP::VEC3 tmpP(0.0);
//						unsigned int N = m_map.faceDegree(ditface);
//						typename PFP::REAL K0 = float(N+5)/float(4*N);//(1.0 / 4.0) + (5.0 / 4.0) * double(N);
//						tmpP += K0 * m_position[ditface];
//						unsigned int j = 1;
//						Dart tmp = m_map.phi1(ditface);
//						do
//						{
//							typename PFP::REAL Kj = (3.0 + 2.0 * cos(2.0 * double(j) * M_PI / double(N))) / (4.0 * N);
//							tmpP += Kj * m_position[tmp];
//							tmp = m_map.phi1(tmp);
//							++j;
//						}while(tmp != ditface);
//
//						p += tmpP;
//						++count;
//						ditface = m_map.phi2(m_map.phi_1(ditface));
//					}
//					while(ditface != dit);
//
//					p /= count;
//
//					m_map.incCurrentLevel();
//
//					m_position[dit] = p;
//
//					m_map.decCurrentLevel();
//				}
//			}
//		}
	}
} ;


} // namespace Masks

} // namespace Dual

} // namespace MR

} // namespace Volume

} // namespace Algo

} // namespace CGoGN

#endif

