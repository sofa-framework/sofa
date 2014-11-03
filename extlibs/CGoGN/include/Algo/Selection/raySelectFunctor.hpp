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

#ifndef RAYSELECTFUNCTOR_H_
#define RAYSELECTFUNCTOR_H_

#include "Geometry/distances.h"
#include "Geometry/intersection.h"

#include "Algo/Geometry/centroid.h"

namespace CGoGN
{

namespace Algo
{

namespace Selection
{

namespace Parallel
{

template <typename PFP>
class FuncVertexInter: public FunctorMapThreaded<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	const VertexAttribute<VEC3, MAP>& m_positions;
	const VEC3& m_A;
	const VEC3& m_AB;
	float m_AB2;
	float m_distMax;
	std::vector<std::pair<REAL, Dart> > m_vd;

public:
	FuncVertexInter(MAP& map, const VertexAttribute<VEC3, MAP>& position, const VEC3& A, const VEC3& AB, REAL AB2, REAL dm2):
		FunctorMapThreaded<typename PFP::MAP>(map),
		m_positions(position),
		m_A(A),
		m_AB(AB),
		m_AB2(AB2),
		m_distMax(dm2)
	{}

	void run(Dart d, unsigned int thread)
	{
		const VEC3& P = m_positions[d];
		float dist = Geom::squaredDistanceLine2Point(m_A, m_AB, m_AB2, P);
		if (dist < m_distMax)
		{
			REAL distA = (P-m_A).norm2();
			m_vd.push_back(std::pair<REAL, Dart>(distA,d));
		}

	}

	const std::vector<std::pair<REAL, Dart> >& getVertexDistances() { return m_vd; }
};

template <typename PFP>
class FuncEdgeInter: public FunctorMapThreaded<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	const VertexAttribute<VEC3, MAP>& m_positions;
	const VEC3& m_A;
	const VEC3& m_AB;
	float m_AB2;
	float m_distMax;
	std::vector<std::pair<REAL, Dart> > m_ed;

public:
	FuncEdgeInter(MAP& map, const VertexAttribute<VEC3, MAP>& position, const VEC3& A, const VEC3& AB, REAL AB2, REAL dm2):
		FunctorMapThreaded<typename PFP::MAP>(map),
		m_positions(position),
		m_A(A),
		m_AB(AB),
		m_AB2(AB2),
		m_distMax(dm2)
	{}

	void run(Dart d, unsigned int thread)
	{
		// get back position of segment PQ
		const VEC3& P = m_positions[d];
		Dart dd = this->m_map.phi1(d);
		const VEC3& Q = m_positions[dd];
		// the three distance to P, Q and (PQ) not used here
		float dist = Geom::squaredDistanceLine2Seg(m_A, m_AB, m_AB2, P, Q);
		if (dist < m_distMax)
		{
			VEC3 M = (P+Q)/2.0;
			REAL distA = (M-m_A).norm2();
			m_ed.push_back(std::pair<REAL, Dart>(distA,d));
		}
	}

	const std::vector<std::pair<REAL, Dart> >& getEdgeDistances() { return m_ed; }
};

template <typename PFP>
class FuncFaceInter: public FunctorMapThreaded<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	const VertexAttribute<VEC3, MAP>& m_positions;
	const VEC3& m_A;
	const VEC3& m_AB;
	std::vector<std::pair<REAL, Dart> > m_fd;

public:
	FuncFaceInter(MAP& map, const VertexAttribute<VEC3, MAP>& position, const VEC3& A, const VEC3& AB):
		FunctorMapThreaded<typename PFP::MAP>(map),
		m_positions(position),
		m_A(A),
		m_AB(AB)
	{}

	void run(Dart d, unsigned int thread)
	{
		const VEC3& Ta = m_positions[d];

		Dart dd  = this->m_map.phi1(d);
		Dart ddd = this->m_map.phi1(dd);
		bool notfound = true;
		do
		{
			// get back position of triangle Ta,Tb,Tc
			const VEC3& Tb = m_positions[dd];
			const VEC3& Tc = m_positions[ddd];
			VEC3 I;
			if (Geom::intersectionRayTriangleOpt<VEC3>(m_A, m_AB, Ta, Tb, Tc, I))
			{
				typename PFP::REAL dist = (I-m_A).norm2();
				m_fd.push_back(std::pair<REAL, Dart>(dist,d));
				notfound = false;
			}
			// next triangle if we are in polygon
			dd = ddd;
			ddd = this->m_map.phi1(dd);
		} while ((ddd != d) && notfound);
	}

	const std::vector<std::pair<REAL, Dart> >& getFaceDistances() { return m_fd; }
};

} // namespace Parallel

} // namespace Selection

} // namespace Algo

} // namespace CGoGN

#endif
