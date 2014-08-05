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

#ifndef __2MR_BERTRAM_FILTER__
#define __2MR_BERTRAM_FILTER__

#include <cmath>
#include "Algo/Geometry/centroid.h"
#include "Algo/Modelisation/tetrahedralization.h"
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
 *                           ANALYSIS FILTERS
 *********************************************************************************/

//w-lift(a)
template <typename PFP>
class Ber02OddAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;
	typename VEC3::DATA_TYPE m_a;

public:
	Ber02OddAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p, typename VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorE<MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			VEC3 ve = (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);
			ve *= 2.0 * m_a;

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi1(d) ;
			m_position[midV] -= ve;
			m_map.decCurrentLevel() ;
		}

		TraversorF<MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			VEC3 vf(0.0);
			VEC3 ef(0.0);

			unsigned int count = 0;
			Traversor2FE<MAP> travFE(m_map, d);
			for (Dart dit = travFE.begin(); dit != travFE.end(); dit = travFE.next())
			{
				vf += m_position[dit];
				m_map.incCurrentLevel();
				ef += m_position[m_map.phi1(dit)];
				m_map.decCurrentLevel();
				++count;
			}
			ef /= count;
			ef *= 4.0 * m_a;

			vf /= count;
			vf *= 4.0 * m_a * m_a;

			m_map.incCurrentLevel() ;
			Dart midF = m_map.phi1(m_map.phi1(d));
			m_position[midF] -= vf + ef ;
			m_map.decCurrentLevel() ;
		}
	}
};

// s-lift(a)
template <typename PFP>
class Ber02EvenAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;
	typename VEC3::DATA_TYPE m_a;

public:
	Ber02EvenAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p, typename VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorE<MAP> travE(m_map);
		for(Dart d = travE.begin() ; d != travE.end() ; d = travE.next())
		{
			if(!m_map.isBoundaryEdge(d))
			{
				unsigned int count = 0;

				VEC3 fe(0);
				Traversor2EF<MAP> travEF(m_map, d);
				for(Dart dit = travEF.begin() ; dit != travEF.end() ; dit = travEF.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi1(m_map.phi1(dit));
					fe += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}

				fe /= count;
				fe *= 2 * m_a;

				m_map.incCurrentLevel() ;
				Dart midF = m_map.phi1(d);
				m_position[midF] -= fe;
				m_map.decCurrentLevel() ;
			}
		}

		TraversorV<MAP> travV(m_map);
		for(Dart d = travV.begin() ; d != travV.end() ; d = travV.next())
		{
			VEC3 ev(0.0);
			VEC3 fv(0.0);
			if(m_map.isBoundaryVertex(d))
			{
				Dart db = m_map.findBoundaryEdgeOfVertex(d);
				m_map.incCurrentLevel() ;
				ev += (m_position[m_map.phi1(db)] + m_position[m_map.phi_1(db)]) * typename PFP::REAL(0.5);
				//ev = (m_position[m_map.phi1(db)] + m_position[m_map.phi_1(db)]);
				m_map.decCurrentLevel() ;
				ev *= 2 * m_a;
				//ev *= m_a;

				m_position[d] -= ev;
			}
			else
			{
				unsigned int count = 0;

				Traversor2VF<MAP> travVF(m_map,d);
				for(Dart dit = travVF.begin(); dit != travVF.end() ; dit = travVF.next())
				{
					m_map.incCurrentLevel() ;

					Dart midEdgeV = m_map.phi1(dit);
					ev += m_position[midEdgeV];
					fv += m_position[m_map.phi1(midEdgeV)];

					m_map.decCurrentLevel() ;
					++count;
				}
				fv /= count;
				fv *= 4 * m_a * m_a;

				ev /= count;
				ev *= 4 * m_a;

				m_position[d] -= fv + ev;
			}
		}
	}
};

// s-scale(a)
template <typename PFP>
class Ber02ScaleAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;
	typename VEC3::DATA_TYPE m_a;

public:
	Ber02ScaleAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p, typename VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorE<MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			m_map.incCurrentLevel() ;
			Dart midE = m_map.phi1(d);
			if(!m_map.isBoundaryVertex(midE))
				m_position[midE] /= m_a ;
			m_map.decCurrentLevel() ;
		}

		TraversorV<MAP> travV(m_map) ;
		for (Dart d = travV.begin(); d != travV.end(); d = travV.next())
		{
			if(m_map.isBoundaryVertex(d))
				m_position[d] /= m_a;
			else
				m_position[d] /= m_a * m_a;
		}
	}
};



/*********************************************************************************
 *                           SYNTHESIS FILTERS
 *********************************************************************************/

//w-lift(a)
template <typename PFP>
class Ber02OddSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;
	typename VEC3::DATA_TYPE m_a;


public:
	Ber02OddSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p, typename VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorF<MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			VEC3 vf(0.0);
			VEC3 ef(0.0);

			unsigned int count = 0;
			Traversor2FE<MAP> travFE(m_map, d);
			for (Dart dit = travFE.begin(); dit != travFE.end(); dit = travFE.next())
			{
				vf += m_position[dit];
				m_map.incCurrentLevel();
				ef += m_position[m_map.phi1(dit)];
				m_map.decCurrentLevel();
				++count;
			}
			ef /= count;
			ef *= 4.0 * m_a;

			vf /= count;
			vf *= 4.0 * m_a * m_a;

			m_map.incCurrentLevel() ;
			Dart midF = m_map.phi1(m_map.phi1(d));
			m_position[midF] += vf + ef ;
			m_map.decCurrentLevel() ;

		}

		TraversorE<MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			VEC3 ve = (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);
			ve *= 2.0 * m_a;

			m_map.incCurrentLevel() ;
			Dart midE = m_map.phi1(d) ;
			m_position[midE] += ve;
			m_map.decCurrentLevel() ;
		}
	}
} ;

// s-lift(a)
template <typename PFP>
class Ber02EvenSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;
	typename VEC3::DATA_TYPE m_a;

public:
	Ber02EvenSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p, typename VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorV<MAP> travV(m_map);
		for(Dart d = travV.begin() ; d != travV.end() ; d = travV.next())
		{
			VEC3 ev(0.0);
			VEC3 fv(0.0);
			if(m_map.isBoundaryVertex(d))
			{
				Dart db = m_map.findBoundaryEdgeOfVertex(d);
				m_map.incCurrentLevel() ;
				ev += (m_position[m_map.phi1(db)] + m_position[m_map.phi_1(db)]) * typename PFP::REAL(0.5);
				m_map.decCurrentLevel() ;
				ev *= 2 * m_a;

				m_position[db] += ev;
			}
			else
			{
				unsigned int count = 0;
				Traversor2VF<MAP> travVF(m_map,d);
				for(Dart dit = travVF.begin(); dit != travVF.end() ; dit = travVF.next())
				{
					m_map.incCurrentLevel() ;

					Dart midEdgeV = m_map.phi1(dit);
					ev += m_position[midEdgeV];
					fv += m_position[m_map.phi1(midEdgeV)];

					m_map.decCurrentLevel() ;
					++count;
				}
				fv /= count;
				fv *= 4 * m_a * m_a;

				ev /= count;
				ev *= 4 * m_a;
				m_position[d] += fv + ev;
			}
		}

		TraversorE<MAP> travE(m_map);
		for(Dart d = travE.begin() ; d != travE.end() ; d = travE.next())
		{
			if(!m_map.isBoundaryEdge(d))
			{
				unsigned int count = 0;

				VEC3 fe(0.0);
				Traversor2EF<MAP> travEF(m_map, d);
				for(Dart dit = travEF.begin() ; dit != travEF.end() ; dit = travEF.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi1(m_map.phi1(dit));
					fe += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}

				fe /= count;
				fe *= 2 * m_a;

				m_map.incCurrentLevel() ;
				Dart midF = m_map.phi1(d);
				m_position[midF] += fe;
				m_map.decCurrentLevel() ;
			}
		}
	}
} ;

// s-scale(a)
template <typename PFP>
class Ber02ScaleSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;
	typename VEC3::DATA_TYPE m_a;

public:
	Ber02ScaleSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p, typename VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorV<MAP> travV(m_map) ;
		for (Dart d = travV.begin(); d != travV.end(); d = travV.next())
		{
			if(m_map.isBoundaryVertex(d))
				m_position[d] *= m_a;
			else
				m_position[d] *= m_a * m_a;
		}

		TraversorE<MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			m_map.incCurrentLevel() ;
			Dart midE = m_map.phi1(d);
			if(!m_map.isBoundaryVertex(midE))
				m_position[midE] *= m_a ;
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

#endif /* __2MR_FILTERS_PRIMAL__ */
