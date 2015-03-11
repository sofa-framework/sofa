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

#ifndef __3MR_BERTRAM_FILTER__
#define __3MR_BERTRAM_FILTER__

#include <cmath>
#include "Algo/Geometry/centroid.h"
#include "Algo/Modelisation/tetrahedralization.h"
#include "Algo/Multiresolution/filter.h"
#include "Topology/generic/traversor/traversor2_closed.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace MR
{

namespace Primal
{

namespace Filters
{

/*******************************************************************************
 *							Without features preserving
 *******************************************************************************/

//
// Analysis
//

//w-lift(a)
template <typename PFP>
class Ber02OddAnalysisFilter : public Algo::MR::Filter
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;
	typename PFP::VEC3::DATA_TYPE m_a;

public:
	Ber02OddAnalysisFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p, typename PFP::VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorE<typename PFP::MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			typename PFP::VEC3 ve = (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);
			ve *= 2.0 * m_a;

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi1(d) ;
			m_position[midV] -= ve;
			m_map.decCurrentLevel() ;
		}

		TraversorF<typename PFP::MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			typename PFP::VEC3 vf(0.0);
			typename PFP::VEC3 ef(0.0);

			unsigned int count = 0;
			Traversor3FE<typename PFP::MAP> travFE(m_map, d);
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

		TraversorW<typename PFP::MAP> travW(m_map) ;
		for (Dart d = travW.begin(); d != travW.end(); d = travW.next())
		{
			typename PFP::VEC3 vc = Algo::Surface::Geometry::volumeCentroid<PFP>(m_map, d, m_position);
			vc *= 8 * m_a * m_a * m_a;

			unsigned int count = 0;
			typename PFP::VEC3 ec(0.0);
			Traversor3WE<typename PFP::MAP> travWE(m_map, d);
			for (Dart dit = travWE.begin(); dit != travWE.end(); dit = travWE.next())
			{
				m_map.incCurrentLevel();
				ec += m_position[m_map.phi1(dit)];
				m_map.decCurrentLevel();
				++count;
			}
			ec /= count;
			ec *= 12 * m_a * m_a;

			count = 0;
			typename PFP::VEC3 fc(0.0);
			Traversor3WF<typename PFP::MAP> travWF(m_map, d);
			for (Dart dit = travWF.begin(); dit != travWF.end(); dit = travWF.next())
			{
				m_map.incCurrentLevel();
				fc += m_position[m_map.phi1(m_map.phi1(dit))];
				m_map.decCurrentLevel();
				++count;
			}
			fc /= count;
			fc *= 6 * m_a;

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(d)));
			m_position[midV] -= vc + ec + fc;
			m_map.decCurrentLevel() ;
		}
	}
} ;

// s-lift(a)
template <typename PFP>
class Ber02EvenAnalysisFilter : public Algo::MR::Filter
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;
	typename PFP::VEC3::DATA_TYPE m_a;

public:
	Ber02EvenAnalysisFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p, typename PFP::VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorF<typename PFP::MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			if(!m_map.isBoundaryFace(d))
			{
				unsigned int count = 0;

				typename PFP::VEC3 cf(0.0);
				Traversor3FW<typename PFP::MAP> travFW(m_map, d);
				for(Dart dit = travFW.begin() ; dit != travFW.end() ; dit = travFW.next())
				{
					m_map.incCurrentLevel();
					Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(dit)));
					cf += m_position[midV];
					m_map.decCurrentLevel();
					++count;
				}
				cf /= count;
				cf *= 2 * m_a;

				m_map.incCurrentLevel() ;
				Dart midF = m_map.phi1(m_map.phi1(d));
				m_position[midF] -= cf;
				m_map.decCurrentLevel() ;
			}
		}

		TraversorE<typename PFP::MAP> travE(m_map);
		for(Dart d = travE.begin() ; d != travE.end() ; d = travE.next())
		{
			if(m_map.isBoundaryEdge(d))
			{
				Dart db = m_map.findBoundaryFaceOfEdge(d);
				typename PFP::VEC3 fe(0.0);

//				unsigned int count = 2;
//				m_map.incCurrentLevel() ;
//				Dart midV = m_map.phi1(m_map.phi1(db));
//				fe += m_position[midV];
//				midV = m_map.phi_1(m_map.phi2(db));
//				fe += m_position[midV];
//				m_map.decCurrentLevel() ;

				//TODO Replace do--while with a Traversor2 on Boundary
				unsigned int count = 0;
				Traversor2EF<typename PFP::MAP> travEF(m_map, db);
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
				Dart midE = m_map.phi1(db);
				m_position[midE] -= fe;
				m_map.decCurrentLevel() ;
			}
			else
			{
				unsigned int count = 0;

				typename PFP::VEC3 ce(0.0);
				Traversor3EW<typename PFP::MAP> travEW(m_map, d);
				for(Dart dit = travEW.begin() ; dit != travEW.end() ; dit = travEW.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(dit)));
					ce += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				ce /= count;
				ce *= 4 * m_a * m_a;

				typename PFP::VEC3 fe(0.0);
				count = 0;
				Traversor3EF<typename PFP::MAP> travEF(m_map, d);
				for(Dart dit = travEF.begin() ; dit != travEF.end() ; dit = travEF.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi1(m_map.phi1(dit));
					fe += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				fe /= count;
				fe *= 4 * m_a;

				m_map.incCurrentLevel() ;
				Dart midE = m_map.phi1(d);
				m_position[midE] -= ce + fe;
				m_map.decCurrentLevel() ;
			}
		}

		TraversorV<typename PFP::MAP> travV(m_map);
		for(Dart d = travV.begin() ; d != travV.end() ; d = travV.next())
		{
			if(m_map.isBoundaryVertex(d))
			{
				Dart db = m_map.findBoundaryFaceOfVertex(d);

				unsigned int count = 0;
				typename PFP::VEC3 ev(0.0);
				typename PFP::VEC3 fv(0.0);

//				Dart dit = db;
//				do
//				{
//					m_map.incCurrentLevel() ;
//
//					Dart midEdgeV = m_map.phi1(dit);
//					ev += m_position[midEdgeV];
//					fv += m_position[m_map.phi1(midEdgeV)];
//
//					m_map.decCurrentLevel() ;
//					++count;
//
//					dit = m_map.phi2(m_map.phi_1(dit));
//
//				}while(dit != db);

				Traversor2VF<typename PFP::MAP> travVF(m_map,db);
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
			else
			{
				unsigned int count = 0;

				typename PFP::VEC3 cv(0.0);
				Traversor3VW<typename PFP::MAP> travVW(m_map,d);
				for(Dart dit = travVW.begin(); dit != travVW.end() ; dit = travVW.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(dit)));
					cv += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				cv /= count;
				cv *= 8 * m_a * m_a * m_a;

				typename PFP::VEC3 fv(0.0);
				count = 0;
				Traversor3VF<typename PFP::MAP> travVF(m_map,d);
				for(Dart dit = travVF.begin(); dit != travVF.end() ; dit = travVF.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi1(m_map.phi1(dit));
					fv += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				fv /= count;
				fv *= 12 * m_a * m_a;

				typename PFP::VEC3 ev(0.0);
				count = 0;
				Traversor3VE<typename PFP::MAP> travVE(m_map,d);
				for(Dart dit = travVE.begin(); dit != travVE.end() ; dit = travVE.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi1(dit);
					ev += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				ev /= count;
				ev *= 6 * m_a;

				m_position[d] -= cv + fv + ev;
			}
		}
	}
} ;

// s-scale(a)
template <typename PFP>
class Ber02ScaleAnalysisFilter : public Algo::MR::Filter
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;
	typename PFP::VEC3::DATA_TYPE m_a;

public:
	Ber02ScaleAnalysisFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p, typename PFP::VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorF<typename PFP::MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
				m_map.incCurrentLevel() ;
				Dart midF = m_map.phi1(m_map.phi1(d));
				if(!m_map.isBoundaryVertex(midF))
					m_position[midF] /= m_a ;
				m_map.decCurrentLevel() ;

		}

		TraversorE<typename PFP::MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{

			m_map.incCurrentLevel() ;
			Dart midE = m_map.phi1(d);
			if(m_map.isBoundaryVertex(midE))
				m_position[midE] /= m_a;
			else
				m_position[midE] /= m_a * m_a;
			m_map.decCurrentLevel() ;

		}

		TraversorV<typename PFP::MAP> travV(m_map) ;
		for (Dart d = travV.begin(); d != travV.end(); d = travV.next())
		{
			if(m_map.isBoundaryVertex(d))
				m_position[d] /= m_a * m_a;
			else
				m_position[d] /= m_a *m_a * m_a;
		}
	}
} ;


//
// Synthesis
//

//w-lift(a)
template <typename PFP>
class Ber02OddSynthesisFilter : public Algo::MR::Filter
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;
	typename PFP::VEC3::DATA_TYPE m_a;

public:
	Ber02OddSynthesisFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p, typename PFP::VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorW<typename PFP::MAP> travW(m_map) ;
		for (Dart d = travW.begin(); d != travW.end(); d = travW.next())
		{
			typename PFP::VEC3 vc = Algo::Surface::Geometry::volumeCentroid<PFP>(m_map, d, m_position);
			vc *= 8 * m_a * m_a * m_a;

			unsigned int count = 0;
			typename PFP::VEC3 ec(0.0);
			Traversor3WE<typename PFP::MAP> travWE(m_map, d);
			for (Dart dit = travWE.begin(); dit != travWE.end(); dit = travWE.next())
			{
				m_map.incCurrentLevel();
				ec += m_position[m_map.phi1(dit)];
				m_map.decCurrentLevel();
				++count;
			}
			ec /= count;
			ec *= 12 * m_a * m_a;

			count = 0;
			typename PFP::VEC3 fc(0.0);
			Traversor3WF<typename PFP::MAP> travWF(m_map, d);
			for (Dart dit = travWF.begin(); dit != travWF.end(); dit = travWF.next())
			{
				m_map.incCurrentLevel();
				fc += m_position[m_map.phi1(m_map.phi1(dit))];
				m_map.decCurrentLevel();
				++count;
			}
			fc /= count;
			fc *= 6 * m_a;

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(d)));
			m_position[midV] += vc + ec + fc;
			m_map.decCurrentLevel() ;
		}

		TraversorF<typename PFP::MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			typename PFP::VEC3 vf(0.0);
			typename PFP::VEC3 ef(0.0);

			unsigned int count = 0;
			Traversor3FE<typename PFP::MAP> travFE(m_map, d);
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

		TraversorE<typename PFP::MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			typename PFP::VEC3 ve = (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);
			ve *= 2.0 * m_a;

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi1(d) ;
			m_position[midV] += ve;
			m_map.decCurrentLevel() ;
		}
	}
} ;

// s-lift(a)
template <typename PFP>
class Ber02EvenSynthesisFilter : public Algo::MR::Filter
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;
	typename PFP::VEC3::DATA_TYPE m_a;

public:
	Ber02EvenSynthesisFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p, typename PFP::VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorV<typename PFP::MAP> travV(m_map);
		for(Dart d = travV.begin() ; d != travV.end() ; d = travV.next())
		{
			if(m_map.isBoundaryVertex(d))
			{
				Dart db = m_map.findBoundaryFaceOfVertex(d);

				unsigned int count = 0;
				typename PFP::VEC3 ev(0.0);
				typename PFP::VEC3 fv(0.0);

//				Dart dit = db;
//				do
//				{
//					m_map.incCurrentLevel() ;
//
//					Dart midEdgeV = m_map.phi1(dit);
//					ev += m_position[midEdgeV];
//					fv += m_position[m_map.phi1(midEdgeV)];
//
//					m_map.decCurrentLevel() ;
//					++count;
//
//					dit = m_map.phi2(m_map.phi_1(dit));
//
//				}while(dit != db);

				//TODO Replace do--while with a Traversor2 on Boundary
				Traversor2VF<typename PFP::MAP> travVF(m_map,db);
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
			else
			{
				unsigned int count = 0;

				typename PFP::VEC3 cv(0.0);
				Traversor3VW<typename PFP::MAP> travVW(m_map,d);
				for(Dart dit = travVW.begin(); dit != travVW.end() ; dit = travVW.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(dit)));
					cv += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				cv /= count;
				cv *= 8 * m_a * m_a * m_a;

				typename PFP::VEC3 fv(0.0);
				count = 0;
				Traversor3VF<typename PFP::MAP> travVF(m_map,d);
				for(Dart dit = travVF.begin(); dit != travVF.end() ; dit = travVF.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi1(m_map.phi1(dit));
					fv += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				fv /= count;
				fv *= 12 * m_a * m_a;

				typename PFP::VEC3 ev(0.0);
				count = 0;
				Traversor3VE<typename PFP::MAP> travVE(m_map,d);
				for(Dart dit = travVE.begin(); dit != travVE.end() ; dit = travVE.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi1(dit);
					ev += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				ev /= count;
				ev *= 6 * m_a;

				m_position[d] += cv + fv + ev;
			}
		}

		TraversorE<typename PFP::MAP> travE(m_map);
		for(Dart d = travE.begin() ; d != travE.end() ; d = travE.next())
		{
			if(m_map.isBoundaryEdge(d))
			{
				Dart db = m_map.findBoundaryFaceOfEdge(d);
				typename PFP::VEC3 fe(0.0);

//				unsigned int count = 2;
//				m_map.incCurrentLevel() ;
//				Dart midV = m_map.phi1(m_map.phi1(db));
//				fe += m_position[midV];
//				midV = m_map.phi_1(m_map.phi2(db));
//				fe += m_position[midV];
//				m_map.decCurrentLevel() ;

				//TODO Replace do--while with a Traversor2 on Boundary
				unsigned int count = 0;
				Traversor2EF<typename PFP::MAP> travEF(m_map, db);
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
				Dart midE = m_map.phi1(db);
				m_position[midE] += fe;
				m_map.decCurrentLevel() ;
			}
			else
			{
				unsigned int count = 0;

				typename PFP::VEC3 ce(0.0);
				Traversor3EW<typename PFP::MAP> travEW(m_map, d);
				for(Dart dit = travEW.begin() ; dit != travEW.end() ; dit = travEW.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(dit)));
					ce += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				ce /= count;
				ce *= 4 * m_a * m_a;

				typename PFP::VEC3 fe(0.0);
				count = 0;
				Traversor3EF<typename PFP::MAP> travEF(m_map, d);
				for(Dart dit = travEF.begin() ; dit != travEF.end() ; dit = travEF.next())
				{
					m_map.incCurrentLevel() ;
					Dart midV = m_map.phi1(m_map.phi1(dit));
					fe += m_position[midV];
					m_map.decCurrentLevel() ;
					++count;
				}
				fe /= count;
				fe *= 4 * m_a;

				m_map.incCurrentLevel() ;
				Dart midE = m_map.phi1(d);
				m_position[midE] += ce + fe;
				m_map.decCurrentLevel() ;
			}
		}

		TraversorF<typename PFP::MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			if(!m_map.isBoundaryFace(d))
			{
				unsigned int count = 0;

				typename PFP::VEC3 cf(0.0);
				Traversor3FW<typename PFP::MAP> travFW(m_map, d);
				for(Dart dit = travFW.begin() ; dit != travFW.end() ; dit = travFW.next())
				{
					m_map.incCurrentLevel();
					Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(dit)));
					cf += m_position[midV];
					m_map.decCurrentLevel();
					++count;
				}
				cf /= count;
				cf *= 2 * m_a;

				m_map.incCurrentLevel() ;
				Dart midF = m_map.phi1(m_map.phi1(d));
				m_position[midF] += cf;
				m_map.decCurrentLevel() ;
			}
		}
	}
} ;

// s-scale(a)
template <typename PFP>
class Ber02ScaleSynthesisFilter : public Algo::MR::Filter
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;
	typename PFP::VEC3::DATA_TYPE m_a;

public:
	Ber02ScaleSynthesisFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p, typename PFP::VEC3::DATA_TYPE a) : m_map(m), m_position(p), m_a(a)
	{}

	void operator() ()
	{
		TraversorV<typename PFP::MAP> travV(m_map) ;
		for (Dart d = travV.begin(); d != travV.end(); d = travV.next())
		{
			if(m_map.isBoundaryVertex(d))
				m_position[d] *= m_a * m_a;
			else
				m_position[d] *= m_a *m_a * m_a;
		}

		TraversorE<typename PFP::MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{

			m_map.incCurrentLevel() ;
			Dart midE = m_map.phi1(d);
			if(m_map.isBoundaryVertex(midE))
				m_position[midE] *= m_a;
			else
				m_position[midE] *= m_a * m_a;
			m_map.decCurrentLevel() ;

		}

		TraversorF<typename PFP::MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
				m_map.incCurrentLevel() ;
				Dart midF = m_map.phi1(m_map.phi1(d));
				if(!m_map.isBoundaryVertex(midF))
					m_position[midF] *= m_a ;
				m_map.decCurrentLevel() ;

		}
	}
} ;



} // namespace Filters

} // namespace Primal

} // namespace MR

} // namespace Volume

} // namespace Algo

} // namespace CGoGN


#endif /* __3MR_FILTERS_PRIMAL__ */
