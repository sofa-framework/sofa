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

#ifndef __2MR_CC_FILTER__
#define __2MR_CC_FILTER__

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

/*
	﻿@article {springerlink:10.1007/s00371-006-0074-7,
	   author = {Wang, Huawei and Qin, Kaihuai and Tang, Kai},
	   affiliation = {Hong Kong University of Science & Technology Department of Mechanical Engineering Hong Kong China},
	   title = {Efficient wavelet construction with Catmull–Clark subdivision},
	   journal = {The Visual Computer},
	   publisher = {Springer Berlin / Heidelberg},
	   issn = {0178-2789},
	   keyword = {Computer Science},
	   pages = {874-884},
	   volume = {22},
	   issue = {9},
	   url = {http://dx.doi.org/10.1007/s00371-006-0074-7},
	   note = {10.1007/s00371-006-0074-7},
	   year = {2006}
	}
*/

/*********************************************************************************
 *                      	     Lazy Wavelet
 *********************************************************************************/
template <typename PFP>
class CCInitEdgeSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;
	bool first;

public:
	CCInitEdgeSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p), first(true)
	{}

	void operator() ()
	{
		if(first)
		{
			TraversorE<MAP> trav(m_map) ;
			for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
			{
				m_map.incCurrentLevel() ;
				m_position[m_map.phi1(d)] = VEC3(0.0);
				m_map.decCurrentLevel() ;
			}
			first = false;
		}
	}
} ;

template <typename PFP>
class CCInitFaceSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;
	bool first;

public:
	CCInitFaceSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p), first(true)
	{}

	void operator() ()
	{
		if(first)
		{
			TraversorF<MAP> trav(m_map) ;
			for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
			{
				m_map.incCurrentLevel() ;
				m_position[m_map.phi2(m_map.phi1(d))] = VEC3(0.0); // ou phi2(d)
				m_map.decCurrentLevel() ;
			}
			first = false;
		}
	}
} ;

template <typename PFP>
class CCEdgeSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	CCEdgeSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			VEC3 ei =  (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi1(d) ;
			m_position[midV] += ei ;
			m_map.decCurrentLevel() ;
		}
	}
} ;

template <typename PFP>
class CCFaceSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	CCFaceSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorF<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			float u = 1.0/2.0;

			VEC3 v(0.0);
			VEC3 e(0.0);
			unsigned int degree = 0;

			Dart dit = d;
			do
			{
				v += m_position[dit];

				m_map.incCurrentLevel() ;
				e += m_position[m_map.phi1(dit)];
				m_map.decCurrentLevel() ;

				++degree;

				dit = m_map.phi1(dit);
			}
			while(dit != d);

			v *= (1.0 - u) / degree;
			e *= u / degree;

			m_map.incCurrentLevel() ;
			m_position[m_map.phi2(m_map.phi1(d))] += v + e ;
			m_map.decCurrentLevel() ;
		}

	}
} ;

template <typename PFP>
class CCVertexSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	CCVertexSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			VEC3 np1(0) ;
			VEC3 np2(0) ;
			unsigned int degree1 = 0 ;
			unsigned int degree2 = 0 ;
			Dart it = d ;
			do
			{
				++degree1 ;
				Dart dd = m_map.phi1(it) ;
				np1 += m_position[dd] ;
				Dart end = m_map.phi_1(it) ;
				dd = m_map.phi1(dd) ;
				do
				{
					++degree2 ;
					np2 += m_position[dd] ;
					dd = m_map.phi1(dd) ;
				} while(dd != end) ;
				it = m_map.alpha1(it) ;
			} while(it != d) ;

			float beta = 3.0 / (2.0 * degree1) ;
			float gamma = 1.0 / (4.0 * degree2) ;
			np1 *= beta / degree1 ;
			np2 *= gamma / degree2 ;

			VEC3 vp = m_position[d] ;
			vp *= 1.0 - beta - gamma ;

			m_map.incCurrentLevel() ;
			m_position[d] = np1 + np2 + vp ;
			m_map.decCurrentLevel() ;
		}
	}
} ;

template <typename PFP>
class CCScalingSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	CCScalingSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			m_map.incCurrentLevel() ;

			VEC3 ei = m_position[m_map.phi1(d)];

			VEC3 f = m_position[m_map.phi2(m_map.phi1(d))];
			f += m_position[m_map.phi_1(m_map.phi2(d))];
			f *= 1.0 / 2.0;

			ei += f;
			ei *= 1.0 / 2.0;

			m_position[m_map.phi1(d)] = ei;
			m_map.decCurrentLevel() ;
		}
	}
} ;

template <typename PFP>
class CCScalingAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	CCScalingAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			m_map.incCurrentLevel() ;

			VEC3 ei = m_position[m_map.phi1(d)];

			VEC3 f = m_position[m_map.phi2(m_map.phi1(d))];
			f += m_position[m_map.phi_1(m_map.phi2(d))];
			f *= 1.0 / 2.0;

			ei *= 2.0;
			ei -= f;

			m_position[m_map.phi1(d)] = ei;
			m_map.decCurrentLevel() ;
		}
	}
} ;

template <typename PFP>
class CCVertexAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	CCVertexAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			m_map.incCurrentLevel() ;
			VEC3 np1(0) ;
			VEC3 np2(0) ;
			unsigned int degree1 = 0 ;
			unsigned int degree2 = 0 ;
			Dart it = d ;
			do
			{
				++degree1 ;
				Dart dd = m_map.phi1(it) ;
				np1 += m_position[dd] ;
				Dart end = m_map.phi_1(it) ;
				dd = m_map.phi1(dd) ;
				do
				{
					++degree2 ;
					np2 += m_position[dd] ;
					dd = m_map.phi1(dd) ;
				} while(dd != end) ;
				it = m_map.alpha1(it) ;
			} while(it != d) ;

			float beta = 3.0 / (2.0 * degree1) ;
			float gamma = 1.0 / (4.0 * degree2) ;
			np1 *= beta / degree1 ;
			np2 *= gamma / degree2 ;

			VEC3 vd = m_position[d] ;

			m_map.decCurrentLevel() ;

			m_position[d] = vd - np1 - np2;
			m_position[d] /= 1.0 - beta - gamma ;

		}
	}
} ;

template <typename PFP>
class CCFaceAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	CCFaceAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorF<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{

			float u = 1.0/2.0;

			VEC3 v(0.0);
			VEC3 e(0.0);
			unsigned int degree = 0;

			Dart dit = d;
			do
			{
				v += m_position[dit];

				m_map.incCurrentLevel() ;
				e += m_position[m_map.phi1(dit)];
				m_map.decCurrentLevel() ;

				++degree;

				dit = m_map.phi1(dit);
			}
			while(dit != d);

			v *= (1.0 - u) / degree;
			e *= u / degree;

			m_map.incCurrentLevel() ;
			m_position[m_map.phi2(m_map.phi1(d))] = m_position[m_map.phi2(m_map.phi1(d))] - v - e ;
			m_map.decCurrentLevel() ;
		}
	}
} ;

template <typename PFP>
class CCEdgeAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	CCEdgeAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			VEC3 ei =  (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi1(d) ;
			m_position[midV] -= ei ;
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
