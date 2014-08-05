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

#ifndef __2MR_SQRT3_FILTER__
#define __2MR_SQRT3_FILTER__

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
	@article{10.1109/TVCG.2007.1031,
		author = {Huawei Wang and Kaihuai Qin and Hanqiu Sun},
		title = {$\sqrt{3}$-Subdivision-Based Biorthogonal Wavelets},
		journal ={IEEE Transactions on Visualization and Computer Graphics},
		volume = {13},
		issn = {1077-2626},
		year = {2007},
		pages = {914-925},
		doi = {http://doi.ieeecomputersociety.org/10.1109/TVCG.2007.1031},
		publisher = {IEEE Computer Society},
		address = {Los Alamitos, CA, USA},
	}
*/

inline double omega12(unsigned int n)
{
	switch(n)
	{
		case 3: return -0.138438 ;
		case 4: return -0.193032 ;
		case 5: return -0.216933 ;
		case 6: return -0.229537 ;
		case 7: return -0.237236 ;
		case 8: return -0.242453 ;
		case 9: return -0.246257 ;
		case 10: return -0.249180 ;
		case 15: return -0.257616 ;
		case 20: return -0.261843 ;
		default: return 0.0;
			//find formulation ?
	}
}

inline double omega0(unsigned int n)
{
	switch(n)
	{
		case 3: return -0.684601 ;
		case 4: return -0.403537 ;
		case 5: return -0.288813 ;
		case 6: return -0.229537 ;
		case 7: return -0.193385 ;
		case 8: return -0.168740 ;
		case 9: return -0.150618 ;
		case 10: return -0.136570 ;
		case 15: return -0.095201 ;
		case 20: return -0.073924 ;
		default: return 0.0;
			//find formulation ?
	}
}

/*********************************************************************************
 *                      	     Lazy Wavelet
 *********************************************************************************/

template <typename PFP>
class Sqrt3FaceSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	Sqrt3FaceSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorF<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			Dart d1 = m_map.phi1(d) ;
			Dart d2 = m_map.phi1(d1) ;

			VEC3 p0 = m_position[d] ;
			VEC3 p1 = m_position[d1] ;
			VEC3 p2 = m_position[d2] ;

			p0 *= 1.0 / 3.0 ;
			p1 *= 1.0 / 3.0 ;
			p2 *= 1.0 / 3.0 ;

			if(m_map.isFaceIncidentToBoundary(d))
			{
				Dart df = m_map.findBoundaryEdgeOfFace(d);
				m_map.incCurrentLevel() ;
				m_position[m_map.phi_1(m_map.phi2(df))] += p0 + p1 + p2 ;
			}
			else
			{
				m_map.incCurrentLevel() ;
				m_position[m_map.phi2(d)] += p0 + p1 + p2 ;
			}

			m_map.decCurrentLevel() ;
		}
	}
} ;

template <typename PFP>
class Sqrt3VertexSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	Sqrt3VertexSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			if(m_map.isBoundaryVertex(d))
			{
				Dart df = m_map.findBoundaryEdgeOfVertex(d);

				if((m_map.getCurrentLevel()%2 == 0))
				{
					VEC3 np(0) ;
					VEC3 nl(0) ;
					VEC3 nr(0) ;

					VEC3 pi = m_position[df];
					VEC3 pi_1 = m_position[m_map.phi_1(df)];
					VEC3 pi1 = m_position[m_map.phi1(df)];

					np += pi_1 * 4 + pi * 19 + pi1 * 4;
					np /= 27;

					nl +=  pi_1 * 10 + pi * 16 + pi1;
					nl /= 27;

					nr += pi_1 + pi * 16 + pi1 * 10;
					nr /= 27;

					m_map.incCurrentLevel() ;

					m_position[df] = np;
					m_position[m_map.phi_1(df)] = nl;
					m_position[m_map.phi1(df)] = nr;

					m_map.decCurrentLevel() ;
				}
			}
			else
			{
				VEC3 nf(0) ;
				unsigned int degree = 0 ;

				m_map.incCurrentLevel() ;
				Dart df = m_map.phi2(m_map.phi1(d));

				Traversor2VVaE<MAP> trav(m_map, df) ;
				for(Dart it = trav.begin(); it != trav.end(); it = trav.next())
				{
					++degree ;
					nf += m_position[it] ;

				}
				m_map.decCurrentLevel() ;

				float alpha = 1.0/9.0 * ( 4.0 - 2.0 * cos(2.0 * M_PI / degree));
				float teta = 1 - (3 * alpha) / 2;
				float sigma = (3 * alpha) / (2 * degree);

				nf *= sigma;

				VEC3 vp = m_position[d] ;
				vp *= teta ;

				m_map.incCurrentLevel() ;

				m_position[df] = vp + nf;

				m_map.decCurrentLevel() ;
			}
		}
	}
} ;

template <typename PFP>
class Sqrt3VertexAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	Sqrt3VertexAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			VEC3 nf(0) ;
			unsigned int degree = 0 ;

			m_map.incCurrentLevel() ;
			Dart df = m_map.phi2(m_map.phi1(d));

			Traversor2VVaE<MAP> trav(m_map, df) ;
			for(Dart it = trav.begin(); it != trav.end(); it = trav.next())
			{
				++degree ;
				nf += m_position[it] ;

			}
			m_map.decCurrentLevel() ;

			float alpha = 1.0/9.0 * ( 4.0 - 2.0 * cos(2.0 * M_PI / degree));
			float teta = 1 - (3 * alpha) / 2;
			float sigma = (3 * alpha) / (2 * degree);

			nf *= sigma;

			VEC3 vp = m_position[d] ;
			vp -= nf ;

			m_map.incCurrentLevel() ;

			m_position[df] = vp * (1.0 / teta) ;

			m_map.decCurrentLevel() ;

		}
	}
} ;

template <typename PFP>
class Sqrt3FaceAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	Sqrt3FaceAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorF<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			Dart d1 = m_map.phi1(d) ;
			Dart d2 = m_map.phi1(d1) ;

			VEC3 p0 = m_position[d] ;
			VEC3 p1 = m_position[d1] ;
			VEC3 p2 = m_position[d2] ;

			p0 *= 1.0 / 3.0 ;
			p1 *= 1.0 / 3.0 ;
			p2 *= 1.0 / 3.0 ;

			m_map.incCurrentLevel() ;

			m_position[m_map.phi2(d)] -= p0 + p1 + p2 ;

			m_map.decCurrentLevel() ;

		}
	}
} ;

/*********************************************************************************
 *                          Three-Point Orthogonalization
 *********************************************************************************/


/*********************************************************************************
 *                          Six-Point Orthogonalization
 *********************************************************************************/

} // namespace Filters

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif

