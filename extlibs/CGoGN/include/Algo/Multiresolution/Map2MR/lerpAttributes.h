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

#ifndef __MR_LERPATTRIB_FILTER__
#define __MR_LERPATTRIB_FILTER__

#include <cmath>
#include "Algo/Multiresolution/filter.h"
#include "Topology/generic/functor.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MR
{

/*********************************************************************************
 *                           SYNTHESIS FILTERS
 *********************************************************************************/

template <typename PFP>
class vertexFilter : public Algo::MR::Filter
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	vertexFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<typename PFP::MAP> tE(m_map);
		for(Dart d = tE.begin() ; d != tE.end() ; d = tE.next())
		{
			Dart dit = d;
			Dart dres = d;
			bool found = false;
			do
			{
				if((m_map.getDartLevel(dit) == (m_map.getCurrentLevel()-1)) && (m_map.getDartLevel(m_map.phi2(dit)) == m_map.getCurrentLevel()))
				{
					dres = dit;
					found = true;
				}

				dit = m_map.phi2(m_map.phi_1(dit));

			}while(!found && dit!=d);

//			m_map.template setOrbitEmbedding<VERTEX>(dres, m_map.template getEmbedding<VERTEX>(dres));
			Algo::Topo::setOrbitEmbedding<VERTEX>(m_map,template getEmbedding<VERTEX>(dres));
		}

//		SelectorEdgeLevel<typename PFP::MAP> ml(m_map, m_map.getCurrentLevel());
//
//		TraversorV<typename PFP::MAP> tV(m_map, ml, true);
//		for(Dart d = tV.begin() ; d != tV.end() ; d = tV.next())
//		{
//			m_map.template embedOrbit<VERTEX>(d, m_map.template getEmbedding<VERTEX>(d));
//		}
	}
};

template <typename PFP>
class LerpVertexSynthesisFilter : public Algo::MR::Filter
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_attrib ;

public:
	LerpVertexSynthesisFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_attrib(p)
	{}

	void operator() ()
	{
		if(m_attrib.isValid())
		{
			TraversorV<typename PFP::MAP> tE(m_map);
			for(Dart d = tE.begin() ; d != tE.end() ; d = tE.next())
			{
				Dart dit = d;
				Dart dres = d;
				bool found = false;
				//1ere boucle est-ce qu'il y a un brin de niveau i

				do
				{
					if((m_map.getDartLevel(dit) < m_map.getCurrentLevel()) && (m_map.getDartLevel(m_map.phi2(dit)) == m_map.getCurrentLevel()))
					{
						dres = dit;
						found = true;
					}

					dit = m_map.phi2(m_map.phi_1(dit));

				}while(!found && dit!=d);


				if(found)
				{
					m_map.decCurrentLevel();
					typename PFP::VEC3 c = m_attrib[dres];
					m_map.incCurrentLevel();

					m_attrib[dres] = c;
				}
			}
		}
	}
} ;


template <typename PFP>
class LerpVertexAnalysisFilter : public Algo::MR::Filter
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_attrib ;

public:
	LerpVertexAnalysisFilter(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_attrib(p)
	{}

	void operator() ()
	{
		if(m_attrib.isValid())
		{
			TraversorV<typename PFP::MAP> trav(m_map) ;
			for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
			{
				Dart dit = d;
				Dart dres = d;
				bool found = false;

				do
				{

					m_map.incCurrentLevel();
					Dart d2_1 = m_map.phi2(dit);
					m_map.decCurrentLevel();

					if((m_map.getDartLevel(dit) == m_map.getCurrentLevel()) && m_map.phi2(dit) != d2_1)
					{
						dres = dit;
						found = true;
					}

					dit = m_map.phi2(m_map.phi_1(dit));

				}while(!found && dit!=d);

				if(found)
				{
					m_map.incCurrentLevel();

					typename PFP::VEC3 c1 = m_attrib[m_map.phi2(m_map.phi_1(dres))];
					typename PFP::VEC3 c2 = m_attrib[m_map.phi1(m_map.phi2(m_map.phi_1(m_map.phi2(dres))))];
					m_map.decCurrentLevel();

					m_attrib[dres] = c1 + c2;
					m_attrib[dres] /= 2;
				}
			}
		}
	}
} ;

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif

//			Dart dit = d;
//			Dart ds = d;
//			bool found = false;
//			//1ere boucle est-ce qu'il y a un brin de niveau i
//			do
//			{
//				if((m_map.getDartLevel(dit) < m_map.getCurrentLevel()) && (m_map.getDartLevel(m_map.phi2(dit)) == m_map.getCurrentLevel()))
//				{
//					ds = dit;
//					found = true;
//				}
//
//				dit = m_map.phi2(m_map.phi_1(dit));
//
//			}while(!found && dit!=d);
//
//
//			if(found)
//			{
//				bool found2 = false;
//				dit = d;
//				Dart dres = dit;
//				//2e boucle chercher un brin de niveau i-1
//				do
//				{
//					if(m_map.getDartLevel(dit) < (m_map.getCurrentLevel()))
//					{
//						if(m_map.getDartLevel(dit) > m_map.getDartLevel(dres))
//							dres = dit;
//
//						//found2 = true;
//					}
//
//					dit = m_map.phi2(m_map.phi_1(dit));
//
//				}while(dit!=d);
//
//				//if(!found2)
//				//	std::cout << "non trouve......." << std::endl;
//
//				//if(dres != d)
//					m_attrib[ds] = m_attrib[dres];
//			}

