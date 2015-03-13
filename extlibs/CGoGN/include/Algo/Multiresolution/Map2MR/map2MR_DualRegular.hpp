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

#include "Algo/Topo/embedding.h"

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

namespace Regular
{

template <typename PFP>
Map2MR<PFP>::Map2MR(typename PFP::MAP& map) :
	MapManipulator("DualRegular2",&map),
	m_map(map),
	shareVertexEmbeddings(false)
{
}

template <typename PFP>
Map2MR<PFP>::~Map2MR()
{
	unsigned int level = m_map.getCurrentLevel();
	unsigned int maxL = m_map.getMaxLevel();

	for(unsigned int i = maxL ; i > level ; --i)
		m_map.removeLevelBack();

	for(unsigned int i = 0 ; i < level ; ++i)
		m_map.removeLevelFront();
}

template <typename PFP>
MapManipulator* Map2MR<PFP>::create(GenericMap *gm)
{
	typename PFP::MAP* map = dynamic_cast<typename PFP::MAP*>(gm);
	if (map != NULL)
		return (new Map2MR<PFP>(*map));
	else
		return NULL;
}


template <typename PFP>
void Map2MR<PFP>::addNewLevel(bool embedNewVertices)
{
	m_map.pushLevel() ;

	m_map.addLevelBack() ;
	m_map.duplicateDarts(m_map.getMaxLevel());
	m_map.setCurrentLevel(m_map.getMaxLevel()) ;


	m_map.decCurrentLevel();
	TraversorE<typename PFP::MAP> te(m_map);
	for (Dart d = te.begin(); d != te.end(); d = te.next())
	{
		m_map.incCurrentLevel();

		Dart d2 = m_map.phi2(d);

		m_map.unsewFaces(d,false);
		Dart nf = m_map.newFace(4,false);
		m_map.sewFaces(d,nf,false);
		m_map.sewFaces(d2,m_map.phi1(m_map.phi1(nf)),false);

		// take care of edge embedding
		if(m_map.template isOrbitEmbedded<EDGE>())
		{
//			m_map.template setOrbitEmbedding<EDGE>(nf, m_map.template getEmbedding<EDGE>(d));
//			m_map.template setOrbitEmbedding<EDGE>(m_map.phi1(m_map.phi1(nf)), m_map.template getEmbedding<EDGE>(d2));
			Algo::Topo::setOrbitEmbedding<EDGE>(m_map, nf, m_map.template getEmbedding<EDGE>(d));
			Algo::Topo::setOrbitEmbedding<EDGE>(m_map, m_map.phi1(m_map.phi1(nf)), m_map.template getEmbedding<EDGE>(d2));
		}

		m_map.decCurrentLevel();
	}


	TraversorV<typename PFP::MAP> tv(m_map);
	for(Dart d = tv.begin() ; d != tv.end() ; d = tv.next())
	{
		m_map.incCurrentLevel();

		m_map.PFP::MAP::TOPO_MAP::closeHole(m_map.phi1(m_map.phi2(d)),false);

		Dart temp = m_map.phi2(m_map.phi1(m_map.phi2(d)));
		Dart stop = temp;

		do
		{
			if(m_map.template isOrbitEmbedded<EDGE>())
			{
//				m_map.template setOrbitEmbedding<EDGE>(temp, m_map.template getEmbedding<EDGE>(	m_map.phi2(temp)));
				Algo::Topo::setOrbitEmbedding<EDGE>(m_map, temp, m_map.template getEmbedding<EDGE>(	m_map.phi2(temp)));
			}

			if(!shareVertexEmbeddings)
			{
				//if(m_map.template getEmbedding<VERTEX>(d) == EMBNULL)
				Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(m_map, temp);

				//m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(d2) ;
			}


			temp = m_map.phi1(temp);
		}
		while(temp != stop);


		m_map.decCurrentLevel();
	}

	m_map.incCurrentLevel() ;

	m_map.popLevel() ;
}

template <typename PFP>
void Map2MR<PFP>::analysis()
{
	assert(m_map.getCurrentLevel() > 0 || !"analysis : called on level 0") ;

	m_map.decCurrentLevel() ;

	for(unsigned int i = 0; i < analysisFilters.size(); ++i)
		(*analysisFilters[i])() ;
}

template <typename PFP>
void Map2MR<PFP>::synthesis()
{
	assert(m_map.getCurrentLevel() < m_map.getMaxLevel() || !"synthesis : called on max level") ;

	for(unsigned int i = 0; i < synthesisFilters.size(); ++i)
		(*synthesisFilters[i])() ;

	m_map.incCurrentLevel() ;
}


} // namespace Regular

} // namespace Dual

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
