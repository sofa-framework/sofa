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

#include "Algo/Modelisation/polyhedron.h"

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

namespace Regular
{

template <typename PFP>
Map3MR<PFP>::Map3MR(typename PFP::MAP& map) :
	m_map(map),
	shareVertexEmbeddings(false)
{

}

template <typename PFP>
Map3MR<PFP>::~Map3MR()
{
	unsigned int level = m_map.getCurrentLevel();
	unsigned int maxL = m_map.getMaxLevel();

	for(unsigned int i = maxL ; i > level ; --i)
		m_map.removeLevelBack();

	for(unsigned int i = 0 ; i < level ; ++i)
		m_map.removeLevelFront();
}

template <typename PFP>
void Map3MR<PFP>::addNewLevel(bool embedNewVertices)
{
	m_map.pushLevel() ;

	m_map.addLevelBack() ;
	m_map.duplicateDarts(m_map.getMaxLevel());
	m_map.setCurrentLevel(m_map.getMaxLevel()) ;

	m_map.decCurrentLevel();


	TraversorF<typename PFP::MAP> tf(m_map);
	for (Dart d = tf.begin(); d != tf.end(); d = tf.next())
	{
		if(!m_map.isBoundaryFace(d))
		{
			unsigned int nbSides = m_map.faceDegree(d);
			Dart d3 = m_map.phi3(d);

			m_map.incCurrentLevel();

			m_map.unsewVolumes(d,false);

			Dart nf = Algo::Surface::Modelisation::createPrism<PFP>(m_map, nbSides, false);

			m_map.sewVolumes(d,nf,false);
			m_map.sewVolumes(d3,m_map.phi2(m_map.phi1(m_map.phi1(m_map.phi2(nf)))),false);

			m_map.decCurrentLevel();
		}
	}


	TraversorE<typename PFP::MAP> te(m_map);
	for(Dart d = te.begin() ; d != te.end() ; d = te.next())
	{
		if(!m_map.isBoundaryEdge(d))
		{
			m_map.incCurrentLevel();

			m_map.PFP::MAP::TOPO_MAP::closeHole(m_map.phi3(d),false);

			m_map.decCurrentLevel();
		}
	}


//	TraversorV<typename PFP::MAP> tv(m_map);
//	for(Dart d = tv.begin() ; d != tv.end() ; d = tv.next())
//	{
//		m_map.incCurrentLevel();
//
//		m_map.PFP::MAP::TOPO_MAP::closeHole(m_map.phi2(m_map.phi_1(m_map.phi3(m_map.phi2(d)))),false);
//
//		Dart temp = m_map.phi2(m_map.phi1(m_map.phi2(d)));
//		Dart stop = temp;
//
//		do
//		{
//			if(!shareVertexEmbeddings)
//			{
//				//if(m_map.template getEmbedding<VERTEX>(d) == EMBNULL)
//				m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(temp) ;
//				//m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(d2) ;
//			}
//
//
//			temp = m_map.phi1(temp);
//		}
//		while(temp != stop);
//
//
//		m_map.decCurrentLevel();
//	}

	m_map.incCurrentLevel() ;

	m_map.popLevel() ;
}

template <typename PFP>
void Map3MR<PFP>::analysis()
{
	assert(m_map.getCurrentLevel() > 0 || !"analysis : called on level 0") ;

	m_map.decCurrentLevel() ;

	for(unsigned int i = 0; i < analysisFilters.size(); ++i)
		(*analysisFilters[i])() ;
}

template <typename PFP>
void Map3MR<PFP>::synthesis()
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
