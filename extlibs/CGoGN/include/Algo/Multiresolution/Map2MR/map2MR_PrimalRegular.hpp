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

#include "Algo/Import/importMRDAT.h"

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

namespace Regular
{

template <typename PFP>
Map2MR<PFP>::Map2MR(typename PFP::MAP& map) :
	m_map(map),
	shareVertexEmbeddings(true)
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
void Map2MR<PFP>::addNewLevel(bool triQuad, bool embedNewVertices)
{
	m_map.pushLevel() ;

	m_map.addLevelBack() ;
	m_map.duplicateDarts(m_map.getMaxLevel());
	m_map.setCurrentLevel(m_map.getMaxLevel()) ;

	// cut edges
	TraversorE<MAP> travE(m_map) ;
	for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
	{
//		if(!shareVertexEmbeddings && embedNewVertices)
//		{
//			if(m_map.template getEmbedding<VERTEX>(d) == EMBNULL)
//				m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(d) ;
//			if(m_map.template getEmbedding<VERTEX>(m_map.phi1(d)) == EMBNULL)
//				m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(d) ;
//		}

		m_map.cutEdge(d) ;
		travE.skip(d) ;
		travE.skip(m_map.phi1(d)) ;

		//std::cout << "is EMB NULL : " << ( m_map.template getEmbedding<VERTEX>(m_map.phi1(d)) == EMBNULL ? "true" : "false" ) << std::endl;

		//if(embedNewVertices)
		//	m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(m_map.phi1(d)) ;
	}

	// split faces
	TraversorF<MAP> travF(m_map) ;
	for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
	{
		Dart old = d ;
		if(m_map.getDartLevel(old) == m_map.getMaxLevel())
			old = m_map.phi1(old) ;

		m_map.decCurrentLevel() ;
		unsigned int degree = m_map.faceDegree(old) ;
		m_map.incCurrentLevel() ;

		if(triQuad && (degree == 3))					// if subdividing a triangle
		{
			Dart dd = m_map.phi1(old) ;
			Dart e = m_map.phi1(m_map.phi1(dd)) ;
			m_map.splitFace(dd, e) ;
			travF.skip(dd) ;

			dd = e ;
			e = m_map.phi1(m_map.phi1(dd)) ;
			m_map.splitFace(dd, e) ;
			travF.skip(dd) ;

			dd = e ;
			e = m_map.phi1(m_map.phi1(dd)) ;
			m_map.splitFace(dd, e) ;
			travF.skip(dd) ;

			travF.skip(e) ;
		}
		else							// if subdividing a polygonal face
		{
			Dart dd = m_map.phi1(old) ;
			Dart next = m_map.phi1(m_map.phi1(dd)) ;
			m_map.splitFace(dd, next) ;		// insert a first edge

			Dart ne = m_map.phi2(m_map.phi_1(dd)) ;
			m_map.cutEdge(ne) ;				// cut the new edge to insert the central vertex
			travF.skip(dd) ;

			//if(embedNewVertices)
			//		m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(m_map.phi1(ne)) ;

			dd = m_map.phi1(m_map.phi1(next)) ;
			while(dd != ne)				// turn around the face and insert new edges
			{							// linked to the central vertex
				Dart tmp = m_map.phi1(ne) ;
				m_map.splitFace(tmp, dd) ;
				travF.skip(tmp) ;
				dd = m_map.phi1(m_map.phi1(dd)) ;
			}
			travF.skip(ne) ;
		}
	}

	m_map.popLevel() ;
}

template <typename PFP>
void Map2MR<PFP>::addNewLevelSqrt3()
{
	m_map.pushLevel() ;

	m_map.addLevelBack() ;
	m_map.duplicateDarts(m_map.getMaxLevel());
	m_map.setCurrentLevel(m_map.getMaxLevel()) ;

	//split faces
	TraversorF<MAP> t(m_map) ;
	for (Dart dit = t.begin(); dit != t.end(); dit = t.next())
	{
		//if it is an even level (triadic refinement) and a boundary face
		if((m_map.getCurrentLevel()%2 == 0) && m_map.isFaceIncidentToBoundary(dit))
		{
			//find the boundary edge
			Dart df = m_map.findBoundaryEdgeOfFace(dit);
			//trisection of the boundary edge
			m_map.cutEdge(df) ;
			m_map.splitFace(m_map.phi2(df), m_map.phi1(m_map.phi1(m_map.phi2(df)))) ;

			df = m_map.phi1(df);
			m_map.cutEdge(df) ;
			m_map.splitFace(m_map.phi2(df), m_map.phi1(m_map.phi1(m_map.phi2(df)))) ;
		}
		else
		{
			Dart d1 = m_map.phi1(dit);
			m_map.splitFace(dit, d1) ;
			m_map.cutEdge(m_map.phi_1(dit)) ;
			Dart x = m_map.phi2(m_map.phi_1(dit)) ;
			Dart dd = m_map.phi1(m_map.phi1(m_map.phi1((x))));
			while(dd != x)
			{
				Dart next = m_map.phi1(dd) ;
				m_map.splitFace(dd, m_map.phi1(x)) ;
				dd = next ;
			}

			Dart cd = m_map.phi2(x);

			Dart fit = cd ;
			do
			{
				t.skip(fit);
				fit = m_map.phi2(m_map.phi_1(fit));
			} while(fit != cd);
		}
	}

	//swap edges
	TraversorE<typename PFP::MAP> te(m_map) ;
	for (Dart dit = te.begin(); dit != te.end(); dit = te.next())
	{
		if(!m_map.isBoundaryEdge(dit) && m_map.getDartLevel(dit) < m_map.getCurrentLevel())
			m_map.flipEdge(dit);
	}

	m_map.popLevel() ;
}

template <typename PFP>
void Map2MR<PFP>::addNewLevelSqrt2()
{
	m_map.pushLevel() ;

	m_map.addLevelBack() ;
	m_map.duplicateDarts(m_map.getMaxLevel());
	m_map.setCurrentLevel(m_map.getMaxLevel()) ;

	//split faces
	TraversorF<MAP> t(m_map) ;
	for (Dart dit = t.begin(); dit != t.end(); dit = t.next())
	{
		Dart d1 = m_map.phi1(dit);
		m_map.splitFace(dit, d1) ;
		m_map.cutEdge(m_map.phi_1(dit)) ;
		Dart x = m_map.phi2(m_map.phi_1(dit)) ;
		Dart dd = m_map.phi1(m_map.phi1(m_map.phi1((x))));
		while(dd != x)
		{
			Dart next = m_map.phi1(dd) ;
			m_map.splitFace(dd, m_map.phi1(x)) ;
			dd = next ;
		}

		Dart cd = m_map.phi2(x);

		Dart fit = cd ;
		do
		{
			t.skip(fit);
			fit = m_map.phi2(m_map.phi_1(fit));
		} while(fit != cd);
	}

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

template <typename PFP>
void Map2MR<PFP>::addLevelFront()
{
	DartMarker<MAP> md(m_map);

	m_map.addLevelFront() ;
	m_map.duplicateDarts(0);
	m_map.setCurrentLevel(0);

	std::vector<Dart> visitedVertices;
	visitedVertices.reserve(1024);

	//look for an irregular vertex

	TraversorV<MAP> tv(m_map);
	bool found = false;
	for(Dart d = tv.begin() ; !found && d != tv.end() ; d = tv.next())
	{
		if(m_map.vertexDegree(d) != 6)
		{
			found = true;
			visitedVertices.push_back(d);
		}
	}

	std::cout << "d = " << visitedVertices[0] << std::endl;

	for(unsigned int i = 0 ; i < visitedVertices.size() ; ++i)
	{
		Dart d = visitedVertices[i];

			Dart fit1 = m_map.phi2(m_map.phi1(d));
			//m_map.mergeFaces(fit1) ;

//		Traversor2VE<typename PFP::MAP> tve(m_map, d);
//		for(Dart eit = tve.begin() ; eit != tve.end() ; eit = tve.next())
//		{
//			//coarse all faces around the vertex
//			if(!md.isMarked(eit))
//			{
//				unsigned int degree = m_map.faceDegree(eit);

//				if(degree == 3)
//				{
//					Dart fit1 = m_map.phi2(m_map.phi1(eit));
//					//Dart fit2 = m_map.phi1(fit1);
//					//Dart fit3 = m_map.phi1(fit2);

//					m_map.mergeFaces(fit1) ;
//					//m_map.mergeFaces(fit2) ;
//					//m_map.mergeFaces(fit3) ;
//				}
//				else
//				{

//				}

//				//visitedVertices.push_back(m_map.phi1(m_map.phi1(eit)));
//				//visitedVertices.push_back(m_map.phi_1(m_map.phi_1(eit)));
//			}
//		}

//		for(Dart eit = tve.begin() ; eit != tve.end() ; eit = tve.next())
//		{
//			if(!md.isMarked(eit))
//			{
//				//coarse all edges around the vertex
//				m_map.uncutEdge(eit) ;
//				md.markOrbit<EDGE>(eit);
//			}
//		}
	}
}

template <typename PFP>
void Map2MR<PFP>::import(Algo::Surface::Import::QuadTree& qt)
{
	std::cout << "  Create finer resolution levels.." << std::flush ;

	for(unsigned int i = 0; i < qt.depth; ++i)
		addNewLevel(true, false) ;

	std::cout << "..done" << std::endl ;
	std::cout << "  Embed finer resolution levels.." << std::flush ;

	m_map.setCurrentLevel(0) ;
	qt.embed<PFP>(m_map) ;
	m_map.setCurrentLevel(m_map.getMaxLevel()) ;

	std::cout << "..done" << std::endl ;
}

} // namespace Regular

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
