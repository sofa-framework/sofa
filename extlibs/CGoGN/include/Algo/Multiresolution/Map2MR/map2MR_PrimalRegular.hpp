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
void Map2MR<PFP>::addNewLevel(bool triQuad)
{
	m_map.pushLevel() ;

	m_map.addLevelBack() ;
	m_map.duplicateDarts(m_map.getMaxLevel());
	m_map.setCurrentLevel(m_map.getMaxLevel()) ;

	// cut edges
	TraversorE<typename PFP::MAP> travE(m_map) ;
	for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
	{
//		if(!shareVertexEmbeddings)
//		{
//			if(m_map.template getEmbedding<VERTEX>(d) == EMBNULL)
//				m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(d) ;
//			if(m_map.template getEmbedding<VERTEX>(m_map.phi1(d)) == EMBNULL)
//				m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(d) ;
//		}

		m_map.cutEdge(d) ;
		travE.skip(d) ;
		travE.skip(m_map.phi1(d)) ;

	}

	// split faces
	TraversorF<typename PFP::MAP> travF(m_map) ;
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
	TraversorF<typename PFP::MAP> t(m_map) ;
	for (Dart dit = t.begin(); dit != t.end(); dit = t.next())
	{
		//if it is an even level (triadic refinement) and a boundary face
		if((m_map.getCurrentLevel()%2 == 0) && m_map.isBoundaryFace(dit))
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
	TraversorF<typename PFP::MAP> t(m_map) ;
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


} // namespace Regular

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
