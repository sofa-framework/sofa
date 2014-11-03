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

#include "Algo/Multiresolution/Map2MR/map2MR_PrimalAdapt.h"

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

namespace Adaptive
{

template <typename PFP>
Map2MR<PFP>::Map2MR(typename PFP::MAP& map) :
	m_map(map),
	shareVertexEmbeddings(true),
	vertexVertexFunctor(NULL),
	edgeVertexFunctor(NULL),
	faceVertexFunctor(NULL)
{

}

/***************************************************
 *               CELLS INFORMATION                 *
 ***************************************************/
template <typename PFP>
unsigned int Map2MR<PFP>::edgeLevel(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"edgeLevel : called with a dart inserted after current level") ;

	unsigned int ld = m_map.getDartLevel(d) ;
	unsigned int ldd = m_map.getDartLevel(m_map.phi2(d)) ;	// the level of an edge is the maximum of the
	return ld > ldd ? ld : ldd ;				// insertion levels of its two darts
}

template <typename PFP>
unsigned int Map2MR<PFP>::faceLevel(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"faceLevel : called with a dart inserted after current level") ;

	if(m_map.getCurrentLevel() == 0)
		return 0 ;

	Dart it = d ;
	unsigned int min1 = m_map.getDartLevel(it) ;		// the level of a face is the second minimum of the
	it = m_map.phi1(it) ;
	unsigned int min2 = m_map.getDartLevel(it) ;		// insertion levels of its darts

	if(min2 < min1)
	{
		unsigned int tmp = min1 ;
		min1 = min2 ;
		min2 = tmp ;
	}

	it = m_map.phi1(it) ;
	while(it != d)
	{
		unsigned int dl = m_map.getDartLevel(it) ;
		if(dl < min2)
		{
			if(dl < min1)
			{
				min2 = min1 ;
				min1 = dl ;
			}
			else
				min2 = dl ;
		}
		it = m_map.phi1(it) ;
	}

	return min2 ;
}

template <typename PFP>
Dart Map2MR<PFP>::faceOrigin(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"faceOrigin : called with a dart inserted after current level") ;

	m_map.pushLevel() ;
	Dart p = d ;
	unsigned int pLevel = m_map.getDartLevel(p) ;
	while(pLevel > 0)
	{
		p = m_map.faceOldestDart(p) ;
		pLevel = m_map.getDartLevel(p) ;
		m_map.setCurrentLevel(pLevel) ;
	}
	m_map.popLevel() ;
	return p ;
}

template <typename PFP>
Dart Map2MR<PFP>::faceOldestDart(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"faceOldestDart : called with a dart inserted after current level") ;

	Dart it = d ;
	Dart oldest = it ;
	unsigned int l_old = m_map.getDartLevel(oldest) ;
	do
	{
		unsigned int l = m_map.getDartLevel(it) ;
		if(l == 0)
			return it ;
		if(l < l_old)
		{
			oldest = it ;
			l_old = l ;
		}
		it = m_map.phi1(it) ;
	} while(it != d) ;
	return oldest ;
}

template <typename PFP>
bool Map2MR<PFP>::edgeIsSubdivided(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"edgeIsSubdivided : called with a dart inserted after current level") ;

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		return false ;

	Dart d2 = m_map.phi2(d) ;
	m_map.incCurrentLevel() ;
	Dart d2_l = m_map.phi2(d) ;
	m_map.decCurrentLevel() ;
	if(d2 != d2_l)
		return true ;
	else
		return false ;
}

template <typename PFP>
bool Map2MR<PFP>::edgeCanBeCoarsened(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"edgeCanBeCoarsened : called with a dart inserted after current level") ;

	if(edgeIsSubdivided(d))
	{
		bool subdOnce = true ;
		bool degree2 = false ;

		Dart d2 = m_map.phi2(d) ;
		m_map.incCurrentLevel() ;
		if(m_map.vertexDegree(m_map.phi1(d)) == 2)
		{
			degree2 = true ;
			if(edgeIsSubdivided(d) || edgeIsSubdivided(d2))
				subdOnce = false ;
		}
		m_map.decCurrentLevel() ;

		return degree2 && subdOnce ;
	}

	return false ;
}

template <typename PFP>
bool Map2MR<PFP>::faceIsSubdivided(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"faceIsSubdivided : called with a dart inserted after current level") ;

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		return false ;

	unsigned int fLevel = faceLevel(d) ;
	if(fLevel < m_map.getCurrentLevel())	// a face whose level in the current level map is lower than
		return false ;				// the current level can not be subdivided to higher levels

	bool subd = false ;
	m_map.incCurrentLevel() ;
	if(m_map.getDartLevel(m_map.phi1(m_map.phi1(d))) == m_map.getCurrentLevel())
		subd = true ;
	m_map.decCurrentLevel() ;
	return subd ;
}

template <typename PFP>
bool Map2MR<PFP>::faceIsSubdividedOnce(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"faceIsSubdividedOnce : called with a dart inserted after current level") ;

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		return false ;

	unsigned int fLevel = faceLevel(d) ;
	if(fLevel < m_map.getCurrentLevel())	// a face whose level in the current level map is lower than
		return false ;				// the current level can not be subdivided to higher levels

	unsigned int degree = 0 ;
	bool subd = false ;
	bool subdOnce = true ;

	m_map.incCurrentLevel() ;
	if(m_map.getDartLevel(m_map.phi1(m_map.phi1(d))) == m_map.getCurrentLevel())
		subd = true ;
	m_map.decCurrentLevel() ;

	if(subd)
	{
		m_map.incCurrentLevel() ;

		if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		{
			m_map.decCurrentLevel() ;
			return true ;
		}

		Dart fit = d ;
		do
		{
			m_map.incCurrentLevel() ;
			if(m_map.getDartLevel(m_map.phi1(m_map.phi1(fit))) == m_map.getCurrentLevel())
				subdOnce = false ;
				m_map.decCurrentLevel() ;
			++degree ;
			fit = m_map.phi1(fit) ;
		} while(subdOnce && fit != d) ;

		if(degree == 3 && subdOnce)
		{
			Dart cf = m_map.phi2(m_map.phi1(d)) ;
			m_map.incCurrentLevel() ;
			if(m_map.getDartLevel(m_map.phi1(m_map.phi1(cf))) == m_map.getCurrentLevel())
				subdOnce = false ;
			m_map.decCurrentLevel() ;
		}

		m_map.decCurrentLevel() ;

		return subdOnce ;
	}

	return false ;
}

/***************************************************
 *               SUBDIVISION                       *
 ***************************************************/

template <typename PFP>
Dart Map2MR<PFP>::cutEdge(Dart d)
{
	Dart dd = m_map.phi2(d) ;
	Dart d11 = m_map.phi1(d);
	Dart dd11 = m_map.phi1(dd);

	m_map.duplicateDart(d);
	m_map.duplicateDart(dd);
	m_map.duplicateDart(d11);
	m_map.duplicateDart(dd11);

	Dart nd = m_map.cutEdge(d) ;

	Dart d1 = m_map.phi1(d);
	Dart dd1 = m_map.phi1(dd);

	m_map.propagateDartRelation(d, m_map.m_phi1) ;
	m_map.propagateDartRelation(d, m_map.m_phi2) ;

	m_map.propagateDartRelation(dd, m_map.m_phi1) ;
	m_map.propagateDartRelation(dd, m_map.m_phi2) ;

	m_map.propagateDartRelation(d1, m_map.m_phi1) ;
	m_map.propagateDartRelation(d1, m_map.m_phi_1) ;
	m_map.propagateDartRelation(d1, m_map.m_phi2) ;

	m_map.propagateDartRelation(dd1, m_map.m_phi1) ;
	m_map.propagateDartRelation(dd1, m_map.m_phi_1) ;
	m_map.propagateDartRelation(dd1, m_map.m_phi2) ;

	m_map.propagateDartRelation(d11, m_map.m_phi_1) ;
	m_map.propagateDartRelation(dd11, m_map.m_phi_1) ;

	return nd ;
}

template <typename PFP>
void Map2MR<PFP>::splitFace(Dart d, Dart e)
{
	Dart dprev = m_map.phi_1(d) ;
	Dart eprev = m_map.phi_1(e) ;

	m_map.duplicateDart(d);
	m_map.duplicateDart(e);
	m_map.duplicateDart(dprev);
	m_map.duplicateDart(eprev);

	m_map.splitFace(d, e) ;
	Dart dd = m_map.phi1(dprev) ;
	Dart ee = m_map.phi1(eprev) ;

	m_map.propagateDartRelation(dd, d, dprev, m_map.m_phi_1, m_map.m_phi1) ;
	m_map.propagateDartRelation(ee, e, eprev, m_map.m_phi_1, m_map.m_phi1) ;

	m_map.propagateDartRelation(dd, m_map.m_phi1) ;
//	//m_map.propagateDartRelation(dd, m_map.m_phi_1) ;
	m_map.propagateDartRelation(dd, m_map.m_phi2) ;

	m_map.propagateDartRelation(ee, m_map.m_phi1) ;
//	//m_map.propagateDartRelation(ee, m_map.m_phi_1) ;
	m_map.propagateDartRelation(ee, m_map.m_phi2) ;

	m_map.propagateDartRelation(dprev, d, m_map.m_phi1) ;
	m_map.propagateDartRelation(eprev, e, m_map.m_phi1) ;
//	//m_map.propagateDartRelation(dprev, m_map.m_phi1) ;
//	//m_map.propagateDartRelation(eprev, m_map.m_phi1) ;

	m_map.propagateDartRelation(d, m_map.m_phi_1) ;
	m_map.propagateDartRelation(e, m_map.m_phi_1) ;

	m_map.template propagateDartEmbedding<VERTEX>(dd) ;
	m_map.template propagateDartEmbedding<VERTEX>(ee) ;

}

template <typename PFP>
void Map2MR<PFP>::flipBackEdge(Dart d)
{
	Dart dprev = m_map.phi_1(d) ;
	Dart dnext = m_map.phi_1(d) ;

	Dart d2 = m_map.phi2(d);
	Dart d2prev = m_map.phi_1(d2) ;
	Dart d2next = m_map.phi_1(d2) ;

	m_map.duplicateDart(d);
	m_map.duplicateDart(dprev);
	m_map.duplicateDart(dnext);

	m_map.duplicateDart(d2);
	m_map.duplicateDart(d2prev);
	m_map.duplicateDart(d2next);

	m_map.flipBackEdge(d) ;
}

template <typename PFP>
void Map2MR<PFP>::subdivideEdge(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideEdge : called with a dart inserted after current level") ;
	assert(!edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;

	assert(m_map.getCurrentLevel() == edgeLevel(d) || !"Trying to subdivide an edge on a bad current level") ;

	m_map.incCurrentLevel() ;

	//Dart nd = cutEdge(d) ;
	Dart nd = m_map.cutEdge(d);

	(*edgeVertexFunctor)(nd) ;

	m_map.decCurrentLevel() ;
}

template <typename PFP>
void Map2MR<PFP>::coarsenEdge(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"coarsenEdge : called with a dart inserted after current level") ;
	assert(edgeCanBeCoarsened(d) || !"Trying to coarsen an edge that can not be coarsened") ;

	m_map.incCurrentLevel() ;
	m_map.uncutEdge(d) ;
	m_map.decCurrentLevel() ;

	unsigned int maxL = m_map.getMaxLevel() ;
	if(m_map.getCurrentLevel() == maxL - 1 && m_map.getNbInsertedDarts(maxL) == 0)
		m_map.removeLevelBack() ;
}

//template <typename PFP>
//unsigned int Map2MR<PFP>::subdivideFace(Dart d, bool triQuad, bool OneLevelDifference)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideFace : called with a dart inserted after current level") ;
//	assert(!faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

//	unsigned int fLevel = faceLevel(d) ;
//	Dart old = faceOldestDart(d) ;

//	m_map.pushLevel() ;
//	m_map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

//	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
//		m_map.addLevelBack();

//	unsigned int degree = 0 ;
//	Dart it = old ;
//	do
//	{
//		++degree ;						// compute the degree of the face

//		if(OneLevelDifference)
//		{
//			Dart nf = m_map.phi2(it) ;
//			if(faceLevel(nf) == fLevel - 1)	// check if neighboring faces have to be subdivided first
//				subdivideFace(nf,triQuad) ;
//		}

//		if(!edgeIsSubdivided(it))
//			subdivideEdge(it) ;			// and cut the edges (if they are not already)
//		it = m_map.phi1(it) ;
//	} while(it != old) ;

//	m_map.setCurrentLevel(fLevel + 1) ;	// go to the next level to perform face subdivision

//	if(triQuad && degree == 3)					// if subdividing a triangle
//	{
//		Dart dd = m_map.phi1(old) ;
//		Dart e = m_map.phi1(dd) ;
//		(*vertexVertexFunctor)(e) ;
//		e = m_map.phi1(e) ;
//		splitFace(dd, e) ;

//		dd = e ;
//		e = m_map.phi1(dd) ;
//		(*vertexVertexFunctor)(e) ;
//		e = m_map.phi1(e) ;
//		splitFace(dd, e) ;

//		dd = e ;
//		e = m_map.phi1(dd) ;
//		(*vertexVertexFunctor)(e) ;
//		e = m_map.phi1(e) ;
//		splitFace(dd, e) ;
//	}
//	else							// if subdividing a polygonal face
//	{
//		Dart dd = m_map.phi1(old) ;
//		Dart next = m_map.phi1(dd) ;
//		(*vertexVertexFunctor)(next) ;
//		next = m_map.phi1(next) ;
//		splitFace(dd, next) ;			// insert a first edge
//		Dart ne = m_map.phi2(m_map.phi_1(dd)) ;

//		cutEdge(ne) ;					// cut the new edge to insert the central vertex

//		dd = m_map.phi1(next) ;
//		(*vertexVertexFunctor)(dd) ;
//		dd = m_map.phi1(dd) ;
//		while(dd != ne)					// turn around the face and insert new edges
//		{								// linked to the central vertex
//			splitFace(m_map.phi1(ne), dd) ;
//			dd = m_map.phi1(dd) ;
//			(*vertexVertexFunctor)(dd) ;
//			dd = m_map.phi1(dd) ;
//		}

//		(*faceVertexFunctor)(m_map.phi1(ne)) ;
//	}

//	m_map.popLevel() ;

//	return fLevel ;
//}

template <typename PFP>
unsigned int Map2MR<PFP>::subdivideFace(Dart d, bool triQuad, bool OneLevelDifference)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideFace : called with a dart inserted after current level") ;
	assert(!faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int fLevel = faceLevel(d) ;
	Dart old = faceOldestDart(d) ;

	m_map.pushLevel() ;
	m_map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

	unsigned int degree = 0 ;
	Dart it = old ;
	do
	{
		++degree ;						// compute the degree of the face

		if(OneLevelDifference)
		{
			Dart nf = m_map.phi2(it) ;
			if(faceLevel(nf) == fLevel - 1)	// check if neighboring faces have to be subdivided first
				subdivideFace(nf,triQuad) ;
		}

		if(!edgeIsSubdivided(it))
			subdivideEdge(it) ;			// and cut the edges (if they are not already)
		it = m_map.phi1(it) ;
	} while(it != old) ;

	m_map.setCurrentLevel(fLevel + 1) ;	// go to the next level to perform face subdivision

	if(triQuad && degree == 3)					// if subdividing a triangle
	{
		Dart dd = m_map.phi1(old) ;
		Dart e = m_map.phi1(dd) ;
		(*vertexVertexFunctor)(e) ;
		e = m_map.phi1(e) ;
		//splitFace(dd, e) ;
		m_map.splitFace(dd, e) ;

		dd = e ;
		e = m_map.phi1(dd) ;
		(*vertexVertexFunctor)(e) ;
		e = m_map.phi1(e) ;
		//splitFace(dd, e) ;
		m_map.splitFace(dd, e) ;

		dd = e ;
		e = m_map.phi1(dd) ;
		(*vertexVertexFunctor)(e) ;
		e = m_map.phi1(e) ;
		//splitFace(dd, e) ;
		m_map.splitFace(dd, e) ;
	}
	else							// if subdividing a polygonal face
	{
		std::cout << "wrong" << std::endl;
		Dart dd = m_map.phi1(old) ;
		Dart next = m_map.phi1(dd) ;
		(*vertexVertexFunctor)(next) ;
		next = m_map.phi1(next) ;
		splitFace(dd, next) ;			// insert a first edge
		Dart ne = m_map.phi2(m_map.phi_1(dd)) ;

		cutEdge(ne) ;					// cut the new edge to insert the central vertex

		dd = m_map.phi1(next) ;
		(*vertexVertexFunctor)(dd) ;
		dd = m_map.phi1(dd) ;
		while(dd != ne)					// turn around the face and insert new edges
		{								// linked to the central vertex
			splitFace(m_map.phi1(ne), dd) ;
			dd = m_map.phi1(dd) ;
			(*vertexVertexFunctor)(dd) ;
			dd = m_map.phi1(dd) ;
		}

		(*faceVertexFunctor)(m_map.phi1(ne)) ;
	}

	m_map.popLevel() ;

	return fLevel ;
}


template <typename PFP>
void Map2MR<PFP>::coarsenFace(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"coarsenFace : called with a dart inserted after current level") ;
	assert(faceIsSubdividedOnce(d) || !"Trying to coarsen a non-subdivided face or a more than once subdivided face") ;

	unsigned int degree = 0 ;
	Dart fit = d ;
	do
	{
		++degree ;
		fit = m_map.phi1(fit) ;
	} while(fit != d) ;

	if(degree == 3)
	{
		fit = d ;
		do
		{
			m_map.incCurrentLevel() ;
			Dart innerEdge = m_map.phi1(fit) ;
			m_map.setCurrentLevel(m_map.getMaxLevel()) ;
			m_map.mergeFaces(innerEdge) ;
			m_map.decCurrentLevel() ;
			fit = m_map.phi1(fit) ;
		} while(fit != d) ;
	}
	else
	{
		m_map.incCurrentLevel() ;
		Dart centralV = m_map.phi1(m_map.phi1(d)) ;
		m_map.setCurrentLevel(m_map.getMaxLevel()) ;
		m_map.deleteVertex(centralV) ;
		m_map.decCurrentLevel() ;
	}

	fit = d ;
	do
	{
		if(edgeCanBeCoarsened(fit))
			coarsenEdge(fit) ;
		fit = m_map.phi1(fit) ;
	} while(fit != d) ;

	unsigned int maxL = m_map.getMaxLevel() ;
	if(m_map.getCurrentLevel() == maxL - 1 && m_map.getNbInsertedDarts(maxL) == 0)
		m_map.removeLevelBack() ;
}

template <typename PFP>
unsigned int Map2MR<PFP>::subdivideFaceSqrt3(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideFace : called with a dart inserted after current level") ;
	//assert(!faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int fLevel = faceLevel(d) ;
	Dart old = faceOldestDart(d) ;

	m_map.pushLevel() ;
	m_map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

	if(m_map.getCurrentLevel() == m_map.getMaxLevel())
		m_map.addLevelBack();

	m_map.setCurrentLevel(fLevel + 1) ;	// go to the next level to perform face subdivision

	//if it is an even level (triadic refinement) and a boundary face
	if((m_map.getCurrentLevel()%2 == 0) && m_map.isBoundaryFace(d))
	{
//		//find the boundary edge
//		Dart df = m_map.findBoundaryEdgeOfFace(d);
//		//trisection of the boundary edge
//		cutEdge(df) ;
//		splitFace(m_map.phi2(df), m_map.phi1(m_map.phi1(m_map.phi2(df)))) ;
//		(*vertexVertexFunctor)(m_map.phi1(df) ;
//
//		df = m_map.phi1(df);
//		cutEdge(df) ;
//		splitFace(m_map.phi2(df), m_map.phi1(m_map.phi1(m_map.phi2(df)))) ;
//		//(*vertexVertexFunctor)(m_map.phi1(df)) ;
	}
	else
	{
		Dart d1 = m_map.phi1(old);
		(*vertexVertexFunctor)(old) ;
		splitFace(old, d1) ;
		(*vertexVertexFunctor)(d1) ;
		cutEdge(m_map.phi_1(old)) ;
		Dart x = m_map.phi2(m_map.phi_1(old)) ;
		Dart dd = m_map.phi1(m_map.phi1(m_map.phi1((x))));
		while(dd != x)
		{
			(*vertexVertexFunctor)(dd) ;
			Dart next = m_map.phi1(dd) ;
			splitFace(dd, m_map.phi1(x)) ;
			dd = next ;
		}

		Dart cd = m_map.phi2(x);
		(*faceVertexFunctor)(cd) ;

		Dart dit = cd;
		do
		{
			Dart dit12 = m_map.phi2(m_map.phi1(dit));

			dit = m_map.phi2(m_map.phi_1(dit));

			if(faceLevel(dit12) == (fLevel + 1)  && !m_map.isBoundaryEdge(dit12))
					flipBackEdge(dit12);
		}
		while(dit != cd);
	}

	m_map.popLevel() ;

	return fLevel ;
}

} // namespace Adaptive

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
