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

#include "Algo/Multiresolution/IHM2/ihm2_PrimalAdapt.h"

namespace CGoGN
{

namespace Algo
{

namespace MR
{

namespace Primal
{

namespace Adaptive
{

template <typename PFP>
IHM2<PFP>::IHM2(MAP& map) :
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
unsigned int IHM2<PFP>::edgeLevel(Dart d)
{
    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    unsigned int ld = m_map.getDartLevel(d);
//	unsigned int ldd = m_dartLevel[phi2(d)] ;	// the level of an edge is the maximum of the
    unsigned int ldd = m_map.getDartLevel(m_map.phi1(d));
    return ld < ldd ? ldd : ld ;
}

template <typename PFP>
unsigned int IHM2<PFP>::faceLevel(Dart d)
{
    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

    if(m_map.getCurrentLevel() == 0)
        return 0 ;

// Methode 1
//	unsigned int cur = m_curLevel ;
//	Dart it = d ;
//	Dart end = d ;
//	bool resetEnd = true ;
//	bool firstEdge = true ;
//	do
//	{
//		if(!resetEnd)
//			firstEdge = false ;
//
//		unsigned int eId = m_edgeId[it] ;
//		Dart next = it ;
//		do
//		{
//			unsigned int l = edgeLevel(next) ;
//			if(l < m_curLevel)
//				m_curLevel = l ;
//			else // l == curLevel
//			{
//				if(!firstEdge)
//				{
//					--m_curLevel ;
//					next = it ;
//				}
//			}
//			next = phi1(next) ;
//		} while(m_edgeId[next] == eId) ;
//		it = next ;
//
//		if(resetEnd)
//		{
//			end = it ;
//			resetEnd = false ;
//		}
//
//	} while(!firstEdge && it != end) ;
//
//	unsigned int fLevel = m_curLevel ;
//	m_curLevel = cur ;

//Methode 2
    Dart it = d ;
    Dart old = it ;
    unsigned int l_old = m_map.getDartLevel(old) ;
    unsigned int fLevel = edgeLevel(it) ;
    do
    {
        it = m_map.phi1(it) ;
        unsigned int dl = m_map.getDartLevel(it) ;
        if(dl < l_old)							// compute the oldest dart of the face
        {										// in the same time
            old = it ;
            l_old = dl ;
        }										// in a first time, the level of a face
        unsigned int l = edgeLevel(it) ;		// is the minimum of the levels
        fLevel = l < fLevel ? l : fLevel ;		// of its edges
    } while(it != d) ;

    unsigned int cur = m_map.getCurrentLevel() ;
    m_map.setCurrentLevel(fLevel) ;

//    unsigned int nbSubd = 0 ;
//    it = old ;
//    unsigned int eId = m_map.getEdgeId(old) ;			// the particular case of a face
//    do											// with all neighboring faces regularly subdivided
//    {											// but not the face itself
//        ++nbSubd ;								// is treated here
//        it = m_map.phi1(it) ;
//    } while(m_map.getEdgeId(it) == eId) ;

//    while(nbSubd > 1)
//    {
//        nbSubd /= 2 ;
//        --fLevel ;
//    }

    m_map.setCurrentLevel(cur) ;

    return fLevel ;
}


template <typename PFP>
Dart IHM2<PFP>::faceOrigin(Dart d)
{
    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    unsigned int cur = m_map.getCurrentLevel() ;
    Dart p = d ;
    unsigned int pLevel = m_map.getDartLevel(p) ;
    while(pLevel > 0)
    {
        p = faceOldestDart(p) ;
        pLevel = m_map.getDartLevel(p) ;
        m_map.setCurrentLevel(pLevel) ;
    }
    m_map.setCurrentLevel(cur) ;
    return p ;
}

template <typename PFP>
Dart IHM2<PFP>::faceOldestDart(Dart d)
{
    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    Dart it = d ;
    Dart oldest = it ;
    unsigned int l_old = m_map.getDartLevel(oldest) ;
    do
    {
        unsigned int l = m_map.getDartLevel(it) ;
        if(l == 0)
            return it ;
        if(l < l_old)
//		if(l < l_old || (l == l_old && it < oldest))
        {
            oldest = it ;
            l_old = l ;
        }
        it = m_map.phi1(it) ;
    } while(it != d) ;
    return oldest ;
}

template <typename PFP>
bool IHM2<PFP>::edgeIsSubdivided(Dart d)
{
    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

    if(m_map.getCurrentLevel() == m_map.getMaxLevel())
        return false ;

//	Dart d2 = phi2(d) ;
    Dart d1 = m_map.phi1(d) ;
    m_map.incCurrentLevel() ;
//	Dart d2_l = phi2(d) ;
    Dart d1_l = m_map.phi1(d) ;
    m_map.decCurrentLevel();
    if(d1 != d1_l)
        return true ;
    else
        return false ;
}

template <typename PFP>
bool IHM2<PFP>::edgeCanBeCoarsened(Dart d)
{
    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    bool subd = false ;
    bool subdOnce = true ;
    bool degree2 = false ;
    if(edgeIsSubdivided(d))
    {
        subd = true ;
        Dart d2 = m_map.phi2(d) ;
        m_map.incCurrentLevel() ;
        if(m_map.vertexDegree(m_map.phi1(d)) == 2)
        {
            degree2 = true ;
            if(edgeIsSubdivided(d) || edgeIsSubdivided(d2))
                subdOnce = false ;
        }
        m_map.decCurrentLevel() ;

    }
    return subd && degree2 && subdOnce ;
}

template <typename PFP>
bool IHM2<PFP>::faceIsSubdivided(Dart d)
{
    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;

    unsigned int fLevel = faceLevel(d) ;
    if(fLevel <= m_map.getCurrentLevel())
        return false ;

    bool subd = false ;
    m_map.incCurrentLevel() ;
    if(m_map.getDartLevel(m_map.phi1(d)) == m_map.getCurrentLevel() && m_map.getEdgeId(m_map.phi1(d)) != m_map.getEdgeId(d))
        subd = true ;
    m_map.decCurrentLevel() ;
    return subd ;
}

template <typename PFP>
bool IHM2<PFP>::faceIsSubdividedOnce(Dart d)
{
    assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
    unsigned int fLevel = faceLevel(d) ;
    if(fLevel < m_map.getCurrentLevel())		// a face whose level in the current level map is lower than
        return false ;			// the current level can not be subdivided to higher levels

    unsigned int degree = 0 ;
    bool subd = false ;
    bool subdOnce = true ;
    Dart fit = d ;
    do
    {
        m_map.incCurrentLevel() ;
        if(m_map.getDartLevel(m_map.phi1(fit)) == m_map.getCurrentLevel() && m_map.getEdgeId(m_map.phi1(fit)) != m_map.getEdgeId(fit))
        {
            subd = true ;
            m_map.incCurrentLevel() ;
            if(m_map.getDartLevel(m_map.phi1(fit)) == m_map.getCurrentLevel() && m_map.getEdgeId(m_map.phi1(fit)) != m_map.getEdgeId(fit))
                subdOnce = false ;
            m_map.decCurrentLevel() ;
        }
        m_map.decCurrentLevel() ;
        ++degree ;
        fit = m_map.phi1(fit) ;
    } while(subd && subdOnce && fit != d) ;

    if(degree == 3 && subd)
    {
        m_map.incCurrentLevel() ;
        Dart cf = m_map.phi2(m_map.phi1(d)) ;
        m_map.incCurrentLevel() ;
        if(m_map.getDartLevel(m_map.phi1(cf)) == m_map.getCurrentLevel() && m_map.getEdgeId(m_map.phi1(cf)) != m_map.getEdgeId(cf))
            subdOnce = false ;
        m_map.decCurrentLevel() ;
        m_map.decCurrentLevel() ;
    }

    return subd && subdOnce ;
}

/***************************************************
 *               SUBDIVISION                       *
 ***************************************************/

template <typename PFP>
void IHM2<PFP>::subdivideEdge(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideEdge : called with a dart inserted after current level") ;
    assert(!edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;

    unsigned int eLevel = edgeLevel(d) ;

	unsigned int cur = m_map.getCurrentLevel() ;
	m_map.setCurrentLevel(eLevel) ;

    Dart dd = m_map.phi2(d) ;

	m_map.setCurrentLevel(eLevel + 1) ;

    m_map.cutEdge(d) ;
	unsigned int eId = m_map.getEdgeId(d) ;
    m_map.setEdgeId(m_map.phi1(d), eId) ;
    m_map.setEdgeId(m_map.phi1(dd), eId) ;
    (*edgeVertexFunctor)(m_map.phi1(d)) ;

	m_map.setCurrentLevel(cur) ;
}

template <typename PFP>
void IHM2<PFP>::coarsenEdge(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"coarsenEdge : called with a dart inserted after current level") ;
	assert(m_map.edgeCanBeCoarsened(d) || !"Trying to coarsen an edge that can not be coarsened") ;


	unsigned int cur = m_map.getCurrentLevel() ;
//	Dart e = m_map.phi2(d) ;
	m_map.setCurrentLevel(cur + 1) ;
//	unsigned int dl = m_map.getDartLevel(e) ;
//	m_map.setDartLevel(m_map.phi1(e), dl) ;
//	m_map.collapseEdge(e) ;
	m_map.uncutEdge(d) ;
	m_map.setCurrentLevel(cur) ;
}

template <typename PFP>
unsigned int IHM2<PFP>::subdivideFace(Dart d, bool triQuad, bool OneLevelDifference)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideFace : called with a dart inserted after current level") ;
    assert(!faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

    unsigned int fLevel = faceLevel(d) ;
    Dart old = faceOldestDart(d) ;

	//std::cout << "faceLevel = " << fLevel << std::endl;

	unsigned int cur = m_map.getCurrentLevel() ;
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


	if((degree == 3) && triQuad)					// if subdividing a triangle
    {
        Dart dd = m_map.phi1(old) ;
        Dart e = m_map.phi1(dd) ;
        (*vertexVertexFunctor)(e) ;

        e = m_map.phi1(e) ;
        m_map.splitFace(dd, e) ;					// insert a new edge
//        unsigned int id = m_map.getNewEdgeId() ;
		unsigned int id = m_map.getTriRefinementEdgeId(m_map.phi_1(dd));
		m_map.setEdgeId(m_map.phi_1(dd), id) ;		// set the edge id of the inserted
		m_map.setEdgeId(m_map.phi_1(e), id) ;		// edge to the next available id

        dd = e ;
        e = m_map.phi1(dd) ;
        (*vertexVertexFunctor)(e) ;
        e = m_map.phi1(e) ;
        m_map.splitFace(dd, e) ;
		//id = m_map.getNewEdgeId() ;
		id = m_map.getTriRefinementEdgeId(m_map.phi_1(dd));
        m_map.setEdgeId(m_map.phi_1(dd), id) ;
        m_map.setEdgeId(m_map.phi_1(e), id) ;

        dd = e ;
        e = m_map.phi1(dd) ;
        (*vertexVertexFunctor)(e) ;
        e = m_map.phi1(e) ;
        m_map.splitFace(dd, e) ;
		//id = m_map.getNewEdgeId() ;
		id = m_map.getTriRefinementEdgeId(m_map.phi_1(dd));
        m_map.setEdgeId(m_map.phi_1(dd), id) ;
        m_map.setEdgeId(m_map.phi_1(e), id) ;
    }
    else											// if subdividing a polygonal face
    {
        Dart dd = m_map.phi1(old) ;
        Dart next = m_map.phi1(dd) ;
        (*vertexVertexFunctor)(next) ;
        next = m_map.phi1(next) ;
        m_map.splitFace(dd, next) ;	// insert a first edge
        Dart ne = m_map.alpha1(dd) ;
        Dart ne2 = m_map.phi2(ne) ;

        m_map.cutEdge(ne) ;							// cut the new edge to insert the central vertex
		//unsigned int id = m_map.getNewEdgeId() ;
		unsigned int id = m_map.getQuadRefinementEdgeId(m_map.phi2(ne));
        m_map.setEdgeId(ne, id) ;
        m_map.setEdgeId(m_map.phi2(ne), id) ;			// set the edge id of the inserted
		//id = m_map.getNewEdgeId() ;
		id = m_map.getQuadRefinementEdgeId(ne2);
        m_map.setEdgeId(ne2, id) ;					// edges to the next available ids
        m_map.setEdgeId(m_map.phi2(ne2), id) ;

        dd = m_map.phi1(next) ;
        (*vertexVertexFunctor)(dd) ;
        dd = m_map.phi1(dd) ;
        while(dd != ne)								// turn around the face and insert new edges
        {											// linked to the central vertex
            m_map.splitFace(m_map.phi1(ne), dd) ;
            Dart nne = m_map.alpha1(dd) ;
			//id = m_map.getNewEdgeId() ;
			id = m_map.getQuadRefinementEdgeId(m_map.phi2(nne));
            m_map.setEdgeId(nne, id) ;
            m_map.setEdgeId(m_map.phi2(nne), id) ;
            dd = m_map.phi1(dd) ;
            (*vertexVertexFunctor)(dd) ;
            dd = m_map.phi1(dd) ;
        }

        (*faceVertexFunctor)(m_map.phi1(ne)) ;
    }

	m_map.setCurrentLevel(cur) ;

	return fLevel ;
}

template <typename PFP>
void IHM2<PFP>::coarsenFace(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"coarsenFace : called with a dart inserted after current level") ;
    assert(faceIsSubdividedOnce(d) || !"Trying to coarsen a non-subdivided face or a more than once subdivided face") ;

	unsigned int cur = m_map.getCurrentLevel() ;

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
			m_map.setCurrentLevel(cur + 1) ;
			Dart innerEdge = m_map.phi1(fit) ;
			m_map.setCurrentLevel(m_map.getMaxLevel()) ;
			m_map.mergeFaces(innerEdge) ;
			m_map.setCurrentLevel(cur) ;
			fit = m_map.phi1(fit) ;
		} while(fit != d) ;
	}
	else
	{
		m_map.setCurrentLevel(cur + 1) ;
		Dart centralV = m_map.phi1(m_map.phi1(d)) ;
		m_map.setCurrentLevel(m_map.getMaxLevel()) ;
		m_map.deleteVertex(centralV) ;
		m_map.setCurrentLevel(cur) ;
	}

	fit = d ;
	do
	{
		if(m_map.edgeCanBeCoarsened(fit))
			coarsenEdge(fit) ;
		fit = m_map.phi1(fit) ;
	} while(fit != d) ;
}


} // namespace Adaptive

} // namespace Primal

} // namespace MR

} // namespace Algo

} // namespace CGoGN
