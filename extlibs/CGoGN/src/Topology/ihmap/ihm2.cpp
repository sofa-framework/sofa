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

#include "Topology/ihmap/ihm2.h"
#include <math.h>

namespace CGoGN
{

ImplicitHierarchicalMap2::ImplicitHierarchicalMap2() : m_curLevel(0), m_maxLevel(0), m_idCount(0)
{
    m_dartLevel = Map2::addAttribute<unsigned int, DART, ImplicitHierarchicalMap2, HandlerAccessorPolicy >("dartLevel") ;
    m_edgeId = Map2::addAttribute<unsigned int, DART, ImplicitHierarchicalMap2, HandlerAccessorPolicy >("edgeId") ;
//    for(unsigned int i = 0; i < NB_ORBITS; ++i)
//        m_nextLevelCell[i] = NULL ;
    //    this->initImplicitProperties();
}

ImplicitHierarchicalMap2::~ImplicitHierarchicalMap2()
{
    removeAttribute(m_edgeId) ;
    removeAttribute(m_dartLevel) ;
}

void ImplicitHierarchicalMap2::clear(bool removeAttrib)
{
    Map2::clear(removeAttrib) ;
    if (removeAttrib)
    {
        m_dartLevel = Map2::addAttribute<unsigned int, DART, ImplicitHierarchicalMap2, HandlerAccessorPolicy >("dartLevel") ;
        m_edgeId = Map2::addAttribute<unsigned int, DART, ImplicitHierarchicalMap2, HandlerAccessorPolicy >("edgeId") ;

//        for(unsigned int i = 0; i < NB_ORBITS; ++i)
//            m_nextLevelCell[i] = NULL ;
    }
}

void ImplicitHierarchicalMap2::initImplicitProperties()
{
    //initEdgeId() ;

    //init each edge Id at 0
    for(Dart d = Map2::begin(); d != Map2::end(); Map2::next(d))
    {
        this->setEdgeId(d, 0u);
    }

//    for(unsigned int orbit = 0; orbit < NB_ORBITS; ++orbit)
//    {
//        if(m_nextLevelCell[orbit] != NULL)
//        {
//            AttributeContainer& cellCont = m_attribs[orbit] ;
//            for(unsigned int i = cellCont.begin(); i < cellCont.end(); cellCont.next(i))
//                m_nextLevelCell[orbit]->operator[](i) = EMBNULL ;
//        }
//    }
}

void ImplicitHierarchicalMap2::initEdgeId()
{
    m_idCount = 0 ;
    DartMarker<Map2> edgeMark(*this) ;
    for(Dart d = Map2::begin(); d != Map2::end(); Map2::next(d))
    {
        if(!edgeMark.isMarked(d))
        {
            edgeMark.markOrbit<EDGE>(d) ;
            m_edgeId[d] = m_idCount ;
            m_edgeId[Map2::phi2(d)] = m_idCount++ ;
        }
    }
}

unsigned int ImplicitHierarchicalMap2::vertexDegree(Dart d)
{
    unsigned int count = 0 ;
    Dart it = d ;
    do
    {
        ++count ;
        it = phi2(phi_1(it)) ;
    } while (it != d) ;
    return count ;
}

//unsigned int ImplicitHierarchicalMap2::faceLevel(Dart d)
//{
//	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

//	if(m_curLevel == 0)
//		return 0 ;

////	unsigned int cur = m_curLevel ;
////	Dart it = d ;
////	Dart end = d ;
////	bool resetEnd = true ;
////	bool firstEdge = true ;
////	do
////	{
////		if(!resetEnd)
////			firstEdge = false ;
////
////		unsigned int eId = m_edgeId[it] ;
////		Dart next = it ;
////		do
////		{
////			unsigned int l = edgeLevel(next) ;
////			if(l < m_curLevel)
////				m_curLevel = l ;
////			else // l == curLevel
////			{
////				if(!firstEdge)
////				{
////					--m_curLevel ;
////					next = it ;
////				}
////			}
////			next = phi1(next) ;
////		} while(m_edgeId[next] == eId) ;
////		it = next ;
////
////		if(resetEnd)
////		{
////			end = it ;
////			resetEnd = false ;
////		}
////
////	} while(!firstEdge && it != end) ;
////
////	unsigned int fLevel = m_curLevel ;
////	m_curLevel = cur ;

//	Dart it = d ;
//	Dart old = it ;
//	unsigned int l_old = m_dartLevel[old] ;
//	unsigned int fLevel = edgeLevel(it) ;
//	do
//	{
//		it = phi1(it) ;
//		unsigned int dl = m_dartLevel[it] ;
//		if(dl < l_old)							// compute the oldest dart of the face
//		{										// in the same time
//			old = it ;
//			l_old = dl ;
//		}										// in a first time, the level of a face
//		unsigned int l = edgeLevel(it) ;		// is the minimum of the levels
//		fLevel = l < fLevel ? l : fLevel ;		// of its edges
//	} while(it != d) ;

//	unsigned int cur = m_curLevel ;
//	m_curLevel = fLevel ;

//	unsigned int nbSubd = 0 ;
//	it = old ;
//	unsigned int eId = m_edgeId[old] ;			// the particular case of a face
//	do											// with all neighboring faces regularly subdivided
//	{											// but not the face itself
//		++nbSubd ;								// is treated here
//		it = phi1(it) ;
//	} while(m_edgeId[it] == eId) ;

//	while(nbSubd > 1)
//	{
//		nbSubd /= 2 ;
//		--fLevel ;
//	}

//	m_curLevel = cur ;

//	return fLevel ;
//}

//Dart ImplicitHierarchicalMap2::faceOrigin(Dart d)
//{
//	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
//	unsigned int cur = m_curLevel ;
//	Dart p = d ;
//	unsigned int pLevel = m_dartLevel[p] ;
//	while(pLevel > 0)
//	{
//		p = faceOldestDart(p) ;
//		pLevel = m_dartLevel[p] ;
//		m_curLevel = pLevel ;
//	}
//	m_curLevel = cur ;
//	return p ;
//}

//Dart ImplicitHierarchicalMap2::faceOldestDart(Dart d)
//{
//	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
//	Dart it = d ;
//	Dart oldest = it ;
//	unsigned int l_old = m_dartLevel[oldest] ;
//	do
//	{
//		unsigned int l = m_dartLevel[it] ;
//		if(l == 0)
//			return it ;
//		if(l < l_old)
////		if(l < l_old || (l == l_old && it < oldest))
//		{
//			oldest = it ;
//			l_old = l ;
//		}
//		it = phi1(it) ;
//	} while(it != d) ;
//	return oldest ;
//}

//bool ImplicitHierarchicalMap2::edgeIsSubdivided(Dart d)
//{
//	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
////	Dart d2 = phi2(d) ;
//	Dart d1 = phi1(d) ;
//	++m_curLevel ;
////	Dart d2_l = phi2(d) ;
//	Dart d1_l = phi1(d) ;
//	--m_curLevel ;
//	if(d1 != d1_l)
//		return true ;
//	else
//		return false ;
//}

//bool ImplicitHierarchicalMap2::edgeCanBeCoarsened(Dart d)
//{
//	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
//	bool subd = false ;
//	bool subdOnce = true ;
//	bool degree2 = false ;
//	if(edgeIsSubdivided(d))
//	{
//		subd = true ;
//		Dart d2 = phi2(d) ;
//		++m_curLevel ;
//		if(vertexDegree(phi1(d)) == 2)
//		{
//			degree2 = true ;
//			if(edgeIsSubdivided(d) || edgeIsSubdivided(d2))
//				subdOnce = false ;
//		}
//		--m_curLevel ;
//	}
//	return subd && degree2 && subdOnce ;
//}

//bool ImplicitHierarchicalMap2::faceIsSubdivided(Dart d)
//{
//	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
//	unsigned int fLevel = faceLevel(d) ;
//	if(fLevel < m_curLevel)
//		return false ;

//	bool subd = false ;
//	++m_curLevel ;
//	if(m_dartLevel[phi1(d)] == m_curLevel && m_edgeId[phi1(d)] != m_edgeId[d])
//		subd = true ;
//	--m_curLevel ;
//	return subd ;
//}

//bool ImplicitHierarchicalMap2::faceIsSubdividedOnce(Dart d)
//{
//	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
//	unsigned int fLevel = faceLevel(d) ;
//	if(fLevel < m_curLevel)		// a face whose level in the current level map is lower than
//		return false ;			// the current level can not be subdivided to higher levels

//	unsigned int degree = 0 ;
//	bool subd = false ;
//	bool subdOnce = true ;
//	Dart fit = d ;
//	do
//	{
//		++m_curLevel ;
//		if(m_dartLevel[phi1(fit)] == m_curLevel && m_edgeId[phi1(fit)] != m_edgeId[fit])
//		{
//			subd = true ;
//			++m_curLevel ;
//			if(m_dartLevel[phi1(fit)] == m_curLevel && m_edgeId[phi1(fit)] != m_edgeId[fit])
//				subdOnce = false ;
//			--m_curLevel ;
//		}
//		--m_curLevel ;
//		++degree ;
//		fit = phi1(fit) ;
//	} while(subd && subdOnce && fit != d) ;

//	if(degree == 3 && subd)
//	{
//		++m_curLevel ;
//		Dart cf = phi2(phi1(d)) ;
//		++m_curLevel ;
//		if(m_dartLevel[phi1(cf)] == m_curLevel && m_edgeId[phi1(cf)] != m_edgeId[cf])
//			subdOnce = false ;
//		--m_curLevel ;
//		--m_curLevel ;
//	}

//	return subd && subdOnce ;
//}

} //namespace CGoGN
