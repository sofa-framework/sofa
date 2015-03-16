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

#include "Topology/ihmap/ihm3.h"
#include <math.h>

namespace CGoGN
{

ImplicitHierarchicalMap3::ImplicitHierarchicalMap3() : m_curLevel(0), m_maxLevel(0), m_edgeIdCount(0), m_faceIdCount(0)
{
    m_dartLevel = Map3::addAttribute<unsigned int, DART, ImplicitHierarchicalMap3>("dartLevel") ;
    m_edgeId = Map3::addAttribute<unsigned int, DART, ImplicitHierarchicalMap3>("edgeId") ;
    m_faceId = Map3::addAttribute<unsigned int, DART, ImplicitHierarchicalMap3>("faceId") ;

    for(unsigned int i = 0; i < NB_ORBITS; ++i)
        m_nextLevelCell[i] = NULL ;
}

ImplicitHierarchicalMap3::~ImplicitHierarchicalMap3()
{
    removeAttribute(m_edgeId) ;
    removeAttribute(m_faceId) ;
    removeAttribute(m_dartLevel) ;
}

void ImplicitHierarchicalMap3::clear(bool removeAttrib)
{
    Map3::clear(removeAttrib) ;
    if (removeAttrib)
    {
        m_dartLevel = Map3::addAttribute<unsigned int, DART, ImplicitHierarchicalMap3>("dartLevel") ;
        m_faceId = Map3::addAttribute<unsigned int, DART, ImplicitHierarchicalMap3>("faceId") ;
        m_edgeId = Map3::addAttribute<unsigned int, DART, ImplicitHierarchicalMap3>("edgeId") ;

        for(unsigned int i = 0; i < NB_ORBITS; ++i)
            m_nextLevelCell[i] = NULL ;
    }
}

void ImplicitHierarchicalMap3::initImplicitProperties()
{
	//initEdgeId() ;
	//initFaceId();

	for(Dart d = Map3::begin(); d != Map3::end(); Map3::next(d))
	{
		m_edgeId[d] = 0;
		m_faceId[d] = 0;
	}

    for(unsigned int orbit = 0; orbit < NB_ORBITS; ++orbit)
    {
        if(m_nextLevelCell[orbit] != NULL)
        {
            AttributeContainer& cellCont = m_attribs[orbit] ;
            for(unsigned int i = cellCont.begin(); i < cellCont.end(); cellCont.next(i))
                m_nextLevelCell[orbit]->operator[](i) = EMBNULL ;
        }
    }
}

void ImplicitHierarchicalMap3::initEdgeId()
{
    DartMarkerStore<Map3> edgeMark(*this) ;
    for(Dart d = Map3::begin(); d != Map3::end(); Map3::next(d))
    {
        if(!edgeMark.isMarked(d))
        {
            Dart e = d;
            do
            {
                m_edgeId[e] = m_edgeIdCount;
                edgeMark.mark(e);

                m_edgeId[Map3::phi2(e)] = m_edgeIdCount ;
                edgeMark.mark(Map3::phi2(e));

                e = Map3::alpha2(e);
            } while(e != d);

            m_edgeIdCount++;
        }
    }
}

void ImplicitHierarchicalMap3::initFaceId()
{
    DartMarkerStore<Map3> faceMark(*this) ;
    for(Dart d = Map3::begin(); d != Map3::end(); Map3::next(d))
    {
        if(!faceMark.isMarked(d))
        {
            Dart e = d;
            do
            {
                m_faceId[e] = m_faceIdCount ;
                faceMark.mark(e);

                Dart e3 = Map3::phi3(e);
                m_faceId[e3] = m_faceIdCount ;
                faceMark.mark(e3);

                e = Map3::phi1(e);
            } while(e != d);

            m_faceIdCount++;
        }
    }
}

} //namespace CGoGN
