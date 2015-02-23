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
#include "ihm2.h"
namespace CGoGN
{

template <typename T, unsigned int ORBIT>
inline void AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::registerInMap()
{
    boost::mutex::scoped_lock lockAH(m_map->attributeHandlersMutex);
    m_map->attributeHandlers.insert(std::pair<AttributeMultiVectorGen*, AttributeHandlerGen*>(m_attrib, this)) ;
}

template <typename T, unsigned int ORBIT>
inline void AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::unregisterFromMap()
{
    typedef std::multimap<AttributeMultiVectorGen*, AttributeHandlerGen*>::iterator IT ;

    boost::mutex::scoped_lock lockAH(m_map->attributeHandlersMutex);
    std::pair<IT, IT> bounds = m_map->attributeHandlers.equal_range(m_attrib) ;
    for(IT i = bounds.first; i != bounds.second; ++i)
    {
        if((*i).second == this)
        {
            m_map->attributeHandlers.erase(i) ;
            return ;
        }
    }
    assert(false || !"Should not get here") ;
}


template <typename T, unsigned int ORBIT>
AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::AttributeHandler() :
    AttributeHandlerGen(false),
    m_map(NULL),
    m_attrib(NULL)
{
//    std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
}

template <typename T, unsigned int ORBIT>
AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::AttributeHandler(MAP* m, AttributeMultiVector<T>* amv) :
    AttributeHandlerGen(false),
    m_map(m),
    m_attrib(amv)
{
    if(m != NULL && amv != NULL && amv->getIndex() != AttributeContainer::UNKNOWN)
    {
        assert(ORBIT == amv->getOrbit() || !"AttributeHandler: orbit incompatibility") ;
        valid = true ;
        registerInMap() ;
    }
    else
        valid = false ;
}

template <typename T, unsigned int ORBIT>
AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::AttributeHandler(const AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>& ta) :
    AttributeHandlerGen(ta.valid),
    m_map(ta.m_map),
    m_attrib(ta.m_attrib)
{
    if(valid)
        registerInMap() ;
}

template <typename T, unsigned int ORBIT>
inline AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>& AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::operator=(const AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>& ta)
{

    if(valid)
        unregisterFromMap() ;
    m_map = ta.m_map ;
    m_attrib = ta.m_attrib ;
    valid = ta.valid ;
    if(valid)
        registerInMap() ;
    return *this ;
}

template <typename T, unsigned int ORBIT>
AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::~AttributeHandler()
{
    if(valid)
        unregisterFromMap() ;
}

template <typename T, unsigned int ORBIT>
AttributeMultiVector<T>* AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::getDataVector() const
{
    return m_attrib ;
}

template <typename T, unsigned int ORBIT>
AttributeMultiVectorGen* AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::getDataVectorGen() const
{
    return m_attrib ;
}

template <typename T, unsigned int ORBIT>
int AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::getSizeOfType() const
{
    return sizeof(T) ;
}

template <typename T, unsigned int ORBIT>
unsigned int AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::getOrbit() const
{
    return ORBIT ;
}

template <typename T, unsigned int ORBIT>
inline unsigned int AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::getIndex() const
{
    return m_attrib->getIndex() ;
}

template <typename T, unsigned int ORBIT>
inline const std::string& AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::name() const
{
    return m_attrib->getName() ;
}

template <typename T, unsigned int ORBIT>
inline const std::string& AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::typeName() const
{
    return m_attrib->getTypeName();
}


template <typename T, unsigned int ORBIT>
inline unsigned int AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::nbElements() const
{
    return m_map->template getAttributeContainer<ORBIT>().size() ;
}

template <typename T, unsigned int ORBIT>
inline T& AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::operator[](Cell<ORBIT> c)
{
    assert(m_map->m_dartLevel[c.dart] <= m_map->m_curLevel || !"Access to a dart introduced after current level") ;
    assert(m_map->vertexInsertionLevel(c) <= m_map->m_curLevel || !"Access to the embedding of a vertex inserted after current level") ;

    const unsigned int orbit = this->getOrbit() ;
    const unsigned int nbSteps = m_map->m_curLevel - m_map->vertexInsertionLevel(c) ;
    unsigned int index = m_map->getEmbedding<ORBIT>(c) ;

    if(index == EMBNULL)
    {
        index = Algo::Topo::setOrbitEmbeddingOnNewCell<ORBIT>(*m_map, c) ;
        m_map->m_nextLevelCell[orbit]->operator[](index) = EMBNULL ;
    }

    AttributeContainer& cont = m_map->getAttributeContainer<ORBIT>() ;
    unsigned int step = 0 ;
    while(step < nbSteps)
    {
        step++ ;
        unsigned int nextIdx = m_map->m_nextLevelCell[orbit]->operator[](index) ;
        if (nextIdx == EMBNULL)
        {
            nextIdx = m_map->newCell<ORBIT>() ;
            m_map->copyCell<ORBIT>(nextIdx, index) ;
            m_map->m_nextLevelCell[orbit]->operator[](index) = nextIdx ;
            m_map->m_nextLevelCell[orbit]->operator[](nextIdx) = EMBNULL ;
            cont.refLine(index) ;
        }
        index = nextIdx ;
    }
    return this->m_attrib->operator[](index);
}

template <typename T, unsigned int ORBIT>
inline const T& AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::operator[](Cell<ORBIT> c) const
{
    assert(m_map->m_dartLevel[c.dart] <= m_map->m_curLevel || !"Access to a dart introduced after current level") ;
    assert(m_map->vertexInsertionLevel(c) <= m_map->m_curLevel || !"Access to the embedding of a vertex inserted after current level") ;

    const unsigned int orbit = this->getOrbit() ;
    const unsigned int nbSteps = m_map->m_curLevel - m_map->vertexInsertionLevel(c) ;
    unsigned int index = m_map->getEmbedding<ORBIT>(c) ;

    unsigned int step = 0 ;
    while(step < nbSteps)
    {
        step++ ;
        unsigned int next = m_map->m_nextLevelCell[orbit]->operator[](index) ;
        if(next != EMBNULL) index = next ;
        else break ;
    }
    return this->m_attrib->operator[](index);
}

template <typename T, unsigned int ORBIT>
inline T& AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::operator[](unsigned int a)
{
    assert(valid || !"Invalid AttributeHandler") ;
    return m_attrib->operator[](a) ;
}

template <typename T, unsigned int ORBIT>
inline const T& AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::operator[](unsigned int a) const
{
    assert(valid || !"Invalid AttributeHandler") ;
    return m_attrib->operator[](a) ;
}

template <typename T, unsigned int ORBIT>
inline unsigned int AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::insert(const T& elt)
{
    assert(valid || !"Invalid AttributeHandler") ;
    unsigned int idx = m_map->template getAttributeContainer<ORBIT>().insertLine() ;
    m_attrib->operator[](idx) = elt ;
    return idx ;
}

template <typename T, unsigned int ORBIT>
inline unsigned int AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::newElt()
{
    assert(valid || !"Invalid AttributeHandler") ;
    unsigned int idx = m_map->template getAttributeContainer<ORBIT>().insertLine() ;
    return idx ;
}

template <typename T, unsigned int ORBIT>
inline void AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::setAllValues(const T& v)
{
    for(unsigned int i = begin(); i != end(); next(i))
        m_attrib->operator[](i) = v ;
}

template <typename T, unsigned int ORBIT>
inline unsigned int AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::begin() const
{
    assert(valid || !"Invalid AttributeHandler") ;
    return m_map->template getAttributeContainer<ORBIT>().begin() ;
}

template <typename T, unsigned int ORBIT>
inline unsigned int AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::end() const
{
    assert(valid || !"Invalid AttributeHandler") ;
    return m_map->template getAttributeContainer<ORBIT>().end() ;
}

template <typename T, unsigned int ORBIT>
inline void AttributeHandler<T, ORBIT, ImplicitHierarchicalMap2>::next(unsigned int& iter) const
{
    assert(valid || !"Invalid AttributeHandler") ;
    m_map->template getAttributeContainer<ORBIT>().next(iter) ;
}




/***************************************************
 *             ATTRIBUTES MANAGEMENT               *
 ***************************************************/

template <typename T, unsigned int ORBIT, typename MAP>
AttributeHandler<T, ORBIT, MAP> ImplicitHierarchicalMap2::addAttribute(const std::string& nameAttr)
{
    bool addNextLevelCell = false ;
    if(!isOrbitEmbedded<ORBIT>())
        addNextLevelCell = true ;

    AttributeHandler<T, ORBIT, MAP> h = TOPO_MAP::addAttribute<T, ORBIT, MAP>(nameAttr) ;

    if(addNextLevelCell)
    {
        AttributeContainer& cellCont = m_attribs[ORBIT] ;
        AttributeMultiVector<unsigned int>* amv = cellCont.addAttribute<unsigned int>("nextLevelCell") ;
        m_nextLevelCell[ORBIT] = amv ;
        for(unsigned int i = cellCont.begin(); i < cellCont.end(); cellCont.next(i))
            amv->operator[](i) = EMBNULL ;
    }

    return AttributeHandler<T, ORBIT, MAP>(this, h.getDataVector()) ;
}

template <typename T, unsigned int ORBIT, typename MAP>
AttributeHandler<T, ORBIT, MAP> ImplicitHierarchicalMap2::getAttribute(const std::string& nameAttr)
{
    return AttributeHandler<T, ORBIT, MAP>(this, TOPO_MAP::getAttribute<T, ORBIT, MAP>(nameAttr).getDataVector()) ;
}


inline void ImplicitHierarchicalMap2::update_topo_shortcuts()
{
    //	TOPO_MAP::update_topo_shortcuts();
    m_dartLevel = EmbeddedMap2::getAttribute<unsigned int, DART, EmbeddedMap2>("dartLevel") ;
    m_edgeId = EmbeddedMap2::getAttribute<unsigned int, DART, EmbeddedMap2>("edgeId") ;

    //AttributeContainer& cont = m_attribs[DART] ;
    //m_nextLevelCell = cont.getDataVector<unsigned int>(cont.getAttributeIndex("nextLevelCell")) ;
}

/***************************************************
 *                 MAP TRAVERSAL                   *
 ***************************************************/

inline Dart ImplicitHierarchicalMap2::newDart()
{
    Dart d = TOPO_MAP::newDart() ;
    m_dartLevel[d] = m_curLevel ;
    if(m_curLevel > m_maxLevel)			// update max level
        m_maxLevel = m_curLevel ;		// if needed
    return d ;
}

inline Dart ImplicitHierarchicalMap2::phi1(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
    bool finished = false ;
    unsigned int edgeId = m_edgeId[d] ;
    Dart it = d ;
    do
    {
        it = TOPO_MAP::phi1(it) ;
        if(m_dartLevel[it] <= m_curLevel)
            finished = true ;
        else
        {
            while(m_edgeId[it] != edgeId)
                it = TOPO_MAP::phi1(TOPO_MAP::phi2(it)) ;
        }
    } while(!finished) ;
    return it ;
}

inline Dart ImplicitHierarchicalMap2::phi_1(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
    bool finished = false ;
    Dart it = TOPO_MAP::phi_1(d) ;
    unsigned int edgeId = m_edgeId[it] ;
    do
    {
        if(m_dartLevel[it] <= m_curLevel)
            finished = true ;
        else
        {
            it = TOPO_MAP::phi_1(it) ;
            while(m_edgeId[it] != edgeId)
                it = TOPO_MAP::phi_1(TOPO_MAP::phi2(it)) ;
        }
    } while(!finished) ;
    return it ;
}

inline Dart ImplicitHierarchicalMap2::phi2(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
    if(TOPO_MAP::phi2(d) == d)
        return d ;
    return TOPO_MAP::phi2(TOPO_MAP::phi_1(phi1(d))) ;
}

inline Dart ImplicitHierarchicalMap2::alpha0(Dart d) const
{
    return phi2(d) ;
}

inline Dart ImplicitHierarchicalMap2::alpha1(Dart d) const
{
    return TOPO_MAP::alpha1(d) ;
}

inline Dart ImplicitHierarchicalMap2::alpha_1(Dart d) const
{
    return TOPO_MAP::alpha_1(d) ;
}

inline Dart ImplicitHierarchicalMap2::begin() const
{
    Dart d = TOPO_MAP::begin() ;
    //	while(d != TOPO_MAP::end() && m_dartLevel[d] > m_curLevel)
    //		TOPO_MAP::next(d) ;
    return d ;
}

inline Dart ImplicitHierarchicalMap2::end() const
{
    return TOPO_MAP::end() ;
}

inline void ImplicitHierarchicalMap2::next(Dart& d) const
{
    //	do
    //	{
    //		TOPO_MAP::next(d) ;
    //	} while(d != TOPO_MAP::end() && m_dartLevel[d] > m_curLevel) ;
    TOPO_MAP::next(d) ;
    if(m_dartLevel[d] > m_curLevel)
        d = TOPO_MAP::end() ;
}

//template <unsigned int ORBIT, typename FUNC>
//void ImplicitHierarchicalMap2::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f) const
//{
//	switch(ORBIT)
//	{
//		case DART:		f(c); break;
//		case VERTEX: 	foreach_dart_of_vertex(c, f); break;
//		case EDGE: 		foreach_dart_of_edge(c, f); break;
//		case FACE: 		foreach_dart_of_face(c, f); break;
//		case VOLUME: 	foreach_dart_of_volume(c, f); break;
//		case VERTEX1: 	foreach_dart_of_vertex1(c, f); break;
//		case EDGE1: 	foreach_dart_of_edge1(c, f); break;
//		default: 		assert(!"Cells of this dimension are not handled"); break;
//	}
//}

template <unsigned int ORBIT, typename FUNC>
void ImplicitHierarchicalMap2::foreach_dart_of_orbit(Cell<ORBIT> c, const FUNC& f) const
{
    switch(ORBIT)
    {
    case DART:		f(c); break;
    case VERTEX: 	foreach_dart_of_vertex(c, f); break;
    case EDGE: 		foreach_dart_of_edge(c, f); break;
    case FACE: 		foreach_dart_of_face(c, f); break;
    case VOLUME: 	foreach_dart_of_volume(c, f); break;
    case VERTEX1: 	foreach_dart_of_vertex1(c, f); break;
    case EDGE1: 	foreach_dart_of_edge1(c, f); break;
    default: 		assert(!"Cells of this dimension are not handled"); break;
    }
}

template <typename FUNC>
inline void ImplicitHierarchicalMap2::foreach_dart_of_vertex(Dart d, const FUNC& f) const
{
    Dart dNext = d;
    do
    {
        f(dNext);
        dNext = alpha1(dNext);
    } while (dNext != d);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap2::foreach_dart_of_edge(Dart d, const FUNC& f) const
{
    f(d);
    Dart d2 = phi2(d);
    if (d2 != d)
        f(d2);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap2::foreach_dart_of_oriented_face(Dart d, const FUNC& f) const
{
    Dart dNext = d ;
    do
    {
        f(dNext);
        dNext = phi1(dNext) ;
    } while (dNext != d) ;
}

template <typename FUNC>
inline void ImplicitHierarchicalMap2::foreach_dart_of_face(Dart d, const FUNC& f) const
{
    foreach_dart_of_oriented_face(d, f) ;
}

template <typename FUNC>
inline void ImplicitHierarchicalMap2::foreach_dart_of_oriented_volume(Dart d, const FUNC& f) const
{
    DartMarkerStore<Map2> mark(*this);	// Lock a marker

    std::list<Dart> visitedFaces;	// Faces that are traversed
    visitedFaces.push_back(d);		// Start with the face of d
    std::list<Dart>::iterator face;

    // For every face added to the list
    for (face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
    {
        if (!mark.isMarked(*face))		// Face has not been visited yet
        {
            // Apply functor to the darts of the face
            foreach_dart_of_oriented_face(*face, f);

            // mark visited darts (current face)
            // and add non visited adjacent faces to the list of face
            Dart dNext = *face ;
            do
            {
                mark.mark(dNext);					// Mark
                Dart adj = phi2(dNext);				// Get adjacent face
                if (adj != dNext && !mark.isMarked(adj))
                    visitedFaces.push_back(adj);	// Add it
                dNext = phi1(dNext);
            } while(dNext != *face);
        }
    }
}

template <typename FUNC>
inline void ImplicitHierarchicalMap2::foreach_dart_of_volume(Dart d, const FUNC& f) const
{
    foreach_dart_of_oriented_volume(d, f) ;
}

template <typename FUNC>
inline void ImplicitHierarchicalMap2::foreach_dart_of_vertex1(Dart d, const FUNC& f) const
{
    f(d);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap2::foreach_dart_of_edge1(Dart d, const FUNC& f) const
{
    f(d);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap2::foreach_dart_of_cc(Dart d, const FUNC& f) const
{
    foreach_dart_of_oriented_volume(d, f) ;
}

/***************************************************
 *               MAP MANIPULATION                  *
 ***************************************************/

inline void ImplicitHierarchicalMap2::splitFace(Dart d, Dart e)
{
    EmbeddedMap2::splitFace(d, e) ;
    if(isOrbitEmbedded<FACE>())
    {
        unsigned int cur = m_curLevel ;
        m_curLevel = m_maxLevel ;
        Algo::Topo::setOrbitEmbedding<FACE>(*this, d, this->getEmbedding<FACE>(d)) ;
        Algo::Topo::setOrbitEmbedding<FACE>(*this, e, this->getEmbedding<FACE>(e)) ;
        m_curLevel = cur ;
    }
}

/***************************************************
 *              LEVELS MANAGEMENT                  *
 ***************************************************/

inline unsigned int ImplicitHierarchicalMap2::getCurrentLevel()
{
    return m_curLevel ;
}

inline void ImplicitHierarchicalMap2::setCurrentLevel(unsigned int l)
{
    m_curLevel = l ;
}

inline void ImplicitHierarchicalMap2::incCurrentLevel()
{
    if(m_curLevel < m_maxLevel)
        ++m_curLevel ;
    else
        CGoGNout << "incCurrentLevel : already at maximum resolution level" << CGoGNendl ;
}

inline void ImplicitHierarchicalMap2::decCurrentLevel()
{
    if(m_curLevel > 0)
        --m_curLevel ;
    else
        CGoGNout << "decCurrentLevel : already at minimum resolution level" << CGoGNendl ;
}

inline unsigned int ImplicitHierarchicalMap2::getMaxLevel()
{
    return m_maxLevel ;
}

inline unsigned int ImplicitHierarchicalMap2::getDartLevel(Dart d)
{
    return m_dartLevel[d] ;
}

inline void ImplicitHierarchicalMap2::setDartLevel(Dart d, unsigned int l)
{
    m_dartLevel[d] = l ;
}

inline void ImplicitHierarchicalMap2::setMaxLevel(unsigned int l)
{
    m_maxLevel = l;
}


/***************************************************
 *             EDGE ID MANAGEMENT                  *
 ***************************************************/

inline unsigned int ImplicitHierarchicalMap2::getNewEdgeId()
{
    return m_idCount++ ;
}

inline unsigned int ImplicitHierarchicalMap2::getEdgeId(Dart d)
{
    return m_edgeId[d] ;
}

inline void ImplicitHierarchicalMap2::setEdgeId(Dart d, unsigned int i)
{
    m_edgeId[d] = i ;
}

inline unsigned int ImplicitHierarchicalMap2::getTriRefinementEdgeId(Dart d)
{
    unsigned int dId = getEdgeId(phi_1(d));
    unsigned int eId = getEdgeId(phi1(d));

    unsigned int id = dId + eId;

    if(id == 0)
        return 1;
    else if(id == 1)
        return 2;
    else if(id == 2)
    {
        if(dId == eId)
            return 0;
        else
            return 1;
    }

    //else if(id == 3)
    return 0;
}

inline unsigned int ImplicitHierarchicalMap2::getQuadRefinementEdgeId(Dart d)
{
    unsigned int eId = getEdgeId(phi1(d));

    if(eId == 0)
        return 1;

    //else if(eId == 1)
    return 0;
}

/***************************************************
 *               CELLS INFORMATION                 *
 ***************************************************/

inline unsigned int ImplicitHierarchicalMap2::faceDegree(Dart d)
{
    unsigned int count = 0 ;
    Dart it = d ;
    do
    {
        ++count ;
        it = phi1(it) ;
    } while (it != d) ;
    return count ;
}



inline unsigned int ImplicitHierarchicalMap2::vertexInsertionLevel(Dart d)
{
//    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
    return m_dartLevel[d] ;
}

//inline unsigned int ImplicitHierarchicalMap2::edgeLevel(Dart d)
//{
//	assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
//	unsigned int ld = m_dartLevel[d] ;
////	unsigned int ldd = m_dartLevel[phi2(d)] ;	// the level of an edge is the maximum of the
//	unsigned int ldd = m_dartLevel[phi1(d)] ;
//	return ld < ldd ? ldd : ld ;				// insertion levels of its two darts
//}








// NEW

Dart IHM2::phi_1(Dart d) const
{
    assert((*m_dartLevel)[d.index] <= m_curLevel || !"Access to a dart introduced after current level") ;
    bool finished = false ;
    Dart it = TOPO_MAP::phi_1(d) ;
    unsigned int edgeId = m_edgeId[it] ;
    do
    {
        if((*m_dartLevel)[it.index] <= m_curLevel)
            finished = true ;
        else
        {
            it = TOPO_MAP::phi_1(it) ;
            while(m_edgeId[it] != edgeId)
                it = TOPO_MAP::phi_1(TOPO_MAP::phi2(it)) ;
        }
    } while(!finished) ;
    return it ;
}

Dart IHM2::phi2(Dart d) const
{
    assert((*m_dartLevel)[d.index] <= m_curLevel || !"Access to a dart introduced after current level") ;
    if(TOPO_MAP::phi2(d) == d)
        return d ;
    return TOPO_MAP::phi2(TOPO_MAP::phi_1(phi1(d))) ;
}

Dart IHM2::alpha0(Dart d) const
{
    return phi2(d);
}

Dart IHM2::alpha1(Dart d) const
{
     return TOPO_MAP::alpha1(d) ;
}

Dart IHM2::alpha_1(Dart d) const
{
    return TOPO_MAP::alpha_1(d) ;
}

Dart IHM2::phi1(Dart d) const
{
    assert((*m_dartLevel)[d.index] <= m_curLevel || !"Access to a dart introduced after current level") ;
    bool finished = false ;
    unsigned int edgeId = m_edgeId[d] ;
    Dart it = d ;
    do
    {
        it = TOPO_MAP::phi1(it) ;
        if((*m_dartLevel)[it.index] <= m_curLevel)
            finished = true ;
        else
        {
            while(m_edgeId[it] != edgeId)
                it = TOPO_MAP::phi1(TOPO_MAP::phi2(it)) ;
        }
    } while(!finished) ;
    return it ;
}












} //namespace CGoGN
