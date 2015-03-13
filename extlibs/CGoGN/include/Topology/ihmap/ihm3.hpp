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

/***************************************************
 *             ATTRIBUTES MANAGEMENT               *
 ***************************************************/
//template <typename T, unsigned int ORBIT, typename MAP>
//AttributeHandler_IHM<T, ORBIT, MAP> ImplicitHierarchicalMap3::addAttribute(const std::string& nameAttr)
//{
//	bool addNextLevelCell = false ;
//	if(!isOrbitEmbedded<ORBIT>())
//		addNextLevelCell = true ;

//	AttributeHandler<T, ORBIT, ImplicitHierarchicalMap3> h = Map3::addAttribute<T, ORBIT, ImplicitHierarchicalMap3>(nameAttr) ;

//	if(addNextLevelCell)
//	{
//		AttributeContainer& cellCont = m_attribs[ORBIT] ;
//		AttributeMultiVector<unsigned int>* amv = cellCont.addAttribute<unsigned int>("nextLevelCell") ;
//		m_nextLevelCell[ORBIT] = amv ;
//		for(unsigned int i = cellCont.begin(); i < cellCont.end(); cellCont.next(i))
//			amv->operator[](i) = EMBNULL ;
//	}

//	return AttributeHandler_IHM<T, ORBIT>(this, h.getDataVector()) ;
//}

//template <typename T, unsigned int ORBIT, typename MAP>
//AttributeHandler_IHM<T, ORBIT, MAP> ImplicitHierarchicalMap3::getAttribute(const std::string& nameAttr)
//{
//	AttributeHandler<T, ORBIT, ImplicitHierarchicalMap3> h = Map3::getAttribute<T, ORBIT, ImplicitHierarchicalMap3>(nameAttr) ;
//	return AttributeHandler_IHM<T, ORBIT>(this, h.getDataVector()) ;
//}

inline void ImplicitHierarchicalMap3::update_topo_shortcuts()
{
//	TOPO_MAP::update_topo_shortcuts();
    m_dartLevel = TOPO_MAP::getAttribute<unsigned int, DART, ImplicitHierarchicalMap3>("dartLevel") ;
    m_faceId = TOPO_MAP::getAttribute<unsigned int, DART, ImplicitHierarchicalMap3>("faceId") ;
    m_edgeId = TOPO_MAP::getAttribute<unsigned int, DART, ImplicitHierarchicalMap3>("edgeId") ;

    //AttributeContainer& cont = m_attribs[DART] ;
    //m_nextLevelCell = cont.getDataVector<unsigned int>(cont.getAttributeIndex("nextLevelCell")) ;
}

/***************************************************
 *                 MAP TRAVERSAL                   *
 ***************************************************/
inline Dart ImplicitHierarchicalMap3::newDart()
{
	Dart d = TOPO_MAP::newDart() ;
    m_dartLevel[d] = m_curLevel ;
    if(m_curLevel > m_maxLevel)			// update max level
        m_maxLevel = m_curLevel ;		// if needed
    return d ;
}

inline Dart ImplicitHierarchicalMap3::phi1(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
    bool finished = false ;

    unsigned int edgeId = m_edgeId[d] ;
    Dart it = d ;

    do
    {

        it = Map3::phi1(it) ;
        if(m_dartLevel[it] <= m_curLevel)
            finished = true ;
        else
        {
            while(m_edgeId[it] != edgeId)
                it = Map3::phi1(phi2bis(it));

        }
    } while(!finished) ;

    return it ;
}

inline Dart ImplicitHierarchicalMap3::phi_1(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
    bool finished = false ;

    Dart it = Map3::phi_1(d) ;
    unsigned int edgeId = m_edgeId[it] ;
    do
    {
        if(m_dartLevel[it] <= m_curLevel)
            finished = true ;
        else
        {
            it = Map3::phi_1(it) ;
            while(m_edgeId[it] != edgeId)
                it = Map3::phi_1(phi2bis(it));
        }
    } while(!finished) ;

    return it ;
}

inline Dart ImplicitHierarchicalMap3::phi2bis(Dart d) const
{
    unsigned int faceId = m_faceId[d];
    Dart it = d;

    it = Map3::phi2(it) ;

    /* du cote des volumes non subdivise (subdiv adapt) */
    if(m_faceId[it] == faceId)
        return it;
    else
    {
        do
        {
            it = Map3::phi2(Map3::phi3(it));
        }
        while(m_faceId[it] != faceId);

        return it;
    }
}

inline Dart ImplicitHierarchicalMap3::phi2(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

    return Map3::phi2(Map3::phi_1(phi1(d))) ;
}

inline Dart ImplicitHierarchicalMap3::phi3(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

    if(Map3::phi3(d) == d)
        return d;

    return Map3::phi3(Map3::phi_1(phi1(d)));
}

inline Dart ImplicitHierarchicalMap3::alpha0(Dart d) const
{
    return phi3(d) ;
}

inline Dart ImplicitHierarchicalMap3::alpha1(Dart d) const
{
    //assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

    return phi3(phi_1(d)) ;
}

inline Dart ImplicitHierarchicalMap3::alpha2(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

    return phi3(phi2(d));
}

inline Dart ImplicitHierarchicalMap3::alpha_2(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;

    return phi2(phi3(d));
}

inline Dart ImplicitHierarchicalMap3::begin() const
{
    Dart d = Map3::begin() ;
    while(m_dartLevel[d] > m_curLevel)
        Map3::next(d) ;
    return d ;
}

inline Dart ImplicitHierarchicalMap3::end() const
{
    return Map3::end() ;
}

inline void ImplicitHierarchicalMap3::next(Dart& d) const
{
    do
    {
        Map3::next(d) ;
    } while(d != Map3::end() && m_dartLevel[d] > m_curLevel) ;
}

//template <unsigned int ORBIT, typename FUNC>
//void ImplicitHierarchicalMap3::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f) const
//{
//    switch(ORBIT)
//    {
//        case DART:		f(c); break;
//		case VERTEX: 	foreach_dart_of_vertex(c, f); break;
//		case EDGE: 		foreach_dart_of_edge(c, f); break;
//		case FACE: 		foreach_dart_of_face(c, f); break;
//		case VOLUME: 	foreach_dart_of_volume(c, f); break;
//		case VERTEX1: 	foreach_dart_of_vertex1(c, f); break;
//		case EDGE1: 	foreach_dart_of_edge1(c, f); break;
//		case VERTEX2: 	foreach_dart_of_vertex2(c, f); break;
//		case EDGE2:		foreach_dart_of_edge2(c, f); break;
//		case FACE2:		foreach_dart_of_face2(c, f); break;
//        default: 		assert(!"Cells of this dimension are not handled"); break;
//    }
//}

template <unsigned int ORBIT, typename FUNC>
void ImplicitHierarchicalMap3::foreach_dart_of_orbit(Cell<ORBIT> c, const FUNC& f) const
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
		case VERTEX2: 	foreach_dart_of_vertex2(c, f); break;
		case EDGE2:		foreach_dart_of_edge2(c, f); break;
		case FACE2:		foreach_dart_of_face2(c, f); break;
        default: 		assert(!"Cells of this dimension are not handled"); break;
    }
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_vertex(Dart d, const FUNC& f) const
{
	DartMarkerStore<Map3> mv(*this);	// Lock a marker

    std::vector<Dart> darts;	// Darts that are traversed
    darts.reserve(256);
    darts.push_back(d);			// Start with the dart d
    mv.mark(d);

    for(unsigned int i = 0; i < darts.size(); ++i)
    {
        // add phi21 and phi23 successor if they are not marked yet
        Dart d2 = phi2(darts[i]);
        Dart d21 = phi1(d2); // turn in volume
        Dart d23 = phi3(d2); // change volume

        if(!mv.isMarked(d21))
        {
            darts.push_back(d21);
            mv.mark(d21);
        }
        if(!mv.isMarked(d23))
        {
            darts.push_back(d23);
            mv.mark(d23);
        }

        f(darts[i]);
    }
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_edge(Dart d, const FUNC& f) const
{
    Dart dNext = d;
    do {
		foreach_dart_of_edge2(dNext, f);
        dNext = alpha2(dNext);
    } while (dNext != d);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_oriented_face(Dart d, const FUNC& f) const
{
    Dart dNext = d ;
    do
    {
        f(dNext);
        dNext = phi1(dNext) ;
    } while (dNext != d) ;
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_face(Dart d, const FUNC& f) const
{
	foreach_dart_of_oriented_face(d, f);
	foreach_dart_of_oriented_face(phi3(d), f);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_oriented_volume(Dart d, const FUNC& f) const
{
	DartMarkerStore<Map3> mark(*this);	// Lock a marker

    std::vector<Dart> visitedFaces;	// Faces that are traversed
    visitedFaces.reserve(1024) ;
    visitedFaces.push_back(d);		// Start with the face of d

    // For every face added to the list
    for(unsigned int i = 0; i < visitedFaces.size(); ++i)
    {
        if (!mark.isMarked(visitedFaces[i]))	// Face has not been visited yet
        {
            // Apply functor to the darts of the face
            foreach_dart_of_oriented_face(visitedFaces[i], f);

            // mark visited darts (current face)
            // and add non visited adjacent faces to the list of face
            Dart e = visitedFaces[i] ;
            do
            {
                mark.mark(e);				// Mark
                Dart adj = phi2(e);			// Get adjacent face
                if (!mark.isMarked(adj))
                    visitedFaces.push_back(adj);	// Add it
                e = phi1(e);
            } while(e != visitedFaces[i]);
        }
    }
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_volume(Dart d, const FUNC& f) const
{
	foreach_dart_of_oriented_volume(d, f) ;
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_vertex1(Dart d, const FUNC& f) const
{
    f(d);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_edge1(Dart d, const FUNC& f) const
{
    f(d);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_vertex2(Dart d, const FUNC& f) const
{
    Dart dNext = d;
    do
    {
        f(dNext);
        dNext = phi2(phi_1(dNext));
    } while (dNext != d);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_edge2(Dart d, const FUNC& f) const
{
    f(d);
    f(phi2(d));
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_face2(Dart d, const FUNC& f) const
{
	foreach_dart_of_oriented_face(d, f);
}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_cc(Dart d, const FUNC& f) const
{
	DartMarkerStore<Map3> mark(*this);	// Lock a marker

    std::vector<Dart> visitedFaces;	// Faces that are traversed
    visitedFaces.reserve(1024) ;
    visitedFaces.push_back(d);		// Start with the face of d

    // For every face added to the list
    for(unsigned int i = 0; i < visitedFaces.size(); ++i)
    {
        if (!mark.isMarked(visitedFaces[i]))	// Face has not been visited yet
        {
            // Apply functor to the darts of the face
            foreach_dart_of_face(visitedFaces[i], f);

            // mark visited darts (current face)
            // and add non visited adjacent faces to the list of face
            Dart e = visitedFaces[i] ;
            do
            {
                mark.mark(e);				// Mark
                Dart adj = phi2(e);			// Get adjacent face
                if (!mark.isMarked(adj))
                    visitedFaces.push_back(adj);	// Add it
                e = phi1(e);
            } while(e != visitedFaces[i]);
        }
    }

//	// foreach_dart_of_oriented_volume(d, f) ;
//	DartMarkerStore mv(*this);	// Lock a marker
//
//	std::vector<Dart> darts;	// Darts that are traversed
//	darts.reserve(1024);
//	darts.push_back(d);			// Start with the dart d
//	mv.mark(d);
//
//	for(unsigned int i = 0; i < darts.size(); ++i)
//	{
//		// add all successors if they are not marked yet
//		Dart d2 = phi1(darts[i]); // turn in face
//		Dart d3 = phi2(darts[i]); // change face
//		Dart d4 = phi3(darts[i]); // change volume
//
//		if (!mv.isMarked(d2))
//		{
//			darts.push_back(d2);
//			mv.mark(d2);
//		}
//		if (!mv.isMarked(d3))
//		{
//			darts.push_back(d2);
//			mv.mark(d2);
//		}
//		if (!mv.isMarked(d4))
//		{
//			darts.push_back(d4);
//			mv.mark(d4);
//		}
//
//		f(darts[i]);
//	}
}


/***************************************************
 *              LEVELS MANAGEMENT                  *
 ***************************************************/

inline void ImplicitHierarchicalMap3::incCurrentLevel()
{
	assert(m_curLevel < m_maxLevel || "incCurrentLevel : already at maximum resolution level");
	++m_curLevel ;
}

inline void ImplicitHierarchicalMap3::decCurrentLevel()
{
	assert(m_curLevel > 0 || "decCurrentLevel : already at minimum resolution level");
	--m_curLevel ;
}

inline unsigned int ImplicitHierarchicalMap3::getCurrentLevel() const
{
    return m_curLevel ;
}

inline void ImplicitHierarchicalMap3::setCurrentLevel(unsigned int l)
{
    m_curLevel = l ;
}

inline unsigned int ImplicitHierarchicalMap3::getMaxLevel() const
{
    return m_maxLevel ;
}

inline unsigned int ImplicitHierarchicalMap3::getDartLevel(Dart d) const
{
    return m_dartLevel[d] ;
}

inline void ImplicitHierarchicalMap3::setDartLevel(Dart d, unsigned int l)
{
    m_dartLevel[d] = l ;
}

/***************************************************
 *             EDGE ID MANAGEMENT                  *
 ***************************************************/

inline unsigned int ImplicitHierarchicalMap3::getNewEdgeId()
{
    return m_edgeIdCount++ ;
}


inline unsigned int ImplicitHierarchicalMap3::getEdgeId(Dart d)
{
    return m_edgeId[d] ;
}

inline void ImplicitHierarchicalMap3::setEdgeId(Dart d, unsigned int i)
{
	Dart e = d;

	do
	{
		m_edgeId[e] = i;
		m_edgeId[Map3::phi2(e)] = i;

		e = Map3::alpha2(e);
	} while(e != d);

}

inline void ImplicitHierarchicalMap3::setDartEdgeId(Dart d, unsigned int i)
{
	m_edgeId[d] = i;
}

inline unsigned int ImplicitHierarchicalMap3::triRefinementEdgeId(Dart d)
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

inline unsigned int ImplicitHierarchicalMap3::quadRefinementEdgeId(Dart d)
{
	unsigned int eId = getEdgeId(phi1(d));

	if(eId == 0)
		return 1;

	//else if(eId == 1)
	return 0;
}

/***************************************************
 *             FACE ID MANAGEMENT                  *
 ***************************************************/

inline unsigned int ImplicitHierarchicalMap3::getNewFaceId()
{
    return m_faceIdCount++;
}

inline unsigned int ImplicitHierarchicalMap3::faceId(Dart d)
{
	unsigned int fId = getFaceId(phi2(d));

	if(fId == 0)
		return 1;
	else if(fId == 1)
		return 2;
	else if(fId == 2)
	{
//		if(dId == eId)
			return 0;
//		else
//			return 1;
	}

	//else if(id == 3)
	return 0;
}

inline unsigned int ImplicitHierarchicalMap3::getFaceId(Dart d)
{
    return m_faceId[d] ;
}

inline void ImplicitHierarchicalMap3::setFaceId(unsigned int orbit, Dart d)
{
    //Mise a jour de l'id de face pour les brins autour d'une arete
    if(orbit == EDGE)
    {
        Dart e = d;

        do
        {
            m_faceId[Map3::phi1(e)] = m_faceId[e];

            e = Map3::alpha2(e);
        }while(e != d);

    }
}

inline void ImplicitHierarchicalMap3::setFaceId(Dart d, unsigned int i, unsigned int orbit)
{

    //Mise a jour de l'id de face pour les brins autour de la face
    if(orbit == FACE)
    {
        Dart e = d;

        do
        {
            m_faceId[e] = i;

            Dart e3 = Map3::phi3(e);
            if(e3 != e)
                m_faceId[e3] = i;

            e = Map3::phi1(e);
        } while(e != d);
    }
    else if(orbit == DART)
    {
        m_faceId[d] = i;

        if(Map3::phi3(d) != d)
            m_faceId[Map3::phi3(d)] = i;
    }
}

/***************************************************
 *               CELLS INFORMATION                 *
 ***************************************************/

inline unsigned int ImplicitHierarchicalMap3::vertexInsertionLevel(Dart d) const
{
    assert(m_dartLevel[d] <= m_curLevel || !"Access to a dart introduced after current level") ;
    return m_dartLevel[d] ;
}

/***************************************************
 *               ATTRIBUTE HANDLER                 *
 ***************************************************/

//template <typename T, unsigned int ORBIT>
//T& AttributeHandler_IHM<T, ORBIT>::operator[](Dart d)
//{
//    ImplicitHierarchicalMap3* m = reinterpret_cast<ImplicitHierarchicalMap3*>(this->m_map) ;
//    assert(m->m_dartLevel[d] <= m->m_curLevel || !"Access to a dart introduced after current level") ;
//    assert(m->vertexInsertionLevel(d) <= m->m_curLevel || !"Access to the embedding of a vertex inserted after current level") ;

//    unsigned int orbit = this->getOrbit() ;
//    unsigned int nbSteps = m->m_curLevel - m->vertexInsertionLevel(d) ;
//    unsigned int index = m->getEmbedding<ORBIT>(d) ;

//    if(index == EMBNULL)
//    {
//        index = Algo::Topo::setOrbitEmbeddingOnNewCell<ORBIT>(*m, d) ;
//        m->m_nextLevelCell[orbit]->operator[](index) = EMBNULL ;
//    }

//    AttributeContainer& cont = m->getAttributeContainer<ORBIT>() ;
//    unsigned int step = 0 ;
//    while(step < nbSteps)
//    {
//        step++ ;
//        unsigned int nextIdx = m->m_nextLevelCell[orbit]->operator[](index) ;
//        if (nextIdx == EMBNULL)
//        {
//            nextIdx = m->newCell<ORBIT>() ;
//            m->copyCell<ORBIT>(nextIdx, index) ;
//            m->m_nextLevelCell[orbit]->operator[](index) = nextIdx ;
//            m->m_nextLevelCell[orbit]->operator[](nextIdx) = EMBNULL ;
//            cont.refLine(index) ;
//        }
//        index = nextIdx ;
//    }

//    return this->m_attrib->operator[](index);
//}

//template <typename T, unsigned int ORBIT>
//const T& AttributeHandler_IHM<T, ORBIT>::operator[](Dart d) const
//{
//    ImplicitHierarchicalMap3* m = reinterpret_cast<ImplicitHierarchicalMap3*>(this->m_map) ;
//    assert(m->m_dartLevel[d] <= m->m_curLevel || !"Access to a dart introduced after current level") ;
//    assert(m->vertexInsertionLevel(d) <= m->m_curLevel || !"Access to the embedding of a vertex inserted after current level") ;

//    unsigned int orbit = this->getOrbit() ;
//    unsigned int nbSteps = m->m_curLevel - m->vertexInsertionLevel(d) ;
//    unsigned int index = m->getEmbedding<ORBIT>(d) ;

//    unsigned int step = 0 ;
//    while(step < nbSteps)
//    {
//        step++ ;
//        unsigned int nextIdx = m->m_nextLevelCell[ORBIT]->operator[](index) ;
//        if(nextIdx != EMBNULL) index = nextIdx ;
//        else break ;
//    }
//    return this->m_attrib->operator[](index);
//}

} //namespace CGoGN
