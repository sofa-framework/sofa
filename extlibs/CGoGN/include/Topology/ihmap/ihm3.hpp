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
#include "ihm3.h"
#include <algorithm>
namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace IHM
{

template <int N>
inline Dart ImplicitHierarchicalMap3::phi(Dart d) const{
//    assert( (N >0) || !"negative parameters not allowed in template multi-phi");
    if (N<10)
    {
        switch(N)
        {
            case 1 : return this->phi1(d) ;
            case 2 : return this->phi2(d) ;
            case 3 : return phi3(d) ;
            default : assert(!"Wrong multi-phi relation value") ; return d ;
        }
    }
    switch(N%10)
    {
        case 1 : return this->phi1(phi<N/10>(d)) ;
        case 2 : return this->phi2(phi<N/10>(d)) ;
        case 3 : return phi3(phi<N/10>(d)) ;
        default : assert(!"Wrong multi-phi relation value") ; return d ;
    }
//    return Parent::phi<N>(d);
}

inline Dart ImplicitHierarchicalMap3::phi1(Dart d) const
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
    if (getCurrentLevel() == getMaxLevel())
    {
        return phi1MaxLvl(d);
    }

    bool finished = false ;
    unsigned int edgeId = getEdgeId(d) ;
    Dart it = d ;

    do
    {
        it = phi1MaxLvl(it) ;
        if(getDartLevel(it) <= getCurrentLevel())
        {
            finished = true ;
        } else
        {
            while(getEdgeId(it) != edgeId)
            {
                it = this->phi1MaxLvl(phi2bis(it));
            }
        }
    } while(!finished) ;

    return it ;
}

inline Dart ImplicitHierarchicalMap3::phi_1(Dart d) const
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;

    if (getCurrentLevel() == getMaxLevel())
    {
        return phi_1MaxLvl(d);
    }

    bool finished = false ;

    Dart it = this->phi_1MaxLvl(d) ;
    const unsigned int edgeId = getEdgeId(it) ;
    do
    {
        if(getDartLevel(it) <= getCurrentLevel())
        {
            finished = true ;
        } else
        {
            it = this->phi_1MaxLvl(it) ;
            while(getEdgeId(it) != edgeId)
            {
                it = this->phi_1MaxLvl(phi2bis(it));
            }
        }
    } while(!finished) ;
//    std::cerr << "ImplicitHierarchicalMap3::phi_1(" << d << ") = " << it << std::endl;
    return it ;
//    return phi_1MaxLvl(d);
}

inline Dart ImplicitHierarchicalMap3::phi2bis(Dart d) const
{
    const unsigned int faceId = getFaceId(d);
    Dart it = d;

    it = this->phi2MaxLvl(it) ;

    /* du cote des volumes non subdivise (subdiv adapt) */
    if(getFaceId(it) == faceId)
        return it;
    else
    {
        do
        {
            it = this->phi2MaxLvl(this->phi3MaxLvl(it));
        } while(getFaceId(it) != faceId);

        return it;
    }
}

inline Dart ImplicitHierarchicalMap3::phi2(Dart d) const
{
    return (getCurrentLevel() == getMaxLevel())? phi2MaxLvl(d) : (phi2MaxLvl(this->phi_1MaxLvl(phi1(d))));
}

inline Dart ImplicitHierarchicalMap3::phi3(Dart d) const
{
//    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
    return (getCurrentLevel() == getMaxLevel())? phi3MaxLvl(d) : (phi3MaxLvl(this->phi_1MaxLvl(phi1(d))));
//    return (phi3MaxLvl(d) == d) ? d : (phi3MaxLvl(this->phi_1MaxLvl(phi1(d))));
}

inline Dart ImplicitHierarchicalMap3::alpha0(Dart d) const
{
    return phi3(d) ;
}

inline Dart ImplicitHierarchicalMap3::alpha1(Dart d) const
{
    return phi3(phi_1(d)) ;
}

inline Dart ImplicitHierarchicalMap3::alpha2(Dart d) const
{
    return phi3(phi2(d));
}

inline Dart ImplicitHierarchicalMap3::alpha_2(Dart d) const
{
    return phi2(phi3(d));
}


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

//template <unsigned int ORBIT, typename FUNC>
//void ImplicitHierarchicalMap3::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC& f) const
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
//		case VERTEX2: 	foreach_dart_of_vertex2(c, f); break;
//		case EDGE2:		foreach_dart_of_edge2(c, f); break;
//		case FACE2:		foreach_dart_of_face2(c, f); break;
//		default: 		assert(!"Cells of this dimension are not handled"); break;
//	}
//}

template <typename FUNC>
inline void ImplicitHierarchicalMap3::foreach_dart_of_vertex(Dart d, const FUNC& f) const
{
    DartMarkerStore< MAP > mv(*this);	// Lock a marker

    std::vector<Dart> darts;	// Darts that are traversed
    darts.reserve(64);
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
    DartMarkerStore< MAP > mark(*this);	// Lock a marker

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
    DartMarkerStore< MAP > mark(*this);	// Lock a marker

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




/***************************************************
 *             EDGE ID MANAGEMENT                  *
 ***************************************************/




inline void ImplicitHierarchicalMap3::setEdgeId(Dart d, unsigned int i, unsigned int orbit)
{
	if(orbit == EDGE)
	{
		Dart e = d;

		do
		{
            Parent::setEdgeId(e, i);
            Parent::setEdgeId(phi2MaxLvl(e), i);
            e = this->alpha2MaxLvl(e);
		} while(e != d);
	}
	else if(orbit == DART)
	{
        Parent::setEdgeId(d, i);
	}
}

/***************************************************
 *             FACE ID MANAGEMENT                  *
 ***************************************************/


inline void ImplicitHierarchicalMap3::setFaceId(unsigned int orbit, Dart d)
{
    //Mise a jour de l'id de face pour les brins autour d'une arete
    if(orbit == EDGE)
    {
        Dart e = d;

        do
        {
            Parent::setFaceId(phi1MaxLvl(e), getFaceId(e));
            e = alpha2MaxLvl(e);
        } while(e != d);
    } else
    {
        std::exit(1);
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
            Parent::setFaceId(e, i);
            const Dart e3 = this->phi3MaxLvl(e);
			if(e3 != e)
            {
                Parent::setFaceId(e3, i);
            }
            e = this->phi1MaxLvl(e);
		} while(e != d);
	}
	else if(orbit == DART)
	{
        Parent::setFaceId(d, i);
        const Dart e3 = this->phi3MaxLvl(d);
        if(e3 != d)
        {
            Parent::setFaceId(e3, i);
        }
	}
}

/***************************************************
 *               CELLS INFORMATION                 *
 ***************************************************/

//TODO
inline unsigned int ImplicitHierarchicalMap3::vertexInsertionLevel(Dart d) const
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
    return getDartLevel(d) ;
}

inline unsigned int ImplicitHierarchicalMap3::edgeLevel(Dart d) const
{
    assert(getDartLevel(d) <= m_curLevel || !"Access to a dart introduced after current level") ;
    return std::max(getDartLevel(d), getDartLevel(phi2(d)));
}

template< unsigned int ORBIT>
inline unsigned int ImplicitHierarchicalMap3::getEmbedding(Cell< ORBIT > c) const
{
//    std::cerr << __FILE__ << ":" << __LINE__ << ": " << __func__ <<std::endl;
//    std::cerr << "ORBIT = " << ORBIT << std::endl;
    if (ORBIT == DART)
    {
        return this->dartIndex(c.dart);
    }

    if (ORBIT == VERTEX)
    {
//        std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
//        const_cast<MAP*>(this)->compactOrbitContainer(VERTEX);
//        const unsigned int nbSteps = getCurrentLevel() - vertexInsertionLevel(c.dart);
//        std::cerr << "nbsteps = " << nbSteps;
//        unsigned int index = Parent::getEmbedding(c);
//        std::cerr << " emb = " << index << std::endl;
//        std::cerr << "Parent::getEmbedding(" << c <<  ") =  " << index << std::endl;
//        if (index == EMBNULL)
//        {
//            return EMBNULL;
//        }

//        if(index == EMBNULL)
//        {
//            index = Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(*const_cast<MAP*>(this), c.dart) ;
//            const_cast<MAP*>(this)->m_nextLevelCell->operator[](index) = EMBNULL ;
//        }


//        AttributeContainer& cont = const_cast<MAP*>(this)->getAttributeContainer<VERTEX>() ;
//        unsigned int step = 0u;
//        while(step < nbSteps)
//        {
//            const unsigned int next = m_nextLevelCell->operator[](index);
//            if (next != EMBNULL)
//            {
//                index = next;
//            } else
//            {
////                assert(false);
//                break;
//            }
//            ++step;
////            unsigned int nextIdx = this->m_nextLevelCell->operator[](index) ;
////            if (nextIdx == EMBNULL)
////            {
////                nextIdx = const_cast<MAP*>(this)->newCell<VERTEX>() ;
////                const_cast<MAP*>(this)->copyCell<VERTEX>(nextIdx, index) ;
////                const_cast<MAP*>(this)->m_nextLevelCell->operator[](index) = nextIdx ;
//////                std::cerr << "m_nextLevelCell[" << index << "] = " << nextIdx << std::endl;
////                const_cast<MAP*>(this)->m_nextLevelCell->operator[](nextIdx) = EMBNULL ;
////                cont.refLine(index) ;
////            }
////            index = nextIdx ;
//        }
        return Parent::getEmbedding(c);
    }

    if (ORBIT == EDGE)
    {
        return m_embeddings[EDGE]->operator [](this->dartIndex(this->edgeNewestDart(c)));
    }

    if (ORBIT == FACE)
    {
        return m_embeddings[FACE]->operator [](this->dartIndex(this->faceNewestDart(FaceCell(c.dart))));
    }
    if (ORBIT == VOLUME)
    {
//        const Dart vnd = this->volumeNewestDart(c);
//        TraversorDartsOfOrbit< MAP, VOLUME > traDoo(*this, VolumeCell(c.dart));
//        std::cerr << "volume orbit of " << c << std::endl;
//        for(Dart dit = traDoo.begin(); dit != traDoo.end() ; dit = traDoo.next())
//        {
//            std::cerr << dit << " (lvl " << getDartLevel(dit) /*<< ") " */;
//            std::cerr << " ; emb " << ParentMap::getEmbedding<VOLUME>(dit) << ") ";
//        }
//        std::cerr << std::endl;
//        const unsigned vndLvl = getDartLevel(vnd);
//        const unsigned curr = getCurrentLevel();
//        assert(vndLvl == curr);
//        std::cerr << "volumeNewestDart(" << c << ") = " << vnd << " ( dartlvl(" << c << ") = " << this->getDartLevel(c) << " )." << std::endl;
//        std::cerr << "getCurrentLevel() : " << getCurrentLevel() <<  " dartlvl(vnd) = " << getDartLevel(vnd) << std::endl;
        return m_embeddings[VOLUME]->operator [](this->dartIndex(this->volumeNewestDart(VolumeCell(c.dart))));
    }

    return Parent::getEmbedding(c);
}


inline bool ImplicitHierarchicalMap3::isWellEmbedded()
{
	TraversorV<ImplicitHierarchicalMap3> tv(*this);

	for(Vertex dv = tv.begin() ; dv.dart != tv.end() ; dv = tv.next())
	{
		unsigned int curem = this->getEmbedding(dv);
		//std::cout << "current emb = " << curem << std::endl;

		unsigned int count = 0;
		TraversorDartsOfOrbit<ImplicitHierarchicalMap3, VERTEX> to(*this, dv);
		for(Dart dit = to.begin() ; dit != to.end() ; dit = to.next())
		{
			//std::cout << getDartLevel(dit) << std::endl;

			if(curem != this->getEmbedding<VERTEX>(dit))
			{
				std::cout << "erreur dart #" << dit;
				std::cout << " / curem = " << curem;
				std::cout << " / em = " << this->getEmbedding<VERTEX>(dit);
				std::cout << std::endl;
			}

            std::cout << this->getEmbedding<VERTEX>(dit) << "(" << this->getEmbedding<VERTEX>(dit) << ")" << " / ";
			++count;
		}
		std::cout << " / vertex degree = " << count << std::endl;

	}

	return true;
}



template<unsigned int ORBIT>
ImplicitHierarchicalMap3::OrbitAttributeBrowser<ORBIT>::OrbitAttributeBrowser(ImplicitHierarchicalMap3 *ihm3) :
    ContainerBrowser()
     ,m_enabled(false)
    ,m_ihm3(ihm3)

{
    assert(ihm3 != NULL);
    m_orbitQT = &(m_ihm3->m_quickTraversal[ORBIT]);
}

template<unsigned int ORBIT>
unsigned int ImplicitHierarchicalMap3::OrbitAttributeBrowser<ORBIT>::begin() const
{
    unsigned int begin = m_ihm3->m_attribs[ORBIT].realBegin();
    if (!m_enabled)
    {
        return begin;
    }
    AttributeMultiVector<Dart> * const orbQT = *m_orbitQT;
    assert(orbQT != NULL);
    const unsigned int end = this->end();
    const unsigned int currLVL = m_ihm3->getCurrentLevel();
    unsigned int cellLevel = m_ihm3->template getCellLevel<ORBIT>(begin);
    while(( cellLevel != m_ihm3->template getMaxCellLevel<ORBIT>(orbQT->operator [](begin)) && ORBIT != VERTEX ) || (cellLevel > currLVL) )
    {
        m_ihm3->m_attribs[ORBIT].realNext(begin);
        if (begin == end)
        {
            return end;
        }
        cellLevel = m_ihm3->template getCellLevel<ORBIT>(begin);
    }
    return begin;
}

template<unsigned int ORBIT>
unsigned int ImplicitHierarchicalMap3::OrbitAttributeBrowser<ORBIT>::end() const
{
    return m_ihm3->m_attribs[ORBIT].realEnd();
}
template<unsigned int ORBIT>
void ImplicitHierarchicalMap3::OrbitAttributeBrowser<ORBIT>::enable()
{
    m_enabled = true;
}

template<unsigned int ORBIT>
void ImplicitHierarchicalMap3::OrbitAttributeBrowser<ORBIT>::next(unsigned int &it) const
{
    m_ihm3->m_attribs[ORBIT].realNext(it);
    const unsigned int end = this->end();
    if (!m_enabled || it == end)
    {
        return;
    }
    AttributeMultiVector<Dart> * const orbQT = *m_orbitQT;
    assert(orbQT != NULL);
    const unsigned int currLVL = m_ihm3->getCurrentLevel();
    unsigned int cellLevel = m_ihm3->template getCellLevel<ORBIT>(it);
    while ( (cellLevel != m_ihm3->template getMaxCellLevel<ORBIT>(orbQT->operator [](it)) && ORBIT !=VERTEX)  || (cellLevel > currLVL) )
    {
        m_ihm3->m_attribs[ORBIT].realNext(it);
        if (it == end)
        {
            return;
        }
        cellLevel = m_ihm3->template getCellLevel<ORBIT>(it);
    }
}

template<unsigned int ORBIT>
void ImplicitHierarchicalMap3::OrbitAttributeBrowser<ORBIT>::disable()
{
    m_enabled = false;
}

template<unsigned int ORBIT>
ImplicitHierarchicalMap3::OrbitAttributeBrowser<ORBIT>::~OrbitAttributeBrowser()
{

}


} //namespace IHM
} // Volume
} //namespace Algo
} //namespace CGoGN
