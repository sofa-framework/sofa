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
#include "Utils/static_assert.h"
#include "Container/attributeMultiVector.h"
#include "Container/fakeAttribute.h"

namespace CGoGN
{

//**********************
// Marker for traversor
//**********************

template <typename MAP, unsigned int ORBIT>
MarkerForTraversor<MAP, ORBIT>::MarkerForTraversor(const MAP& map, bool forceDartMarker, unsigned int thread) :
    m_map(map),
    m_dmark(NULL),
    m_cmark(NULL)
{
    if(!forceDartMarker && map.isOrbitEmbedded(ORBIT))
        m_cmark = new CellMarkerStore<MAP, ORBIT>(map, thread) ;
    else
        m_dmark = new DartMarkerStore<MAP>(map, thread) ;
}

template <typename MAP, unsigned int ORBIT>
MarkerForTraversor<MAP, ORBIT>::~MarkerForTraversor()
{
    if (m_cmark)
        delete m_cmark;
    if (m_dmark)
        delete m_dmark;
}

template <typename MAP, unsigned int ORBIT>
void MarkerForTraversor<MAP, ORBIT>::mark(Cell<ORBIT> c)
{
    if (m_cmark)
        m_cmark->mark(c);
    else
        m_dmark->markOrbit(c);
}

template <typename MAP, unsigned int ORBIT>
void MarkerForTraversor<MAP, ORBIT>::unmark(Cell<ORBIT> c)
{
    if (m_cmark)
        m_cmark->unmark(c);
    else
        m_dmark->unmarkOrbit(c);
}

template <typename MAP, unsigned int ORBIT>
bool MarkerForTraversor<MAP, ORBIT>::isMarked(Cell<ORBIT> c)
{
    if (m_cmark)
        return m_cmark->isMarked(c);
    return m_dmark->isMarked(c);
}

template <typename MAP, unsigned int ORBIT>
CellMarkerStore<MAP, ORBIT>* MarkerForTraversor<MAP, ORBIT>::cmark()
{
    return m_cmark;
}

template <typename MAP, unsigned int ORBIT>
DartMarkerStore<MAP>* MarkerForTraversor<MAP, ORBIT>::dmark()
{
    return m_dmark;
}

//**************************************
// Traversor cellX Y incident to cell X
//**************************************

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Traversor3XY<MAP, ORBX, ORBY>::Traversor3XY(const MAP& map, Cell<ORBX> c, bool forceDartMarker, unsigned int thread) :
    m_map(map),
    m_dmark(NULL),
    m_cmark(NULL),
    m_tradoo(map, c, thread),
    m_QLT(NULL),
    m_allocated(true),
    m_first(true)
{
    const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<ORBX,ORBY>() ;
    if (quickTraversal != NULL)
    {
        m_QLT = &(quickTraversal->operator[](map.getEmbedding(c)));
    }
    else
    {
        if(!forceDartMarker && map.isOrbitEmbedded(ORBY)) {
            m_cmark = new CellMarkerStore<MAP, ORBY>(map, thread) ;
        }
        else
            m_dmark = new DartMarkerStore<MAP>(map, thread) ;
    }
}

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Traversor3XY<MAP, ORBX, ORBY>::Traversor3XY(const MAP& map, Cell<ORBX> c, MarkerForTraversor<MAP, ORBY>& tmo, bool /*forceDartMarker*/, unsigned int thread) :
    m_map(map),
    m_tradoo(map, c, thread),
    m_QLT(NULL),
    m_allocated(false),
    m_first(true)
{
    m_cmark = tmo.cmark();
    m_dmark = tmo.dmark();
}

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Traversor3XY<MAP, ORBX, ORBY>::Traversor3XY(const Traversor3XY& tra3xy)
    : m_map(tra3xy.m_map)
    ,m_dmark(NULL)
    ,m_cmark(NULL)
    ,m_tradoo(tra3xy.m_tradoo)
    ,m_QLT(tra3xy.m_QLT)
    ,m_allocated(tra3xy.m_allocated)
    ,m_first(tra3xy.m_first)
{
    std::cerr << "Traversor3XY copy constructor (should not happen)...ORBX =" << ORBX << " & ORBY = " << ORBY << std::endl;
    if (m_QLT == NULL) {
        if(tra3xy.m_cmark != NULL)
            m_cmark = new CellMarkerStore<MAP, ORBY>(m_map, tra3xy.m_cmark->getThread()) ;
        else
            m_dmark = new DartMarkerStore<MAP>(m_map, tra3xy.m_dmark->getThread()) ;
    }

}



template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Traversor3XY<MAP, ORBX, ORBY>::~Traversor3XY()
{
    if (m_allocated)
    {
        if (m_cmark)
            delete m_cmark;
        if (m_dmark)
            delete m_dmark;
    }
}

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Cell<ORBY> Traversor3XY<MAP, ORBX, ORBY>::begin()
{
    if(m_QLT != NULL)
    {
        m_ItDarts = m_QLT->begin();
        return *m_ItDarts++;
    }

    if (!m_first)
    {
        if (m_cmark)
            m_cmark->unmarkAll();
        else
            m_dmark->unmarkAll();
    }
    m_first = false;

    m_current = m_tradoo.begin() ;
    // for the case of beginning with a given MarkerForTraversor
    if (!m_allocated)
    {
        if (m_cmark)
        {
            while ((m_current != NIL) && m_cmark->isMarked(m_current))
                m_current = m_tradoo.next();
        }
        else
        {
            while ((m_current != NIL) && m_dmark->isMarked(m_current))
                m_current = m_tradoo.next();
        }
    }

    if ((ORBY == VOLUME) && (m_current != NIL))
    {
        if(m_map.template isBoundaryMarked<3>(m_current))
            m_current = next();
    }

    return m_current;
}

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Cell<ORBY> Traversor3XY<MAP, ORBX, ORBY>::end()
{
    return NIL ;
}

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Cell<ORBY> Traversor3XY<MAP, ORBX, ORBY>::next()
{
    if(m_QLT != NULL)
    {
        return *m_ItDarts++;
    }

    //    if(m_current != NIL)
    //	{
    if (m_cmark)
    {
        m_cmark->mark(m_current);
        m_current = m_tradoo.next();
        if(ORBY == VOLUME && m_current != NIL)
        {
            if(m_map.template isBoundaryMarked<3>(m_current))
                m_cmark->mark(m_current);
        }
        while ((m_current != NIL) && m_cmark->isMarked(m_current))
            m_current = m_tradoo.next();
    }
    else
    {
        if (ORBX == VOLUME)
        {
            // if allocated we are in a local traversal of volume so we can mark only darts of volume
            if (m_allocated)
                m_dmark->template markOrbit<ORBY + MAP::IN_PARENT>(Dart::create(m_current.index()));
            else
                m_dmark->markOrbit(m_current); // here we need to mark all the darts
        }
        else
            m_dmark->markOrbit(m_current);
        m_current = m_tradoo.next();
        if(ORBY == VOLUME)
        {
            if(m_map.template isBoundaryMarked<3>(m_current))
            {
                if (ORBX == VOLUME)
                {
                    // if allocated we are in a local traversal of volume so we can mark only darts of volume
                    if (m_allocated)
                        m_dmark->template markOrbit<ORBY + MAP::IN_PARENT>(Dart::create(m_current.index()));
                    else
                        m_dmark->markOrbit(m_current); // here we need to mark all the darts
                }
                else
                    m_dmark->markOrbit(m_current);
            }
        }
        while ((m_current != NIL) && m_dmark->isMarked(m_current))
            m_current = m_tradoo.next();
    }
    //	}
    return m_current ;
}

// ARRAY VERSION
//template <typename MAP, unsigned int ORBX, unsigned int ORBY>
//Traversor3XYArray<MAP, ORBX, ORBY>::Traversor3XYArray(const MAP& map, Cell<ORBX> c, bool , unsigned int thread ) :
//    m_thread(thread),
//    m_cells(GenericMap::askDartBuffer(thread))
//{
//    const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<ORBX,ORBY>() ;
//    if (quickTraversal != NULL) {
//        const std::vector<Dart>& m_QLT = quickTraversal->operator[](map.getEmbedding(c));
//        *m_cells = m_QLT;
//    } else {
//        CellMarkerStore<MAP, ORBY> m_cmark(map, thread) ;
//        TraversorDartsOfOrbit<MAP, ORBX> m_tradoo(map, c, thread);

//        for (Dart curr = m_tradoo.begin() ; !curr.isNil() ; curr = m_tradoo.next()) {
//            if (!m_cmark.isMarked(curr)) {
//                m_cmark.mark(curr);
//                if ( (ORBY != VOLUME) || (!map.template isBoundaryMarked<3>(curr)))
//                    m_cells->push_back(curr);
//            }
//        }
//    }
//}

//template <typename MAP, unsigned int ORBX, unsigned int ORBY>
//Traversor3XYArray<MAP, ORBX, ORBY>::Traversor3XYArray(const Traversor3XYArray&)
//{
//    std::cerr << "CGoGN : Copy constructor of Traversor3XYArray<MAP, ORBX, ORBY> called. Should not happen. Aborting." << std::endl;
//    std::exit(1);
//}

//*********************************************
// Traversor cellX to cellX adjacent by cell Y
//*********************************************

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Traversor3XXaY<MAP, ORBX, ORBY>::Traversor3XXaY(const MAP& map, Cell<ORBX> c, bool forceDartMarker, unsigned int thread):
    m_map(map),
    m_QLT(NULL)
{
    const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal =  map.template getQuickAdjacentTraversal<ORBX,ORBY>() ;
    if (quickTraversal != NULL)
    {
        m_QLT  = &(quickTraversal->operator[](map.getEmbedding(c)));
    }
    else
    {
        MarkerForTraversor<MAP, ORBX> mk(map, forceDartMarker, thread);
        mk.mark(c);

        Traversor3XY<MAP, ORBX, ORBY> traAdj(map, c, forceDartMarker, thread);
        for (Dart d = traAdj.begin(); d != traAdj.end(); d = traAdj.next())
        {
            Traversor3XY<MAP, ORBY, ORBX> traInci(map, d, mk, forceDartMarker, thread);
            for (Dart e = traInci.begin(); e != traInci.end(); e = traInci.next())
                m_vecDarts.push_back(e);
        }
        m_vecDarts.push_back(NIL);
    }
}

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Cell<ORBX> Traversor3XXaY<MAP, ORBX, ORBY>::begin()
{
    if(m_QLT != NULL)
    {
        m_ItDarts = m_QLT->begin();
        return *m_ItDarts++;
    }

    m_iter = m_vecDarts.begin();
    return *m_iter;
}

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Cell<ORBX> Traversor3XXaY<MAP, ORBX, ORBY>::end()
{
    return NIL;
}

template <typename MAP, unsigned int ORBX, unsigned int ORBY>
Cell<ORBX> Traversor3XXaY<MAP, ORBX, ORBY>::next()
{
    if(m_QLT != NULL)
    {
        return *m_ItDarts++;
    }

    if (*m_iter != NIL)
        m_iter++;
    return *m_iter ;
}


// ARRAY VERSION
//template <typename MAP, unsigned int ORBX, unsigned int ORBY>
//Traversor3XXaYArray<MAP, ORBX, ORBY>::Traversor3XXaYArray(const MAP& map, Cell<ORBX> c, bool , unsigned int thread ) :
//    m_thread(thread),
//    m_cells(GenericMap::askDartBuffer(thread))
//{
//    const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal =  map.template getQuickAdjacentTraversal<ORBX,ORBY>() ;
//    if (quickTraversal != NULL)
//    {
//        const std::vector<Dart>& m_QLT  = quickTraversal->operator[](map.getEmbedding(c));
//        *m_cells = m_QLT;
//    }
//    else
//    {
//        MarkerForTraversor<MAP, ORBX> mk(map, false, thread);
//        mk.mark(c);

//        Traversor3XY<MAP, ORBX, ORBY> traAdj(map, c, false, thread);
//        for (Dart d = traAdj.begin(), endAdj = traAdj.end(); d != endAdj ; d = traAdj.next())
//        {
//            Traversor3XY<MAP, ORBY, ORBX> traInci(map, d, mk, false, thread);
//            for (Dart e = traInci.begin(), endInci = traInci.end() ; e != endInci; e = traInci.next())
//                m_cells->push_back(e);
//        }
//    }
//}

//template <typename MAP, unsigned int ORBX, unsigned int ORBY>
//Traversor3XXaYArray<MAP, ORBX, ORBY>::Traversor3XXaYArray(const Traversor3XXaYArray&)
//{
//    std::cerr << "CGoGN : Copy constructor of Traversor3XYArray<MAP, ORBX, ORBY> called. Should not happen. Aborting." << std::endl;
//    std::exit(1);
//}


//template<typename MAP>
//Traversor3<MAP>* Traversor3<MAP>::createXY(MAP& map, Dart dart, unsigned int orbX, unsigned int orbY)
//{
//	int code = 0x10*(orbX-VERTEX) + orbY-VERTEX;
//
//	switch(code)
//	{
//	case 0x01:
//		return new Traversor3XY<MAP, VERTEX, EDGE>(map,dart);
//		break;
//	case 0x02:
//		return new Traversor3XY<MAP, VERTEX, FACE>(map,dart);
//		break;
//	case 0x03:
//		return new Traversor3XY<MAP, VERTEX, VOLUME>(map,dart);
//		break;
//
//	case 0x10:
//		return new Traversor3XY<MAP, EDGE, VERTEX>(map,dart);
//		break;
//	case 0x12:
//		return new Traversor3XY<MAP, EDGE, FACE>(map,dart);
//		break;
//	case 0x13:
//		return new Traversor3XY<MAP, EDGE, VOLUME>(map,dart);
//		break;
//
//	case 0x20:
//		return new Traversor3XY<MAP, FACE, VERTEX>(map,dart);
//		break;
//	case 0x21:
//		return new Traversor3XY<MAP, FACE, EDGE>(map,dart);
//		break;
//	case 0x23:
//		return new Traversor3XY<MAP, FACE, VOLUME>(map,dart);
//		break;
//
//	case 0x30:
//		return new Traversor3XY<MAP, VOLUME, VERTEX>(map,dart);
//		break;
//	case 0x31:
//		return new Traversor3XY<MAP, VOLUME, EDGE>(map,dart);
//		break;
//	case 0x32:
//		return new Traversor3XY<MAP, VOLUME, FACE>(map,dart);
//		break;
//
//	default:
//		return NULL;
//		break;
//	}
//	return NULL;
//}
//
//
//template<typename MAP>
//Traversor3<MAP>* Traversor3<MAP>::createXXaY(MAP& map, Dart dart, unsigned int orbX, unsigned int orbY)
//{
//	int code = 0x10*(orbX-VERTEX) + orbY-VERTEX;
//
//	switch(code)
//	{
//	case 0x01:
//		return new Traversor3XXaY<MAP, VERTEX, EDGE>(map,dart);
//		break;
//	case 0x02:
//		return new Traversor3XXaY<MAP, VERTEX, FACE>(map,dart);
//		break;
//	case 0x03:
//		return new Traversor3XXaY<MAP, VERTEX, VOLUME>(map,dart);
//		break;
//
//	case 0x10:
//		return new Traversor3XXaY<MAP, EDGE, VERTEX>(map,dart);
//		break;
//	case 0x12:
//		return new Traversor3XXaY<MAP, EDGE, FACE>(map,dart);
//		break;
//	case 0x13:
//		return new Traversor3XXaY<MAP, EDGE, VOLUME>(map,dart);
//		break;
//
//	case 0x20:
//		return new Traversor3XXaY<MAP, FACE, VERTEX>(map,dart);
//		break;
//	case 0x21:
//		return new Traversor3XXaY<MAP, FACE, EDGE>(map,dart);
//		break;
//	case 0x23:
//		return new Traversor3XXaY<MAP, FACE, VOLUME>(map,dart);
//		break;
//
//	case 0x30:
//		return new Traversor3XXaY<MAP, VOLUME, VERTEX>(map,dart);
//		break;
//	case 0x31:
//		return new Traversor3XXaY<MAP, VOLUME, EDGE>(map,dart);
//		break;
//	case 0x32:
//		return new Traversor3XXaY<MAP, VOLUME, FACE>(map,dart);
//		break;
//
//	default:
//		return NULL;
//		break;
//	}
//	return NULL;
//}

} // namespace CGoGN
