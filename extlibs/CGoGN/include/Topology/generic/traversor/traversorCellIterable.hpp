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
#ifndef TRAVERSORCELLITERABLE_HPP
#define TRAVERSORCELLITERABLE_HPP

namespace CGoGN
{

template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::TraversorCellIterable(const MAP& map, bool, unsigned int) {
    quickTraversal = map.template getQuickTraversal<ORBIT>() ;
    assert(quickTraversal != NULL);
    cont = &(map.template getAttributeContainer<ORBIT>()) ;

}


template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::TraversorCellIterable(const TraversorCellIterable&) {
    std::cerr << "CGoGN : Copy constructor of TraversorCellIterable called. Should not happen. Aborting." << std::endl;
    std::exit(1);
}


template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::~TraversorCellIterable()
{}

template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::Iterator::Iterator(const TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>& tr, unsigned int qCurr) :
    m_trav(tr),
    qCurrent(qCurr)
{
    if (qCurr == m_trav.cont->realEnd())
        current = NIL;
    else
        current = tr.quickTraversal->operator[](qCurrent) ;
}

template<typename MAP,unsigned int ORBIT>
typename TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::Iterator& TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::Iterator::operator=(const Iterator&)
{
    std::cerr << "CGoGN : TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::Iterator::operator= called. Should not happen. Aborting." << std::endl;
    std::exit(1);
    return *this;
}

template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::Iterator::Iterator(const Iterator& it) :
    m_trav(it.m_trav),
    qCurrent(it.qCurrent),
    current(it.current)
{}

template<typename MAP,unsigned int ORBIT>
typename TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::Iterator& TraversorCellIterable<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::Iterator::operator++()
{
    m_trav.cont->realNext(qCurrent) ;
    if (qCurrent != m_trav.cont->realEnd())
        current = m_trav.quickTraversal->operator[](qCurrent) ;
    else current = NIL;
    return *this;
}
// cell marking version


template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_CELL_MARKING>::TraversorCellIterable(const MAP& map, bool, unsigned int thread) :
    m_thread(thread)
{
    m_begin = map.begin();
    while((m_begin != map.end()) && (map.isBoundaryMarkedCurrent(m_begin)))
        map.next(m_begin) ;
    if(m_begin == map.end())
        m_begin = NIL ;
}

template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_CELL_MARKING>::TraversorCellIterable(const TraversorCellIterable&) {
    std::cerr << "CGoGN : Copy constructor of TraversorCellIterable called. Should not happen. Aborting." << std::endl;
    std::exit(1);
}


template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_CELL_MARKING>::Iterator::Iterator(const TraversorCellIterable& tr, Cell<ORBIT> curr) :
    //    m_trav(tr),
    current(curr),
    cmark(NULL)
{
    if (!current.isNil()) {
        cmark = new CellMarker<MAP, ORBIT>(tr.m_map, tr.m_thread);
        cmark->mark(current) ;
    }
}


template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_CELL_MARKING>::Iterator::Iterator(const Iterator& it) :
    //    m_trav(it.m_trav),
    current(it.current),
    cmark(NULL)
{
    if (!current.isNil()) {
        cmark = new CellMarker<MAP, ORBIT>(*(it.cmark));
        cmark->mark(current) ;
    }
}

template<typename MAP,unsigned int ORBIT>
typename TraversorCellIterable<MAP, ORBIT, FORCE_CELL_MARKING>::Iterator& TraversorCellIterable<MAP, ORBIT, FORCE_CELL_MARKING>::Iterator::operator++()
{
    assert(cmark != NULL);
    const MAP& m = cmark->getMap();
    bool ismarked = cmark->isMarked(current) ;
    while((!current.isNil()) && (ismarked || m.isBoundaryMarkedCurrent(current)))
    {
        m.next(current) ;
        if(current == m.end())
            current = NIL ;
        else
            ismarked = cmark->isMarked(current) ;
    }
    if(current != NIL)
        cmark->mark(current) ;
    return *this;
}

template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, FORCE_CELL_MARKING>::Iterator::~Iterator()
{
    delete cmark;
}


// auto version

template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, AUTO>::TraversorCellIterable(const MAP& map, bool, unsigned int thread) :
    m_begin(NIL),
    m_map(map),
    quickTraversal(map.template getQuickTraversal<ORBIT>()),
    cont(NULL),
    m_thread(thread)
{
    if (quickTraversal != NULL) {
        cont = &(map.template getAttributeContainer<ORBIT>()) ;
        m_begin = quickTraversal->operator[](cont->realBegin());
    } else {
        m_begin = map.begin();
        while((m_begin != map.end()) && (map.isBoundaryMarkedCurrent(m_begin)))
            map.next(m_begin) ;
        if(m_begin == map.end())
            m_begin = NIL ;
    }
}

template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, AUTO>::TraversorCellIterable(const TraversorCellIterable&)
{
    std::cerr << "CGoGN : Copy constructor of TraversorCellIterable<MAP, ORBIT, AUTO>::TraversorCellIterable called. Should not happen. Aborting." << std::endl;
    std::exit(1);
}

template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, AUTO>::~TraversorCellIterable()
{}

template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, AUTO>::Iterator::Iterator(const TraversorCellIterable<MAP, ORBIT, AUTO>& tr, bool beginning) :
    cmark(NULL),
    current(NIL),
    m_trav(tr)
{
    if (!beginning)
        current = NIL;
    else {
        if (m_trav.quickTraversal != NULL) {
            qCurrent = m_trav.cont->realBegin();
            current = m_trav.quickTraversal->operator[](qCurrent) ;
            assert(current = m_trav.m_begin);
        } else {
            cmark = new CellMarker<MAP, ORBIT>(m_trav.m_map, m_trav.m_thread);
            current = m_trav.m_begin;
            if (!current.isNil())
                cmark->mark(current);
        }
    }
}


template<typename MAP,unsigned int ORBIT>
TraversorCellIterable<MAP, ORBIT, AUTO>::Iterator::~Iterator() {
    delete cmark;
}

template<typename MAP,unsigned int ORBIT>
typename TraversorCellIterable<MAP, ORBIT, AUTO>::Iterator& TraversorCellIterable<MAP, ORBIT, AUTO>::Iterator::operator++() {
    if (m_trav.quickTraversal != NULL) {
        m_trav.cont->realNext(qCurrent) ;
        if (qCurrent != m_trav.cont->realEnd())
            current = m_trav.quickTraversal->operator[](qCurrent) ;
        else current = NIL;

    } else {
        assert(cmark != NULL);
        const MAP& m = cmark->getMap();
        bool ismarked = cmark->isMarked(current) ;
        while((!current.isNil()) && (ismarked || m.isBoundaryMarkedCurrent(current)))
        {
            m.next(current) ;
            if(current == m.end())
                current = NIL ;
            else
                ismarked = cmark->isMarked(current) ;
        }
        if(current != NIL)
            cmark->mark(current) ;
    }
    return *this;
}


/**
 * TRAVERSORCELL -- ARRAY VERSION
 * Generate an array of cells during the construction, then return iterators to this array
 */

// Quick Traversal version
template<typename MAP,unsigned int ORBIT>
TraversorCellArray<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::~TraversorCellArray()
{
    GenericMap::releaseDartBuffer(m_cells, m_thread);
}

template<typename MAP,unsigned int ORBIT>
TraversorCellArray<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::TraversorCellArray(const MAP& map, bool , unsigned int thread) :
    m_thread(thread),
    m_cells(GenericMap::askDartBuffer(thread))
{
    const AttributeMultiVector<Dart>* quickTraversal = map.template getQuickTraversal<ORBIT>() ;
    assert(quickTraversal != NULL);
    const AttributeContainer& cont= map.template getAttributeContainer<ORBIT>() ;
    for (unsigned int qCurrent = cont.realBegin(), end = cont.realEnd() ; qCurrent != end ; cont.next(qCurrent))
        m_cells->push_back(quickTraversal->operator[](qCurrent));
}

template<typename MAP,unsigned int ORBIT>
TraversorCellArray<MAP, ORBIT, FORCE_QUICK_TRAVERSAL>::TraversorCellArray(const TraversorCellArray&) :
    m_thread(0)
{
    std::cerr << "CGoGN : Copy constructor of TraversorCellArray<MAP, ORBIT, FORCE_QUICK_TRAVERSAL> called. Should not happen. Aborting." << std::endl;
    std::exit(1);
}


// Cell Marking version
template<typename MAP,unsigned int ORBIT>
TraversorCellArray<MAP, ORBIT, FORCE_CELL_MARKING>::~TraversorCellArray()
{
    GenericMap::releaseDartBuffer(m_cells, m_thread);
}

template<typename MAP,unsigned int ORBIT>
TraversorCellArray<MAP, ORBIT, FORCE_CELL_MARKING>::TraversorCellArray(const MAP& map, bool , unsigned int thread) :
    m_thread(thread),
    m_cells(GenericMap::askDartBuffer(thread))
{
    CellMarker<MAP, ORBIT> cmark(map,thread);
    for (Dart current = map.begin(), end = map.end() ; current != end ; map.next(current)) {
        if ( (!map.isBoundaryMarkedCurrent(current)) && (!cmark.isMarked(current))) {
            cmark.mark(current);
            m_cells->push_back(current);
        }
    }
//    std::cerr << "Iterating on " << m_cells->size() << " " << orbitName<ORBIT>() << std::endl;
}

template<typename MAP,unsigned int ORBIT>
TraversorCellArray<MAP, ORBIT, FORCE_CELL_MARKING>::TraversorCellArray(const TraversorCellArray&) :
    m_thread(0)
{
    std::cerr << "CGoGN : Copy constructor of TraversorCellArray<MAP, ORBIT, FORCE_QUICK_TRAVERSAL> called. Should not happen. Aborting." << std::endl;
    std::exit(1);
}

// AUTO version
template<typename MAP,unsigned int ORBIT>
TraversorCellArray<MAP, ORBIT, AUTO>::~TraversorCellArray()
{
    GenericMap::releaseDartBuffer(m_cells, m_thread);
}

template<typename MAP,unsigned int ORBIT>
TraversorCellArray<MAP, ORBIT, AUTO>::TraversorCellArray(const MAP& map, bool , unsigned int thread) :
    m_thread(thread),
    m_cells(GenericMap::askDartBuffer(thread))
{
    const AttributeMultiVector<Dart>* quickTraversal = map.template getQuickTraversal<ORBIT>() ;
    if (quickTraversal == NULL) {
        CellMarker<MAP, ORBIT> cmark(map,thread);
        for (Dart current = map.begin(), end = map.end() ; current != end ; map.next(current)) {
            if ( (!map.isBoundaryMarkedCurrent(current)) && (!cmark.isMarked(current))) {
                cmark.mark(current);
                m_cells->push_back(current);
            }
        }
    } else {
        const AttributeContainer& cont= map.template getAttributeContainer<ORBIT>() ;
        for (unsigned int qCurrent = cont.realBegin(), end = cont.realEnd() ; qCurrent != end ; cont.next(qCurrent))
            m_cells->push_back(quickTraversal->operator[](qCurrent));
    }
//    std::cerr << "Iterating on " << m_cells->size() << " " << orbitName<ORBIT>() << std::endl;
}

template<typename MAP,unsigned int ORBIT>
TraversorCellArray<MAP, ORBIT, AUTO>::TraversorCellArray(const TraversorCellArray&) :
    m_thread(0)
{
    std::cerr << "CGoGN : Copy constructor of TraversorCellArray<MAP, ORBIT, FORCE_QUICK_TRAVERSAL> called. Should not happen. Aborting." << std::endl;
    std::exit(1);
}



} // namespace cgogn
#endif // TRAVERSORCELLITERABLE_HPP
