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

#ifndef __TRAVERSOR_CELL_H__
#define __TRAVERSOR_CELL_H__

#include "Topology/generic/dart.h"
#include "Topology/generic/cells.h"
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversor/traversorGen.h"

#include <functional>

namespace CGoGN
{
/**
 * Travsersor class
 * template params:
 *  - MAP type of map
 *  - ORBIT orbit of the cell
 *  - OPT type of optimization
 */

enum TraversalOptim {AUTO=0, FORCE_DART_MARKING, FORCE_CELL_MARKING, FORCE_QUICK_TRAVERSAL};

template <typename MAP, unsigned int ORBIT, TraversalOptim OPT = AUTO>
class TraversorCell
{
protected:
    const MAP& m ;
    unsigned int dimension ;

    const AttributeContainer* cont ;
    unsigned int qCurrent ;

    DartMarker<MAP>* dmark ;
    CellMarker<MAP, ORBIT>* cmark ;
    const AttributeMultiVector<Dart>* quickTraversal ;

    Cell<ORBIT> current ;
    bool firstTraversal ;

    // just for odd/even versions


public:
    TraversorCell(const MAP& map, bool forceDartMarker = false, unsigned int thread = 0) ;
    TraversorCell(const TraversorCell<MAP, ORBIT, OPT>& tc);
    ~TraversorCell() ;

    inline Cell<ORBIT> begin() ;
    inline Cell<ORBIT> end() ;
    inline Cell<ORBIT> next() ;

    inline void skip(Cell<ORBIT> c);
} ;

// qt version

template <typename MAP, unsigned int ORBIT, TraversalOptim OPT>
class TraversorCellIterable {
    BOOST_STATIC_ASSERT((sizeof(MAP) == 0)); // always false
};

template<typename MAP,unsigned int ORBIT>
class TraversorCellIterable<MAP,ORBIT, FORCE_QUICK_TRAVERSAL> {
public:
    class Iterator {
        Iterator(const TraversorCellIterable& tr, unsigned int qCurr);
        Iterator(const Iterator& it);
        Iterator& operator++() ;

        inline bool operator!=(const Iterator& it) const { return current != it.current; }
        inline bool operator==(const Iterator& it) const { return current == it.current; }
        // Warning : does not return a reference but a value.
        inline Cell<ORBIT> operator*() const {return current;}
        inline const Cell<ORBIT>* operator->() const {return &current;}

    private:
        Iterator();
        Iterator& operator=(const Iterator& it);
        // Never use it++, instead you'll need to use ++it.
        Iterator& operator++(int) ;
    private:
        Cell<ORBIT> current;
        unsigned int qCurrent;
        const TraversorCellIterable& m_trav;
    };

public:
    TraversorCellIterable(const MAP& map, bool, unsigned int);
    TraversorCellIterable(const MAP& map);
    TraversorCellIterable(const TraversorCellIterable& );
    ~TraversorCellIterable();
    inline Iterator begin() const {return Iterator(*this, cont->realBegin()); }
    inline Iterator end() const { return Iterator(*this, this->cont->realEnd()); }
private:
    const AttributeMultiVector<Dart>* quickTraversal ;
    const AttributeContainer* cont ;
};



// cell marking version

template<typename MAP,unsigned int ORBIT>
class TraversorCellIterable<MAP,ORBIT, FORCE_CELL_MARKING> {
public:
    class Iterator {
        Iterator(const TraversorCellIterable& tr, Cell<ORBIT> curr) ;
        Iterator(const Iterator& it) ;
        Iterator& operator++() ;
        ~Iterator();

        inline bool operator!=(const Iterator& it) const { return current != it.current; }
        inline bool operator==(const Iterator& it) const { return current == it.current; }
        // Warning : does not return a reference but a value.
        inline Cell<ORBIT> operator*() const { return current; }
        inline const Cell<ORBIT>* operator->() const { return &current; }

    private:
        Iterator();
        Iterator& operator=(const Iterator& it);
        // Never use it++, instead you'll need to use ++it.
        Iterator& operator++(int) ;
    private:
        CellMarker<MAP, ORBIT>* cmark ;
        Cell<ORBIT> current;
        const TraversorCellIterable& m_trav;
    };

public:
    TraversorCellIterable(const MAP& map, bool forceDartmarking = false, unsigned int thread = 0);
    TraversorCellIterable(const TraversorCellIterable& );
    ~TraversorCellIterable();
    inline Iterator begin() { return Iterator(*this,m_begin); }
    inline Iterator end()   { return Iterator(*this,NIL); }

private:
    Cell<ORBIT> m_begin;
    const MAP& m_map;
    const unsigned m_thread;
};


template <typename MAP, unsigned int ORBIT, TraversalOptim OPT = AUTO>
class TraversorCellEven : public TraversorCell<MAP, ORBIT, OPT>
{
public:
    TraversorCellEven(const TraversorCell<MAP,ORBIT, OPT>& tra):
        TraversorCell<MAP, ORBIT, OPT>(tra) {}
    ~TraversorCellEven() { this->cmark = NULL; this->dmark = NULL; }
    inline Cell<ORBIT> begin() ;
} ;


template <typename MAP, unsigned int ORBIT, TraversalOptim OPT = AUTO>
class TraversorCellOdd : public TraversorCell<MAP, ORBIT, OPT>
{
public:
    TraversorCellOdd(const TraversorCell<MAP, ORBIT, OPT>& tra):
        TraversorCell<MAP, ORBIT, OPT>(tra) {}
    ~TraversorCellOdd() {this->cmark = NULL; this->dmark = NULL; }
    inline Cell<ORBIT> begin() ;
    inline Cell<ORBIT> next() ;
} ;



//template <typename MAP, unsigned int ORBIT, const class FUNC& F, TraversalOptim OPT = AUTO>
//class TraversorCellConditional : public TraversorCell<MAP, ORBIT, OPT> {
//public:
//    TraversorCellConditional(const MAP& map, bool forceDartMarker = false, unsigned int thread = 0)  : TraversorCell<MAP, ORBIT, OPT>(map, forceDartMarker, thread) {}
//    TraversorCellConditional(const TraversorCellConditional& tcc) : TraversorCell<MAP, ORBIT, OPT>(tcc) {}
//    inline Cell<ORBIT> begin() ;
//    inline Cell<ORBIT> next() ;
//} ;

/*
 * Executes function f on each ORBIT
 */
template <unsigned int ORBIT, typename MAP, typename FUNC>
inline void foreach_cell(const MAP& map, FUNC f, TraversalOptim opt = AUTO, unsigned int thread = 0);


/*
 * Executes function f on each ORBIT until f returns false
 */
template <unsigned int ORBIT, typename MAP, typename FUNC>
inline void foreach_cell_until(const MAP& map, FUNC f, TraversalOptim opt = AUTO, unsigned int thread = 0);


namespace Parallel
{

/**
 * @brief foreach_cell
 * @param map
 * @param func function to apply on cells
 * @param needMarkers func need markers ?
 * @param opt optimization param of traversal
 * @param nbth number of used thread (0:for traversal, [1,nbth-1] for func computing
*/
template <unsigned int ORBIT, typename MAP, typename FUNC>
void foreach_cell(MAP& map, FUNC func, TraversalOptim opt = AUTO, unsigned int nbth = NumberOfThreads);

} // namespace Parallel





template <typename MAP, TraversalOptim OPT = AUTO>
class TraversorV : public TraversorCell<MAP, VERTEX, OPT>
{
public:
    TraversorV(const MAP& m, unsigned int thread = 0) : TraversorCell<MAP, VERTEX>(m, false, thread)
    {}
};

template <typename MAP, TraversalOptim OPT = AUTO>
class TraversorE : public TraversorCell<MAP, EDGE, OPT>
{
public:
    TraversorE(const MAP& m, unsigned int thread = 0) : TraversorCell<MAP, EDGE>(m, false, thread)
    {}
};

template <typename MAP, TraversalOptim OPT = AUTO>
class TraversorF : public TraversorCell<MAP, FACE, OPT>
{
public:
    TraversorF(const MAP& m, unsigned int thread = 0) : TraversorCell<MAP, FACE>(m, false, thread)
    {}
};

template <typename MAP, TraversalOptim OPT = AUTO>
class TraversorW : public TraversorCell<MAP, VOLUME, OPT>
{
public:
    TraversorW(const MAP& m, unsigned int thread = 0) : TraversorCell<MAP, VOLUME>(m, false, thread)
    {}
};



} // namespace CGoGN

#include "Topology/generic/traversor/traversorCell.hpp"

#endif
