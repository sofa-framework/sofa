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
#ifndef TRAVERSORCELLITERABLE_H
#define TRAVERSORCELLITERABLE_H

#include "Topology/generic/traversor/traversorCell.h"

namespace CGoGN
{

// qt version
template <typename MAP, unsigned int ORBIT, TraversalOptim OPT>
class TraversorCellIterable {
    BOOST_STATIC_ASSERT((sizeof(MAP) == 0)); // always false
};

template<typename MAP,unsigned int ORBIT>
class TraversorCellIterable<MAP,ORBIT, FORCE_QUICK_TRAVERSAL> {
public:
    class Iterator {
    public:
        Iterator(const TraversorCellIterable& tr, unsigned int qCurr);
        Iterator(const Iterator& it);
        Iterator& operator=(const Iterator& it);
        Iterator& operator++() ;

        inline bool operator!=(const Iterator& it) const { return current != it.current; }
        inline bool operator==(const Iterator& it) const { return current == it.current; }
        // Warning : does not return a reference but a value.
        inline Cell<ORBIT> operator*() const {return current;}
        inline const Cell<ORBIT>* operator->() const {return &current;}
    private:
        Iterator();
        // Never use it++, instead you'll need to use ++it.
        Iterator& operator++(int) ;
    private:
        Cell<ORBIT> current;
        unsigned int qCurrent;
        const TraversorCellIterable& m_trav;
    };

public:
    typedef Iterator iterator;
    TraversorCellIterable(const MAP& map, bool = 0, unsigned int = 0);
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
    public:
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
//        const TraversorCellIterable& m_trav;
    };

public:
    typedef Iterator iterator;
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



// AUTO VERSION (--> qt if possible, otherwise cm)
template<typename MAP,unsigned int ORBIT>
class TraversorCellIterable<MAP,ORBIT, AUTO> {
public:
    class Iterator {
    public:
        Iterator(const TraversorCellIterable& tr, bool beginning = true) ; // true for beginning, false for end
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
        unsigned int qCurrent;
        const TraversorCellIterable& m_trav;
    };

public:
    typedef Iterator iterator;
    TraversorCellIterable(const MAP& map, bool forceDartmarking = false, unsigned int thread = 0);
    TraversorCellIterable(const TraversorCellIterable& );
    ~TraversorCellIterable();
    inline Iterator begin() { return Iterator(*this,m_begin); }
    inline Iterator end()   { return Iterator(*this,NIL); }

private:
    Cell<ORBIT> m_begin;
    const MAP& m_map;
    const AttributeMultiVector<Dart>* quickTraversal ;
    const AttributeContainer* cont ;
    const unsigned m_thread;

};


template <typename MAP, unsigned int ORBIT, TraversalOptim OPT>
class TraversorCellArray {
    BOOST_STATIC_ASSERT(sizeof(MAP) == 0); // always false
};


template<typename MAP,unsigned int ORBIT>
class TraversorCellArray<MAP,ORBIT, FORCE_QUICK_TRAVERSAL> {
public:
    typedef std::vector<Dart>::iterator iterator;
    TraversorCellArray(const MAP& map, bool = 0, unsigned int = 0);
    TraversorCellArray(const TraversorCellArray& );
    ~TraversorCellArray();
    inline iterator begin() const {return m_cells->begin(); }
    inline iterator end() const { return m_cells->end(); }
private:
    const unsigned int m_thread;
    std::vector<Dart>* m_cells;
};

template<typename MAP,unsigned int ORBIT>
class TraversorCellArray<MAP,ORBIT, FORCE_CELL_MARKING> {
public:
    typedef std::vector<Dart>::iterator iterator;
    TraversorCellArray(const MAP& map, bool = 0, unsigned int = 0);
    TraversorCellArray(const TraversorCellArray& );
    ~TraversorCellArray();
    inline iterator begin() const { return m_cells->begin(); }
    inline iterator end() const { return m_cells->end(); }
private:
    const unsigned int m_thread;
    std::vector<Dart>* m_cells;
};

template<typename MAP,unsigned int ORBIT>
class TraversorCellArray<MAP,ORBIT, AUTO> {
public:
    typedef std::vector<Dart>::iterator iterator;
    TraversorCellArray(const MAP& map, bool = 0, unsigned int = 0);
    TraversorCellArray(const TraversorCellArray& );
    ~TraversorCellArray();
    inline iterator begin() const {return m_cells->begin(); }
    inline iterator end() const { return m_cells->end(); }
private:
    const unsigned int m_thread;
    std::vector<Dart>* m_cells;
};


} // namespace cgogn
#include "Topology/generic/traversor/traversorCellIterable.hpp"
#endif // TRAVERSORCELLITERABLE_H
