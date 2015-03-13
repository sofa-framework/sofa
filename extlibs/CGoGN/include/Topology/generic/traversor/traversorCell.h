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



public:
    // just for odd/even versions
    TraversorCell(const TraversorCell<MAP, ORBIT, OPT>& tc);


    TraversorCell(const MAP& map, bool forceDartMarker = false) ;
    ~TraversorCell() ;

    inline Cell<ORBIT> begin() ;
    inline Cell<ORBIT> end() ;
    inline Cell<ORBIT> next() ;

    inline void skip(Cell<ORBIT> c);
} ;




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






/*
 * Executes function f on each ORBIT
 */
template <unsigned int ORBIT, typename MAP, typename FUNC>
inline void foreach_cell(const MAP& map, FUNC f, TraversalOptim opt = AUTO);


/*
 * Executes function f on each ORBIT until f returns false
 */
template <unsigned int ORBIT, typename MAP, typename FUNC>
inline void foreach_cell_until(const MAP& map, FUNC f, TraversalOptim opt = AUTO);


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
	TraversorV(const MAP& m) : TraversorCell<MAP, VERTEX>(m, false)
	{}
};

template <typename MAP, TraversalOptim OPT = AUTO>
class TraversorE : public TraversorCell<MAP, EDGE, OPT>
{
public:
	TraversorE(const MAP& m) : TraversorCell<MAP, EDGE>(m, false)
	{}
};

template <typename MAP, TraversalOptim OPT = AUTO>
class TraversorF : public TraversorCell<MAP, FACE, OPT>
{
public:
	TraversorF(const MAP& m) : TraversorCell<MAP, FACE>(m, false)
	{}
};

template <typename MAP, TraversalOptim OPT = AUTO>
class TraversorW : public TraversorCell<MAP, VOLUME, OPT>
{
public:
	TraversorW(const MAP& m) : TraversorCell<MAP, VOLUME>(m, false)
	{}
};

template <typename MAP, unsigned int ORBIT, TraversalOptim OPT = AUTO>
class allCells: public TraversorCell<MAP,ORBIT,OPT>
{
public:
	allCells(const MAP& map, bool forceDartMarker = false):
		TraversorCell<MAP,ORBIT,OPT>(map,forceDartMarker) {}


	class iterator
	{
		TraversorCell<MAP,ORBIT,OPT>* m_ptr;
		Cell<ORBIT> m_index;

	public:

		inline iterator(allCells<MAP,ORBIT,OPT>* p, Cell<ORBIT> i): m_ptr(p),m_index(i){}

		inline iterator& operator++()
		{
			m_index = m_ptr->next();
			return *this;
		}

		inline Cell<ORBIT>& operator*()
		{
			return m_index;
		}

		inline bool operator!=(iterator it)
		{
			return m_index.dart != it.m_index.dart;
		}

	};

	inline iterator begin()
	{
		return iterator(this,TraversorCell<MAP,ORBIT,OPT>::begin());
	}

	inline iterator end()
	{
		return iterator(this,TraversorCell<MAP,ORBIT,OPT>::end());
	}

};

template <typename MAP>
inline allCells<MAP, VERTEX, AUTO> allVerticesOf(const MAP& m)
{
	return allCells<MAP,VERTEX,AUTO>(m, false);
}

template <typename MAP>
inline allCells<MAP, EDGE, AUTO> allEdgesOf(const MAP& m)
{
	return allCells<MAP,EDGE,AUTO>(m, false);
}

template <typename MAP>
inline allCells<MAP, FACE, AUTO> allFacesOf(const MAP& m)
{
	return allCells<MAP,FACE,AUTO>(m, false);
}

template <typename MAP>
inline allCells<MAP, VOLUME, AUTO> allVolumesOf(const MAP& m)
{
	return allCells<MAP,VOLUME,AUTO>(m, false);
}


template <TraversalOptim OPT, typename MAP>
inline allCells<MAP, VERTEX, OPT> allVerticesOf(const MAP& m)
{
	return allCells<MAP,VERTEX,OPT>(m, false);
}

template <TraversalOptim OPT, typename MAP>
inline allCells<MAP, EDGE, OPT> allEdgesOf(const MAP& m)
{
	return allCells<MAP,EDGE,OPT>(m, false);
}

template <TraversalOptim OPT, typename MAP>
inline allCells<MAP, FACE, OPT> allFacesOf(const MAP& m)
{
	return allCells<MAP,FACE,OPT>(m, false);
}

template <TraversalOptim OPT, typename MAP>
inline allCells<MAP, VOLUME, OPT> allVolumesOf(const MAP& m)
{
	return allCells<MAP,VOLUME,OPT>(m, false);
}


/*
template <typename MAP, TraversalOptim OPT = AUTO>
class allVertices : public allCells<MAP, VERTEX, OPT>
{
public:
	allVertices(const MAP& m) : allCells<MAP, VERTEX>(m, false)
	{}
};


template <typename MAP, TraversalOptim OPT = AUTO>
class allEdges : public allCells<MAP, EDGE, OPT>
{
public:
	allEdges(const MAP& m) : allCells<MAP, EDGE>(m, false)
	{}
};

template <typename MAP, TraversalOptim OPT = AUTO>
class allFaces : public allCells<MAP, FACE, OPT>
{
public:
	allFaces(const MAP& m) : allCells<MAP, FACE>(m, false)
	{}
};

template <typename MAP, TraversalOptim OPT = AUTO>
class allVolumes : public allCells<MAP, VOLUME, OPT>
{
public:
	allVolumes(const MAP& m) : allCells<MAP, VOLUME>(m, false)
	{}
};

*/



} // namespace CGoGN

#include "Topology/generic/traversor/traversorCell.hpp"

#endif
