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

#ifndef __TRAVERSOR3_H__
#define __TRAVERSOR3_H__

#include "Topology/generic/dart.h"
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversor/traversorCell.h"
//#include "Topology/generic/traversor/traversorGen.h"
#include "Topology/generic/traversor/traversorDoO.h"

namespace CGoGN
{

/**
 * class Marker for Traversor usefull to combine
 * several TraversorXY
 */
template <typename MAP, unsigned int ORBIT>
class MarkerForTraversor
{
private:
	const MAP& m_map ;
    DartMarkerStore<MAP>* m_dmark ;
    CellMarkerStore<MAP, ORBIT>* m_cmark ;

public:
	MarkerForTraversor(const MAP& map, bool forceDartMarker = false, unsigned int thread = 0) ;
	~MarkerForTraversor();

    DartMarkerStore<MAP>* dmark();
    CellMarkerStore<MAP, ORBIT>* cmark();
	void mark(Cell<ORBIT> c);
	void unmark(Cell<ORBIT> c);
	bool isMarked(Cell<ORBIT> c);
} ;

/**
 * Generic class Traversor (do not use directly)
 * Traverse all Y incident to X
 */
template <typename MAP, unsigned int ORBX, unsigned int ORBY>
class Traversor3XY
{
    BOOST_STATIC_ASSERT(MAP::DIMENSION >= 3u) ;
private:
	const MAP& m_map ;
    DartMarkerStore<MAP>* m_dmark ;
    CellMarkerStore<MAP, ORBY>* m_cmark ;
	Cell<ORBY> m_current ;
	TraversorDartsOfOrbit<MAP, ORBX> m_tradoo;

	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;

	bool m_allocated;
	bool m_first;

public:
	Traversor3XY(const MAP& map, Cell<ORBX> c, bool forceDartMarker = false, unsigned int thread = 0) ;
	Traversor3XY(const MAP& map, Cell<ORBX> c, MarkerForTraversor<MAP, ORBY>& tmo, bool forceDartMarker = false, unsigned int thread = 0) ;
    Traversor3XY(const Traversor3XY& tra3xy);
	~Traversor3XY();

	Cell<ORBY> begin() ;
	Cell<ORBY> end() ;
	Cell<ORBY> next() ;
} ;

//template <typename MAP, unsigned int ORBX, unsigned int ORBY>
//class Traversor3XYArray {
//public:
//    typedef std::vector<Dart>::iterator iterator;
//    Traversor3XYArray(const MAP& map,Cell<ORBX> c, bool = false, unsigned int = 0);
//    Traversor3XYArray(const Traversor3XYArray& );
//    ~Traversor3XYArray();
//    inline iterator begin() const { return m_cells->begin(); }
//    inline iterator end() const { return m_cells->end(); }
//private:
//    const unsigned int m_thread;
//    std::vector<Dart>* m_cells;
//};




/**
 * Generic class Traversor (do not use directly)
 * Traverse all X adjacent to X by an Y
 */
template <typename MAP, unsigned int ORBX, unsigned int ORBY>
class Traversor3XXaY
{
private:
	const MAP& m_map ;
	std::vector<Dart> m_vecDarts;
	std::vector<Dart>::iterator m_iter;

	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;

public:
	Traversor3XXaY(const MAP& map, Cell<ORBX> c, bool forceDartMarker = false, unsigned int thread = 0);

	Cell<ORBX> begin();
	Cell<ORBX> end();
	Cell<ORBX> next();
};


//template <typename MAP, unsigned int ORBX, unsigned int ORBY>
//class Traversor3XXaYArray {
//public:
//    typedef std::vector<Dart>::iterator iterator;
//    Traversor3XXaYArray(const MAP& map, Cell<ORBX> c, bool forceDartMarker = false, unsigned int thread = 0);
//    Traversor3XXaYArray(const Traversor3XXaYArray& );
//    ~Traversor3XXaYArray();
//    inline iterator begin() const { return m_cells->begin(); }
//    inline iterator end() const { return m_cells->end(); }
//private:
//    const unsigned int m_thread;
//    std::vector<Dart>* m_cells;
//};

/**
 * Traverse vertices incident to volume
 */
template <typename MAP>
class Traversor3WV: public Traversor3XY<MAP, VOLUME, VERTEX>
{
public:
	Traversor3WV(const MAP& m, Vol dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VOLUME, VERTEX>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse edges incident to volume
 */
template <typename MAP>
class Traversor3WE: public Traversor3XY<MAP, VOLUME, EDGE>
{
public:
	Traversor3WE(const MAP& m, Vol dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VOLUME, EDGE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse faces incident to volume
 */
template <typename MAP>
class Traversor3WF: public Traversor3XY<MAP, VOLUME, FACE>
{
public:
	Traversor3WF(const MAP& m, Vol dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VOLUME, FACE>(m, dart, forceDartMarker, thread) {}
};

/**
 * Traverse vertices incident to face
 */
template <typename MAP>
class Traversor3FV: public Traversor3XY<MAP, FACE, VERTEX>
{
public:
	Traversor3FV(const MAP& m, Face dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, FACE, VERTEX>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse edges incident to face
 */
template <typename MAP>
class Traversor3FE: public Traversor3XY<MAP, FACE, EDGE>
{
public:
	Traversor3FE(const MAP& m, Face dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, FACE, EDGE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes incident to face
 */
template <typename MAP>
class Traversor3FW: public Traversor3XY<MAP, FACE, VOLUME>
{
public:
	Traversor3FW(const MAP& m, Face dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, FACE, VOLUME>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices incident to edge
 */
template <typename MAP>
class Traversor3EV: public Traversor3XY<MAP, EDGE, VERTEX>
{
public:
	Traversor3EV(const MAP& m, Edge dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, EDGE, VERTEX>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse faces incident to edge
 */
template <typename MAP>
class Traversor3EF: public Traversor3XY<MAP, EDGE, FACE>
{
public:
	Traversor3EF(const MAP& m, Edge dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, EDGE, FACE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes incident to edge
 */
template <typename MAP>
class Traversor3EW: public Traversor3XY<MAP, EDGE, VOLUME>
{
public:
	Traversor3EW(const MAP& m, Edge dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, EDGE, VOLUME>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse edges incident to vertex
 */
template <typename MAP>
class Traversor3VE: public Traversor3XY<MAP, VERTEX, EDGE>
{
public:
	Traversor3VE(const MAP& m, Vertex dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VERTEX, EDGE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse faces incident to vertex
 */
template <typename MAP>
class Traversor3VF: public Traversor3XY<MAP, VERTEX, FACE>
{
public:
	Traversor3VF(const MAP& m, Vertex dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VERTEX, FACE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes incident to vertex
 */
template <typename MAP>
class Traversor3VW: public Traversor3XY<MAP, VERTEX, VOLUME>
{
public:
	Traversor3VW(const MAP& m, Vertex dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VERTEX, VOLUME>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices adjacent to a vertex by an edge
 */
template <typename MAP>
class Traversor3VVaE: public Traversor3XXaY<MAP, VERTEX, EDGE>
{
public:
	Traversor3VVaE(const MAP& m, Vertex d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VERTEX, EDGE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices adjacent to a vertex by a face
 */
template <typename MAP>
class Traversor3VVaF: public Traversor3XXaY<MAP, VERTEX, FACE>
{
public:
	Traversor3VVaF(const MAP& m, Vertex d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VERTEX, FACE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices adjacent to a vertex by a volume
 */
template <typename MAP>
class Traversor3VVaW: public Traversor3XXaY<MAP, VERTEX, VOLUME>
{
public:
	Traversor3VVaW(const MAP& m, Vertex d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VERTEX, VOLUME>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse edges adjacent to an egde by a vertex
 */
template <typename MAP>
class Traversor3EEaV: public Traversor3XXaY<MAP, EDGE, VERTEX>
{
public:
	Traversor3EEaV(const MAP& m, Edge d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, EDGE, VERTEX>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse edges adjacent to an egde by a face
 */
template <typename MAP>
class Traversor3EEaF: public Traversor3XXaY<MAP, EDGE, FACE>
{
public:
	Traversor3EEaF(const MAP& m, Edge d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, EDGE, FACE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse edges adjacent to an egde by a volume
 */
template <typename MAP>
class Traversor3EEaW: public Traversor3XXaY<MAP, EDGE, VOLUME>
{
public:
	Traversor3EEaW(const MAP& m, Edge d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, EDGE, VOLUME>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse faces adjacent to a face by a vertex
 */
template <typename MAP>
class Traversor3FFaV: public Traversor3XXaY<MAP, FACE, VERTEX>
{
public:
	Traversor3FFaV(const MAP& m, Face d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, FACE, VERTEX>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse faces adjacent to a face by an edge
 */
template <typename MAP>
class Traversor3FFaE: public Traversor3XXaY<MAP, FACE, EDGE>
{
public:
	Traversor3FFaE(const MAP& m, Face d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, FACE, EDGE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse faces adjacent to a face by a volume
 */
template <typename MAP>
class Traversor3FFaW: public Traversor3XXaY<MAP, FACE, VOLUME>
{
public:
	Traversor3FFaW(const MAP& m, Face d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, FACE, VOLUME>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes adjacent to a volume by a vertex
 */
template <typename MAP>
class Traversor3WWaV: public Traversor3XXaY<MAP, VOLUME, VERTEX>
{
public:
	Traversor3WWaV(const MAP& m, Vol d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VOLUME, VERTEX>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes adjacent to a volume by an edge
 */
template <typename MAP>
class Traversor3WWaE: public Traversor3XXaY<MAP, VOLUME, EDGE>
{
public:
	Traversor3WWaE(const MAP& m, Vol d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VOLUME, EDGE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes adjacent to a volume by a face
 */
template <typename MAP>
class Traversor3WWaF: public Traversor3XXaY<MAP, VOLUME, FACE>
{
public:
	Traversor3WWaF(const MAP& m, Vol d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VOLUME, FACE>(m, d, forceDartMarker, thread)	{}
};


template <unsigned int ORBIT_TO, unsigned int ORBIT_FROM, typename MAP, typename FUNC>
inline void foreach_incident3(MAP& map, Cell<ORBIT_FROM> c, FUNC f, bool forceDartMarker = false, unsigned int thread = 0)
{
	Traversor3XY<MAP,ORBIT_FROM,ORBIT_TO> trav(const_cast<const MAP&>(map),c,forceDartMarker,thread);
    for (Cell<ORBIT_TO> c = trav.begin(), e = trav.end(); c != e; c = trav.next())
		f(c);
}


template <unsigned int THRU, unsigned int ORBIT, typename MAP, typename FUNC>
inline void foreach_adjacent3(MAP& map, Cell<ORBIT> c, FUNC f, bool forceDartMarker = false, unsigned int thread = 0)
{
	Traversor3XXaY<MAP,ORBIT,THRU> trav(const_cast<const MAP&>(map),c,forceDartMarker,thread);
    for (Cell<ORBIT> c = trav.begin(), e = trav.end(); c != e; c = trav.next())
		f(c);
}



} // namespace CGoGN

#include "Topology/generic/traversor/traversor3.hpp"

#endif
