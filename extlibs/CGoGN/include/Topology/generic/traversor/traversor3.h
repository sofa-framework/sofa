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
#include "Topology/generic/cells.h"
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversor/traversorCell.h"
//#include "Topology/generic/traversor/traversorGen.h"
#include "Topology/generic/traversor/traversorDoO.h"
#include "Topology/generic/traversor/iterTrav.h"

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
	MarkerForTraversor(const MAP& map, bool forceDartMarker = false) ;
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
//    BOOST_STATIC_ASSERT(MAP::DIMENSION >= 3u) ;   // WARNING: have to remove the assertion because it makes mpl::if_ fail !
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
	Traversor3XY(const MAP& map, Cell<ORBX> c, bool forceDartMarker = false) ;
	Traversor3XY(const MAP& map, Cell<ORBX> c, MarkerForTraversor<MAP, ORBY>& tmo, bool forceDartMarker = false) ;
    Traversor3XY(const Traversor3XY& tra3xy);
	~Traversor3XY();

//	Traversor3XY(Traversor3XY<MAP,ORBX,ORBY>&& tra):
//	m_map(tra.m_map),m_tradoo(std::move(tra.m_tradoo))
//	{
//		m_dmark = tra.m_dmark;
//		m_cmark = tra.m_cmark;
//		m_current = tra.m_current;
//	}

	Cell<ORBY> begin() ;
	Cell<ORBY> end() ;
	Cell<ORBY> next() ;

	typedef Cell<ORBY> IterType;
	typedef Cell<ORBX> ParamType;
	typedef MAP MapType;
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
	Traversor3XXaY(const MAP& map, Cell<ORBX> c, bool forceDartMarker = false);

	Cell<ORBX> begin();
	Cell<ORBX> end();
	Cell<ORBX> next();

	typedef Cell<ORBX> IterType;
	typedef Cell<ORBX> ParamType;
	typedef MAP MapType;
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
	Traversor3WV(const MAP& m, Vol dart, bool forceDartMarker = false) : Traversor3XY<MAP, VOLUME, VERTEX>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse edges incident to volume
 */
template <typename MAP>
class Traversor3WE: public Traversor3XY<MAP, VOLUME, EDGE>
{
public:
	Traversor3WE(const MAP& m, Vol dart, bool forceDartMarker = false) : Traversor3XY<MAP, VOLUME, EDGE>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse faces incident to volume
 */
template <typename MAP>
class Traversor3WF: public Traversor3XY<MAP, VOLUME, FACE>
{
public:
	Traversor3WF(const MAP& m, Vol dart, bool forceDartMarker = false) : Traversor3XY<MAP, VOLUME, FACE>(m, dart, forceDartMarker) {}
};

/**
 * Traverse vertices incident to face
 */
template <typename MAP>
class Traversor3FV: public Traversor3XY<MAP, FACE, VERTEX>
{
public:
	Traversor3FV(const MAP& m, Face dart, bool forceDartMarker = false) : Traversor3XY<MAP, FACE, VERTEX>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse edges incident to face
 */
template <typename MAP>
class Traversor3FE: public Traversor3XY<MAP, FACE, EDGE>
{
public:
	Traversor3FE(const MAP& m, Face dart, bool forceDartMarker = false) : Traversor3XY<MAP, FACE, EDGE>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse volumes incident to face
 */
template <typename MAP>
class Traversor3FW: public Traversor3XY<MAP, FACE, VOLUME>
{
public:
	Traversor3FW(const MAP& m, Face dart, bool forceDartMarker = false) : Traversor3XY<MAP, FACE, VOLUME>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse vertices incident to edge
 */
template <typename MAP>
class Traversor3EV: public Traversor3XY<MAP, EDGE, VERTEX>
{
public:
	Traversor3EV(const MAP& m, Edge dart, bool forceDartMarker = false) : Traversor3XY<MAP, EDGE, VERTEX>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse faces incident to edge
 */
template <typename MAP>
class Traversor3EF: public Traversor3XY<MAP, EDGE, FACE>
{
public:
	Traversor3EF(const MAP& m, Edge dart, bool forceDartMarker = false) : Traversor3XY<MAP, EDGE, FACE>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse volumes incident to edge
 */
template <typename MAP>
class Traversor3EW: public Traversor3XY<MAP, EDGE, VOLUME>
{
public:
	Traversor3EW(const MAP& m, Edge dart, bool forceDartMarker = false) : Traversor3XY<MAP, EDGE, VOLUME>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse edges incident to vertex
 */
template <typename MAP>
class Traversor3VE: public Traversor3XY<MAP, VERTEX, EDGE>
{
public:
	Traversor3VE(const MAP& m, Vertex dart, bool forceDartMarker = false) : Traversor3XY<MAP, VERTEX, EDGE>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse faces incident to vertex
 */
template <typename MAP>
class Traversor3VF: public Traversor3XY<MAP, VERTEX, FACE>
{
public:
	Traversor3VF(const MAP& m, Vertex dart, bool forceDartMarker = false) : Traversor3XY<MAP, VERTEX, FACE>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse volumes incident to vertex
 */
template <typename MAP>
class Traversor3VW: public Traversor3XY<MAP, VERTEX, VOLUME>
{
public:
	Traversor3VW(const MAP& m, Vertex dart, bool forceDartMarker = false) : Traversor3XY<MAP, VERTEX, VOLUME>(m, dart, forceDartMarker)	{}
};

/**
 * Traverse vertices adjacent to a vertex by an edge
 */
template <typename MAP>
class Traversor3VVaE: public Traversor3XXaY<MAP, VERTEX, EDGE>
{
public:
	Traversor3VVaE(const MAP& m, Vertex d, bool forceDartMarker = false) : Traversor3XXaY<MAP, VERTEX, EDGE>(m, d, forceDartMarker)	{}
};

/**
 * Traverse vertices adjacent to a vertex by a face
 */
template <typename MAP>
class Traversor3VVaF: public Traversor3XXaY<MAP, VERTEX, FACE>
{
public:
	Traversor3VVaF(const MAP& m, Vertex d, bool forceDartMarker = false) : Traversor3XXaY<MAP, VERTEX, FACE>(m, d, forceDartMarker)	{}
};

/**
 * Traverse vertices adjacent to a vertex by a volume
 */
template <typename MAP>
class Traversor3VVaW: public Traversor3XXaY<MAP, VERTEX, VOLUME>
{
public:
	Traversor3VVaW(const MAP& m, Vertex d, bool forceDartMarker = false) : Traversor3XXaY<MAP, VERTEX, VOLUME>(m, d, forceDartMarker)	{}
};

/**
 * Traverse edges adjacent to an egde by a vertex
 */
template <typename MAP>
class Traversor3EEaV: public Traversor3XXaY<MAP, EDGE, VERTEX>
{
public:
	Traversor3EEaV(const MAP& m, Edge d, bool forceDartMarker = false) : Traversor3XXaY<MAP, EDGE, VERTEX>(m, d, forceDartMarker)	{}
};

/**
 * Traverse edges adjacent to an egde by a face
 */
template <typename MAP>
class Traversor3EEaF: public Traversor3XXaY<MAP, EDGE, FACE>
{
public:
	Traversor3EEaF(const MAP& m, Edge d, bool forceDartMarker = false) : Traversor3XXaY<MAP, EDGE, FACE>(m, d, forceDartMarker)	{}
};

/**
 * Traverse edges adjacent to an egde by a volume
 */
template <typename MAP>
class Traversor3EEaW: public Traversor3XXaY<MAP, EDGE, VOLUME>
{
public:
	Traversor3EEaW(const MAP& m, Edge d, bool forceDartMarker = false) : Traversor3XXaY<MAP, EDGE, VOLUME>(m, d, forceDartMarker)	{}
};

/**
 * Traverse faces adjacent to a face by a vertex
 */
template <typename MAP>
class Traversor3FFaV: public Traversor3XXaY<MAP, FACE, VERTEX>
{
public:
	Traversor3FFaV(const MAP& m, Face d, bool forceDartMarker = false) : Traversor3XXaY<MAP, FACE, VERTEX>(m, d, forceDartMarker)	{}
};

/**
 * Traverse faces adjacent to a face by an edge
 */
template <typename MAP>
class Traversor3FFaE: public Traversor3XXaY<MAP, FACE, EDGE>
{
public:
	Traversor3FFaE(const MAP& m, Face d, bool forceDartMarker = false) : Traversor3XXaY<MAP, FACE, EDGE>(m, d, forceDartMarker)	{}
};

/**
 * Traverse faces adjacent to a face by a volume
 */
template <typename MAP>
class Traversor3FFaW: public Traversor3XXaY<MAP, FACE, VOLUME>
{
public:
	Traversor3FFaW(const MAP& m, Face d, bool forceDartMarker = false) : Traversor3XXaY<MAP, FACE, VOLUME>(m, d, forceDartMarker)	{}
};

/**
 * Traverse volumes adjacent to a volume by a vertex
 */
template <typename MAP>
class Traversor3WWaV: public Traversor3XXaY<MAP, VOLUME, VERTEX>
{
public:
	Traversor3WWaV(const MAP& m, Vol d, bool forceDartMarker = false) : Traversor3XXaY<MAP, VOLUME, VERTEX>(m, d, forceDartMarker)	{}
};

/**
 * Traverse volumes adjacent to a volume by an edge
 */
template <typename MAP>
class Traversor3WWaE: public Traversor3XXaY<MAP, VOLUME, EDGE>
{
public:
	Traversor3WWaE(const MAP& m, Vol d, bool forceDartMarker = false) : Traversor3XXaY<MAP, VOLUME, EDGE>(m, d, forceDartMarker)	{}
};

/**
 * Traverse volumes adjacent to a volume by a face
 */
template <typename MAP>
class Traversor3WWaF: public Traversor3XXaY<MAP, VOLUME, FACE>
{
public:
	Traversor3WWaF(const MAP& m, Vol d, bool forceDartMarker = false) : Traversor3XXaY<MAP, VOLUME, FACE>(m, d, forceDartMarker)	{}
};


template <unsigned int ORBIT_TO, unsigned int ORBIT_FROM, typename MAP, typename FUNC>
inline void foreach_incident3(MAP& map, Cell<ORBIT_FROM> c, FUNC f, bool forceDartMarker = false)
{
	Traversor3XY<MAP,ORBIT_FROM,ORBIT_TO> trav(const_cast<const MAP&>(map),c,forceDartMarker);
	for (Cell<ORBIT_TO> c = trav.begin(), e = trav.end(); c.dart != e.dart; c = trav.next())
		f(c);
}


template <unsigned int THRU, unsigned int ORBIT, typename MAP, typename FUNC>
inline void foreach_adjacent3(MAP& map, Cell<ORBIT> c, FUNC f, bool forceDartMarker = false)
{
	Traversor3XXaY<MAP,ORBIT,THRU> trav(const_cast<const MAP&>(map),c,forceDartMarker);
	for (Cell<ORBIT> c = trav.begin(), e = trav.end(); c.dart != e.dart; c = trav.next())
		f(c);
}


/**
 * template classs that add iterator to Traversor
 * to allow the use of c++11 syntax for (auto d : v)
 */
//template <typename TRAV>
//class Iteratorize3: public TRAV
//{
//public:
//	typedef typename TRAV::MapType MAP;
//	typedef typename TRAV::IterType ITER;
//	typedef typename TRAV::ParamType PARAM;

//	Iteratorize3(const MAP& map, PARAM p):
//		TRAV(map,p),m_begin(this,TRAV::begin()),m_end(this,TRAV::end())
//	{
////		m_begin = this->begin();
////		m_end = this->end();
//	}


//	class iterator
//	{
//		Iteratorize3<TRAV>* m_ptr;
//		ITER m_index;

//	public:

//		inline iterator(Iteratorize3<TRAV>* p, ITER i): m_ptr(p),m_index(i){}

//		inline iterator& operator++()
//		{
//			m_index = m_ptr->next();
//			return *this;
//		}

//		inline ITER& operator*()
//		{
//			return m_index;
//		}

//		inline bool operator!=(const iterator& it)
//		{
//			return m_index.dart != it.m_index.dart;
//		}

//	};

//	inline iterator begin()
//	{
////		return iterator(this,TRAV::begin());
//		return m_begin;
//	}

//	inline iterator end()
//	{
////		return iterator(this,TRAV::end());
//		return m_end;
//	}

//protected:
//	iterator m_begin;
//	iterator m_end;
//};

// functions that return the traversor+iterator
// functions instead of typedef because function
// allows the compiler to deduce template param

template <typename MAP>
inline Iteratorize< Traversor3VE<MAP> > edgesIncidentToVertex3(const MAP& m, Vertex c)
{
	return Iteratorize< Traversor3VE<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3VF<MAP> > facesIncidentToVertex3(const MAP& m, Vertex c)
{
	return Iteratorize< Traversor3VF<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3VW<MAP> > VolumesIncidentToVertex3(const MAP& m, Vertex c)
{
	return Iteratorize< Traversor3VW<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3EV<MAP> > verticesIncidentToEdge3(const MAP& m, Edge c)
{
	return Iteratorize< Traversor3EV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3EF<MAP> > facesIncidentToEdge3(const MAP& m, Edge c)
{
	return Iteratorize< Traversor3EF<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3EW<MAP> > volumesIncidentToEdge3(const MAP& m, Edge c)
{
	return Iteratorize< Traversor3EW<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3FV<MAP> > verticesIncidentToFace3(const MAP& m, Face c)
{
	return Iteratorize< Traversor3FV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3FE<MAP> > edgesIncidentToFace3(const MAP& m, Face c)
{
	return Iteratorize< Traversor3FE<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3FW<MAP> > volumesIncidentToFace3(const MAP& m, Face c)
{
	return Iteratorize< Traversor3FW<MAP> >(m, c);
}


template <typename MAP>
inline Iteratorize< Traversor3WV<MAP> > verticesIncidentToVolume3(const MAP& m, Vol c)
{
	return Iteratorize< Traversor3WV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3WE<MAP> > edgesIncidentToVolume3(const MAP& m, Vol c)
{
	return Iteratorize< Traversor3WE<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3WF<MAP> > facesIncidentToVolume3(const MAP& m, Vol c)
{
	return Iteratorize< Traversor3WF<MAP> >(m, c);
}


template <typename MAP>
inline Iteratorize< Traversor3VVaE<MAP> > verticesAdjacentByEdge3(const MAP& m, Vertex c)
{
	return Iteratorize< Traversor3VVaE<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3VVaF<MAP> > verticesAdjacentByFace3(const MAP& m, Vertex c)
{
	return Iteratorize< Traversor3VVaF<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3VVaW<MAP> > verticesAdjacentByVolume3(const MAP& m, Vertex c)
{
	return Iteratorize< Traversor3VVaW<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3EEaV<MAP> > edgesAdjacentByVertex3(const MAP& m, Edge c)
{
	return Iteratorize< Traversor3EEaV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3EEaF<MAP> > edgesAdjacentByFace3(const MAP& m, Edge c)
{
	return Iteratorize< Traversor3EEaF<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3EEaW<MAP> > edgesAdjacentByVolume3(const MAP& m, Edge c)
{
	return Iteratorize< Traversor3EEaW<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3FFaV<MAP> > facesAdjacentByVertex3(const MAP& m, Face c)
{
	return Iteratorize< Traversor3FFaV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3FFaE<MAP> > facesAdjacentByEdge3(const MAP& m, Face c)
{
	return Iteratorize< Traversor3FFaE<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3FFaW<MAP> > facesAdjacentByVolume3(const MAP& m, Face c)
{
	return Iteratorize< Traversor3FFaW<MAP> >(m, c);
}


template <typename MAP>
inline Iteratorize< Traversor3WWaV<MAP> > volumesAdjacentByVertex3(const MAP& m, Vol c)
{
	return Iteratorize< Traversor3WWaV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3WWaE<MAP> > volumesAdjacentByEdge3(const MAP& m, Vol c)
{
	return Iteratorize< Traversor3WWaE<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor3WWaF<MAP> > volumesAdjacentByFace3(const MAP& m, Vol c)
{
	return Iteratorize< Traversor3WWaF<MAP> >(m, c);
}



} // namespace CGoGN

#include "Topology/generic/traversor/traversor3.hpp"

#endif
