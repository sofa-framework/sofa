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
#include "Topology/generic/traversorCell.h"
//#include "Topology/generic/traversorGen.h"
#include "Topology/generic/traversorDoO.h"

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
	MAP& m_map ;
	DartMarkerStore* m_dmark ;
	CellMarkerStore<ORBIT>* m_cmark ;
public:
	MarkerForTraversor(MAP& map, bool forceDartMarker = false, unsigned int thread = 0) ;
	~MarkerForTraversor();
	DartMarkerStore* dmark();
	CellMarkerStore<ORBIT>* cmark();
	void mark(Dart d);
	void unmark(Dart d);
	bool isMarked(Dart d);
} ;


/**
 * Generic class Traversor (do not use directly)
 * Traverse all Y incident to X
 */
template <typename MAP, unsigned int ORBX, unsigned int ORBY>
class Traversor3XY//: public Traversor<MAP>
{
private:
	MAP& m_map ;
	DartMarkerStore* m_dmark ;
	CellMarkerStore<ORBY>* m_cmark ;
	Dart m_current ;
	TraversorDartsOfOrbit<MAP, ORBX> m_tradoo;

	std::vector<Dart>* m_QLT;
	std::vector<Dart>::iterator m_ItDarts;

	bool m_allocated;
	bool m_first;
public:
	Traversor3XY(MAP& map, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) ;
	Traversor3XY(MAP& map, Dart dart, MarkerForTraversor<MAP, ORBY>& tmo, bool forceDartMarker = false, unsigned int thread = 0) ;
	~Traversor3XY();
	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

/**
 * Generic class Traversor (do not use directly)
 * Traverse all X adjacent to X by an Y
 */
template <typename MAP, unsigned int ORBX, unsigned int ORBY>
class Traversor3XXaY//: public Traversor<MAP>
{
private:
	MAP& m_map ;
	std::vector<Dart> m_vecDarts;
	std::vector<Dart>::iterator m_iter;

	std::vector<Dart>* m_QLT;
	std::vector<Dart>::iterator m_ItDarts;
public:
	Traversor3XXaY(MAP& map, Dart dart, bool forceDartMarker = false, unsigned int thread = 0);

	Dart begin();

	Dart end();

	Dart next();
};


/**
 * Traverse vertices incident to volume
 */
template <typename MAP>
class Traversor3WV: public Traversor3XY<MAP, VOLUME, VERTEX>
{
public:
	Traversor3WV(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VOLUME, VERTEX>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse edges incident to volume
 */
template <typename MAP>
class Traversor3WE: public Traversor3XY<MAP, VOLUME, EDGE>
{
public:
	Traversor3WE(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VOLUME, EDGE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse faces incident to volume
 */
template <typename MAP>
class Traversor3WF: public Traversor3XY<MAP, VOLUME, FACE>
{
public:
	Traversor3WF(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VOLUME, FACE>(m, dart, forceDartMarker, thread) {}
};

/**
 * Traverse vertices incident to face
 */
template <typename MAP>
class Traversor3FV: public Traversor3XY<MAP, FACE, VERTEX>
{
public:
	Traversor3FV(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, FACE, VERTEX>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse edges incident to face
 */
template <typename MAP>
class Traversor3FE: public Traversor3XY<MAP, FACE, EDGE>
{
public:
	Traversor3FE(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, FACE, EDGE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes incident to face
 */
template <typename MAP>
class Traversor3FW: public Traversor3XY<MAP, FACE, VOLUME>
{
public:
	Traversor3FW(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, FACE, VOLUME>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices incident to edge
 */
template <typename MAP>
class Traversor3EV: public Traversor3XY<MAP, EDGE, VERTEX>
{
public:
	Traversor3EV(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, EDGE, VERTEX>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse faces incident to edge
 */
template <typename MAP>
class Traversor3EF: public Traversor3XY<MAP, EDGE, FACE>
{
public:
	Traversor3EF(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, EDGE, FACE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes incident to edge
 */
template <typename MAP>
class Traversor3EW: public Traversor3XY<MAP, EDGE, VOLUME>
{
public:
	Traversor3EW(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, EDGE, VOLUME>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse edges incident to vertex
 */
template <typename MAP>
class Traversor3VE: public Traversor3XY<MAP, VERTEX, EDGE>
{
public:
	Traversor3VE(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VERTEX, EDGE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse faces incident to vertex
 */
template <typename MAP>
class Traversor3VF: public Traversor3XY<MAP, VERTEX, FACE>
{
public:
	Traversor3VF(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VERTEX, FACE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes incident to vertex
 */
template <typename MAP>
class Traversor3VW: public Traversor3XY<MAP, VERTEX, VOLUME>
{
public:
	Traversor3VW(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XY<MAP, VERTEX, VOLUME>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices adjacent to a vertex by an edge
 */
template <typename MAP>
class Traversor3VVaE: public Traversor3XXaY<MAP, VERTEX, EDGE>
{
public:
	Traversor3VVaE(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VERTEX, EDGE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices adjacent to a vertex by a face
 */
template <typename MAP>
class Traversor3VVaF: public Traversor3XXaY<MAP, VERTEX, FACE>
{
public:
	Traversor3VVaF(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VERTEX, FACE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices adjacent to a vertex by a volume
 */
template <typename MAP>
class Traversor3VVaW: public Traversor3XXaY<MAP, VERTEX, VOLUME>
{
public:
	Traversor3VVaW(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VERTEX, VOLUME>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse edges adjacent to an egde by a vertex
 */
template <typename MAP>
class Traversor3EEaV: public Traversor3XXaY<MAP, EDGE, VERTEX>
{
public:
	Traversor3EEaV(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, EDGE, VERTEX>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse edges adjacent to an egde by a face
 */
template <typename MAP>
class Traversor3EEaF: public Traversor3XXaY<MAP, EDGE, FACE>
{
public:
	Traversor3EEaF(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, EDGE, FACE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse edges adjacent to an egde by a volume
 */
template <typename MAP>
class Traversor3EEaW: public Traversor3XXaY<MAP, EDGE, VOLUME>
{
public:
	Traversor3EEaW(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, EDGE, VOLUME>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse faces adjacent to a face by a vertex
 */
template <typename MAP>
class Traversor3FFaV: public Traversor3XXaY<MAP, FACE, VERTEX>
{
public:
	Traversor3FFaV(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, FACE, VERTEX>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse faces adjacent to a face by an edge
 */
template <typename MAP>
class Traversor3FFaE: public Traversor3XXaY<MAP, FACE, EDGE>
{
public:
	Traversor3FFaE(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, FACE, EDGE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse faces adjacent to a face by a volume
 */
template <typename MAP>
class Traversor3FFaW: public Traversor3XXaY<MAP, FACE, VOLUME>
{
public:
	Traversor3FFaW(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, FACE, VOLUME>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes adjacent to a volume by a vertex
 */
template <typename MAP>
class Traversor3WWaV: public Traversor3XXaY<MAP, VOLUME, VERTEX>
{
public:
	Traversor3WWaV(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VOLUME, VERTEX>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes adjacent to a volume by an edge
 */
template <typename MAP>
class Traversor3WWaE: public Traversor3XXaY<MAP, VOLUME, EDGE>
{
public:
	Traversor3WWaE(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VOLUME, EDGE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes adjacent to a volume by a face
 */
template <typename MAP>
class Traversor3WWaF: public Traversor3XXaY<MAP, VOLUME, FACE>
{
public:
	Traversor3WWaF(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : Traversor3XXaY<MAP, VOLUME, FACE>(m, d, forceDartMarker, thread)	{}
};

} // namespace CGoGN

#include "Topology/generic/traversor3.hpp"

#endif
