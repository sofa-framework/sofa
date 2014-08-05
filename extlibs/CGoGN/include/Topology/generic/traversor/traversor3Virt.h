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

#ifndef __VTraversor3_VIRT_H__
#define __VTraversor3_VIRT_H__

#include "Topology/generic/dart.h"
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversor/traversorCell.h"
#include "Topology/generic/traversor/traversorGen.h"
#include "Topology/generic/traversor/traversorDoO.h"

namespace CGoGN
{

/**
 * class Marker for Traversor usefull to combine
 * several TraversorXY
 */
template <typename MAP, unsigned int ORBIT>
class VMarkerForTraversor
{
private:
	MAP& m_map ;
	DartMarkerStore<MAP>* m_dmark ;
	CellMarkerStore<MAP, ORBIT>* m_cmark ;

public:
	VMarkerForTraversor(MAP& map, bool forceDartMarker = false, unsigned int thread = 0) ;
	~VMarkerForTraversor();
	DartMarkerStore<MAP>* dmark();
	CellMarkerStore<MAP, ORBIT>* cmark();
	void mark(Dart d);
	void unmark(Dart d);
	bool isMarked(Dart d);
} ;


/**
 * Generic class Traversor (do not use directly)
 * Traverse all Y incident to X
 */
template <typename MAP, unsigned int ORBX, unsigned int ORBY>
class VTraversor3XY: public Traversor
{
private:
	MAP& m_map ;
	DartMarkerStore<MAP>* m_dmark ;
	CellMarkerStore<MAP, ORBY>* m_cmark ;
	Dart m_current ;
	TraversorDartsOfOrbit<MAP, ORBX> m_tradoo;

	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;

	bool m_allocated;
	bool m_first;

public:
	VTraversor3XY(MAP& map, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) ;
	VTraversor3XY(MAP& map, Dart dart, VMarkerForTraversor<MAP, ORBY>& tmo, bool forceDartMarker = false, unsigned int thread = 0) ;
	~VTraversor3XY();

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

/**
 * Generic class Traversor (do not use directly)
 * Traverse all X adjacent to X by an Y
 */
template <typename MAP, unsigned int ORBX, unsigned int ORBY>
class VTraversor3XXaY: public Traversor
{
private:
	MAP& m_map ;
	std::vector<Dart> m_vecDarts;
	std::vector<Dart>::iterator m_iter;

	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;

public:
	VTraversor3XXaY(MAP& map, Dart dart, bool forceDartMarker = false, unsigned int thread = 0);

	Dart begin();
	Dart end();
	Dart next();
};


/**
 * Traverse vertices incident to volume
 */
template <typename MAP>
class VTraversor3WV: public VTraversor3XY<MAP, VOLUME, VERTEX>
{
public:
	VTraversor3WV(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, VOLUME, VERTEX>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse edges incident to volume
 */
template <typename MAP>
class VTraversor3WE: public VTraversor3XY<MAP, VOLUME, EDGE>
{
public:
	VTraversor3WE(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, VOLUME, EDGE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse faces incident to volume
 */
template <typename MAP>
class VTraversor3WF: public VTraversor3XY<MAP, VOLUME, FACE>
{
public:
	VTraversor3WF(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, VOLUME, FACE>(m, dart, forceDartMarker, thread) {}
};

/**
 * Traverse vertices incident to face
 */
template <typename MAP>
class VTraversor3FV: public VTraversor3XY<MAP, FACE, VERTEX>
{
public:
	VTraversor3FV(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, FACE, VERTEX>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse edges incident to face
 */
template <typename MAP>
class VTraversor3FE: public VTraversor3XY<MAP, FACE, EDGE>
{
public:
	VTraversor3FE(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, FACE, EDGE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes incident to face
 */
template <typename MAP>
class VTraversor3FW: public VTraversor3XY<MAP, FACE, VOLUME>
{
public:
	VTraversor3FW(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, FACE, VOLUME>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices incident to edge
 */
template <typename MAP>
class VTraversor3EV: public VTraversor3XY<MAP, EDGE, VERTEX>
{
public:
	VTraversor3EV(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, EDGE, VERTEX>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse faces incident to edge
 */
template <typename MAP>
class VTraversor3EF: public VTraversor3XY<MAP, EDGE, FACE>
{
public:
	VTraversor3EF(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, EDGE, FACE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes incident to edge
 */
template <typename MAP>
class VTraversor3EW: public VTraversor3XY<MAP, EDGE, VOLUME>
{
public:
	VTraversor3EW(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, EDGE, VOLUME>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse edges incident to vertex
 */
template <typename MAP>
class VTraversor3VE: public VTraversor3XY<MAP, VERTEX, EDGE>
{
public:
	VTraversor3VE(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, VERTEX, EDGE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse faces incident to vertex
 */
template <typename MAP>
class VTraversor3VF: public VTraversor3XY<MAP, VERTEX, FACE>
{
public:
	VTraversor3VF(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, VERTEX, FACE>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes incident to vertex
 */
template <typename MAP>
class VTraversor3VW: public VTraversor3XY<MAP, VERTEX, VOLUME>
{
public:
	VTraversor3VW(MAP& m, Dart dart, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XY<MAP, VERTEX, VOLUME>(m, dart, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices adjacent to a vertex by an edge
 */
template <typename MAP>
class VTraversor3VVaE: public VTraversor3XXaY<MAP, VERTEX, EDGE>
{
public:
	VTraversor3VVaE(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, VERTEX, EDGE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices adjacent to a vertex by a face
 */
template <typename MAP>
class VTraversor3VVaF: public VTraversor3XXaY<MAP, VERTEX, FACE>
{
public:
	VTraversor3VVaF(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, VERTEX, FACE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse vertices adjacent to a vertex by a volume
 */
template <typename MAP>
class VTraversor3VVaW: public VTraversor3XXaY<MAP, VERTEX, VOLUME>
{
public:
	VTraversor3VVaW(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, VERTEX, VOLUME>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse edges adjacent to an egde by a vertex
 */
template <typename MAP>
class VTraversor3EEaV: public VTraversor3XXaY<MAP, EDGE, VERTEX>
{
public:
	VTraversor3EEaV(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, EDGE, VERTEX>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse edges adjacent to an egde by a face
 */
template <typename MAP>
class VTraversor3EEaF: public VTraversor3XXaY<MAP, EDGE, FACE>
{
public:
	VTraversor3EEaF(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, EDGE, FACE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse edges adjacent to an egde by a volume
 */
template <typename MAP>
class VTraversor3EEaW: public VTraversor3XXaY<MAP, EDGE, VOLUME>
{
public:
	VTraversor3EEaW(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, EDGE, VOLUME>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse faces adjacent to a face by a vertex
 */
template <typename MAP>
class VTraversor3FFaV: public VTraversor3XXaY<MAP, FACE, VERTEX>
{
public:
	VTraversor3FFaV(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, FACE, VERTEX>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse faces adjacent to a face by an edge
 */
template <typename MAP>
class VTraversor3FFaE: public VTraversor3XXaY<MAP, FACE, EDGE>
{
public:
	VTraversor3FFaE(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, FACE, EDGE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse faces adjacent to a face by a volume
 */
template <typename MAP>
class VTraversor3FFaW: public VTraversor3XXaY<MAP, FACE, VOLUME>
{
public:
	VTraversor3FFaW(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, FACE, VOLUME>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes adjacent to a volume by a vertex
 */
template <typename MAP>
class VTraversor3WWaV: public VTraversor3XXaY<MAP, VOLUME, VERTEX>
{
public:
	VTraversor3WWaV(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, VOLUME, VERTEX>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes adjacent to a volume by an edge
 */
template <typename MAP>
class VTraversor3WWaE: public VTraversor3XXaY<MAP, VOLUME, EDGE>
{
public:
	VTraversor3WWaE(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, VOLUME, EDGE>(m, d, forceDartMarker, thread)	{}
};

/**
 * Traverse volumes adjacent to a volume by a face
 */
template <typename MAP>
class VTraversor3WWaF: public VTraversor3XXaY<MAP, VOLUME, FACE>
{
public:
	VTraversor3WWaF(MAP& m, Dart d, bool forceDartMarker = false, unsigned int thread = 0) : VTraversor3XXaY<MAP, VOLUME, FACE>(m, d, forceDartMarker, thread)	{}
};

} // namespace CGoGN

#include "Topology/generic/traversor/traversor3Virt.hpp"

#endif
