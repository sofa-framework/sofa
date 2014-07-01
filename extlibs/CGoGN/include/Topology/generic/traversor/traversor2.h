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

#ifndef __TRAVERSOR2_H__
#define __TRAVERSOR2_H__

#include "Topology/generic/dart.h"
//#include "Topology/generic/traversorGen.h"
#include "Topology/generic/cells.h"
#include <functional>

namespace CGoGN
{

/*******************************************************************************
					VERTEX CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the edges incident to a given vertex
template <typename MAP>
class Traversor2VE//: public Traversor<MAP>
{
private:
	const MAP& m ;
	Edge start ;
	Edge current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2VE(const MAP& map, Vertex dart) ;

	inline Edge begin() ;
	inline Edge end() ;
	inline Edge next() ;
} ;

// Traverse the faces incident to a given vertex
template <typename MAP>
class Traversor2VF //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Face start ;
	Face current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2VF(const MAP& map, Vertex dart) ;

	inline Face begin() ;
	inline Face end() ;
	inline Face next() ;
} ;

// Traverse the vertices adjacent to a given vertex through sharing a common edge
template <typename MAP>
class Traversor2VVaE //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Vertex start ;
	Vertex current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2VVaE(const MAP& map, Vertex dart) ;

	inline Vertex begin() ;
	inline Vertex end() ;
	inline Vertex next() ;
} ;

// Traverse the vertices adjacent to a given vertex through sharing a common face
template <typename MAP>
class Traversor2VVaF //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Vertex start ;
	Vertex current ;

	Vertex stop ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2VVaF(const MAP& map, Vertex dart) ;

	inline Vertex begin() ;
	inline Vertex end() ;
	inline Vertex next() ;
} ;

/*******************************************************************************
					EDGE CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the vertices incident to a given edge
template <typename MAP>
class Traversor2EV //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Vertex start ;
	Vertex current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2EV(const MAP& map, Edge dart) ;

	inline Vertex begin() ;
	inline Vertex end() ;
	inline Vertex next() ;
} ;

// Traverse the faces incident to a given edge
template <typename MAP>
class Traversor2EF //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Face start ;
	Face current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2EF(const MAP& map, Edge dart) ;

	inline Face begin() ;
	inline Face end() ;
	inline Face next() ;
} ;

// Traverse the edges adjacent to a given edge through sharing a common vertex
template <typename MAP>
class Traversor2EEaV //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Edge start ;
	Edge current ;

	Edge stop1, stop2 ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2EEaV(const MAP& map, Edge dart) ;

	inline Edge begin() ;
	inline Edge end() ;
	inline Edge next() ;
} ;

// Traverse the edges adjacent to a given edge through sharing a common face
template <typename MAP>
class Traversor2EEaF //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Edge start ;
	Edge current ;

	Edge stop1, stop2 ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2EEaF(const MAP& map, Edge dart) ;

	inline Edge begin() ;
	inline Edge end() ;
	inline Edge next() ;
} ;

/*******************************************************************************
					FACE CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the vertices incident to a given face
template <typename MAP>
class Traversor2FV //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Vertex start ;
	Vertex current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2FV(const MAP& map, Face dart) ;

	inline Vertex begin() ;
	inline Vertex end() ;
	inline Vertex next() ;
} ;


//// Traverse the edges incident to a given face (equivalent to vertices)
//template <typename MAP>
//class Traversor2FE: public Traversor2FV<MAP>
//{
//public:
//	Traversor2FE(const MAP& map, Dart dart):Traversor2FV<MAP>(map,dart){}
//} ;

// Traverse the vertices incident to a given face
template <typename MAP>
class Traversor2FE //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Edge start ;
	Edge current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2FE(const MAP& map, Face dart) ;

	inline Edge begin() ;
	inline Edge end() ;
	inline Edge next() ;
} ;


// Traverse the faces adjacent to a given face through sharing a common vertex
template <typename MAP>
class Traversor2FFaV //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Face start ;
	Face current ;

	Face stop ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2FFaV(const MAP& map, Face dart) ;

	inline Face begin() ;
	inline Face end() ;
	inline Face next() ;
} ;

// Traverse the faces adjacent to a given face through sharing a common edge
// Warning mult-incidence is not managed (some faces can be send several times)
template <typename MAP>
class Traversor2FFaE //: public Traversor<MAP>
{
private:
	const MAP& m ;
	Face start ;
	Face current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	Traversor2FFaE(const MAP& map, Face dart) ;

	inline Face begin() ;
	inline Face end() ;
	inline Face next() ;
} ;


template <typename MAP, unsigned int F, unsigned int T>
class IncidentTrav2
{
public:
	IncidentTrav2(const MAP&, Cell<F>) {}

};


template <typename MAP>
class IncidentTrav2<MAP,VERTEX,EDGE>
{
public:
	Traversor2VE<MAP> t;
	IncidentTrav2(const MAP& m, Vertex d):t(m,d) {}
};

template <typename MAP>
class IncidentTrav2<MAP,VERTEX,FACE>
{
public:
	Traversor2VF<MAP> t;
	IncidentTrav2(const MAP& m, Vertex d):t(m,d) {}
};

template <typename MAP>
class IncidentTrav2<MAP,EDGE,VERTEX>
{
public:
	Traversor2EV<MAP> t;
	IncidentTrav2(const MAP& m, Edge d):t(m,d) {}
};

template <typename MAP>
class IncidentTrav2<MAP,EDGE,FACE>
{
public:
	Traversor2EF<MAP> t;
	IncidentTrav2(const MAP& m, Edge d):t(m,d) {}
};

template <typename MAP>
class IncidentTrav2<MAP,FACE,VERTEX>
{
public:
	Traversor2FV<MAP> t;
	IncidentTrav2(const MAP& m, Face d):t(m,d) {}
};

template <typename MAP>
class IncidentTrav2<MAP,FACE,EDGE>
{
public:
	Traversor2FE<MAP> t;
	IncidentTrav2(const MAP& m, Face d):t(m,d) {}
};



template <typename MAP, unsigned int F, unsigned int T>
class AdjacentTrav2
{
public:
	AdjacentTrav2(const MAP&, Cell<F>) {}
};


template <typename MAP>
class AdjacentTrav2<MAP,VERTEX,EDGE>
{
public:
	Traversor2VVaE<MAP> t;
	AdjacentTrav2(const MAP& m, Vertex d):t(m,d) {}
};

template <typename MAP>
class AdjacentTrav2<MAP,VERTEX,FACE>
{
public:
	Traversor2VVaF<MAP> t;
	AdjacentTrav2(const MAP& m, Vertex d):t(m,d) {}
};

template <typename MAP>
class AdjacentTrav2<MAP,EDGE,VERTEX>
{
public:
	Traversor2EEaV<MAP> t;
	AdjacentTrav2(const MAP& m, Edge d):t(m,d) {}
};

template <typename MAP>
class AdjacentTrav2<MAP,EDGE,FACE>
{
public:
	Traversor2EEaF<MAP> t;
	AdjacentTrav2(const MAP& m, Edge d):t(m,d) {}
};

template <typename MAP>
class AdjacentTrav2<MAP,FACE,VERTEX>
{
public:
	Traversor2FFaV<MAP> t;
	AdjacentTrav2(const MAP& m, Face d):t(m,d) {}
};

template <typename MAP>
class AdjacentTrav2<MAP,FACE,EDGE>
{
public:
	Traversor2FFaE<MAP> t;
	AdjacentTrav2(const MAP& m, Face d):t(m,d) {}
};


template <unsigned int ORBIT_TO, unsigned int ORBIT_FROM, typename MAP, typename FUNC>
inline void foreach_incident2(MAP& map, Cell<ORBIT_FROM> c, FUNC f)
{
	IncidentTrav2<MAP,ORBIT_FROM,ORBIT_TO> trav(const_cast<const MAP&>(map),c);
    for (Cell<ORBIT_TO> c = trav.t.begin(), e = trav.t.end(); c != e; c = trav.t.next())
		f(c);
}

template <unsigned int THRU, unsigned int ORBIT, typename MAP, typename FUNC>
inline void foreach_adjacent2(MAP& map, Cell<ORBIT> c, FUNC f)
{
	AdjacentTrav2<MAP,ORBIT,THRU> trav(const_cast<const MAP&>(map),c);
    for (Cell<ORBIT> c = trav.t.begin(), e = trav.t.end(); c != e; c = trav.t.next())
		f(c);
}



} // namespace CGoGN

#include "Topology/generic/traversor/traversor2.hpp"

#endif
