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
#include "Topology/generic/cells.h"
#include "Topology/generic/traversor/iterTrav.h"

namespace CGoGN
{

/*******************************************************************************
                    VERTEX CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the edges incident to a given vertex
template <typename MAP>
class Traversor2VE
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

    typedef Edge IterType;
    typedef Vertex ParamType;
    typedef MAP MapType;
} ;

// Traverse the faces incident to a given vertex
template <typename MAP>
class Traversor2VF
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

    typedef Face IterType;
    typedef Vertex ParamType;
    typedef MAP MapType;
} ;

// Traverse the vertices adjacent to a given vertex through sharing a common edge
template <typename MAP>
class Traversor2VVaE
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

    typedef Vertex IterType;
    typedef Vertex ParamType;
    typedef MAP MapType;
} ;

// Traverse the vertices adjacent to a given vertex through sharing a common face
template <typename MAP>
class Traversor2VVaF
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

    typedef Vertex IterType;
    typedef Vertex ParamType;
    typedef MAP MapType;
} ;

/*******************************************************************************
                    EDGE CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the vertices incident to a given edge
template <typename MAP>
class Traversor2EV
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

    typedef Vertex IterType;
    typedef Edge ParamType;
    typedef MAP MapType;
} ;

// Traverse the faces incident to a given edge
template <typename MAP>
class Traversor2EF
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

    typedef Face IterType;
    typedef Edge ParamType;
    typedef MAP MapType;
} ;

// Traverse the edges adjacent to a given edge through sharing a common vertex
template <typename MAP>
class Traversor2EEaV
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

    typedef Edge IterType;
    typedef Edge ParamType;
    typedef MAP MapType;
} ;

// Traverse the edges adjacent to a given edge through sharing a common face
template <typename MAP>
class Traversor2EEaF
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

    typedef Edge IterType;
    typedef Edge ParamType;
    typedef MAP MapType;
} ;

/*******************************************************************************
                    FACE CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the vertices incident to a given face
template <typename MAP>
class Traversor2FV
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

    typedef Vertex IterType;
    typedef Face ParamType;
    typedef MAP MapType;
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
class Traversor2FE
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

    typedef Edge IterType;
    typedef Face ParamType;
    typedef MAP MapType;
} ;


// Traverse the faces adjacent to a given face through sharing a common vertex
template <typename MAP>
class Traversor2FFaV
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

    typedef Face IterType;
    typedef Face ParamType;
    typedef MAP MapType;
} ;

// Traverse the faces adjacent to a given face through sharing a common edge
// Warning mult-incidence is not managed (some faces can be send several times)
template <typename MAP>
class Traversor2FFaE
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

    typedef Face IterType;
    typedef Face ParamType;
    typedef MAP MapType;
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

// foreach traversal function

template <unsigned int ORBIT_TO, unsigned int ORBIT_FROM, typename MAP, typename FUNC>
inline void foreach_incident2(MAP& map, Cell<ORBIT_FROM> c, FUNC f)
{
    IncidentTrav2<MAP,ORBIT_FROM,ORBIT_TO> trav(const_cast<const MAP&>(map),c);
    for (Cell<ORBIT_TO> c = trav.t.begin(), e = trav.t.end(); c.dart != e.dart; c = trav.t.next())
        f(c);
}

template <unsigned int THRU, unsigned int ORBIT, typename MAP, typename FUNC>
inline void foreach_adjacent2(MAP& map, Cell<ORBIT> c, FUNC f)
{
    AdjacentTrav2<MAP,ORBIT,THRU> trav(const_cast<const MAP&>(map),c);
    for (Cell<ORBIT> c = trav.t.begin(), e = trav.t.end(); c.dart != e.dart; c = trav.t.next())
        f(c);
}



/**
 * template classs that add iterator to Traversor
 * to allow the use of c++11 syntax for (auto d : v)
 */
//template <typename TRAV>
//class Iteratorize: public TRAV
//{
//public:
//	typedef typename TRAV::MapType MAP;
//	typedef typename TRAV::IterType ITER;
//	typedef typename TRAV::ParamType PARAM;

//	Iteratorize(const MAP& map, PARAM p):
//		TRAV(map,p){}

//	class iterator
//	{
//		Iteratorize<TRAV>* m_ptr;
//		ITER m_index;

//	public:

//		inline iterator(Iteratorize<TRAV>* p, ITER i): m_ptr(p),m_index(i){}

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
//		return iterator(this,TRAV::begin());
//	}

//	inline iterator end()
//	{
//		return iterator(this,TRAV::end());
//	}

//};

// functions that return the traversor+iterator
// functions instead of typedef because function
// allows the compiler to deduce template param

template <typename MAP>
inline Iteratorize< Traversor2VE<MAP> > edgesIncidentToVertex2(const MAP& m, Vertex c)
{
    return Iteratorize< Traversor2VE<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2VF<MAP> > facesIncidentToVertex2(const MAP& m, Vertex c)
{
    return Iteratorize< Traversor2VF<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2EV<MAP> > verticesIncidentToEdge2(const MAP& m, Edge c)
{
    return Iteratorize< Traversor2EV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2EF<MAP> > facesIncidentToEdge2(const MAP& m, Edge c)
{
    return Iteratorize< Traversor2EF<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2FV<MAP> > verticesIncidentToFace2(const MAP& m, Face c)
{
    return Iteratorize< Traversor2FV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2FE<MAP> > edgesIncidentToFace2(const MAP& m, Face c)
{
    return Iteratorize< Traversor2FE<MAP> >(m, c);
}


template <typename MAP>
inline Iteratorize< Traversor2VVaE<MAP> > verticesAdjacentByEdge2(const MAP& m, Vertex c)
{
    return Iteratorize< Traversor2VVaE<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2VVaF<MAP> > verticesAdjacentByFace2(const MAP& m, Vertex c)
{
    return Iteratorize< Traversor2VVaF<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2EEaV<MAP> > edgesAdjacentByVertex2(const MAP& m, Edge c)
{
    return Iteratorize< Traversor2EEaV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2EEaF<MAP> > edgesAdjacentByFace2(const MAP& m, Edge c)
{
    return Iteratorize< Traversor2EEaF<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2FFaV<MAP> > facesAdjacentByVertex2(const MAP& m, Face c)
{
    return Iteratorize< Traversor2FFaV<MAP> >(m, c);
}

template <typename MAP>
inline Iteratorize< Traversor2FFaE<MAP> > facesAdjacentByEdge2(const MAP& m, Face c)
{
    return Iteratorize< Traversor2FFaE<MAP> >(m, c);
}



} // namespace CGoGN

#include "Topology/generic/traversor/traversor2.hpp"

#endif
