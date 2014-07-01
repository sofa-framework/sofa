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

#ifndef __FUNCTOR_H__
#define __FUNCTOR_H__

#include "Topology/generic/dart.h"
//#include "Topology/generic/marker.h"

//#include "Container/attributeMultiVector.h"
//#include <vector>

namespace CGoGN
{

// Base Class for Functors: object function that is applied to darts
/********************************************************/

class FunctorType
{
public:
	FunctorType() {}
	virtual ~FunctorType() {}
	virtual bool operator()(Dart d) = 0;
};


template <typename MAP>
class FunctorMap : public virtual FunctorType
{
protected:
	MAP& m_map ;
public:
	FunctorMap(MAP& m): m_map(m) {}
};





// Selector functors : return true to select or false to not select a dart
/********************************************************/

class FunctorSelect
{
public:
	FunctorSelect() {}
	virtual ~FunctorSelect() {}
	virtual bool operator()(Dart d) const = 0 ;
	virtual FunctorSelect* copy() const = 0;
};

class SelectorTrue : public FunctorSelect
{
public:
	bool operator()(Dart) const { return true; }
	FunctorSelect* copy() const { return new SelectorTrue();}

};

class SelectorFalse : public FunctorSelect
{
public:
	bool operator()(Dart) const { return false; }
	FunctorSelect* copy() const { return new SelectorFalse();}
};

const SelectorTrue allDarts = SelectorTrue() ;


class SelectorAnd : public FunctorSelect
{
protected:
	const FunctorSelect* m_sel1;
	const FunctorSelect* m_sel2;

public:
	SelectorAnd(const FunctorSelect& fs1, const FunctorSelect& fs2) { m_sel1 = fs1.copy(); m_sel2 = fs2.copy();}
	bool operator()(Dart d) const { return m_sel1->operator()(d) && m_sel2->operator()(d); }
	~SelectorAnd() { delete m_sel1; delete m_sel2;}
	FunctorSelect* copy() const  {  return new SelectorAnd(*m_sel1,*m_sel2);}
};

class SelectorOr : public FunctorSelect
{
protected:
	const FunctorSelect* m_sel1;
	const FunctorSelect* m_sel2;

public:
	SelectorOr(const FunctorSelect& fs1, const FunctorSelect& fs2) { m_sel1 = fs1.copy(); m_sel2 = fs2.copy();}
	bool operator()(Dart d) const { return m_sel1->operator()(d) || m_sel2->operator()(d); }
	~SelectorOr() { delete m_sel1; delete m_sel2;}
	FunctorSelect* copy() const  { return new SelectorOr(*m_sel1,*m_sel2);}
};

inline SelectorAnd operator&&(const FunctorSelect& fs1, const FunctorSelect& fs2)
{
	return SelectorAnd(fs1,fs2);
}

inline SelectorOr operator||(const FunctorSelect& fs1, const FunctorSelect& fs2)
{
	return SelectorOr(fs1,fs2);
}

template <typename MAP>
class SelectorVertexBoundary : public FunctorSelect
{
public:
protected:
	const MAP& m_map;
public:
	SelectorVertexBoundary(const MAP& m): m_map(m) {}
	bool operator()(Dart d) const { return m_map.isBoundaryVertex(d); }
	FunctorSelect* copy() const { return new SelectorVertexBoundary(m_map);}
};

template <typename MAP>
class SelectorVertexNoBoundary : public FunctorSelect
{
public:
protected:
	const MAP& m_map;
public:
	SelectorVertexNoBoundary(const MAP& m): m_map(m) {}
	bool operator()(Dart d) const { return !m_map.isBoundaryVertex(d); }
	FunctorSelect* copy() const { return new SelectorVertexNoBoundary(m_map);}
};

template <typename MAP>
class SelectorEdgeBoundary : public FunctorSelect
{
public:
protected:
	const MAP& m_map;
public:
	SelectorEdgeBoundary(const MAP& m): m_map(m) {}
	bool operator()(Dart d) const { return m_map.isBoundaryEdge(d); }
	FunctorSelect* copy() const { return new SelectorEdgeBoundary(m_map);}
};


template <typename MAP>
class SelectorEdgeNoBoundary : public FunctorSelect
{
public:
protected:
	const MAP& m_map;
public:
	SelectorEdgeNoBoundary(const MAP& m): m_map(m) {}
	bool operator()(Dart d) const { return !m_map.isBoundaryEdge(d); }
	FunctorSelect* copy() const { return new SelectorEdgeNoBoundary(m_map);}
};

/**
 * Selector for darts of boundary (of current dimension)
 */
template <typename MAP>
class SelectorDartBoundary : public FunctorSelect
{
public:
protected:
	const MAP& m_map;
public:
	SelectorDartBoundary(const MAP& m): m_map(m) {}
	bool operator()(Dart d) const { return m_map.template isBoundaryMarked<MAP::DIMENSION>(d); }
	FunctorSelect* copy() const { return new SelectorDartBoundary(m_map);}
};

/**
 * Selector for darts not of boundary (of current dimension)
 */

template <typename MAP>
class SelectorDartNoBoundary : public FunctorSelect
{
public:
protected:
	const MAP& m_map;
public:
	SelectorDartNoBoundary(const MAP& m): m_map(m) {}
	bool operator()(Dart d) const { return !m_map.template isBoundaryMarked<MAP::DIMENSION>(d); }
	FunctorSelect* copy() const { return new SelectorDartNoBoundary(m_map);}
};

template <typename MAP>
class SelectorLevel : public FunctorSelect
{
public:
protected:
	const MAP& m_map;
	unsigned int m_level;
public:
	SelectorLevel(const MAP& m, unsigned int l): m_map(m), m_level(l) {}
	bool operator()(Dart d) const { return m_map.getDartLevel(d) == m_level; }
	FunctorSelect* copy() const { return new SelectorLevel(m_map, m_level);}
};

template <typename MAP>
class SelectorEdgeLevel : public FunctorSelect
{
public:
protected:
	const MAP& m_map;
	unsigned int m_level;
public:
	SelectorEdgeLevel(const MAP& m, unsigned int l): m_map(m), m_level(l) {}
	bool operator()(Dart d) const { return (m_map.getDartLevel(d) == m_level) && (m_map.getDartLevel(m_map.phi2(d)) == m_level); }
	FunctorSelect* copy() const { return new SelectorEdgeLevel(m_map, m_level);}
};



// Multiple Functor: to apply several Functors in turn to a dart
/********************************************************/

class FunctorDoubleFunctor : public FunctorType
{
protected:
	FunctorType& m_fonct1;
	FunctorType& m_fonct2;
public:
	FunctorDoubleFunctor(FunctorType& f1, FunctorType& f2) : m_fonct1(f1), m_fonct2(f2) {}
	bool operator()(Dart d)
	{
		if (m_fonct1(d)) return true;
		return m_fonct2(d);
	}
};



//
// FOR PARALLEL TRAVERSALS
//

/**
 * Functor class for parallel::foreach_orbit/cell/dart
 * Overload run
 * Overload duplicate if necessary (no sharing of functors)
 */
template<typename MAP>
class FunctorMapThreaded
{
protected:
	MAP& m_map ;

public:
	FunctorMapThreaded(MAP& m): m_map(m) {}

	virtual ~FunctorMapThreaded() {}

	/**
	 * @return a pointer on a copy of the object.
	 */
	virtual FunctorMapThreaded<MAP>* duplicate() const { return NULL; }

	/**
	 * insert your code here:
	 * @param d the dart on which apply functor
	 * @param threadID the id of thread currently running your code
	 */
	virtual void run(Dart d, unsigned int threadID) = 0;
};

/**
 * Functor class for parallel::foreach_attrib
 * Overload run
 * Overload duplicate if necessary (no sharing of functors)
 */
class FunctorAttribThreaded
{
public:
	virtual ~FunctorAttribThreaded() {}

	/**
	 * @return a pointer on a copy of the object.
	 */
	virtual FunctorAttribThreaded* duplicate() const { return NULL;}

	/**
	 * insert your code here:
	 * @param d the dart on which apply functor
	 * @param threadID the id of thread currently running your code
	 */
	virtual void run(unsigned int i, unsigned int threadID) = 0;
};

} //namespace CGoGN

#endif
