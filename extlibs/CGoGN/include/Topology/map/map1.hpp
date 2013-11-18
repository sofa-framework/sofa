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

namespace CGoGN
{

/// INLINE FUNCTIONS

inline void Map1::init()
{
	m_phi1 = addRelation("phi1") ;
	m_phi_1 = addRelation("phi_1") ;
}

inline Map1::Map1() : AttribMap()
{
	init() ;
}

inline std::string Map1::mapTypeName() const
{
	return "Map1" ;
}

inline unsigned int Map1::dimension() const
{
	return 1 ;
}

inline void Map1::clear(bool removeAttrib)
{
	AttribMap::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

inline void Map1::update_topo_shortcuts()
{
	GenericMap::update_topo_shortcuts();
	m_phi1 = getRelation("phi1");
	m_phi_1 = getRelation("phi_1");
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

inline Dart Map1::newDart()
{
	Dart d = GenericMap::newDart() ;
	unsigned int d_index = dartIndex(d) ;
	(*m_phi1)[d_index] = d ;
	(*m_phi_1)[d_index] = d ;
	if(m_isMultiRes)
	{
		pushLevel() ;
		for(unsigned int i = m_mrCurrentLevel + 1;  i < m_mrDarts.size(); ++i)
		{
			setCurrentLevel(i) ;
			unsigned int d_index = dartIndex(d) ;
			(*m_phi1)[d_index] = d ;
			(*m_phi_1)[d_index] = d ;
		}
		popLevel() ;
	}
	return d ;
}

inline Dart Map1::phi1(Dart d)
{
//	unsigned int d_index = dartIndex(d);
	return (*m_phi1)[dartIndex(d)] ;
}

inline Dart Map1::phi_1(Dart d)
{
//	unsigned int d_index = dartIndex(d);
	return (*m_phi_1)[dartIndex(d)] ;
}

template <int N>
inline Dart Map1::phi(Dart d)
{
	assert((N > 0) || !"negative parameters not allowed in template multi-phi");
	if (N < 10)
	{
		switch(N)
		{
			case 1 : return phi1(d) ;
			default : assert(!"Wrong multi-phi relation value") ; return d ;
		}
	}
	switch(N%10)
	{
		case 1 : return phi1(phi<N/10>(d)) ;
		default : assert(!"Wrong multi-phi relation value") ; return d ;
	}
}

inline Dart Map1::alpha1(Dart d)
{
	return phi1(d) ;
}

inline Dart Map1::alpha_1(Dart d)
{
	return phi_1(d) ;
}

inline void Map1::phi1sew(Dart d, Dart e)
{
	unsigned int d_index = dartIndex(d);
	unsigned int e_index = dartIndex(e);
	Dart f = (*m_phi1)[d_index] ;
	Dart g = (*m_phi1)[e_index] ;
	(*m_phi1)[d_index] = g ;
	(*m_phi1)[e_index] = f ;
	(*m_phi_1)[dartIndex(g)] = d ;
	(*m_phi_1)[dartIndex(f)] = e ;
}

inline void Map1::phi1unsew(Dart d)
{
	unsigned int d_index = dartIndex(d);
	Dart e = (*m_phi1)[d_index] ;
	unsigned int e_index = dartIndex(e);
	Dart f = (*m_phi1)[e_index] ;
	(*m_phi1)[d_index] = f ;
	(*m_phi1)[e_index] = e ;
	(*m_phi_1)[dartIndex(f)] = d ;
	(*m_phi_1)[e_index] = e ;
}

/*! @name Topological Operators
 *  Topological operations on 1-maps
 *************************************************************************/

inline Dart Map1::cutEdge(Dart d)
{
	Dart e = newDart() ;	// Create a new dart
	phi1sew(d, e) ;			// Insert dart e between d and phi1(d)

	if (isBoundaryMarked2(d))
		boundaryMark2(e);

	if (isBoundaryMarked3(d))
		boundaryMark3(e);

	return e ;
}

inline void Map1::uncutEdge(Dart d)
{
	Dart d1 = phi1(d) ;
	phi1unsew(d) ;		// Dart d is linked to the successor of its successor
	deleteDart(d1) ;	// Dart d1 is erased
}

inline void Map1::collapseEdge(Dart d)
{
	phi1unsew(phi_1(d)) ;	// Dart before d is linked to its successor
	deleteDart(d) ;			// Dart d is erased
}

inline void Map1::splitCycle(Dart d, Dart e)
{
	assert(d != e && sameCycle(d, e)) ;
	phi1sew(phi_1(d), phi_1(e)) ;
}

inline void Map1::mergeCycles(Dart d, Dart e)
{
	assert(!sameCycle(d, e)) ;
	phi1sew(phi_1(d), phi_1(e)) ;
}

inline void Map1::linkCycles(Dart d, Dart e)
{
	assert(d != e && !sameCycle(d, e)) ;
	Map1::cutEdge(phi_1(d));		// cut the edge before d (insert a new dart before d)
	Map1::cutEdge(phi_1(e));		// cut the edge before e (insert a new dart before e)
	phi1sew(phi_1(d), phi_1(e)) ;	// phi1sew between the 2 new inserted darts
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

inline bool Map1::sameCycle(Dart d, Dart e)
{
	Dart it = d ;
	do
	{
		if(it == e)
			return true ;
		it = phi1(it) ;
	} while(it != d) ;
	return false ;
}

inline unsigned int Map1::cycleDegree(Dart d)
{
	unsigned int count = 0 ;
	Dart it = d ;
	do
	{
		++count ;
		it = phi1(it) ;
	} while (it != d) ;
	return count ;
}


inline int Map1::checkCycleDegree(Dart d, unsigned int degree)
{
	unsigned int count = 0 ;
	Dart it = d ;
	do
	{
		++count ;
		it = phi1(it) ;
	} while ((count<=degree) && (it != d)) ;

	return count-degree;
}


inline bool Map1::isCycleTriangle(Dart d)
{
	return (phi1(d) != d) && (phi1(phi1(phi1(d))) == d) ;
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

inline bool Map1::foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	return f(d) ;
}

inline bool Map1::foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	return f(d) ;
}

inline bool Map1::foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	Dart it = d ;
	do
	{
		if (f(it))
			return true ;
		it = phi1(it) ;
	} while (it != d) ;
	return false ;
}


} // namespace CGoGN
