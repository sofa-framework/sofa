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

template <typename MAP_IMPL>
inline void Map1<MAP_IMPL>::init()
{
	MAP_IMPL::addPermutation() ;
}

template <typename MAP_IMPL>
inline Map1<MAP_IMPL>::Map1() : MapCommon<MAP_IMPL>()
{
	init() ;
}

template <typename MAP_IMPL>
inline std::string Map1<MAP_IMPL>::mapTypeName() const
{
	return "Map1" ;
}

template <typename MAP_IMPL>
inline unsigned int Map1<MAP_IMPL>::dimension() const
{
	return 1 ;
}

template <typename MAP_IMPL>
inline void Map1<MAP_IMPL>::clear(bool removeAttrib)
{
	ParentMap::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

template <typename MAP_IMPL>
inline unsigned int Map1<MAP_IMPL>::getNbInvolutions() const
{
	return 0;
}

template <typename MAP_IMPL>
inline unsigned int Map1<MAP_IMPL>::getNbPermutations() const
{
	return 1;
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

template <typename MAP_IMPL>
inline Dart Map1<MAP_IMPL>::phi1(Dart d) const
{
	return MAP_IMPL::template getPermutation<0>(d);
}

template <typename MAP_IMPL>
inline Dart Map1<MAP_IMPL>::phi_1(Dart d) const
{
	return MAP_IMPL::template getPermutationInv<0>(d);
}

template <typename MAP_IMPL>
template <int N>
inline Dart Map1<MAP_IMPL>::phi(Dart d) const
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

template <typename MAP_IMPL>
inline Dart Map1<MAP_IMPL>::alpha1(Dart d) const
{
	return phi1(d) ;
}

template <typename MAP_IMPL>
inline Dart Map1<MAP_IMPL>::alpha_1(Dart d) const
{
	return phi_1(d) ;
}

template <typename MAP_IMPL>
inline void Map1<MAP_IMPL>::phi1sew(Dart d, Dart e)
{
	MAP_IMPL::template permutationSew<0>(d,e);
}

template <typename MAP_IMPL>
inline void Map1<MAP_IMPL>::phi1unsew(Dart d)
{
	MAP_IMPL::template permutationUnsew<0>(d);
}

/*! @name Generator and Deletor
 *  To generate or delete faces in a 1-map
 *************************************************************************/

template <typename MAP_IMPL>
Dart Map1<MAP_IMPL>::newCycle(unsigned int nbEdges)
{
	assert(nbEdges > 0 || !"Cannot create a face with no edge") ;
	Dart d = this->newDart() ;	// Create the first edge
	for (unsigned int i = 1 ; i < nbEdges ; ++i)
		Map1<MAP_IMPL>::cutEdge(d) ;		// Subdivide nbEdges-1 times this edge
	return d ;
}

template <typename MAP_IMPL>
void Map1<MAP_IMPL>::deleteCycle(Dart d)
{
	Dart e = phi1(d) ;
	while (e != d)
	{
		Dart f = phi1(e) ;
		this->deleteDart(e) ;
		e = f ;
	}
	this->deleteDart(d) ;
}

/*! @name Topological Operators
 *  Topological operations on 1-maps
 *************************************************************************/

template <typename MAP_IMPL>
inline Dart Map1<MAP_IMPL>::cutEdge(Dart d)
{
	Dart e = this->newDart() ;	// Create a new dart
	phi1sew(d, e) ;				// Insert dart e between d and phi1(d)

	if (this->template isBoundaryMarked<2>(d))
		this->template boundaryMark<2>(e);

	if (this->template isBoundaryMarked<3>(d))
		this->template boundaryMark<3>(e);

	return e ;
}

template <typename MAP_IMPL>
inline void Map1<MAP_IMPL>::uncutEdge(Dart d)
{
	Dart d1 = phi1(d) ;
	phi1unsew(d) ;			// Dart d is linked to the successor of its successor
	this->deleteDart(d1) ;	// Dart d1 is erased
}

template <typename MAP_IMPL>
inline void Map1<MAP_IMPL>::collapseEdge(Dart d)
{
	phi1unsew(phi_1(d)) ;	// Dart before d is linked to its successor
	this->deleteDart(d) ;	// Dart d is erased
}

template <typename MAP_IMPL>
inline void Map1<MAP_IMPL>::splitCycle(Dart d, Dart e)
{
	assert(d != e && sameCycle(d, e)) ;
	phi1sew(phi_1(d), phi_1(e)) ;
}

template <typename MAP_IMPL>
inline void Map1<MAP_IMPL>::mergeCycles(Dart d, Dart e)
{
	assert(!sameCycle(d, e)) ;
	phi1sew(phi_1(d), phi_1(e)) ;
}

template <typename MAP_IMPL>
inline void Map1<MAP_IMPL>::linkCycles(Dart d, Dart e)
{
	assert(d != e && !sameCycle(d, e)) ;
	Map1<MAP_IMPL>::cutEdge(phi_1(d));		// cut the edge before d (insert a new dart before d)
	Map1<MAP_IMPL>::cutEdge(phi_1(e));		// cut the edge before e (insert a new dart before e)
	phi1sew(phi_1(d), phi_1(e)) ;	// phi1sew between the 2 new inserted darts
}

template <typename MAP_IMPL>
void Map1<MAP_IMPL>::reverseCycle(Dart d)
{
	Dart e = phi1(d) ;			// Dart e is the first edge of the new face
	if (e == d) return ;		// Only one edge: nothing to do
	if (phi1(e) == d) return ;	// Only two edges: nothing to do

	phi1unsew(d) ;				// Detach e from the face of d

	Dart dNext = phi1(d) ;		// While the face of d contains more than two edges
	while (dNext != d)
	{
		phi1unsew(d) ;			// Unsew the edge after d
		phi1sew(e, dNext) ;		// Sew it after e (thus in reverse order)
		dNext = phi1(d) ;
	}
	phi1sew(e, d) ;				// Sew the last edge
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

template <typename MAP_IMPL>
inline bool Map1<MAP_IMPL>::sameCycle(Dart d, Dart e) const
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

template <typename MAP_IMPL>
inline unsigned int Map1<MAP_IMPL>::cycleDegree(Dart d) const
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

template <typename MAP_IMPL>
inline int Map1<MAP_IMPL>::checkCycleDegree(Dart d, unsigned int degree) const
{
	unsigned int count = 0 ;
	Dart it = d ;
	do
	{
		++count ;
		it = phi1(it) ;
	} while ((count <= degree) && (it != d)) ;
	return count-degree;
}

template <typename MAP_IMPL>
inline bool Map1<MAP_IMPL>::isCycleTriangle(Dart d) const
{
	return (phi1(d) != d) && (phi1(phi1(phi1(d))) == d) ;
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

template <typename MAP_IMPL>
template <unsigned int ORBIT, typename FUNC>
void Map1<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, const FUNC& f, unsigned int thread) const
{
	switch(ORBIT)
	{
		case DART:		f(c); break;
		case VERTEX: 	foreach_dart_of_vertex(c, f, thread); break;
		case EDGE: 		foreach_dart_of_edge(c, f, thread); break;
		default: 		assert(!"Cells of this dimension are not handled"); break;
	}
}

//template <typename MAP_IMPL>
//template <unsigned int ORBIT, typename FUNC>
//void Map1<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f, unsigned int thread) const
//{
//	switch(ORBIT)
//	{
//		case DART:		f(c); break;
//		case VERTEX: 	foreach_dart_of_vertex(c, f, thread); break;
//		case EDGE: 		foreach_dart_of_edge(c, f, thread); break;
//		default: 		assert(!"Cells of this dimension are not handled"); break;
//	}
//}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map1<MAP_IMPL>::foreach_dart_of_vertex(Dart d,const  FUNC& f, unsigned int /*thread*/) const
{
	f(d) ;
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map1<MAP_IMPL>::foreach_dart_of_edge(Dart d, const FUNC& f, unsigned int /*thread*/) const
{
	f(d) ;
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map1<MAP_IMPL>::foreach_dart_of_cc(Dart d, const FUNC& f, unsigned int /*thread*/) const
{
	Dart it = d ;
	do
	{
		f(it);
		it = phi1(it) ;
	} while (it != d) ;
}

} // namespace CGoGN
