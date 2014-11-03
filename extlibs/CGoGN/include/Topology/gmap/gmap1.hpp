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
inline void GMap1<MAP_IMPL>::init()
{
	MAP_IMPL::addInvolution() ;
}

template <typename MAP_IMPL>
inline GMap1<MAP_IMPL>::GMap1() : GMap0<MAP_IMPL>()
{
	init() ;
}

template <typename MAP_IMPL>
inline std::string GMap1<MAP_IMPL>::mapTypeName() const
{
	return "GMap1";
}

template <typename MAP_IMPL>
inline unsigned int GMap1<MAP_IMPL>::dimension() const
{
	return 1;
}

template <typename MAP_IMPL>
inline void GMap1<MAP_IMPL>::clear(bool removeAttrib)
{
	ParentMap::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

template <typename MAP_IMPL>
inline unsigned int GMap1<MAP_IMPL>::getNbInvolutions() const
{
	return 1 + ParentMap::getNbInvolutions();
}

template <typename MAP_IMPL>
inline unsigned int GMap1<MAP_IMPL>::getNbPermutations() const
{
	return ParentMap::getNbPermutations();
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

template <typename MAP_IMPL>
inline Dart GMap1<MAP_IMPL>::beta1(Dart d) const
{
	return MAP_IMPL::template getInvolution<1>(d);
}

template <typename MAP_IMPL>
template <int N>
inline Dart GMap1<MAP_IMPL>::beta(const Dart d) const
{
	assert( (N > 0) || !"negative parameters not allowed in template multi-beta");
	if (N<10)
	{
		switch(N)
		{
			case 0 : return this->beta0(d) ;
			case 1 : return beta1(d) ;
			default : assert(!"Wrong multi-beta relation value") ;
		}
	}
	switch(N%10)
	{
		case 0 : return beta0(beta<N/10>(d)) ;
		case 1 : return beta1(beta<N/10>(d)) ;
		default : assert(!"Wrong multi-beta relation value") ;
	}
}

template <typename MAP_IMPL>
inline Dart GMap1<MAP_IMPL>::phi1(Dart d) const
{
	return beta1(this->beta0(d)) ;
}

template <typename MAP_IMPL>
inline Dart GMap1<MAP_IMPL>::phi_1(Dart d) const
{
	return this->beta0(beta1(d)) ;
}

template <typename MAP_IMPL>
template <int N>
inline Dart GMap1<MAP_IMPL>::phi(Dart d) const
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
inline Dart GMap1<MAP_IMPL>::alpha1(Dart d) const
{
	return beta1(this->beta0(d)) ;
}

template <typename MAP_IMPL>
inline Dart GMap1<MAP_IMPL>::alpha_1(Dart d) const
{
	return beta0(beta1(d)) ;
}

template <typename MAP_IMPL>
inline void GMap1<MAP_IMPL>::beta1sew(Dart d, Dart e)
{
	MAP_IMPL::template involutionSew<1>(d,e);
}

template <typename MAP_IMPL>
inline void GMap1<MAP_IMPL>::beta1unsew(Dart d)
{
	MAP_IMPL::template involutionUnsew<1>(d);
}

/*! @name Constructors and Destructors
 *  To generate or delete faces in a 1-G-map
 *************************************************************************/

template <typename MAP_IMPL>
Dart GMap1<MAP_IMPL>::newCycle(unsigned int nbEdges)
{
	assert(nbEdges > 0 || !"Cannot create a face with no edge") ;

	Dart d0 = ParentMap::newEdge();	// create the first edge
	Dart dp = this->beta0(d0);		// store an extremity
	for (unsigned int i = 1; i < nbEdges; ++i)
	{
		Dart di = ParentMap::newEdge();	// create the next edge
		beta1sew(dp,di);
		dp = this->beta0(di);	// change the preceding
	}
	beta1sew(dp,d0);	// sew the last with the first
	return d0;
}

template <typename MAP_IMPL>
void GMap1<MAP_IMPL>::deleteCycle(Dart d)
{
	Dart e = phi1(d);
	while (e != d)
	{
		Dart f = phi1(e);
		this->deleteEdge(e);
		e = f;
	}
	this->deleteEdge(d);
}

/*! @name Topological Operators
 *  Topological operations on 1-G-maps
 *************************************************************************/

template <typename MAP_IMPL>
inline Dart GMap1<MAP_IMPL>::cutEdge(Dart d)
{
	Dart dd = this->beta0(d) ;
	Dart e = this->newDart();
	Dart f = this->newDart();
	beta1sew(e, f) ;
	this->beta0unsew(d) ;
	this->beta0sew(e, d) ;
	this->beta0sew(f, dd) ;

	if (this->template isBoundaryMarked<2>(d))
	{
		this->template boundaryMark<2>(e);
		this->template boundaryMark<2>(f);
	}

	if (this->template isBoundaryMarked<3>(d))
	{
		this->template boundaryMark<3>(e);
		this->template boundaryMark<3>(f);
	}

	return f ;
}

template <typename MAP_IMPL>
inline void GMap1<MAP_IMPL>::uncutEdge(Dart d)
{
	Dart d0 = this->beta0(d) ;
	Dart d1 = phi1(d) ;
	Dart d10 = this->beta0(d1) ;
	this->beta0unsew(d) ;
	this->beta0unsew(d10) ;
	this->beta0sew(d, d10) ;
	this->deleteDart(d0) ;
	this->deleteDart(d1) ;
}

template <typename MAP_IMPL>
inline void GMap1<MAP_IMPL>::collapseEdge(Dart d)
{
	Dart d1 = beta1(d) ;
	Dart dd = this->beta0(d) ;
	Dart dd1 = beta1(dd) ;
	beta1unsew(d) ;
	beta1unsew(dd) ;
	beta1sew(d1, dd1) ;
	this->deleteEdge(d) ;
}

template <typename MAP_IMPL>
inline void GMap1<MAP_IMPL>::splitCycle(Dart d, Dart e)
{
	assert(d != e && sameCycle(d, e)) ;

	if(!sameOrientedCycle(d, e))
		e = beta1(e) ;

	Dart d1 = beta1(d) ;
	Dart e1 = beta1(e) ;
	beta1unsew(d) ;
	beta1unsew(e) ;
	beta1sew(d, e1) ;
	beta1sew(e, d1) ;
}

template <typename MAP_IMPL>
inline void GMap1<MAP_IMPL>::mergeCycles(Dart d, Dart e)
{
	assert(!sameCycle(d, e)) ;

	Dart d1 = beta1(d) ;
	Dart e1 = beta1(e) ;
	beta1unsew(d) ;
	beta1unsew(e) ;
	beta1sew(d, e1) ;
	beta1sew(e, d1) ;
}

template <typename MAP_IMPL>
inline void GMap1<MAP_IMPL>::linkCycles(Dart d, Dart e)
{
	assert(d != e && !sameCycle(d, e)) ;
	Dart d1 = beta1(d) ;
	Dart e1 = beta1(e) ;
	Dart dd = this->newEdge() ;
	Dart ee = this->newEdge() ;
	beta1unsew(d) ;
	beta1unsew(e) ;
	beta1sew(d, dd) ;
	beta1sew(e1, this->beta0(dd)) ;
	beta1sew(e, ee) ;
	beta1sew(d1, this->beta0(ee)) ;
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

template <typename MAP_IMPL>
inline bool GMap1<MAP_IMPL>::sameOrientedCycle(Dart d, Dart e) const
{
	Dart it = d ;
	do
	{
		if (it == e)
			return true ;
		it = phi1(it) ;
	} while (it != d) ;
	return false ;
}

template <typename MAP_IMPL>
inline bool GMap1<MAP_IMPL>::sameCycle(Dart d, Dart e) const
{
	Dart it = d ;
	do
	{
		if (it == e)
			return true ;
		it = this->beta0(it);
		if (it == e)
			return true ;
		it = beta1(it) ;
	} while (it != d) ;
	return false ;
}

template <typename MAP_IMPL>
inline unsigned int GMap1<MAP_IMPL>::cycleDegree(Dart d) const
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
inline int GMap1<MAP_IMPL>::checkCycleDegree(Dart d, unsigned int degree) const
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

template <typename MAP_IMPL>
inline bool GMap1<MAP_IMPL>::isCycleTriangle(Dart d) const
{
	return (phi1(d) != d) && (phi1(phi1(phi1(d))) == d) ;
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

template <typename MAP_IMPL>
template <unsigned int ORBIT, typename FUNC>
void GMap1<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f, unsigned int thread) const
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
//void GMap1<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC& f, unsigned int thread) const
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
inline void GMap1<MAP_IMPL>::foreach_dart_of_vertex(Dart d, FUNC& f, unsigned int /*thread*/) const
{
	f(d);
	Dart d1 = beta1(d);
	if (d1 != d)
		f(d1);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap1<MAP_IMPL>::foreach_dart_of_edge(Dart d, FUNC& f, unsigned int /*thread*/) const
{
	f(d);
	Dart d1 = this->beta0(d);
	if (d1 != d)
		f(d1);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap1<MAP_IMPL>::foreach_dart_of_oriented_cc(Dart d, FUNC& f, unsigned int /*thread*/) const
{
	Dart it = d ;
	do
	{
		f(it);
		it = phi1(it) ;
	} while (it != d) ;
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap1<MAP_IMPL>::foreach_dart_of_cc(Dart d, FUNC& f, unsigned int thread) const
{
	GMap1<MAP_IMPL>::foreach_dart_of_oriented_cc(d, f, thread);
	GMap1<MAP_IMPL>::foreach_dart_of_oriented_cc(this->beta0(d), f, thread);
}

} // namespace CGoGN
