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

inline void GMap1::init()
{
	m_beta1 = addRelation("beta1") ;
}

inline GMap1::GMap1() : GMap0()
{
	init() ;
}

inline std::string GMap1::mapTypeName() const
{
	return "GMap1";
}

inline unsigned int GMap1::dimension() const
{
	return 1;
}

inline void GMap1::clear(bool removeAttrib)
{
	GMap0::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

inline void GMap1::update_topo_shortcuts()
{
	GMap0::update_topo_shortcuts();
	m_beta1 = getRelation("beta1");
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

inline Dart GMap1::newDart()
{
	Dart d = GMap0::newDart() ;
	(*m_beta1)[d.index] = d ;
	return d ;
}

inline Dart GMap1::beta1(Dart d)
{
	return (*m_beta1)[d.index] ;
}

template <int N>
inline Dart GMap1::beta(const Dart d)
{
	assert( (N > 0) || !"negative parameters not allowed in template multi-beta");
	if (N<10)
	{
		switch(N)
		{
		case 0 : return beta0(d) ;
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

inline Dart GMap1::phi1(Dart d)
{
	return beta1(beta0(d)) ;
}

inline Dart GMap1::phi_1(Dart d)
{
	return beta0(beta1(d)) ;
}

template <int N>
inline Dart GMap1::phi(Dart d)
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

inline Dart GMap1::alpha1(Dart d)
{
	return beta1(beta0(d)) ;
}

inline Dart GMap1::alpha_1(Dart d)
{
	return beta0(beta1(d)) ;
}

inline void GMap1::beta1sew(Dart d, Dart e)
{
	assert((*m_beta1)[d.index] == d) ;
	assert((*m_beta1)[e.index] == e) ;
	(*m_beta1)[d.index] = e ;
	(*m_beta1)[e.index] = d ;
}

inline void GMap1::beta1unsew(Dart d)
{
	Dart e = (*m_beta1)[d.index] ;
	(*m_beta1)[d.index] = d ;
	(*m_beta1)[e.index] = e ;
}

/*! @name Topological Operators
 *  Topological operations on 1-G-maps
 *************************************************************************/

inline Dart GMap1::cutEdge(Dart d)
{
	Dart dd = beta0(d) ;
	Dart e = newDart();
	Dart f = newDart();
	beta1sew(e, f) ;
	beta0unsew(d) ;
	beta0sew(e, d) ;
	beta0sew(f, dd) ;

	if (isBoundaryMarked2(d))
	{
		boundaryMark2(e);
		boundaryMark2(f);
	}

	if (isBoundaryMarked3(d))
	{
		boundaryMark3(e);
		boundaryMark3(f);
	}

	return f ;
}

inline void GMap1::uncutEdge(Dart d)
{
	Dart d0 = beta0(d) ;
	Dart d1 = phi1(d) ;
	Dart d10 = beta0(d1) ;
	beta0unsew(d) ;
	beta0unsew(d10) ;
	beta0sew(d, d10) ;
	deleteDart(d0) ;
	deleteDart(d1) ;
}

inline void GMap1::collapseEdge(Dart d)
{
	Dart d1 = beta1(d) ;
	Dart dd = beta0(d) ;
	Dart dd1 = beta1(dd) ;
	beta1unsew(d) ;
	beta1unsew(dd) ;
	beta1sew(d1, dd1) ;
	deleteEdge(d) ;
}

inline void GMap1::splitCycle(Dart d, Dart e)
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

inline void GMap1::mergeCycles(Dart d, Dart e)
{
	assert(!sameCycle(d, e)) ;

	Dart d1 = beta1(d) ;
	Dart e1 = beta1(e) ;
	beta1unsew(d) ;
	beta1unsew(e) ;
	beta1sew(d, e1) ;
	beta1sew(e, d1) ;
}

inline void GMap1::linkCycles(Dart d, Dart e)
{
	assert(d != e && !sameCycle(d, e)) ;
	Dart d1 = beta1(d) ;
	Dart e1 = beta1(e) ;
	Dart dd = newEdge() ;
	Dart ee = newEdge() ;
	beta1unsew(d) ;
	beta1unsew(e) ;
	beta1sew(d, dd) ;
	beta1sew(e1, beta0(dd)) ;
	beta1sew(e, ee) ;
	beta1sew(d1, beta0(ee)) ;
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

inline bool GMap1::sameOrientedCycle(Dart d, Dart e)
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

inline bool GMap1::sameCycle(Dart d, Dart e)
{
	Dart it = d ;
	do
	{
		if (it == e)
			return true ;
		it = beta0(it);
		if (it == e)
			return true ;
		it = beta1(it) ;
	} while (it != d) ;
	return false ;
}

inline unsigned int GMap1::cycleDegree(Dart d)
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

inline int GMap1::checkCycleDegree(Dart d, unsigned int degree)
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


inline bool GMap1::isCycleTriangle(Dart d)
{
	return (phi1(d) != d) && (phi1(phi1(phi1(d))) == d) ;
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

inline bool GMap1::foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	if (f(d)) return true;
	Dart d1 = beta1(d);
	if (d1 != d) return f(d1);
	return false;
}

inline bool GMap1::foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	if (f(d)) return true;
	Dart d1 = beta0(d);
	if (d1 != d) return f(d1);
	return false;
}

inline bool GMap1::foreach_dart_of_oriented_cc(Dart d, FunctorType& f, unsigned int /*thread*/)
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

inline bool GMap1::foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread)
{
	return GMap1::foreach_dart_of_oriented_cc(d, f, thread) || GMap1::foreach_dart_of_oriented_cc(beta0(d), f, thread) ;

//	Dart it = d ;
//	do
//	{
//		if (f(it))
//			return true ;
//		it = beta0(it);
//		if (f(it))
//			return true ;
//		it = beta1(it) ;
//	} while (it != d) ;
//	return false ;
}

} // namespace CGoGN
