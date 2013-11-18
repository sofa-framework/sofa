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

inline void GMap2::init()
{
	m_beta2 = addRelation("beta2") ;
}

inline GMap2::GMap2() : GMap1()
{
	init() ;
}

inline std::string GMap2::mapTypeName() const
{
	return "GMap2";
}

inline unsigned int GMap2::dimension() const
{
	return 2;
}

inline void GMap2::clear(bool removeAttrib)
{
	GMap1::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

inline void GMap2::update_topo_shortcuts()
{
	GMap1::update_topo_shortcuts();
	m_beta2 = getRelation("beta2");
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

inline Dart GMap2::newDart()
{
	Dart d = GMap1::newDart() ;
	(*m_beta2)[d.index] = d ;
	return d ;
}

inline Dart GMap2::beta2(Dart d)
{
	return (*m_beta2)[d.index] ;
}

template <int N>
inline Dart GMap2::beta(const Dart d)
{
	assert( (N > 0) || !"negative parameters not allowed in template multi-beta");
	if (N<10)
	{
		switch(N)
		{
		case 0 : return beta0(d) ;
		case 1 : return beta1(d) ;
		case 2 : return beta2(d) ;
		default : assert(!"Wrong multi-beta relation value") ;
		}
	}
	switch(N%10)
	{
	case 0 : return beta0(beta<N/10>(d)) ;
	case 1 : return beta1(beta<N/10>(d)) ;
	case 2 : return beta2(beta<N/10>(d)) ;
	default : assert(!"Wrong multi-beta relation value") ;
	}
}

inline Dart GMap2::phi2(Dart d)
{
	return beta2(beta0(d)) ;
}

template <int N>
inline Dart GMap2::phi(Dart d)
{
	assert( (N >0) || !"negative parameters not allowed in template multi-phi");
	if (N<10)
	{
		switch(N)
		{
		case 1 : return phi1(d) ;
		case 2 : return phi2(d) ;
		default : assert(!"Wrong multi-phi relation value") ; return d ;
		}
	}
	switch(N%10)
	{
	case 1 : return phi1(phi<N/10>(d)) ;
	case 2 : return phi2(phi<N/10>(d)) ;
	default : assert(!"Wrong multi-phi relation value") ; return d ;
	}
}

inline Dart GMap2::alpha0(Dart d)
{
	return beta2(beta0(d)) ;
}

inline Dart GMap2::alpha1(Dart d)
{
	return beta2(beta1(d)) ;
}

inline Dart GMap2::alpha_1(Dart d)
{
	return beta1(beta2(d)) ;
}

inline void GMap2::beta2sew(Dart d, Dart e)
{
	assert((*m_beta2)[d.index] == d) ;
	assert((*m_beta2)[e.index] == e) ;
	(*m_beta2)[d.index] = e ;
	(*m_beta2)[e.index] = d ;
}

inline void GMap2::beta2unsew(Dart d)
{
	Dart e = (*m_beta2)[d.index] ;
	(*m_beta2)[d.index] = d ;
	(*m_beta2)[e.index] = e ;
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

inline bool GMap2::sameVertex(Dart d, Dart e)
{
	return sameOrientedVertex(d, e) || sameOrientedVertex(beta2(d), e) ;
}

inline bool GMap2::sameEdge(Dart d, Dart e)
{
	return d == e || beta2(d) == e || beta0(d) == e || beta2(beta0(d)) == e ;
}

inline bool GMap2::isBoundaryEdge(Dart d)
{
	return isBoundaryMarked2(d) || isBoundaryMarked2(beta2(d));
}

inline bool GMap2::sameOrientedFace(Dart d, Dart e)
{
	return GMap1::sameOrientedCycle(d, e) ;
}

inline bool GMap2::sameFace(Dart d, Dart e)
{
	return GMap1::sameCycle(d, e) ;
}

inline unsigned int GMap2::faceDegree(Dart d)
{
	return GMap1::cycleDegree(d) ;
}

inline int GMap2::checkFaceDegree(Dart d, unsigned int le)
{
	return GMap1::checkCycleDegree(d,le) ;
}


inline bool GMap2::sameVolume(Dart d, Dart e)
{
	return sameOrientedVolume(d, e) || sameOrientedVolume(beta2(d), e) ;
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

inline bool GMap2::foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int thread)
{
	return GMap2::foreach_dart_of_oriented_vertex(d, f, thread) || GMap2::foreach_dart_of_oriented_vertex(beta1(d), f, thread) ;
}

inline bool GMap2::foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread)
{
	return GMap2::foreach_dart_of_oriented_cc(d, f, thread) || GMap2::foreach_dart_of_oriented_cc(beta0(d), f, thread) ;
}

inline bool GMap2::foreach_dart_of_face(Dart d, FunctorType& f, unsigned int thread)
{
	return GMap1::foreach_dart_of_cc(d, f, thread);
}

inline bool GMap2::foreach_dart_of_oriented_face(Dart d, FunctorType& f, unsigned int thread)
{
	return GMap1::foreach_dart_of_oriented_cc(d, f, thread);
}

inline bool GMap2::foreach_dart_of_vertex1(Dart d, FunctorType& f, unsigned int thread)
{
	return GMap1::foreach_dart_of_vertex(d,f,thread);
}

inline bool GMap2::foreach_dart_of_edge1(Dart d, FunctorType& f, unsigned int thread)
{
	return GMap1::foreach_dart_of_edge(d,f,thread);
}

} // namespace CGoGN
