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

inline void GMap3::init()
{
	m_beta3 = addRelation("beta3") ;
}

inline GMap3::GMap3() : GMap2()
{
	init() ;
}

inline std::string GMap3::mapTypeName() const
{
	return "GMap3";
}

inline unsigned int GMap3::dimension() const
{
	return 3;
}

inline void GMap3::clear(bool removeAttrib)
{
	GMap2::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

inline void GMap3::update_topo_shortcuts()
{
	GMap2::update_topo_shortcuts();
	m_beta3 = getRelation("beta3");
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

inline Dart GMap3::newDart()
{
	Dart d = GMap2::newDart() ;
	(*m_beta3)[d.index] = d ;
	return d ;
}

inline Dart GMap3::beta3(Dart d) const
{
	return (*m_beta3)[d.index] ;
}

template <int N>
inline Dart GMap3::beta(const Dart d) const
{
	assert( (N > 0) || !"negative parameters not allowed in template multi-beta");
	if (N<10)
	{
		switch(N)
		{
		case 0 : return beta0(d) ;
		case 1 : return beta1(d) ;
		case 2 : return beta2(d) ;
		case 3 : return beta2(d) ;
		default : assert(!"Wrong multi-beta relation value") ;
		}
	}
	switch(N%10)
	{
	case 0 : return beta0(beta<N/10>(d)) ;
	case 1 : return beta1(beta<N/10>(d)) ;
	case 2 : return beta2(beta<N/10>(d)) ;
	case 3 : return beta3(beta<N/10>(d)) ;
	default : assert(!"Wrong multi-beta relation value") ;
	}
}

inline Dart GMap3::phi3(Dart d) const
{
	return beta3(beta0(d)) ;
}

template <int N>
inline Dart GMap3::phi(Dart d) const
{
	assert( (N >0) || !"negative parameters not allowed in template multi-phi");
	if (N<10)
	{
		switch(N)
		{
		case 1 : return phi1(d) ;
		case 2 : return phi2(d) ;
		case 3 : return phi3(d) ;
		default : assert(!"Wrong multi-phi relation value") ; return d ;
		}
	}
	switch(N%10)
	{
	case 1 : return phi1(phi<N/10>(d)) ;
	case 2 : return phi2(phi<N/10>(d)) ;
	case 3 : return phi3(phi<N/10>(d)) ;
	default : assert(!"Wrong multi-phi relation value") ; return d ;
	}
}

inline Dart GMap3::alpha0(Dart d) const
{
	return beta3(beta0(d)) ;
}

inline Dart GMap3::alpha1(Dart d) const
{
	return beta3(beta1(d)) ;
}

inline Dart GMap3::alpha2(Dart d) const
{
	return beta3(beta2(d)) ;
}

inline Dart GMap3::alpha_2(Dart d) const
{
	return beta2(beta3(d)) ;
}

inline void GMap3::beta3sew(Dart d, Dart e)
{
	assert((*m_beta3)[d.index] == d) ;
	assert((*m_beta3)[e.index] == e) ;
	(*m_beta3)[d.index] = e ;
	(*m_beta3)[e.index] = d ;
}

inline void GMap3::beta3unsew(Dart d)
{
	Dart e = (*m_beta3)[d.index] ;
	(*m_beta3)[d.index] = d ;
	(*m_beta3)[e.index] = e ;
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

inline bool GMap3::sameFace(Dart d, Dart e) const
{
	return GMap2::sameFace(d, e) || GMap2::sameFace(beta3(d), e) ;
}

inline bool GMap3::isBoundaryFace(Dart d) const
{
	return isBoundaryMarked3(d) || isBoundaryMarked3(beta3(d));
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

inline bool GMap3::foreach_dart_of_face(Dart d, FunctorType& f, unsigned int thread) const
{
	return GMap2::foreach_dart_of_face(d, f, thread) || GMap2::foreach_dart_of_face(beta3(d), f, thread);
}

inline bool GMap3::foreach_dart_of_volume(Dart d, FunctorType& f, unsigned int thread) const
{
	return GMap2::foreach_dart_of_cc(d, f, thread);
}

inline bool GMap3::foreach_dart_of_oriented_volume(Dart d, FunctorType& f, unsigned int thread) const
{
	return GMap2::foreach_dart_of_oriented_cc(d, f, thread);
}

inline bool GMap3::foreach_dart_of_vertex2(Dart d, FunctorType& f, unsigned int thread) const
{
	return GMap2::foreach_dart_of_vertex(d, f, thread);
}

inline bool GMap3::foreach_dart_of_edge2(Dart d, FunctorType& f, unsigned int thread) const
{
	return GMap2::foreach_dart_of_edge(d, f, thread);
}

inline bool GMap3::foreach_dart_of_face2(Dart d, FunctorType& f, unsigned int thread) const
{
	return GMap2::foreach_dart_of_face(d, f, thread);
}

} // namespace CGoGN
