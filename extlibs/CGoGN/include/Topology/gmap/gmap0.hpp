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

inline void GMap0::init()
{
	m_beta0 = addRelation("beta0") ;
}

inline GMap0::GMap0() : AttribMap()
{
	init() ;
}

inline std::string GMap0::mapTypeName() const
{
	return "GMap0";
}

inline unsigned int GMap0::dimension() const
{
	return 0;
}

inline void GMap0::clear(bool removeAttrib)
{
	AttribMap::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

inline void GMap0::update_topo_shortcuts()
{
	GenericMap::update_topo_shortcuts();
	m_beta0 = getRelation("beta0");
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

inline Dart GMap0::newDart()
{
	Dart d = GenericMap::newDart() ;
	(*m_beta0)[d.index] = d ;
	return d ;
}

inline Dart GMap0::beta0(Dart d) const
{
	return (*m_beta0)[d.index] ;
}

inline void GMap0::beta0sew(Dart d, Dart e)
{
	assert((*m_beta0)[d.index] == d) ;
	assert((*m_beta0)[e.index] == e) ;
	(*m_beta0)[d.index] = e ;
	(*m_beta0)[e.index] = d ;
}

inline void GMap0::beta0unsew(Dart d)
{
	Dart e = (*m_beta0)[d.index] ;
	(*m_beta0)[d.index] = d ;
	(*m_beta0)[e.index] = e ;
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

inline bool GMap0::foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int /*thread*/) const
{
	return f(d) ;
}

//inline bool GMap0::foreach_dart_of_vertex(Dart d, FunctorConstType& f, unsigned int /*thread*/) const
//{
//	return f(d) ;
//}

inline bool GMap0::foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int /*thread*/) const
{
	if (f(d)) return true;
	Dart d1 = beta0(d);
	if (d1 != d) return f(d1);
	return false;
}


//inline bool GMap0::foreach_dart_of_edge(Dart d, FunctorConstType& f, unsigned int /*thread*/)
//{
//	if (f(d)) return true;
//	Dart d1 = beta0(d);
//	if (d1 != d) return f(d1);
//	return false;
//}



} // namespace CGoGN
