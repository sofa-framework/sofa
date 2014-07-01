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

template <typename MAP_IMPL>
inline void GMap0<MAP_IMPL>::init()
{
	MAP_IMPL::addInvolution() ;
}

template <typename MAP_IMPL>
inline GMap0<MAP_IMPL>::GMap0() : MapCommon<MAP_IMPL>()
{
	init() ;
}

template <typename MAP_IMPL>
inline std::string GMap0<MAP_IMPL>::mapTypeName() const
{
	return "GMap0";
}

template <typename MAP_IMPL>
inline unsigned int GMap0<MAP_IMPL>::dimension() const
{
	return 0;
}

template <typename MAP_IMPL>
inline void GMap0<MAP_IMPL>::clear(bool removeAttrib)
{
	MAP_IMPL::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

template <typename MAP_IMPL>
inline unsigned int GMap0<MAP_IMPL>::getNbInvolutions() const
{
	return 1;
}

template <typename MAP_IMPL>
inline unsigned int GMap0<MAP_IMPL>::getNbPermutations() const
{
	return 0;
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

template <typename MAP_IMPL>
inline Dart GMap0<MAP_IMPL>::beta0(Dart d) const
{
	return MAP_IMPL::template getInvolution<0>(d);
}

template <typename MAP_IMPL>
inline void GMap0<MAP_IMPL>::beta0sew(Dart d, Dart e)
{
	MAP_IMPL::template involutionSew<0>(d,e);
}

template <typename MAP_IMPL>
inline void GMap0<MAP_IMPL>::beta0unsew(Dart d)
{
	MAP_IMPL::template involutionUnsew<0>(d);
}

/*! @name Constructors and Destructors
 *  To generate or delete edges in a 0-G-map
 *************************************************************************/

template <typename MAP_IMPL>
Dart GMap0<MAP_IMPL>::newEdge()
{
	Dart d1 = this->newDart();
	Dart d2 = this->newDart();
	beta0sew(d1,d2);
	return d1;
}

template <typename MAP_IMPL>
void GMap0<MAP_IMPL>::deleteEdge(Dart d)
{
	this->deleteDart(beta0(d));
	this->deleteDart(d);
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

template <typename MAP_IMPL>
template <unsigned int ORBIT, typename FUNC>
void GMap0<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f, unsigned int thread) const
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
//void GMap0<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC& f, unsigned int thread) const
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
inline void GMap0<MAP_IMPL>::foreach_dart_of_vertex(Dart d, FUNC& f, unsigned int /*thread*/) const
{
	f(d) ;
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap0<MAP_IMPL>::foreach_dart_of_edge(Dart d, FUNC& f, unsigned int /*thread*/) const
{
	f(d);
	Dart d1 = beta0(d);
	if (d1 != d)
		f(d1);
}

} // namespace CGoGN
