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

#ifndef __XMAP_H__
#define __XMAP_H__

#include "Topology/map/map3.h"
#include "Topology/generic/functor.h"

namespace CGoGN
{

template <typename DART>
class tXMap : public tMap3<DART>
{
public:
	typedef typename tMap3<DART>::Dart Dart;

	/**
	* Apply a functor on each dart of a vertex
	* @param d a dart of the vertex
	* @param fonct the functor
	*/
	bool foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int thread=0);

	/**
	* Apply a functor on each dart of an edge
	* @param d a dart of the oriented edge
	* @param fonct the functor
	*/
	bool foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int thread=0);

	bool foreach_dart_of_open_edge(Dart d, FunctorType& f, unsigned int thread=0);

	/**
	* Apply a functor on each dart of an oriented face
	* @param d a dart of the oriented face
	* @param fonct the functor
	*/
	bool foreach_dart_of_face(Dart d, FunctorType& f, unsigned int thread=0);

	/**
	* Apply a functor on each dart of an volume
	* @param d a dart of the volume
	* @param fonct the functor
	*/
	// TODO change to oriented volume to handle higher dimension ?
	bool foreach_dart_of_volume(Dart d, FunctorType& f, unsigned int thread=0)
	{
		return foreach_dart_of_oriented_volume(d,f);
	};

	/**
	* Apply a functor on each dart of a cc
	* @param d a dart of the cc
	* @param fonct the functor
	*/
	bool foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread=0);
};

} // namespace CGoGN

#ifndef TEMPLATE_INSTANCIED
#include "xmap/xmap.hpp"
#endif

#endif
