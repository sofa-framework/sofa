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

#ifndef _HYPER_MAP_H
#define _HYPER_MAP_H

#include "Topology/generic/genericmap.h"

namespace CGoGN
{


#define NIL this->end()

template <typename DART>
class tHyperMap : public tGenericMap<DART>
{
	/**
	* short type definition for local dart iterator
	*/
	typedef typename std::list<DART>::iterator Dart;

	/**
	* insert a new Dart in the map
	*/
	Dart tHyperMap<DART>::newHyperDart();

	/**
	* apply permutation alpha i
	*/
	Dart alpha(int i,Dart d);

	/**
	* apply permutation alpha i
	*/
	Dart alpha_(int i,Dart d);

	/**
	* sew two darts by alpha i
	*/
	void sewAlpha(int i,Dart d, Dart e);

	/**
	* unsew by alpha i
	*/
	void unsewAlpha(int i,Dart d);

	/**
	* create a pseudo edge (2 darts sewed by alpha0)
	*/
	Dart createPseudoEdge();

	/**
	* create a face
	*/
	Dart createFace(int nbEdges);

	/**
	* sew to face by edge fusion
	*/
	void edgeFusion(Dart d, Dart e);

	template <typename MAP>
	bool foreach_dart_of_face(Dart d, FunctorType<MAP>& fonct);
};


} // namespace CGoGN

#ifndef TEMPLATE_INSTANCIED
#include "hypermap/hypermap2.hpp"
#endif


#endif