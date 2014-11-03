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

#ifndef __TRAVERSOR1_H__
#define __TRAVERSOR1_H__

#include "Topology/generic/dart.h"
#include "Topology/generic/traversor/traversorGen.h"

namespace CGoGN
{

/*******************************************************************************
					VERTEX CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the edges incident to a given vertex
template <typename MAP>
class Traversor1VE: public Traversor
{
private:
	const MAP& m ;
	Dart start ;
	Dart current ;

	Dart d2 ;

public:
	Traversor1VE(const MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the vertices adjacent to a given vertex through sharing a common edge
template <typename MAP>
class Traversor1VVaE: public Traversor
{
private:
	const MAP& m ;
	Dart start ;
	Dart current ;

	Dart d2 ;

public:
	Traversor1VVaE(const MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

/*******************************************************************************
					EDGE CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the vertices incident to a given edge
template <typename MAP>
class Traversor1EV: public Traversor
{
private:
	const MAP& m ;
	Dart start ;
	Dart current ;

	Dart d2 ;

public:
	Traversor1EV(const MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the edges adjacent to a given edge through sharing a common vertex
template <typename MAP>
class Traversor1EEaV: public Traversor
{
private:
	const MAP& m ;
	Dart start ;
	Dart current ;

	Dart d2 ;

public:
	Traversor1EEaV(const MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

} // namespace CGoGN

#include "Topology/generic/traversor/traversor1.hpp"

#endif
