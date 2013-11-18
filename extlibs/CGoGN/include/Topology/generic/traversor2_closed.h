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

#ifndef __TRAVERSOR2_CLOSED_H__
#define __TRAVERSOR2_CLOSED_H__

#include "Topology/generic/dart.h"
#include "Topology/generic/traversorGen.h"

namespace CGoGN
{

namespace ClosedMap
{


/*******************************************************************************
					VERTEX CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the edges incident to a given vertex
template <typename MAP>
class Traversor2VE: public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

public:
	Traversor2VE(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the faces incident to a given vertex
template <typename MAP>
class Traversor2VF : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

public:
	Traversor2VF(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the vertices adjacent to a given vertex through sharing a common edge
template <typename MAP>
class Traversor2VVaE : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

public:
	Traversor2VVaE(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the vertices adjacent to a given vertex through sharing a common face
template <typename MAP>
class Traversor2VVaF : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

	Dart stop ;

public:
	Traversor2VVaF(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

/*******************************************************************************
					EDGE CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the vertices incident to a given edge
template <typename MAP>
class Traversor2EV : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

public:
	Traversor2EV(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the faces incident to a given edge
template <typename MAP>
class Traversor2EF : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

public:
	Traversor2EF(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the edges adjacent to a given edge through sharing a common vertex
template <typename MAP>
class Traversor2EEaV : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

	Dart stop1, stop2 ;

public:
	Traversor2EEaV(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the edges adjacent to a given edge through sharing a common face
template <typename MAP>
class Traversor2EEaF : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

	Dart stop1, stop2 ;

public:
	Traversor2EEaF(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

/*******************************************************************************
					FACE CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the vertices incident to a given face
template <typename MAP>
class Traversor2FV : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

public:
	Traversor2FV(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;


// Traverse the edges incident to a given face (equivalent to vertices)
template <typename MAP>
class Traversor2FE: public Traversor2FV<MAP>
{
public:
	Traversor2FE(MAP& map, Dart dart):Traversor2FV<MAP>(map,dart){}
} ;

// Traverse the faces adjacent to a given face through sharing a common vertex
template <typename MAP>
class Traversor2FFaV : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

	Dart stop ;

public:
	Traversor2FFaV(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the faces adjacent to a given face through sharing a common edge
template <typename MAP>
class Traversor2FFaE : public Traversor<MAP>
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

public:
	Traversor2FFaE(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

} // namespace ClosedMap

} // namespace CGoGN

#include "Topology/generic/traversor2_closed.hpp"

#endif
