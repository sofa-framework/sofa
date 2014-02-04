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

#ifndef __VTraversor2_VIRT_H__
#define __VTraversor2_VIRT_H__

#include "Topology/generic/dart.h"
#include "Topology/generic/traversorGen.h"

namespace CGoGN
{

/*******************************************************************************
					VERTEX CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the edges incident to a given vertex
template <typename MAP>
class VTraversor2VE: public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2VE(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the faces incident to a given vertex
template <typename MAP>
class VTraversor2VF : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2VF(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the vertices adjacent to a given vertex through sharing a common edge
template <typename MAP>
class VTraversor2VVaE : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2VVaE(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the vertices adjacent to a given vertex through sharing a common face
template <typename MAP>
class VTraversor2VVaF : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

	Dart stop ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2VVaF(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

/*******************************************************************************
					EDGE CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the vertices incident to a given edge
template <typename MAP>
class VTraversor2EV : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2EV(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the faces incident to a given edge
template <typename MAP>
class VTraversor2EF : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2EF(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the edges adjacent to a given edge through sharing a common vertex
template <typename MAP>
class VTraversor2EEaV : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

	Dart stop1, stop2 ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2EEaV(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the edges adjacent to a given edge through sharing a common face
template <typename MAP>
class VTraversor2EEaF : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

	Dart stop1, stop2 ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2EEaF(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

/*******************************************************************************
					FACE CENTERED TRAVERSALS
*******************************************************************************/

// Traverse the vertices incident to a given face
template <typename MAP>
class VTraversor2FV : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2FV(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;


// Traverse the edges incident to a given face (equivalent to vertices)
template <typename MAP>
class VTraversor2FE: public VTraversor2FV<MAP>
{
public:
	VTraversor2FE(MAP& map, Dart dart):VTraversor2FV<MAP>(map,dart){}
} ;

// Traverse the faces adjacent to a given face through sharing a common vertex
template <typename MAP>
class VTraversor2FFaV : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;

	Dart stop ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2FFaV(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

// Traverse the faces adjacent to a given face through sharing a common edge
template <typename MAP>
class VTraversor2FFaE : public Traversor
{
private:
	MAP& m ;
	Dart start ;
	Dart current ;
	const std::vector<Dart>* m_QLT;
	std::vector<Dart>::const_iterator m_ItDarts;
public:
	VTraversor2FFaE(MAP& map, Dart dart) ;

	Dart begin() ;
	Dart end() ;
	Dart next() ;
} ;

} // namespace CGoGN

#include "Topology/generic/traversor2Virt.hpp"

#endif
