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

#ifndef __TRAVERSOR_CELLVIRT_H__
#define __TRAVERSOR_CELLVIRT_H__

#include "Topology/generic/dart.h"
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversor/traversorGen.h"

namespace CGoGN
{

template <typename MAP, unsigned int ORBIT>
class VTraversorCell : public Traversor
{
private:
	const MAP& m ;

	const AttributeContainer* cont ;
	unsigned int qCurrent ;

	DartMarker<MAP>* dmark ;
	CellMarker<MAP, ORBIT>* cmark ;
	AttributeMultiVector<Dart>* quickTraversal ;

	Dart current ;
	bool firstTraversal ;

public:
	VTraversorCell(const MAP& map, bool forceDartMarker = false, unsigned int thread = 0) ;

	~VTraversorCell() ;

	Dart begin() ;

	Dart end() ;

	Dart next() ;

	void skip(Dart d);
} ;

//template <typename MAP>
//class TraversorV : public VTraversorCell<MAP, VERTEX>
//{
//public:
//	TraversorV(MAP& m, bool forceDartMarker = false, unsigned int thread = 0) : VTraversorCell<MAP, VERTEX>(m, forceDartMarker, thread)
//	{}
//};

//template <typename MAP>
//class TraversorE : public VTraversorCell<MAP, EDGE>
//{
//public:
//	TraversorE(MAP& m, bool forceDartMarker = false, unsigned int thread = 0) : VTraversorCell<MAP, EDGE>(m, forceDartMarker, thread)
//	{}
//};

//template <typename MAP>
//class TraversorF : public VTraversorCell<MAP, FACE>
//{
//public:
//	TraversorF(MAP& m, bool forceDartMarker = false, unsigned int thread = 0) : VTraversorCell<MAP, FACE>(m, forceDartMarker, thread)
//	{}
//};

//template <typename MAP>
//class TraversorW : public VTraversorCell<MAP, VOLUME>
//{
//public:
//	TraversorW(MAP& m, bool forceDartMarker = false, unsigned int thread = 0) : VTraversorCell<MAP, VOLUME>(m, forceDartMarker, thread)
//	{}
//};

/*
template <typename MAP>
class TraversorV : public VTraversorCell<MAP, VERTEX>
{
public:
	TraversorV(MAP& m, unsigned int thread = 0) : VTraversorCell<MAP, VERTEX>(m, false, thread)
	{}
};

template <typename MAP>
class TraversorE : public VTraversorCell<MAP, EDGE>
{
public:
	TraversorE(MAP& m, unsigned int thread = 0) : VTraversorCell<MAP, EDGE>(m, false, thread)
	{}
};

template <typename MAP>
class TraversorF : public VTraversorCell<MAP, FACE>
{
public:
	TraversorF(MAP& m, unsigned int thread = 0) : VTraversorCell<MAP, FACE>(m, false, thread)
	{}
};

template <typename MAP>
class TraversorW : public VTraversorCell<MAP, VOLUME>
{
public:
	TraversorW(MAP& m, unsigned int thread = 0) : VTraversorCell<MAP, VOLUME>(m, false, thread)
	{}
};
*/	

} // namespace CGoGN

#include "Topology/generic/traversor/traversorCellVirt.hpp"

#endif
