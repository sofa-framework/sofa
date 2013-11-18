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

#ifndef __LINEAR_SOLVING_VARIABLES_SETUP__
#define __LINEAR_SOLVING_VARIABLES_SETUP__

namespace CGoGN
{

namespace LinearSolving
{

template <typename PFP, typename ATTR_TYPE>
class FunctorMeshToSolver_Scalar : public FunctorType
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const CellMarker<VERTEX>& freeMarker ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;
	bool lockedVertices ;

public:
	FunctorMeshToSolver_Scalar(
		const VertexAttribute<unsigned int>& index,
		const CellMarker<VERTEX>& fm,
		const VertexAttribute<ATTR_TYPE>& attr
	) :	indexTable(index), freeMarker(fm), attrTable(attr), lockedVertices(false)
	{}

	bool operator()(Dart d)
	{
		nlSetVariable(indexTable[d], attrTable[d]);
		if(!freeMarker.isMarked(d))
		{
			nlLockVariable(indexTable[d]);
			lockedVertices = true ;
		}
		return false ;
	}

	bool hasLockedVertices() { return lockedVertices ; }
} ;

template <typename PFP, typename ATTR_TYPE>
class FunctorMeshToSolver_Vector : public FunctorType
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const CellMarker<VERTEX>& freeMarker ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;
	unsigned int coord ;
	bool lockedVertices ;

public:
	FunctorMeshToSolver_Vector(
		const VertexAttribute<unsigned int>& index,
		const CellMarker<VERTEX>& fm,
		const VertexAttribute<ATTR_TYPE>& attr,
		unsigned int c
	) :	indexTable(index), freeMarker(fm), attrTable(attr), coord(c), lockedVertices(false)
	{}

	bool operator()(Dart d)
	{
		nlSetVariable(indexTable[d], (attrTable[d])[coord]);
		if(!freeMarker.isMarked(d))
		{
			nlLockVariable(indexTable[d]);
			lockedVertices = true ;
		}
		return false ;
	}

	bool hasLockedVertices() { return lockedVertices ; }
} ;

template <typename PFP, typename ATTR_TYPE>
class FunctorSolverToMesh_Scalar : public FunctorType
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	VertexAttribute<ATTR_TYPE>& attrTable ;

public:
	FunctorSolverToMesh_Scalar(
		const VertexAttribute<unsigned int>& index,
		VertexAttribute<ATTR_TYPE>& attr
	) :	indexTable(index), attrTable(attr)
	{}

	bool operator()(Dart d)
	{
		attrTable[d] = nlGetVariable(indexTable[d]) ;
		return false ;
	}
} ;

template <typename PFP, typename ATTR_TYPE>
class FunctorSolverToMesh_Vector : public FunctorType
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	VertexAttribute<ATTR_TYPE>& attrTable ;
	unsigned int coord ;

public:
	FunctorSolverToMesh_Vector(
		const VertexAttribute<unsigned int>& index,
		VertexAttribute<ATTR_TYPE>& attr,
		unsigned int c
	) :	indexTable(index), attrTable(attr), coord(c)
	{}

	bool operator()(Dart d)
	{
		(attrTable[d])[coord] = nlGetVariable(indexTable[d]) ;
		return false ;
	}
} ;

} // namespace LinearSolving

} // namespace CGoGN

#endif
