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

#ifndef __MAP2MR_PRIMAL_ADAPT__
#define __MAP2MR_PRIMAL_ADAPT__

#include "Topology/map/embeddedMap2.h"
#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor2.h"

#include <cmath>

namespace CGoGN
{

namespace Algo
{

namespace MR
{

namespace Primal
{

namespace Adaptive
{

template <typename PFP>
class IHM2
{

public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	MAP& m_map;
	bool shareVertexEmbeddings ;

	FunctorType* vertexVertexFunctor ;
	FunctorType* edgeVertexFunctor ;
	FunctorType* faceVertexFunctor ;

public:
	IHM2(MAP& map) ;


protected:
	/***************************************************
	 *               SUBDIVISION                       *
	 ***************************************************/

	/**
	 * subdivide the edge of d to the next level
	 */
	void subdivideEdge(Dart d) ;

	/**
	 * coarsen the edge of d from the next level
	 */
	void coarsenEdge(Dart d) ;

public:
	/**
	 * subdivide the face of d to the next level
	 */
	unsigned int subdivideFace(Dart d, bool triQuad = true, bool OneLevelDifference = true);

	/**
	 *
	 */
	unsigned int subdivideFaceSqrt3(Dart d);

	/**
	 * coarsen the face of d from the next level
	 */
	void coarsenFace(Dart d) ;

	/**
	 * vertices attributes management
	 */
	void setVertexVertexFunctor(FunctorType* f) { vertexVertexFunctor = f ; }
	void setEdgeVertexFunctor(FunctorType* f) { edgeVertexFunctor = f ; }
	void setFaceVertexFunctor(FunctorType* f) { faceVertexFunctor = f ; }
} ;

} // namespace Adaptive

} // namespace Primal

} // namespace MR

} // namespace Algo

} // namespace CGoGN

#include "Algo/Multiresolution/IHM2/ihm2_PrimalAdapt.hpp"

#endif
