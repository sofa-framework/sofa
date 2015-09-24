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

#ifndef __IMPLICIT_HIERARCHICAL_MESH_SUBDIVISION3__
#define __IMPLICIT_HIERARCHICAL_MESH_SUBDIVISION3__


namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace IHM
{

enum SubdivideType
{
	S_TRI,
	S_QUAD
} ;




template <typename PFP>
void newLevelHexa(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);


/***********************************************************************************
 *								 Subdivision									   *
 ***********************************************************************************/


template <typename PFP>
void subdivideEdge(typename PFP::MAP& map, Dart d, typename AttributeHandler_Traits< typename PFP::VEC3, VERTEX, typename PFP::MAP >::Handler& position) ;

template <typename PFP>
void subdivideFace(typename PFP::MAP& map, Dart d, typename AttributeHandler_Traits< typename PFP::VEC3, VERTEX, typename PFP::MAP >::Handler& position, SubdivideType sType = S_TRI);

template <typename PFP>
Dart subdivideVolumeClassic(typename PFP::MAP& map, Dart d, typename AttributeHandler_Traits< typename PFP::VEC3, VERTEX, typename PFP::MAP >::Handler& position );








template <typename PFP>
void subdivideEdgeWrong(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position) ;

template <typename PFP>
void subdivideFaceWrong(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, SubdivideType sType = S_TRI);

template <typename PFP>
Dart subdivideVolumeClassicWrong(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);









template <typename PFP>
Dart subdivideVolume(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);





template <typename PFP>
Dart subdivideVolumeGen(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);



template <typename PFP>
Dart subdivideVolumeClassic2(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);


template <typename PFP>
void subdivideLoop(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);


/***********************************************************************************
 *								 Simplification									   *
 ***********************************************************************************/

template <typename PFP>
void coarsenEdge(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);

template <typename PFP>
void coarsenFace(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, SubdivideType sType = S_TRI);

template <typename PFP>
void coarsenVolume(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);

/***********************************************************************************
 *												Raffinement
 ***********************************************************************************/
/*
 * Un brin de la face oppose aux faces a spliter
 */
template <typename PFP>
void splitVolume(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position);




} //namespace IHM
} // Volume
} //namespace Algo
} //namespace CGoGN

#include "Algo/ImplicitHierarchicalMesh/subdivision3.hpp"

#endif
