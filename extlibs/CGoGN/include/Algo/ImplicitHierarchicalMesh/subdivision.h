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

#ifndef __IMPLICIT_HIERARCHICAL_MESH_SUBDIVISION__
#define __IMPLICIT_HIERARCHICAL_MESH_SUBDIVISION__

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace IHM
{

template <typename PFP>
void subdivideEdge(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3>& position) ;

template <typename PFP>
void subdivideFace(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3>& position, bool forceTri = true);

template <typename PFP>
void coarsenEdge(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3>& position) ;

template <typename PFP>
void coarsenFace(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3>& position) ;

} //namespace IHM
} // Surface
} //namespace Algo
} //namespace CGoGN

#include "Algo/ImplicitHierarchicalMesh/subdivision.hpp"

#endif
