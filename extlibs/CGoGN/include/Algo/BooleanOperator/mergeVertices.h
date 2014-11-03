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

#ifndef __ALGO_BOOLEANOPERATOR_VERTICES_H__
#define __ALGO_BOOLEANOPERATOR_VERTICES_H__

#include "Geometry/basic.h"
#include "Geometry/inclusion.h"
#include "Geometry/orientation.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace BooleanOperator
{

template <typename PFP>
bool isBetween(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, Dart d, Dart e, Dart f) ;

template <typename PFP>
void mergeVertex(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, Dart d, Dart e);

template <typename PFP>
void mergeVertices(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions);

}

}

}

}

#include "mergeVertices.hpp"

#endif
