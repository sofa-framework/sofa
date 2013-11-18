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

#ifndef _MAP_GL_RENDER
#define _MAP_GL_RENDER

#include <list>
#include <GL/glew.h>

#include "Topology/generic/functor.h"

/**
* A set of functions that allow the creation of rendering
* object using Vertex-Buffer-Object.
* Function are made for dual-2-map and can be used on
* any subset of a dual-N-map which is a 2-map
*/
namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL1
{

enum RenderType { NO_LIGHT=1, LINE, FLAT, SMOOTH };
enum RenderPrimitives { NONE=0, TRIANGLES=3, QUADS=4, POLYGONS=5, TRIFAN=6 };

/**
* @param the_map the map to render
* @param rt type of rendu (FLAT, SMOOTH, FIL)
* @param explode face exploding coefficient
*/
template <typename PFP>
void renderTriQuadPoly(typename PFP::MAP& the_map, RenderType rt, float explode,
		const VertexAttribute<typename PFP::VEC3>& position, const VertexAttribute<typename PFP::VEC3>& normal);

template <typename PFP>
void renderTriQuadPoly(typename PFP::MAP& the_map, RenderType rt, float explode,
		const VertexAttribute<typename PFP::VEC3>& position, const VertexAttribute<typename PFP::VEC3>& normal, const VertexAttribute<typename PFP::VEC3>& color);

template <typename PFP>
void renderNormalVertices(typename PFP::MAP& the_map,
		const VertexAttribute<typename PFP::VEC3>& position, const VertexAttribute<typename PFP::VEC3>& normal, float scale);

template <typename PFP>
void renderFrameVertices(typename PFP::MAP& the_map,
		const VertexAttribute<typename PFP::VEC3>& position, const VertexAttribute<typename PFP::VEC3> frame[3], float scale);

} // namespace Direct

} // namespace Render

} // namespace Algo

} // namespace CGoGN

#include "Algo/Render/GL1/renderFunctor.h"
#include "Algo/Render/GL1/map_glRender.hpp"

#endif
