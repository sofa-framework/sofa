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

#ifndef _MAP_SVG_RENDER_
#define _MAP_SVG_RENDER_

#include <vector>
#include <fstream>
#include <sstream>

#include "Utils/svg.h"
#include "Topology/generic/traversor/traversorCell.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace SVG
{

/**
 * render vertices in a SVGOut
 * @warning no depth ordering
 */
template <typename PFP>
void renderVertices(Utils::SVG::SVGOut& svg, typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, unsigned int thread = 0);

/**
 * render colored vertices in a SVGOut
 * @warning no depth ordering
 */
template <typename PFP>
void renderVertices(Utils::SVG::SVGOut& svg, typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& color, unsigned int thread = 0);

/**
 * render edges in a SVGOut
 * @warning no depth ordering
 */
template <typename PFP>
void renderEdges(Utils::SVG::SVGOut& svg, typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, unsigned int thread = 0);

/**
 * render colored edges in a SVGOut
 * @warning no depth ordering
 */
template <typename PFP>
void renderEdges(Utils::SVG::SVGOut& svg, typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& color, unsigned int thread = 0);

} // namespace SVG

} // namespace Render

} // namespace Algo

} // namespace CGoGN

#include "Algo/Render/SVG/mapSVGRender.hpp"

#endif
