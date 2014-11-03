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

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace SVG
{

template <typename PFP>
void renderVertices(Utils::SVG::SVGOut& svg, typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, unsigned int /*thread*/)
{
	Utils::SVG::SvgGroup* svg1 = new Utils::SVG::SvgGroup("vertices", svg.m_model, svg.m_proj);
	TraversorV<typename PFP::MAP> trac(map);
	svg1->beginPoints();
	for (Dart d = trac.begin(); d != trac.end(); d = trac.next())
		svg1->addPoint(position[d]);
	svg1->endPoints();
	svg.addGroup(svg1);
}

template <typename PFP>
void renderVertices(Utils::SVG::SVGOut& svg, typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& color, unsigned int thread)
{
	Utils::SVG::SvgGroup* svg1 = new Utils::SVG::SvgGroup("vertices", svg.m_model, svg.m_proj);
	TraversorV<typename PFP::MAP> trac(map);
	svg1->beginPoints();
	for (Dart d = trac.begin(); d != trac.end(); d = trac.next())
		svg1->addPoint(position[d], color[d]);
	svg1->endPoints();
	svg.addGroup(svg1);
}

template <typename PFP>
void renderEdges(Utils::SVG::SVGOut& svg, typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, unsigned int /*thread*/)
{
	Utils::SVG::SvgGroup* svg1 = new Utils::SVG::SvgGroup("edges", svg.m_model, svg.m_proj);
	TraversorE<typename PFP::MAP> trac(map);
	svg1->beginLines();
	for (Dart d = trac.begin(); d != trac.end(); d = trac.next())
		svg1->addLine(position[d], position[map.phi1(d)]);
	svg1->endLines();
	svg.addGroup(svg1);
}

template <typename PFP>
void renderEdges(Utils::SVG::SVGOut& svg, typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& color, unsigned int thread)
{
	Utils::SVG::SvgGroup* svg1 = new Utils::SVG::SvgGroup("edges", svg.m_model, svg.m_proj);
	TraversorE<typename PFP::MAP> trac(map);
	svg1->beginLines();
	for (Dart d = trac.begin(); d != trac.end(); d = trac.next())
		svg1->addLine(position[d], position[map.phi1(d)], color[d]);
	svg1->endLines();
	svg.addGroup(svg1);
}

} // namespace SVG

} // namespace Render

} // namespace Algo

} // namespace CGoGN
