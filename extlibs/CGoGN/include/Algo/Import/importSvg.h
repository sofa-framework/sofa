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

#ifndef __IMPORTSVG_H__
#define __IMPORTSVG_H__

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import
{

/**
 * check if an xml node has a given name
 * @param node the xml node
 * @param name the name
 * @ return true if node has the good name
 */
bool checkXmlNode(xmlNodePtr node, const std::string& name);

template <typename PFP>
void readCoordAndStyle(xmlNode* cur_path,
		std::vector<std::vector<VEC3 > >& allPoly,
		std::vector<std::vector<VEC3 > >& allBrokenLines,
		std::vector<float>& allBrokenLinesWidth);

template <typename PFP>
bool importSVG(typename PFP::MAP& map, const std::string& filename, VertexAttribute<typename PFP::VEC3>& position, CellMarker<EDGE>& polygons, CellMarker<FACE>& polygonsFaces);

/**
 *
 */

template <typename PFP>
bool readSVG(const std::string& filename, std::vector<std::vector<typename PFP::VEC3 > > &allPoly);

template <typename PFP>
bool importBB(const std::string& filename, std::vector<Geom::BoundingBox<typename PFP::VEC3> > &bb);

template <typename PFP>
bool importSVG(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames);

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN

#include "Algo/Import/importSvg.hpp"

#endif
