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

#ifndef __IMPORT_H__
#define __IMPORT_H__

#include "Topology/generic/attributeHandler.h"
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/dartmarker.h"

#include "Algo/Import/import2tables.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import 
{

/**
* import a mesh
* @param map the map in which the function imports the mesh
* @param filename
* @param attrNames attribute names
* @param mergeCloseVertices a boolean indicating if close vertices should be merged during import
* @return a boolean indicating if import was successful
*/
template <typename PFP>
bool importMesh(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, bool mergeCloseVertices = false);

/**
* import a voxellisation
* @param map the map in which the function imports the mesh
* @param voxellisation
* @param attrNames attribute names
* @param mergeCloseVertices a boolean indicating if close vertices should be merged during import
* @return a boolean indicating if import was successful
*/
template <typename PFP>
bool importVoxellisation(typename PFP::MAP& map, Algo::Surface::Modelisation::Voxellisation& voxellisation, std::vector<std::string>& attrNames, bool mergeCloseVertices=false);

/**
 * import a Choupi file
 * @param map
 * @param filename
 * @return
 */
template <typename PFP>
bool importChoupi(const std::string& filename, const std::vector<typename PFP::VEC3>& tabV, const std::vector<unsigned int>& tabE);

} // namespace Import

} // Surface


namespace Volume
{

namespace Import
{
template <typename PFP>
Dart createStampVolume(typename PFP::MAP& map);
/**
 * import a volumetric mesh
 * @param map the map in which the function imports the mesh
 * @param filename
 * @param attrNames attribute names
 * @param mergeCloseVertices a boolean indicating if close vertices should be merged during import
 * @return a boolean indicating if import was successful
 */
template <typename PFP>
bool importMesh(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, bool mergeCloseVertices = false);

/**
 * import a mesh and extrude it
 * @param map the map in which the function imports the mesh
 * @param filename
 * @param attrNames attribute names
 * @param mergeCloseVertices a boolean indicating if close vertices should be merged during import
 * @return a boolean indicating if import was successful
 */
template <typename PFP>
bool importMeshToExtrude(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, float scale = 5.0f, unsigned int nbStage = 1);

/**
 *
 */
template <typename PFP>
bool importMeshSAsV(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames);

template <typename PFP>
bool importMSH(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, float scaleFactor = 1.0f);

} // Import

} // Volume


} // namespace Algo

} // namespace CGoGN

#include "Algo/Import/import.hpp"
#include "Algo/Import/importChoupi.hpp"
//#include "Algo/Import/importMSH.hpp"

//#include "Algo/Import/importMoka.hpp"
//#include "Algo/Import/importObjTex.hpp"

#endif
