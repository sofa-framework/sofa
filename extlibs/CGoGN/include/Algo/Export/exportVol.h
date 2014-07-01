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

#ifndef __EXPORT_VOL_H__
#define __EXPORT_VOL_H__

#include "Topology/generic/attributeHandler.h"
#include <stdint.h>

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Export
{
/**
* Export a mesh choosing the format according to filename extension
* @param the_map map to be exported
* @param position the position container
* @param filename filename of mesh file
* @return true
 */
template <typename PFP>
bool exportMesh(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const std::string& filename) ;

/**
* export the map into a .nas (nastran file)
* @param the_map map to be exported
* @param position the position container
* @param filename filename of mesh file
* @return true
*/
template <typename PFP>
bool exportNAS(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const char* filename) ;


/**
* export the map into a vtu file (vtk unstructured grid)
* @param the_map map to be exported
* @param position the position container
* @param filename filename of mesh file
* @return true
*/
template <typename PFP>
bool exportVTU(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const char* filename) ;


/**
* export the map into a .msh (gmesh file)
* @param the_map map to be exported
* @param position the position container
* @param filename filename of mesh file
* @return true
*/
template <typename PFP>
bool exportMSH(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const char* filename) ;


/**
* export the map into a .tet file
* @param the_map map to be exported
* @param position the position container
* @param filename filename of mesh file
* @return true
*/
template <typename PFP>
bool exportTet(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const char* filename) ;


/**
* export the map into a .node/.ele file pair
* @param the_map map to be exported
* @param position the position container
* @param filename filename of mesh file
* @return true
*/
template <typename PFP>
bool exportNodeEle(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const char* filename) ;


/**
* export in binary volume file (nb_vert,nb_tetra,nb_hexa, vertices, tetra, hexa)
* @param the_map map to be exported
* @param position the position container
* @param filename filename of mesh file
 */
template <typename PFP>
bool exportVolBinGz(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const char* filename);


/**
* export tetmesh format
* @warning if macro _OPTIMIZED_FOR_TETRA_ONLY_ (before include) is define assume map contain only tetrahedrons
* @param the_map map to be exported
* @param position the position container
* @param filename filename of mesh file
 */
template <typename PFP>
bool exportTetmesh(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const char* filename);


} // namespace Export

} // namespace Volume

} // namespace Algo

} // namespace CGoGN

#include "Algo/Export/exportVol.hpp"

#endif
