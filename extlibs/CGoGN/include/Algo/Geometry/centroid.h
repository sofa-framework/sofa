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

#ifndef __ALGO_GEOMETRY_CENTROID_H__
#define __ALGO_GEOMETRY_CENTROID_H__

#include "Geometry/basic.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

/**
* Compute volume centroid (generic version)
*  Pre: closed volume & embedded vertices
* Template param:
*  PFP:  as usual
*  V_ATT: attributes vector type  or cell type (VertexCell, FaceCell, ...)
* @param map the map
* @param d a dart of the face
* @param attributs the vector of attribute or cell
*/
template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE volumeCentroid(typename PFP::MAP& map, Vol d, const V_ATT& attributs, unsigned int thread = 0);

/**
* Compute volume centroid weighted by edge length (generic version)
*  Pre: closed volume & embedded vertices
* Template param:
*  PFP:  as usual
*  V_ATT: attributes vector type  or cell type (VertexCell, FaceCell, ...)
* @param map the map
* @param d a dart of the face
* @param attributs the vector of attribute or cell
*/
template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE volumeCentroidELW(typename PFP::MAP& map, Vol d, const V_ATT& attributs, unsigned int thread = 0);

/**
 * Compute face centroid (generic version)
 * Template param:
 *  PFP:  as usual
 *  V_ATT: attributes vector type  or cell type (VertexCell, FaceCell, ...)
 * @param map the map
 * @param d a dart of the face
 * @param attributs the vector of attribute or cell
 */
template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE faceCentroid(typename PFP::MAP& map, Face d, const V_ATT& attributs);

/**
 * Compute face centroid weighted by edge length (generic version)
 * Template param:
 *  PFP:  as usual
 *  V_ATT: attributes vector type  or cell type (VertexCell, FaceCell, ...)
 * @param map the map
 * @param d a dart of the face
 * @param attributs the vector of attribute or cell
 */
template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE faceCentroidELW(typename PFP::MAP& map, Face d, const V_ATT& attributs);

/**
 * Compute vertex neighbours centroid (generic version)
 * Template param:
 *  PFP:  as usual
 *  V_ATT: attributes vector type  or cell type (VertexCell, FaceCell, ...)
 * @param map the map
 * @param d a dart of the face
 * @param position the vector of attribute or cell
 */
template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE vertexNeighborhoodCentroid(typename PFP::MAP& map, Vertex d, const V_ATT& attributs);

/**
 * Compute centroid of all faces
 * @param map the map
 * @param position position vertex attribute
 * @param face_centroid centroid face attribute
 * @param thread the thread id (default 0)
 */
template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid, unsigned int thread = 0);

/**
 * Compute centroid of all faces (Edge Length Weighted)
 * @param map the map
 * @param position position vertex attribute
 * @param face_centroid centroid face attribute
 * @param thread the thread id (default 0)
 */
template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidELWFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid, unsigned int thread = 0) ;

/**
 * Compute neighborhood centroid of all vertices
 * @param map the map
 * @param position position vertex attribute
 * @param vertex_centroid centroid vertex attribute
 * @param thread the thread id (default 0)
 */
template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& vertex_centroid, unsigned int thread = 0) ;


namespace Parallel
{

/**
 * Compute centroid of all faces
 * @param map the map
 * @param position position vertex attribute
 * @param face_centroid centroid face attribute
 */
template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidFaces(typename PFP::MAP& map,
		const V_ATT& position, F_ATT& face_centroid) ;

/**
 * Compute centroid of all faces (Edge Length Weighted)
 * @param map the map
 * @param position position vertex attribute
 * @param face_centroid centroid face attribute
 */
template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidELWFaces(typename PFP::MAP& map,
		const V_ATT& position, F_ATT& face_centroid) ;

/**
 * Compute neighborhood centroid of all vertices (in parallel)
 * @param map the map
 * @param position position vertex attribute
 * @param vertex_centroid centroid vertex attribute
 */
template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map,
		const V_ATT& position, V_ATT& vertex_centroid) ;

} // namespace Parallel



} // namespace Geometry

} // namespace Surface

namespace Volume
{

namespace Geometry
{

/**
 * Compute vertex neighbours centroid in map of dimension 3(generic version)
 * Template param:
 *  PFP:  as usual
 *  V_ATT: attributes vector type  or cell type (VertexCell, FaceCell, ...)
 * @param map the map
 * @param d a dart of the face
 * @param position the vector of attribute or cell
 */
template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE vertexNeighborhoodCentroid(typename PFP::MAP& map, Vertex d, const V_ATT& attributs, unsigned int thread = 0);

/**
 * compute centroid of all volumes
 * @param map the map
 * @param position vertex attribute of position
 * @param vol_centroid volume attribute where to store the centroids
 */
template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidVolumes(typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid, unsigned int thread = 0);

template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidELWVolumes(typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid, unsigned int thread = 0);

/**
 * compute centroid of all vertices
 * @param map the map
 * @param position vertex attribute of position
 * @param vertex_centroid vertex attribute to store the centroids
 */
template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& vertex_centroid, unsigned int thread = 0) ;


namespace Parallel
{

template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidVolumes(typename PFP::MAP& map,
		const V_ATT& position, W_ATT& vol_centroid) ;


template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidELWVolumes(typename PFP::MAP& map,
		const V_ATT& position, W_ATT& vol_centroid) ;

		
template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map,
		const V_ATT& position, V_ATT& vertex_centroid) ;

} // namespace Parallel


} // namespace Geometry

} // namespace Volume

} // namespace Algo

} // namespace CGoGN

#include "Algo/Geometry/centroid.hpp"

#endif
