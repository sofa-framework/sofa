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

#ifndef __ALGO_GEOMETRY_AREA_H__
#define __ALGO_GEOMETRY_AREA_H__

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

//! \brief Compute triangle area
//! @param map
//! @param d
//! @param position
//! @return
template <typename PFP>
typename PFP::REAL triangleArea(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position) ;

//! \brief Compute convex polygonal face area
//! @param map
//! @param d
//! @param position
//! @return
template <typename PFP>
typename PFP::REAL convexFaceArea(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position) ;

//! \brief Compute the total area of a mesh by summing all face areas.
//! @param map
//! @param position
//! @return
template <typename PFP>
typename PFP::REAL totalArea(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, unsigned int thread = 0) ;

//! \brief Compute the area of the faces around a vertex.
//! @param map
//! @param d
//! @param position
//! @return
template <typename PFP>
typename PFP::REAL vertexOneRingArea(typename PFP::MAP& map, Vertex v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position) ;

template <typename PFP>
typename PFP::REAL vertexBarycentricArea(typename PFP::MAP& map, Vertex v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position) ;

template <typename PFP>
typename PFP::REAL vertexVoronoiArea(typename PFP::MAP& map, Vertex v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position) ;

template <typename PFP>
void computeAreaFaces(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, FaceAttribute<typename PFP::REAL, typename PFP::MAP>& face_area, unsigned int thread = 0) ;

template <typename PFP>
void computeOneRingAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::REAL, typename PFP::MAP>& vertex_area, unsigned int thread = 0) ;

template <typename PFP>
void computeBarycentricAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::REAL, typename PFP::MAP>& vertex_area, unsigned int thread = 0) ;

template <typename PFP>
void computeVoronoiAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::REAL, typename PFP::MAP>& vertex_area, unsigned int thread = 0) ;

namespace Parallel
{
template <typename PFP>
void computeAreaFaces(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, FaceAttribute<typename PFP::REAL, typename PFP::MAP>& area);

template <typename PFP>
void computeOneRingAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::REAL, typename PFP::MAP>& area);

template <typename PFP>
void computeBarycentricAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::REAL, typename PFP::MAP>& vertex_area);

template <typename PFP>
void computeVoronoiAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::REAL, typename PFP::MAP>& area);
} // namespace Parallel

} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Geometry/area.hpp"

#endif
