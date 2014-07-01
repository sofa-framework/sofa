/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2011, IGG Team, LSIIT, University of Strasbourg           *
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
* Web site: http://cgogn.u-strasbg.fr/                                         *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef __ALGO_GEOMETRY_FEATURE_H__
#define __ALGO_GEOMETRY_FEATURE_H__

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

enum
{
	EMPTY,
	SEGMENT,
	BARY,
	FEATURE
};

typedef struct { Dart d ; float w ; } e0point ;
typedef struct { e0point p1 ; e0point p2 ; unsigned char type ; } e0segment ;
typedef NoTypeNameAttribute<e0segment> ridgeSegment ;

template <typename PFP>
void featureEdgeDetection(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	CellMarker<typename PFP::MAP, EDGE>& featureEdge) ;

template <typename PFP>
void computeFaceGradient(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_normal,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& scalar,
	const FaceAttribute<typename PFP::REAL, typename PFP::MAP>& face_area,
	FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_gradient,
	unsigned int thread = 0) ;

template <typename PFP>
typename PFP::VEC3 faceGradient(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_normal,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& scalar,
	const FaceAttribute<typename PFP::REAL, typename PFP::MAP>& face_area) ;

template <typename PFP>
void computeVertexGradient(
	typename PFP::MAP& map,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_gradient,
	const FaceAttribute<typename PFP::REAL, typename PFP::MAP>& face_area,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& vertex_gradient,
	unsigned int thread = 0) ;

template <typename PFP>
typename PFP::VEC3 vertexGradient(
	typename PFP::MAP& map,
	Dart d,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_gradient,
	const FaceAttribute<typename PFP::REAL, typename PFP::MAP>& area) ;

//template <typename PFP>
//typename PFP::REAL extremality(
//	typename PFP::MAP& map,
//	Dart d,
//	const typename PFP::VEC3& K,
//	const FaceAttribute<typename PFP::VEC3>& face_gradient,
//	const FaceAttribute<typename PFP::REAL>& face_area) ;

template <typename PFP>
void computeTriangleType(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& K,
	CellMarker<typename PFP::MAP, FACE>& regularMarker,
	unsigned int thread = 0) ;

template <typename PFP>
bool isTriangleRegular(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& K) ;

template <typename PFP>
void initRidgeSegments(
	typename PFP::MAP& map,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments,
	unsigned int thread = 0) ;

template <typename PFP>
void computeRidgeLines(
	typename PFP::MAP& map,
	CellMarker<typename PFP::MAP, FACE>& regularMarker,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& K,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& vertex_gradient,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& k,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& k2,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments,
	unsigned int thread = 0) ;

template <typename PFP>
void ridgeLines(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_area,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& K,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& vertex_gradient,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& k,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& k2,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments) ;

template <typename PFP>
void computeSingularTriangle(
	typename PFP::MAP& map,
	CellMarker<typename PFP::MAP, FACE>& regularMarker,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments,
	unsigned int thread = 0) ;

template <typename PFP>
void singularTriangle(
	typename PFP::MAP& map,
	Dart d,
	CellMarker<typename PFP::MAP, FACE>& regularMarker,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments) ;

template <typename PFP>
bool isEdgeInTriangle(
	typename PFP::MAP& map,
	Dart edge,
	Dart triangle) ;

template <typename PFP>
std::vector<typename PFP::VEC3> occludingContoursDetection(
	typename PFP::MAP& map,
	const typename PFP::VEC3& cameraPosition,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal) ;

} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Geometry/feature.hpp"

#endif
