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

#ifndef RAYSELECTOR_H_
#define RAYSELECTOR_H_

#include <vector>
#include "Algo/Selection/raySelectFunctor.hpp"

namespace CGoGN
{

namespace Algo
{

namespace Selection
{

/**
 * Function that does the selection of faces, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param position the vertex attribute storing positions
 * @param rayA first point of ray (user side)
 * @param rayAB direction of ray (directed to the scene)
 * @param vecFaces (out) vector to store the intersected faces
 * @param iPoints (out) vector to store the intersection points
 */
template<typename PFP>
void facesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Face>& vecFaces,
		std::vector<typename PFP::VEC3>& iPoints);

/**
 * Function that does the selection of faces, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param position the vertex attribute storing positions
 * @param rayA first point of ray (user side)
 * @param rayAB direction of ray (directed to the scene)
 * @param vecFaces (out) vector to store the intersected faces
 */
template<typename PFP>
void facesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Face>& vecFaces);

/**
 * Function that does the selection of one face
 * @param map the map we want to test
 * @param position the vertex attribute storing positions
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param face (out) selected face (set to NIL if no face selected)
 */
template<typename PFP>
void faceRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		Face& face);

/**
 * Function that does the selection of edges, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param position the vertex attribute storing positions
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vecEdges (out) vector to store intersected edges
 * @param distMax radius of the cylinder of selection
 */
template<typename PFP>
void edgesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Edge>& vecEdges,
		float distMax);

/**
 * Function that does the selection of one vertex
 * @param map the map we want to test
 * @param position the vertex attribute storing positions
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param edge (out) selected edge (set to NIL if no edge selected)
 */
template<typename PFP>
void edgeRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		Edge& edge);

/**
 * Function that does the selection of vertices, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param position the vertex attribute storing positions
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vecVertices (out) vector to store intersected vertices
 * @param dist radius of the cylinder of selection
 */
template<typename PFP>
void verticesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Vertex>& vecVertices,
		float dist);

/**
 * Function that does the selection of one vertex
 * @param map the map we want to test
 * @param position the vertex attribute storing positions
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vertex (out) selected vertex (set to NIL if no vertex selected)
 */
template<typename PFP>
void vertexRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		Vertex& vertex);

/**
 * Volume selection, not yet functional
 */
template<typename PFP>
void volumesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Vol>& vecVolumes);

template<typename PFP>
void facesPlanSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename Geom::Plane3D<typename PFP::VEC3::DATA_TYPE>& plan,
		std::vector<Face>& vecFaces);

/**
 * Function that does the selection of vertices in a cone, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param position the position attribute
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param angle angle of the cone in degree.
 * @param vecVertices (out) vector to store intersected vertices
 */
template<typename PFP>
void verticesConeSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		float angle,
		std::vector<Vertex>& vecVertices);

/**
 * Function that does the selection of edges, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param position the position attribute
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param angle radius of the cylinder of selection
 * @param vecEdges (out) vector to store intersected edges
 */
template<typename PFP>
void edgesConeSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		float angle,
		std::vector<Edge>& vecEdges);

/**
 * Function that select the closest vertex in the bubble
 * @param map the map we want to test
 * @param position the position attribute
 * @param cursor the cursor position (center of bubble)
 * @param radiusMax max radius of selection
 */
template<typename PFP>
Vertex verticesBubbleSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& cursor,
		typename PFP::REAL radiusMax);

/**
 * Function that select the closest edge in the bubble
 * @param map the map we want to test
 * @param position the position attribute
 * @param cursor the cursor position (center of bubble)
 * @param radiusMax max radius of selection
 */
template<typename PFP>
Edge edgesBubbleSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
		const typename PFP::VEC3& cursor,
		typename PFP::REAL radiusMax);

/**
 * Fonction that do the selection of darts, returned darts are sorted from closest to farthest
 * Dart is here considered as a triangle formed by the 2 end vertices of the edge and the face centroid
 * @param map the map we want to test
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vecDarts (out) vector to store dart of intersected darts
 */
//template<typename PFP>
//void dartsRaySelection(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, std::vector<Dart>& vecDarts);

//namespace Parallel
//{
//template<typename PFP>
//void facesRaySelection(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const FunctorSelect& good, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, std::vector<Dart>& vecFaces, unsigned int nbth=0, unsigned int current_thread=0);
//template<typename PFP>
//void edgesRaySelection(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const FunctorSelect& good, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, std::vector<Dart>& vecEdges, float dist, unsigned int nbth=0, unsigned int current_thread=0);
//template<typename PFP>
//void vertexRaySelection(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, Dart& vertex, unsigned int nbth=0, unsigned int current_thread=0);
//}

} //namespace Selection

} //namespace Algo

} //namespace CGoGN

#include "Algo/Selection/raySelector.hpp"

#endif /* RAYSELECTOR_H_ */
