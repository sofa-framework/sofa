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

#ifndef EXTRUSION_H
#define EXTRUSION_H

#include <math.h>
#include <vector>
#include "Algo/Tiling/Surface/square.h"

#include "Algo/Modelisation/subdivision.h"
#include "Algo/Geometry/normal.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Modelisation
{

/**
* extrusion with scale
* WARNING: this function use OpenGL to create transformation matrix !
* @param the_map the map in which include created surface
* @param profile vector of points that describe the profile (must be almost coplanar for best result)
* @param center profile virtual point defined as the center of the profile (follow the path)
* @param normalProfile normal direction of profile plane
* @param profile_closed profile is a closed polygon or not ?
* @param path the vector of points that define the path to follow
* @param path_closed path is a closed polygon or not ?
* @param scalePath a vector of scale value to apply on profile at each node of path,
* size of vector must be the same as path
*/
template<typename PFP>
Dart extrusion_scale(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& centerProfile,
	const typename PFP::VEC3& normalProfile,
	bool profile_closed,
	const std::vector<typename PFP::VEC3>& path,
	bool path_closed,
	const std::vector<float>& scalePath);

/**
* extrusion with scale, return a Polyhedron that can be easily transform
* WARNING: this function use OpenGL to create transformation matrix !
* WARNING: return a pointer on Polyhedron it is up to the use to delete this Polyhedron
* @param the_map the map in which include created surface
* @param profile vector of points that describe the profile (must be almost coplanar for best result)
* @param center profile virtual point defined as the center of the profile (follow the path)
* @param normalProfile normal direction of profile plane
* @param profile_closed profile is a closed polygon or not ?
* @param path the vector of points that define the path to follow
* @param path_closed path is a closed polygon or not ?
* @param scalePath a vector of scale value to apply on profile at each node of path,
* size of vector must be the same as path
*/
template<typename PFP>
//Polyhedron<PFP>* extrusion_scale_prim(typename PFP::MAP& the_map, VertexAttribute<typename PFP::VEC3>& positions,
Algo::Surface::Tilings::Tiling<PFP>* extrusion_scale_prim(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& centerProfile,
	const typename PFP::VEC3& normalProfile,
	bool profile_closed,
	const std::vector<typename PFP::VEC3>& path,
	bool path_closed,
	const std::vector<float>& scalePath);

/**
* extrusion with scale, return a Polyhedron that can be easily transform
* WARNING: this function use OpenGL to create transformation matrix !
* WARNING: return a pointer on Polyhedron it is up to the use to delete this Polyhedron
* @param the_map the map in which include created surface
* @param profile vector of points that describe the profile (must be almost coplanar for best result)
* @param center profile virtual point defined as the center of the profile (follow the path)
* @param normalProfile normal direction of profile plane
* @param profile_closed profile is a closed polygon or not ?
* @param path the vector of points that define the path to follow
* @param path_closed path is a closed polygon or not ?
* @param scalePath a vector of scale value to apply on profile at each node of path,
* size of vector must be the same as path
*/
template<typename PFP>
//Polyhedron<PFP>* extrusion_prim(typename PFP::MAP& the_map, VertexAttribute<typename PFP::VEC3>& positions,
Algo::Surface::Tilings::Tiling<PFP>* extrusion_prim(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& centerProfile,
	const typename PFP::VEC3& normalProfile,
	bool profile_closed,
	const std::vector<typename PFP::VEC3>& path,
	bool path_closed);

/**
* extrusion!
* WARNING: this function use OpenGL to create transformation matrix !
* @param the_map the map in which include created surface
* @param profile vector of points that describe the profile (must be almost coplanar for best result)
* @param centerProfile virtual point defined as the center of the profile (follow the path)
* @param normalProfile normal direction of profile plane
* @param profile_closed profile is a closed polygon or not ?
* @param path the vector of points that define the path to follow
* @param path_closed path is a closed polygon or not ?
* size of vector must be the same as path
*/
template<typename PFP>
Dart extrusion(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& centerProfile,
	const typename PFP::VEC3& normal,
	bool profile_closed,
	const std::vector<typename PFP::VEC3>& path,
	bool path_closed);

/**
* revolution!
* WARNING: this function use OpenGL to create transformation matrix !
* @param the_map the map in which include created surface
* @param profile vector of points that describe the profile (must be almost coplanar for best result)
* @param center center (point) of axis revolution
* @param axis direction of axis revolution
* @param profile_closed profile is a closed polygon or not ?
* @param nbSide number of steps around the revolution
*/
template<typename PFP>
//Polyhedron<PFP>* revolution_prim(typename PFP::MAP& the_map, VertexAttribute<typename PFP::VEC3>& positions,
Algo::Surface::Tilings::Tiling<PFP>* revolution_prim(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& center,
	const typename PFP::VEC3& axis,
	bool profile_closed,
	int nbSides);

/**
* revolution!
* WARNING: this function use OpenGL to create transformation matrix !
* @param the_map the map in which include created surface
* @param profile vector of points that describe the profile (must be almost coplanar for best result)
* @param center center (point) of axis revolution
* @param axis direction of axis revolution
* @param profile_closed profile is a closed polygon or not ?
* @param nbSide number of steps around the revolution
*/
template<typename PFP>
Dart revolution(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& center,
	const typename PFP::VEC3& axis,
	bool profile_closed,
	int nbSides);

/**
* Face extrusion
* @param the_map the map in which include created surface
* @param d a dart of the face to extrude
* @param N the vector use to extrude face center (point) of axis revolution
*/
template<typename PFP>
Dart extrudeFace(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,
	Dart d,
	const typename PFP::VEC3& N);

/**
* Face extrusion
* @param the_map the map in which include created surface
* @param d a dart of the face to extrude
* @param dist the height to extrude face
*/
template<typename PFP>
Dart extrudeFace(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,
	Dart d,
	float dist);

template<typename PFP>
Dart extrudeRegion(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	Dart d,
	const CellMarker<typename PFP::MAP, FACE>& cm);

} // namespace Modelisation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Modelisation/extrusion.hpp"

#endif
