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

#ifndef __ALGO_GEOMETRY_NORMAL_H__
#define __ALGO_GEOMETRY_NORMAL_H__

#include "Geometry/basic.h"


namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template<typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE triangleNormal(typename PFP::MAP& map, Face f, const V_ATT& position) ;

template<typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE newellNormal(typename PFP::MAP& map, Face f, const V_ATT& position);

template<typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE faceNormal(typename PFP::MAP& map, Face f, const V_ATT& position) ;

template<typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE vertexNormal(typename PFP::MAP& map, Vertex v, const V_ATT& position) ;

template<typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE vertexBorderNormal(typename PFP::MAP& map, Vertex v, const V_ATT& position) ;

template <typename PFP, typename V_ATT, typename F_ATT>
void computeNormalFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_normal, unsigned int thread = 0) ;

/**
 * compute normals of  vertices
 * @param map the map on which we work
 * @param position the position of vertices attribute handler
 * @param normal the normal handler in which the result will be stored
 * @param the selector
 * @ param th the thread number
 */
template <typename PFP,typename V_ATT>
void computeNormalVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& normal, unsigned int thread = 0) ;


template <typename PFP, typename V_ATT>
typename PFP::REAL computeAngleBetweenNormalsOnEdge(typename PFP::MAP& map, Edge d, const V_ATT& position) ;

template <typename PFP, typename V_ATT, typename E_ATT>
void computeAnglesBetweenNormalsOnEdges(typename PFP::MAP& map, const V_ATT& position, E_ATT& angles, unsigned int thread = 0) ;



namespace Parallel
{

template <typename PFP,typename V_ATT>
void computeNormalVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& normal) ;

template <typename PFP, typename V_ATT, typename F_ATT>
void computeNormalFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_normal) ;

template <typename PFP, typename V_ATT, typename E_ATT>
void computeAnglesBetweenNormalsOnEdges(typename PFP::MAP& map, const V_ATT& position, E_ATT& angles) ;

}


} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Geometry/normal.hpp"

#endif
