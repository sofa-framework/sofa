/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2013, IGG Team, ICube, University of Strasbourg           *
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

#ifndef __DECIMATION_H__
#define __DECIMATION_H__

#include "Algo/Decimation/edgeSelector.h"
#include "Algo/Decimation/halfEdgeSelector.h"
#include "Algo/Decimation/geometryApproximator.h"
#include "Algo/Decimation/colorPerVertexApproximator.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

/**
 * \fn decimate
 * Function that decimates the provided mesh through successive edge collapses
 * by using a declared selector and approximator type (see \file approximator.h and \file selector.h).
 *
 * \param map the map to decimate
 * \param s the SelectorType
 * \param a the ApproximatorType
 * \param position the vertex position embeddings
 * \param nbWantedVertices the aimed amount of vertices after decimation
 * \param edgeErrors will (if not null) contain the edge errors computed by the approximator/selector (default NULL)
 * \param callback_wrapper a callback function for progress monitoring (default NULL)
 * \param callback_object the object to call the callback on (default NULL)
 *
 * \return >= 0 if finished correctly : 1 if no more edges are collapsible, 0 is nbWantedVertices achieved, -1 if the initialisation of the selector failed
 */
template <typename PFP>
int decimate(
	typename PFP::MAP& map,
	SelectorType s,
	ApproximatorType a,
	std::vector<VertexAttribute<typename PFP::VEC3, typename PFP::MAP>*>& attribs,
	unsigned int nbWantedVertices,
	EdgeAttribute<typename PFP::REAL, typename PFP::MAP>* edgeErrors = NULL,
	void (*callback_wrapper)(void*, const void*) = NULL, void* callback_object = NULL
) ;

/**
 *\fn decimate
 * Function that decimates the provided mesh through successive edge collapses
 * by providing the selector and the approximators
 *
 * \param map the map to decimate
 * \param s the selector
 * \param a a vector containing the approximators
 * \param nbWantedVertices the aimed amount of vertices after decimation
 * \param recomputePriorityList if false and a priority list exists, it is not recomputed
 * \param edgeErrors will (if not null) contain the edge errors computed by the approximator/selector (default NULL)
 * \param callback_wrapper a callback function for progress monitoring (default NULL)
 * \param callback_object the object to call the callback on (default NULL)
 *
 * \return >= 0 if finished correctly : 1 if no more edges are collapsible, 0 is nbWantedVertices achieved, -1 if the initialisation of the selector failed
 */
template <typename PFP>
int decimate(
	typename PFP::MAP& map,
	Selector<PFP>* s,
	std::vector<ApproximatorGen<PFP>*>& a,
	unsigned int nbWantedVertices,
	bool recomputePriorityList = true,
	EdgeAttribute<typename PFP::REAL, typename PFP::MAP>* edgeErrors = NULL,
	void (*callback_wrapper)(void*, const void*) = NULL, void* callback_object = NULL
) ;

} // namespace Decimation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Decimation/decimation.hpp"

#endif
