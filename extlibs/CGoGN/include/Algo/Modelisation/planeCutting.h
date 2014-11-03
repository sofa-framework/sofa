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

#ifndef __PLANE_CUTTING_H__
#define __PLANE_CUTTING_H__

#include <math.h>
#include <vector>
#include "Geometry/plane_3d.h"
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversor/traversorCell.h"
#include "Algo/Modelisation/triangulation.h"
#include "Algo/Modelisation/subdivision.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Modelisation
{

/**
*/
template <typename PFP>
void planeCut(
	typename PFP::MAP& map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const Geom::Plane3D<typename PFP::REAL>& plane,
	CellMarker<typename PFP::MAP, FACE>& cmf_over,
	bool keepTriangles = false,
	bool with_unsew = true
);

template <typename PFP>
void planeCut2(
	typename PFP::MAP& map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const Geom::Plane3D<typename PFP::REAL>& plane,
	CellMarker<typename PFP::MAP, FACE>& cmf_over,
	bool with_unsew
);

} // namespace Modelisation

} // namespace Surface

namespace Volume
{

namespace Modelisation
{

template <typename PFP>
void planeCut(
	typename PFP::MAP& map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const Geom::Plane3D<typename PFP::REAL>& plane,
	CellMarker<typename PFP::MAP, FACE>& cmv_over,
	bool keepTetrahedra = false,
	bool with_unsew = true
);

} // namespace Modelisation

} // namespace Volume

} // namespace Algo

} // namespace CGoGN

#include "Algo/Modelisation/planeCutting.hpp"

#endif
