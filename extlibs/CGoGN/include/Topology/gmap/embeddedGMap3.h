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

#ifndef __EMBEDDED_GMAP3_H__
#define __EMBEDDED_GMAP3_H__

#include "Topology/gmap/gmap3.h"
#include "Topology/generic/mapImpl/mapMono.h"

namespace CGoGN
{

/**
* Class of 3-dimensional G-maps
* with managed embeddings
*/
class EmbeddedGMap3 : public GMap3<MapMono>
{
	EmbeddedGMap3(const EmbeddedGMap3& m):GMap3<MapMono>(m) {}
public:
	typedef MapMono IMPL;
	typedef GMap3<MapMono> TOPO_MAP;

	static const unsigned int DIMENSION = TOPO_MAP::DIMENSION ;

	EmbeddedGMap3() {}

	/*!
	 *
	 */
	virtual Dart deleteVertex(Dart d);

	//! Cut the edge of d
	/*! @param d a dart of the edge to cut
	 */
	virtual Dart cutEdge(Dart d);

	/*! The attributes attached to the edge of d are kept on the resulting edge
	 *  @param d a dart of the edge to cut
	 */
	virtual bool uncutEdge(Dart d);

	//!
	/*!
	 */
	virtual Dart deleteEdge(Dart d);

	/*!
	 *
	 */
	virtual void splitFace(Dart d, Dart e);

	/*!
	 *
	 */
	virtual void sewVolumes(Dart d, Dart e, bool withBoundary = true);

	/*!
	 *
	 */
	virtual void unsewVolumes(Dart d);

	/*!
	 *
	 */
	virtual bool mergeVolumes(Dart d);

	/*!
	 *
	 */
	virtual void splitVolume(std::vector<Dart>& vd);

	/**
	 * No attribute is attached to the new volume
	 */
	virtual unsigned int closeHole(Dart d, bool forboundary = true);

	/*!
	 *
	 */
	virtual bool check();
} ;

} // namespace CGoGN

#endif
