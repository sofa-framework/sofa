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

#ifndef __EMBEDDED_MAP3_H__
#define __EMBEDDED_MAP3_H__

#include "Topology/map/map3.h"

namespace CGoGN
{

/*! Class of 3-dimensional maps with managed embeddings
 */
class EmbeddedMap3 : public Map3
{
public:
	typedef Map3 TOPO_MAP;

	static const unsigned int DIMENSION = 3 ;



	//!
	/*!
	 *
	 */
	virtual Dart splitVertex(std::vector<Dart>& vd);

	//!
	/*!
	 */
	virtual Dart deleteVertex(Dart d);

	//! No attribute is attached to the new vertex
	/*! The attributes attached to the old edge are duplicated on both resulting edges
	 *  @param d a dart
	 */
	virtual Dart cutEdge(Dart d);

	//! The attributes attached to the edge of d are kept on the resulting edge
	/*!  @param d a dart of the edge to cut
	 */
	virtual bool uncutEdge(Dart d);

	//!
	/*!
	 */
	virtual Dart deleteEdge(Dart d);

	//!
	/*!
	 */
	bool edgeCanCollapse(Dart d);

	//!
	/*!
	 */
	virtual Dart collapseEdge(Dart d, bool delDegenerateVolumes=true);

	//!
	/*!
	 */
//	virtual bool collapseDegeneratedFace(Dart d);

	//!
	/*!
	 */
	virtual void splitFace(Dart d, Dart e);

	/**
	 * The attributes attached to the face of dart d are kept on the resulting face
	 */
	virtual bool mergeFaces(Dart d);

	//!
	/*!
	 *
	 */
	virtual Dart collapseFace(Dart d, bool delDegenerateVolumes = true);

	//!
	/*!
	 */
	virtual void sewVolumes(Dart d, Dart e, bool withBoundary = true);

	//!
	/*!
	 */
	virtual void unsewVolumes(Dart d, bool withBoundary = true);

	//!
	/*!
	 */
	virtual bool mergeVolumes(Dart d, bool deleteFace = true);

	//!
	/*!
	 */
	virtual void splitVolume(std::vector<Dart>& vd);

	//!
	virtual void splitVolumeWithFace(std::vector<Dart>& vd, Dart d);


	//!
	/*!
	 */
	virtual Dart collapseVolume(Dart d, bool delDegenerateVolumes = true);

	//!
	/*! No attribute is attached to the new volume
	 */
	virtual unsigned int closeHole(Dart d, bool forboundary = true);

	//!
	/*!
	 */
	virtual bool check();
} ;

} // namespace CGoGN

#endif
