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

#ifndef __EMBEDDED_GMAP2_H__
#define __EMBEDDED_GMAP2_H__

#include "Topology/gmap/gmap2.h"
#include "Topology/generic/mapImpl/mapMono.h"

namespace CGoGN
{

/**
* Class of 2-dimensional G-maps
* with managed embeddings
*/
class EmbeddedGMap2 : public GMap2<MapMono>
{
	EmbeddedGMap2(const EmbeddedGMap2& m):GMap2<MapMono>(m) {}
public:
	typedef MapMono IMPL;
	typedef GMap2<MapMono> TOPO_MAP;

	static const unsigned int DIMENSION = TOPO_MAP::DIMENSION ;

	EmbeddedGMap2() {}

	/**
	 *	create a new face with managed embeddings
	 */
	Dart newFace(unsigned int nbEdges, bool withBoundary = true) ;

	/**
	 * The attributes attached to the old vertex are duplicated on both resulting vertices
	 * No attribute is attached to the new edge
	 */
	void splitVertex(Dart d, Dart e) ;

	/**
	 * The attributes attached to the face of d are kept on the resulting face
	 */
	Dart deleteVertex(Dart d) ;

	/**
	 * No attribute is attached to the new vertex
	 * The attributes attached to the old edge are duplicated on both resulting edges
	 */
	Dart cutEdge(Dart d) ;

	/**
	 * The attributes attached to the edge of d are kept on the resulting edge
	 */
	bool uncutEdge(Dart d) ;

	/**
	 * Check if the edge of d can be collapsed or not based on some topological conditions
	 * @param d a dart of the edge to test
	 * @return true if the edge can be collapsed, false otherwise
	 */
	bool edgeCanCollapse(Dart d) ;

	/**
	 * The attributes attached to the vertex of dart d are kept on the resulting vertex
	 * See 'collapseDegeneratedFace' to see what can happen to edges attributes
	 * Nothing has to be done for the faces (some degenerate ones can be deleted)
	 */
	Dart collapseEdge(Dart d, bool delDegenerateFaces = true) ;

	/**
	 * No cell is created or deleted
	 */
	bool flipEdge(Dart d) ;

	/**
	 * No cell is created or deleted
	 */
	bool flipBackEdge(Dart d) ;

//	/**
//	 * The attributes attached to the vertex of dart d are kept on the resulting vertex
//	 * The attributes attached to the face of dart d are overwritten on the face of dart e
//	 */
//	virtual void insertEdgeInVertex(Dart d, Dart e);
//
//	/**
//	 * The attributes attached to the vertex of dart d are kept on the resulting vertex
//	 * The attributes attached to the face of dart d are overwritten on the face of dart e
//	 */
//	virtual void removeEdgeFromVertex(Dart d);

	/**
	 * The attributes attached to the vertices of the edge of d are kept on the vertices of the resulting edge
	 * The attributes attached to the edge of d are kept on the resulting edge
	 */
	void sewFaces(Dart d, Dart e, bool withBoundary = true) ;

	/**
	 * The attributes attached to the vertices of the old edge of d are duplicated on the vertices of both resulting edges
	 * The attributes attached to the old edge are duplicated on both resulting edges
	 */
	void unsewFaces(Dart d) ;

	/**
	 * The attributes attached to the edge of d are kept on the resulting edge
	 */
	bool collapseDegeneratedFace(Dart d);

	/**
	 * No attribute is attached to the new edge
	 * The attributes attached to the old face are duplicated on both resulting faces
	 */
	void splitFace(Dart d, Dart e) ;

	/**
	 * The attributes attached to the face of dart d are kept on the resulting face
	 */
	bool mergeFaces(Dart d) ;

	/**
	 * The attributes attached to the vertices of the face of d are kept on the resulting vertices
	 * The attributes attached to the edges of the face of d are kept on the resulting edges
	 */
	bool mergeVolumes(Dart d, Dart e) ;

	/**
	 * No attribute is attached to the new face
	 */
	unsigned int closeHole(Dart d, bool forboundary = true);

	bool check() ;
} ;

} // namespace CGoGN

#endif
