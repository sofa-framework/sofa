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

#ifndef __EMBEDDED_MAP2_MR_H__
#define __EMBEDDED_MAP2_MR_H__

#include "Topology/map/map2.h"
#include "Topology/generic/mapImpl/mapMulti.h"

namespace sofa {
namespace cgogn_plugin {
namespace test {
    class CGoGN_test ;
}
}
}

namespace CGoGN
{

/**
* Class of 2-dimensional maps
* with managed embeddings
*/
class EmbeddedMap2_MR : public Map2<MapMulti>
{
    friend class ::sofa::cgogn_plugin::test::CGoGN_test;
public:
	typedef MapMulti IMPL;
	typedef Map2<MapMulti> TOPO_MAP;

	static const unsigned int DIMENSION = TOPO_MAP::DIMENSION ;

	/*
	 */
	virtual Dart newPolyLine(unsigned int nbEdges) ;

	/*
	 */
	virtual Dart newFace(unsigned int nbEdges, bool withBoundary = true) ;

	/**
	 * The attributes attached to the old vertex are duplicated on both resulting vertices
	 */
	virtual void splitVertex(Dart d, Dart e) ;

	/**
	 * The attributes attached to the face of d are kept on the resulting face
	 */
	virtual Dart deleteVertex(Dart d) ;

	/**
	 * The attributes attached to the old edge are duplicated on both resulting edges
	 */
	virtual Dart cutEdge(Dart d) ;

	/**
	 * The attributes attached to the edge of d are kept on the resulting edge
	 */
	virtual bool uncutEdge(Dart d) ;

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
	virtual Dart collapseEdge(Dart d, bool delDegenerateFaces = true) ;

	/**
	 * No cell is created or deleted
	 */
	virtual bool flipEdge(Dart d) ;

	/**
	 * No cell is created or deleted
	 */
	virtual bool flipBackEdge(Dart d) ;

	/*
	 *
	 */
	virtual void swapEdges(Dart d, Dart e);

	/**
	 * The attributes attached to the vertex of dart d are kept on the resulting vertex
	 * The attributes attached to the face of dart d are overwritten on the face of dart e
	 */
	virtual void insertEdgeInVertex(Dart d, Dart e);

	/**
	 * The attributes attached to the vertex of dart d are kept on the resulting vertex
	 * The attributes attached to the face of dart d are overwritten on the face of dart e
	 */
	virtual bool removeEdgeFromVertex(Dart d);

	/**
	 * The attributes attached to the vertices of the edge of d are kept on the vertices of the resulting edge
	 * The attributes attached to the edge of d are kept on the resulting edge
	 */
	virtual void sewFaces(Dart d, Dart e, bool withBoundary = true) ;

	/**
	 * The attributes attached to the vertices of the old edge of d are duplicated on the vertices of both resulting edges
	 * The attributes attached to the old edge are duplicated on both resulting edges
	 */
	virtual void unsewFaces(Dart d, bool withBoundary = true) ;

	/**
	 * The attributes attached to the edge of d are kept on the resulting edge
	 */
	virtual bool collapseDegeneratedFace(Dart d);

	/**
	 * The attributes attached to the old face are duplicated on both resulting faces
	 */
	virtual void splitFace(Dart d, Dart e) ;

	/**
	 * The attributes attached to the face of dart d are kept on the resulting face
	 */
	virtual bool mergeFaces(Dart d) ;

	/**
	 * The attributes attached to the vertices of the face of d are kept on the resulting vertices
	 * The attributes attached to the edges of the face of d are kept on the resulting edges
	 */
	virtual bool mergeVolumes(Dart d, Dart e, bool deleteFace = true) ;

	/**
	 *
	 */
	virtual void splitSurface(std::vector<Dart>& vd, bool firstSideClosed = true, bool secondSideClosed = true);

	/**
	 *
	 */
	virtual unsigned int closeHole(Dart d, bool forboundary = true);

	virtual bool check() ;
} ;

} // namespace CGoGN

#endif
