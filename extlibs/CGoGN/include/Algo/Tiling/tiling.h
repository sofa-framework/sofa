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

#ifndef _TILING_H_
#define _TILING_H_

#include "Geometry/transfo.h"
#include "Topology/generic/cellmarker.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Tilings
{

/*! \brief The class of regular tiling
 */
template <typename PFP>
class Tiling
{
	typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;

protected:
    /**
    * Map in which we are working
    */
    MAP& m_map;

    unsigned int m_nx, m_ny, m_nz;

    VEC3 m_center;

    /**
        * Reference dart of Polyhedron
        */
    Dart m_dart;

    /**
    * Table of vertex darts (one dart per vertex)
    * Order depend on tiling kind
    */
    std::vector<Dart> m_tableVertDarts;

public:
    Tiling(MAP& map, unsigned int x, unsigned int y, unsigned int z):
        m_map(map),
        m_nx(x), m_ny(y), m_nz(z)
	{}

    Tiling(MAP& map) :
        m_map(map),
        m_nx(-1), m_ny(-1), m_nz(-1)
	{}

    Tiling(const Tiling<PFP>& t1, const Tiling<PFP> t2);

    /**
    * get the table of darts (one per vertex)
    */
    std::vector<Dart>& getVertexDarts() { return m_tableVertDarts; }

	void computeCenter(VertexAttribute<VEC3, MAP>& position);

    //void Polyhedron<PFP>::transform(float* matrice)
	void transform(VertexAttribute<VEC3, MAP>& position, const Geom::Matrix44f& matrice);

	void mark(CellMarker<MAP, VERTEX>& m);

    /*
	* get the reference dart
	*/
    Dart getDart() { return m_dart; }

	bool exportPositions(const VertexAttribute<VEC3, MAP>& position, const char* filename);

	//	/**
	//	* mark all embedded vertices of the Polyhedron
	//	* @param m the marker to use
	//	*/
	////	void markEmbVertices(Mark m);

	//	/**
	//	* test if a vertex is in the Polyhedron
	//	* @param d a dart of the vertex to test
	//	*/
	//	bool containVertex(Dart d);
};

} // namespace Tilings

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Tiling/tiling.hpp"

#endif //_TILING_H_
