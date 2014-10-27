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

#ifndef __TETRAHEDRALIZATION_H__
#define __TETRAHEDRALIZATION_H__

//#include "tetgen/tetgen.h"
//#include <Topology/generic/parameters.h>
#include "Algo/Geometry/normal.h"
#include <set>

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Modelisation
{

//TODO change namespace
namespace Tetrahedralization
{


template <typename PFP>
class EarTriangulation
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;

protected:
	// forward declaration
	class VertexPoly;

	// multiset typedef for simple writing
	typedef std::multiset<VertexPoly,VertexPoly> VPMS;
	typedef typename VPMS::iterator VMPSITER;
	typedef NoTypeNameAttribute<VMPSITER> EarAttr ;

	class VertexPoly
	{
	public:
		Dart dart;
		float angle;
		float length;

        VertexPoly()
        {}

        VertexPoly(Dart d, float v, float l) : dart(d), angle(v), length(l)
        {}

        bool operator()(const VertexPoly& vp1, const VertexPoly& vp2)
        {
            if (fabs(vp1.angle - vp2.angle) < 0.2f)
                return vp1.length < vp2.length;
            return vp1.angle < vp2.angle;
        }
    };

protected:
	MAP& m_map;

	VertexAutoAttribute<EarAttr, MAP> m_dartEars;

	VertexAttribute<VEC3, MAP> m_position;

	std::vector<Dart> m_resTets;

	VPMS m_ears;

	bool inTriangle(const VEC3& P, const VEC3& normal, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc);

	void recompute2Ears(Dart d, const VEC3& normalPoly, bool convex);

	float computeEarInit(Dart d, const VEC3& normalPoly, float& val);

public:

	EarTriangulation(MAP& map) : m_map(map), m_dartEars(map)
	{
        m_position = map.template getAttribute<VEC3, VERTEX, MAP>("position");
	}

//	void trianguleFace(Dart d, DartMarker& mark);
	void trianguleFace(Dart d);

	void triangule(unsigned int thread = 0);

    std::vector<Dart> getResultingTets() const { return m_resTets; }
};


///**
//* subdivide a hexahedron into 5 tetrahedron
//*/
//template <typename PFP>
//void hexahedronToTetrahedron(typename PFP::MAP& map, Dart d);
//
///**
//* WARNING : assume all volumes to be hexahedrons
//* subdivide a hexahedron mesh into a tetrahedron mesh
//*/
//template <typename PFP>
//void hexahedronsToTetrahedrons(typename PFP::MAP& map);
//
//
//template <typename PFP>
//void tetrahedrizeVolume(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position);

/************************************************************************************************
 * 									Collapse / Split Operators									*
 ************************************************************************************************/

//!
/*!
 *
 */
template <typename PFP>
Dart splitVertex(typename PFP::MAP& map, std::vector<Dart>& vd);

/************************************************************************************************
 * 									Tetrahedron functions										*
 ************************************************************************************************/

//!
/*!
 *
 */
template <typename PFP>
bool isTetrahedron(typename PFP::MAP& map, Dart d, unsigned int thread = 0);

//!
/*!
 *
 */
template <typename PFP>
bool isTetrahedralization(typename PFP::MAP& map);


/************************************************************************************************
 *										Swap Functions 											*
 ************************************************************************************************/

//!
/*!
 *
 */
template <typename PFP>
Dart swap2To2(typename PFP::MAP& map, Dart d);

//!
/*!
 *
 */
template <typename PFP>
Dart swap4To4(typename PFP::MAP& map, Dart d);

//!
/*!
 *
 */
template <typename PFP>
Dart swap3To2(typename PFP::MAP& map, Dart d);

//!
/*!
 *
 */
template <typename PFP>
Dart swap2To3(typename PFP::MAP& map, Dart d);

//!
/*!
 *
 */
template <typename PFP>
Dart swap5To4(typename PFP::MAP& map, Dart d);

//!
/*!
 *  called edge removal (equivalent to G32)
 * Connect the vertex of dart d to each vertex of the polygonal face
 * @return A dart from the vertex that is incident to the tetrahedra created during the swap
 */
template <typename PFP>
Dart swapGen3To2(typename PFP::MAP& map, Dart d);


//!
/*!
 * Edge removal
 * Optimized version : do an ear cutting on the sandwiched polygonal face
 * @return : A dart of each tetrahedra created during the swap
 */
template <typename PFP>
std::vector<Dart> swapGen3To2Optimized(typename PFP::MAP& map, Dart d);

//!
/*!
 * called multi-face removal (equivalent to G23 )
 */
template <typename PFP>
void swapGen2To3(typename PFP::MAP& map, Dart d);

/************************************************************************************************
 *											Flip Functions 										*
 ************************************************************************************************/

//!
/*!
 *
 */
template <typename PFP>
Dart flip1To4(typename PFP::MAP& map, Dart d);

//!
/*!
 *
 */
template <typename PFP>
Dart flip1To3(typename PFP::MAP& map, Dart d);

/************************************************************************************************
 *									Bisection Functions 										*
 ************************************************************************************************/

//!
/*!
 *
 */
template <typename PFP>
Dart edgeBisection(typename PFP::MAP& map, Dart d);



//namespace Tetgen
//{

///**
// * generate tetrahedra based on an surface mesh object
// */
//template <typename PFP>
//bool tetrahedralize(const typename PFP::MAP2& map2, typename PFP::MAP3& map3, bool add_steiner_points_on_exterior_boundary,
//                    bool add_steiner_points_on_interior_boundary, double max_volume, double max_shape);

///**
// * generate tetrahedra based on an surface mesh object
// */
//template <typename PFP>
//bool process(const std::string& filename, typename PFP::MAP3& map3, bool add_steiner_points_on_exterior_boundary,
//             bool add_steiner_points_on_interior_boundary, double max_volume, double max_shape);

//} //namespace Tetgen




} // namespace Tetrahedralization

} // namespace Volume

} // namespace Modelisation

} // namespace Algo

} // namespace CGoGN

#include "Algo/Modelisation/tetrahedralization.hpp"

#endif
