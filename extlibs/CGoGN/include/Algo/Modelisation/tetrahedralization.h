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
bool isTetrahedron(typename PFP::MAP& the_map, Dart d, unsigned int thread=0);

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
void swap4To4(typename PFP::MAP& map, Dart d);

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
 */
template <typename PFP>
void swapGen3To2(typename PFP::MAP& map, Dart d);

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

} // namespace Tetrahedralization

}

} // namespace Modelisation

} // namespace Algo

} // namespace CGoGN

#include "Algo/Modelisation/tetrahedralization.hpp"

#endif
