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

#ifndef __MAP3MR_PRIMAL_ADAPT__
#define __MAP3MR_PRIMAL_ADAPT__

#include "Topology/map/embeddedMap3.h"
#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor3.h"
#include "Algo/Modelisation/tetrahedralization.h"
#include <limits>

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace MR
{

namespace Primal
{

namespace Adaptive
{

/*! \brief The class of adaptive 3-map MR
 */

template <typename PFP>
class Map3MR
{

public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	MAP& m_map;
	bool shareVertexEmbeddings;

	FunctorType* vertexVertexFunctor ;
	FunctorType* edgeVertexFunctor ;
	FunctorType* faceVertexFunctor ;
	FunctorType* volumeVertexFunctor ;

public:
	Map3MR(MAP& map);

	/*! @name Cells informations
	 *
	 *************************************************************************/
	//@{
	//! Return the level of the edge of d in the current level map
	/* @param d Dart from the edge
	 */
	unsigned int edgeLevel(Dart d);

	//! Return the level of the face of d in the current level map
	/* @param d Dart from the face
	 */
	unsigned int faceLevel(Dart d);

	//! Return the level of the volume of d in the current level map
	/* @param d Dart from the volume
	 */
	unsigned int volumeLevel(Dart d);

	//! Return the oldest dart of the face of d in the current level map
	/* @param d Dart from the edge
	 */
	Dart faceOldestDart(Dart d);

	//! Return the level of the edge of d in the current level map
	/* @param d Dart from the edge
	 */
	Dart volumeOldestDart(Dart d);

	//! Return true if the edge of d in the current level map
	//! has already been subdivided to the next level
	/*! @param d Dart from the edge
	 */
	bool edgeIsSubdivided(Dart d) ;

	//! Return true if the face of d in the current level map
	//! has already been subdivided to the next level
	/*! @param d Dart from the face
	 */
	bool faceIsSubdivided(Dart d) ;

	//! Return true if the volume of d in the current level map
	//! has already been subdivided to the next level
	/*! @param d Dart from the volume
	 */
	bool volumeIsSubdivided(Dart d);
	//@}

//protected:
	/*! @name Topological helping functions
	 *
	 *************************************************************************/
	//@{
	//!
	/*!
	 */
	void swapEdges(Dart d, Dart e);

	void splitVolume(std::vector<Dart>& vd);

	Dart cutEdge(Dart d) ;

	void splitFace(Dart d, Dart e) ;
	//@}

	/*! @name Subdivision
	 *
	 *************************************************************************/
	//@{
	//! Subdivide the edge of d to the next level
	/*! @param d Dart from the edge
	 */
	void subdivideEdge(Dart d) ;

	//!
	/*!
	 */
	void coarsenEdge(Dart d);

	//! Subdivide the edge of d to the next level
	/*! @param d Dart frome the face
	 */
	void subdivideFace(Dart d, bool triQuad) ;

	//!
	/*!
	 */
	void coarsenFace(Dart d);

public:
	//! Subdivide the volume of d to hexahedral cells
	/*! @param d Dart from the volume
	 */
	unsigned int subdivideVolume(Dart d, bool triQuad = true, bool OneLevelDifference = true);

	/*!
	 * \brief subdivideHexa
	 *
	 * Detailed description of the function
	 * \param d
	 * \param OneLevelDifference
	 * \return
	 */
	unsigned int subdivideHexa(Dart d, bool OneLevelDifference = true);

	//! Subdivide the volume of d to hexahedral cells
	/*! @param d Dart from the volume
	 */
	void subdivideVolumeTetOcta(Dart d) ;
	//@}

	/*! @name Vertices Attributes management
	 *
	 *************************************************************************/
	//@{
	void setVertexVertexFunctor(FunctorType* f) { vertexVertexFunctor = f ; }
	void setEdgeVertexFunctor(FunctorType* f) { edgeVertexFunctor = f ; }
	void setFaceVertexFunctor(FunctorType* f) { faceVertexFunctor = f ; }
	void setVolumeVertexFunctor(FunctorType* f) { volumeVertexFunctor = f ; }
	//@}

	unsigned int subdivideHexa2(Dart d, bool OneLevelDifference = true);
	void subdivideFace2(Dart d, bool triQuad);
};

} // namespace Adaptive

} // namespace Primal

} // namespace MR

} // namespace Volume

} // namespace Algo

} // namespace CGoGN

#include "Algo/Multiresolution/Map3MR/map3MR_PrimalAdapt.hpp"

#endif /* __MAP3MR_PRIMAL__ */
