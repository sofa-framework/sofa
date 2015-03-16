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

#include "Topology/ihmap/ihm3.h"
#include "Topology/generic/traversor/traversorCell.h"
#include "Topology/generic/traversor/traversor3.h"

#include <cmath>

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

template <typename PFP>
class IHM3
{

public:
    typedef typename PFP::MAP MAP ;

protected:
    MAP& m_map;
    bool shareVertexEmbeddings ;

    FunctorType* vertexVertexFunctor ;
    FunctorType* edgeVertexFunctor ;
    FunctorType* faceVertexFunctor ;
    FunctorType* volumeVertexFunctor ;

public:
    IHM3(MAP& map) ;

    /***************************************************
     *               CELLS INFORMATION                 *
     ***************************************************/

    //! Return the level of the edge of d in the current level map
    /*!
     */
    unsigned int edgeLevel(Dart d) ;

    //! Return the level of the face of d in the current level map
    /*!
     */
    unsigned int faceLevel(Dart d);

    //! Return the level of the volume of d in the current level map
    /*!
     */
    unsigned int volumeLevel(Dart d);

    //! Return the oldest dart of the face of d in the current level map
    /*!
     */
    Dart faceOldestDart(Dart d);

    //! Return the oldest dart of the volume of d in the current level map
    /*!
     */
    Dart volumeOldestDart(Dart d);

    //! Return true if the edge of d in the current level map
    //! has already been subdivided to the next level
    /*!
     */
    bool edgeIsSubdivided(Dart d) ;

    //! Return true if the edge of d in the current level map
    //! is subdivided to the next level,
    //! none of its resulting edges is in turn subdivided to the next level
    //! and the middle vertex is of degree 2
    /*!
     */
    bool edgeCanBeCoarsened(Dart d);

    //! Return true if the face of d in the current level map
    //! has already been subdivided to the next level
    /*!
     */
    bool faceIsSubdivided(Dart d) ;

    //!
    /*!
     */
    bool faceCanBeCoarsened(Dart d);

    //! Return true if the volume of d in the current level map
    //! has already been subdivided to the next level
    /*!
     */
    bool volumeIsSubdivided(Dart d);

    //!
    /*!
     */
    bool volumeIsSubdividedOnce(Dart d);

protected:
    /***************************************************
     *               SUBDIVISION                       *
     ***************************************************/
public:
    /**
     * subdivide the edge of d to the next level
     */
    void subdivideEdge(Dart d) ;

    /**
     * coarsen the edge of d from the next level
     */
    void coarsenEdge(Dart d) ;

    /**
     * subdivide the face of d to the next level
     */
    unsigned int subdivideFace(Dart d, bool triQuad);

    /**
     * coarsen the face of d from the next level
     */
    void coarsenFace(Dart d) ;


    //! Subdivide the volume of d to hexahedral cells
    /*! @param d Dart from the volume
     */
	Dart subdivideVolume(Dart d, bool triQuad = true, bool OneLevelDifference = true);

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

	void coarsenVolume(Dart d) ;

    /**
     * vertices attributes management
     */
    void setVertexVertexFunctor(FunctorType* f) { vertexVertexFunctor = f ; }
    void setEdgeVertexFunctor(FunctorType* f) { edgeVertexFunctor = f ; }
    void setFaceVertexFunctor(FunctorType* f) { faceVertexFunctor = f ; }
    void setVolumeVertexFunctor(FunctorType* f) { volumeVertexFunctor = f ; }

} ;

} // namespace Adaptive

} // namespace Primal

} // namespace MR

} // namespace Volume

} // namespace Algo

} // namespace CGoGN

#include "Algo/Multiresolution/IHM3/ihm3_PrimalAdapt.hpp"

#endif
