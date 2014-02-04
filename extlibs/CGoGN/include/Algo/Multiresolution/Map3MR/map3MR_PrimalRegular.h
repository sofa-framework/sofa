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

#ifndef __MAP3MR_PRIMAL_REGULAR_
#define __MAP3MR_PRIMAL_REGULAR_

#include "Topology/map/embeddedMap3.h"
#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor3.h"

#include "Algo/Multiresolution/filter.h"
#include "Algo/Modelisation/tetrahedralization.h"
#include "Algo/Modelisation/subdivision.h"

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

namespace Regular
{

/*! \brief The class of regular 3-map MR
 */

template <typename PFP>
class Map3MR
{

public:
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

protected:
	MAP& m_map;
	bool shareVertexEmbeddings;

	std::vector<Algo::MR::Filter*> synthesisFilters ;
	std::vector<Algo::MR::Filter*> analysisFilters ;


public:
	Map3MR(MAP& map);

	~Map3MR();

private:
	/*! @name Topological helping functions
	 *
	 *************************************************************************/
	//@{


	void splitSurfaceInVolume(std::vector<Dart>& vd, bool firstSideClosed = true, bool secondSideClosed = false);
	//@}

public:

	void swapEdges(Dart d, Dart e);


    Dart swap2To2(Dart d);
    void swap4To4(Dart d);
	Dart swap2To3(Dart d);
	void swapGen3To2(Dart d);

	/*! @name Level creation
	 *
	 *************************************************************************/
	//@{
	//!
	/*
	 */
	void addNewLevelTetraOcta();

	//!
	/*
	 */
	void addNewLevelHexa();

	//!
	/*
	 */
	void addNewLevel();

	void addNewLevelSqrt3(bool embedNewVertices = false);

	void addNewLevelSqrt3(bool embedNewVertices, VertexAttribute<typename PFP::VEC3> position);
    void addNewLevelSqrt3Geom(bool embedNewVertices, VertexAttribute<typename PFP::VEC3> position);

	//!
	/*
	 */
	//void addNewLevel(bool embedNewVertices);
	//@}

	/*! @name Geometry modification
	 *  Analysis / Synthesis
	 *************************************************************************/
	//@{

	void setSharingVertexEmbeddings(bool b) { shareVertexEmbeddings = b; }

	void addSynthesisFilter(Algo::MR::Filter* f) { synthesisFilters.push_back(f) ; }
	void addAnalysisFilter(Algo::MR::Filter* f) { analysisFilters.push_back(f) ; }

	void clearSynthesisFilters() { synthesisFilters.clear() ; }
	void clearAnalysisFilters() { analysisFilters.clear() ; }

	void analysis() ;
	void synthesis() ;
	//@}
};

} // namespace Regular

} // namespace Primal

} // namespace MR

} // namespace Volume

} // namespace Algo

} // namespace CGoGN

#include "Algo/Multiresolution/Map3MR/map3MR_PrimalRegular.hpp"

#endif /* __MAP3MR_PRIMAL__ */
