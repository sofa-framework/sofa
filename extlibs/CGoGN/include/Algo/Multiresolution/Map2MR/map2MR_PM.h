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

#ifndef __MAP2MR_PM__
#define __MAP2MR_PM__

#include "Topology/map/embeddedMap2.h"
#include "Topology/generic/traversor/traversorCell.h"
#include "Topology/generic/traversor/traversor2.h"

#include "Container/attributeContainer.h"

#include "Algo/Decimation/selector.h"
#include "Algo/Decimation/edgeSelector.h"
#include "Algo/Decimation/geometryApproximator.h"
#include "Algo/Decimation/geometryPredictor.h"
#include "Algo/Decimation/lightfieldApproximator.h"

#include "Algo/Multiresolution/filter.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MR
{

template <typename PFP>
class Map2MR_PM
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

private:
	MAP& m_map ;
	VertexAttribute<VEC3>& m_position;

	bool m_initOk ;

	Algo::Surface::Decimation::EdgeSelector<PFP>* m_selector ;
	std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*> m_approximators ;
	std::vector<Algo::Surface::Decimation::PredictorGen<PFP>*> m_predictors ;

	Algo::Surface::Decimation::Approximator<PFP, VEC3, EDGE>* m_positionApproximator ;

	std::vector<Algo::MR::Filter*> synthesisFilters ;
	std::vector<Algo::MR::Filter*> analysisFilters ;

public:
	Map2MR_PM(MAP& map, VertexAttribute<VEC3>& position);

	~Map2MR_PM();

	//create a progressive mesh (a coarser level)
	void createPM(Algo::Surface::Decimation::SelectorType s, Algo::Surface::Decimation::ApproximatorType a) ;

	void addNewLevel(unsigned int percentWantedVertices);

	void collapseEdge(Dart d);

	//coarsen the mesh -> analysis
	void coarsen() ;

	//refine the mesh -> synthesis
	void refine() ;

	bool initOk() { return m_initOk; }

	void addSynthesisFilter(Algo::MR::Filter* f) { synthesisFilters.push_back(f) ; }
	void addAnalysisFilter(Algo::MR::Filter* f) { analysisFilters.push_back(f) ; }

	void clearSynthesisFilters() { synthesisFilters.clear() ; }
	void clearAnalysisFilters() { analysisFilters.clear() ; }

	/**
	 * Given the vertex of d in the current level,
	 * return a dart of from the vertex of the current level
	 */
	Dart vertexOrigin(Dart d) ;

//	/**
//	 * Return the level of the vertex of d in the current level map
//	 */
//	unsigned int vertexLevel(Dart d);
} ;

} // namespace Multiresolution

} // namespace Surface

} // namespace Algo

} // namespace CGoGN


#include "Algo/Multiresolution/Map2MR/map2MR_PM.hpp"

#endif
