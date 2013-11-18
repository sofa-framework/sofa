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

#ifndef __MAP3MR_PM__
#define __MAP3MR_PM__

#include "Topology/map/embeddedMap3.h"
#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor3.h"

#include "Container/attributeContainer.h"

#include "Algo/DecimationVolumes/selector.h"
#include "Algo/DecimationVolumes/edgeSelector.h"
#include "Algo/DecimationVolumes/geometryApproximator.h"


#include "Algo/Multiresolution/filter.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace MR
{

template <typename PFP>
class Map3MR_PM
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

private:
	MAP& m_map ;
	VertexAttribute<VEC3>& m_position;

	bool m_initOk ;

	Algo::Volume::Decimation::EdgeSelector<PFP>* m_selector ;
	std::vector<Algo::Volume::Decimation::ApproximatorGen<PFP>*> m_approximators ;
	std::vector<Algo::Volume::Decimation::PredictorGen<PFP>*> m_predictors ;

	Algo::Volume::Decimation::Approximator<PFP, VEC3>* m_positionApproximator ;

	std::vector<Filter*> synthesisFilters ;
	std::vector<Filter*> analysisFilters ;

public:
	Map3MR_PM(MAP& map, VertexAttribute<VEC3>& position);

	~Map3MR_PM();

	//create a progressive mesh (a coarser level)
	void createPM(Algo::Volume::Decimation::SelectorType s, Algo::Volume::Decimation::ApproximatorType a) ;

	void addNewLevel(unsigned int percentWantedVertices);

	void collapseEdge(Dart d);

	//coarsen the mesh -> analysis
	void coarsen() ;

	//refine the mesh -> synthesis
	void refine() ;

	bool initOk() { return m_initOk; }

	void addSynthesisFilter(Filter* f) { synthesisFilters.push_back(f) ; }
	void addAnalysisFilter(Filter* f) { analysisFilters.push_back(f) ; }

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


#include "Algo/Multiresolution/Map3MR/map3MR_PM.hpp"

#endif
