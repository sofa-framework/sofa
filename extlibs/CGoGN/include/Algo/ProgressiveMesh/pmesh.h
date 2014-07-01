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

#ifndef __PMESH__
#define __PMESH__

#include "Algo/ProgressiveMesh/vsplit.h"

#include "Algo/Decimation/selector.h"
#include "Algo/Decimation/edgeSelector.h"
#include "Algo/Decimation/geometryApproximator.h"
#include "Algo/Decimation/geometryPredictor.h"
#include "Algo/Decimation/colorPerVertexApproximator.h"

#include "Utils/quantization.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace PMesh
{

template <typename PFP>
class ProgressiveMesh
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

private:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& position ;

	DartMarker<MAP>& inactiveMarker ;

	Algo::Surface::Decimation::Selector<PFP>* m_selector ;
	std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*> m_approximators ;
	std::vector<Algo::Surface::Decimation::PredictorGen<PFP>*> m_predictors ;
	std::vector<VSplit<PFP>*> m_splits ;
	unsigned int m_cur ;

	Algo::Surface::Decimation::Approximator<PFP, VEC3, EDGE>* m_positionApproximator ;

	bool m_initOk ;

	double m_detailAmount ;
	bool m_localFrameDetailVectors ;

	std::vector<VEC3> originalDetailVectors ;
	bool quantizationInitialized, quantizationApplied ;
	Algo::PMesh::Quantization<VEC3>* q ;

public:
	ProgressiveMesh(
		MAP& map,
		DartMarker<MAP>& inactive,
		Algo::Surface::Decimation::SelectorType s,
		Algo::Surface::Decimation::ApproximatorType a,
		VertexAttribute<VEC3, MAP>& position
	) ;
	ProgressiveMesh(
			MAP& map, DartMarker& inactive,
			Algo::Surface::Decimation::Selector<PFP>* selector, std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>& approximators,
			VertexAttribute<typename PFP::VEC3>& position) ;
	~ProgressiveMesh() ;

	bool initOk() { return m_initOk ; }

	void createPM(unsigned int percentWantedVertices) ;

	std::vector<VSplit<PFP>*>& splits() { return m_splits ; }
	Algo::Surface::Decimation::Selector<PFP>* selector() { return m_selector ; }
	std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>& approximators() { return m_approximators ; }
	std::vector<Algo::Surface::Decimation::PredictorGen<PFP>*>& predictors() { return m_predictors ; }

	void edgeCollapse(VSplit<PFP>* vs) ;
	void vertexSplit(VSplit<PFP>* vs) ;

	void coarsen() ;
	void refine() ;

	void gotoLevel(unsigned int goal) ;
	unsigned int& currentLevel() { return m_cur ; }
	unsigned int nbSplits() { return m_splits.size() ; }

	void recomputeApproxAndDetails() ;

	double detailAmount() { return m_detailAmount ; }
	void setDetailAmount(double a) ;

	void localizeDetailVectors() ;
	void globalizeDetailVectors() ;

	void quantizeDetailVectors(unsigned int nbClasses) ;
	void quantizeDetailVectors(float distortion) ;
	void resetDetailVectors() ;

//	float getDifferentialEntropy() { return q->getDifferentialEntropy() ; }
//	float getDiscreteEntropy() { return q->getDiscreteEntropy() ; }

//	void calculCourbeDebitDistortion(float distortion) ;

private:
	void initQuantization() ;
} ;

} //namespace PMesh

} // Surface

} //namespace Algo

} //namespace CGoGN

#include "Algo/ProgressiveMesh/pmesh.hpp"

#endif
