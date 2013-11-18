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

#include "Algo/Geometry/localFrame.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace VPMesh
{

template <typename PFP>
VolumetricProgressiveMesh<PFP>::VolumetricProgressiveMesh(
		MAP& map, DartMarker& inactive,
		Algo::Decimation::SelectorType s, Algo::Decimation::ApproximatorType a,
		VertexAttribute<typename PFP::VEC3>& position
	) :
	m_map(map), positionsTable(position), inactiveMarker(inactive), dartSelect(inactiveMarker)
{
	CGoGNout << "  creating approximator and predictor.." << CGoGNflush ;
	switch(a)
	{
		case Algo::Decimation::A_QEM : {
			m_approximators.push_back(new Algo::Decimation::Approximator_QEM<PFP>(m_map, positionsTable)) ;
			break ; }
		default :
			CGoGNout << "not yet implemented" << CGoGNendl;
			break;
	}
	CGoGNout << "..done" << CGoGNendl ;

	CGoGNout << "  creating selector.." << CGoGNflush ;
	switch(s)
	{
		case Algo::Decimation::S_QEM : {
			m_selector = new Algo::Decimation::EdgeSelector_QEM<PFP>(m_map, positionsTable, m_approximators, dartSelect) ;
			break ; }
		default:
			CGoGNout << "not yet implemented" << CGoGNendl;
			break;
	}
	CGoGNout << "..done" << CGoGNendl ;

	m_initOk = true ;

	CGoGNout << "  initializing approximators.." << CGoGNflush ;
	for(typename std::vector<Algo::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
	{
		if(! (*it)->init())
			m_initOk = false ;
		if((*it)->getApproximatedAttributeName() == "position")
			m_positionApproximator = reinterpret_cast<Algo::Decimation::Approximator<PFP, VEC3>*>(*it) ;
	}
	CGoGNout << "..done" << CGoGNendl ;

	CGoGNout << "  initializing predictors.." << CGoGNflush ;
	for(typename std::vector<Algo::Decimation::PredictorGen<PFP>*>::iterator it = m_predictors.begin(); it != m_predictors.end(); ++it)
		if(! (*it)->init())
			m_initOk = false ;
	CGoGNout << "..done" << CGoGNendl ;

	CGoGNout << "  initializing selector.." << CGoGNflush ;
	m_initOk = m_selector->init() ;
	CGoGNout << "..done" << CGoGNendl ;
}

template <typename PFP>
VolumetricProgressiveMesh<PFP>::~VolumetricProgressiveMesh()
{
	for(unsigned int i = 0; i < m_splits.size(); ++i)
		delete m_splits[i] ;
	if(m_selector)
		delete m_selector ;
	for(typename std::vector<Algo::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
		delete (*it) ;
	for(typename std::vector<Algo::Decimation::PredictorGen<PFP>*>::iterator it = m_predictors.begin(); it != m_predictors.end(); ++it)
		delete (*it) ;
}

template <typename PFP>
void VolumetricProgressiveMesh<PFP>::createPM(unsigned int percentWantedVertices)
{
	unsigned int nbVertices = m_map.template getNbOrbits<VERTEX>() ;
	unsigned int nbWantedVertices = nbVertices * percentWantedVertices / 100 ;
	CGoGNout << "  creating PM (" << nbVertices << " vertices).." << /* flush */ CGoGNendl ;

	bool finished = false ;
	Dart d ;
	while(!finished)
	{
		if(!m_selector->nextEdge(d))
			break ;

		--nbVertices ;
		Dart d2 = m_map.phi2(m_map.phi_1(d)) ;
		Dart dd2 = m_map.phi2(m_map.phi_1(m_map.phi2(d))) ;

		VSplit<PFP>* vs = new VSplit<PFP>(m_map, d, dd2, d2) ;	// create new VSplit node
		m_splits.push_back(vs) ;								// and store it

		for(typename std::vector<Algo::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
		{
			(*it)->approximate(d) ;					// compute approximated attributes with its associated detail
			(*it)->saveApprox(d) ;
		}

		m_selector->updateBeforeCollapse(d) ;		// update selector

		edgeCollapse(vs) ;							// collapse edge

		unsigned int newV = m_map.template setOrbitEmbeddingOnNewCell<VERTEX>(d2) ;
		unsigned int newE1 = m_map.template setOrbitEmbeddingOnNewCell<EDGE>(d2) ;
		unsigned int newE2 = m_map.template setOrbitEmbeddingOnNewCell<EDGE>(dd2) ;
		vs->setApproxV(newV) ;
		vs->setApproxE1(newE1) ;
		vs->setApproxE2(newE2) ;

		for(typename std::vector<Algo::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
			(*it)->affectApprox(d2);				// affect data to the resulting vertex

		m_selector->updateAfterCollapse(d2, dd2) ;	// update selector

		if(nbVertices <= nbWantedVertices)
			finished = true ;
	}
	delete m_selector ;
	m_selector = NULL ;

	m_cur = m_splits.size() ;
	CGoGNout << "..done (" << nbVertices << " vertices)" << CGoGNendl ;






//	unsigned int nbVertices = m_map.template getNbOrbits<VERTEX>() ;
//	unsigned int nbWantedVertices = nbVertices * percentWantedVertices / 100 ;
//	CGoGNout << "  creating PM (" << nbVertices << " vertices).." << /* flush */ CGoGNendl ;
//
//	bool finished = false ;
//	Dart d ;
//	while(!finished)
//	{
//		//Next Operator to perform
//		Algo::DecimationVolumes::Operator<PFP> *op;
//
//		if((op = m_selector->nextOperator()) == NULL)
//			break;
//
//		m_nodes->add(op);
//
//		for(typename std::vector<Algo::DecimationVolumes::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
//		{
//			(*it)->approximate(op) ;					// compute approximated attributes with its associated detail
//			(*it)->saveApprox(op) ;
//		}
//
//		//Update the selector before performing operation
//		if(!m_selector->updateBeforeOperation(op))
//			break;
//
//		nbVertices -= op->collapse(m_map, positionsTable);
//
//		for(typename std::vector<Algo::DecimationVolumes::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
//			(*it)->affectApprox(op);				// affect data to the resulting vertex
//
//		m_selector->updateAfterOperation(op) ;	// update selector
//
//		if(nbVertices <= 3) //<= nbWantedVertices)
//			finished = true ;
//	}
//
//	m_selector->finish() ;
//
//	delete m_selector ;
//	m_selector = NULL ;
//
//	m_level = m_nodes->size() ;
//	CGoGNout << "..done (" << nbVertices << " vertices)" << CGoGNendl ;
}

template <typename PFP>
void VolumetricProgressiveMesh<PFP>::edgeCollapse(VSplit<PFP>* vs)
{
	Dart d = vs->getEdge() ;
	Dart dd = m_map.phi2(d) ;

	inactiveMarker.markOrbit<FACE>(d) ;
	inactiveMarker.markOrbit<FACE>(dd) ;

	//m_map.extractTrianglePair(d) ;
	m_map.collapseEdge(d);
}

template <typename PFP>
void VolumetricProgressiveMesh<PFP>::vertexSplit(VSplit<PFP>* vs)
{
	Dart d = vs->getEdge() ;
	Dart dd = m_map.phi2(d) ;
	Dart d2 = vs->getLeftEdge() ;
	Dart dd2 = vs->getRightEdge() ;

	m_map.insertTrianglePair(d, d2, dd2) ;

	inactiveMarker.unmarkOrbit<FACE>(d) ;
	inactiveMarker.unmarkOrbit<FACE>(dd) ;
}

template <typename PFP>
void VolumetricProgressiveMesh<PFP>::coarsen()
{
	if(m_cur == m_splits.size())
		return ;

	VSplit<PFP>* vs = m_splits[m_cur] ; // get the split node
	++m_cur ;

	Dart d = vs->getEdge() ;
	Dart dd = m_map.phi2(d) ;		// get some darts
	Dart d2 = vs->getLeftEdge() ;
	Dart dd2 = vs->getRightEdge() ;

	edgeCollapse(vs) ;	// collapse edge

	m_map.template setOrbitEmbedding<VERTEX>(d2, vs->getApproxV()) ;
	m_map.template setOrbitEmbedding<EDGE>(d2, vs->getApproxE1()) ;
	m_map.template setOrbitEmbedding<EDGE>(dd2, vs->getApproxE2()) ;
}

template <typename PFP>
void VolumetricProgressiveMesh<PFP>::refine()
{
	if(m_cur == 0)
		return ;

	--m_cur ;
	VSplit<PFP>* vs = m_splits[m_cur] ; // get the split node

	Dart d = vs->getEdge() ;
	Dart dd = m_map.phi2(d) ; 		// get some darts
	Dart dd2 = vs->getRightEdge() ;
	Dart d2 = vs->getLeftEdge() ;
	Dart d1 = m_map.phi2(d2) ;
	Dart dd1 = m_map.phi2(dd2) ;

	unsigned int v1 = m_map.template getEmbedding<VERTEX>(d) ;				// get the embedding
	unsigned int v2 = m_map.template getEmbedding<VERTEX>(dd) ;			// of the new vertices
	unsigned int e1 = m_map.template getEmbedding<EDGE>(m_map.phi1(d)) ;
	unsigned int e2 = m_map.template getEmbedding<EDGE>(m_map.phi_1(d)) ;	// and new edges
	unsigned int e3 = m_map.template getEmbedding<EDGE>(m_map.phi1(dd)) ;
	unsigned int e4 = m_map.template getEmbedding<EDGE>(m_map.phi_1(dd)) ;

//	if(!m_predictors.empty())
//	{
//		for(typename std::vector<Algo::Decimation::PredictorGen<PFP>*>::iterator pit = m_predictors.begin();
//			pit != m_predictors.end();
//			++pit)
//		{
//			(*pit)->predict(d2, dd2) ;
//		}
//	}

//	typename PFP::MATRIX33 invLocalFrame ;
//	if(m_localFrameDetailVectors)
//	{
//		typename PFP::MATRIX33 localFrame = Algo::Geometry::vertexLocalFrame<PFP>(m_map, dd2, positionsTable) ;
//		localFrame.invert(invLocalFrame) ;
//	}

	vertexSplit(vs) ; // split vertex

	m_map.template setOrbitEmbedding<VERTEX>(d, v1) ;		// embed the
	m_map.template setOrbitEmbedding<VERTEX>(dd, v2) ;	// new vertices
	m_map.template setOrbitEmbedding<EDGE>(d1, e1) ;
	m_map.template setOrbitEmbedding<EDGE>(d2, e2) ;		// and new edges
	m_map.template setOrbitEmbedding<EDGE>(dd1, e3) ;
	m_map.template setOrbitEmbedding<EDGE>(dd2, e4) ;

//	if(!m_predictors.empty())
//	{
//		typename std::vector<Algo::Decimation::PredictorGen<PFP>*>::iterator pit ;
//		typename std::vector<Algo::Decimation::ApproximatorGen<PFP>*>::iterator ait ;
//		for(pit = m_predictors.begin(), ait = m_approximators.begin();
//			pit != m_predictors.end();
//			++pit, ++ait)
//		{
//			typename PFP::MATRIX33* detailTransform = NULL ;
//			if(m_localFrameDetailVectors)
//				detailTransform = &invLocalFrame ;
//
//			(*pit)->affectPredict(d) ;
//			if((*ait)->getType() == Algo::Decimation::A_HalfCollapse)
//			{
//				(*ait)->addDetail(dd, m_detailAmount, true, detailTransform) ;
//			}
//			else
//			{
//				(*ait)->addDetail(d, m_detailAmount, true, detailTransform) ;
//				(*ait)->addDetail(dd, m_detailAmount, false, detailTransform) ;
//			}
//		}
//	}
}


template <typename PFP>
void VolumetricProgressiveMesh<PFP>::gotoLevel(unsigned int l)
{
	if(l == m_cur || l > m_splits.size() || l < 0)
		return ;

	if(l > m_cur)
		while(m_cur != l)
			coarsen() ;
	else
		while(m_cur != l)
			refine() ;

//	unsigned int i=0;
//	if(l == m_level || l > m_nodes->size() || l < 0)
//		return ;
//
//	if(l > m_level)
//		for(i=m_level ; i<l ; i++)
//			m_nodes->coarsen(positionsTable);
//	else
//		for(i=l ; i<m_level ; i++)
//			m_nodes->refine(positionsTable);
//
//	m_level = i;
}


} //namespace VPMesh

}

} //namespace Algo

} //namespace CGoGN
