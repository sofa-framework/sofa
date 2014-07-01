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

namespace Surface
{

namespace PMesh
{

template <typename PFP>
ProgressiveMesh<PFP>::ProgressiveMesh(
		MAP& map,
		DartMarker<MAP>& inactive,
		Algo::Surface::Decimation::SelectorType s,
		Algo::Surface::Decimation::ApproximatorType a,
		VertexAttribute<VEC3, MAP>& pos
	) :
	m_map(map),
	position(pos),
	inactiveMarker(inactive)
{
	CGoGNout << "  creating approximator and predictor.." << CGoGNflush ;

	std::vector<VertexAttribute<VEC3, MAP>*> pos_v ;
	pos_v.push_back(&position) ;
	switch(a)
	{
		case Algo::Surface::Decimation::A_QEM : {
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_QEM<PFP>(m_map, pos_v)) ;
			break ; }
		case Algo::Surface::Decimation::A_MidEdge : {
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_MidEdge<PFP>(m_map, pos_v)) ;
			break ; }
		case Algo::Surface::Decimation::A_hHalfCollapse : {
			Algo::Surface::Decimation::Predictor_HalfCollapse<PFP>* pred = new Algo::Surface::Decimation::Predictor_HalfCollapse<PFP>(m_map, position) ;
			m_predictors.push_back(pred) ;
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_HalfCollapse<PFP>(m_map, pos_v, pred)) ;
			break ; }
		case Algo::Surface::Decimation::A_CornerCutting : {
			Algo::Surface::Decimation::Predictor_CornerCutting<PFP>* pred = new Algo::Surface::Decimation::Predictor_CornerCutting<PFP>(m_map, position) ;
			m_predictors.push_back(pred) ;
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_CornerCutting<PFP>(m_map, pos_v, pred)) ;
			break ; }
		case Algo::Surface::Decimation::A_TangentPredict1 : {
			Algo::Surface::Decimation::Predictor_TangentPredict1<PFP>* pred = new Algo::Surface::Decimation::Predictor_TangentPredict1<PFP>(m_map, position) ;
			m_predictors.push_back(pred) ;
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_MidEdge<PFP>(m_map, pos_v, pred)) ;
			break ; }
		case Algo::Surface::Decimation::A_TangentPredict2 : {
			Algo::Surface::Decimation::Predictor_TangentPredict2<PFP>* pred = new Algo::Surface::Decimation::Predictor_TangentPredict2<PFP>(m_map, position) ;
			m_predictors.push_back(pred) ;
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_MidEdge<PFP>(m_map, pos_v, pred)) ;
			break ; }
	}
	CGoGNout << "..done" << CGoGNendl ;

	CGoGNout << "  creating selector.." << CGoGNflush ;
	switch(s)
	{
		case Algo::Surface::Decimation::S_MapOrder : {
			m_selector = new Algo::Surface::Decimation::EdgeSelector_MapOrder<PFP>(m_map, positionsTable, m_approximators) ;
			break ; }
		case Algo::Surface::Decimation::S_Random : {
			m_selector = new Algo::Surface::Decimation::EdgeSelector_Random<PFP>(m_map, positionsTable, m_approximators) ;
			break ; }
		case Algo::Surface::Decimation::S_EdgeLength : {
			m_selector = new Algo::Surface::Decimation::EdgeSelector_Length<PFP>(m_map, positionsTable, m_approximators) ;
			break ; }
		case Algo::Surface::Decimation::S_QEM : {
			m_selector = new Algo::Surface::Decimation::EdgeSelector_QEM<PFP>(m_map, positionsTable, m_approximators) ;
			break ; }
		case Algo::Surface::Decimation::S_MinDetail : {
			m_selector = new Algo::Surface::Decimation::EdgeSelector_MinDetail<PFP>(m_map, positionsTable, m_approximators) ;
			break ; }
		case Algo::Surface::Decimation::S_Curvature : {
			m_selector = new Algo::Surface::Decimation::EdgeSelector_Curvature<PFP>(m_map, positionsTable, m_approximators) ;
			break ; }
	}
	CGoGNout << "..done" << CGoGNendl ;

	m_initOk = true ;

	CGoGNout << "  initializing approximators.." << CGoGNflush ;
	for(typename std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
	{
		if(! (*it)->init())
			m_initOk = false ;
		if((*it)->getApproximatedAttributeName() == "position")
			m_positionApproximator = reinterpret_cast<Algo::Surface::Decimation::Approximator<PFP, VEC3, EDGE>*>(*it) ;
	}
	CGoGNout << "..done" << CGoGNendl ;

	CGoGNout << "  initializing predictors.." << CGoGNflush ;
	for(typename std::vector<Algo::Surface::Decimation::PredictorGen<PFP>*>::iterator it = m_predictors.begin(); it != m_predictors.end(); ++it)
		if(! (*it)->init())
			m_initOk = false ;
	CGoGNout << "..done" << CGoGNendl ;

	CGoGNout << "  initializing selector.." << CGoGNflush ;
	m_initOk = m_selector->init() ;
	CGoGNout << "..done" << CGoGNendl ;

	m_detailAmount = REAL(1) ;
	m_localFrameDetailVectors = false ;
	quantizationInitialized = false ;
	quantizationApplied = false ;
}

template <typename PFP>
ProgressiveMesh<PFP>::ProgressiveMesh(
		MAP& map, DartMarker& inactive,
		Algo::Surface::Decimation::Selector<PFP>* selector, std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>& approximators,
		VertexAttribute<typename PFP::VEC3>& position
	) :
	m_map(map), m_selector(selector), m_approximators(approximators), positionsTable(position), inactiveMarker(inactive)
{
	CGoGNout << "  initializing approximators.." << CGoGNflush ;
	for(typename std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
	{
		if(! (*it)->init())
			m_initOk = false ;
		if((*it)->getApproximatedAttributeName() == "position")
			m_positionApproximator = reinterpret_cast<Algo::Surface::Decimation::Approximator<PFP, VEC3, EDGE>*>(*it) ;
	}
	CGoGNout << "..done" << CGoGNendl ;

	CGoGNout << "  initializing predictors.." << CGoGNflush ;
	for(typename std::vector<Algo::Surface::Decimation::PredictorGen<PFP>*>::iterator it = m_predictors.begin(); it != m_predictors.end(); ++it)
		if(! (*it)->init())
			m_initOk = false ;
	CGoGNout << "..done" << CGoGNendl ;

	CGoGNout << "  initializing selector.." << CGoGNflush ;
	m_initOk = m_selector->init() ;
	CGoGNout << "..done" << CGoGNendl ;

	m_detailAmount = REAL(1) ;
	m_localFrameDetailVectors = false ;
	quantizationInitialized = false ;
	quantizationApplied = false ;
}

template <typename PFP>
ProgressiveMesh<PFP>::~ProgressiveMesh()
{
	for(unsigned int i = 0; i < m_splits.size(); ++i)
		delete m_splits[i] ;
	if(m_selector)
		delete m_selector ;
	for(typename std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
		delete (*it) ;
	for(typename std::vector<Algo::Surface::Decimation::PredictorGen<PFP>*>::iterator it = m_predictors.begin(); it != m_predictors.end(); ++it)
		delete (*it) ;
	if(quantizationInitialized)
		delete q ;
}

template <typename PFP>
void ProgressiveMesh<PFP>::createPM(unsigned int percentWantedVertices)
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

		for(typename std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
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

		for(typename std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
			(*it)->affectApprox(d2);				// affect data to the resulting vertex

		m_selector->updateAfterCollapse(d2, dd2) ;	// update selector

		if(nbVertices <= nbWantedVertices)
			finished = true ;
	}
	delete m_selector ;
	m_selector = NULL ;

	m_cur = m_splits.size() ;
	CGoGNout << "..done (" << nbVertices << " vertices)" << CGoGNendl ;

	initQuantization() ;
}

template <typename PFP>
void ProgressiveMesh<PFP>::edgeCollapse(VSplit<PFP>* vs)
{
	Dart d = vs->getEdge() ;
	Dart dd = m_map.phi2(d) ;

	inactiveMarker.template markOrbit<FACE>(d) ;
	inactiveMarker.template markOrbit<FACE>(dd) ;

	m_map.extractTrianglePair(d) ;
}

template <typename PFP>
void ProgressiveMesh<PFP>::vertexSplit(VSplit<PFP>* vs)
{
	Dart d = vs->getEdge() ;
	Dart dd = m_map.phi2(d) ;
	Dart d2 = vs->getLeftEdge() ;
	Dart dd2 = vs->getRightEdge() ;

	m_map.insertTrianglePair(d, d2, dd2) ;

	inactiveMarker.template unmarkOrbit<FACE>(d) ;
	inactiveMarker.template unmarkOrbit<FACE>(dd) ;
}

template <typename PFP>
void ProgressiveMesh<PFP>::coarsen()
{
	if(m_cur == m_splits.size())
		return ;

	VSplit<PFP>* vs = m_splits[m_cur] ; // get the split node
	++m_cur ;

	// Dart d = vs->getEdge() ;
	// Dart dd = m_map.phi2(d) ;		// get some darts
	Dart d2 = vs->getLeftEdge() ;
	Dart dd2 = vs->getRightEdge() ;

	edgeCollapse(vs) ;	// collapse edge

	m_map.template setOrbitEmbedding<VERTEX>(d2, vs->getApproxV()) ;
	m_map.template setOrbitEmbedding<EDGE>(d2, vs->getApproxE1()) ;
	m_map.template setOrbitEmbedding<EDGE>(dd2, vs->getApproxE2()) ;
}

template <typename PFP>
void ProgressiveMesh<PFP>::refine()
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

	if(!m_predictors.empty())
	{
		for(typename std::vector<Algo::Surface::Decimation::PredictorGen<PFP>*>::iterator pit = m_predictors.begin();
			pit != m_predictors.end();
			++pit)
		{
			(*pit)->predict(d2, dd2) ;
		}
	}

	typename PFP::MATRIX33 invLocalFrame ;
	if(m_localFrameDetailVectors)
	{
		typename PFP::MATRIX33 localFrame = Algo::Geometry::vertexLocalFrame<PFP>(m_map, dd2, position) ;
		localFrame.invert(invLocalFrame) ;
	}

	vertexSplit(vs) ; // split vertex

	m_map.template setOrbitEmbedding<VERTEX>(d, v1) ;	// embed the
	m_map.template setOrbitEmbedding<VERTEX>(dd, v2) ;	// new vertices
	m_map.template setOrbitEmbedding<EDGE>(d1, e1) ;
	m_map.template setOrbitEmbedding<EDGE>(d2, e2) ;	// and new edges
	m_map.template setOrbitEmbedding<EDGE>(dd1, e3) ;
	m_map.template setOrbitEmbedding<EDGE>(dd2, e4) ;

	if(!m_predictors.empty())
	{
		typename std::vector<Algo::Surface::Decimation::PredictorGen<PFP>*>::iterator pit ;
		typename std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>::iterator ait ;
		for(pit = m_predictors.begin(), ait = m_approximators.begin();
			pit != m_predictors.end();
			++pit, ++ait)
		{
			typename PFP::MATRIX33* detailTransform = NULL ;
			if(m_localFrameDetailVectors)
				detailTransform = &invLocalFrame ;

			(*pit)->affectPredict(d) ;
			if((*ait)->getType() == Algo::Surface::Decimation::A_hHalfCollapse)
			{
				(*ait)->addDetail(dd, m_detailAmount, true, detailTransform) ;
			}
			else
			{
				(*ait)->addDetail(d, m_detailAmount, true, detailTransform) ;
				(*ait)->addDetail(dd, m_detailAmount, false, detailTransform) ;
			}
		}
	}
}

template <typename PFP>
void ProgressiveMesh<PFP>::gotoLevel(unsigned int l)
{
	if(l == m_cur || l > m_splits.size())
		return ;

	if(l > m_cur)
		while(m_cur != l)
			coarsen() ;
	else
		while(m_cur != l)
			refine() ;
}

template <typename PFP>
void ProgressiveMesh<PFP>::recomputeApproxAndDetails()
{
//	if(!m_predictors.empty())
//	{
//		gotoLevel(0) ;
//		while(m_cur < nbSplits())
//		{
//			VSplit<PFP>* vs = m_splits[m_cur] ;
//			++m_cur ;
//			unsigned int e = vs->getApprox() ;
//			m_approximator->approximate(vs, e) ;
//			edgeCollapse(vs, e) ;
//		}
//	}
}

template <typename PFP>
void ProgressiveMesh<PFP>::setDetailAmount(double a)
{
	m_detailAmount = a ;
	unsigned int c = m_cur ;
	gotoLevel(nbSplits()) ;
	gotoLevel(c) ;
}

template <typename PFP>
void ProgressiveMesh<PFP>::localizeDetailVectors()
{
	if(m_positionApproximator->getPredictor() && !m_localFrameDetailVectors)
	{
		bool quantizationWasApplied = quantizationApplied ;
		unsigned int nbCodeVectors = 0 ;
		if(quantizationWasApplied)
		{
			nbCodeVectors = q->getNbCodeVectors() ;
			resetDetailVectors() ;
		}
		m_localFrameDetailVectors = true ;
		gotoLevel(nbSplits()) ;
		while(m_cur > 0)
		{
			Dart d = m_splits[m_cur-1]->getEdge() ;
			Dart dd2 = m_splits[m_cur-1]->getRightEdge() ;
			typename PFP::MATRIX33 localFrame = Algo::Geometry::vertexLocalFrame<PFP>(m_map, dd2, position) ;
			VEC3 det = m_positionApproximator->getDetail(d) ;
			det = localFrame * det ;
			m_positionApproximator->setDetail(d, det) ;
			refine() ;
		}
		quantizationInitialized = false ;
		initQuantization() ;
		if(quantizationWasApplied)
			quantizeDetailVectors(nbCodeVectors) ;
	}
}

template <typename PFP>
void ProgressiveMesh<PFP>::globalizeDetailVectors()
{
	if(!m_predictors.empty() && m_localFrameDetailVectors)
	{
		bool quantizationWasApplied = quantizationApplied ;
		unsigned int nbCodeVectors = 0 ;
		if(quantizationWasApplied)
		{
			nbCodeVectors = q->getNbCodeVectors() ;
			resetDetailVectors() ;
		}
		m_localFrameDetailVectors = false ;
		gotoLevel(nbSplits()) ;
		while(m_cur > 0)
		{
			Dart d = m_splits[m_cur-1]->getEdge() ;
			Dart dd2 = m_splits[m_cur-1]->getRightEdge() ;
			typename PFP::MATRIX33 localFrame = Algo::Geometry::vertexLocalFrame<PFP>(m_map, dd2, position) ;
			typename PFP::MATRIX33 invLocalFrame ;
			localFrame.invert(invLocalFrame) ;
			VEC3 det = m_positionApproximator->getDetail(d) ;
			det = invLocalFrame * det ;
			m_positionApproximator->setDetail(d, det) ;
			refine() ;
		}
		quantizationInitialized = false ;
		initQuantization() ;
		if(quantizationWasApplied)
			quantizeDetailVectors(nbCodeVectors) ;
	}
}

template <typename PFP>
void ProgressiveMesh<PFP>::initQuantization()
{
	if(m_positionApproximator->getPredictor() && !quantizationInitialized)
	{
		gotoLevel(nbSplits()) ;
		originalDetailVectors.resize(m_splits.size()) ;
		for(unsigned int i = 0; i < m_splits.size(); ++i)
			originalDetailVectors[i] = m_positionApproximator->getDetail(m_splits[i]->getEdge(),0) ;
		q = new Algo::PMesh::Quantization<VEC3>(originalDetailVectors) ;
		quantizationInitialized = true ;
		CGoGNout << "  Differential Entropy -> " << q->getDifferentialEntropy() << CGoGNendl ;
	}
}

template <typename PFP>
void ProgressiveMesh<PFP>::quantizeDetailVectors(unsigned int nbClasses)
{
	initQuantization() ;
	if(quantizationInitialized)
	{
		gotoLevel(nbSplits()) ;
		std::vector<VEC3> resultat;
		q->vectorQuantizationNbRegions(nbClasses, resultat) ;
		for(unsigned int i = 0; i < m_splits.size(); ++i)
			m_positionApproximator->setDetail(m_splits[i]->getEdge(), 0, resultat[i]) ;
		quantizationApplied = true ;
		gotoLevel(0) ;
		CGoGNout << "Discrete Entropy -> " << q->getDiscreteEntropy() << " (codebook size : " << q->getNbCodeVectors() << ")" << CGoGNendl ;
/*
		Point p;
		p.x = q->getEntropieDiscrete() ;
		p.y = computeDistance2() ;
		p.nbClasses = q->getNbClasses() ;
		courbe.push_back(p) ;
*/
	}
}

template <typename PFP>
void ProgressiveMesh<PFP>::quantizeDetailVectors(float distortion)
{
	initQuantization() ;
	if(quantizationInitialized)
	{
		gotoLevel(nbSplits()) ;
		std::vector<typename PFP::VEC3> resultat;
		q->vectorQuantizationDistortion(distortion, resultat) ;
		for(unsigned int i = 0; i < m_splits.size(); ++i)
			m_positionApproximator->setDetail(m_splits[i]->getEdge(), resultat[i]) ;
		quantizationApplied = true ;
		gotoLevel(0) ;
		CGoGNout << "Discrete Entropy -> " << q->getDiscreteEntropy() << " (codebook size : " << q->getNbCodeVectors() << ")" << CGoGNendl ;
	}
}

template <typename PFP>
void ProgressiveMesh<PFP>::resetDetailVectors()
{
	if(quantizationInitialized)
	{
		gotoLevel(nbSplits()) ;
		for(unsigned int i = 0; i < m_splits.size(); ++i)
			m_positionApproximator->setDetail(m_splits[i]->getEdge(), originalDetailVectors[i]) ;
		delete q ;
		quantizationInitialized = false ;
		quantizationApplied = false ;
		gotoLevel(0) ;
	}
}

/*
template <typename PFP>
float ProgressiveMesh<PFP>::computeDistance2()
{
	float distance = 0; // sum of 2-distance between original vertices and new vertices

	gotoLevel(0) ; // mesh reconstruction from detail vectors
	DartMarker mUpdate(m_map) ;
	for(Dart d = m_map.begin(); d != m_map.end(); m_map.next(d)) // vertices loop
	{
		if(!mUpdate.isMarked(d))
		{
			mUpdate.markOrbit<VERTEX>(d) ;
			EMB* dEmb = reinterpret_cast<EMB*>(m_map.getVertexEmb(d)) ;
			// computes the 2-distance between original vertex and new vertex
			dEmb->updateDistance2() ;
			distance += dEmb->getDistance2() ;
		}
	}

	return distance ;
}

template <typename PFP>
void ProgressiveMesh<PFP>::calculCourbeDebitDistortion()
{
	Dart d;
	EMB* dEmb;
	std::vector<Vector3f> source;
	std::vector<Vector3f> resultat;
	float distance;
	Point p;

	CGoGNout << "calcul de la courbe débit distortion " << CGoGNendl;

	// get original detail vectors
	for(unsigned int i = 0; i < m_splits.size(); ++i)
	{
		source.push_back(Vector3f(*(m_splits.at(i)->getDetailDown())));
		source.push_back(Vector3f(*(m_splits.at(i)->getDetailUp())));
	}

	// vector quantization initialisation
	Quantization<Vector3f> q (source);
	q.vectorQuantizationInit();
	entropieDifferentielle = q.getEntropieDifferentielle();
	determinantSigma = q.getDeterminantSigma();
	traceSigma = q.getTraceSigma();


	// several quantizations of the same detail vectors to compute the curve
	for(unsigned int i = 8 ; i < m_splits.size() ; i *= 2)
	{
		q.vectorQuantization(i, resultat);

		// insert new vectors into the model to compute the distance
		for(unsigned int j = 0; j < m_splits.size(); ++j)
		{
			gmtl::Vec3f* v = resultat.at(j*2).getGmtl();
			m_splits.at(j)->setDetailDown(v);
			v = resultat.at(j*2+1).getGmtl();
			m_splits.at(j)->setDetailUp(v);
		}
		distance = computeDistance2();

		p.x = q.getEntropieDiscrete();
		p.y = distance;
		p.nbClasses = q.getNbClasses();
		courbe.push_back(p);
		// returns to coarse mesh
		gotoLevel(nbSplits());
		CGoGNout << "..." << CGoGNendl;
	}
	q.erase();
}
*/

} // namespace PMesh

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
