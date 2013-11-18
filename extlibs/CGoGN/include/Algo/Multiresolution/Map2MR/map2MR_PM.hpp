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

#include "Container/attributeMultiVector.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MR
{

template <typename PFP>
Map2MR_PM<PFP>::Map2MR_PM(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position) : m_map(map), m_position(position)
{

}

template <typename PFP>
Map2MR_PM<PFP>::~Map2MR_PM()
{
	if(m_selector)
		delete m_selector ;
	for(typename std::vector<Algo::Surface::Decimation::ApproximatorGen<PFP>*>::iterator it = m_approximators.begin(); it != m_approximators.end(); ++it)
		delete (*it) ;
	for(typename std::vector<Algo::Surface::Decimation::PredictorGen<PFP>*>::iterator it = m_predictors.begin(); it != m_predictors.end(); ++it)
		delete (*it) ;
}

template <typename PFP>
void Map2MR_PM<PFP>::createPM(Algo::Surface::Decimation::SelectorType s, Algo::Surface::Decimation::ApproximatorType a)
{
	CGoGNout << "  creating approximator and predictor.." << CGoGNflush ;

	std::vector<VertexAttribute< typename PFP::VEC3>* > pos_v ;
	pos_v.push_back(&m_position) ;
	switch(a)
	{
		case Algo::Surface::Decimation::A_QEM : {
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_QEM<PFP>(m_map, pos_v)) ;
			break ; }
		case Algo::Surface::Decimation::A_MidEdge : {
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_MidEdge<PFP>(m_map, pos_v)) ;
			break ; }
		case Algo::Surface::Decimation::A_hHalfCollapse : {
			Algo::Surface::Decimation::Predictor_HalfCollapse<PFP>* pred = new Algo::Surface::Decimation::Predictor_HalfCollapse<PFP>(m_map, m_position) ;
			m_predictors.push_back(pred) ;
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_HalfCollapse<PFP>(m_map, pos_v, pred)) ;
			break ; }
		case Algo::Surface::Decimation::A_CornerCutting : {
			Algo::Surface::Decimation::Predictor_CornerCutting<PFP>* pred = new Algo::Surface::Decimation::Predictor_CornerCutting<PFP>(m_map, m_position) ;
			m_predictors.push_back(pred) ;
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_CornerCutting<PFP>(m_map, pos_v, pred)) ;
			break ; }
		case Algo::Surface::Decimation::A_TangentPredict1 : {
			Algo::Surface::Decimation::Predictor_TangentPredict1<PFP>* pred = new Algo::Surface::Decimation::Predictor_TangentPredict1<PFP>(m_map, m_position) ;
			m_predictors.push_back(pred) ;
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_MidEdge<PFP>(m_map, pos_v, pred)) ;
			break ; }
		case Algo::Surface::Decimation::A_TangentPredict2 : {
			Algo::Surface::Decimation::Predictor_TangentPredict2<PFP>* pred = new Algo::Surface::Decimation::Predictor_TangentPredict2<PFP>(m_map, m_position) ;
			m_predictors.push_back(pred) ;
			m_approximators.push_back(new Algo::Surface::Decimation::Approximator_MidEdge<PFP>(m_map, pos_v, pred)) ;
			break ; }
		case Algo::Surface::Decimation::A_NormalArea :
		case Algo::Surface::Decimation::A_ColorQEMext :
		case Algo::Surface::Decimation::A_hQEM :
		case Algo::Surface::Decimation::A_hLightfieldHalf :
		case Algo::Surface::Decimation::A_Lightfield :
		default: break;
	}
	CGoGNout << "..done" << CGoGNendl ;

	CGoGNout << "  creating selector.." << CGoGNflush ;
	switch(s)
	{
	case Algo::Surface::Decimation::S_MapOrder : {
		m_selector = new Algo::Surface::Decimation::EdgeSelector_MapOrder<PFP>(m_map, m_position, m_approximators) ;
		break ; }
	case Algo::Surface::Decimation::S_Random : {
		m_selector = new Algo::Surface::Decimation::EdgeSelector_Random<PFP>(m_map, m_position, m_approximators) ;
		break ; }
	case Algo::Surface::Decimation::S_EdgeLength : {
		m_selector = new Algo::Surface::Decimation::EdgeSelector_Length<PFP>(m_map, m_position, m_approximators) ;
		break ; }
	case Algo::Surface::Decimation::S_QEM : {
		m_selector = new Algo::Surface::Decimation::EdgeSelector_QEM<PFP>(m_map, m_position, m_approximators) ;
		break ; }
	case Algo::Surface::Decimation::S_MinDetail : {
		m_selector = new Algo::Surface::Decimation::EdgeSelector_MinDetail<PFP>(m_map, m_position, m_approximators) ;
		break ; }
	case Algo::Surface::Decimation::S_CurvatureTensor : {
		m_selector = new Algo::Surface::Decimation::EdgeSelector_Curvature<PFP>(m_map, m_position, m_approximators) ;
		break ; }
	case Algo::Surface::Decimation::S_QEMml :
	case Algo::Surface::Decimation::S_Curvature :
	case Algo::Surface::Decimation::S_NormalArea :
	case Algo::Surface::Decimation::S_ColorNaive :
	case Algo::Surface::Decimation::S_QEMextColor :
	case Algo::Surface::Decimation::S_hQEMextColor :
	case Algo::Surface::Decimation::S_hQEMml :
	case Algo::Surface::Decimation::S_Lightfield :
	case Algo::Surface::Decimation::S_hLightfield :
	case Algo::Surface::Decimation::S_hLightfieldExp :
	case Algo::Surface::Decimation::S_hLightfieldKCL :
	case Algo::Surface::Decimation::S_hColorExperimental :
	case Algo::Surface::Decimation::S_hLFexperimental :
	case Algo::Surface::Decimation::S_hColorPerFace :
	case Algo::Surface::Decimation::S_hLFperFace :
	default: break;
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


}

template <typename PFP>
void Map2MR_PM<PFP>::addNewLevel(unsigned int percentWantedVertices)
{
	unsigned int nbVertices = m_map.template getNbOrbits<VERTEX>() ;
	unsigned int nbWantedVertices = nbVertices * percentWantedVertices / 100 ;

	unsigned int nbDeletedVertex=0;
	unsigned int percentWantedPerLevel = 50;
	//unsigned int nbWantedPerLevel = nbWantedVertices * percentWantedPerLevel / 100 ;
	unsigned int nbWantedPerLevel = nbVertices * percentWantedPerLevel / 100 ;

	CGoGNout << "  initializing selector.." << CGoGNflush ;
	m_initOk = m_selector->init() ;
	CGoGNout << "..done" << CGoGNendl ;

	std::vector<Dart> edges;
	edges.reserve(nbWantedPerLevel);

	std::cout << "stops at  : " << nbWantedPerLevel << std::endl;

	DartMarkerStore me(m_map); 	//mark edges not to collapse

	bool finished = false ;
	Dart d ;

	while(!finished)
	{
		if(!m_selector->nextEdge(d))
			break ;

		if(!me.isMarked(d))
		{
			//Mark le 1 voisinage
			Dart dt = d;
			do
			{
				Traversor2VE<typename PFP::MAP> tf(m_map, dt) ;
				for(Dart it = tf.begin(); it != tf.end(); it = tf.next())
				{
					me.markOrbit<EDGE>(it);
					me.markOrbit<EDGE>(m_map.phi1(it));
				}

				dt = m_map.phi1(dt);
			}while(dt != d);

			Traversor2VE<typename PFP::MAP> tf(m_map, m_map.phi_1(m_map.phi2(d))) ;
			for(Dart it = tf.begin(); it != tf.end(); it = tf.next())
			{
				me.markOrbit<EDGE>(it);
				me.markOrbit<EDGE>(m_map.phi1(it));
			}

			++nbDeletedVertex ;

			edges.push_back(d);
		}

		m_selector->updateWithoutCollapse();

		if(nbDeletedVertex >= nbWantedPerLevel)
			finished = true ;
	}


	std::cout << "nbDeletedVertices  : " << nbDeletedVertex << std::endl;

	if(!edges.empty())
	{
		//create the new level
		m_map.addLevelFront();
		m_map.setCurrentLevel(0);

		AttributeContainer& attribs = m_map.getMRAttributeContainer();
		AttributeMultiVector<unsigned int>* attribLevel = m_map.getMRLevelAttributeVector();
		AttributeMultiVector<unsigned int>* attribDarts = m_map.getMRDartAttributeVector(0);

		for(unsigned int i = attribs.begin(); i != attribs.end(); attribs.next(i))
		{
			if((*attribDarts)[i] == MRNULL)
				++(*attribLevel)[i];
		}

		for(std::vector<Dart>::iterator it = edges.begin() ; it != edges.end() ; ++it)
		{
			collapseEdge(*it);
		}
	}

	//m_map.printMR();
}


template <typename PFP>
void Map2MR_PM<PFP>::collapseEdge(Dart d)
{
	//incremente le dartLevel des brins des faces a supprimer
	m_map.incDartLevel(d);
	m_map.incDartLevel(m_map.phi1(d));
	m_map.incDartLevel(m_map.phi_1(d));
	m_map.incDartLevel(m_map.phi2(d));
	m_map.incDartLevel(m_map.phi_1(m_map.phi2(d)));
	m_map.incDartLevel(m_map.phi1(m_map.phi2(d)));

	m_map.duplicateDartAtOneLevel(m_map.phi2(m_map.phi1(d)), 0);
	m_map.duplicateDartAtOneLevel(m_map.phi2(m_map.phi_1(d)), 0);
	m_map.duplicateDartAtOneLevel(m_map.phi2(m_map.phi1(m_map.phi2(d))), 0);
	m_map.duplicateDartAtOneLevel(m_map.phi2(m_map.phi_1(m_map.phi2(d))), 0);

	m_map.duplicateDartAtOneLevel(d, 0);
	m_map.duplicateDartAtOneLevel(m_map.phi1(d), 0);
	m_map.duplicateDartAtOneLevel(m_map.phi_1(d), 0);
	m_map.duplicateDartAtOneLevel(m_map.phi2(d), 0);
	m_map.duplicateDartAtOneLevel(m_map.phi_1(m_map.phi2(d)), 0);
	m_map.duplicateDartAtOneLevel(m_map.phi1(m_map.phi2(d)), 0);

	m_map.collapseEdge(d);
}

//analysis
template <typename PFP>
void Map2MR_PM<PFP>::coarsen()
{
	assert(m_map.getCurrentLevel() > 0 || !"coarsen : called on level 0") ;

	m_map.decCurrentLevel() ;

	for(unsigned int i = 0; i < analysisFilters.size(); ++i)
		(*analysisFilters[i])() ;
}

//synthesis
template <typename PFP>
void Map2MR_PM<PFP>::refine()
{
	assert(m_map.getCurrentLevel() < m_map.getMaxLevel() || !"refine: called on max level") ;

	m_map.incCurrentLevel() ;

	for(unsigned int i = 0; i < synthesisFilters.size(); ++i)
		(*synthesisFilters[i])() ;
}

template <typename PFP>
Dart Map2MR_PM<PFP>::vertexOrigin(Dart d)
{
//	Dart dit = d;
//	do
//	{
//		if(m_map.getDartLevel(dit) == m_map.currentLevel())
//			return dit;
//
//		dit = m_map.phi2(m_map.phi_1(dit));
//	}
//	while(dit != d);

	return NIL;
}

//template <typename PFP>
//unsigned int Map2MR_PM<PFP>::vertexLevel(Dart d)
//{
//	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"edgeLevel : called with a dart inserted after current level") ;
//
//	Dart dit = d;
//	do
//	{
//		unsigned int ldit = m_map.getDartLevel(dit) ;
//
//		dit = m_map.phi2(m_map.phi_1(dit));
//	}
//	while(dit != d);
//
//
//	unsigned int ldd = m_map.getDartLevel(m_map.phi2(d)) ;	// the level of an edge is the maximum of the
//	return ld > ldd ? ld : ldd ;				// insertion levels of its two darts
//}


} // namespace Multiresolution

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
