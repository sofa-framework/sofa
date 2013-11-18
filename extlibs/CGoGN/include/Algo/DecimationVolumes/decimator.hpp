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

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Decimation
{

template <typename PFP>
void decimate(typename PFP::MAP& map, SelectorType s, ApproximatorType a,
	VertexAttribute<typename PFP::VEC3>& position, unsigned int percentWantedVertices)
{
	std::vector<ApproximatorGen<PFP>*> approximators ;
	Selector<PFP>* selector = NULL ;

	//choose the Approximator
	switch(a)
	{
		case A_MidEdge :
			approximators.push_back(new Approximator_MidEdge<PFP>(map, position)) ;
			break ;
		default :
			CGoGNout << "not yet implemented" << CGoGNendl;
			break;
	}

	//choose the Selector
	switch(s)
	{
		case S_MapOrder :
			selector = new Algo::Volume::Decimation::EdgeSelector_MapOrder<PFP>(map, position, approximators) ;
			break ;
		case S_Random :
			selector = new Algo::Volume::Decimation::EdgeSelector_Random<PFP>(map, position, approximators) ;
			break ;
		default:
			CGoGNout << "not yet implemented" << CGoGNendl;
			break;
	}

	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = approximators.begin(); it != approximators.end(); ++it)
		(*it)->init() ;

	if(!selector->init())
		return ;


	unsigned int nbVertices = map.template getNbOrbits<VERTEX>() ;
	unsigned int nbWantedVertices = nbVertices * percentWantedVertices / 100 ;
	CGoGNout << " decimate (" << nbVertices << " vertices).." << /* flush */ CGoGNendl ;
	bool finished = false ;

	Dart d;

	while(!finished)
	{
		if(!selector->nextEdge(d))
			break ;

		std::cout << "d = " << d << std::endl;

		--nbVertices ;

		Dart d2 = map.phi2(map.phi_1(d)) ;
		Dart dd2 = map.phi2(map.phi_1(map.phi2(d))) ;

		for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = approximators.begin(); it != approximators.end(); ++it)
		{
			(*it)->approximate(d) ;				// compute approximated attributes
			(*it)->saveApprox(d) ;
		}

		selector->updateBeforeCollapse(d) ;		// update selector

		map.collapseEdge(d) ;					// collapse edge

		if(!map.check())
			finished = true;

		for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = approximators.begin(); it != approximators.end(); ++it)
			(*it)->affectApprox(d2);			// affect data to the resulting vertex

		selector->updateAfterCollapse(d2, dd2) ;// update selector

		if(nbVertices <= nbWantedVertices)
			finished = true ;
	}

	CGoGNout << "..done (" << nbVertices << " vertices)" << CGoGNendl ;

	delete selector ;

	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = approximators.begin(); it != approximators.end(); ++it)
		delete (*it) ;
}

} //namespace Decimation

} //namespace Volume

} //namespace Algo

} //namespace CGoGN

//		if(!selector->nextEdge(d))
//			break ;
//
//
//		Dart d2 = map.phi2(map.phi_1(d)) ;
//		Dart dd2 = map.phi2(map.phi_1(map.phi2(d))) ;
//
//		std::cout << "bin a contracter : " << d << std::endl;
//		std::cout << "voisin d2 : " << d2 << std::endl;
//		std::cout << "voisin dd2 : " << dd2 << std::endl;
//
//		--nbVertices ;
//
//		for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = approximators.begin(); it != approximators.end(); ++it)
//		{
//			(*it)->approximate(d) ;				// compute approximated attributes
//			(*it)->saveApprox(d) ;
//		}
//
//		selector->updateBeforeCollapse(d) ;		// update selector
//
//		map.collapseEdge(d) ;					// collapse edge
//
//		for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = approximators.begin(); it != approximators.end(); ++it)
//			(*it)->affectApprox(d2);			// affect data to the resulting vertex
//
//		selector->updateAfterCollapse(d2, dd2) ;// update selector
//
//		if(nbVertices <= nbWantedVertices)
//			finished = true ;

