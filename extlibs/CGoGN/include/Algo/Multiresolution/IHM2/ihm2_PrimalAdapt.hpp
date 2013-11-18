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

#include "Algo/Multiresolution/IHM2/ihm2_PrimalAdapt.h"

namespace CGoGN
{

namespace Algo
{

namespace MR
{

namespace Primal
{

namespace Adaptive
{

template <typename PFP>
IHM2<PFP>::IHM2(typename PFP::MAP& map) :
	m_map(map),
	shareVertexEmbeddings(true),
	vertexVertexFunctor(NULL),
	edgeVertexFunctor(NULL),
	faceVertexFunctor(NULL)
{

}


template <typename PFP>
void IHM2<PFP>::subdivideEdge(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideEdge : called with a dart inserted after current level") ;
	assert(!m_map.edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;

	unsigned int eLevel = m_map.edgeLevel(d) ;

	unsigned int cur = m_map.getCurrentLevel() ;
	m_map.setCurrentLevel(eLevel) ;

	m_map.setCurrentLevel(eLevel + 1) ;

	Dart nd = m_map.cutEdge(d) ;
	unsigned int eId = m_map.getEdgeId(d) ;
	m_map.setEdgeId(nd, eId) ;
	m_map.setEdgeId(m_map.phi2(d), eId) ;
	(*edgeVertexFunctor)(nd) ;

	m_map.setCurrentLevel(cur) ;
}

template <typename PFP>
void IHM2<PFP>::coarsenEdge(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"coarsenEdge : called with a dart inserted after current level") ;
	assert(m_map.edgeCanBeCoarsened(d) || !"Trying to coarsen an edge that can not be coarsened") ;


	unsigned int cur = m_map.getCurrentLevel() ;
//	Dart e = m_map.phi2(d) ;
	m_map.setCurrentLevel(cur + 1) ;
//	unsigned int dl = m_map.getDartLevel(e) ;
//	m_map.setDartLevel(m_map.phi1(e), dl) ;
//	m_map.collapseEdge(e) ;
	m_map.uncutEdge(d) ;
	m_map.setCurrentLevel(cur) ;
}

template <typename PFP>
unsigned int IHM2<PFP>::subdivideFace(Dart d, bool triQuad, bool OneLevelDifference)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"subdivideFace : called with a dart inserted after current level") ;
	assert(!m_map.faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int fLevel = m_map.faceLevel(d) ;
	Dart old = m_map.faceOldestDart(d) ;

	unsigned int cur = m_map.getCurrentLevel() ;
	m_map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

	unsigned int degree = 0 ;
	Dart it = old ;
	do
	{
		++degree ;						// compute the degree of the face

		//if(OneLevelDifference)
		//{
		//	Dart nf = m_map.phi2(it) ;
		//	if(m_map.faceLevel(nf) == fLevel - 1)	// check if neighboring faces have to be subdivided first
		//		subdivideFace(nf) ;
		//}

		if(!m_map.edgeIsSubdivided(it))
			subdivideEdge(it) ;			// and cut the edges (if they are not already)
		it = m_map.phi1(it) ;
	} while(it != old) ;

	m_map.setCurrentLevel(fLevel + 1) ;	// go to the next level to perform face subdivision


	if(triQuad && degree == 3)					// if subdividing a triangle
	{
		Dart dd = m_map.phi1(old) ;
		Dart e = m_map.phi1(dd) ;
		//(*vertexVertexFunctor)(e) ;

		e = m_map.phi1(e) ;
		m_map.splitFace(dd, e) ;					// insert a new edge
		unsigned int id = m_map.getNewEdgeId() ;
		m_map.setEdgeId(m_map.phi_1(dd), id) ;		// set the edge id of the inserted
		m_map.setEdgeId(m_map.phi_1(e), id) ;		// edge to the next available id

		dd = e ;
		e = m_map.phi1(dd) ;
		//(*vertexVertexFunctor)(e) ;
		e = m_map.phi1(e) ;
		m_map.splitFace(dd, e) ;
		id = m_map.getNewEdgeId() ;
		m_map.setEdgeId(m_map.phi_1(dd), id) ;
		m_map.setEdgeId(m_map.phi_1(e), id) ;

		dd = e ;
		e = m_map.phi1(dd) ;
		//(*vertexVertexFunctor)(e) ;
		e = m_map.phi1(e) ;
		m_map.splitFace(dd, e) ;
		id = m_map.getNewEdgeId() ;
		m_map.setEdgeId(m_map.phi_1(dd), id) ;
		m_map.setEdgeId(m_map.phi_1(e), id) ;
	}
	else											// if subdividing a polygonal face
	{
		Dart dd = m_map.phi1(old) ;
		Dart next = m_map.phi1(dd) ;
		//(*vertexVertexFunctor)(next) ;
		next = m_map.phi1(next) ;
		m_map.splitFace(dd, next) ;	// insert a first edge
		Dart ne = m_map.alpha1(dd) ;
		Dart ne2 = m_map.phi2(ne) ;

		m_map.cutEdge(ne) ;							// cut the new edge to insert the central vertex
		unsigned int id = m_map.getNewEdgeId() ;
		m_map.setEdgeId(ne, id) ;
		m_map.setEdgeId(m_map.phi2(ne), id) ;			// set the edge id of the inserted
		id = m_map.getNewEdgeId() ;
		m_map.setEdgeId(ne2, id) ;					// edges to the next available ids
		m_map.setEdgeId(m_map.phi2(ne2), id) ;

		dd = m_map.phi1(next) ;
		(*vertexVertexFunctor)(dd) ;
		dd = m_map.phi1(dd) ;
		while(dd != ne)								// turn around the face and insert new edges
		{											// linked to the central vertex
			m_map.splitFace(m_map.phi1(ne), dd) ;
			Dart nne = m_map.alpha1(dd) ;
			id = m_map.getNewEdgeId() ;
			m_map.setEdgeId(nne, id) ;
			m_map.setEdgeId(m_map.phi2(nne), id) ;
			dd = m_map.phi1(dd) ;
			//(*vertexVertexFunctor)(dd) ;
			dd = m_map.phi1(dd) ;
		}

		(*faceVertexFunctor)(m_map.phi1(ne)) ;
	}


	m_map.setCurrentLevel(cur) ;

	return fLevel ;
}

template <typename PFP>
void IHM2<PFP>::coarsenFace(Dart d)
{
	assert(m_map.getDartLevel(d) <= m_map.getCurrentLevel() || !"coarsenFace : called with a dart inserted after current level") ;
	assert(m_map.faceIsSubdividedOnce(d) || !"Trying to coarsen a non-subdivided face or a more than once subdivided face") ;

	unsigned int cur = m_map.getCurrentLevel() ;

	unsigned int degree = 0 ;
	Dart fit = d ;
	do
	{
		++degree ;
		fit = m_map.phi1(fit) ;
	} while(fit != d) ;

	if(degree == 3)
	{
		fit = d ;
		do
		{
			m_map.setCurrentLevel(cur + 1) ;
			Dart innerEdge = m_map.phi1(fit) ;
			m_map.setCurrentLevel(m_map.getMaxLevel()) ;
			m_map.mergeFaces(innerEdge) ;
			m_map.setCurrentLevel(cur) ;
			fit = m_map.phi1(fit) ;
		} while(fit != d) ;
	}
	else
	{
		m_map.setCurrentLevel(cur + 1) ;
		Dart centralV = m_map.phi1(m_map.phi1(d)) ;
		m_map.setCurrentLevel(m_map.getMaxLevel()) ;
		m_map.deleteVertex(centralV) ;
		m_map.setCurrentLevel(cur) ;
	}

	fit = d ;
	do
	{
		if(m_map.edgeCanBeCoarsened(fit))
			coarsenEdge(fit) ;
		fit = m_map.phi1(fit) ;
	} while(fit != d) ;
}


} // namespace Adaptive

} // namespace Primal

} // namespace MR

} // namespace Algo

} // namespace CGoGN
