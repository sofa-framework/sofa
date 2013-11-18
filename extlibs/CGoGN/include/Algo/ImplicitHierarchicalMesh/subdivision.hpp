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

namespace Surface
{

namespace IHM
{

template <typename PFP>
void subdivideEdge(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3>& position)
{
	assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
	assert(!map.edgeIsSubdivided(d) || !"Trying to subdivide an already subdivided edge") ;

	unsigned int eLevel = map.edgeLevel(d) ;

	unsigned int cur = map.getCurrentLevel() ;
	map.setCurrentLevel(eLevel) ;

	Dart dd = map.phi2(d) ;
	typename PFP::VEC3 p1 = position[d] ;
	typename PFP::VEC3 p2 = position[map.phi1(d)] ;

	map.setCurrentLevel(eLevel + 1) ;

	map.cutEdge(d) ;
	unsigned int eId = map.getEdgeId(d) ;
	map.setEdgeId(map.phi1(d), eId) ;
	map.setEdgeId(map.phi1(dd), eId) ;
	position[map.phi1(d)] = (p1 + p2) * typename PFP::REAL(0.5) ;

	map.setCurrentLevel(cur) ;
}

template <typename PFP>
void subdivideFace(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3>& position, bool forceTri)
{
	assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
	assert(!map.faceIsSubdivided(d) || !"Trying to subdivide an already subdivided face") ;

	unsigned int fLevel = map.faceLevel(d) ;
	Dart old = map.faceOldestDart(d) ;

	unsigned int cur = map.getCurrentLevel() ;
	map.setCurrentLevel(fLevel) ;		// go to the level of the face to subdivide its edges

	std::cout << "subdiv" << std::endl;

	unsigned int degree = 0 ;
	typename PFP::VEC3 p ;
	Dart it = old ;
	do
	{
		++degree ;
		p += position[it] ;
		if(!map.edgeIsSubdivided(it))							// first cut the edges (if they are not already)
			IHM::subdivideEdge<PFP>(map, it, position) ;	// and compute the degree of the face
		it = map.phi1(it) ;
	} while(it != old) ;
	p /= typename PFP::REAL(degree) ;

	map.setCurrentLevel(fLevel + 1) ;			// go to the next level to perform face subdivision

	if(degree == 3 && forceTri)								// if subdividing a triangle
	{
		Dart dd = map.phi1(old) ;
		Dart e = map.phi1(map.phi1(dd)) ;
		map.splitFace(dd, e) ;					// insert a new edge
		//unsigned int id = map.getNewEdgeId() ;
		//map.setEdgeId(map.phi_1(dd), id) ;		// set the edge id of the inserted
		//map.setEdgeId(map.phi_1(e), id) ;		// edge to the next available id

		dd = e ;
		e = map.phi1(map.phi1(dd)) ;
		map.splitFace(dd, e) ;
		//id = map.getNewEdgeId() ;
		//map.setEdgeId(map.phi_1(dd), id) ;
		//map.setEdgeId(map.phi_1(e), id) ;

		dd = e ;
		e = map.phi1(map.phi1(dd)) ;
		map.splitFace(dd, e) ;
		//id = map.getNewEdgeId() ;
		//map.setEdgeId(map.phi_1(dd), id) ;
		//map.setEdgeId(map.phi_1(e), id) ;


		Dart stop = map.phi2(map.phi1(old));
		Dart dit = stop;
		do
		{
			unsigned int dId = map.getEdgeId(map.phi_1(map.phi2(dit)));
			unsigned int eId = map.getEdgeId(map.phi1(map.phi2(dit)));

			unsigned int t = dId + eId;

			if(t == 0)
			{
				map.setEdgeId(dit, 1);
				map.setEdgeId(map.phi2(dit), 1);
			}
			else if(t == 1)
			{
				map.setEdgeId(dit, 2);
				map.setEdgeId(map.phi2(dit), 2);
			}
			else if(t == 2)
			{
				if(dId == eId)
				{
					map.setEdgeId(dit, 0);
					map.setEdgeId(map.phi2(dit), 0);
				}
				else
				{
					map.setEdgeId(dit, 1);
					map.setEdgeId(map.phi2(dit), 1);
				}
			}
			else if(t == 3)
			{
				map.setEdgeId(dit, 0);
				map.setEdgeId(map.phi2(dit), 0);
			}

			dit = map.phi1(dit);
		}while(dit != stop);

	}
	else											// if subdividing a polygonal face
	{
		Dart dd = map.phi1(old) ;
		map.splitFace(dd, map.phi1(map.phi1(dd))) ;	// insert a first edge
		Dart ne = map.alpha1(dd) ;
		//Dart ne2 = map.phi2(ne) ;

		map.cutEdge(ne) ;							// cut the new edge to insert the central vertex
		//unsigned int id = map.getNewEdgeId() ;
		//map.setEdgeId(ne, id) ;
		//map.setEdgeId(map.phi2(ne), id) ;			// set the edge id of the inserted
		//id = map.getNewEdgeId() ;
		//map.setEdgeId(ne2, id) ;					// edges to the next available ids
		//map.setEdgeId(map.phi2(ne2), id) ;

		position[map.phi2(ne)] = p ;

		dd = map.phi1(map.phi1(map.phi1(map.phi1(ne)))) ;
		while(dd != ne)								// turn around the face and insert new edges
		{											// linked to the central vertex
			Dart next = map.phi1(map.phi1(dd)) ;
			map.splitFace(map.phi1(ne), dd) ;
			//Dart nne = map.alpha1(dd) ;
			//id = map.getNewEdgeId() ;
			//map.setEdgeId(nne, id) ;
			//map.setEdgeId(map.phi2(nne), id) ;
			dd = next ;
		}

		Dart dit = map.phi2(ne);
		do
		{
			unsigned int eId = map.getEdgeId(map.phi1(dit));
			if(eId == 0)
			{
				map.setEdgeId(dit, 1);
				map.setEdgeId(map.phi2(dit), 1);
			}
			else if(eId == 1)
			{
				map.setEdgeId(dit, 0);
				map.setEdgeId(map.phi2(dit), 0);
			}
			dit = map.phi2(map.phi_1(dit));
		}
		while(dit != map.phi2(ne));
	}

	map.setCurrentLevel(cur) ;
}

template <typename PFP>
void coarsenEdge(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3>& position)
{
	assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
	assert(map.edgeCanBeCoarsened(d) || !"Trying to coarsen an edge that can not be coarsened") ;

	unsigned int cur = map.getCurrentLevel() ;
//	Dart e = map.phi2(d) ;
	map.setCurrentLevel(cur + 1) ;
//	unsigned int dl = map.getDartLevel(e) ;
//	map.setDartLevel(map.phi1(e), dl) ;
//	map.collapseEdge(e) ;
	map.uncutEdge(d) ;
	map.setCurrentLevel(cur) ;
}

template <typename PFP>
void coarsenFace(typename PFP::MAP& map, Dart d, VertexAttribute<typename PFP::VEC3>& position)
{
	assert(map.getDartLevel(d) <= map.getCurrentLevel() || !"Access to a dart introduced after current level") ;
	assert(map.faceIsSubdividedOnce(d) || !"Trying to coarsen a non-subdivided face or a more than once subdivided face") ;

	unsigned int cur = map.getCurrentLevel() ;

	unsigned int degree = 0 ;
	Dart fit = d ;
	do
	{
		++degree ;
		fit = map.phi1(fit) ;
	} while(fit != d) ;

	if(degree == 3)
	{
		fit = d ;
		do
		{
			map.setCurrentLevel(cur + 1) ;
			Dart innerEdge = map.phi1(fit) ;
			map.setCurrentLevel(map.getMaxLevel()) ;
			map.mergeFaces(innerEdge) ;
			map.setCurrentLevel(cur) ;
			fit = map.phi1(fit) ;
		} while(fit != d) ;
	}
	else
	{
		map.setCurrentLevel(cur + 1) ;
		Dart centralV = map.phi1(map.phi1(d)) ;
		map.setCurrentLevel(map.getMaxLevel()) ;
		map.deleteVertex(centralV) ;
		map.setCurrentLevel(cur) ;
	}

	fit = d ;
	do
	{
		if(map.edgeCanBeCoarsened(fit))
			coarsenEdge<PFP>(map, fit, position) ;
		fit = map.phi1(fit) ;
	} while(fit != d) ;
}

} //namespace IHM
} // Surface
} //namespace Algo
} //namespace CGoGN
