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

#include "Topology/map/map2.h"
#include "Topology/generic/traversorCell.h"
#include "Topology/generic/dartmarker.h"

namespace CGoGN
{

void Map2::rdfi(Dart t, DartMarker& m1, DartMarker& m2)
{
	Dart p = NIL;
	while (!(p == NIL && (t == NIL || (m1.isMarked(t) || m2.isMarked(t)) ) ) )
	{
		if (t == NIL || (m1.isMarked(t) || m2.isMarked(t)))
		{
			if (m2.isMarked(p))			// pop
			{
				Dart q = phi2(p);		//	q = p->s1;
				unsigned int pi=dartIndex(p);
				(*m_phi2)[pi]=t;		//	p->s1 = t;
				t = p;
				p = q;
	     	}
			else						// swing
	     	{
				m2.mark(p);				//	p->val = 2;
				Dart q = phi1(p);		//	q = p->s0;
				unsigned int pi=dartIndex(p);
				(*m_phi1)[pi]=t;		//	p->s0 = t;
				t = phi2(p);			//	t = p->s1;
				(*m_phi2)[pi]=q;		//	p->s1 = q;
			}
		}
		else							 // push
		{
			m1.mark(t);					//	t->val = 1;
			Dart q = phi1(t);			//	q = t->s0;
			unsigned int ti=dartIndex(t);
			(*m_phi1)[ti]=p;			//	t->s0 = p;
			p = t;
			t = q;
		}
	}
}

void Map2::compactTopoRelations(const std::vector<unsigned int>& oldnew)
{
	for (unsigned int i = m_attribs[DART].begin(); i != m_attribs[DART].end(); m_attribs[DART].next(i))
	{
		unsigned int d_index = dartIndex(m_phi1->operator[](i));
		if (d_index != oldnew[d_index])
			m_phi1->operator[](i) = Dart(oldnew[d_index]);

		d_index = dartIndex(m_phi_1->operator[](i));
		if (d_index != oldnew[d_index])
			m_phi_1->operator[](i) = Dart(oldnew[d_index]);

		d_index = dartIndex(m_phi2->operator[](i));
		if (d_index != oldnew[d_index])
			m_phi2->operator[](i) = Dart(oldnew[d_index]);

//		{
//			Dart& d = m_phi1->operator [](i);
//			Dart e = Dart(oldnew[d.index]);
//			if (d != e)
//				d = e;
//		}
//		{
//			Dart& d = m_phi_1->operator [](i);
//			Dart e = Dart(oldnew[d.index]);
//			if (d != e)
//				d = e;
//		}
//		{
//			Dart& d = m_phi2->operator [](i);
//			Dart e = Dart(oldnew[d.index]);
//			if (d != e)
//				d = e;
//		}
	}
}

/*! @name Generator and Deletor
 *  To generate or delete faces in a 2-map
 *************************************************************************/

Dart Map2::newPolyLine(unsigned int nbEdges)
{
	Dart d = Map1::newCycle(2*nbEdges);
	{
		Dart it1 = d;
		Dart it2 = phi_1(d);
		for(unsigned int i = 0; i<nbEdges ; ++i)
		{
			phi2sew(it1, it2);
			it1 = phi1(it1);
			it2 = phi_1(it2);
		}
	}
	return d;
}

Dart Map2::newFace(unsigned int nbEdges, bool withBoundary)
{
	Dart d = Map1::newCycle(nbEdges);
	if (withBoundary)
	{
		Dart e = newBoundaryCycle(nbEdges);

		Dart it = d;
		do
		{
			phi2sew(it, e);
			it = phi1(it);
			e = phi_1(e);
		} while (it != d);
	}
	return d;
}

void Map2::deleteFace(Dart d, bool withBoundary)
{
	assert(!isBoundaryMarked2(d)) ;
	if (withBoundary)
	{
		Dart it = d ;
		do
		{
			if(!isBoundaryEdge(it))
				unsewFaces(it) ;
			it = phi1(it) ;
		} while(it != d) ;
		Dart dd = phi2(d) ;
		Map1::deleteCycle(d) ;
		Map1::deleteCycle(dd) ;
		return;
	}
	//else with remove the face and create fixed points
	Dart it = d ;
	do
	{
		phi2unsew(it);
		it = phi1(it) ;
	} while(it != d) ;
	Map1::deleteCycle(d);
}

void Map2::deleteCC(Dart d)
{
	DartMarkerStore mark(*this);

	std::vector<Dart> visited;
	visited.reserve(1024) ;
	visited.push_back(d);
	mark.mark(d) ;

	for(unsigned int i = 0; i < visited.size(); ++i)
	{
		Dart d1 = phi1(visited[i]) ;
		if(!mark.isMarked(d1))
		{
			visited.push_back(d1) ;
			mark.mark(d1);
		}
		Dart d2 = phi2(visited[i]) ;
		if(!mark.isMarked(d2))
		{
			visited.push_back(d2) ;
			mark.mark(d2);
		}
	}

	for(std::vector<Dart>::iterator it = visited.begin(); it != visited.end(); ++it)
		deleteDart(*it) ;
}

void Map2::fillHole(Dart d)
{
	assert(isBoundaryEdge(d)) ;
	Dart dd = d ;
	if(!isBoundaryMarked2(dd))
		dd = phi2(dd) ;
	boundaryUnmarkOrbit<FACE,2>(dd) ;
}

void Map2::createHole(Dart d)
{
	assert(!isBoundaryEdge(d)) ;
	boundaryMarkOrbit<FACE,2>(d) ;
}

/*! @name Topological Operators
 *  Topological operations on 2-maps
 *************************************************************************/

void Map2::splitVertex(Dart d, Dart e)
{
	assert(sameVertex(d, e)) ;
	Dart d2 = phi2(d) ; assert(d != d2) ;
	Dart e2 = phi2(e) ; assert(e != e2) ;
	Dart nd = Map1::cutEdge(d2) ;	// Cut the edge of dd (make a new half edge)
	Dart ne = Map1::cutEdge(e2) ;	// Cut the edge of ee (make a new half edge)
	phi2sew(nd, ne) ;				// Sew the two faces along the new edge
}

Dart Map2::deleteVertex(Dart d)
{
	//TODO utile ?
	if(isBoundaryVertex(d))
		return NIL ;

	Dart res = NIL ;
	Dart vit = d ;
	do
	{
		if(res == NIL && phi1(phi1(d)) != d)
			res = phi1(d) ;

		Dart f = phi_1(phi2(vit)) ;
		phi1sew(vit, f) ;

		vit = phi2(phi_1(vit)) ;
	} while(vit != d) ;
	Map1::deleteCycle(d) ;
	return res ;
}

Dart Map2::cutEdge(Dart d)
{
	Dart e = phi2(d);
	phi2unsew(d);					// remove old phi2 links
	Dart nd = Map1::cutEdge(d);		// Cut the 1-edge of d
	Dart ne = Map1::cutEdge(e);		// Cut the 1-edge of phi2(d)
	phi2sew(d, ne);					// Correct the phi2 links
	phi2sew(e, nd);
	return nd;
}

bool Map2::uncutEdge(Dart d)
{
	if(vertexDegree(phi1(d)) == 2)
	{
		Dart e = phi2(phi1(d)) ;
		phi2unsew(e) ;
		phi2unsew(d) ;
		Map1::uncutEdge(d) ;
		Map1::uncutEdge(e) ;
		phi2sew(d, e) ;
		return true ;
	}
	return false ;
}

Dart Map2::collapseEdge(Dart d, bool delDegenerateFaces)
{
	Dart resV = NIL ;

	Dart e = phi2(d);
	phi2unsew(d);	// Unlink the opposite edges

	Dart f = phi1(e) ;
	Dart h = phi2(phi_1(e));

	if (h != e)
		resV = h;

	if (f != e && delDegenerateFaces)
	{
		Map1::collapseEdge(e) ;		// Collapse edge e
		collapseDegeneratedFace(f) ;// and collapse its face if degenerated
	}
	else
		Map1::collapseEdge(e) ;	// Just collapse edge e

	f = phi1(d) ;
	if(resV == NIL)
	{
		h = phi2(phi_1(d));
		if (h != d)
			resV = h;
	}

	if (f != d && delDegenerateFaces)
	{
		Map1::collapseEdge(d) ;		// Collapse edge d
		collapseDegeneratedFace(f) ;// and collapse its face if degenerated
	}
	else
		Map1::collapseEdge(d) ;	// Just collapse edge d

	return resV ;
}

bool Map2::flipEdge(Dart d)
{
	if (!isBoundaryEdge(d))
	{
		Dart e = phi2(d);
		Dart dNext = phi1(d);
		Dart eNext = phi1(e);
		Dart dPrev = phi_1(d);
		Dart ePrev = phi_1(e);
		phi1sew(d, ePrev);		// Detach the two
		phi1sew(e, dPrev);		// vertices of the edge
		phi1sew(d, dNext);		// Insert the edge in its
		phi1sew(e, eNext);		// new vertices after flip
		return true ;
	}
	return false ; // cannot flip a border edge
}

bool Map2::flipBackEdge(Dart d)
{
	if (!isBoundaryEdge(d))
	{
		Dart e = phi2(d);
		Dart dPrev = phi_1(d);
		Dart ePrev = phi_1(e);
		phi1sew(d, ePrev);			// Detach the two
		phi1sew(e, dPrev);			// vertices of the edge
		phi1sew(e, phi_1(dPrev));	// Insert the edge in its
		phi1sew(d, phi_1(ePrev));	// new vertices after flip
		return true ;
	}
	return false ; // cannot flip a border edge
}

void Map2::swapEdges(Dart d, Dart e)
{
	assert(!Map2::isBoundaryEdge(d) && !Map2::isBoundaryEdge(e));

	Dart d2 = phi2(d);
	Dart e2 = phi2(e);

	phi2unsew(d);
	phi2unsew(e) ;

	phi2sew(d, e);
	phi2sew(d2, e2);
}

void Map2::insertEdgeInVertex(Dart d, Dart e)
{
	assert(!sameVertex(d,e));
	assert(phi2(e) == phi_1(e));
	phi1sew(phi_1(d), phi_1(e));
}

bool Map2::removeEdgeFromVertex(Dart d)
{
	if (!isBoundaryEdge(d))
	{
		phi1sew(phi_1(d), phi2(d)) ;
		return true ;
	}
	return false ;
}

void Map2::sewFaces(Dart d, Dart e, bool withBoundary)
{
	// if sewing with fixed points
	if (!withBoundary)
	{
		assert(phi2(d) == d && phi2(e) == e) ;
		phi2sew(d, e) ;
		return ;
	}

	assert(isBoundaryEdge(d) && isBoundaryEdge(e)) ;

	Dart dd = phi2(d) ;
	Dart ee = phi2(e) ;

	phi2unsew(d) ;	// unsew faces from boundary
	phi2unsew(e) ;

	if (ee != phi_1(dd))
		phi1sew(ee, phi_1(dd)) ;	// remove the boundary edge
	if (dd != phi_1(ee))
		phi1sew(dd, phi_1(ee)) ;	// and properly close incident boundaries
	Map1::deleteCycle(dd) ;

	phi2sew(d, e) ; // sew the faces
}

void Map2::unsewFaces(Dart d, bool withBoundary)
{
	if (!withBoundary)
	{
		phi2unsew(d) ;
		return ;
	}

	assert(!Map2::isBoundaryEdge(d)) ;

	Dart dd = phi2(d) ;

	Dart e = newBoundaryCycle(2) ;
	Dart ee = phi1(e) ;

	Dart f = findBoundaryEdgeOfVertex(d) ;
	Dart ff = findBoundaryEdgeOfVertex(dd) ;

	if(f != NIL)
		phi1sew(e, phi_1(f)) ;

	if(ff != NIL)
		phi1sew(ee, phi_1(ff)) ;

	phi2unsew(d) ;

	phi2sew(d, e) ;		// sew faces
	phi2sew(dd, ee) ;	// to the boundary
}

bool Map2::collapseDegeneratedFace(Dart d)
{
	Dart e = phi1(d) ;				// Check if the face is degenerated
	if (phi1(e) == d)				// Yes: it contains one or two edge(s)
	{
		Dart d2 = phi2(d) ;			// Check opposite edges
		Dart e2 = phi2(e) ;
		phi2unsew(d) ;
		if(d != e)
		{
			phi2unsew(e) ;
			phi2sew(d2, e2) ;
		}
		else
		{
			phi1sew(d2, phi_1(d2)) ;
			Map1::deleteCycle(d2) ;
		}
		Map1::deleteCycle(d) ;
		return true ;
	}
	return false ;
}

void Map2::splitFace(Dart d, Dart e)
{
	assert(d != e && Map2::sameFace(d, e)) ;
	Dart dd = Map1::cutEdge(phi_1(d)) ;
	Dart ee = Map1::cutEdge(phi_1(e)) ;
	Map1::splitCycle(dd, ee) ;
	phi2sew(dd, ee);
}

bool Map2::mergeFaces(Dart d)
{
	if (!isBoundaryEdge(d))
	{
		Dart e = phi2(d) ;
		phi2unsew(d) ;
		Map1::mergeCycles(d, phi1(e)) ;
		Map1::splitCycle(e, phi1(d)) ;
		Map1::deleteCycle(d) ;
		return true ;
	}
	return false ;
}

void Map2::extractTrianglePair(Dart d)
{
	Dart e = phi2(d) ;

	assert(!isBoundaryFace(d) && !isBoundaryFace(e)) ;
	assert(faceDegree(d) == 3 && faceDegree(e) == 3) ;

	Dart d1 = phi2(phi1(d)) ;
	Dart d2 = phi2(phi_1(d)) ;
	phi2unsew(d1) ;
	phi2unsew(d2) ;
	phi2sew(d1, d2) ;

	Dart e1 = phi2(phi1(e)) ;
	Dart e2 = phi2(phi_1(e)) ;
	phi2unsew(e1) ;
	phi2unsew(e2) ;
	phi2sew(e1, e2) ;
}

void Map2::insertTrianglePair(Dart d, Dart v1, Dart v2)
{
	Dart e = phi2(d) ;

	assert(v1 != v2 && sameOrientedVertex(v1, v2)) ;
	assert(faceDegree(d) == 3 && faceDegree(phi2(d)) == 3) ;

	Dart vv1 = phi2(v1) ;
	phi2unsew(v1) ;
	phi2sew(phi_1(d), v1) ;
	phi2sew(phi1(d), vv1) ;

	Dart vv2 = phi2(v2) ;
	phi2unsew(v2) ;
	phi2sew(phi_1(e), v2) ;
	phi2sew(phi1(e), vv2) ;
}

bool Map2::mergeVolumes(Dart d, Dart e, bool deleteFace)
{
	assert(!isBoundaryMarked2(d) && !isBoundaryMarked2(e)) ;

	if (Map2::isBoundaryFace(d) || Map2::isBoundaryFace(e))
		return false;

	// First traversal of both faces to check the face sizes
	// and store their edges to efficiently access them further

	std::vector<Dart> dDarts;
	std::vector<Dart> eDarts;
	dDarts.reserve(16);		// usual faces have less than 16 edges
	eDarts.reserve(16);

	Dart dFit = d ;
	Dart eFit = phi1(e) ;	// must take phi1 because of the use
	do						// of reverse iterator for sewing loop
	{
		dDarts.push_back(dFit) ;
		dFit = phi1(dFit) ;
	} while(dFit != d) ;
	do
	{
		eDarts.push_back(eFit) ;
		eFit = phi1(eFit) ;
	} while(eFit != phi1(e)) ;

	if(dDarts.size() != eDarts.size())
		return false ;

	// Make the sewing: take darts in initial order (clockwise) in first face
	// and take darts in reverse order (counter-clockwise) in the second face
	std::vector<Dart>::iterator dIt;
	std::vector<Dart>::reverse_iterator eIt;
	for (dIt = dDarts.begin(), eIt = eDarts.rbegin(); dIt != dDarts.end(); ++dIt, ++eIt)
	{
		Dart d2 = phi2(*dIt);	// Search the faces adjacent to dNext and eNext
		Dart e2 = phi2(*eIt);
		phi2unsew(d2);		// Unlink the two adjacent faces from dNext and eNext
		phi2unsew(e2);
		phi2sew(d2, e2);	// Link the two adjacent faces together
	}

	if(deleteFace)
	{
		Map1::deleteCycle(d);		// Delete the two alone faces
		Map1::deleteCycle(e);
	}

	return true ;
}

void Map2::splitSurface(std::vector<Dart>& vd, bool firstSideClosed, bool secondSideClosed)
{
//	assert(checkSimpleOrientedPath(vd)) ;

	Dart e = vd.front() ;
	Dart e2 = phi2(e) ;

	//unsew the edge path
	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		//if(!Map2::isBoundaryEdge(*it))
			unsewFaces(*it) ;
	}

	if(firstSideClosed)
		Map2::fillHole(e) ;

	if(secondSideClosed)
		Map2::fillHole(e2) ;
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

bool Map2::sameOrientedVertex(Dart d, Dart e) const
{
	Dart it = d;				// Foreach dart dNext in the vertex of d
	do
	{
		if (it == e)			// Test equality with e
			return true;
		it = phi2(phi_1(it));
	} while (it != d);
	return false;				// None is equal to e => vertices are distinct
}

unsigned int Map2::vertexDegree(Dart d) const
{
	unsigned int count = 0 ;
	Dart it = d ;
	do
	{
		++count ;
		it = phi2(phi_1(it)) ;
	} while (it != d) ;
	return count ;
}

int Map2::checkVertexDegree(Dart d, unsigned int vd) const
{
	unsigned int count = 0 ;
	Dart it = d ;
	do
	{
		++count ;
		it = phi2(phi_1(it)) ;
	} while ((count<=vd) && (it != d)) ;

	return count-vd;
}

bool Map2::isBoundaryVertex(Dart d) const
{
	Dart it = d ;
	do
	{
		if (isBoundaryMarked2(it))
			return true ;
		it = phi2(phi_1(it)) ;
	} while (it != d) ;
	return false ;
}

Dart Map2::findBoundaryEdgeOfVertex(Dart d) const
{
	Dart it = d ;
	do
	{
		if (isBoundaryMarked2(it))
			return it ;
		it = phi2(phi_1(it)) ;
	} while (it != d) ;
	return NIL ;
}

Dart Map2::findBoundaryEdgeOfFace(Dart d) const
{
	Dart it = d ;
	do
	{
		if (isBoundaryMarked2(phi2(it)))
			return phi2(it) ;
		it = phi1(it) ;
	} while (it != d) ;
	return NIL ;
}

bool Map2::isBoundaryFace(Dart d) const
{
	Dart it = d ;
	do
	{
		if (isBoundaryMarked2(phi2(it)))
			return true ;
		it = phi1(it) ;
	} while (it != d) ;
	return false ;
}

bool Map2::sameOrientedVolume(Dart d, Dart e) const
{
	DartMarkerStore mark(*this);	// Lock a marker

	std::list<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.push_back(d);		// Start with the face of d
	std::list<Dart>::iterator face;

	// For every face added to the list
	for (face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
	{
		if (!isBoundaryMarked2(*face) && !mark.isMarked(*face))		// Face has not been visited yet
		{
			Dart it = *face ;
			do
			{
				if(it == e)
					return true;

				mark.mark(it);						// Mark
				Dart adj = phi2(it);				// Get adjacent face
				if (!isBoundaryMarked2(adj) && !mark.isMarked(adj))
					visitedFaces.push_back(adj);	// Add it
				it = phi1(it);
			} while(it != *face);
		}
	}
	return false;
}

unsigned int Map2::volumeDegree(Dart d) const
{
	unsigned int count = 0;
	DartMarkerStore mark(*this);		// Lock a marker

	std::vector<Dart> visitedFaces;		// Faces that are traversed
	visitedFaces.reserve(16);
	visitedFaces.push_back(d);			// Start with the face of d
	std::vector<Dart>::iterator face;

	// For every face added to the list
	for (unsigned int i = 0; i != visitedFaces.size(); ++i)
	{
		Dart df = visitedFaces[i];
		if (!isBoundaryMarked2(df) && !mark.isMarked(df))		// Face has not been visited yet
		{
			++count;
			Dart it = df ;
			do
			{
				mark.mark(it);					// Mark
				Dart adj = phi2(it);			// Get adjacent face
				if ( !isBoundaryMarked2(adj) && !mark.isMarked(adj) )
					visitedFaces.push_back(adj);// Add it
				it = phi1(it);
			} while(it != df);
		}
	}

	return count;
}



int Map2::checkVolumeDegree(Dart d, unsigned int volDeg) const
{
	unsigned int count = 0;
	DartMarkerStore mark(*this);		// Lock a marker

	std::vector<Dart> visitedFaces;		// Faces that are traversed
	visitedFaces.reserve(16);
	visitedFaces.push_back(d);			// Start with the face of d

	// For every face added to the list
	for (unsigned int i = 0; i != visitedFaces.size(); ++i)
	{
		Dart df = visitedFaces[i];
		if (!isBoundaryMarked2(df) && !mark.isMarked(df))		// Face has not been visited yet
		{
			++count;
			Dart it = df ;
			do
			{
				mark.mark(it);					// Mark
				Dart adj = phi2(it);			// Get adjacent face
				if ( !isBoundaryMarked2(adj) && !mark.isMarked(adj) )
					visitedFaces.push_back(adj);// Add it
				it = phi1(it);
			} while(it != df);
		}
		if (count > volDeg)
			break;
	}

	return count - volDeg;
}



bool Map2::isTriangular() const
{
	TraversorF<Map2> t(*this) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
	{
		if(faceDegree(d) != 3)
			return false ;
	}
	return true ;
}

bool Map2::check() const
{
	CGoGNout << "Check: topology begin" << CGoGNendl;
	DartMarker m(*this);
	for(Dart d = Map2::begin(); d != Map2::end(); Map2::next(d))
	{
		Dart d2 = phi2(d);
		if (phi2(d2) != d)	// phi2 involution ?
		{
			CGoGNout << "Check: phi2 is not an involution" << CGoGNendl;
			return false;
		}
		if(d2 == d)
		{
			CGoGNout << "Check: phi2 fixed point" << CGoGNendl;
			return false;
		}

		Dart d1 = phi1(d);
		if (phi_1(d1) != d)	// phi1 a une image correcte ?
		{
			CGoGNout << "Check: inconsistent phi_1 link" << CGoGNendl;
			return false;
		}

		if (m.isMarked(d1))	// phi1 a un seul antécédent ?
		{
			CGoGNout << "Check: dart with two phi1 predecessors" << CGoGNendl;
			return false;
		}
		m.mark(d1);

		if (d1 == d)
			CGoGNout << "Check: (warning) face loop (one edge)" << CGoGNendl;
		if (phi1(d1) == d)
			CGoGNout << "Check: (warning) face with only two edges" << CGoGNendl;
		if (phi2(d1) == d)
			CGoGNout << "Check: (warning) dangling edge" << CGoGNendl;
	}

	for(Dart d = Map2::begin(); d != Map2::end(); Map2::next(d))
	{
		if (!m.isMarked(d))	// phi1 a au moins un antécédent ?
		{
			CGoGNout << "Check: dart with no phi1 predecessor" << CGoGNendl;
			return false;
		}
	}

	CGoGNout << "Check: topology ok" << CGoGNendl;

	return true;
}

bool Map2::checkSimpleOrientedPath(std::vector<Dart>& vd)
{
	DartMarkerStore dm(*this) ;
	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		if(dm.isMarked(*it))
			return false ;
		dm.markOrbit<VERTEX>(*it) ;

		std::vector<Dart>::iterator prev ;
		if(it == vd.begin())
			prev = vd.end() - 1 ;
		else
			prev = it - 1 ;

		if(!sameVertex(*it, phi1(*prev)))
			return false ;
	}
	return true ;
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

bool Map2::foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int /*thread*/) const
{
	Dart dNext = d;
	do
	{
		if (f(dNext))
			return true;
		dNext = phi2(phi_1(dNext));
 	} while (dNext != d);
 	return false;
}

bool Map2::foreach_dart_of_edge(Dart d, FunctorType& fonct, unsigned int /*thread*/) const
{
	if (fonct(d))
		return true;
	return fonct(phi2(d));
}

bool Map2::foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread) const
{
	DartMarkerStore mark(*this, thread);	// Lock a marker
	bool found = false;				// Last functor return value

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(d);		// Start with the face of d

	// For every face added to the list
	for(unsigned int i = 0; !found && i < visitedFaces.size(); ++i)
	{
		if (!mark.isMarked(visitedFaces[i]))	// Face has not been visited yet
		{
			// Apply functor to the darts of the face
			found = Map2::foreach_dart_of_face(visitedFaces[i], f);

			// If functor returns false then mark visited darts (current face)
			// and add non visited adjacent faces to the list of face
			if (!found)
			{
				Dart e = visitedFaces[i] ;
				do
				{
					mark.mark(e);				// Mark
					Dart adj = phi2(e);			// Get adjacent face
					if (!mark.isMarked(adj))
						visitedFaces.push_back(adj);	// Add it
					e = phi1(e);
				} while(e != visitedFaces[i]);
			}
		}
	}
	return found;
}

/*! @name Close map after import or creation
 *  These functions must be used with care, generally only by import/creation algorithms
 *************************************************************************/

Dart Map2::newBoundaryCycle(unsigned int nbE)
{
	Dart d = Map1::newCycle(nbE);
	boundaryMarkOrbit<FACE,2>(d);
	return d;
}

unsigned int Map2::closeHole(Dart d, bool forboundary)
{
	assert(phi2(d) == d);		// Nothing to close

	Dart first = newDart();		// First edge of the face that will fill the hole
	unsigned int countEdges = 1;

	phi2sew(d, first);	// phi2-link the new edge to the hole

	Dart dNext = d;	// Turn around the hole
	Dart dPhi1;		// to complete the face
	do
	{
		do
		{
			dPhi1 = phi1(dNext);	// Search and put in dNext
			dNext = phi2(dPhi1);	// the next dart of the hole
		} while (dNext != dPhi1 && dPhi1 != d);

		if (dPhi1 != d)
		{
			Dart next = newDart();	// Add a new edge there and link it to the face
			++countEdges;
			phi1sew(first, next);	// the edge is linked to the face
			phi2sew(dNext, next);	// the face is linked to the hole
		}
	} while (dPhi1 != d);

	if (forboundary)
		boundaryMarkOrbit<FACE,2>(phi2(d));

	return countEdges ;
}

unsigned int Map2::closeMap(bool forboundary)
{
	// Search the map for topological holes (fix points of phi2)
	unsigned int nb = 0 ;
	for (Dart d = begin(); d != end(); next(d))
	{
		if (phi2(d) == d)
		{
			++nb ;
			closeHole(d, forboundary);
		}
	}
	return nb ;
}

/*! @name Compute dual
 * These functions compute the dual mesh
 *************************************************************************/

void Map2::reverseOrientation()
{
	DartAttribute<unsigned int> emb0(this, getEmbeddingAttributeVector<VERTEX>()) ;
	if(emb0.isValid())
	{
		DartAttribute<unsigned int> new_emb0 = addAttribute<unsigned int, DART>("new_EMB_0") ;
		for(Dart d = begin(); d != end(); next(d))
			new_emb0[d] = emb0[phi1(d)] ;

		swapAttributes<unsigned int>(emb0, new_emb0) ;
		removeAttribute(new_emb0) ;
	}

	DartAttribute<Dart> n_phi1 = getAttribute<Dart, DART>("phi1") ;
	DartAttribute<Dart> n_phi_1 = getAttribute<Dart, DART>("phi_1") ;
	swapAttributes<Dart>(n_phi1, n_phi_1) ;
}

void Map2::computeDual()
{
	DartAttribute<Dart> old_phi1 = getAttribute<Dart, DART>("phi1");
	DartAttribute<Dart> old_phi_1 = getAttribute<Dart, DART>("phi_1") ;
	DartAttribute<Dart> new_phi1 = addAttribute<Dart, DART>("new_phi1") ;
	DartAttribute<Dart> new_phi_1 = addAttribute<Dart, DART>("new_phi_1") ;

	for(Dart d = begin(); d != end(); next(d))
	{
		Dart dd = phi1(phi2(d));

		new_phi1[d] = dd ;
		new_phi_1[dd] = d ;
	}

	swapAttributes<Dart>(old_phi1, new_phi1) ;
	swapAttributes<Dart>(old_phi_1, new_phi_1) ;

	removeAttribute(new_phi1) ;
	removeAttribute(new_phi_1) ;

	swapEmbeddingContainers(VERTEX, FACE) ;

	reverseOrientation() ;

	//boundary management
	for(Dart d = begin(); d != end(); next(d))
	{
		if(isBoundaryMarked2(d))
		{
			boundaryMarkOrbit<FACE,2>(deleteVertex(phi2(d)));
		}
	}
}

} // namespace CGoGN
