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

#include "Topology/gmap/gmap2.h"
#include "Topology/generic/traversorCell.h"

namespace CGoGN
{

void GMap2::compactTopoRelations(const std::vector<unsigned int>& oldnew)
{
	for (unsigned int i = m_attribs[DART].begin(); i != m_attribs[DART].end(); m_attribs[DART].next(i))
	{
		{
			Dart& d = m_beta0->operator [](i);
			Dart e = Dart(oldnew[d.index]);
			if (d != e)
				d = e;
		}
		{
			Dart& d = m_beta1->operator [](i);
			Dart e = Dart(oldnew[d.index]);
			if (d != e)
				d = e;
		}
		{
			Dart& d = m_beta2->operator [](i);
			Dart e = Dart(oldnew[d.index]);
			if (d != e)
				d = e;
		}
	}
}

/*! @name Generator and Deletor
 *  To generate or delete faces in a 2-G-map
 *************************************************************************/

Dart GMap2::newFace(unsigned int nbEdges, bool withBoundary)
{
	Dart d = GMap1::newCycle(nbEdges);
	if (withBoundary)
	{
		Dart e = newBoundaryCycle(nbEdges);

		Dart it = d;
		do
		{
			beta2sew(it, beta0(e));
			beta2sew(beta0(it), e);
			it = phi1(it);
			e = phi_1(e);
		} while (it != d);
	}
	return d;
}

void GMap2::deleteFace(Dart d)
{
	assert(!isBoundaryMarked2(d)) ;
	Dart it = d ;
	do
	{
		if(!isBoundaryEdge(it))
			unsewFaces(it) ;
		it = phi1(it) ;
	} while(it != d) ;
	Dart dd = phi2(d) ;
	GMap1::deleteCycle(d) ;
	GMap1::deleteCycle(dd) ;
}

void GMap2::deleteCC(Dart d)
{
	DartMarkerStore mark(*this);

	std::vector<Dart> visited;
	visited.reserve(1024) ;
	visited.push_back(d);
	mark.mark(d) ;

	for(unsigned int i = 0; i < visited.size(); ++i)
	{
		Dart d0 = beta0(visited[i]) ;
		if(!mark.isMarked(d0))
		{
			visited.push_back(d0) ;
			mark.mark(d0);
		}
		Dart d1 = beta1(visited[i]) ;
		if(!mark.isMarked(d1))
		{
			visited.push_back(d1) ;
			mark.mark(d1);
		}
		Dart d2 = beta2(visited[i]) ;
		if(!mark.isMarked(d2))
		{
			visited.push_back(d2) ;
			mark.mark(d2);
		}
	}

	for(std::vector<Dart>::iterator it = visited.begin(); it != visited.end(); ++it)
		deleteDart(*it) ;
}

void GMap2::fillHole(Dart d)
{
	assert(isBoundaryEdge(d)) ;
	Dart dd = d ;
	if(!isBoundaryMarked2(dd))
		dd = phi2(dd) ;
	boundaryUnmarkOrbit<FACE,2>(dd) ;
}

/*! @name Topological Operators
 *  Topological operations on 2-G-maps
 *************************************************************************/

void GMap2::splitVertex(Dart d, Dart e)
{
	assert(sameVertex(d, e));

	if(!sameOrientedVertex(d, e))
		e = beta2(e) ;

	Dart dd = phi2(d) ;
	Dart ee = phi2(e) ;
	beta2unsew(d) ;
	beta2unsew(e) ;

	GMap1::cutEdge(dd);			// Cut the edge of dd (make a new edge)
	GMap1::cutEdge(ee);			// Cut the edge of ee (make a new edge)

	beta2sew(phi1(dd), beta0(phi1(ee)));	// Sew the two faces along the new edge
	beta2sew(phi1(ee), beta0(phi1(dd)));
	beta2sew(d, beta0(dd)) ;
	beta2sew(e, beta0(ee)) ;
}

Dart GMap2::deleteVertex(Dart d)
{
	if(isBoundaryVertex(d))
		return NIL ;

	Dart res = NIL ;
	Dart vit = d ;
	do
	{
		if(res == NIL && phi1(phi1(d)) != d)
			res = phi1(d) ;

		Dart d0 = beta0(vit) ;
		Dart d02 = beta2(d0) ;
		Dart d01 = beta1(d0) ;
		Dart d021 = beta1(d02) ;
		beta1unsew(d0) ;
		beta1unsew(d02) ;
		beta1sew(d0, d02) ;
		beta1sew(d01, d021) ;

		vit = alpha1(vit) ;
	} while(vit != d) ;
	GMap1::deleteCycle(d) ;
	return res ;
}

Dart GMap2::cutEdge(Dart d)
{
	Dart e = phi2(d) ;
	Dart nd = GMap1::cutEdge(d) ;
	Dart ne = GMap1::cutEdge(e) ;

	beta2sew(nd, beta1(ne)) ;
	beta2sew(ne, beta1(nd)) ;

	return nd ;
}

bool GMap2::uncutEdge(Dart d)
{
	if(vertexDegree(phi1(d)) == 2)
	{
		GMap1::uncutEdge(d) ;
		GMap1::uncutEdge(beta2(d)) ;
		return true ;
	}
	return false ;
}

Dart GMap2::collapseEdge(Dart d, bool delDegenerateFaces)
{
	Dart resV = NIL ;

	Dart e = phi2(d);
	beta2unsew(d);	// Unlink the opposite edges
	beta2unsew(e);

	Dart f = phi1(e) ;
	Dart h = alpha1(e);

	if (h != e)
		resV = h;

	if (f != e && delDegenerateFaces)
	{
		GMap1::collapseEdge(e) ;		// Collapse edge e
		collapseDegeneratedFace(f) ;	// and collapse its face if degenerated
	}
	else
		GMap1::collapseEdge(e) ;	// Just collapse edge e

	f = phi1(d) ;
	if(resV == NIL)
	{
		h = alpha1(d);
		if (h != d)
			resV = h;
	}

	if (f != d && delDegenerateFaces)
	{
		GMap1::collapseEdge(d) ;		// Collapse edge d
		collapseDegeneratedFace(f) ;	// and collapse its face if degenerated
	}
	else
		GMap1::collapseEdge(d) ;	// Just collapse edge d

	return resV ;
}

bool GMap2::flipEdge(Dart d)
{
	if (!isBoundaryEdge(d))
	{
		Dart e = phi2(d);
		Dart dNext = phi1(d);
		Dart eNext = phi1(e);
		Dart dPrev = phi_1(d);
		Dart ePrev = phi_1(e);
		Dart dNext2 = phi1(dNext);
		Dart eNext2 = phi1(eNext);

		beta1unsew(d) ;
		beta1unsew(eNext) ;
		beta1unsew(e) ;
		beta1unsew(dNext) ;
		beta1unsew(dNext2) ;
		beta1unsew(eNext2) ;

		beta1sew(beta0(e), eNext2) ;
		beta1sew(d, beta0(eNext)) ;
		beta1sew(beta0(d), dNext2) ;
		beta1sew(e, beta0(dNext)) ;
		beta1sew(eNext, beta0(dPrev)) ;
		beta1sew(dNext, beta0(ePrev)) ;

		return true ;
	}
	return false ; // cannot flip a border edge
}

bool GMap2::flipBackEdge(Dart d)
{
	if (!isBoundaryEdge(d))
	{
		Dart e = phi2(d);
		Dart dNext = phi1(d);
		Dart eNext = phi1(e);
		Dart dPrev = phi_1(d);
		Dart ePrev = phi_1(e);
		Dart dPrev1 = beta1(dPrev) ;
		Dart ePrev1 = beta1(ePrev) ;

		beta1unsew(d) ;
		beta1unsew(eNext) ;
		beta1unsew(e) ;
		beta1unsew(dNext) ;
		beta1unsew(dPrev) ;
		beta1unsew(ePrev) ;

		beta1sew(beta0(e), dPrev) ;
		beta1sew(d, dPrev1) ;
		beta1sew(beta0(d), ePrev) ;
		beta1sew(e, ePrev1) ;
		beta1sew(eNext, beta0(dPrev)) ;
		beta1sew(dNext, beta0(ePrev)) ;

		return true ;
	}
	return false ; // cannot flip a border edge
}

//void GMap2::insertEdgeInVertex(Dart d, Dart e)
//{
//	assert(!sameVertex(d,e) && phi2(e)==phi_1(e));
//
//	phi1sew(phi_1(d),phi_1(e));
//}
//
//void GMap2::removeEdgeFromVertex(Dart d)
//{
//	assert(phi2(d)!=d);
//
//	phi1sew(phi_1(d),phi2(d));
//}

void GMap2::sewFaces(Dart d, Dart e, bool withBoundary)
{
	// if sewing with fixed points
	if (!withBoundary)
	{
		assert(beta2(d) == d && beta2(beta0(d)) == beta0(d) && beta2(e) == e && beta2(beta0(e)) == beta0(e)) ;
		beta2sew(d, beta0(e)) ;
		beta2sew(e, beta0(d)) ;
		return ;
	}

	assert(isBoundaryEdge(d) && isBoundaryEdge(e)) ;

	Dart dd = phi2(d) ;
	Dart ee = phi2(e) ;

	beta2unsew(d) ;	// unsew faces from boundary
	beta2unsew(beta0(d)) ;
	beta2unsew(e) ;
	beta2unsew(beta0(e)) ;

	if (ee != phi_1(dd))
	{
		Dart eeN = phi1(ee) ;		// remove the boundary edge
		Dart dd1 = beta1(dd) ;
		beta1unsew(eeN) ;
		beta1unsew(dd1) ;
		beta1sew(beta0(ee), dd) ;
		beta1sew(eeN, dd1) ;
	}
	if (dd != phi_1(ee))
	{
		Dart ddN = phi1(dd) ;		// and properly close incident boundaries
		Dart ee1 = beta1(ee) ;
		beta1unsew(ddN) ;
		beta1unsew(ee1) ;
		beta1sew(beta0(dd), ee) ;
		beta1sew(ddN, ee1) ;
	}
	GMap1::deleteCycle(dd) ;

	beta2sew(d, beta0(e)) ; // sew the faces
	beta2sew(e, beta0(d)) ;
}

void GMap2::unsewFaces(Dart d)
{
	assert(!isBoundaryEdge(d)) ;

	Dart dd = phi2(d);

	Dart e = newBoundaryCycle(2);
	Dart ee = phi1(e) ;

	Dart f = findBoundaryEdgeOfVertex(d) ;
	if (f != NIL)
	{
		Dart f1 = beta1(f) ;
		beta1unsew(ee) ;
		beta1unsew(f) ;
		beta1sew(f, beta0(e)) ;
		beta1sew(f1, ee) ;
	}

	f = findBoundaryEdgeOfVertex(dd) ;
	if (f != NIL)
	{
		Dart f1 = beta1(f) ;
		beta1unsew(e) ;
		beta1unsew(f) ;
		beta1sew(f, beta0(ee)) ;
		beta1sew(f1, e) ;
	}

	beta2unsew(d) ;
	beta2unsew(beta0(d)) ;
	beta2unsew(dd) ;
	beta2unsew(beta0(dd)) ;

	beta2sew(d, beta0(e)) ;		// sew faces
	beta2sew(e, beta0(d)) ;
	beta2sew(dd, beta0(ee)) ;	// to the boundary
	beta2sew(ee, beta0(dd)) ;
}

bool GMap2::collapseDegeneratedFace(Dart d)
{
	Dart e = phi1(d);				// Check if the face is a loop
	if (phi1(e) == d)				// Yes: it contains one or two edge(s)
	{
		Dart d2 = phi2(d) ;			// Check opposite edges
		Dart e2 = phi2(e) ;
		beta2unsew(d) ;
		beta2unsew(beta0(d)) ;
		if(d != e)
		{
			beta2unsew(e) ;
			beta2unsew(beta0(e)) ;
			beta2sew(d2, beta0(e2)) ;
			beta2sew(e2, beta0(d2)) ;
		}
		else
		{
			Dart d21 = beta1(d2) ;
			Dart d2N = phi1(d2) ;
			beta1unsew(d2) ;
			beta1unsew(d2N) ;
			beta1sew(d21, d2N) ;
			beta1sew(d2, beta0(d2)) ;
			GMap1::deleteCycle(d2) ;
		}
		GMap1::deleteCycle(d) ;
		return true ;
	}
	return false ;
}

void GMap2::splitFace(Dart d, Dart e)
{
	assert(d != e && sameFace(d, e)) ;

	if(!sameOrientedFace(d, e))
		e = beta1(e) ;

	Dart dprev = phi_1(d) ;
	Dart eprev = phi_1(e) ;

	//required to unsew and resew because we use GMap1 cutEdge
	//which insert new darts within the cut edge
	beta2unsew(beta1(d)) ;
	beta2unsew(beta1(e)) ;

	Dart dd = GMap1::cutEdge(phi_1(d)) ;
	Dart ee = GMap1::cutEdge(phi_1(e)) ;
	GMap1::splitCycle(dd, ee) ;
	beta2sew(dd, beta1(e)) ;
	beta2sew(ee, beta1(d)) ;

	beta2sew(beta0(dprev), beta0(beta2(dprev))) ;
	beta2sew(beta0(eprev), beta0(beta2(eprev))) ;
}

bool GMap2::mergeFaces(Dart d)
{
	if (!isBoundaryEdge(d))
	{
		Dart e = phi2(d) ;
		beta2unsew(d) ;
		beta2unsew(e) ;
		GMap1::mergeCycles(d, phi1(e)) ;
		GMap1::splitCycle(e, phi1(d)) ;
		GMap1::deleteCycle(d) ;
		return true ;
	}
	return false ;
}

void GMap2::extractTrianglePair(Dart d)
{
	Dart e = phi2(d) ;

	assert(!isBoundaryFace(d) && !isBoundaryFace(e)) ;
	assert(faceDegree(d) == 3 && faceDegree(e) == 3) ;

	Dart d1 = phi2(phi1(d)) ;
	Dart d2 = phi2(phi_1(d)) ;
	beta2unsew(d1) ;
	beta2unsew(beta0(d1)) ;
	beta2unsew(d2) ;
	beta2unsew(beta0(d2)) ;
	beta2sew(d1, beta0(d2)) ;
	beta2sew(d2, beta0(d1)) ;

	Dart e1 = phi2(phi1(e)) ;
	Dart e2 = phi2(phi_1(e)) ;
	beta2unsew(e1) ;
	beta2unsew(beta0(e1)) ;
	beta2unsew(e2) ;
	beta2unsew(beta0(e2)) ;
	beta2sew(e1, beta0(e2)) ;
	beta2sew(e2, beta0(e1)) ;
}

void GMap2::insertTrianglePair(Dart d, Dart v1, Dart v2)
{
	Dart e = phi2(d) ;

	assert(v1 != v2 && sameOrientedVertex(v1, v2)) ;
	assert(faceDegree(d) == 3 && faceDegree(phi2(d)) == 3) ;

	Dart vv1 = phi2(v1) ;
	beta2unsew(v1) ;
	beta2unsew(vv1) ;
	beta2sew(phi_1(d), beta0(v1)) ;
	beta2sew(beta1(d), v1) ;
	beta2sew(phi1(d), beta0(vv1)) ;
	beta2sew(beta0(phi1(d)), vv1) ;

	Dart vv2 = phi2(v2) ;
	beta2unsew(v2) ;
	beta2unsew(vv2) ;
	beta2sew(phi_1(e), beta0(v2)) ;
	beta2sew(beta1(e), v2) ;
	beta2sew(phi1(e), beta0(vv2)) ;
	beta2sew(beta0(phi1(e)), vv2) ;
}

bool GMap2::mergeVolumes(Dart d, Dart e)
{
	assert(!isBoundaryMarked2(d) && !isBoundaryMarked2(e)) ;

	if (GMap2::isBoundaryFace(d) || GMap2::isBoundaryFace(e))
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
		beta2unsew(d2);		// Unlink the two adjacent faces from dNext and eNext
		beta2unsew(beta0(d2));
		beta2unsew(e2);
		beta2unsew(beta0(e2));
		beta2sew(d2, beta0(e2));	// Link the two adjacent faces together
		beta2sew(e2, beta0(d2));
	}
	GMap1::deleteCycle(d);		// Delete the two alone faces
	GMap1::deleteCycle(e);

	return true ;
}

void GMap2::splitSurface(std::vector<Dart>& vd, bool firstSideClosed, bool secondSideClosed)
{
	//assert(checkSimpleOrientedPath(vd)) ;
	Dart e = vd.front() ;
	Dart e2 = phi2(e) ;

	//unsew the edge path
	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		if(!GMap2::isBoundaryEdge(*it))
			unsewFaces(*it) ;
	}

	if(firstSideClosed)
		GMap2::fillHole(e) ;

	if(secondSideClosed)
		GMap2::fillHole(e2) ;
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

bool GMap2::sameOrientedVertex(Dart d, Dart e)
{
	Dart it = d;
	do
	{
		if (it == e)	// Test equality with e
			return true;
		it = alpha1(it);
	} while (it != d);
	return false;		// None is equal to e => vertices are distinct
}

unsigned int GMap2::vertexDegree(Dart d)
{
	unsigned int count = 0 ;
	Dart it = d ;
	do
	{
		++count ;
		it = alpha1(it) ;
	} while (it != d) ;
	return count ;
}

int GMap2::checkVertexDegree(Dart d, unsigned int vd)
{
	unsigned int count = 0 ;
	Dart it = d ;
	do
	{
		++count ;
		it = alpha1(it) ;
	} while ((count<=vd) && (it != d)) ;

	return count-vd;
}


bool GMap2::isBoundaryVertex(Dart d)
{
	Dart it = d ;
	do
	{
		if (isBoundaryMarked2(it))
			return true ;
		it = alpha1(it) ;
	} while (it != d) ;
	return false ;
}

Dart GMap2::findBoundaryEdgeOfVertex(Dart d)
{
	Dart it = d ;
	do
	{
		if (isBoundaryMarked2(it))
			return it ;
		it = alpha1(it) ;
	} while (it != d) ;
	return NIL ;
}

bool GMap2::isBoundaryFace(Dart d)
{
	Dart it = d ;
	do
	{
		if (isBoundaryMarked2(beta2(it)))
			return true ;
		it = phi1(it) ;
	} while (it != d) ;
	return false ;
}

bool GMap2::sameOrientedVolume(Dart d, Dart e)
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

unsigned int GMap2::volumeDegree(Dart d)
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


int GMap2::checkVolumeDegree(Dart d, unsigned int volDeg)
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
		if (count > volDeg)
			break;
	}

	return count - volDeg;
}



bool GMap2::isTriangular()
{
	TraversorF<GMap2> t(*this) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
	{
		if(faceDegree(d) != 3)
			return false ;
	}
	return true ;
}

bool GMap2::check()
{
	CGoGNout << "Check: topology begin" << CGoGNendl;
	for(Dart d = begin(); d != end(); next(d))
	{
		Dart dd = beta0(d);
		if (beta0(dd) != d)	// beta0 involution ?
		{
			CGoGNout << "Check: beta0 is not an involution" << CGoGNendl;
			return false;
		}

		dd = beta1(d);
		if (beta1(dd) != d)	// beta1 involution ?
		{
			CGoGNout << "Check: beta1 is not an involution" << CGoGNendl;
			return false;
		}

		dd = beta2(d);
		if (beta2(dd) != d)	// beta2 involution ?
		{
			CGoGNout << "Check: beta2 is not an involution" << CGoGNendl;
			return false;
		}
		if(dd == d)
			CGoGNout << "Check (warning): beta2 has fix points" << CGoGNendl;
	}

	CGoGNout << "Check: topology ok" << CGoGNendl;

	return true;
}

bool GMap2::checkSimpleOrientedPath(std::vector<Dart>& vd)
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

bool GMap2::foreach_dart_of_oriented_vertex(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	Dart it = d;
	do
	{
		if (f(it))
			return true;
		it = alpha1(it);
 	} while (it != d);
 	return false;
}

bool GMap2::foreach_dart_of_oriented_edge(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	if (f(d))
		return true ;
	Dart e = beta2(beta0(d)) ;
	if (f(e))
		return true ;

	return false ;
}

bool GMap2::foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int /*thread*/)
{
	if (f(d))
		return true ;
	Dart e = beta0(d) ;
	if (f(e))
		return true ;
	e = beta2(d) ;
	if (f(e))
		return true ;
	e = beta0(e) ;
	if (f(e))
		return true ;

	return false ;
}

bool GMap2::foreach_dart_of_oriented_cc(Dart d, FunctorType& f, unsigned int thread)
{
	DartMarkerStore mark(*this, thread);	// Lock a marker
	bool found = false;				// Last functor return value

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(d);		// Start with the face of d

	// For every face added to the list
	for(unsigned int i = 0; !found && i < visitedFaces.size(); ++i)
	{
		if (!mark.isMarked(visitedFaces[i]))		// Face has not been visited yet
		{
			// Apply functor to the darts of the face
			found = foreach_dart_of_oriented_face(visitedFaces[i], f);

			// If functor returns false then mark visited darts (current face)
			// and add non visited adjacent faces to the list of face
			if (!found)
			{
				Dart e = visitedFaces[i] ;
				do
				{
					mark.mark(e);					// Mark
					Dart adj = phi2(e);				// Get adjacent face
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

Dart GMap2::newBoundaryCycle(unsigned int nbE)
{
	Dart d = GMap1::newCycle(nbE);
	boundaryMarkOrbit<FACE,2>(d);
	return d;
}

unsigned int GMap2::closeHole(Dart d, bool forboundary)
{
	assert(beta2(d) == d);		// Nothing to close

	Dart first = newEdge();		// First edge of the face that will fill the hole
	unsigned int countEdges = 1;

	beta2sew(d, beta0(first));	// sew the new edge to the hole
	beta2sew(first, beta0(d));

	Dart prev = first ;
	Dart dNext = d;	// Turn around the hole
	Dart dPhi1;		// to complete the face
	do
	{
		dPhi1 = phi1(dNext) ;
		dNext = beta2(dPhi1) ;
		while(dNext != dPhi1 && dPhi1 != d)
		{
			dPhi1 = beta1(dNext) ;	// Search and put in dNext
			dNext = beta2(dPhi1) ;	// the next dart of the hole
		}

		if (dPhi1 != d)
		{
			Dart next = newEdge();	// Add a new edge there and link it to the face
			++countEdges;
			beta1sew(beta0(next), prev);	// the edge is linked to the face
			prev = next ;
			beta2sew(dNext, beta0(next));	// the face is linked to the hole
			beta2sew(next, beta0(dNext));
		}
	} while (dPhi1 != d);

	beta1sew(prev, beta0(first)) ;

	if(forboundary)
		boundaryMarkOrbit<FACE,2>(phi2(d));

	return countEdges ;
}

unsigned int GMap2::closeMap()
{
	// Search the map for topological holes (fix points of phi2)
	unsigned int nb = 0 ;
	for (Dart d = begin(); d != end(); next(d))
	{
		if (beta2(d) == d)
		{
			++nb ;
			closeHole(d);
		}
	}
	return nb ;
}

/*! @name Compute dual
 * These functions compute the dual mesh
 *************************************************************************/

void GMap2::computeDual()
{
//	DartAttribute<Dart> old_beta0 = this->getAttribute<Dart, DART>("beta0");
//	DartAttribute<Dart> old_beta2 = this->getAttribute<Dart, DART>("beta2") ;
//
//	swapAttributes<Dart>(old_beta0, old_beta2) ;
//
//	swapEmbeddingContainers(VERTEX, FACE) ;
//
//	//boundary management ?
}

} // namespace CGoGN
