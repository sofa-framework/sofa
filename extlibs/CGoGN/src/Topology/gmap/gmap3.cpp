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

#include "Topology/gmap/gmap3.h"
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/traversor3.h"

namespace CGoGN
{

void GMap3::compactTopoRelations(const std::vector<unsigned int>& oldnew)
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
		{
			Dart& d = m_beta3->operator [](i);
			Dart e = Dart(oldnew[d.index]);
			if (d != e)
				d = e;
		}
	}
}

/*! @name Generator and Deletor
 *  To generate or delete volumes in a 3-G-map
 *************************************************************************/

void GMap3::deleteVolume(Dart d)
{
	DartMarkerStore mark(*this);		// Lock a marker

	std::vector<Dart> visitedFaces;		// Faces that are traversed
	visitedFaces.reserve(512);
	visitedFaces.push_back(d);			// Start with the face of d
	mark.markOrbit<FACE>(d) ;

	// For every face added to the list
	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		Dart e = visitedFaces[i] ;

		if(!isBoundaryFace(e))
			unsewVolumes(e) ;

		do	// add all face neighbours to the table
		{
			Dart ee = phi2(e) ;
			if(!mark.isMarked(ee)) // not already marked
			{
				visitedFaces.push_back(ee) ;
				mark.markOrbit<FACE>(ee) ;
			}
			e = phi1(e) ;
		} while(e != visitedFaces[i]) ;
	}

	Dart dd = phi3(d) ;
	GMap2::deleteCC(d) ;
	GMap2::deleteCC(dd) ;
}

void GMap3::fillHole(Dart d)
{
	assert(isBoundaryFace(d)) ;
	Dart dd = d ;
	if(!isBoundaryMarked3(dd))
		dd = phi3(dd) ;
	boundaryUnmarkOrbit<VOLUME,3>(dd) ;
}

/*! @name Topological Operators
 *  Topological operations on 3-G-maps
 *************************************************************************/

Dart GMap3::deleteVertex(Dart d)
{
	if(isBoundaryVertex(d))
		return NIL ;

	// Save the darts around the vertex
	// (one dart per face should be enough)
	std::vector<Dart> fstoretmp;
	fstoretmp.reserve(128);
	FunctorStore fs(fstoretmp);
	foreach_dart_of_vertex(d, fs);

	// just one dart per face
	std::vector<Dart> fstore;
	fstore.reserve(128);
	DartMarker mf(*this);
	for(unsigned int i = 0; i < fstoretmp.size(); ++i)
	{
		if(!mf.isMarked(fstoretmp[i]))
		{
			mf.markOrbit<FACE>(fstoretmp[i]);
			fstore.push_back(fstoretmp[i]);
		}
	}

	Dart res = NIL ;
	for(std::vector<Dart>::iterator it = fstore.begin() ; it != fstore.end() ; ++it)
	{
		Dart fit = *it ;
		Dart end = phi_1(fit) ;
		fit = phi1(fit) ;
		while(fit != end)
		{
			Dart d2 = phi2(fit) ;
			Dart d3 = phi3(fit) ;
			Dart d32 = phi2(d3) ;

			if(res == NIL)
				res = d2 ;

			beta2unsew(d2) ;
			beta2unsew(beta0(d2)) ;

			beta2unsew(d32) ;
			beta2unsew(beta0(d32)) ;

			beta2sew(d2, beta0(d32)) ;
			beta2sew(beta0(d2), d32) ;
			beta2sew(fit, beta0(d3)) ;
			beta2sew(beta0(fit), d3) ;

			fit = phi1(fit) ;
		}
	}

	GMap2::deleteCC(d) ;

	return res ;
}

Dart GMap3::cutEdge(Dart d)
{
	Dart prev = d ;
	Dart dd = alpha2(d) ;
	Dart nd = GMap2::cutEdge(d) ;

	while(dd != d)
	{
		prev = dd ;
		dd = alpha2(dd) ;

		GMap2::cutEdge(prev) ;

		Dart d3 = beta3(prev);
		beta3sew(beta0(prev), beta0(d3));
		beta3sew(phi1(prev), phi1(d3));
	}

	Dart d3 = beta3(d);
	beta3sew(beta0(d), beta0(d3));
	beta3sew(phi1(d), phi1(d3));

	return nd ;
}

bool GMap3::uncutEdge(Dart d)
{
	if(vertexDegree(phi1(d)) == 2)
	{
		Dart prev = d ;

		Dart dd = d;
		do
		{
			prev = dd;
			dd = alpha2(dd);

			GMap2::uncutEdge(prev);
		} while (dd != d) ;

		return true;
	}
	return false;
}

bool GMap3::deleteEdgePreCond(Dart d)
{
	unsigned int nb1 = vertexDegree(d);
	unsigned int nb2 = vertexDegree(phi1(d));
	return (nb1!=2) && (nb2!=2);
}

Dart GMap3::deleteEdge(Dart d)
{
	assert(deleteEdgePreCond(d));

	if(isBoundaryEdge(d))
		return NIL ;

	Dart res = NIL ;
	Dart dit = d ;
	do
	{
		Dart fit = dit ;
		Dart end = fit ;
		fit = phi1(fit) ;
		while(fit != end)
		{
			Dart d2 = phi2(fit) ;
			Dart d3 = phi3(fit) ;
			Dart d32 = phi2(d3) ;

			if(res == NIL)
				res = d2 ;

			beta2unsew(d2) ;
			beta2unsew(beta0(d2)) ;
			beta2unsew(d32) ;
			beta2unsew(beta0(d32)) ;

			beta2sew(d2, beta0(d32)) ;
			beta2sew(beta0(d2), d32) ;
			beta2sew(fit, beta0(d3)) ;
			beta2sew(beta0(fit), d3) ;

			fit = phi1(fit) ;
		}
		dit = alpha2(dit) ;
	} while(dit != d) ;

	GMap2::deleteCC(d) ;

	return res ;
}

bool GMap3::splitFacePreCond(Dart d, Dart e)
{
	return (d != e && GMap2::sameFace(d, e)) ;
}

void GMap3::splitFace(Dart d, Dart e)
{
	assert(splitFacePreCond(d, e));

	if(!sameOrientedFace(d, e))
		e = beta1(e) ;

	Dart dd = beta1(beta3(d));
	Dart ee = beta1(beta3(e));

	Dart dprev = phi_1(d) ;
	Dart eprev = phi_1(e) ;
	Dart ddprev = phi_1(dd) ;
	Dart eeprev = phi_1(ee) ;

	beta3unsew(beta1(d)) ;
	beta3unsew(beta1(e)) ;
	beta3unsew(beta1(dd)) ;
	beta3unsew(beta1(ee)) ;

	GMap2::splitFace(d, e);
	GMap2::splitFace(dd, ee);
	beta3sew(beta1(d), phi_1(ee));
	beta3sew(phi_1(d), beta1(ee));
	beta3sew(beta1(e), phi_1(dd));
	beta3sew(phi_1(e), beta1(dd));

	beta3sew(beta0(dprev), beta0(beta3(dprev))) ;
	beta3sew(beta0(eprev), beta0(beta3(eprev))) ;
	beta3sew(beta0(ddprev), beta0(beta3(ddprev))) ;
	beta3sew(beta0(eeprev), beta0(beta3(eeprev))) ;
}

void GMap3::sewVolumes(Dart d, Dart e, bool withBoundary)
{
	assert(faceDegree(d) == faceDegree(e));

	// if sewing with fixed points
	if (!withBoundary)
	{
		assert(beta3(d) == d && beta3(e) == e) ;
		Dart fitD = d ;
		Dart fitE = e ;
		do
		{
			beta3sew(fitD, beta0(fitE)) ;
			beta3sew(beta0(fitD), fitE) ;
			fitD = phi1(fitD) ;
			fitE = phi_1(fitE) ;
		} while(fitD != d) ;
		return ;
	}

	Dart dd = beta3(d) ;
	Dart ee = beta3(e) ;

	Dart fitD = dd ;
	Dart fitE = ee ;
	do
	{
		Dart fitD2 = beta2(fitD) ;
		Dart fitE2 = beta2(fitE) ;
		if(fitD2 != fitE)
		{
			beta2unsew(fitD) ;
			beta2unsew(fitE) ;
			beta2unsew(beta0(fitD)) ;
			beta2unsew(beta0(fitE)) ;
			beta2sew(fitD2, beta0(fitE2)) ;
			beta2sew(beta0(fitD2), fitE2) ;
			beta2sew(fitD, beta0(fitE)) ;
			beta2sew(beta0(fitD), fitE) ;
		}
		beta3unsew(fitD) ;
		beta3unsew(beta0(fitD)) ;
		beta3unsew(fitE) ;
		beta3unsew(beta0(fitE)) ;
		fitD = phi1(fitD) ;
		fitE = phi_1(fitE) ;
	} while(fitD != dd) ;
	GMap2::deleteCC(dd) ;

	fitD = d ;
	fitE = e ;
	do
	{
		beta3sew(fitD, beta0(fitE)) ;
		beta3sew(beta0(fitD), fitE) ;
		fitD = phi1(fitD) ;
		fitE = phi_1(fitE) ;
	} while(fitD != d) ;
}

void GMap3::unsewVolumes(Dart d)
{
	assert(!isBoundaryFace(d)) ;

	unsigned int nbE = faceDegree(d) ;
	Dart d3 = phi3(d);

	Dart b1 = newBoundaryCycle(nbE) ;
	Dart b2 = newBoundaryCycle(nbE) ;

	Dart fit1 = d ;
	Dart fit2 = d3 ;
	Dart fitB1 = b1 ;
	Dart fitB2 = b2 ;
	do
	{
		Dart f = findBoundaryFaceOfEdge(fit1) ;
		if(f != NIL)
		{
			Dart f2 = phi2(f) ;
			beta2unsew(f) ;
			beta2unsew(beta0(f)) ;
			beta2sew(fitB1, beta0(f)) ;
			beta2sew(beta0(fitB1), f) ;
			beta2sew(fitB2, beta0(f2)) ;
			beta2sew(beta0(fitB2), f2) ;
		}
		else
		{
			beta2sew(fitB1, beta0(fitB2)) ;
			beta2sew(beta0(fitB1), fitB2) ;
		}

		beta3unsew(fit1) ;
		beta3unsew(beta0(fit1)) ;
		beta3sew(fit1, beta0(fitB1)) ;
		beta3sew(beta0(fit1), fitB1) ;
		beta3sew(fit2, beta0(fitB2)) ;
		beta3sew(beta0(fit2), fitB2) ;

		fit1 = phi1(fit1) ;
		fit2 = phi_1(fit2) ;
		fitB1 = phi_1(fitB1) ;
		fitB2 = phi1(fitB2) ;
	} while(fitB1 != b1) ;
}

bool GMap3::mergeVolumes(Dart d)
{
	if(!GMap3::isBoundaryFace(d))
	{
		GMap2::mergeVolumes(d, phi3(d)); // merge the two volumes along common face
		return true ;
	}
	return false ;
}

void GMap3::splitVolume(std::vector<Dart>& vd)
{
	assert(checkSimpleOrientedPath(vd)) ;

	Dart e = vd.front();
	Dart e2 = phi2(e);

	GMap2::splitSurface(vd,true,true);

	//sew the two connected components
	GMap3::sewVolumes(phi2(e), phi2(e2), false);

//	Dart e = vd.front();
//	Dart e2 = phi2(e);
//
//	//unsew the edge path
//	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
//		GMap2::unsewFaces(*it);
//
//	GMap2::fillHole(e) ;
//	GMap2::fillHole(e2) ;
//
//	//sew the two connected components
//	GMap3::sewVolumes(beta2(e), beta2(e2), false);
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

bool GMap3::sameOrientedVertex(Dart d, Dart e) const
{
	DartMarkerStore mv(*this);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		if(darts[i] == e)
			return true;

		// add phi21 and phi23 successor if they are not marked yet
		Dart d2 = phi2(darts[i]);
		Dart d21 = phi1(d2); // turn in volume
		Dart d23 = phi3(d2); // change volume

		if(!mv.isMarked(d21))
		{
			darts.push_back(d21);
			mv.mark(d21);
		}
		if(!mv.isMarked(d23))
		{
			darts.push_back(d23);
			mv.mark(d23);
		}
	}
	return false;
}

bool GMap3::sameVertex(Dart d, Dart e) const
{
	DartMarkerStore mv(*this);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		if(darts[i] == e)
			return true;

		Dart dx = beta1(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = beta2(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = beta3(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
	}
	return false;
}

//unsigned int GMap3::vertexDegree(Dart d)
//{
//	unsigned int count = 0;
//	DartMarkerStore mv(*this);	// Lock a marker

//	std::vector<Dart> darts;	// Darts that are traversed
//	darts.reserve(256);
//	darts.push_back(d);			// Start with the dart d
//	mv.mark(d);

//	for(unsigned int i = 0; i < darts.size(); ++i)
//	{
//		//add phi21 and phi23 successor if they are not marked yet
//		Dart d2 = phi2(darts[i]);
//		Dart d21 = phi1(d2); // turn in volume
//		Dart d23 = phi3(d2); // change volume

//		if(!mv.isMarked(d21))
//		{
//			darts.push_back(d21);
//			mv.mark(d21);
//		}
//		if(!mv.isMarked(d23))
//		{
//			darts.push_back(d23);
//			mv.mark(d23);
//		}
//	}

//	DartMarkerStore me(*this);
//	for(std::vector<Dart>::iterator it = darts.begin(); it != darts.end() ; ++it)
//	{
//		if(!me.isMarked(*it))
//		{
//			++count;
//			me.markOrbit<EDGE>(*it);
//		}
//	}

//	return count;
//}

unsigned int GMap3::vertexDegree(Dart d) const
{
	unsigned int count = 0;

	Traversor3VE<GMap3> trav3VE(*this, d);
	for(Dart dit = trav3VE.begin() ; dit != trav3VE.end() ; dit = trav3VE.next())
	{
		++count;
	}

	return count;
}


int GMap3::checkVertexDegree(Dart d, unsigned int vd) const
{
	unsigned int count = 0;

	Traversor3VE<GMap3> trav3VE(*this, d);
	Dart dit = trav3VE.begin();
	for( ; (count<=vd) && (dit != trav3VE.end()) ; dit = trav3VE.next())
	{
		++count;
	}

	return count - vd;
}

bool GMap3::isBoundaryVertex(Dart d) const
{
	DartMarkerStore mv(*this);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		if(isBoundaryMarked3(darts[i]))
			return true ;

		//add phi21 and phi23 successor if they are not marked yet
		Dart d2 = phi2(darts[i]);
		Dart d21 = phi1(d2); // turn in volume
		Dart d23 = phi3(d2); // change volume

		if(!mv.isMarked(d21))
		{
			darts.push_back(d21);
			mv.mark(d21);
		}
		if(!mv.isMarked(d23))
		{
			darts.push_back(d23);
			mv.mark(d23);
		}
	}
	return false ;
}

bool GMap3::sameOrientedEdge(Dart d, Dart e) const
{
	Dart it = d;
	do
	{
		if(it == e || phi2(it) == e)
			return true;
		it = alpha2(it);
	} while (it != d);
	return false;
}

bool GMap3::sameEdge(Dart d, Dart e) const
{
	Dart it = d;
	do
	{
		if(it == e || beta0(it) == e || beta2(it) == e || phi2(it) == e)
			return true;

		it = alpha2(it);
	} while (it != d);
	return false;
}

unsigned int GMap3::edgeDegree(Dart d) const
{
	unsigned int deg = 0;
	Dart it = d;
	do
	{
		++deg;
		it = alpha2(it);
	} while(it != d);
	return deg;
}

bool GMap3::isBoundaryEdge(Dart d) const
{
	Dart it = d;
	do
	{
		if(isBoundaryMarked3(it))
			return true ;
		it = alpha2(it);
	} while(it != d);
	return false;
}

Dart GMap3::findBoundaryFaceOfEdge(Dart d) const
{
	Dart it = d;
	do
	{
		if (isBoundaryMarked3(it))
			return it ;
		it = alpha2(it);
	} while(it != d);
	return NIL ;
}

bool GMap3::sameOrientedFace(Dart d, Dart e) const
{
	Dart it = d;
	do
	{
		if(it == e || phi3(it) == e)
			return true;
		it = phi1(it);
	} while (it != d);
	return false;
}

bool GMap3::isBoundaryVolume(Dart d) const
{
	DartMarkerStore mark(*this);	// Lock a marker

	std::vector<Dart> visitedFaces ;
	visitedFaces.reserve(128) ;
	visitedFaces.push_back(d) ;
	mark.markOrbit<FACE>(d) ;

	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		if (isBoundaryMarked3(beta3(visitedFaces[i])))
			return true ;

		Dart e = visitedFaces[i] ;
		do	// add all face neighbours to the table
		{
			Dart ee = phi2(e) ;
			if(!mark.isMarked(ee)) // not already marked
			{
				visitedFaces.push_back(ee) ;
				mark.markOrbit<FACE>(ee) ;
			}
			e = phi1(e) ;
		} while(e != visitedFaces[i]) ;
	}
	return false;
}

bool GMap3::check() const
{
    CGoGNout << "Check: topology begin" << CGoGNendl;
    DartMarker m(*this);
    m.unmarkAll();
    for(Dart d = this->begin(); d != this->end(); this->next(d))
    {
        Dart d0 = beta0(d);
        if (beta0(d0) != d) // beta0 involution ?
		{
             CGoGNout << "Check: beta0 is not an involution" << CGoGNendl;
            return false;
        }

        Dart d3 = beta3(d);
        if (beta3(d3) != d) // beta3 involution ?
		{
             CGoGNout << "Check: beta3 is not an involution" << CGoGNendl;
            return false;
        }

        if(d3 != d)
        {
        	if(beta1(d3) != beta3(beta1(d)))
        	{
        		CGoGNout << "Check: beta3 , faces are not entirely sewn" << CGoGNendl;
        		return false;
        	}
        }

        Dart d2 = beta2(d);
        if (beta2(d2) != d) // beta2 involution ?
		{
            CGoGNout << "Check: beta2 is not an involution" << CGoGNendl;
            return false;
        }

        Dart d1 = phi1(d);
        if (phi_1(d1) != d) // phi1 a une image correcte ?
		{
            CGoGNout << "Check: unconsistent phi_1 link" << CGoGNendl;
            return false;
        }

        if (m.isMarked(d1)) // phi1 a un seul antécédent ?
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
            CGoGNout << "Check: (warning) dandling edge (phi2)" << CGoGNendl;

        if (phi3(d1) == d)
            CGoGNout << "Check: (warning) dandling edge (phi3)" << CGoGNendl;
    }

    for(Dart d = this->begin(); d != this->end(); this->next(d))
    {
        if (!m.isMarked(d)) // phi1 a au moins un antécédent ?
		{
        	std::cout << "dart = " << d << std::endl;
            CGoGNout << "Check: dart with no phi1 predecessor" << CGoGNendl;
            return false;
        }
    }

    CGoGNout << "Check: topology ok" << CGoGNendl;

    return true;
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

bool GMap3::foreach_dart_of_oriented_vertex(Dart d, FunctorType& f, unsigned int thread) const
{
	DartMarkerStore mv(*this, thread);	// Lock a marker
	bool found = false;					// Last functor return value

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; !found && i < darts.size(); ++i)
	{
		// add phi21 and phi23 successor if they are not marked yet
		Dart d2 = phi2(darts[i]);
		Dart d21 = phi1(d2); // turn in volume
		Dart d23 = phi3(d2); // change volume

		if(!mv.isMarked(d21))
		{
			darts.push_back(d21);
			mv.mark(d21);
		}
		if(!mv.isMarked(d23))
		{
			darts.push_back(d23);
			mv.mark(d23);
		}

		found = f(darts[i]);
	}
	return found;
}

bool GMap3::foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int thread) const
{
	DartMarkerStore mv(*this, thread);	// Lock a marker
	bool found = false;					// Last functor return value

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; !found && i < darts.size(); ++i)
	{
		Dart dx = beta1(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = beta2(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = beta3(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}

		found = f(darts[i]);
	}
	return found;
}

bool GMap3::foreach_dart_of_oriented_edge(Dart d, FunctorType& f, unsigned int thread) const
{
	Dart it = d;
	do
	{
		if (GMap2::foreach_dart_of_oriented_edge(it, f, thread))
			return true;
		it = alpha2(it);
	} while (it != d);
	return false;
}

bool GMap3::foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int thread) const
{
	Dart it = d;
	do
	{
		if (GMap2::foreach_dart_of_edge(it, f, thread))
			return true;
		it = alpha2(it);
	} while (it != d);
	return false;
}

bool GMap3::foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread) const
{
	DartMarkerStore mv(*this,thread);	// Lock a marker
	bool found = false;					// Last functor return value

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(1024);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; !found && i < darts.size(); ++i)
	{
		Dart dx = beta0(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = beta1(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = beta2(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = beta3(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}

		found =  f(darts[i]);
	}
	return found;
}

/*! @name Close map after import or creation
 *  These functions must be used with care, generally only by import/creation algorithms
 *************************************************************************/

Dart GMap3::newBoundaryCycle(unsigned int nbE)
{
	Dart d = GMap1::newCycle(nbE);
	boundaryMarkOrbit<FACE,3>(d);
	return d;
}

unsigned int GMap3::closeHole(Dart d, bool forboundary)
{
	assert(beta3(d) == d);		// Nothing to close
	DartMarkerStore m(*this) ;

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(d);		// Start with the face of d
	m.markOrbit<FACE>(d) ;

	unsigned int count = 0 ;

	// For every face added to the list
	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		Dart f = visitedFaces[i] ;
		unsigned int degree = faceDegree(f) ;
		Dart b = newBoundaryCycle(degree) ;
		++count ;

		Dart bit = b ;
		do
		{
			Dart e = alpha2(f) ;
			bool found = false ;
			do
			{
				if(beta3(e) == e)
				{
					found = true ;
					if(!m.isMarked(e))
					{
						visitedFaces.push_back(e) ;
						m.markOrbit<FACE>(e) ;
					}
				}
				else if(isBoundaryMarked3(e))
				{
					found = true ;
					beta2sew(e, bit) ;
					beta2sew(beta0(e), beta0(bit)) ;
				}
				else
					e = alpha2(e) ;
			} while(!found) ;

			beta3sew(f, bit) ;
			beta3sew(beta0(f), beta0(bit)) ;
			bit = phi1(bit) ;
			f = phi1(f);
		} while(f != visitedFaces[i]);
	}

	return count ;
}

unsigned int GMap3::closeMap()
{
	// Search the map for topological holes (fix points of beta3)
	unsigned int nb = 0 ;
	for (Dart d = begin(); d != end(); next(d))
	{
		if (beta3(d) == d)
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

void GMap3::computeDual()
{
//	DartAttribute<Dart> old_beta0 = getAttribute<Dart, DART>("beta0");
//	DartAttribute<Dart> old_beta1 = getAttribute<Dart, DART>("beta1");
//	DartAttribute<Dart> old_beta2 = getAttribute<Dart, DART>("beta2");
//	DartAttribute<Dart> old_beta3 = getAttribute<Dart, DART>("beta3") ;
//
//	swapAttributes<Dart>(old_beta0, old_beta3) ;
//	swapAttributes<Dart>(old_beta1, old_beta2) ;
//
//	swapEmbeddingContainers(VERTEX, FACE) ;
//
//	//boundary management ?
}


} // namespace CGoGN
