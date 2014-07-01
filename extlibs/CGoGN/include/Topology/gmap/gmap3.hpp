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

#include "Topology/generic/traversor/traversor3.h"

namespace CGoGN
{

template <typename MAP_IMPL>
inline void GMap3<MAP_IMPL>::init()
{
	MAP_IMPL::addInvolution() ;
}

template <typename MAP_IMPL>
inline GMap3<MAP_IMPL>::GMap3() : GMap2<MAP_IMPL>()
{
	init() ;
}

template <typename MAP_IMPL>
inline std::string GMap3<MAP_IMPL>::mapTypeName() const
{
	return "GMap3";
}

template <typename MAP_IMPL>
inline unsigned int GMap3<MAP_IMPL>::dimension() const
{
	return 3;
}

template <typename MAP_IMPL>
inline void GMap3<MAP_IMPL>::clear(bool removeAttrib)
{
	ParentMap::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

template <typename MAP_IMPL>
inline unsigned int GMap3<MAP_IMPL>::getNbInvolutions() const
{
	return 1 + ParentMap::getNbInvolutions();
}

template <typename MAP_IMPL>
inline unsigned int GMap3<MAP_IMPL>::getNbPermutations() const
{
	return ParentMap::getNbPermutations();
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

template <typename MAP_IMPL>
inline Dart GMap3<MAP_IMPL>::beta3(Dart d) const
{
	return MAP_IMPL::template getInvolution<3>(d);
}

template <typename MAP_IMPL>
template <int N>
inline Dart GMap3<MAP_IMPL>::beta(const Dart d) const
{
	assert( (N > 0) || !"negative parameters not allowed in template multi-beta");
	if (N<10)
	{
		switch(N)
		{
			case 0 : return this->beta0(d) ;
			case 1 : return this->beta1(d) ;
			case 2 : return this->beta2(d) ;
			case 3 : return beta3(d) ;
			default : assert(!"Wrong multi-beta relation value") ;
		}
	}
	switch(N%10)
	{
		case 0 : return beta0(beta<N/10>(d)) ;
		case 1 : return beta1(beta<N/10>(d)) ;
		case 2 : return beta2(beta<N/10>(d)) ;
		case 3 : return beta3(beta<N/10>(d)) ;
		default : assert(!"Wrong multi-beta relation value") ;
	}
}

template <typename MAP_IMPL>
inline Dart GMap3<MAP_IMPL>::phi3(Dart d) const
{
	return beta3(this->beta0(d)) ;
}

template <typename MAP_IMPL>
template <int N>
inline Dart GMap3<MAP_IMPL>::phi(Dart d) const
{
	assert( (N >0) || !"negative parameters not allowed in template multi-phi");
	if (N<10)
	{
		switch(N)
		{
			case 1 : return this->phi1(d) ;
			case 2 : return this->phi2(d) ;
			case 3 : return phi3(d) ;
			default : assert(!"Wrong multi-phi relation value") ; return d ;
		}
	}
	switch(N%10)
	{
		case 1 : return this->phi1(phi<N/10>(d)) ;
		case 2 : return this->phi2(phi<N/10>(d)) ;
		case 3 : return phi3(phi<N/10>(d)) ;
		default : assert(!"Wrong multi-phi relation value") ; return d ;
	}
}

template <typename MAP_IMPL>
inline Dart GMap3<MAP_IMPL>::alpha0(Dart d) const
{
	return beta3(this->beta0(d)) ;
}

template <typename MAP_IMPL>
inline Dart GMap3<MAP_IMPL>::alpha1(Dart d) const
{
	return beta3(this->beta1(d)) ;
}

template <typename MAP_IMPL>
inline Dart GMap3<MAP_IMPL>::alpha2(Dart d) const
{
	return beta3(this->beta2(d)) ;
}

template <typename MAP_IMPL>
inline Dart GMap3<MAP_IMPL>::alpha_2(Dart d) const
{
	return beta2(beta3(d)) ;
}

template <typename MAP_IMPL>
inline void GMap3<MAP_IMPL>::beta3sew(Dart d, Dart e)
{
	MAP_IMPL::template involutionSew<3>(d,e);
}

template <typename MAP_IMPL>
inline void GMap3<MAP_IMPL>::beta3unsew(Dart d)
{
	MAP_IMPL::template involutionUnsew<3>(d);
}

/*! @name Generator and Deletor
 *  To generate or delete volumes in a 3-G-map
 *************************************************************************/

template <typename MAP_IMPL>
void GMap3<MAP_IMPL>::deleteVolume(Dart d)
{
	DartMarkerStore<GMap3<MAP_IMPL> > mark(*this);	// Lock a marker

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
			Dart ee = this->phi2(e) ;
			if(!mark.isMarked(ee)) // not already marked
			{
				visitedFaces.push_back(ee) ;
				mark.markOrbit<FACE>(ee) ;
			}
			e = this->phi1(e) ;
		} while(e != visitedFaces[i]) ;
	}

	Dart dd = phi3(d) ;
	ParentMap::deleteCC(d) ;
	ParentMap::deleteCC(dd) ;
}

template <typename MAP_IMPL>
void GMap3<MAP_IMPL>::fillHole(Dart d)
{
	assert(isBoundaryFace(d)) ;
	Dart dd = d ;
	if(!this->template isBoundaryMarked<3>(dd))
		dd = phi3(dd) ;
	Algo::Topo::boundaryUnmarkOrbit<3,VOLUME>(*this, dd) ;
}

/*! @name Topological Operators
 *  Topological operations on 3-G-maps
 *************************************************************************/

template <typename MAP_IMPL>
Dart GMap3<MAP_IMPL>::deleteVertex(Dart d)
{
	if(isBoundaryVertex(d))
		return NIL ;

	// Save the darts around the vertex
	// (one dart per face should be enough)
	std::vector<Dart> fstoretmp;
	fstoretmp.reserve(128);
//	this->template foreach_dart_of_orbit<VERTEX>(d, [&] (Dart it) { fstoretmp.push_back(it); });
    this->template foreach_dart_of_orbit<VERTEX>(d, boost::bind(&std::vector<Dart>::push_back, boost::ref(fstoretmp), bl::_1));

	// just one dart per face
	std::vector<Dart> fstore;
	fstore.reserve(128);
	DartMarker<GMap3<MAP_IMPL> > mf(*this);
	for(unsigned int i = 0; i < fstoretmp.size(); ++i)
	{
		if(!mf.isMarked(fstoretmp[i]))
		{
			mf.template markOrbit<FACE>(fstoretmp[i]);
			fstore.push_back(fstoretmp[i]);
		}
	}

	Dart res = NIL ;
	for(std::vector<Dart>::iterator it = fstore.begin() ; it != fstore.end() ; ++it)
	{
		Dart fit = *it ;
		Dart end = this->phi_1(fit) ;
		fit = this->phi1(fit) ;
		while(fit != end)
		{
			Dart d2 = this->phi2(fit) ;
			Dart d3 = phi3(fit) ;
			Dart d32 = this->phi2(d3) ;

			if(res == NIL)
				res = d2 ;

			this->beta2unsew(d2) ;
			this->beta2unsew(this->beta0(d2)) ;

			this->beta2unsew(d32) ;
			this->beta2unsew(this->beta0(d32)) ;

			this->beta2sew(d2, this->beta0(d32)) ;
			this->beta2sew(this->beta0(d2), d32) ;
			this->beta2sew(fit, this->beta0(d3)) ;
			this->beta2sew(this->beta0(fit), d3) ;

			fit = this->phi1(fit) ;
		}
	}

	ParentMap::deleteCC(d) ;

	return res ;
}

template <typename MAP_IMPL>
Dart GMap3<MAP_IMPL>::cutEdge(Dart d)
{
	Dart prev = d ;
	Dart dd = alpha2(d) ;
	Dart nd = ParentMap::cutEdge(d) ;

	while(dd != d)
	{
		prev = dd ;
		dd = alpha2(dd) ;

		ParentMap::cutEdge(prev) ;

		Dart d3 = beta3(prev);
		beta3sew(this->beta0(prev), this->beta0(d3));
		beta3sew(this->phi1(prev), this->phi1(d3));
	}

	Dart d3 = beta3(d);
	beta3sew(this->beta0(d), this->beta0(d3));
	beta3sew(this->phi1(d), this->phi1(d3));

	return nd ;
}

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::uncutEdge(Dart d)
{
	if(vertexDegree(this->phi1(d)) == 2)
	{
		Dart prev = d ;

		Dart dd = d;
		do
		{
			prev = dd;
			dd = alpha2(dd);

			ParentMap::uncutEdge(prev);
		} while (dd != d) ;

		return true;
	}
	return false;
}

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::deleteEdgePreCond(Dart d)
{
	unsigned int nb1 = vertexDegree(d);
	unsigned int nb2 = vertexDegree(this->phi1(d));
	return (nb1!=2) && (nb2!=2);
}

template <typename MAP_IMPL>
Dart GMap3<MAP_IMPL>::deleteEdge(Dart d)
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
		fit = this->phi1(fit) ;
		while(fit != end)
		{
			Dart d2 = this->phi2(fit) ;
			Dart d3 = phi3(fit) ;
			Dart d32 = this->phi2(d3) ;

			if(res == NIL)
				res = d2 ;

			this->beta2unsew(d2) ;
			this->beta2unsew(this->beta0(d2)) ;
			this->beta2unsew(d32) ;
			this->beta2unsew(this->beta0(d32)) ;

			this->beta2sew(d2, this->beta0(d32)) ;
			this->beta2sew(this->beta0(d2), d32) ;
			this->beta2sew(fit, this->beta0(d3)) ;
			this->beta2sew(this->beta0(fit), d3) ;

			fit = this->phi1(fit) ;
		}
		dit = alpha2(dit) ;
	} while(dit != d) ;

	ParentMap::deleteCC(d) ;

	return res ;
}

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::splitFacePreCond(Dart d, Dart e)
{
	return (d != e && ParentMap::sameFace(d, e)) ;
}

template <typename MAP_IMPL>
void GMap3<MAP_IMPL>::splitFace(Dart d, Dart e)
{
	assert(splitFacePreCond(d, e));

	if(!sameOrientedFace(d, e))
		e = this->beta1(e) ;

	Dart dd = this->beta1(beta3(d));
	Dart ee = this->beta1(beta3(e));

	Dart dprev = this->phi_1(d) ;
	Dart eprev = this->phi_1(e) ;
	Dart ddprev = this->phi_1(dd) ;
	Dart eeprev = this->phi_1(ee) ;

	beta3unsew(this->beta1(d)) ;
	beta3unsew(this->beta1(e)) ;
	beta3unsew(this->beta1(dd)) ;
	beta3unsew(this->beta1(ee)) ;

	ParentMap::splitFace(d, e);
	ParentMap::splitFace(dd, ee);
	beta3sew(this->beta1(d), this->phi_1(ee));
	beta3sew(this->phi_1(d), this->beta1(ee));
	beta3sew(this->beta1(e), this->phi_1(dd));
	beta3sew(this->phi_1(e), this->beta1(dd));

	beta3sew(this->beta0(dprev), this->beta0(beta3(dprev))) ;
	beta3sew(this->beta0(eprev), this->beta0(beta3(eprev))) ;
	beta3sew(this->beta0(ddprev), this->beta0(beta3(ddprev))) ;
	beta3sew(this->beta0(eeprev), this->beta0(beta3(eeprev))) ;
}

template <typename MAP_IMPL>
void GMap3<MAP_IMPL>::sewVolumes(Dart d, Dart e, bool withBoundary)
{
	assert(this->faceDegree(d) == this->faceDegree(e));

	// if sewing with fixed points
	if (!withBoundary)
	{
		assert(beta3(d) == d && beta3(e) == e) ;
		Dart fitD = d ;
		Dart fitE = e ;
		do
		{
			beta3sew(fitD, this->beta0(fitE)) ;
			beta3sew(this->beta0(fitD), fitE) ;
			fitD = this->phi1(fitD) ;
			fitE = this->phi_1(fitE) ;
		} while(fitD != d) ;
		return ;
	}

	Dart dd = beta3(d) ;
	Dart ee = beta3(e) ;

	Dart fitD = dd ;
	Dart fitE = ee ;
	do
	{
		Dart fitD2 = this->beta2(fitD) ;
		Dart fitE2 = this->beta2(fitE) ;
		if(fitD2 != fitE)
		{
			this->beta2unsew(fitD) ;
			this->beta2unsew(fitE) ;
			this->beta2unsew(this->beta0(fitD)) ;
			this->beta2unsew(this->beta0(fitE)) ;
			this->beta2sew(fitD2, this->beta0(fitE2)) ;
			this->beta2sew(this->beta0(fitD2), fitE2) ;
			this->beta2sew(fitD, this->beta0(fitE)) ;
			this->beta2sew(this->beta0(fitD), fitE) ;
		}
		beta3unsew(fitD) ;
		beta3unsew(this->beta0(fitD)) ;
		beta3unsew(fitE) ;
		beta3unsew(this->beta0(fitE)) ;
		fitD = this->phi1(fitD) ;
		fitE = this->phi_1(fitE) ;
	} while(fitD != dd) ;
	ParentMap::deleteCC(dd) ;

	fitD = d ;
	fitE = e ;
	do
	{
		beta3sew(fitD, this->beta0(fitE)) ;
		beta3sew(this->beta0(fitD), fitE) ;
		fitD = this->phi1(fitD) ;
		fitE = this->phi_1(fitE) ;
	} while(fitD != d) ;
}

template <typename MAP_IMPL>
void GMap3<MAP_IMPL>::unsewVolumes(Dart d)
{
	assert(!isBoundaryFace(d)) ;

	unsigned int nbE = this->faceDegree(d) ;
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
			Dart f2 = this->phi2(f) ;
			this->beta2unsew(f) ;
			this->beta2unsew(this->beta0(f)) ;
			this->beta2sew(fitB1, this->beta0(f)) ;
			this->beta2sew(this->beta0(fitB1), f) ;
			this->beta2sew(fitB2, this->beta0(f2)) ;
			this->beta2sew(this->beta0(fitB2), f2) ;
		}
		else
		{
			this->beta2sew(fitB1, this->beta0(fitB2)) ;
			this->beta2sew(this->beta0(fitB1), fitB2) ;
		}

		beta3unsew(fit1) ;
		beta3unsew(this->beta0(fit1)) ;
		beta3sew(fit1, this->beta0(fitB1)) ;
		beta3sew(this->beta0(fit1), fitB1) ;
		beta3sew(fit2, this->beta0(fitB2)) ;
		beta3sew(this->beta0(fit2), fitB2) ;

		fit1 = this->phi1(fit1) ;
		fit2 = this->phi_1(fit2) ;
		fitB1 = this->phi_1(fitB1) ;
		fitB2 = this->phi1(fitB2) ;
	} while(fitB1 != b1) ;
}

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::mergeVolumes(Dart d)
{
	if(!GMap3::isBoundaryFace(d))
	{
		ParentMap::mergeVolumes(d, phi3(d)); // merge the two volumes along common face
		return true ;
	}
	return false ;
}

template <typename MAP_IMPL>
void GMap3<MAP_IMPL>::splitVolume(std::vector<Dart>& vd)
{
	assert(ParentMap::checkSimpleOrientedPath(vd)) ;

	Dart e = vd.front();
	Dart e2 = this->phi2(e);

	ParentMap::splitSurface(vd,true,true);

	//sew the two connected components
	sewVolumes(this->phi2(e), this->phi2(e2), false);

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

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::sameOrientedVertex(Dart d, Dart e) const
{
	DartMarkerStore<GMap3<MAP_IMPL> > mv(*this);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		if(darts[i] == e)
			return true;

		// add phi21 and phi23 successor if they are not marked yet
		Dart d2 = this->phi2(darts[i]);
		Dart d21 = this->phi1(d2);	// turn in volume
		Dart d23 = phi3(d2);		// change volume

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

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::sameVertex(Dart d, Dart e) const
{
	DartMarkerStore<GMap3<MAP_IMPL> > mv(*this);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		if(darts[i] == e)
			return true;

		Dart dx = this->beta1(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = this->beta2(darts[i]);
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

//template <typename MAP_IMPL>
//unsigned int GMap3<MAP_IMPL>::vertexDegree(Dart d)
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

template <typename MAP_IMPL>
unsigned int GMap3<MAP_IMPL>::vertexDegree(Dart d) const
{
	unsigned int count = 0;

	Traversor3VE<GMap3<MAP_IMPL> > trav3VE(*this, d);
	for(Dart dit = trav3VE.begin() ; dit != trav3VE.end() ; dit = trav3VE.next())
	{
		++count;
	}

	return count;
}


template <typename MAP_IMPL>
int GMap3<MAP_IMPL>::checkVertexDegree(Dart d, unsigned int vd) const
{
	unsigned int count = 0;

	Traversor3VE<GMap3<MAP_IMPL> > trav3VE(*this, d);
	Dart dit = trav3VE.begin();
	for( ; (count<=vd) && (dit != trav3VE.end()) ; dit = trav3VE.next())
	{
		++count;
	}

	return count - vd;
}

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::isBoundaryVertex(Dart d) const
{
	DartMarkerStore<GMap3<MAP_IMPL> > mv(*this);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		if(this->template isBoundaryMarked<3>(darts[i]))
			return true ;

		//add phi21 and phi23 successor if they are not marked yet
		Dart d2 = this->phi2(darts[i]);
		Dart d21 = this->phi1(d2);	// turn in volume
		Dart d23 = phi3(d2);		// change volume

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

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::sameOrientedEdge(Dart d, Dart e) const
{
	Dart it = d;
	do
	{
		if(it == e || this->phi2(it) == e)
			return true;
		it = alpha2(it);
	} while (it != d);
	return false;
}

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::sameEdge(Dart d, Dart e) const
{
	Dart it = d;
	do
	{
		if(it == e || this->beta0(it) == e || this->beta2(it) == e || this->phi2(it) == e)
			return true;

		it = alpha2(it);
	} while (it != d);
	return false;
}

template <typename MAP_IMPL>
unsigned int GMap3<MAP_IMPL>::edgeDegree(Dart d) const
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

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::isBoundaryEdge(Dart d) const
{
	Dart it = d;
	do
	{
		if(this->template isBoundaryMarked<3>(it))
			return true ;
		it = alpha2(it);
	} while(it != d);
	return false;
}

template <typename MAP_IMPL>
Dart GMap3<MAP_IMPL>::findBoundaryFaceOfEdge(Dart d) const
{
	Dart it = d;
	do
	{
		if (this->template isBoundaryMarked<3>(it))
			return it ;
		it = alpha2(it);
	} while(it != d);
	return NIL ;
}

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::sameOrientedFace(Dart d, Dart e) const
{
	Dart it = d;
	do
	{
		if(it == e || phi3(it) == e)
			return true;
		it = this->phi1(it);
	} while (it != d);
	return false;
}

template <typename MAP_IMPL>
inline bool GMap3<MAP_IMPL>::sameFace(Dart d, Dart e) const
{
	return ParentMap::sameFace(d, e) || ParentMap::sameFace(beta3(d), e) ;
}

template <typename MAP_IMPL>
inline bool GMap3<MAP_IMPL>::isBoundaryFace(Dart d) const
{
	return this->template isBoundaryMarked<3>(d) || this->template isBoundaryMarked<3>(beta3(d));
}

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::isBoundaryAdjacentVolume(Dart d) const
{
	DartMarkerStore<GMap3<MAP_IMPL> > mark(*this);	// Lock a marker

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
			Dart ee = this->phi2(e) ;
			if(!mark.isMarked(ee)) // not already marked
			{
				visitedFaces.push_back(ee) ;
				mark.markOrbit<FACE>(ee) ;
			}
			e = this->phi1(e) ;
		} while(e != visitedFaces[i]) ;
	}
	return false;
}

template <typename MAP_IMPL>
bool GMap3<MAP_IMPL>::check() const
{
	CGoGNout << "Check: topology begin" << CGoGNendl;
	DartMarker<GMap3<MAP_IMPL> > m(*this);
	m.unmarkAll();
	for(Dart d = this->begin(); d != this->end(); this->next(d))
	{
		Dart d0 = this->beta0(d);
		if (this->beta0(d0) != d) // beta0 involution ?
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
			if(this->beta1(d3) != beta3(this->beta1(d)))
			{
				CGoGNout << "Check: beta3 , faces are not entirely sewn" << CGoGNendl;
				return false;
			}
		}

		Dart d2 = this->beta2(d);
		if (this->beta2(d2) != d) // beta2 involution ?
		{
			CGoGNout << "Check: beta2 is not an involution" << CGoGNendl;
			return false;
		}

		Dart d1 = this->phi1(d);
		if (this->phi_1(d1) != d) // phi1 a une image correcte ?
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

		if (this->phi1(d1) == d)
			CGoGNout << "Check: (warning) face with only two edges" << CGoGNendl;

		if (this->phi2(d1) == d)
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

template <typename MAP_IMPL>
template <unsigned int ORBIT, typename FUNC>
void GMap3<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f, unsigned int thread) const
{
	switch(ORBIT)
	{
		case DART:		f(c); break;
		case VERTEX: 	foreach_dart_of_vertex(c, f, thread); break;
		case EDGE: 		foreach_dart_of_edge(c, f, thread); break;
		case FACE: 		foreach_dart_of_face(c, f, thread); break;
		case VOLUME: 	foreach_dart_of_volume(c, f, thread); break;
		case VERTEX1: 	foreach_dart_of_vertex1(c, f, thread); break;
		case EDGE1: 	foreach_dart_of_edge1(c, f, thread); break;
		case VERTEX2: 	foreach_dart_of_vertex2(c, f, thread); break;
		case EDGE2:		foreach_dart_of_edge2(c, f, thread); break;
		case FACE2:		foreach_dart_of_face2(c, f, thread); break;
		default: 		assert(!"Cells of this dimension are not handled"); break;
	}
}

//template <typename MAP_IMPL>
//template <unsigned int ORBIT, typename FUNC>
//void GMap3<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC& f, unsigned int thread) const
//{
//	switch(ORBIT)
//	{
//		case DART:		f(c); break;
//		case VERTEX: 	foreach_dart_of_vertex(c, f, thread); break;
//		case EDGE: 		foreach_dart_of_edge(c, f, thread); break;
//		case FACE: 		foreach_dart_of_face(c, f, thread); break;
//		case VOLUME: 	foreach_dart_of_volume(c, f, thread); break;
//		case VERTEX1: 	foreach_dart_of_vertex1(c, f, thread); break;
//		case EDGE1: 	foreach_dart_of_edge1(c, f, thread); break;
//		case VERTEX2: 	foreach_dart_of_vertex2(c, f, thread); break;
//		case EDGE2:		foreach_dart_of_edge2(c, f, thread); break;
//		case FACE2:		foreach_dart_of_face2(c, f, thread); break;
//		default: 		assert(!"Cells of this dimension are not handled"); break;
//	}
//}

template <typename MAP_IMPL>
template <typename FUNC>
void GMap3<MAP_IMPL>::foreach_dart_of_oriented_vertex(Dart d, FUNC& f, unsigned int thread) const
{
	DartMarkerStore<GMap3<MAP_IMPL> > mv(*this, thread);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		// add phi21 and phi23 successor if they are not marked yet
		Dart d2 = this->phi2(darts[i]);
		Dart d21 = this->phi1(d2); // turn in volume
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

		f(darts[i]);
	}
}

template <typename MAP_IMPL>
template <typename FUNC>
void GMap3<MAP_IMPL>::foreach_dart_of_vertex(Dart d, FUNC& f, unsigned int thread) const
{
	DartMarkerStore<GMap3<MAP_IMPL> > mv(*this, thread);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		Dart dx = this->beta1(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = this->beta2(darts[i]);
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

		f(darts[i]);
	}
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_oriented_edge(Dart d, FUNC& f, unsigned int thread) const
{
	Dart it = d;
	do
	{
		ParentMap::foreach_dart_of_oriented_edge(it, f, thread);
		it = alpha2(it);
	} while (it != d);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_edge(Dart d, FUNC& f, unsigned int thread) const
{
	Dart it = d;
	do
	{
		ParentMap::foreach_dart_of_edge(it, f, thread);
		it = alpha2(it);
	} while (it != d);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_face(Dart d, FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_face(d, f, thread);
	ParentMap::foreach_dart_of_face(beta3(d), f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_volume(Dart d, FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_cc(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_oriented_volume(Dart d, FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_oriented_cc(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_vertex1(Dart d, FUNC& f, unsigned int thread) const
{
	ParentMap::ParentMap::foreach_dart_of_vertex(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_edge1(Dart d, FUNC& f, unsigned int thread) const
{
	ParentMap::ParentMap::foreach_dart_of_edge(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_vertex2(Dart d, FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_vertex(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_edge2(Dart d, FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_edge(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void GMap3<MAP_IMPL>::foreach_dart_of_face2(Dart d, FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_face(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
void GMap3<MAP_IMPL>::foreach_dart_of_cc(Dart d, FUNC& f, unsigned int thread) const
{
	DartMarkerStore<GMap3<MAP_IMPL> > mv(*this, thread);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(1024);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		Dart dx = this->beta0(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = this->beta1(darts[i]);
		if (!mv.isMarked(dx))
		{
			darts.push_back(dx);
			mv.mark(dx);
		}
		dx = this->beta2(darts[i]);
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

		f(darts[i]);
	}
}

/*! @name Close map after import or creation
 *  These functions must be used with care, generally only by import/creation algorithms
 *************************************************************************/

template <typename MAP_IMPL>
Dart GMap3<MAP_IMPL>::newBoundaryCycle(unsigned int nbE)
{
	Dart d = GMap1<MAP_IMPL>::newCycle(nbE);
	Algo::Topo::boundaryMarkOrbit<3,FACE>(*this, d);
	return d;
}

template <typename MAP_IMPL>
unsigned int GMap3<MAP_IMPL>::closeHole(Dart d, bool forboundary)
{
	assert(beta3(d) == d);		// Nothing to close
	DartMarkerStore<GMap3<MAP_IMPL> > m(*this) ;

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(d);		// Start with the face of d
	m.template markOrbit<FACE>(d) ;

	unsigned int count = 0 ;

	// For every face added to the list
	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		Dart f = visitedFaces[i] ;
		unsigned int degree = this->faceDegree(f) ;
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
						m.template markOrbit<FACE>(e) ;
					}
				}
				else if(this->template isBoundaryMarked<3>(e))
				{
					found = true ;
					this->beta2sew(e, bit) ;
					this->beta2sew(this->beta0(e), this->beta0(bit)) ;
				}
				else
					e = alpha2(e) ;
			} while(!found) ;

			beta3sew(f, bit) ;
			beta3sew(this->beta0(f), this->beta0(bit)) ;
			bit = this->phi1(bit) ;
			f = this->phi1(f);
		} while(f != visitedFaces[i]);
	}

	return count ;
}

template <typename MAP_IMPL>
unsigned int GMap3<MAP_IMPL>::closeMap()
{
	// Search the map for topological holes (fix points of beta3)
	unsigned int nb = 0 ;
	for (Dart d = this->begin(); d != this->end(); this->next(d))
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

template <typename MAP_IMPL>
void GMap3<MAP_IMPL>::computeDual()
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
