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
inline void Map3<MAP_IMPL>::init()
{
	MAP_IMPL::addInvolution() ;
}

template <typename MAP_IMPL>
inline Map3<MAP_IMPL>::Map3() : Map2<MAP_IMPL>()
{
	init() ;
}

template <typename MAP_IMPL>
inline std::string Map3<MAP_IMPL>::mapTypeName() const
{
	return "Map3";
}

template <typename MAP_IMPL>
inline unsigned int Map3<MAP_IMPL>::dimension() const
{
	return 3;
}

template <typename MAP_IMPL>
inline void Map3<MAP_IMPL>::clear(bool removeAttrib)
{
	ParentMap::clear(removeAttrib) ;
	if (removeAttrib)
		init() ;
}

template <typename MAP_IMPL>
inline unsigned int Map3<MAP_IMPL>::getNbInvolutions() const
{
	return 1 + ParentMap::getNbInvolutions();
}

template <typename MAP_IMPL>
inline unsigned int Map3<MAP_IMPL>::getNbPermutations() const
{
	return ParentMap::getNbPermutations();
}

/*! @name Basic Topological Operators
 * Access and Modification
 *************************************************************************/

template <typename MAP_IMPL>
inline Dart Map3<MAP_IMPL>::phi3(Dart d) const
{
	return MAP_IMPL::template getInvolution<1>(d);
}

template <typename MAP_IMPL>
template <int N>
inline Dart Map3<MAP_IMPL>::phi(Dart d) const
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
inline Dart Map3<MAP_IMPL>::alpha0(Dart d) const
{
	return phi3(d) ;
}

template <typename MAP_IMPL>
inline Dart Map3<MAP_IMPL>::alpha1(Dart d) const
{
	return phi3(this->phi_1(d)) ;
}

template <typename MAP_IMPL>
inline Dart Map3<MAP_IMPL>::alpha2(Dart d) const
{
	return phi3(this->phi2(d));
}

template <typename MAP_IMPL>
inline Dart Map3<MAP_IMPL>::alpha_2(Dart d) const
{
	return this->phi2(phi3(d));
}

template <typename MAP_IMPL>
inline void Map3<MAP_IMPL>::phi3sew(Dart d, Dart e)
{
	MAP_IMPL::template involutionSew<1>(d,e);
}

template <typename MAP_IMPL>
inline void Map3<MAP_IMPL>::phi3unsew(Dart d)
{
	MAP_IMPL::template involutionUnsew<1>(d);
}

/*! @name Generator and Deletor
 *  To generate or delete volumes in a 3-map
 *************************************************************************/

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::deleteVolume(Dart d, bool withBoundary)
{
	if(withBoundary)
	{
		DartMarkerStore< Map3<MAP_IMPL> > mark(*this);		// Lock a marker

		std::vector<Dart> visitedFaces;		// Faces that are traversed
		visitedFaces.reserve(512);
		visitedFaces.push_back(d);			// Start with the face of d

		mark.template markOrbit<FACE2>(d) ;

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
					mark.template markOrbit<FACE2>(ee) ;
				}
				e = this->phi1(e) ;
			} while(e != visitedFaces[i]) ;
		}

		Dart dd = phi3(d) ;
		ParentMap::deleteCC(d) ; //deleting the volume
		ParentMap::deleteCC(dd) ; //deleting its border (created from the unsew operation)

		return;
	}

	//else remove the CC and create fixed points
	DartMarkerStore< Map3<MAP_IMPL> > mark(*this);		// Lock a marker

	std::vector<Dart> visitedFaces;		// Faces that are traversed
	visitedFaces.reserve(512);
	visitedFaces.push_back(d);			// Start with the face of d

	mark.template markOrbit<FACE2>(d) ;

	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		Dart e = visitedFaces[i] ;

		Dart it = e ;
		do
		{
			phi3unsew(it);
			it = this->phi1(it) ;
		} while(it != e) ;

		do	// add all face neighbours to the table
		{
			Dart ee = this->phi2(e) ;
			if(!mark.isMarked(ee)) // not already marked
			{
				visitedFaces.push_back(ee) ;
				mark.template markOrbit<FACE2>(ee) ;
			}
			e = this->phi1(e) ;
		} while(e != visitedFaces[i]) ;
	}

	ParentMap::deleteCC(d) ; //deleting the volume
}

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::fillHole(Dart d)
{
	assert(isBoundaryFace(d)) ;
	Dart dd = d ;
	if(!this->template isBoundaryMarked<3>(dd))
		dd = phi3(dd) ;
	Algo::Topo::boundaryUnmarkOrbit<3,VOLUME>(*this, dd) ;
}

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::createHole(Dart d)
{
	assert(!isBoundaryFace(d)) ;
	Algo::Topo::boundaryMarkOrbit<3,VOLUME>(*this, d) ;
}

/*! @name Topological Operators
 *  Topological operations on 3-maps
 *************************************************************************/

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::splitVertex(std::vector<Dart>& vd)
{
	//assert(checkPathAroundVertex(vd)) ;

	//bool boundE = false;

	Dart prev = vd.front();	//elt 0

	Dart db1 = NIL;
	if(isBoundaryFace(this->phi2(prev)))
		db1 = this->phi2(phi3(this->phi1(this->phi2(prev))));

	Dart fs = this->phi_1(this->phi2(this->phi_1(prev)));	//first side

	ParentMap::splitVertex(prev, this->phi2(fs));

	for(unsigned int i = 1; i < vd.size(); ++i)
	{
		prev = vd[i];

		Dart fs = this->phi_1(this->phi2(this->phi_1(prev)));	//first side

		ParentMap::splitVertex(prev, this->phi2(fs));

		Dart d1 = this->phi_1(this->phi2(this->phi_1(vd[i-1])));
		Dart d2 = this->phi1(this->phi2(vd[i]));

		phi3sew(d1, d2);
	}

	Dart db2 = NIL;
	if(isBoundaryFace(this->phi2(this->phi_1(prev))))
		db2 = this->phi2(phi3(this->phi2(this->phi_1(prev))));

	if(db1 != NIL && db2 != NIL)
	{
		ParentMap::splitVertex(db1, db2);
		phi3sew(this->phi1(this->phi2(db2)), this->phi_1(phi3(this->phi2(db2))));
		phi3sew(this->phi1(this->phi2(db1)), this->phi_1(phi3(this->phi2(db1))));
	}
	else
	{
		Dart dbegin = this->phi1(this->phi2(vd.front()));
		Dart dend = this->phi_1(this->phi2(this->phi_1(vd.back())));
		phi3sew(dbegin, dend);
	}

	return this->phi_1(this->phi2(this->phi_1(prev)));
}

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::deleteVertex(Dart d)
{
	//if(isBoundaryVertex(d))
	//	return NIL ;

	// Save the darts around the vertex
	// (one dart per face should be enough)
	std::vector<Dart> fstoretmp;
	fstoretmp.reserve(128);
//	this->template foreach_dart_of_orbit<VERTEX>(d, [&] (Dart it) { fstoretmp.push_back(it); });
    this->template foreach_dart_of_orbit<VERTEX>(d, bl::bind(static_cast<void(std::vector<Dart>::*)(const Dart&)>(&std::vector<Dart>::push_back), boost::ref(fstoretmp), bl::_1));

	 // just one dart per face
	std::vector<Dart> fstore;
	fstore.reserve(128);
	DartMarker<Map3<MAP_IMPL> > mf(*this);
	for(unsigned int i = 0; i < fstoretmp.size(); ++i)
	{
		if(!mf.isMarked(fstoretmp[i]))
		{
			mf.template markOrbit<FACE>(fstoretmp[i]);
			fstore.push_back(fstoretmp[i]);
		}
	}

	std::cout << "nb faces " << fstore.size() << std::endl;

	Dart res = NIL ;
	for(std::vector<Dart>::iterator it = fstore.begin() ; it != fstore.end() ; ++it)
	{
		Dart fit = *it ;
		Dart end = this->phi_1(fit) ;
		fit = this->phi1(fit) ;

		if(fit == end)
		{
			std::cout << " mmmmmmmmmmmmmmmmmmmmmerrrrrrrrrrrrrrrrrde !!!!!!!!!!!! " << std::endl;

//			Dart d2 = phi2(fit) ;
//			Dart d23 = phi3(d2) ;
//			Dart d3 = phi3(fit) ;
//			Dart d32 = phi2(d3) ;
//
//			//phi3unsew()
//			phi3sew(d3,23);
//
//			fit = phi_1(fit);
//
//			d2 = phi2(fit) ;
//			d23 = phi3(d2) ;
//			d3 = phi3(fit) ;
//			d32 = phi2(d3) ;
//			phi3sew(d3,23);

//			Map2::deleteCC(fit);
		}
		else
		{
			while(fit != end)
			{
				Dart d2 = this->phi2(fit) ;
				Dart d3 = phi3(fit) ;
				Dart d32 = this->phi2(d3) ;

				if(res == NIL)
					res = d2 ;

				this->phi2unsew(d2) ;
				this->phi2unsew(d32) ;
				this->phi2sew(d2, d32) ;
				this->phi2sew(fit, d3) ;

				fit = this->phi1(fit) ;
			}
		}
	}

	ParentMap::deleteCC(d) ;

	return res ;
}

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::cutEdge(Dart d)
{
	Dart prev = d;
	Dart dd = this->alpha2(d);
	Dart nd = ParentMap::cutEdge(d);

	while (dd != d)
	{
		prev = dd;
		dd = alpha2(dd);

		ParentMap::cutEdge(prev);

		Dart d3 = phi3(prev);
		phi3unsew(prev);
		phi3sew(prev, this->phi1(d3));
		phi3sew(d3, this->phi1(prev));
	}

	Dart d3 = phi3(d);
	phi3unsew(d);
	phi3sew(d, this->phi1(d3));
	phi3sew(d3, this->phi1(d));

	return nd;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::uncutEdge(Dart d)
{
	if(vertexDegree(this->phi1(d)) == 2)
	{
		Dart prev = d ;
		phi3unsew(this->phi1(prev)) ;

		Dart dd = d;
		do
		{
			prev = dd;
			dd = this->alpha2(dd);

			phi3unsew(this->phi2(prev)) ;
			phi3unsew(this->phi2(this->phi1(prev))) ;
			ParentMap::uncutEdge(prev);
			phi3sew(dd, this->phi2(prev));
		} while (dd != d) ;

		return true;
	}
	return false;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::deleteEdgePreCond(Dart d)
{
	unsigned int nb1 = vertexDegree(d);
	unsigned int nb2 = vertexDegree(this->phi1(d));
	return (nb1!=2) && (nb2!=2);
}

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::deleteEdge(Dart d)
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

			this->phi2unsew(d2) ;
			this->phi2unsew(d32) ;
			this->phi2sew(d2, d32) ;
			this->phi2sew(fit, d3) ;

			fit = this->phi1(fit) ;
		}
		dit = this->alpha2(dit) ;
	} while(dit != d) ;

	ParentMap::deleteCC(d) ;

	return res ;
}

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::collapseEdge(Dart d, bool delDegenerateVolumes)
{
	Dart resV = NIL;
	Dart dit = d;

	std::vector<Dart> darts;
	do
	{
		darts.push_back(dit);
		dit = this->alpha2(dit);
	} while(dit != d);

	for (std::vector<Dart>::iterator it = darts.begin(); it != darts.end(); ++it)
	{
		Dart x = this->phi2(this->phi_1(*it));

		Dart resCV = NIL;

		if(!isBoundaryFace(this->phi2(this->phi1(*it))))
			resCV = phi3(this->phi2(this->phi1(*it)));
		else if(!isBoundaryFace(this->phi2(this->phi_1(*it))))
			resCV = phi3(this->phi2(this->phi_1(*it)));

		resV = ParentMap::collapseEdge(*it, true);
		if (delDegenerateVolumes)
			if(collapseDegeneretedVolume(x) && resCV != NIL)
				resV = resCV;
	}

	return resV;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::splitFacePreCond(Dart d, Dart e)
{
	return (d != e && this->sameOrientedFace(d, e)) ;
}

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::splitFace(Dart d, Dart e)
{
//	assert(d != e && sameOrientedFace(d, e)) ;
	assert(splitFacePreCond(d,e));

	Dart dd = this->phi1(phi3(d));
	Dart ee = this->phi1(phi3(e));

	ParentMap::splitFace(d, e);
	ParentMap::splitFace(dd, ee);

	phi3sew(this->phi_1(d), this->phi_1(ee));
	phi3sew(this->phi_1(e), this->phi_1(dd));
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::mergeFaces(Dart d)
{
	assert(edgeDegree(d)==2);

	Dart dd = phi3(d);

	phi3unsew(d);
	phi3unsew(dd);

	//use code of mergesFaces to override the if(isBoundaryEdge)
	//we have to merge the faces if the face is linked to a border also
//	Map2::mergeFaces(d);
	Dart e = this->phi2(d) ;
	this->phi2unsew(d) ;
	Map1<MAP_IMPL>::mergeCycles(d, this->phi1(e)) ;
	Map1<MAP_IMPL>::splitCycle(e, this->phi1(d)) ;
	Map1<MAP_IMPL>::deleteCycle(d) ;
//	ParentMap::mergeFaces(dd);
	e = this->phi2(dd) ;
	this->phi2unsew(dd) ;
	Map1<MAP_IMPL>::mergeCycles(dd, this->phi1(e)) ;
	Map1<MAP_IMPL>::splitCycle(e, this->phi1(dd)) ;
	Map1<MAP_IMPL>::deleteCycle(dd);

	return true;
}

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::collapseFace(Dart d, bool delDegenerateVolumes)
{
	Dart resV = NIL;
	Dart stop = this->phi_1(d);
	Dart dit = d;
	std::vector<Dart> vd;
	vd.reserve(32);

	do
	{
		vd.push_back(this->alpha2(dit));
		dit = this->phi1(dit);
	}
	while(dit != stop);

	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
		resV = Map3::collapseEdge(*it, delDegenerateVolumes);

	return resV;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::collapseDegeneretedVolume(Dart d)
{
	Dart e1 = d;
	Dart e2 = this->phi2(d);

	do
	{
		if (e1 != this->phi2(e2))
			return false;
		e1 = this->phi1(e1);
		e2 = this->phi_1(e2);
	}while (e1 != d);

	if (e2 != this->phi2(d))
		return false;

	// degenerated:
	do
	{
		Dart f1 = phi3(e1);
		Dart f2 = phi3(e2);
		phi3unsew(e1);
		phi3unsew(e2);
		phi3sew(f1,f2);
		e1 = this->phi1(e1);
		e2 = this->phi_1(e2);
	}while (e1 != d);

	ParentMap::deleteCC(d) ;
	return true;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::sewVolumesPreCond(Dart d, Dart e)
{
	return (this->faceDegree(d) == this->faceDegree(e));
}

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::sewVolumes(Dart d, Dart e, bool withBoundary)
{
	assert(sewVolumesPreCond(d,e));

	// if sewing with fixed points
	if (!withBoundary)
	{
		assert(phi3(d) == d && phi3(e) == e) ;
		Dart fitD = d ;
		Dart fitE = e ;
		do
		{
			phi3sew(fitD, fitE) ;
			fitD = this->phi1(fitD) ;
			fitE = this->phi_1(fitE) ;
		} while(fitD != d) ;
		return ;
	}

	Dart dd = phi3(d) ;
	Dart ee = phi3(e) ;

	Dart fitD = dd ;
	Dart fitE = ee ;
	do
	{
		Dart fitD2 = this->phi2(fitD) ;
		Dart fitE2 = this->phi2(fitE) ;
		if(fitD2 != fitE)
		{
			this->phi2unsew(fitD) ;
			this->phi2unsew(fitE) ;
			this->phi2sew(fitD2, fitE2) ;
			this->phi2sew(fitD, fitE) ;
		}
		phi3unsew(fitD) ;
		phi3unsew(fitE) ;
		fitD = this->phi1(fitD) ;
		fitE = this->phi_1(fitE) ;
	} while(fitD != dd) ;
	ParentMap::deleteCC(dd) ;

	fitD = d ;
	fitE = e ;
	do
	{
		phi3sew(fitD, fitE) ;
		fitD = this->phi1(fitD) ;
		fitE = this->phi_1(fitE) ;
	} while(fitD != d) ;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::unsewVolumesPreCond(Dart d)
{
	return (!this->isBoundaryFace(d)) ;
}

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::unsewVolumes(Dart d, bool withBoundary)
{
	assert(unsewVolumesPreCond(d)) ;

	if (!withBoundary)
	{
		Dart fitD = d ;
		do
		{
			phi3unsew(fitD) ;
			fitD = this->phi1(fitD) ;
		} while(fitD != d) ;
		return ;
	}

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
			this->phi2unsew(f) ;
			this->phi2sew(fitB1, f) ;
			this->phi2sew(fitB2, f2) ;
		}
		else
			this->phi2sew(fitB1, fitB2) ;

		phi3unsew(fit1) ;
		phi3sew(fit1, fitB1) ;
		phi3sew(fit2, fitB2) ;

		fit1 = this->phi1(fit1) ;
		fit2 = this->phi_1(fit2) ;
		fitB1 = this->phi_1(fitB1) ;
		fitB2 = this->phi1(fitB2) ;
	} while(fitB1 != b1) ;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::mergeVolumes(Dart d, bool deleteFace)
{
	if(!Map3<MAP_IMPL>::isBoundaryFace(d))
	{
		ParentMap::mergeVolumes(d, phi3(d), deleteFace); // merge the two volumes along common face
		return true ;
	}
	return false ;
}

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::splitVolume(std::vector<Dart>& vd)
{
	//assert(checkSimpleOrientedPath(vd)) ;

	Dart e = vd.front();
	Dart e2 = this->phi2(e);

	ParentMap::splitSurface(vd, true, true);

	//sew the two connected components
	Map3<MAP_IMPL>::sewVolumes(this->phi2(e), this->phi2(e2), false);
}

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::splitVolumeWithFace(std::vector<Dart>& vd, Dart d)
{
	assert(vd.size() == this->faceDegree(d));

	// deconnect edges around the path
	// sew the given face into the paths
	Dart dit = d;
	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		Dart it2 = this->phi2(*it);
		ParentMap::unsewFaces(*it, false) ;

		ParentMap::sewFaces(*it, dit, false);
		ParentMap::sewFaces(it2, phi3(dit), false);

		dit = this->phi_1(dit);
	}
}

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::collapseVolume(Dart d, bool delDegenerateVolumes)
{
	Dart resV = NIL;
	std::vector<Dart> vd;
	vd.reserve(32);

	vd.push_back(d);
	vd.push_back(this->alpha2(this->phi1(d)));
	vd.push_back(this->alpha2(this->phi_1(this->phi2(this->phi1(d)))));

//	Traversor3WF<Map3> tra(*this, phi1(d));
//	for(Dart dit = tra.begin() ; dit != tra.end() ; dit = tra.next())
//	{
//		vd.push_back(alpha2(dit));
//	}
//	vd.pop_back();

	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
		resV = Map3<MAP_IMPL>::collapseEdge(*it, delDegenerateVolumes);

	return resV;
}

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::faceToEdge(Dart d)
{
	Dart dc = this->phi2(this->phi1(d));
	Dart dc1 = this->phi_1(d);
	Dart dc2 = this->phi1(this->phi2(dc));

	ParentMap::unsewFaces(dc, false);
	ParentMap::unsewFaces(dc1, false);
	ParentMap::unsewFaces(dc2, false);

	unsewFaces(phi3(dc), false);
	unsewFaces(phi3(dc1), false);
	unsewFaces(phi3(dc2), false);

	return dc;
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::sameVertex(Dart d, Dart e) const
{
	DartMarkerStore< Map3<MAP_IMPL> > mv(*this);	// Lock a marker

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
	}
	return false;
}

template <typename MAP_IMPL>
unsigned int Map3<MAP_IMPL>::vertexDegree(Dart d) const
{
	unsigned int count = 0;

	Traversor3VE<Map3<MAP_IMPL> > trav3VE(*this, d);
	for(Dart dit = trav3VE.begin() ; dit != trav3VE.end() ; dit = trav3VE.next())
		++count;

	return count;
}

template <typename MAP_IMPL>
int Map3<MAP_IMPL>::checkVertexDegree(Dart d, unsigned int vd) const
{
	unsigned int count = 0;

	Traversor3VE<Map3<MAP_IMPL> > trav3VE(*this, d);
	Dart dit = trav3VE.begin();
	for( ; (count <= vd) && (dit != trav3VE.end()) ; dit = trav3VE.next())
		++count;

	return count - vd;
}

template <typename MAP_IMPL>
unsigned int Map3<MAP_IMPL>::vertexDegreeOnBoundary(Dart d) const
{
	assert(Map3::isBoundaryVertex(d));
	return ParentMap::vertexDegree(d);
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::isBoundaryVertex(Dart d) const
{
	DartMarkerStore< Map3<MAP_IMPL> > mv(*this);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
        if(this->isBoundaryMarked(3, darts[i]))
			return true ;

		//add phi21 and phi23 successor if they are not marked yet
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
	}
	return false ;
}

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::findBoundaryFaceOfVertex(Dart d) const
{
	DartMarkerStore< Map3<MAP_IMPL> > mv(*this);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
        if(this->isBoundaryMarked(3, darts[i]))
			return darts[i];

		//add phi21 and phi23 successor if they are not marked yet
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
	}
	return NIL ;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::sameOrientedEdge(Dart d, Dart e) const
{
	Dart it = d;
	do
	{
		if(it == e)
			return true;
		it = alpha2(it);
	} while (it != d);
	return false;
}

template <typename MAP_IMPL>
inline bool Map3<MAP_IMPL>::sameEdge(Dart d, Dart e) const
{
	return sameOrientedEdge(d, e) || sameOrientedEdge(this->phi2(d), e) ;
}

template <typename MAP_IMPL>
unsigned int Map3<MAP_IMPL>::edgeDegree(Dart d) const
{
	unsigned int deg = 0;
	Dart it = d;
	do
	{
		if(!this->template isBoundaryMarked<3>(it))
			++deg;
		it = alpha2(it);
	} while(it != d);
	return deg;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::isBoundaryEdge(Dart d) const
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
Dart Map3<MAP_IMPL>::findBoundaryFaceOfEdge(Dart d) const
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
inline bool Map3<MAP_IMPL>::sameFace(Dart d, Dart e) const
{
	return ParentMap::sameOrientedFace(d, e) || ParentMap::sameOrientedFace(phi3(d), e) ;
}

template <typename MAP_IMPL>
inline bool Map3<MAP_IMPL>::isBoundaryFace(Dart d) const
{
	return this->template isBoundaryMarked<3>(d) || this->template isBoundaryMarked<3>(phi3(d));
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::isVolumeIncidentToBoundary(Dart d) const
{
	Traversor3WF<Map3<MAP_IMPL> > tra(*this, d);
	for(Dart dit = tra.begin() ; dit != tra.end() ; dit = tra.next())
	{
		if(this->template isBoundaryMarked<3>(phi3(dit)))
			return true ;
	}
	return false;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::hasBoundaryEdge(Dart d) const
{
	Traversor3WE<Map3<MAP_IMPL> > tra(*this, d);
	for(Dart dit = tra.begin() ; dit != tra.end() ; dit = tra.next())
	{
		if(isBoundaryEdge(dit))
			return true;
	}

	return false;
}

template <typename MAP_IMPL>
bool Map3<MAP_IMPL>::check() const
{
	std::cout << "Check: topology begin" << std::endl;
	DartMarkerStore< Map3<MAP_IMPL> > m(*this);
	for(Dart d = Map3::begin(); d != Map3::end(); Map3::next(d))
	{
		Dart d3 = phi3(d);
		if (phi3(d3) != d) // phi3 involution ?
		{
			std::cout << "Check: phi3 is not an involution" << std::endl;
			return false;
		}

		if(this->phi1(d3) != phi3(this->phi_1(d)))
		{
			if(this->template isBoundaryMarked<3>(d))
				std::cout << "Boundary case - Check: phi3 , faces are not entirely sewn" << std::endl;
			else
				std::cout << "Check: phi3 , faces are not entirely sewn" << std::endl;
			std::cout << "face : " << this->phi1(d3) << " and face = " << phi3(this->phi_1(d)) << std::endl;
			return false;
		}

		Dart d2 = this->phi2(d);
		if (this->phi2(d2) != d) // phi2 involution ?
		{
			if(this->template isBoundaryMarked<3>(d))
				std::cout << "Boundary case - ";
			std::cout << "Check: phi2 is not an involution" << std::endl;
			return false;
		}

		Dart d1 = this->phi1(d);
		if (this->phi_1(d1) != d) // phi1 a une image correcte ?
		{
			if(this->template isBoundaryMarked<3>(d))
				std::cout << "Boundary case - ";
			std::cout << "Check: unconsistent phi_1 link" << std::endl;
			return false;
		}

		if (m.isMarked(d1)) // phi1 a un seul antécédent ?
		{
			if(this->template isBoundaryMarked<3>(d))
				std::cout << "Boundary case - ";
			std::cout << "Check: dart with two phi1 predecessors" << std::endl;
			return false;
		}
		m.mark(d1);

		if (d1 == d)
		{
			if(this->template isBoundaryMarked<3>(d))
				std::cout << "Boundary case - ";
			std::cout << "Check: (warning) face loop (one edge)" << std::endl;
		}

		if (this->phi1(d1) == d)
		{
			if(this->template isBoundaryMarked<3>(d))
				std::cout << "Boundary case - ";
			std::cout << "Check: (warning) face with only two edges" << std::endl;
		}

		if (this->phi2(d1) == d)
		{
			if(this->template isBoundaryMarked<3>(d))
				std::cout << "Boundary case - ";
			std::cout << "Check: (warning) dandling edge (phi2)" << std::endl;
		}

		if (phi3(d1) == d)
		{
			if(this->template isBoundaryMarked<3>(d))
				std::cout << "Boundary case - ";
			std::cout << "Check: (warning) dandling edge (phi3)" << std::endl;
		}
	}

	for(Dart d = this->begin(); d != this->end(); this->next(d))
	{
		if (!m.isMarked(d)) // phi1 a au moins un antécédent ?
		{
			if(this->template isBoundaryMarked<3>(d))
				std::cout << "Boundary case - ";
			std::cout << "Check: dart with no phi1 predecessor" << std::endl;
			return false;
		}
	}

	std::cout << "Check: topology ok" << std::endl;

	return true;
}

/*! @name Cell Functors
 *  Apply functors to all darts of a cell
 *************************************************************************/

//template <typename MAP_IMPL>
//template <unsigned int ORBIT, typename FUNC>
//void Map3<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f, unsigned int thread) const
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
template <unsigned int ORBIT, typename FUNC>
void Map3<MAP_IMPL>::foreach_dart_of_orbit(Cell<ORBIT> c, const FUNC& f, unsigned int thread) const
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

template <typename MAP_IMPL>
template <typename FUNC>
void Map3<MAP_IMPL>::foreach_dart_of_vertex(Dart d, const FUNC& f, unsigned int thread) const
{
	DartMarkerStore< Map3<MAP_IMPL> > mv(*this, thread);	// Lock a marker

    std::vector<Dart>& darts = *(this->askDartBuffer(thread));	// Darts that are traversed
//	darts.reserve(256);
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
    this->releaseDartBuffer(&darts, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map3<MAP_IMPL>::foreach_dart_of_edge(Dart d,const  FUNC& f, unsigned int thread) const
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
inline void Map3<MAP_IMPL>::foreach_dart_of_face(Dart d, const FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_face(d, f, thread);
	ParentMap::foreach_dart_of_face(phi3(d), f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map3<MAP_IMPL>::foreach_dart_of_volume(Dart d, const FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_cc(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map3<MAP_IMPL>::foreach_dart_of_vertex1(Dart d, const FUNC& f, unsigned int thread) const
{
	ParentMap::ParentMap::foreach_dart_of_vertex(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map3<MAP_IMPL>::foreach_dart_of_edge1(Dart d, const FUNC& f, unsigned int thread) const
{
	ParentMap::ParentMap::foreach_dart_of_edge(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map3<MAP_IMPL>::foreach_dart_of_vertex2(Dart d, const FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_vertex(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map3<MAP_IMPL>::foreach_dart_of_edge2(Dart d, const FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_edge(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
inline void Map3<MAP_IMPL>::foreach_dart_of_face2(Dart d, const FUNC& f, unsigned int thread) const
{
	ParentMap::foreach_dart_of_face(d, f, thread);
}

template <typename MAP_IMPL>
template <typename FUNC>
void Map3<MAP_IMPL>::foreach_dart_of_cc(Dart d, const FUNC& f, unsigned int thread) const
{
	DartMarkerStore< Map3<MAP_IMPL> > mv(*this,thread);	// Lock a marker

    std::vector<Dart>& darts = *(this->askDartBuffer(thread));	// Darts that are traversed
	darts.reserve(1024);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		// add all successors if they are not marked yet
		Dart d2 = this->phi1(darts[i]); // turn in face
		Dart d3 = this->phi2(darts[i]); // change face
		Dart d4 = phi3(darts[i]); // change volume

		if (!mv.isMarked(d2))
		{
			darts.push_back(d2);
			mv.mark(d2);
		}
		if (!mv.isMarked(d3))
		{
			darts.push_back(d2);
			mv.mark(d2);
		}
		if (!mv.isMarked(d4))
		{
			darts.push_back(d4);
			mv.mark(d4);
		}

		f(darts[i]);
	}
    this->releaseDartBuffer(&darts, thread);
}

/*! @name Close map after import or creation
 *  These functions must be used with care, generally only by import/creation algorithms
 *************************************************************************/

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::newBoundaryCycle(unsigned int nbE)
{
	Dart d = Map1<MAP_IMPL>::newCycle(nbE);
	Algo::Topo::boundaryMarkOrbit<3,FACE>(*this, d);
	return d;
}

template <typename MAP_IMPL>
unsigned int Map3<MAP_IMPL>::closeHole(Dart d, bool forboundary)
{
	assert(phi3(d) == d);		// Nothing to close
	DartMarkerStore< Map3<MAP_IMPL> > m(*this) ;

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(d);		// Start with the face of d
	m.template markOrbit<FACE2>(d) ;

	unsigned int count = 0 ;

	// For every face added to the list
	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		Dart it = visitedFaces[i] ;
		Dart f = it ;

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
				if(phi3(e) == e)
				{
					found = true ;
					if(!m.isMarked(e))
					{
						visitedFaces.push_back(e) ;
						m.template markOrbit<FACE2>(e) ;
					}
				}
				else if(this->template isBoundaryMarked<3>(e))
				{
					found = true ;
					this->phi2sew(e, bit) ;
				}
				else
					e = alpha2(e) ;
			} while(!found) ;

			phi3sew(f, bit) ;
			bit = this->phi_1(bit) ;
			f = this->phi1(f);
		} while(f != it) ;
	}

	return count ;
}

template <typename MAP_IMPL>
unsigned int Map3<MAP_IMPL>::closeMap()
{
	// Search the map for topological holes (fix points of phi3)
	unsigned int nb = 0 ;
	for (Dart d = this->begin(); d != this->end(); this->next(d))
	{
		if (phi3(d) == d)
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
void Map3<MAP_IMPL>::reverseOrientation()
{

}

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::computeDual()
{
	DartAttribute<Dart, MAP_IMPL> old_phi1 = this->template getAttribute<Dart, DART>("phi1") ;
	DartAttribute<Dart, MAP_IMPL> old_phi_1 = this->template getAttribute<Dart, DART>("phi_1") ;
	DartAttribute<Dart, MAP_IMPL> new_phi1 = this->template addAttribute<Dart, DART>("new_phi1") ;
	DartAttribute<Dart, MAP_IMPL> new_phi_1 = this->template addAttribute<Dart, DART>("new_phi_1") ;

	DartAttribute<Dart, MAP_IMPL> old_phi2 = this->template getAttribute<Dart, DART>("phi2") ;
	DartAttribute<Dart, MAP_IMPL> new_phi2 = this->template addAttribute<Dart, DART>("new_phi2") ;

	for(Dart d = this->begin(); d != this->end(); this->next(d))
	{
		Dart dd = this->phi2(phi3(d)) ;
		new_phi1[d] = dd ;
		new_phi_1[dd] = d ;

		Dart ddd = this->phi1(phi3(d));
		new_phi2[d] = ddd;
		new_phi2[ddd] = d;
	}

	this->template swapAttributes<Dart>(old_phi1, new_phi1) ;
	this->template swapAttributes<Dart>(old_phi_1, new_phi_1) ;
	this->template swapAttributes<Dart>(old_phi2, new_phi2) ;

	this->removeAttribute(new_phi1) ;
	this->removeAttribute(new_phi_1) ;
	this->removeAttribute(new_phi2) ;

	this->swapEmbeddingContainers(VERTEX, VOLUME) ;

//	unsigned int count = 0;

//	std::vector<Dart> vbound;

//	//std::cout << "nb faces : " << closeMap() << std::endl;

//	for(Dart d = begin(); d != end(); next(d))
//	{
//		if(isBoundaryMarked3(d) && !isBoundaryMarked3(phi3(d)))
//		{
//			vbound.push_back(d);
//		}
//	}
//
//	std::cout << "vbound size = " << vbound.size() << std::endl;
//
//	for(std::vector<Dart>::iterator it = vbound.begin() ; it != vbound.end() ; ++it)
//	{
//		Dart d = *it;
//		//Dart d3 = phi3(d);
//		phi3unsew(d);
//		//phi3unsew(d3);
//	}
//
//	//std::cout << "nb faces : " << closeMap() << std::endl;
//
//			if(d == 14208)
//			{
//				std::cout << "yeahhhhhhhh" << std::endl;
//				std::cout << "isBoundaryMarked ? " << isBoundaryMarked3(phi3(phi2(14208))) << std::endl;
//
//			}
//
//			//boundaryUnmark<3>(d);
//
//		}
//			if(d == 1569)
//			{
//				std::cout << "d " << std::endl;
//
//				Traversor3WE<Map3> t(*this,d);
//				for(Dart dit = t.begin() ; dit != t.end() ; dit = t.next())
//				{
//					Dart temp = dit;
//					do
//					{
//						if(isBoundaryMarked3(d))
//							std::cout << "d boundary " << std::endl;
//
//						temp = alpha2(temp);
//					}while(temp != dit);
//				}
//
//				if(isBoundaryMarked3(d))
//					std::cout << "d boundary " << std::endl;
//
//				if(isBoundaryMarked3(phi1(d)))
//					std::cout << "phi1(d) boundary " << std::endl;
//
//				if(isBoundaryMarked3(phi_1(d)))
//					std::cout << "phi_1(d) boundary " << std::endl;
//
//				if(isBoundaryMarked3(phi2(d)))
//					std::cout << "phi2(d) boundary " << std::endl;
//
//				if(isBoundaryMarked3(phi3(d)))
//					std::cout << "phi3(d) boundary " << std::endl;
//
//				if(isBoundaryMarked3(phi2(phi3(d))))
//					std::cout << "phi2(phi3(d)) boundary " << std::endl;
//
//				if(isBoundaryMarked3(phi3(phi2(d))))
//					std::cout << "phi3(phi2(d)) boundary " << std::endl;
//
//				if(isBoundaryMarked3(phi1(phi3(d))))
//					std::cout << "phi1(phi3(d)) boundary " << std::endl;
//
//				if(isBoundaryMarked3(phi3(phi1(d))))
//					std::cout << "phi3(phi1(d)) boundary " << std::endl;
//			}
//
//			if(isBoundaryMarked3(d))
//			{
//
//			if(isBoundaryMarked3(d))
//				std::cout << "d = " << d << std::endl;
//
//			if(isBoundaryMarked3(phi3(d)))
//				std::cout << "phi3(d) = " << phi3(d) << std::endl;
//
//			if(isBoundaryMarked3(phi2(d)))
//				std::cout << "phi2(d) = " << phi2(d) << std::endl;
//
//			Dart dit = deleteVertex(phi3(d));
//
//			if(dit == NIL)
//				std::cout << "NILLLLLLLLLLLLLLL" << std::endl;
//
//			++count;
//
//			if(count == 5)
//				return;
//		}
//	}


//	TraversorW<Map3> tW(*this);
//	for(Dart d = tW.begin(); d != tW.end(); d = tW.next())
//	{
//		if(isBoundaryMarked3(d))
//		{
//			boundaryMarkOrbit<3,VOLUME>(d);
//		}
//	}

//	unsigned int count = 0;
//	for(Dart d = begin(); d != end(); next(d))
//	{
//		if(isBoundaryMarked3(d))
//		{
//			++count;
//		}
//	}
//	std::cout << "nb boundar marked = " << count << std::endl;
//
//	count = 0;
//	for(Dart d = begin(); d != end(); next(d))
//	{
//		if(isBoundaryMarked3(d))
//		{
//			++count;
//			std::cout << count << std::endl;
//			//Map3::deleteVolume(d,false);
//			//deleteVolume(d,false);
//		}
//	}


	//std::cout << "Map closed (" << closeMap() <<" boundary faces)" << std::endl;
}

template <typename MAP_IMPL>
Dart Map3<MAP_IMPL>::explodBorderTopo(Dart d)
{
	std::vector<std::pair<Dart,Dart> > ve;
	ve.reserve(1024);

	//stocke un brin par face du bord
	DartMarker<Map3<MAP_IMPL> > me(*this);
	for(Dart dit = this->begin() ; dit != this->end() ; this->next(dit))
	{
		if(this->template isBoundaryMarked<3>(dit) && !me.isMarked(dit))
		{
			ve.push_back(std::make_pair(dit, this->phi2(dit)));
			me.template markOrbit<EDGE>(dit);
		}
	}

	//decoud chaque face
	for(std::vector<std::pair<Dart,Dart> >::iterator it = ve.begin() ; it != ve.end() ; ++it)
	{
		ParentMap::unsewFaces((*it).first, false);
	}

	//triangule chaque face
	DartMarker<Map3<MAP_IMPL> > mf(*this);
	for(std::vector<std::pair<Dart,Dart> >::iterator it = ve.begin() ; it != ve.end() ; ++it)
	{
		Dart first = (*it).first;
		Dart second = (*it).second;

		if(!mf.isMarked(first))
		{
			unsigned int degf = ParentMap::faceDegree(first);

			Dart dnf = ParentMap::newFace(degf, false);
			Dart dit = first;
			do
			{
				ParentMap::sewFaces(dit, dnf, false);
				this->template copyDartEmbedding<VERTEX>(dnf, this->phi1(dit)) ;
				dit = this->phi1(dit);
				dnf = this->phi_1(dnf);
			} while(dit != first);

			mf.template markOrbit<FACE>(first);

			Dart db = dnf;
			Dart d1 = this->phi1(db);
			ParentMap::splitFace(db, d1) ;
			ParentMap::cutEdge(this->phi_1(db)) ;

			Dart x = this->phi2(this->phi_1(db)) ;
			Dart dd = this->phi1(this->phi1(this->phi1(x)));
			while(dd != x)
			{
				Dart next = this->phi1(dd) ;
				ParentMap::splitFace(dd, this->phi1(x)) ;
				dd = next ;
			}

			Dart cd = this->phi_1(db);
			do
			{
				this->template setDartEmbedding<VERTEX>(this->phi2(cd), this->template getEmbedding<VERTEX>(this->phi1(cd))) ;
				cd = this->phi2(this->phi_1(cd));
			} while(cd != this->phi_1(db));
		}

		if(!mf.isMarked(second))
		{
			mf.template markOrbit<FACE>(second);
			unsigned int degf = ParentMap::faceDegree(second);

			Dart dnf = ParentMap::newFace(degf, false);
			Dart dit = second;
			do
			{
				ParentMap::sewFaces(dit, dnf, false);
				this->template copyDartEmbedding<VERTEX>(dnf, this->phi1(dit));
				dit = this->phi1(dit);
				dnf = this->phi_1(dnf);
			} while(dit != second);

			mf.template markOrbit<FACE>(second);

			Dart db = dnf;
			Dart d1 = this->phi1(db);
			ParentMap::splitFace(db, d1);
			ParentMap::cutEdge(this->phi_1(db));

			Dart x = this->phi2(this->phi_1(db)) ;
			Dart dd = this->phi1(this->phi1(this->phi1(x)));
			while(dd != x)
			{
				Dart next = this->phi1(dd) ;
				ParentMap::splitFace(dd, this->phi1(x)) ;
				dd = next ;
			}

			Dart cd = this->phi_1(db);
			do
			{
				this->template setDartEmbedding<VERTEX>(this->phi2(cd), this->template getEmbedding<VERTEX>(this->phi1(cd))) ;
				cd = this->phi2(this->phi_1(cd));
			} while(cd != this->phi_1(db));
		}
	}

	//close de chaque nouveau volume
	for(std::vector<std::pair<Dart,Dart> >::iterator it = ve.begin() ; it != ve.end() ; ++it)
	{
		Dart dit1 = this->phi2((*it).first);
		Dart dit2 = this->phi2((*it).second);
		Map3<MAP_IMPL>::sewVolumes(dit1, dit2, false);
	}

	Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(*this, this->phi_1(this->phi2(ve.front().first)));

	return this->phi_1(this->phi2(ve.front().first));
}

template <typename MAP_IMPL>
void Map3<MAP_IMPL>::computeDualTest()
{
//		unsigned int count = 0;
//		CellMarkerNoUnmark<VERTEX> cv(*this);
//		std::vector<Dart> v;
//		for(Dart d = begin(); d != end(); next(d))
//		{
//			if(!cv.isMarked(d) && isBoundaryMarked3(d))
//			{
//				++count;
//				v.push_back(d);
//				cv.mark(d);
//			}
//		}
//
//		cv.unmarkAll();

//		std::cout << "boundary vertices : " << count << std::endl;

	DartAttribute<Dart, MAP_IMPL> old_phi1 = this->template getAttribute<Dart, DART>("phi1") ;
	DartAttribute<Dart, MAP_IMPL> old_phi_1 = this->template getAttribute<Dart, DART>("phi_1") ;
	DartAttribute<Dart, MAP_IMPL> new_phi1 = this->template addAttribute<Dart, DART>("new_phi1") ;
	DartAttribute<Dart, MAP_IMPL> new_phi_1 = this->template addAttribute<Dart, DART>("new_phi_1") ;

	DartAttribute<Dart, MAP_IMPL> old_phi2 = this->template getAttribute<Dart, DART>("phi2") ;
	DartAttribute<Dart, MAP_IMPL> new_phi2 = this->template addAttribute<Dart, DART>("new_phi2") ;

	for(Dart d = this->begin(); d != this->end(); this->next(d))
	{
		Dart dd = this->phi2(phi3(d)) ;
		new_phi1[d] = dd ;
		new_phi_1[dd] = d ;

		Dart ddd = this->phi1(phi3(d));
		new_phi2[d] = ddd;
		new_phi2[ddd] = d;
	}

	this->template swapAttributes<Dart>(old_phi1, new_phi1) ;
	this->template swapAttributes<Dart>(old_phi_1, new_phi_1) ;
	this->template swapAttributes<Dart>(old_phi2, new_phi2) ;

	this->removeAttribute(new_phi1) ;
	this->removeAttribute(new_phi_1) ;
	this->removeAttribute(new_phi2) ;

	this->swapEmbeddingContainers(VERTEX, VOLUME) ;

	for(Dart d = this->begin(); d != this->end(); this->next(d))
	{
        if(this->isBoundaryMarked<3>(d))
			Map3<MAP_IMPL>::deleteVolume(d, false);
	}

	closeMap();

//	reverseOrientation();
//
//		for(std::vector<Dart>::iterator it = v.begin() ; it != v.end() ; ++it)
//		{
//			boundaryUnmarkOrbit<3,VOLUME>(*it);
//		}
//
//		for(std::vector<Dart>::iterator it = v.begin() ; it != v.end() ; ++it)
//		{
//			Map3::deleteVolume(*it);
//		}
//
//		std::cout << "boundary faces : " << closeMap() << std::endl;

//	//boundary management
//	for(Dart d = begin(); d != end(); next(d))
//	{
//		if(isBoundaryMarked3(d))
//		{
//			//Dart dit = deleteVertex(phi3(d));
//			//deleteVolume(phi3(d));
//			//if(dit == NIL)
//			//{
//			//	std::cout << "ploooooooooooooooooooop" << std::endl;
//			//	return;
//			//}
//			//else
//			//{
//			//	std::cout << "gooooooooooooooooooooood" << std::endl;
//			//	boundaryMarkOrbit<3,VOLUME>(dit);
//			//	return;
//			//}
//			//boundaryUnmarkOrbit<3,VOLUME>(d);
//			//deleteVolume(d);
//		}
//	}
}


template <typename MAP_IMPL>
void Map3<MAP_IMPL>::moveFrom(Map2<MAP_IMPL>& mapf)
{
	GenericMap::moveData(mapf);
	MAP_IMPL::removeLastInvolutionPtr();
	MAP_IMPL::addInvolution() ;
	MAP_IMPL::restore_topo_shortcuts() ;
	GenericMap::garbageMarkVectors();

	closeMap();
}


} // namespace CGoGN
