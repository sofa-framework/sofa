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

#include "Topology/map/map3.h"
#include "Topology/generic/traversor3.h"

namespace CGoGN
{

void Map3::compactTopoRelations(const std::vector<unsigned int>& oldnew)
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

		d_index = dartIndex(m_phi3->operator[](i));
		if (d_index != oldnew[d_index])
			m_phi3->operator[](i) = Dart(oldnew[d_index]);
//
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
//		{
//			Dart& d = m_phi3->operator [](i);
//			Dart e = Dart(oldnew[d.index]);
//			if (d != e)
//				d = e;
//		}
	}
}

/*! @name Generator and Deletor
 *  To generate or delete volumes in a 3-map
 *************************************************************************/

void Map3::deleteVolume(Dart d, bool withBoundary)
{
	if(withBoundary)
	{
		DartMarkerStore mark(*this);		// Lock a marker

		std::vector<Dart> visitedFaces;		// Faces that are traversed
		visitedFaces.reserve(512);
		visitedFaces.push_back(d);			// Start with the face of d

		mark.markOrbit<FACE2>(d) ;

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
					mark.markOrbit<FACE2>(ee) ;
				}
				e = phi1(e) ;
			} while(e != visitedFaces[i]) ;
		}

		Dart dd = phi3(d) ;
		Map2::deleteCC(d) ; //deleting the volume
		Map2::deleteCC(dd) ; //deleting its border (created from the unsew operation)

		return;
	}

	//else remove the CC and create fixed points
	DartMarkerStore mark(*this);		// Lock a marker

	std::vector<Dart> visitedFaces;		// Faces that are traversed
	visitedFaces.reserve(512);
	visitedFaces.push_back(d);			// Start with the face of d

	mark.markOrbit<FACE2>(d) ;

	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		Dart e = visitedFaces[i] ;

		Dart it = e ;
		do
		{
			phi3unsew(it);
			it = phi1(it) ;
		} while(it != e) ;

		do	// add all face neighbours to the table
		{
			Dart ee = phi2(e) ;
			if(!mark.isMarked(ee)) // not already marked
			{
				visitedFaces.push_back(ee) ;
				mark.markOrbit<FACE2>(ee) ;
			}
			e = phi1(e) ;
		} while(e != visitedFaces[i]) ;
	}

	Map2::deleteCC(d) ; //deleting the volume
}

void Map3::fillHole(Dart d)
{
	assert(isBoundaryFace(d)) ;
	Dart dd = d ;
	if(!isBoundaryMarked3(dd))
		dd = phi3(dd) ;
	boundaryUnmarkOrbit<VOLUME,3>(dd) ;
}

void Map3::createHole(Dart d)
{
	assert(!isBoundaryFace(d)) ;
	boundaryMarkOrbit<VOLUME,3>(d) ;
}

/*! @name Topological Operators
 *  Topological operations on 3-maps
 *************************************************************************/

Dart Map3::splitVertex(std::vector<Dart>& vd)
{
	//assert(checkPathAroundVertex(vd)) ;

	//bool boundE = false;

	Dart prev = vd.front();	//elt 0

	Dart db1 = NIL;
	if(isBoundaryFace(phi2(prev)))
	{
		db1 = phi2(phi3(phi1(phi2(prev))));
	}

	Dart fs = phi_1(phi2(phi_1(prev)));	//first side

	Map2::splitVertex(prev, phi2(fs));

	for(unsigned int i = 1; i < vd.size(); ++i)
	{
		prev = vd[i];

		Dart fs = phi_1(phi2(phi_1(prev)));	//first side

		Map2::splitVertex(prev, phi2(fs));

		Dart d1 = phi_1(phi2(phi_1(vd[i-1])));
		Dart d2 = phi1(phi2(vd[i]));

		phi3sew(d1, d2);
	}

	Dart db2 = NIL;
	if(isBoundaryFace(phi2(phi_1(prev))))
	{
		db2 = phi2(phi3(phi2(phi_1(prev))));
	}

	if(db1 != NIL && db2 != NIL)
	{
		Map2::splitVertex(db1, db2);
		phi3sew(phi1(phi2(db2)), phi_1(phi3(phi2(db2))));
		phi3sew(phi1(phi2(db1)), phi_1(phi3(phi2(db1))));
	}
	else
	{
		Dart dbegin = phi1(phi2(vd.front()));
		Dart dend = phi_1(phi2(phi_1(vd.back())));
		phi3sew(dbegin, dend);
	}

	return phi_1(phi2(phi_1(prev)));
}

//	//unsew the face path
//	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
//	{
//		Dart dit = *it;
//
//		Map1::cutEdge(phi_1(phi2(phi_1(dit)))); //comme un vertexSplit
//		Map1::cutEdge(phi2(phi1(phi2(dit))));
//		Map2::sewFaces(phi1(phi2(phi1(phi2(dit)))), phi_1(phi2(phi_1(dit))), false);
//
//
//
//		Dart dit3 = phi3(dit);
//		unsewVolumes(dit);

//		Dart f1 = newFace(3,false);
//		Dart f2 = newFace(3,false);
//		Dart f3 = newFace(3,false);
//		Dart f4 = newFace(3,false);
//
//		sewFaces(f1,f2,false);
//		sewFaces(phi_1(f1), f3, false);
//		sewFaces(phi1(f1), f4, false);
//		sewFaces(phi_1(f2), phi1(f4), false);
//		sewFaces(phi_1(f3), phi1(f2), false);
//		sewFaces(phi1(f3), phi_1(f4), false);
//
//		sewVolumes(dit,f3);
//		sewVolumes(dit3,f4);
//	}

/*
	if(isBoundaryVertex(d))
	{
		unsewVolumes(d);
		unsewVolumes(e);

		Dart dc = phi1(phi2(d));

		//unsewVolumes(phi2(dc));
		Map2::splitVertex(d, phi1(phi2(dc)));


//		Map2::splitFace(d, phi2(dc));

//		Dart ec = phi_1(phi2(e));
//		Map2::splitVertex(e, ec);
//		//Map2::splitFace(e, phi2(ec));
	}
*/


Dart Map3::deleteVertex(Dart d)
{
	//if(isBoundaryVertex(d))
	//	return NIL ;

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

	std::cout << "nb faces " << fstore.size() << std::endl;

	Dart res = NIL ;
	for(std::vector<Dart>::iterator it = fstore.begin() ; it != fstore.end() ; ++it)
	{
		Dart fit = *it ;
		Dart end = phi_1(fit) ;
		fit = phi1(fit) ;

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
				Dart d2 = phi2(fit) ;
				Dart d3 = phi3(fit) ;
				Dart d32 = phi2(d3) ;

				if(res == NIL)
					res = d2 ;

				phi2unsew(d2) ;
				phi2unsew(d32) ;
				phi2sew(d2, d32) ;
				phi2sew(fit, d3) ;

				fit = phi1(fit) ;
			}
		}
	}

	Map2::deleteCC(d) ;

	return res ;
}

Dart Map3::cutEdge(Dart d)
{
	Dart prev = d;
	Dart dd = alpha2(d);
	Dart nd = Map2::cutEdge(d);

	while (dd != d)
	{
		prev = dd;
		dd = alpha2(dd);

		Map2::cutEdge(prev);

		Dart d3 = phi3(prev);
		phi3unsew(prev);
		phi3sew(prev, phi1(d3));
		phi3sew(d3, phi1(prev));
	}

	Dart d3 = phi3(d);
	phi3unsew(d);
	phi3sew(d, phi1(d3));
	phi3sew(d3, phi1(d));

	return nd;
}

bool Map3::uncutEdge(Dart d)
{
	if(vertexDegree(phi1(d)) == 2)
	{
		Dart prev = d ;
		phi3unsew(phi1(prev)) ;

		Dart dd = d;
		do
		{
			prev = dd;
			dd = alpha2(dd);

			phi3unsew(phi2(prev)) ;
			phi3unsew(phi2(phi1(prev))) ;
			Map2::uncutEdge(prev);
			phi3sew(dd, phi2(prev));
		} while (dd != d) ;

		return true;
	}
	return false;
}

bool Map3::deleteEdgePreCond(Dart d)
{
	unsigned int nb1 = vertexDegree(d);
	unsigned int nb2 = vertexDegree(phi1(d));
	return (nb1!=2) && (nb2!=2);
}

Dart Map3::deleteEdge(Dart d)
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

			phi2unsew(d2) ;
			phi2unsew(d32) ;
			phi2sew(d2, d32) ;
			phi2sew(fit, d3) ;

			fit = phi1(fit) ;
		}
		dit = alpha2(dit) ;
	} while(dit != d) ;

	Map2::deleteCC(d) ;

	return res ;
}

//Dart Map3::collapseEdge(Dart d, bool delDegenerateVolumes)
//{
//	Dart resV = NIL;
//
//	Dart dit = d;
//
//	do
//	{
//		Dart e = dit;
//		dit = alpha2(dit);
//
//		//test si un seul polyedre autour de l'arete
//		if(e == dit)
//			resV == phi3(phi2(phi1(e)));
//
//		if(delDegenerateVolumes)
//		{
//			Map2::collapseEdge(e, true);
//			collapseDegeneretedVolume(e);
//		}
//		else
//			Map2::collapseEdge(e, false);
//
//		if(resV == NIL)
//		{
//
//		}
//
//	}while(d != dit);
//
//	return resV;
//}

Dart Map3::collapseEdge(Dart d, bool delDegenerateVolumes)
{
	Dart resV = NIL;
	Dart dit = d;

	std::vector<Dart> darts;
	do
	{
		darts.push_back(dit);
		dit = alpha2(dit);
	}while(dit != d);

	for (std::vector<Dart>::iterator it = darts.begin(); it != darts.end(); ++it)
	{
		Dart x = phi2(phi_1(*it));

		Dart resCV = NIL;

		if(!isBoundaryFace(phi2(phi1(*it))))
			resCV = phi3(phi2(phi1(*it)));
		else if(!isBoundaryFace(phi2(phi_1(*it))))
			resCV = phi3(phi2(phi_1(*it)));

		resV = Map2::collapseEdge(*it, true);
		if (delDegenerateVolumes)
			if(collapseDegeneretedVolume(x) && resCV != NIL)
				resV = resCV;
	}

	return resV;
}


bool Map3::splitFacePreCond(Dart d, Dart e)
{
	return (d != e && sameOrientedFace(d, e)) ;
}

void Map3::splitFace(Dart d, Dart e)
{
//	assert(d != e && sameOrientedFace(d, e)) ;
	assert(splitFacePreCond(d,e));

	Dart dd = phi1(phi3(d));
	Dart ee = phi1(phi3(e));

	Map2::splitFace(d, e);
	Map2::splitFace(dd, ee);

	phi3sew(phi_1(d), phi_1(ee));
	phi3sew(phi_1(e), phi_1(dd));
}

bool Map3::mergeFaces(Dart d)
{
	assert(edgeDegree(d)==2);

	Dart dd = phi3(d);

	phi3unsew(d);
	phi3unsew(dd);

	//use code of mergesFaces to override the if(isBoundaryEdge)
	//we have to merge the faces if the face is linked to a border also
//	Map2::mergeFaces(d);
	Dart e = phi2(d) ;
	phi2unsew(d) ;
	Map1::mergeCycles(d, phi1(e)) ;
	Map1::splitCycle(e, phi1(d)) ;
	Map1::deleteCycle(d) ;
//	Map2::mergeFaces(dd);
	e = phi2(dd) ;
	phi2unsew(dd) ;
	Map1::mergeCycles(dd, phi1(e)) ;
	Map1::splitCycle(e, phi1(dd)) ;
	Map1::deleteCycle(dd);

	return true;
}

Dart Map3::collapseFace(Dart d, bool delDegenerateVolumes)
{
	Dart resV = NIL;
	Dart stop = phi_1(d);
	Dart dit = d;
	std::vector<Dart> vd;
	vd.reserve(32);

	do
	{
		vd.push_back(alpha2(dit));
		dit = phi1(dit);
	}
	while(dit != stop);

	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
		resV = Map3::collapseEdge(*it, delDegenerateVolumes);

	return resV;
}

//bool Map3::collapseDegeneretedVolume(Dart d)
//{
//	Dart e1 = phi2(d);
//	Dart e2 = phi2(phi1(d));
//
//	//Si les deux faces ne sont pas du bord
//	if(!isBoundaryFace(e1) && !isBoundaryFace(e2))
//	{
//		sewVolumes(phi3(e1),phi3(e2));
//		deleteVolume(d);
//		return true;
//	}
//	else
//	{
//		//alors simple suppression du volume degenere
//		deleteVolume(d);
//		return true;
//	}
//
//	return false;
//}

bool Map3::collapseDegeneretedVolume(Dart d)
{
	Dart e1 = d;
	Dart e2 = phi2(d);

	do
	{
		if (e1 != phi2(e2))
			return false;
		e1 = phi1(e1);
		e2 = phi_1(e2);
	}while (e1 != d);

	if (e2 != phi2(d))
		return false;

	// degenerated:
	do
	{
		Dart f1 = phi3(e1);
		Dart f2 = phi3(e2);
		phi3unsew(e1);
		phi3unsew(e2);
		phi3sew(f1,f2);
		e1 = phi1(e1);
		e2 = phi_1(e2);
	}while (e1 != d);

	Map2::deleteCC(d) ;
	return true;
}


bool Map3::sewVolumesPreCond(Dart d, Dart e)
{
	return (faceDegree(d) == faceDegree(e));
}

void Map3::sewVolumes(Dart d, Dart e, bool withBoundary)
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
			fitD = phi1(fitD) ;
			fitE = phi_1(fitE) ;
		} while(fitD != d) ;
		return ;
	}

	Dart dd = phi3(d) ;
	Dart ee = phi3(e) ;

	Dart fitD = dd ;
	Dart fitE = ee ;
	do
	{
		Dart fitD2 = phi2(fitD) ;
		Dart fitE2 = phi2(fitE) ;
		if(fitD2 != fitE)
		{
			phi2unsew(fitD) ;
			phi2unsew(fitE) ;
			phi2sew(fitD2, fitE2) ;
			phi2sew(fitD, fitE) ;
		}
		phi3unsew(fitD) ;
		phi3unsew(fitE) ;
		fitD = phi1(fitD) ;
		fitE = phi_1(fitE) ;
	} while(fitD != dd) ;
	Map2::deleteCC(dd) ;

	fitD = d ;
	fitE = e ;
	do
	{
		phi3sew(fitD, fitE) ;
		fitD = phi1(fitD) ;
		fitE = phi_1(fitE) ;
	} while(fitD != d) ;
}

bool Map3::unsewVolumesPreCond(Dart d)
{
	return (!isBoundaryFace(d)) ;
}


void Map3::unsewVolumes(Dart d, bool withBoundary)
{
	assert(unsewVolumesPreCond(d)) ;

	if (!withBoundary)
	{
		Dart fitD = d ;
		do
		{
			phi3unsew(fitD) ;
			fitD = phi1(fitD) ;
		} while(fitD != d) ;
		return ;
	}

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
			phi2unsew(f) ;
			phi2sew(fitB1, f) ;
			phi2sew(fitB2, f2) ;
		}
		else
			phi2sew(fitB1, fitB2) ;

		phi3unsew(fit1) ;
		phi3sew(fit1, fitB1) ;
		phi3sew(fit2, fitB2) ;

		fit1 = phi1(fit1) ;
		fit2 = phi_1(fit2) ;
		fitB1 = phi_1(fitB1) ;
		fitB2 = phi1(fitB2) ;
	} while(fitB1 != b1) ;
}

bool Map3::mergeVolumes(Dart d, bool deleteFace)
{
	if(!Map3::isBoundaryFace(d))
	{
		Map2::mergeVolumes(d, phi3(d), deleteFace); // merge the two volumes along common face
		return true ;
	}
	return false ;
}

void Map3::splitVolume(std::vector<Dart>& vd)
{
	//assert(checkSimpleOrientedPath(vd)) ;

	Dart e = vd.front();
	Dart e2 = phi2(e);

	Map2::splitSurface(vd,true,true);

	//sew the two connected components
	Map3::sewVolumes(phi2(e), phi2(e2), false);
}

void Map3::splitVolumeWithFace(std::vector<Dart>& vd, Dart d)
{
	assert(vd.size() == faceDegree(d));

	// deconnect edges around the path
	// sew the given face into the paths
	Dart dit = d;
	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		Dart it2 = phi2(*it);
		unsewFaces(*it, false) ;

		sewFaces(*it,dit,false);
		sewFaces(it2, phi3(dit),false);

		dit = phi_1(dit);

	}
}

Dart Map3::collapseVolume(Dart d, bool delDegenerateVolumes)
{
	Dart resV = NIL;
	std::vector<Dart> vd;
	vd.reserve(32);

	vd.push_back(d);
	vd.push_back(alpha2(phi1(d)));
	vd.push_back(alpha2(phi_1(phi2(phi1(d)))));

//	Traversor3WF<Map3> tra(*this, phi1(d));
//	for(Dart dit = tra.begin() ; dit != tra.end() ; dit = tra.next())
//	{
//		vd.push_back(alpha2(dit));
//	}
//	vd.pop_back();

	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
		resV = Map3::collapseEdge(*it, delDegenerateVolumes);

	return resV;
}

/*! @name Topological Queries
 *  Return or set various topological information
 *************************************************************************/

bool Map3::sameVertex(Dart d, Dart e)
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

unsigned int Map3::vertexDegree(Dart d)
{
	unsigned int count = 0;

	Traversor3VE<Map3> trav3VE(*this, d);
	for(Dart dit = trav3VE.begin() ; dit != trav3VE.end() ; dit = trav3VE.next())
	{
		++count;
	}

	return count;
}


int Map3::checkVertexDegree(Dart d, unsigned int vd)
{
	unsigned int count = 0;

	Traversor3VE<Map3> trav3VE(*this, d);
	Dart dit = trav3VE.begin();
	for( ; (count<=vd) && (dit != trav3VE.end()) ; dit = trav3VE.next())
	{
		++count;
	}

	return count - vd;
}


unsigned int Map3::vertexDegreeOnBoundary(Dart d)
{
	assert(Map3::isBoundaryVertex(d));

	return Map2::vertexDegree(d);
}

bool Map3::isBoundaryVertex(Dart d)
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

Dart Map3::findBoundaryFaceOfVertex(Dart d)
{
	DartMarkerStore mv(*this);	// Lock a marker

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(256);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; i < darts.size(); ++i)
	{
		if(isBoundaryMarked3(darts[i]))
			return darts[i];

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
	return NIL ;
}

bool Map3::sameOrientedEdge(Dart d, Dart e)
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

unsigned int Map3::edgeDegree(Dart d)
{
	unsigned int deg = 0;
	Dart it = d;
	do
	{
		if(!isBoundaryMarked3(it))
			++deg;
		it = alpha2(it);
	} while(it != d);
	return deg;
}

bool Map3::isBoundaryEdge(Dart d)
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

Dart Map3::findBoundaryFaceOfEdge(Dart d)
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

bool Map3::isBoundaryVolume(Dart d)
{
	Traversor3WF<Map3> tra(*this, d);
	for(Dart dit = tra.begin() ; dit != tra.end() ; dit = tra.next())
	{
		if(isBoundaryMarked3(phi3(dit)))
			return true ;
	}
	return false;
}

bool Map3::hasBoundaryEdge(Dart d)
{
	Traversor3WE<Map3> tra(*this, d);
	for(Dart dit = tra.begin() ; dit != tra.end() ; dit = tra.next())
	{
		if(isBoundaryEdge(dit))
			return true;
	}

	return false;
}

bool Map3::check()
{
    std::cout << "Check: topology begin" << std::endl;
    DartMarkerStore m(*this);
    for(Dart d = Map3::begin(); d != Map3::end(); Map3::next(d))
    {
        Dart d3 = phi3(d);
        if (phi3(d3) != d) // phi3 involution ?
		{
            std::cout << "Check: phi3 is not an involution" << std::endl;
            return false;
        }

		if(phi1(d3) != phi3(phi_1(d)))
		{
			if(isBoundaryMarked3(d))
				std::cout << "Boundary case - Check: phi3 , faces are not entirely sewn" << std::endl;
			else
				std::cout << "Check: phi3 , faces are not entirely sewn" << std::endl;
            std::cout << "face : " << phi1(d3) << " and face = " << phi3(phi_1(d)) << std::endl;
			return false;
		}

        Dart d2 = phi2(d);
        if (phi2(d2) != d) // phi2 involution ?
		{
        	if(isBoundaryMarked3(d))
        		std::cout << "Boundary case - ";

        	std::cout << "Check: phi2 is not an involution" << std::endl;
            return false;
        }

        Dart d1 = phi1(d);
        if (phi_1(d1) != d) // phi1 a une image correcte ?
		{
        	if(isBoundaryMarked3(d))
        		std::cout << "Boundary case - ";

            std::cout << "Check: unconsistent phi_1 link" << std::endl;
            return false;
        }

        if (m.isMarked(d1)) // phi1 a un seul antécédent ?
		{
        	if(isBoundaryMarked3(d))
        		std::cout << "Boundary case - ";

            std::cout << "Check: dart with two phi1 predecessors" << std::endl;
            return false;
        }
        m.mark(d1);

        if (d1 == d)
        {
        	if(isBoundaryMarked3(d))
        		std::cout << "Boundary case - ";

            std::cout << "Check: (warning) face loop (one edge)" << std::endl;
        }

        if (phi1(d1) == d)
        {
        	if(isBoundaryMarked3(d))
        		std::cout << "Boundary case - ";

            std::cout << "Check: (warning) face with only two edges" << std::endl;
        }

        if (phi2(d1) == d)
        {
        	if(isBoundaryMarked3(d))
        		std::cout << "Boundary case - ";

        	std::cout << "Check: (warning) dandling edge (phi2)" << std::endl;
        }

        if (phi3(d1) == d)
        {
        	if(isBoundaryMarked3(d))
        		std::cout << "Boundary case - ";

            std::cout << "Check: (warning) dandling edge (phi3)" << std::endl;
        }
    }

    for(Dart d = this->begin(); d != this->end(); this->next(d))
    {
        if (!m.isMarked(d)) // phi1 a au moins un antécédent ?
		{
        	if(isBoundaryMarked3(d))
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

bool Map3::foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int thread)
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

bool Map3::foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int thread)
{
	Dart it = d;
	do
	{
		if (Map2::foreach_dart_of_edge(it, f, thread))
			return true;
		it = alpha2(it);
	} while (it != d);
	return false;
}

bool Map3::foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread)
{
	DartMarkerStore mv(*this,thread);	// Lock a marker
	bool found = false;					// Last functor return value

	std::vector<Dart> darts;	// Darts that are traversed
	darts.reserve(1024);
	darts.push_back(d);			// Start with the dart d
	mv.mark(d);

	for(unsigned int i = 0; !found && i < darts.size(); ++i)
	{
		// add all successors if they are not marked yet
		Dart d2 = phi1(darts[i]); // turn in face
		Dart d3 = phi2(darts[i]); // change face
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

		found = f(darts[i]);
	}
	return found;
}

/*! @name Close map after import or creation
 *  These functions must be used with care, generally only by import/creation algorithms
 *************************************************************************/

Dart Map3::newBoundaryCycle(unsigned int nbE)
{
	Dart d = Map1::newCycle(nbE);
	boundaryMarkOrbit<FACE,3>(d);
	return d;
}

unsigned int Map3::closeHole(Dart d, bool forboundary)
{
	assert(phi3(d) == d);		// Nothing to close
	DartMarkerStore m(*this) ;

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(d);		// Start with the face of d
	m.markOrbit<FACE2>(d) ;

	unsigned int count = 0 ;

	// For every face added to the list
	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		Dart it = visitedFaces[i] ;
		Dart f = it ;

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
				if(phi3(e) == e)
				{
					found = true ;
					if(!m.isMarked(e))
					{
						visitedFaces.push_back(e) ;
						m.markOrbit<FACE2>(e) ;
					}
				}
				else if(isBoundaryMarked3(e))
				{
					found = true ;
					phi2sew(e, bit) ;
				}
				else
					e = alpha2(e) ;
			} while(!found) ;

			phi3sew(f, bit) ;
			bit = phi_1(bit) ;
			f = phi1(f);
		} while(f != it) ;
	}

	return count ;
}

unsigned int Map3::closeMap()
{
	// Search the map for topological holes (fix points of phi3)
	unsigned int nb = 0 ;
	for (Dart d = begin(); d != end(); next(d))
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

void Map3::reverseOrientation()
{

}

void Map3::computeDual()
{
	DartAttribute<Dart> old_phi1 = getAttribute<Dart, DART>("phi1") ;
	DartAttribute<Dart> old_phi_1 = getAttribute<Dart, DART>("phi_1") ;
	DartAttribute<Dart> new_phi1 = addAttribute<Dart, DART>("new_phi1") ;
	DartAttribute<Dart> new_phi_1 = addAttribute<Dart, DART>("new_phi_1") ;

	DartAttribute<Dart> old_phi2 = getAttribute<Dart, DART>("phi2") ;
	DartAttribute<Dart> new_phi2 = addAttribute<Dart, DART>("new_phi2") ;

	for(Dart d = begin(); d != end(); next(d))
	{
		Dart dd = phi2(phi3(d)) ;
		new_phi1[d] = dd ;
		new_phi_1[dd] = d ;

		Dart ddd = phi1(phi3(d));
		new_phi2[d] = ddd;
		new_phi2[ddd] = d;
	}

	swapAttributes<Dart>(old_phi1, new_phi1) ;
	swapAttributes<Dart>(old_phi_1, new_phi_1) ;
	swapAttributes<Dart>(old_phi2, new_phi2) ;

	removeAttribute(new_phi1) ;
	removeAttribute(new_phi_1) ;
	removeAttribute(new_phi2) ;

	swapEmbeddingContainers(VERTEX, VOLUME) ;

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
//			boundaryMarkOrbit<VOLUME,3>(d);
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

Dart Map3::explodBorderTopo(Dart d)
{
	std::vector<std::pair<Dart,Dart> > ve;
	ve.reserve(1024);

	//stocke un brin par face du bord
	DartMarker me(*this);
	for(Dart dit = begin() ; dit != end() ; next(dit))
	{
		if(isBoundaryMarked3(dit) && !me.isMarked(dit))
		{
			ve.push_back(std::make_pair(dit,phi2(dit)));
			me.markOrbit<EDGE>(dit);
		}
	}

	//decoud chaque face
	for(std::vector<std::pair<Dart,Dart> >::iterator it = ve.begin() ; it != ve.end() ; ++it)
	{
		Map2::unsewFaces((*it).first,false);
	}

	//triangule chaque face
	DartMarker mf(*this);
	for(std::vector<std::pair<Dart,Dart> >::iterator it = ve.begin() ; it != ve.end() ; ++it)
	{
		Dart first = (*it).first;
		Dart second = (*it).second;

		if(!mf.isMarked(first))
		{
			unsigned int degf = Map2::faceDegree(first);

			Dart dnf = Map2::newFace(degf,false);
            Dart dit = first;
            do
            {
            	Map2::sewFaces(dit,dnf,false);
            	copyDartEmbedding<VERTEX>(dnf, phi1(dit)) ;
            	dit = phi1(dit);
            	dnf = phi_1(dnf);
            }while(dit != first);

			mf.markOrbit<FACE>(first);

			Dart db = dnf;
			Dart d1 = phi1(db);
			Map2::splitFace(db, d1) ;
			Map2::cutEdge(phi_1(db)) ;

			Dart x = phi2(phi_1(db)) ;
			Dart dd = phi1(phi1(phi1(x)));
			while(dd != x)
			{
				Dart next = phi1(dd) ;
				Map2::splitFace(dd, phi1(x)) ;
				dd = next ;
			}

			Dart cd = phi_1(db);
			do
			{
				setDartEmbedding<VERTEX>(phi2(cd), getEmbedding<VERTEX>(phi1(cd))) ;
				cd = phi2(phi_1(cd));
			}while(cd != phi_1(db));

		}

		if(!mf.isMarked(second))
		{
			mf.markOrbit<FACE>(second);
			unsigned int degf = Map2::faceDegree(second);

			Dart dnf = Map2::newFace(degf,false);
            Dart dit = second;
            do
            {
            	Map2::sewFaces(dit,dnf,false);
            	copyDartEmbedding<VERTEX>(dnf, phi1(dit)) ;
            	dit = phi1(dit);
            	dnf = phi_1(dnf);
            }while(dit != second);

			mf.markOrbit<FACE>(second);

			Dart db = dnf;
			Dart d1 = phi1(db);
			Map2::splitFace(db, d1) ;
			Map2::cutEdge(phi_1(db)) ;

			Dart x = phi2(phi_1(db)) ;
			Dart dd = phi1(phi1(phi1(x)));
			while(dd != x)
			{
				Dart next = phi1(dd) ;
				Map2::splitFace(dd, phi1(x)) ;
				dd = next ;
			}

			Dart cd = phi_1(db);
			do
			{
				setDartEmbedding<VERTEX>(phi2(cd), getEmbedding<VERTEX>(phi1(cd))) ;
				cd = phi2(phi_1(cd));
			}while(cd != phi_1(db));
		}

	}

	//close de chaque nouveau volume
	for(std::vector<std::pair<Dart,Dart> >::iterator it = ve.begin() ; it != ve.end() ; ++it)
	{
		Dart dit1 = phi2((*it).first);
		Dart dit2 = phi2((*it).second);
		Map3::sewVolumes(dit1, dit2, false);
	}

	setOrbitEmbeddingOnNewCell<VERTEX>(phi_1(phi2(ve.front().first)));

	return phi_1(phi2(ve.front().first));
}

void Map3::computeDualTest()
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

	DartAttribute<Dart> old_phi1 = getAttribute<Dart, DART>("phi1") ;
	DartAttribute<Dart> old_phi_1 = getAttribute<Dart, DART>("phi_1") ;
	DartAttribute<Dart> new_phi1 = addAttribute<Dart, DART>("new_phi1") ;
	DartAttribute<Dart> new_phi_1 = addAttribute<Dart, DART>("new_phi_1") ;

	DartAttribute<Dart> old_phi2 = getAttribute<Dart, DART>("phi2") ;
	DartAttribute<Dart> new_phi2 = addAttribute<Dart, DART>("new_phi2") ;

	for(Dart d = begin(); d != end(); next(d))
	{
		Dart dd = phi2(phi3(d)) ;
		new_phi1[d] = dd ;
		new_phi_1[dd] = d ;

		Dart ddd = phi1(phi3(d));
		new_phi2[d] = ddd;
		new_phi2[ddd] = d;
	}

	swapAttributes<Dart>(old_phi1, new_phi1) ;
	swapAttributes<Dart>(old_phi_1, new_phi_1) ;
	swapAttributes<Dart>(old_phi2, new_phi2) ;

	removeAttribute(new_phi1) ;
	removeAttribute(new_phi_1) ;
	removeAttribute(new_phi2) ;

	swapEmbeddingContainers(VERTEX, VOLUME) ;

	for(Dart d = begin(); d != end(); next(d))
	{
		if(isBoundaryMarked3(d))
		{
			Map3::deleteVolume(d,false);
		}
	}

	closeMap();

//	reverseOrientation();
//
//		for(std::vector<Dart>::iterator it = v.begin() ; it != v.end() ; ++it)
//		{
//			boundaryUnmarkOrbit<VOLUME,3>(*it);
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
//			//	boundaryMarkOrbit<VOLUME,3>(dit);
//			//	return;
//			//}
//			//boundaryUnmarkOrbit<VOLUME,3>(d);
//			//deleteVolume(d);
//		}
//	}

}

} // namespace CGoGN
