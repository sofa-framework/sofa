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
#include <vector>
#include <algorithm>
#include "Topology/map/embeddedMap3.h"
#include "Topology/generic/traversor3.h"

namespace CGoGN
{

//TODO
Dart EmbeddedMap3::splitVertex(std::vector<Dart>& vd)
{
	Dart d = vd.front();
	Dart d2 = phi1(phi2(d));

	Dart dres = Map3::splitVertex(vd);

	if(isOrbitEmbedded<VERTEX>())
	{
		setOrbitEmbeddingOnNewCell<VERTEX>(d2);
		copyCell<VERTEX>(d2, d);
		setOrbitEmbedding<VERTEX>( d, getEmbedding<VERTEX>(d));
	}

	if(isOrbitEmbedded<EDGE>())
	{
		initOrbitEmbeddingNewCell<EDGE>(dres) ; // TODO : check if dres is a dart of the new edge
	}

	if(isOrbitEmbedded<VOLUME>())
	{
		for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
		{
			setOrbitEmbedding<VOLUME>( *it, getEmbedding<VOLUME>(*it)) ;
		}
	}

	return dres;
}

//TODO
Dart EmbeddedMap3::deleteVertex(Dart d)
{
	Dart v = Map3::deleteVertex(d) ;
	if(v != NIL)
	{
		if (isOrbitEmbedded<VOLUME>())
		{
			setOrbitEmbedding<VOLUME>(v, getEmbedding<VOLUME>(v)) ;
		}
	}
	return v ;
}

Dart EmbeddedMap3::cutEdge(Dart d)
{
	Dart nd = Map3::cutEdge(d);

//	if(isOrbitEmbedded<VERTEX>())
//	{
//		initOrbitEmbeddingNewCell<VERTEX>(nd) ;
//	}

	if(isOrbitEmbedded<EDGE>())
	{
		// embed the new darts created in the cut edge
		setOrbitEmbedding<EDGE>(d, getEmbedding<EDGE>(d)) ;
		// embed a new cell for the new edge and copy the attributes' line (c) Lionel
		setOrbitEmbeddingOnNewCell<EDGE>(nd) ;
		copyCell<EDGE>(nd, d) ;
	}

	if(isOrbitEmbedded<FACE2>())
	{
		Dart f = d;
		do
		{
			Dart f1 = phi1(f) ;

			copyDartEmbedding<FACE2>(f1, f);
			Dart e = phi3(f1);
			copyDartEmbedding<FACE2>(phi1(e), e);
			f = alpha2(f);
		} while(f != d);
	}

	if(isOrbitEmbedded<FACE>())
	{
		Dart f = d;
		do
		{
			unsigned int fEmb = getEmbedding<FACE>(f) ;
			setDartEmbedding<FACE>(phi1(f), fEmb);
			setDartEmbedding<FACE>(phi3(f), fEmb);
			f = alpha2(f);
		} while(f != d);
	}

	if(isOrbitEmbedded<VOLUME>())
	{
		Dart f = d;
		do
		{
			unsigned int vEmb = getEmbedding<VOLUME>(f) ;
			setDartEmbedding<VOLUME>(phi1(f), vEmb);
			setDartEmbedding<VOLUME>(phi2(f), vEmb);
			f = alpha2(f);
		} while(f != d);
	}

	return nd ;
}

bool EmbeddedMap3::uncutEdge(Dart d)
{
	if(Map3::uncutEdge(d))
	{
		//embed all darts from the old two edges to one of the two edge embedding
		if(isOrbitEmbedded<EDGE>())
		{
			setOrbitEmbedding<EDGE>(d, getEmbedding<EDGE>(d)) ;
		}
		return true ;
	}
	return false ;
}

Dart EmbeddedMap3::deleteEdge(Dart d)
{
	Dart v = Map3::deleteEdge(d) ;
	if(v != NIL)
	{
		if(isOrbitEmbedded<VOLUME>())
		{
			setOrbitEmbedding<VOLUME>(v, getEmbedding<VOLUME>(v)) ;
		}
	}
	return v;
}

bool EmbeddedMap3::edgeCanCollapse(Dart d)
{
//	if(isBoundaryVertex(d) || isBoundaryVertex(phi1(d)))
//		return false;
//
//	if(isBoundaryEdge(d))
//		return false;

	CellMarkerStore<VERTEX> mv(*this);

	Traversor3VVaE<TOPO_MAP> t3VVaE_v1(*this,d);
	for(Dart dit = t3VVaE_v1.begin() ; dit != t3VVaE_v1.end() ; dit = t3VVaE_v1.next())
	{
		mv.mark(dit);
	}

	Traversor3EW<TOPO_MAP> t3EW(*this,d);
	for(Dart dit = t3EW.begin() ; dit != t3EW.end() ; dit = t3EW.next())
	{
		mv.unmark(phi_1(dit));
		mv.unmark(phi_1(phi2(dit)));
	}

	Traversor3VVaE<TOPO_MAP> t3VVaE_v2(*this,phi2(d));
	for(Dart dit = t3VVaE_v2.begin() ; dit != t3VVaE_v2.end() ; dit = t3VVaE_v2.next())
	{
		if(mv.isMarked(dit))
			return false;
	}

	return true;
}

Dart EmbeddedMap3::collapseEdge(Dart d, bool delDegenerateVolumes)
{
	unsigned int vEmb = getEmbedding<VERTEX>(d) ;

	Dart d2 = phi2(phi_1(d)) ;
	Dart dd2 = phi2(phi_1(phi2(d))) ;

	Dart resV = Map3::collapseEdge(d, delDegenerateVolumes);

	if(resV != NIL)
	{
		if(isOrbitEmbedded<VERTEX>())
		{
			setOrbitEmbedding<VERTEX>(resV, vEmb);
		}

		if(isOrbitEmbedded<EDGE>())
		{
			setOrbitEmbedding<EDGE>(d2, getEmbedding<EDGE>(d2));
			setOrbitEmbedding<EDGE>(dd2, getEmbedding<EDGE>(dd2));
		}
	}

	return resV;
}

void EmbeddedMap3::splitFace(Dart d, Dart e)
{
	Dart dd = phi1(phi3(d));
	Dart ee = phi1(phi3(e));

	Map3::splitFace(d, e);

	if(isOrbitEmbedded<VERTEX>())
	{
		unsigned int vEmb1 = getEmbedding<VERTEX>(d) ;
		unsigned int vEmb2 = getEmbedding<VERTEX>(e) ;
		setDartEmbedding<VERTEX>(phi_1(e), vEmb1);
		setDartEmbedding<VERTEX>(phi_1(ee), vEmb1);
		setDartEmbedding<VERTEX>(phi_1(d), vEmb2);
		setDartEmbedding<VERTEX>(phi_1(dd), vEmb2);
	}

//	if(isOrbitEmbedded<EDGE>())
//	{
//		initOrbitEmbeddingNewCell<EDGE>(phi_1(d)) ;
//	}

	if(isOrbitEmbedded<FACE2>())
	{
		copyDartEmbedding<FACE2>(phi_1(d), d) ;
		setOrbitEmbeddingOnNewCell<FACE2>(e) ;
		copyCell<FACE2>(e, d) ;

		copyDartEmbedding<FACE2>(phi_1(dd), dd) ;
		setOrbitEmbeddingOnNewCell<FACE2>(ee) ;
		copyCell<FACE2>(ee, dd) ;
	}

	if(isOrbitEmbedded<FACE>())
	{
		unsigned int fEmb = getEmbedding<FACE>(d) ;
		setDartEmbedding<FACE>(phi_1(d), fEmb) ;
		setDartEmbedding<FACE>(phi_1(ee), fEmb) ;
		setOrbitEmbeddingOnNewCell<FACE>(e);
		copyCell<FACE>(e, d);
	}

	if(isOrbitEmbedded<VOLUME>())
	{
		unsigned int vEmb1 = getEmbedding<VOLUME>(d) ;
		setDartEmbedding<VOLUME>(phi_1(d),  vEmb1);
		setDartEmbedding<VOLUME>(phi_1(e),  vEmb1);

		unsigned int vEmb2 = getEmbedding<VOLUME>(dd) ;
		setDartEmbedding<VOLUME>(phi_1(dd),  vEmb2);
		setDartEmbedding<VOLUME>(phi_1(ee),  vEmb2);
	}
}

bool EmbeddedMap3::mergeFaces(Dart d)
{
	Dart d1 = phi1(d);

	if(Map3::mergeFaces(d))
	{
		if(isOrbitEmbedded<FACE2>())
		{
			setOrbitEmbedding<FACE2>(d1, getEmbedding<FACE2>(d1)) ;
		}

		if(isOrbitEmbedded<FACE>())
		{
			setOrbitEmbedding<FACE>(d1, getEmbedding<FACE>(d1)) ;
		}

		return true;
	}

	return false;
}

//!
/*!
 *
 */
Dart EmbeddedMap3::collapseFace(Dart d, bool delDegenerateVolumes)
{
	unsigned int vEmb = getEmbedding<VERTEX>(d) ;

	Dart resV = Map3::collapseFace(d, delDegenerateVolumes);

	if(resV != NIL)
	{
		if(isOrbitEmbedded<VERTEX>())
		{
			setOrbitEmbedding<VERTEX>(resV, vEmb);
		}
	}

	return resV;
}

void EmbeddedMap3::sewVolumes(Dart d, Dart e, bool withBoundary)
{
	if (!withBoundary)
	{
		Map3::sewVolumes(d, e, false) ;
		return ;
	}

	Map3::sewVolumes(d, e, withBoundary);

	// embed the vertex orbits from the oriented face with dart e
	// with vertex orbits value from oriented face with dart d
	if (isOrbitEmbedded<VERTEX>())
	{
		Dart it = d ;
		do
		{
			setOrbitEmbedding<VERTEX>(it, getEmbedding<VERTEX>(it)) ;
			it = phi1(it) ;
		} while(it != d) ;
	}

	// embed the new edge orbit with the old edge orbit value
	// for all the face
	if (isOrbitEmbedded<EDGE>())
	{
		Dart it = d ;
		do
		{
			setOrbitEmbedding<EDGE>(it, getEmbedding<EDGE>(it)) ;
			it = phi1(it) ;
		} while(it != d) ;
	}

	// embed the face orbit from the volume sewn
	if (isOrbitEmbedded<FACE>())
	{
		setOrbitEmbedding<FACE>(e, getEmbedding<FACE>(d)) ;
	}
}

void EmbeddedMap3::unsewVolumes(Dart d, bool withBoundary)
{
	if (!withBoundary)
	{
		Map3::unsewVolumes(d, false) ;
		return ;
	}

	Dart dd = alpha1(d);

	unsigned int fEmb = EMBNULL ;
	if(isOrbitEmbedded<FACE>())
		fEmb = getEmbedding<FACE>(d) ;

	Map3::unsewVolumes(d);

	Dart dit = d;
	do
	{
		// embed the unsewn vertex orbit with the vertex embedding if it is deconnected
		if(isOrbitEmbedded<VERTEX>())
		{
			if(!sameVertex(dit, dd))
			{
				setOrbitEmbedding<VERTEX>(dit, getEmbedding<VERTEX>(dit)) ;
				setOrbitEmbeddingOnNewCell<VERTEX>(dd);
				copyCell<VERTEX>(dd, dit);
			}
			else
			{
				setOrbitEmbedding<VERTEX>(dit, getEmbedding<VERTEX>(dit)) ;
			}
		}

		dd = phi_1(dd);

		// embed the unsewn edge with the edge embedding if it is deconnected
		if(isOrbitEmbedded<EDGE>())
		{
			if(!sameEdge(dit, dd))
			{
				setOrbitEmbeddingOnNewCell<EDGE>(dd);
				copyCell<EDGE>(dd, dit);
				copyDartEmbedding<EDGE>(phi3(dit), dit) ;
			}
			else
			{
				unsigned int eEmb = getEmbedding<EDGE>(dit) ;
				setDartEmbedding<EDGE>(phi3(dit), eEmb) ;
				setDartEmbedding<EDGE>(alpha_2(dit), eEmb) ;
			}
		}

		if(isOrbitEmbedded<FACE>())
		{
			setDartEmbedding<FACE>(phi3(dit), fEmb) ;
		}

		dit = phi1(dit);
	} while(dit != d);

	// embed the unsewn face with the face embedding
	if (isOrbitEmbedded<FACE>())
	{
		setOrbitEmbeddingOnNewCell<FACE>(dd);
		copyCell<FACE>(dd, d);
	}
}

bool EmbeddedMap3::mergeVolumes(Dart d, bool deleteFace)
{
	Dart d2 = phi2(d);

	if(Map3::mergeVolumes(d, deleteFace))
	{
		if (isOrbitEmbedded<VOLUME>())
		{
			setOrbitEmbedding<VOLUME>(d2, getEmbedding<VOLUME>(d2)) ;
		}
		return true;
	}
	return false;
}

void EmbeddedMap3::splitVolume(std::vector<Dart>& vd)
{
	Map3::splitVolume(vd);

	// follow the edge path a second time to embed the vertex, edge and volume orbits
	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		Dart dit = *it;
		Dart dit23 = phi3(phi2(dit));

		// embed the vertex embedded from the origin volume to the new darts
		if(isOrbitEmbedded<VERTEX>())
		{
			copyDartEmbedding<VERTEX>(dit23, dit);
			copyDartEmbedding<VERTEX>(phi2(dit), phi1(dit));
		}

		// embed the edge embedded from the origin volume to the new darts
		if(isOrbitEmbedded<EDGE2>())
		{
			setOrbitEmbeddingOnNewCell<EDGE2>(dit23) ;
			copyCell<EDGE2>(dit23, dit) ;

			copyDartEmbedding<EDGE2>(phi2(dit), dit);
		}

		// embed the edge embedded from the origin volume to the new darts
		if(isOrbitEmbedded<EDGE>())
		{
			unsigned int eEmb = getEmbedding<EDGE>(dit) ;
			setDartEmbedding<EDGE>(dit23, eEmb);
			setDartEmbedding<EDGE>(phi2(dit), eEmb);
		}

		// embed the volume embedded from the origin volume to the new darts
		if(isOrbitEmbedded<VOLUME>())
		{
			copyDartEmbedding<VOLUME>(phi2(dit), dit);
		}
	}

	if(isOrbitEmbedded<VOLUME>())
	{
		Dart v = vd.front() ;
		Dart v23 = phi3(phi2(v));
		setOrbitEmbeddingOnNewCell<VOLUME>(v23) ;
		copyCell<VOLUME>(v23, v) ;
	}
}

//! Split a volume into two volumes along a edge path and add the given face between
void EmbeddedMap3::splitVolumeWithFace(std::vector<Dart>& vd, Dart d)
{
	Map3::splitVolumeWithFace(vd,d);

	// follow the edge path a second time to embed the vertex, edge and volume orbits
	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		Dart dit = *it;
		Dart dit23 = phi3(phi2(dit));

		// embed the vertex embedded from the origin volume to the new darts
		if(isOrbitEmbedded<VERTEX>())
		{
			copyDartEmbedding<VERTEX>(dit23, dit);
			copyDartEmbedding<VERTEX>(phi2(dit), phi1(dit));
		}

		// embed the edge embedded from the origin volume to the new darts
		if(isOrbitEmbedded<EDGE2>())
		{
			setOrbitEmbeddingOnNewCell<EDGE2>(dit23) ;
			copyCell<EDGE2>(dit23, dit) ;

			copyDartEmbedding<EDGE2>(phi2(dit), dit);
		}

		// embed the edge embedded from the origin volume to the new darts
		if(isOrbitEmbedded<EDGE>())
		{
			unsigned int eEmb = getEmbedding<EDGE>(dit) ;
			setDartEmbedding<EDGE>(dit23, eEmb);
			setDartEmbedding<EDGE>(phi2(dit), eEmb);
		}

		// embed the volume embedded from the origin volume to the new darts
		if(isOrbitEmbedded<VOLUME>())
		{
			copyDartEmbedding<VOLUME>(phi2(dit), dit);
		}
	}

	if(isOrbitEmbedded<VOLUME>())
	{
		Dart v = vd.front() ;
		Dart v23 = phi3(phi2(v));
		setOrbitEmbeddingOnNewCell<VOLUME>(v23) ;
		copyCell<VOLUME>(v23, v) ;
	}
}

Dart EmbeddedMap3::collapseVolume(Dart d, bool delDegenerateVolumes)
{
	unsigned int vEmb = getEmbedding<VERTEX>(d) ;

	Dart resV = Map3::collapseVolume(d, delDegenerateVolumes);

	if(resV != NIL)
	{
		if(isOrbitEmbedded<VERTEX>())
		{
			setOrbitEmbedding<VERTEX>(resV, vEmb);
		}
	}

	return resV;
}

unsigned int EmbeddedMap3::closeHole(Dart d, bool forboundary)
{
	unsigned int nbF = Map3::closeHole(d, forboundary) ;

	DartMarkerStore mark(*this);	// Lock a marker

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(phi3(d));// Start with the face of d
	mark.markOrbit<FACE2>(phi3(d)) ;

	// For every face added to the list
	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		Dart it = visitedFaces[i] ;
		Dart f = it ;
		do
		{
			if(isOrbitEmbedded<VERTEX>())
			{
				copyDartEmbedding<VERTEX>(f, alpha1(f)) ;
			}
			if(isOrbitEmbedded<EDGE>())
			{
				copyDartEmbedding<EDGE>(f, phi3(f)) ;
			}
			if(isOrbitEmbedded<FACE>())
			{
				copyDartEmbedding<FACE>(f, phi3(f)) ;
			}

			Dart adj = phi2(f);	// Get adjacent face
			if (!mark.isMarked(adj))
			{
				visitedFaces.push_back(adj);	// Add it
				mark.markOrbit<FACE2>(adj) ;
			}

			f = phi1(f) ;
		} while(f != it) ;
	}

	return nbF ;
}

bool EmbeddedMap3::check()
{
    std::cout << "nb vertex orbits : " << getNbOrbits<VERTEX>() << std::endl ;
    std::cout << "nb vertex cells : " << m_attribs[VERTEX].size() << std::endl ;

    std::cout << "nb edge orbits : " << getNbOrbits<EDGE>() << std::endl ;
    std::cout << "nb edge cells : " << m_attribs[EDGE].size() << std::endl ;

    std::cout << "nb face orbits : " << getNbOrbits<FACE>() << std::endl ;
    std::cout << "nb face cells : " << m_attribs[FACE].size() << std::endl ;

    std::cout << "nb volume orbits : " << getNbOrbits<VOLUME>() << std::endl ;
    std::cout << "nb volume cells : " << m_attribs[VOLUME].size() << std::endl ;


	bool topo = Map3::check() ;
	if (!topo)
		return false ;

	std::cout << "Check: embedding begin" << std::endl ;

	for(Dart d = begin(); d != end(); next(d))
	{
		if(isOrbitEmbedded<VERTEX>())
		{
			if( getEmbedding<VERTEX>(d) != getEmbedding<VERTEX>(alpha1(d)))
			{
				std::cout << "Embedding Check : different embeddings on vertex (alpha1(d) != d)" << std::endl ;
				return false ;
			}
			if(getEmbedding<VERTEX>(d) != getEmbedding<VERTEX>(alpha2(d)) )
			{
				std::cout << "Embedding Check : different embeddings on vertex (alpha2(d) != d)" << std::endl ;
				return false ;
			}
		}

		if(isOrbitEmbedded<EDGE>())
		{
			if( getEmbedding<EDGE>(d) != getEmbedding<EDGE>(phi2(d)) ||
					getEmbedding<EDGE>(d) != getEmbedding<EDGE>(phi3(d)) )
			{
				std::cout << "Embedding Check : different embeddings on edge" << std::endl ;
				return false ;
			}
		}

		if (isOrbitEmbedded<FACE2>())
		{
			if (getEmbedding<FACE2>(d) != getEmbedding<FACE2>(phi1(d)))
			{
				CGoGNout << "Check: different embeddings on oriented face" << CGoGNendl ;
				return false ;
			}
		}

		if (isOrbitEmbedded<FACE>())
		{
			if( getEmbedding<FACE>(d) != getEmbedding<FACE>(phi1(d)) ||
					getEmbedding<FACE>(d) != getEmbedding<FACE>(phi3(d)) )
			{
				CGoGNout << "Check: different embeddings on face" << CGoGNendl ;
				return false ;
			}
		}

		if (isOrbitEmbedded<VOLUME>())
		{
			if( getEmbedding<VOLUME>(d) != getEmbedding<VOLUME>(phi1(d)) ||
					getEmbedding<VOLUME>(d) != getEmbedding<VOLUME>(phi2(d)) )
			{
				CGoGNout << "Check: different embeddings on volume" << CGoGNendl ;
				return false ;
			}
		}
	}

	std::cout << "Check: embedding ok" << std::endl ;

	return true ;
}

} // namespace CGoGN
