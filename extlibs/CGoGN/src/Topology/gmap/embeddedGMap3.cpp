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

#include "Topology/gmap/embeddedGMap3.h"
#include "Topology/generic/traversor/traversor3.h"

namespace CGoGN
{

Dart EmbeddedGMap3::deleteVertex(Dart d)
{
	Dart v = GMap3::deleteVertex(d) ;
	if(v != NIL)
	{
		if (isOrbitEmbedded<VOLUME>())
		{
			Algo::Topo::setOrbitEmbedding<VOLUME>(*this, v, getEmbedding<VOLUME>(v)) ;
		}
	}
	return v ;
}

Dart EmbeddedGMap3::cutEdge(Dart d)
{
	Dart nd = GMap3::cutEdge(d);

	if(isOrbitEmbedded<EDGE>())
	{
		// embed the new darts created in the cut edge
		unsigned int eEmb = getEmbedding<EDGE>(d) ;
		Dart e = d ;
		do
		{
			setDartEmbedding<EDGE>(beta0(e), eEmb) ;
			e = alpha2(e) ;
		} while(e != d) ;

		// embed a new cell for the new edge and copy the attributes' line (c) Lionel
		Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE>(*this, phi1(d)) ;
		Algo::Topo::copyCellAttributes<EDGE>(*this, phi1(d), d) ;
	}

	if(isOrbitEmbedded<FACE>())
	{
		Dart f = d;
		do
		{
			unsigned int fEmb = getEmbedding<FACE>(f) ;
			setDartEmbedding<FACE>(beta0(f), fEmb);
			setDartEmbedding<FACE>(phi1(f), fEmb);
			setDartEmbedding<FACE>(phi3(f), fEmb);
			setDartEmbedding<FACE>(beta1(phi3(f)), fEmb);
			f = alpha2(f);
		} while(f != d);
	}

	if(isOrbitEmbedded<VOLUME>())
	{
		Dart f = d;
		do
		{
			unsigned int vEmb = getEmbedding<VOLUME>(f) ;
			setDartEmbedding<VOLUME>(beta0(f), vEmb);
			setDartEmbedding<VOLUME>(phi1(f), vEmb);
			setDartEmbedding<VOLUME>(phi2(f), vEmb);
			setDartEmbedding<VOLUME>(beta1(phi2(f)), vEmb);
			f = alpha2(f);
		} while(f != d);
	}

	return nd ;
}

bool EmbeddedGMap3::uncutEdge(Dart d)
{
	if(GMap3::uncutEdge(d))
	{
		//embed all darts from the old two edges to one of the two edge embedding
		if(isOrbitEmbedded<EDGE>())
		{
			Algo::Topo::setOrbitEmbedding<EDGE>(*this, d, getEmbedding<EDGE>(d)) ;
		}
		return true ;
	}
	return false ;
}

Dart EmbeddedGMap3::deleteEdge(Dart d)
{
	Dart v = GMap3::deleteEdge(d) ;
	if(v != NIL)
	{
		if (isOrbitEmbedded<VOLUME>())
		{
			Algo::Topo::setOrbitEmbedding<VOLUME>(*this, v, getEmbedding<VOLUME>(v)) ;
		}
	}
	return v ;
}

void EmbeddedGMap3::splitFace(Dart d, Dart e)
{
	Dart dd = beta1(beta3(d));
	Dart ee = beta1(beta3(e));

	GMap3::splitFace(d, e);

	if(isOrbitEmbedded<VERTEX>())
	{
		unsigned int vEmb1 = getEmbedding<VERTEX>(d) ;
		unsigned int vEmb2 = getEmbedding<VERTEX>(e) ;

		setDartEmbedding<VERTEX>(beta1(d), vEmb1);
		setDartEmbedding<VERTEX>(beta2(beta1(d)), vEmb1);
		setDartEmbedding<VERTEX>(beta1(beta2(beta1(d))), vEmb1);
		setDartEmbedding<VERTEX>(beta1(dd), vEmb1);
		setDartEmbedding<VERTEX>(beta2(beta1(dd)), vEmb1);
		setDartEmbedding<VERTEX>(beta1(beta2(beta1(dd))), vEmb1);

		setDartEmbedding<VERTEX>(beta1(e), vEmb2);
		setDartEmbedding<VERTEX>(beta2(beta1(e)), vEmb2);
		setDartEmbedding<VERTEX>(beta1(beta2(beta1(e))), vEmb2);
		setDartEmbedding<VERTEX>(beta1(ee), vEmb2);
		setDartEmbedding<VERTEX>(beta2(beta1(ee)), vEmb2);
		setDartEmbedding<VERTEX>(beta1(beta2(beta1(ee))), vEmb2);
	}

	if(isOrbitEmbedded<EDGE>())
	{
		Algo::Topo::setOrbitEmbedding<EDGE>(*this, beta1(d), getEmbedding<EDGE>(beta1(d))) ;
		Algo::Topo::setOrbitEmbedding<EDGE>(*this, beta1(e), getEmbedding<EDGE>(beta1(e))) ;

		Algo::Topo::setOrbitEmbedding<EDGE>(*this, d, getEmbedding<EDGE>(d)) ;
		Algo::Topo::setOrbitEmbedding<EDGE>(*this, e, getEmbedding<EDGE>(e)) ;
		Algo::Topo::setOrbitEmbedding<EDGE>(*this, beta1(beta2(beta1(d))), getEmbedding<EDGE>(beta1(beta2(beta1(d))))) ;
		Algo::Topo::setOrbitEmbedding<EDGE>(*this, beta1(beta2(beta1(e))), getEmbedding<EDGE>(beta1(beta2(beta1(e))))) ;
	}


	if(isOrbitEmbedded<FACE>())
	{
		unsigned int fEmb = getEmbedding<FACE>(d) ;
		setDartEmbedding<FACE>(beta1(d), fEmb) ;
		setDartEmbedding<FACE>(beta0(beta1(d)), fEmb) ;
		setDartEmbedding<FACE>(beta1(beta0(beta1(d))), fEmb) ;
		setDartEmbedding<FACE>(beta1(ee), fEmb) ;
		setDartEmbedding<FACE>(beta0(beta1(ee)), fEmb) ;
		setDartEmbedding<FACE>(beta1(beta0(beta1(ee))), fEmb) ;
		Algo::Topo::setOrbitEmbeddingOnNewCell<FACE>(*this, e);
		Algo::Topo::copyCellAttributes<FACE>(*this, e, d);
	}

	if(isOrbitEmbedded<VOLUME>())
	{
		unsigned int vEmb1 = getEmbedding<VOLUME>(d) ;
		setDartEmbedding<VOLUME>(beta1(d),  vEmb1);
		setDartEmbedding<VOLUME>(beta0(beta1(d)),  vEmb1);
		setDartEmbedding<VOLUME>(beta1(beta0(beta1(d))), vEmb1) ;
		setDartEmbedding<VOLUME>(beta1(e),  vEmb1);
		setDartEmbedding<VOLUME>(beta0(beta1(e)),  vEmb1);
		setDartEmbedding<VOLUME>(beta1(beta0(beta1(e))), vEmb1) ;

		unsigned int vEmb2 = getEmbedding<VOLUME>(dd) ;
		setDartEmbedding<VOLUME>(beta1(dd),  vEmb2);
		setDartEmbedding<VOLUME>(beta0(beta1(dd)),  vEmb2);
		setDartEmbedding<VOLUME>(beta1(beta0(beta1(dd))), vEmb2) ;
		setDartEmbedding<VOLUME>(beta1(ee),  vEmb2);
		setDartEmbedding<VOLUME>(beta0(beta1(ee)),  vEmb2);
		setDartEmbedding<VOLUME>(beta1(beta0(beta1(ee))), vEmb2) ;
	}
}

void EmbeddedGMap3::sewVolumes(Dart d, Dart e, bool withBoundary)
{
	// topological sewing
	GMap3::sewVolumes(d, e, withBoundary);

	// embed the vertex orbits from the oriented face with dart e
	// with vertex orbits value from oriented face with dart d
	if (isOrbitEmbedded<VERTEX>())
	{
		Dart it = d ;
		do
		{
			Algo::Topo::setOrbitEmbedding<VERTEX>(*this, it, getEmbedding<VERTEX>(it)) ;
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
			Algo::Topo::setOrbitEmbedding<EDGE>(*this, it, getEmbedding<EDGE>(it)) ;
			it = phi1(it) ;
		} while(it != d) ;
	}

	// embed the face orbit from the volume sewn
	if (isOrbitEmbedded<FACE>())
	{
		Algo::Topo::setOrbitEmbedding<FACE>(*this, e, getEmbedding<FACE>(d)) ;
	}
}

void EmbeddedGMap3::unsewVolumes(Dart d)
{
	Dart dd = alpha1(d);

	unsigned int fEmb = EMBNULL ;
	if(isOrbitEmbedded<FACE>())
		fEmb = getEmbedding<FACE>(d) ;

	GMap3::unsewVolumes(d);

	Dart dit = d;
	do
	{
		// embed the unsewn vertex orbit with the vertex embedding if it is deconnected
		if(isOrbitEmbedded<VERTEX>())
		{
			if(!sameVertex(dit, dd))
			{
				Algo::Topo::setOrbitEmbedding<VERTEX>(*this, dit, getEmbedding<VERTEX>(dit)) ;
				Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(*this, dd);
				Algo::Topo::copyCellAttributes<VERTEX>(*this, dd, dit);
			}
			else
			{
				Algo::Topo::setOrbitEmbedding<VERTEX>(*this, dit, getEmbedding<VERTEX>(dit)) ;
			}
		}

		dd = phi_1(dd);

		// embed the unsewn edge with the edge embedding if it is deconnected
		if(isOrbitEmbedded<EDGE>())
		{
			if(!sameEdge(dit, dd))
			{
				Algo::Topo::setOrbitEmbedding<EDGE>(*this, dit, getEmbedding<EDGE>(dit)) ;
				Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE>(*this, dd);
				Algo::Topo::copyCellAttributes<EDGE>(*this, dd, dit);
			}
			else
			{
				Algo::Topo::setOrbitEmbedding<EDGE>(*this, dit, getEmbedding<EDGE>(dit)) ;
			}
		}

		if(isOrbitEmbedded<FACE>())
		{
			setDartEmbedding<FACE>(beta3(dit), fEmb) ;
			setDartEmbedding<FACE>(beta0(beta3(dit)), fEmb) ;
		}

		dit = phi1(dit);
	} while(dit != d);

	// embed the unsewn face with the face embedding
	if (isOrbitEmbedded<FACE>())
	{
		Algo::Topo::setOrbitEmbeddingOnNewCell<FACE>(*this, dd);
		Algo::Topo::copyCellAttributes<FACE>(*this, dd, d);
	}
}

bool EmbeddedGMap3::mergeVolumes(Dart d)
{
	Dart d2 = phi2(d);

	if(GMap3::mergeVolumes(d))
	{
		if (isOrbitEmbedded<VOLUME>())
		{
			Algo::Topo::setOrbitEmbedding<VOLUME>(*this, d2, getEmbedding<VOLUME>(d2)) ;
		}
		return true;
	}
	return false;
}

void EmbeddedGMap3::splitVolume(std::vector<Dart>& vd)
{
	GMap3::splitVolume(vd);

	// follow the edge path a second time to embed the vertex, edge and volume orbits
	for(std::vector<Dart>::iterator it = vd.begin() ; it != vd.end() ; ++it)
	{
		Dart dit = *it;

		// embed the vertex embedded from the origin volume to the new darts
		if(isOrbitEmbedded<VERTEX>())
		{
			unsigned int vEmb = getEmbedding<VERTEX>(dit) ;
			setDartEmbedding<VERTEX>(beta2(dit), vEmb);
			setDartEmbedding<VERTEX>(beta3(beta2(dit)), vEmb);
			setDartEmbedding<VERTEX>(beta1(beta2(dit)), vEmb);
			setDartEmbedding<VERTEX>(beta3(beta1(beta2(dit))), vEmb);
		}

		// embed the edge embedded from the origin volume to the new darts
		if(isOrbitEmbedded<EDGE>())
		{
			unsigned int eEmb = getEmbedding<EDGE>(dit) ;
			setDartEmbedding<EDGE>(beta2(dit), eEmb);
			setDartEmbedding<EDGE>(beta3(beta2(dit)), eEmb);
			setDartEmbedding<EDGE>(beta0(beta2(dit)), eEmb);
			setDartEmbedding<EDGE>(beta0(beta3(beta2(dit))), eEmb);
		}

		// embed the volume embedded from the origin volume to the new darts
		if(isOrbitEmbedded<VOLUME>())
		{
			unsigned int vEmb = getEmbedding<VOLUME>(dit) ;
			setDartEmbedding<VOLUME>(beta2(dit), vEmb);
			setDartEmbedding<VOLUME>(beta0(beta2(dit)), vEmb);
		}
	}

	if(isOrbitEmbedded<VOLUME>())
	{
		Dart v = vd.front() ;
		Dart v23 = alpha2(v) ;
		Algo::Topo::setOrbitEmbeddingOnNewCell<VOLUME>(*this, v23) ;
		Algo::Topo::copyCellAttributes<VOLUME>(*this, v23, v) ;
	}
}

unsigned int EmbeddedGMap3::closeHole(Dart d, bool forboundary)
{
	unsigned int nbF = GMap3::closeHole(d, forboundary) ;

	DartMarkerStore<EmbeddedGMap3> mark(*this);	// Lock a marker

	std::vector<Dart> visitedFaces;	// Faces that are traversed
	visitedFaces.reserve(1024) ;
	visitedFaces.push_back(beta3(d));// Start with the face of d
	mark.markOrbit<FACE>(beta3(d)) ;

	// For every face added to the list
	for(unsigned int i = 0; i < visitedFaces.size(); ++i)
	{
		Dart f = visitedFaces[i] ;
		do
		{
			if(isOrbitEmbedded<VERTEX>())
			{
				unsigned int vEmb = getEmbedding<VERTEX>(beta3(f)) ;
				setDartEmbedding<VERTEX>(f, vEmb) ;
				setDartEmbedding<VERTEX>(beta1(f), vEmb) ;
			}
			if(isOrbitEmbedded<EDGE>())
			{
				unsigned int eEmb = getEmbedding<EDGE>(beta3(f)) ;
				setDartEmbedding<EDGE>(f, eEmb) ;
				setDartEmbedding<EDGE>(beta0(f), eEmb) ;
			}
			if(isOrbitEmbedded<FACE>())
			{
				unsigned int fEmb = getEmbedding<FACE>(beta3(f)) ;
				setDartEmbedding<FACE>(f, fEmb) ;
				setDartEmbedding<FACE>(beta0(f), fEmb) ;
			}

			Dart adj = beta2(f);	// Get adjacent face
			if (!mark.isMarked(adj))
			{
				visitedFaces.push_back(adj);	// Add it
				mark.markOrbit<FACE>(adj) ;
			}

			f = phi1(f) ;
		} while(f != visitedFaces[i]) ;
	}

	return nbF ;
}

bool EmbeddedGMap3::check()
{
	bool topo = GMap3::check() ;
	if (!topo)
		return false ;

	CGoGNout << "Check: embedding begin" << CGoGNendl ;

	for(Dart d = begin(); d != end(); next(d))
	{
		if(isOrbitEmbedded<VERTEX>())
		{
			if( getEmbedding<VERTEX>(d) != getEmbedding<VERTEX>(beta1(d)) ||
				getEmbedding<VERTEX>(d) != getEmbedding<VERTEX>(beta2(d)) ||
				getEmbedding<VERTEX>(d) != getEmbedding<VERTEX>(beta3(d)) )
			{
				std::cout << "Embedding Check : different embeddings on vertex" << std::endl ;
				return false ;
			}
		}

		if(isOrbitEmbedded<EDGE>())
		{
			if( getEmbedding<EDGE>(d) != getEmbedding<EDGE>(beta0(d)) ||
				getEmbedding<EDGE>(d) != getEmbedding<EDGE>(beta2(d)) ||
				getEmbedding<EDGE>(d) != getEmbedding<EDGE>(beta3(d)) )
			{
				std::cout << "Embedding Check : different embeddings on edge" << std::endl ;
				return false ;
			}
		}

		if (isOrbitEmbedded<FACE>())
		{
			if( getEmbedding<FACE>(d) != getEmbedding<FACE>(beta0(d)) ||
				getEmbedding<FACE>(d) != getEmbedding<FACE>(beta1(d)) ||
				getEmbedding<FACE>(d) != getEmbedding<FACE>(beta3(d)) )
			{
				CGoGNout << "Check: different embeddings on face" << CGoGNendl ;
				return false ;
			}
		}

		if (isOrbitEmbedded<VOLUME>())
		{
			if( getEmbedding<VOLUME>(d) != getEmbedding<VOLUME>(beta0(d)) ||
				getEmbedding<VOLUME>(d) != getEmbedding<VOLUME>(beta1(d)) ||
				getEmbedding<VOLUME>(d) != getEmbedding<VOLUME>(beta2(d)) )
			{
				CGoGNout << "Check: different embeddings on volume" << CGoGNendl ;
				return false ;
			}
		}
	}

	CGoGNout << "Check: embedding ok" << CGoGNendl ;

	std::cout << "nb vertex orbits : " << Algo::Topo::getNbOrbits<VERTEX>(*this) << std::endl ;
    std::cout << "nb vertex cells : " << m_attribs[VERTEX].size() << std::endl ;

	std::cout << "nb edge orbits : " << Algo::Topo::getNbOrbits<EDGE>(*this) << std::endl ;
    std::cout << "nb edge cells : " << m_attribs[EDGE].size() << std::endl ;

	std::cout << "nb face orbits : " << Algo::Topo::getNbOrbits<FACE>(*this) << std::endl ;
    std::cout << "nb face cells : " << m_attribs[FACE].size() << std::endl ;

	std::cout << "nb volume orbits : " << Algo::Topo::getNbOrbits<VOLUME>(*this) << std::endl ;
    std::cout << "nb volume cells : " << m_attribs[VOLUME].size() << std::endl ;

	return true ;
}

} // namespace CGoGN
