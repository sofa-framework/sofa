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

#include "Topology/gmap/embeddedGMap2.h"
#include "Topology/generic/traversor/traversor2.h"

namespace CGoGN
{

Dart EmbeddedGMap2::newFace(unsigned int nbEdges, bool withBoundary)
{
	Dart d = GMap2::newFace(nbEdges, withBoundary);

	if(withBoundary)
	{
		if (isOrbitEmbedded<VERTEX>())
		{
			Traversor2FV<EmbeddedGMap2> t(*this, d);
			for(Dart it = t.begin(); it != t.end(); it = t.next())
				Algo::Topo::initOrbitEmbeddingOnNewCell<VERTEX>(*this, it) ;
		}

		if(isOrbitEmbedded<EDGE>())
		{
			Traversor2FE<EmbeddedGMap2> t(*this, d);
			for(Dart it = t.begin(); it != t.end(); it = t.next())
				Algo::Topo::initOrbitEmbeddingOnNewCell<EDGE>(*this, it) ;
		}

		if(isOrbitEmbedded<FACE>())
		{
			Algo::Topo::initOrbitEmbeddingOnNewCell<FACE>(*this, d) ;
			Algo::Topo::initOrbitEmbeddingOnNewCell<FACE>(*this, phi2(d)) ;
		}
	}

//	else
//	{
//		if (isOrbitEmbedded<VERTEX>())
//		{
//			Traversor2FV<EmbeddedGMap2> t(*this, d);
//			for(Dart it = t.begin(); it != t.end(); it = t.next())
//				initOrbitEmbeddingNewCell<VERTEX>(it) ;
//		}

//		if(isOrbitEmbedded<FACE>())
//			initOrbitEmbeddingNewCell<FACE>(d) ;

//	}
	return d ;
}


void EmbeddedGMap2::splitVertex(Dart d, Dart e)
{
	Dart dd = phi2(d) ;
	Dart ee = phi2(e) ;

	GMap2::splitVertex(d, e) ;

	if (isOrbitEmbedded<VERTEX>())
	{
		unsigned int vEmb = getEmbedding<VERTEX>(d) ;
		setDartEmbedding<VERTEX>(phi1(dd), vEmb) ;
		setDartEmbedding<VERTEX>(beta2(phi1(dd)), vEmb) ;
		setDartEmbedding<VERTEX>(beta0(dd), vEmb) ;

		Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(*this, e) ;
		Algo::Topo::copyCellAttributes<VERTEX>(*this, e, d) ;
	}

	if(isOrbitEmbedded<EDGE>())
	{
		Algo::Topo::initOrbitEmbeddingOnNewCell<EDGE>(*this, phi1(dd)) ;
	}

	if(isOrbitEmbedded<FACE>())
	{
		unsigned int f1Emb = getEmbedding<FACE>(dd) ;
		setDartEmbedding<FACE>(phi1(dd), f1Emb) ;
		setDartEmbedding<FACE>(beta0(phi1(dd)), f1Emb) ;
		unsigned int f2Emb = getEmbedding<FACE>(ee) ;
		setDartEmbedding<FACE>(phi1(ee), f2Emb) ;
		setDartEmbedding<FACE>(beta0(phi1(ee)), f2Emb) ;
	}
}

Dart EmbeddedGMap2::deleteVertex(Dart d)
{
	Dart f = GMap2::deleteVertex(d) ;
	if(f != NIL)
	{
		if (isOrbitEmbedded<FACE>())
		{
			Algo::Topo::setOrbitEmbedding<FACE>(*this, f, getEmbedding<FACE>(f)) ;
		}
	}
	return f ;
}

Dart EmbeddedGMap2::cutEdge(Dart d)
{
	Dart nd = GMap2::cutEdge(d) ;

	if(isOrbitEmbedded<VERTEX>())
	{
		Algo::Topo::initOrbitEmbeddingOnNewCell<VERTEX>(*this, nd) ;
	}

	if (isOrbitEmbedded<EDGE>())
	{
		unsigned int eEmb = getEmbedding<EDGE>(d) ;
		setDartEmbedding<EDGE>(phi2(d), eEmb) ;
		setDartEmbedding<EDGE>(beta0(d), eEmb) ;
		Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE>(*this, nd) ;
		Algo::Topo::copyCellAttributes<EDGE>(*this, nd, d) ;
	}

	if(isOrbitEmbedded<FACE>())
	{
		unsigned int f1Emb = getEmbedding<FACE>(d) ;
		setDartEmbedding<FACE>(phi1(d), f1Emb) ;
		setDartEmbedding<FACE>(beta0(d), f1Emb) ;
		Dart e = phi2(nd) ;
		unsigned int f2Emb = getEmbedding<FACE>(d) ;
		setDartEmbedding<FACE>(phi1(e), f2Emb) ;
		setDartEmbedding<FACE>(beta0(e), f2Emb) ;
	}

	return nd ;
}

bool EmbeddedGMap2::uncutEdge(Dart d)
{
	if(GMap2::uncutEdge(d))
	{
		if(isOrbitEmbedded<EDGE>())
		{
			unsigned int eEmb = getEmbedding<EDGE>(d) ;
			setDartEmbedding<EDGE>(phi2(d), eEmb) ;
			setDartEmbedding<EDGE>(beta0(d), eEmb) ;
		}
		return true ;
	}
	return false ;
}

bool EmbeddedGMap2::edgeCanCollapse(Dart d)
{
	if(isBoundaryVertex(d) || isBoundaryVertex(phi1(d)))
		return false ;

	unsigned int val_v1 = vertexDegree(d) ;
	unsigned int val_v2 = vertexDegree(phi1(d)) ;

	if(val_v1 + val_v2 < 8 || val_v1 + val_v2 > 14)
		return false ;

	if(faceDegree(d) == 3)
	{
		if(vertexDegree(phi_1(d)) < 4)
			return false ;
	}

	Dart dd = phi2(d) ;
	if(faceDegree(dd) == 3)
	{
		if(vertexDegree(phi_1(dd)) < 4)
			return false ;
	}

	// Check vertex sharing condition
	std::vector<unsigned int> vu1 ;
	vu1.reserve(32) ;
	Dart vit1 = alpha1(alpha1(d)) ;
	Dart end = phi1(dd) ;
	do
	{
		unsigned int ve = getEmbedding<VERTEX>(phi2(vit1)) ;
		vu1.push_back(ve) ;
		vit1 = alpha1(vit1) ;
	} while(vit1 != end) ;
	end = phi1(d) ;
	Dart vit2 = alpha1(alpha1(dd)) ;
	do
	{
		unsigned int ve = getEmbedding<VERTEX>(phi2(vit2)) ;
		std::vector<unsigned int>::iterator it = std::find(vu1.begin(), vu1.end(), ve) ;
		if(it != vu1.end())
			return false ;
		vit2 = alpha1(vit2) ;
	} while(vit2 != end) ;

	return true ;
}

Dart EmbeddedGMap2::collapseEdge(Dart d, bool delDegenerateFaces)
{
	unsigned int vEmb = EMBNULL ;
	if (isOrbitEmbedded<VERTEX>())
	{
		vEmb = getEmbedding<VERTEX>(d) ;
	}

	Dart dV = GMap2::collapseEdge(d, delDegenerateFaces);

	if (isOrbitEmbedded<VERTEX>())
	{
		Algo::Topo::setOrbitEmbedding<VERTEX>(*this, dV, vEmb) ;
	}

	return dV ;
}

bool EmbeddedGMap2::flipEdge(Dart d)
{
	if(GMap2::flipEdge(d))
	{
		Dart e = phi2(d) ;

		if (isOrbitEmbedded<VERTEX>())
		{
			unsigned int v1Emb = getEmbedding<VERTEX>(beta1(d)) ;
			setDartEmbedding<VERTEX>(d, v1Emb) ;
			setDartEmbedding<VERTEX>(beta2(d), v1Emb) ;
			unsigned int v2Emb = getEmbedding<VERTEX>(beta1(e)) ;
			setDartEmbedding<VERTEX>(e, v2Emb) ;
			setDartEmbedding<VERTEX>(beta2(e), v2Emb) ;
		}
		if (isOrbitEmbedded<FACE>())
		{
			unsigned int f1Emb = getEmbedding<FACE>(d) ;
			setDartEmbedding<FACE>(phi_1(d), f1Emb) ;
			setDartEmbedding<FACE>(beta1(d), f1Emb) ;
			unsigned int f2Emb = getEmbedding<FACE>(e) ;
			setDartEmbedding<FACE>(phi_1(e), f2Emb) ;
			setDartEmbedding<FACE>(beta1(e), f2Emb) ;
		}
		return true ;
	}
	return false ;
}

bool EmbeddedGMap2::flipBackEdge(Dart d)
{
	if(GMap2::flipBackEdge(d))
	{
		Dart e = phi2(d) ;

		if (isOrbitEmbedded<VERTEX>())
		{
			unsigned int v1Emb = getEmbedding<VERTEX>(beta1(d)) ;
			setDartEmbedding<VERTEX>(d, v1Emb) ;
			setDartEmbedding<VERTEX>(beta2(d), v1Emb) ;
			unsigned int v2Emb = getEmbedding<VERTEX>(beta1(e)) ;
			setDartEmbedding<VERTEX>(e, v2Emb) ;
			setDartEmbedding<VERTEX>(beta2(e), v2Emb) ;
		}

		if (isOrbitEmbedded<FACE>())
		{
			unsigned int f1Emb = getEmbedding<FACE>(d) ;
			setDartEmbedding<FACE>(phi1(d), f1Emb) ;
			setDartEmbedding<FACE>(beta0(phi1(d)), f1Emb) ;
			unsigned int f2Emb = getEmbedding<FACE>(e) ;
			setDartEmbedding<FACE>(phi1(e), f2Emb) ;
			setDartEmbedding<FACE>(beta0(phi1(e)), f2Emb) ;
		}
		return true ;
	}
	return false ;
}

//void EmbeddedGMap2::insertEdgeInVertex(Dart d, Dart e)
//{
//	GMap2::insertEdgeInVertex(d, e);
//
//	if (isOrbitEmbedded<VERTEX>())
//	{
//		copyDartEmbedding<VERTEX>(e, d) ;
//		copyDartEmbedding<VERTEX>(beta2(e), d) ;
//	}
//
//	if (isOrbitEmbedded<FACE>())
//	{
//		if(!sameFace(d,e))
//		{
//			setOrbitEmbeddingOnNewCell<FACE>(e);
//			copyCell<FACE>(e, d) ;
//		}
//		else
//		{
//			setOrbitEmbedding<FACE>(d, getEmbedding<FACE>(d)) ;
//		}
//	}
//}
//
//void EmbeddedGMap2::removeEdgeFromVertex(Dart d)
//{
//	Dart dPrev = alpha_1(d);
//
//	GMap2::removeEdgeFromVertex(d);
//
//	if (isOrbitEmbedded<VERTEX>())
//	{
//		setOrbitEmbeddingOnNewCell<VERTEX>(d);
//		copyCell<VERTEX>(d, dPrev);
//	}
//
//	if (isOrbitEmbedded<FACE>())
//	{
//		if(!sameFace(d, dPrev))
//		{
//			setOrbitEmbeddingOnNewCell<FACE>(d);
//			copyCell<FACE>(d, dPrev) ;
//		}
//		else
//		{
//			setOrbitEmbedding<FACE>(d, getEmbedding<FACE>(d)) ;
//		}
//	}
//}

void EmbeddedGMap2::sewFaces(Dart d, Dart e, bool withBoundary)
{
	// for fixed point construction (import & primitives)
	if (!withBoundary)
	{
		GMap2::sewFaces(d, e, false) ;
		if(isOrbitEmbedded<EDGE>())
		{
			Algo::Topo::initOrbitEmbeddingOnNewCell<EDGE>(*this, d) ;
		}
		return ;
	}

	GMap2::sewFaces(d, e, true) ;

	if (isOrbitEmbedded<VERTEX>())
	{
		Algo::Topo::setOrbitEmbedding<VERTEX>(*this, d, getEmbedding<VERTEX>(d)) ;
		Algo::Topo::setOrbitEmbedding<VERTEX>(*this, e, getEmbedding<VERTEX>(beta0(d))) ;
	}

	if (isOrbitEmbedded<EDGE>())
	{
		Algo::Topo::setOrbitEmbedding<EDGE>(*this, e, getEmbedding<EDGE>(d)) ;
	}
}

void EmbeddedGMap2::unsewFaces(Dart d)
{
	Dart e = beta2(d);
	GMap2::unsewFaces(d);

	if (isOrbitEmbedded<VERTEX>())
	{
		if(!sameVertex(d,e))
		{
			Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(*this, e);
			Algo::Topo::copyCellAttributes<VERTEX>(*this, e, d);
		}

		d = beta0(d);
		e = beta0(e);

		if(!sameVertex(d,e))
		{
			Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(*this, e);
			Algo::Topo::copyCellAttributes<VERTEX>(*this, e, d);
		}
	}

	if (isOrbitEmbedded<EDGE>())
	{
		Algo::Topo::setOrbitEmbeddingOnNewCell<EDGE>(*this, e);
		Algo::Topo::copyCellAttributes<EDGE>(*this, e, d);
	}
}

bool EmbeddedGMap2::collapseDegeneratedFace(Dart d)
{
	Dart e = beta2(d) ;
	bool updateEdgeEmb = false ;
	if(phi1(d) != d)
		updateEdgeEmb = true ;

	if(GMap2::collapseDegeneratedFace(d))
	{
		if (isOrbitEmbedded<EDGE>() && updateEdgeEmb)
		{
			unsigned int eEmb = getEmbedding<EDGE>(e) ;
			setDartEmbedding<EDGE>(beta2(e), eEmb) ;
			setDartEmbedding<EDGE>(phi2(e), eEmb) ;
		}
		return true ;
	}
	return false ;
}

void EmbeddedGMap2::splitFace(Dart d, Dart e)
{
	GMap2::splitFace(d, e) ;

	if (isOrbitEmbedded<VERTEX>())
	{
		unsigned int v1Emb = getEmbedding<VERTEX>(d) ;
		setDartEmbedding<VERTEX>(phi_1(e), v1Emb) ;
		setDartEmbedding<VERTEX>(beta1(d), v1Emb) ;
		setDartEmbedding<VERTEX>(beta1(phi_1(e)), v1Emb) ;
		unsigned int v2Emb = getEmbedding<VERTEX>(e) ;
		setDartEmbedding<VERTEX>(phi_1(d), v2Emb) ;
		setDartEmbedding<VERTEX>(beta1(e), v2Emb) ;
		setDartEmbedding<VERTEX>(beta1(phi_1(d)), v2Emb) ;
	}

	if(isOrbitEmbedded<EDGE>())
	{
		Algo::Topo::initOrbitEmbeddingOnNewCell<EDGE>(*this, beta1(d)) ;
	}

	if (isOrbitEmbedded<FACE>())
	{
		unsigned int fEmb = getEmbedding<FACE>(d) ;
		setDartEmbedding<FACE>(phi_1(d), fEmb) ;
		setDartEmbedding<FACE>(beta1(phi_1(d)), fEmb) ;
		Algo::Topo::setOrbitEmbeddingOnNewCell<FACE>(*this, e) ;
		Algo::Topo::copyCellAttributes<FACE>(*this, e, d) ;
	}
}

bool EmbeddedGMap2::mergeFaces(Dart d)
{
	Dart dNext = phi1(d) ;

	if(GMap2::mergeFaces(d))
	{
		if (isOrbitEmbedded<FACE>())
		{
			Algo::Topo::setOrbitEmbedding<FACE>(*this, dNext, getEmbedding<FACE>(dNext)) ;
		}
		return true ;
	}
	return false ;
}

bool EmbeddedGMap2::mergeVolumes(Dart d, Dart e)
{
	std::vector<Dart> darts ;
	std::vector<unsigned int> vEmb ;
	std::vector<unsigned int> eEmb ;
	darts.reserve(32) ;
	vEmb.reserve(32) ;
	eEmb.reserve(32) ;

	Dart fit = d ;
	do
	{
		darts.push_back(phi2(fit)) ;

		if (isOrbitEmbedded<VERTEX>())
		{
			vEmb.push_back(getEmbedding<VERTEX>(phi2(fit))) ;
		}

		if (isOrbitEmbedded<EDGE>())
		{
			eEmb.push_back(getEmbedding<EDGE>(fit)) ;
		}

		fit = phi1(fit) ;
	} while (fit != d) ;

	if(GMap2::mergeVolumes(d, e))
	{
		for(unsigned int i = 0; i < darts.size(); ++i)
		{
			if (isOrbitEmbedded<VERTEX>())
			{
				Algo::Topo::setOrbitEmbedding<VERTEX>(*this, darts[i], vEmb[i]) ;
			}

			if (isOrbitEmbedded<EDGE>())
			{
				Algo::Topo::setOrbitEmbedding<EDGE>(*this, darts[i], eEmb[i]) ;
			}
		}
		return true ;
	}
	return false ;
}

unsigned int EmbeddedGMap2::closeHole(Dart d, bool forboundary)
{
	unsigned int nbE = GMap2::closeHole(d) ;
	Dart dd = phi2(d) ;
	Dart it = dd ;
	do
	{
		if (isOrbitEmbedded<VERTEX>())
		{
			copyDartEmbedding<VERTEX>(it, beta2(it)) ;
			copyDartEmbedding<VERTEX>(beta0(it), phi2(it)) ;
		}
		if (isOrbitEmbedded<EDGE>())
		{
			unsigned int eEmb = getEmbedding<EDGE>(beta2(it)) ;
			setDartEmbedding<EDGE>(it, eEmb) ;
			setDartEmbedding<EDGE>(beta0(it), eEmb) ;
		}
		it = phi1(it) ;
	} while(it != dd) ;

	if(isOrbitEmbedded<FACE>())
	{
		Algo::Topo::initOrbitEmbeddingOnNewCell<FACE>(*this, dd) ;
	}

	return nbE ;
}

bool EmbeddedGMap2::check()
{
	bool topo = GMap2::check() ;
	if (!topo)
		return false ;

	CGoGNout << "Check: embedding begin" << CGoGNendl ;
	for(Dart d = begin(); d != end(); next(d))
	{
		if (isOrbitEmbedded<VERTEX>())
		{
			if (getEmbedding<VERTEX>(d) != getEmbedding<VERTEX>(alpha1(d)))
			{
				CGoGNout << "Check: different embeddings on vertex" << CGoGNendl ;
				return false ;
			}
		}

		if (isOrbitEmbedded<EDGE>())
		{
			if (getEmbedding<EDGE>(d) != getEmbedding<EDGE>(phi2(d)))
			{
				CGoGNout << "Check: different embeddings on edge" << CGoGNendl ;
				return false ;
			}
		}

		if (isOrbitEmbedded<FACE>())
		{
			if (getEmbedding<FACE>(d) != getEmbedding<FACE>(phi1(d)))
		{
				CGoGNout << "Check: different embeddings on face" << CGoGNendl ;
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

	return true ;
}

} // namespace CGoGN
