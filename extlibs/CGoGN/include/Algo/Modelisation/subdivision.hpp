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

#include "Algo/Geometry/basic.h"
#include "Algo/Geometry/centroid.h"
#include "Topology/generic/autoAttributeHandler.h"

#define _USE_MATH_DEFINES
#include <math.h>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Modelisation
{

template <typename PFP>
Dart trianguleFace(typename PFP::MAP& map, Dart d)
{
	Dart d1 = map.phi1(d);
	if (d1 == d)
        std::cerr << "Warning: triangulation of a face with only one edge" << std::endl;
	if (map.phi1(d1) == d)
        std::cerr << "Warning: triangulation of a face with only two edges" << std::endl;
	map.splitFace(d, d1) ;
	map.cutEdge(map.phi_1(d)) ;
	Dart x = map.phi2(map.phi_1(d)) ;
	Dart dd = map.template phi<111>(x) ;
	while(dd != x)
	{
		Dart next = map.phi1(dd) ;
		map.splitFace(dd, map.phi1(x)) ;
		dd = next ;
	}

//    Dart e = map.phi2(x);
//    std::cerr << "checking embeddings :in triangulateFace " << std::endl;
//    std::cerr << map.template getEmbedding<FACE>(e) << std::endl;
//    std::cerr << map.template getEmbedding<FACE>(map.phi2(map.phi_1(e))) << std::endl;
//    Dart f = map.phi2(e);
//    std::cerr << "dart "<< f << " map.template getEmbedding<FACE>(f) " << map.template getEmbedding<FACE>(f) << std::endl;
//    std::cerr <<  "dart "<< map.phi3(f) << " map.template getEmbedding<FACE>(map.phi3(f)) " << map.template getEmbedding<FACE>(map.phi3(f)) << std::endl;
//    Dart g = map.phi1(f);
//    while (g != f) {
//        std::cerr << "map.template getEmbedding<FACE>(g) " << map.template getEmbedding<FACE>(g) << std::endl;
//        std::cerr << "map.template getEmbedding<FACE>(map.phi3(g)) " << map.template getEmbedding<FACE>(map.phi3(g)) << std::endl;
//        g = map.phi1(g);
//    }
	return map.phi2(x);	// Return a dart of the central vertex
}

template <typename PFP, typename EMBV>
void trianguleFaces(typename PFP::MAP& map, EMBV& attributs)
{
	typedef typename EMBV::DATA_TYPE EMB;

	TraversorF<typename PFP::MAP> t(map) ;
	for (Dart d = t.begin(); d != t.end(); d = t.next())
	{
		EMB center = Geometry::faceCentroid<PFP, EMBV>(map, d, attributs);	// compute center
		Dart cd = trianguleFace<PFP>(map, d);	// triangule the face
		attributs[cd] = center;					// affect the data to the central vertex
		Dart fit = cd ;
		do
		{
			t.skip(fit);
			fit = map.phi2(map.phi_1(fit));
		} while(fit != cd);
	}
}

//template <typename PFP>
//void trianguleFaces(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position)
//{
//	trianguleFaces<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map, position) ;
//}

template <typename PFP>
void trianguleFaces(
	typename PFP::MAP& map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& positionF
)
{
	TraversorF<typename PFP::MAP> t(map) ;
	for (Dart d = t.begin(); d != t.end(); d = t.next())
	{
		Dart cd = trianguleFace<PFP>(map, d);	// triangule the face
		position[cd] = positionF[d];			// affect the data to the central vertex
		Dart fit = cd ;
		do
		{
			t.skip(fit);
			fit = map.phi2_1(fit);
		} while(fit != cd);
	}
}

template <typename PFP>
Dart quadranguleFace(typename PFP::MAP& map, Dart d)
{
	d = map.phi1(d) ;
	map.splitFace(d, map.template phi<11>(d)) ;
	map.cutEdge(map.phi_1(d)) ;
	Dart x = map.phi2(map.phi_1(d)) ;
	Dart dd = map.template phi<1111>(x) ;
	while(dd != x)
	{
		Dart next = map.template phi<11>(dd) ;
		map.splitFace(dd, map.phi1(x)) ;
		dd = next ;
	}

	return map.phi2(x);	// Return a dart of the central vertex
}

template <typename PFP, typename EMBV>
void quadranguleFaces(typename PFP::MAP& map, EMBV& attributs)
{
	typedef typename PFP::MAP MAP;
	typedef typename EMBV::DATA_TYPE EMB;

	DartMarker<MAP> me(map) ;
	DartMarker<MAP> mf(map) ;

	// first pass: cut the edges
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if ( !map.template isBoundaryMarked<2>(d) && !me.isMarked(d))
		{
			Dart f = map.phi1(d);
			Dart e = map.cutEdge(d);
//			TODO trouver pourquoi lerp bug avec ECell
//			attributs[m] = AttribOps::lerp<EMB,PFP>(attributs[d],attributs[f], 0.5);

			attributs[e] = attributs[d];
			attributs[e] += attributs[f];
			attributs[e] *= 0.5;

			me.template markOrbit<EDGE>(d);
			me.template markOrbit<EDGE>(e);
			mf.template markOrbit<VERTEX>(e);
		}
	}

	// second pass: quandrangule faces
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if ( !map.template isBoundaryMarked<2>(d) && !mf.isMarked(d))
		{
			EMB center = Geometry::faceCentroid<PFP, EMBV>(map, d, attributs);	// compute center
			Dart cf = quadranguleFace<PFP>(map, d);	// quadrangule the face
			attributs[cf] = center;					// affect the data to the central vertex
			Dart e = cf;
			do
			{
				mf.template markOrbit<FACE>(e);
				e = map.phi2_1(e);
			} while (e != cf);
		}
	}
}

//template <typename PFP>
//void quadranguleFaces(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position)
//{
//	quadranguleFaces<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map, position) ;
//}

template <typename PFP, typename EMBV>
void CatmullClarkSubdivision(typename PFP::MAP& map, EMBV& attributs)
{
	typedef typename PFP::MAP MAP;
	typedef typename EMBV::DATA_TYPE EMB;

	std::vector<Dart> l_middles;
	std::vector<Dart> l_verts;

	CellMarkerNoUnmark<MAP, VERTEX> m0(map);
	DartMarkerNoUnmark<MAP> mf(map);
	DartMarkerNoUnmark<MAP> me(map);

	// first pass: cut edges
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if ( !map.template isBoundaryMarked<2>(d) && !me.isMarked(d))
		{
			if (!m0.isMarked(d))
			{
				m0.mark(d);
				l_verts.push_back(d);
			}
			Dart d2 = map.phi2(d);
			if (!m0.isMarked(d2))
			{
				m0.mark(d2);
				l_verts.push_back(d2);
			}

			Dart f = map.phi1(d);
			Dart e = map.cutEdge(d);

			attributs[e] = attributs[d];
			attributs[e] += attributs[f];
			attributs[e] *= 0.5;

			me.template markOrbit<EDGE>(d);
			me.template markOrbit<EDGE>(e);

			mf.mark(d) ;
			mf.mark(map.phi2(e)) ;

			l_middles.push_back(e);
		}
	}

	// second pass: quandrangule faces
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if ( !map.template isBoundaryMarked<2>(d) && mf.isMarked(d)) // for each face not subdivided
		{
			// compute center skip darts of new vertices non embedded
//			EMB center = AttribOps::zero<EMB,PFP>();
			EMB center(0.0);
			unsigned int count = 0 ;
			mf.template unmarkOrbit<FACE>(d) ;
			Dart it = d;
			do
			{
				center += attributs[it];
				++count ;
				me.template unmarkOrbit<PFP::MAP::EDGE_OF_PARENT>(it);

				it = map.phi1(it) ;
				me.template unmarkOrbit<PFP::MAP::EDGE_OF_PARENT>(it);
				it = map.phi1(it) ;
			} while(it != d) ;
			center /= double(count);
			Dart cf = quadranguleFace<PFP>(map, d);	// quadrangule the face
			attributs[cf] = center;					// affect the data to the central vertex
		}
	}

	// Compute edge points
	for(typename std::vector<Dart>::iterator mid = l_middles.begin(); mid != l_middles.end(); ++mid)
	{
		Dart x = *mid;
		// other side of the edge
		if (!map.isBoundaryEdge(x))
		{
			Dart f1 = map.phi_1(x);
			Dart f2 = map.phi2(map.phi1(map.phi2(x)));
//			EMB temp = AttribOps::zero<EMB,PFP>();
//			temp = attributs[f1];
			EMB temp = attributs[f1];
			temp += attributs[f2];			// E' = (V0+V1+F1+F2)/4
			temp *= 0.25;
			attributs[x] *= 0.5;
			attributs[x] += temp;
		}
		// else nothing to do point already in the middle of segment
	}

	// Compute vertex points
	for(typename std::vector<Dart>::iterator vert = l_verts.begin(); vert != l_verts.end(); ++vert)
	{
		m0.unmark(*vert);

//		EMB temp = AttribOps::zero<EMB,PFP>();
//		EMB temp2 = AttribOps::zero<EMB,PFP>();
		EMB temp(0.0);
		EMB temp2(0.0);

		unsigned int n = 0;
		Dart x = *vert;
		do
		{
			Dart m = map.phi1(x);
			Dart f = map.phi2(m);
			Dart v = map.template phi<11>(f);

			temp += attributs[f];
			temp2 += attributs[v];

			++n;
			x = map.phi2_1(x);
		} while (x != *vert);

		EMB emcp = attributs[*vert];
		emcp *= double((n-2)*n);		// V' = (n-2)/n*V + 1/n2 *(F+E)
		emcp += temp;
		emcp += temp2;
		emcp /= double(n*n);

		attributs[*vert] = emcp ;
	}
}

//template <typename PFP>
//void CatmullClarkSubdivision(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position)
//{
//	CatmullClarkSubdivision<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map, position) ;
//}

inline double betaF(unsigned int n)
{
	switch(n)
	{
		case 1: return 0.234375 ;
		case 2: return 0.609375 ;
		case 3: return 0.5625 ;
		case 4: return 0.484375 ;
		case 5: return 0.420466 ;
		case 6: return 0.375 ;
		case 7: return 0.343174 ;
		case 8: return 0.320542 ;
		case 9: return 0.304065 ;
		case 10: return 0.291778 ;
		case 11: return 0.282408 ;
		case 12: return 0.27512 ;
		case 13: return 0.26935 ;
		case 14: return 0.264709 ;
		default:
			double t = 3.0 + 2.0 * cos((2.0*M_PI)/double(n)) ;
			return 5.0/8.0 - (t * t) / 64.0 ;
	}
}

template <typename PFP, typename EMBV>
void LoopSubdivision(typename PFP::MAP& map, EMBV& attributs)
{
	typedef typename PFP::MAP MAP;
	typedef typename EMBV::DATA_TYPE EMB;

	std::vector<Dart> l_middles;
	std::vector<Dart> l_verts;

	CellMarkerNoUnmark<MAP, VERTEX> m0(map);
	DartMarkerNoUnmark<MAP> mv(map);
	DartMarkerNoUnmark<MAP> me(map);

	// first pass cut edges
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if ( !map.template isBoundaryMarked<2>(d) && !me.isMarked(d))
		{
			if (!m0.isMarked(d))
			{
				m0.mark(d);
				l_verts.push_back(d);
			}
			if (!m0.isMarked(map.phi2(d)))
			{
				m0.mark(map.phi2(d));
				l_verts.push_back(map.phi2(d));
			}

			Dart f = map.phi1(d);
			Dart e = map.cutEdge(d);

			attributs[e] =  attributs[d];
			attributs[e] += attributs[f];
			attributs[e] *= 0.5;

			me.template markOrbit<EDGE>(d);
			me.template markOrbit<EDGE>(e);

			mv.template markOrbit<VERTEX>(e);

			l_middles.push_back(e);
		}
	}

	// Compute edge points
	for(typename std::vector<Dart>::iterator mid = l_middles.begin(); mid != l_middles.end(); ++mid)
	{
		Dart d = *mid;
		if (!map.isBoundaryEdge(d))
		{
			Dart dd = map.phi2(d);
			attributs[d] *= 0.75;
			Dart e1 = map.template phi<111>(d);

//			EMB temp(0.0);
//			temp += attributs[e1];

			EMB temp = attributs[e1];
			e1 = map.phi_1(map.phi_1(dd));
			temp += attributs[e1];
			temp *= 1.0 / 8.0;
			attributs[d] += temp;
		}
		// else nothing to do point already in the middle of segment
	}

	// Compute vertex points
	for(typename std::vector<Dart>::iterator vert = l_verts.begin(); vert != l_verts.end(); ++vert)
	{
		m0.unmark(*vert);

//		EMB temp = AttribOps::zero<EMB,PFP>();
		EMB temp(0.0);
		int n = 0;
		Dart x = *vert;
		do
		{
			Dart y = map.phi1(map.phi1(x));
			temp += attributs[y];
			++n;
			x = map.phi2_1(x);
		} while ((x != *vert));
		EMB emcp = attributs[*vert];
		if (n == 6)
		{
			temp /= 16.0;
			emcp *= 10.0/16.0;
			emcp += temp;
		}
		else
		{
			double beta = betaF(n) ;
			temp *= (beta / double(n));
			emcp *= (1.0 - beta);
			emcp += temp;
		}
		attributs[*vert] = emcp;
	}

	// insert new edges
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (mv.isMarked(d))
		{
			// unmark the darts of the face
			me.template unmarkOrbit<FACE>(d) ;
			mv.template unmarkOrbit<FACE>(d) ;

			Dart dd = d;
			Dart e = map.template phi<11>(dd) ;
			map.splitFace(dd, e);

			dd = e;
			e = map.template phi<11>(dd) ;
			map.splitFace(dd, e);

			dd = e;
			e = map.template phi<11>(dd) ;
			map.splitFace(dd, e);
		}
	}
}

//template <typename PFP>
//void LoopSubdivision(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position)
//{
////	LoopSubdivision<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map, position) ;
//	LoopSubdivisionGen<PFP, VertexAttribute<typename PFP::VEC3> >(map, position) ;
//}

template <typename PFP, typename EMBV>
void TwoNPlusOneSubdivision(typename PFP::MAP& map, EMBV& attributs, float size)
{
	typedef typename PFP::MAP MAP;
	typedef typename EMBV::DATA_TYPE EMB;

	CellMarker<MAP, EDGE> m0(map);
	CellMarker<MAP, FACE> m1(map);

	std::vector<Dart> dOrig;

	//first pass cut edge
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(!m0.isMarked(d))
		{
			if(!m1.isMarked(d))
			{
				m1.mark(d);
				dOrig.push_back(d);
			}

//			if(selected(map.phi2(d)) && !m1.isMarked(map.phi2(d)))   TODO CHECK THIS ALGO
			if (!m1.isMarked(map.phi2(d)))
			{
				m1.mark(map.phi2(d));
				dOrig.push_back(map.phi2(d));
			}

			EMB e1 = attributs[d];
			EMB e2 = attributs[map.phi1(d)];
			map.cutEdge(d);
			attributs[map.phi1(d)] = e1*(1.0f-size)+e2*size;
			map.cutEdge(map.phi1(d));
			attributs[map.phi1(map.phi1(d))] = e2*(1.0f-size)+e1*size;
			m0.mark(d);
			m0.mark(map.phi1(d));
			m0.mark(map.template phi<11>(d));
		}
	}

	CGoGNout << "nb orig : " << dOrig.size() << CGoGNendl;

	DartMarkerNoUnmark<MAP> mCorner(map);
//	//second pass create corner face
	for (std::vector<Dart>::iterator it = dOrig.begin(); it != dOrig.end(); ++it)
	{
//		EMB c = Geometry::faceCentroid<PFP>(map,*it,attributs);
		Dart dd = *it;
		do
		{
			map.splitFace(map.phi1(dd),map.phi_1(dd));
			map.cutEdge(map.phi1(dd));
			mCorner.mark(map.phi2(map.phi1(dd)));
//			attributs[map.template phi<11>(dd)] = c*(1.0-size)+ attributs[dd]*size;
			attributs[map.template phi<11>(dd)] = attributs[dd] 	+ Geometry::vectorOutOfDart<PFP>(map,dd,attributs)
																	- Geometry::vectorOutOfDart<PFP>(map,map.phi_1(dd),attributs);
			dd = map.phi1(map.phi1(map.phi1(map.phi2(map.phi1(dd)))));
		} while(!mCorner.isMarked(dd));
	}

	//third pass create center face
	for (std::vector<Dart>::iterator it = dOrig.begin(); it != dOrig.end(); ++it)
	{
		Dart dd = map.phi2(map.phi1(*it));
		do {
			mCorner.unmark(dd);
			Dart dNext = map.phi1(map.phi1(map.phi1(dd)));
			map.splitFace(dd,dNext);
			dd = dNext;
		} while(mCorner.isMarked(dd));
	}
}


template <typename PFP, typename EMBV>
void DooSabin(typename PFP::MAP& map, EMBV& position)
{
	typedef typename PFP::MAP MAP;
	typedef typename EMBV::DATA_TYPE EMB;

	DartMarker<MAP> dm(map);
	// storage of boundary of hole (missing vertex faces)
	std::vector<Dart> fp;
	fp.reserve(16384);

	// storage of initial faces for position updating
	std::vector<Dart> faces;
	faces.reserve(16384);


	// create the edge faces
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (!dm.isMarked(d))
		{
			faces.push_back(d);
			Dart e = d;
			do
			{
				Dart e2 = map.phi2(e);
				if (!dm.isMarked(e2))
				{
					map.unsewFaces(e,false);
					Dart nf = map.newFace(4,false);
					map.sewFaces(e,nf,false);
					map.sewFaces(e2,map.template phi<11>(nf),false);
					// take care of edge embedding
					if(map.template isOrbitEmbedded<EDGE>())
					{
						map.template setOrbitEmbedding<EDGE>(nf, map.template getEmbedding<EDGE>(e));
						map.template setOrbitEmbedding<EDGE>(map.template phi<11>(nf), map.template getEmbedding<EDGE>(e2));
					}

					dm.markOrbit<FACE>(nf);
					fp.push_back(map.phi1(nf));
					fp.push_back(map.phi_1(nf));
				}
				dm.markOrbit<EDGE1>(e);
				e = map.phi1(e);
			} while (e!=d);
		}
	}
	// fill (create) the new  vertex faces
	for (std::vector<Dart>::iterator di=fp.begin(); di != fp.end(); ++di)
	{
		if (map.phi2(*di) == *di)
		{
			map.PFP::MAP::TOPO_MAP::closeHole(*di,false);

			if(map.template isOrbitEmbedded<EDGE>())
			{
				Dart df = map.phi2(*di);
				Dart d = df;
				do
				{
					map.template setOrbitEmbedding<EDGE>(d,map.template getEmbedding<EDGE>(map.phi2(d)));
					d = map.phi1(d);
				} while (d != df);
			}
		}
	}

	std::vector<EMB> buffer;
	buffer.reserve(8);
	for (std::vector<Dart>::iterator di=faces.begin(); di != faces.end(); ++di)
	{
		Dart e = *di;
		typename PFP::VEC3 center = Geometry::faceCentroid<PFP>(map,e,position);

		do
		{
			// compute DoSabin
			buffer.push_back(position[e]);
			e = map.phi1(e);
		}while (e != * di);

		int N = buffer.size();
		for (int i = 0; i < N; ++i)
		{
			EMB P(0);
			for (int j = 0; j < N; ++j)
			{
				if (j==i)
				{
					/*float*/typename PFP::REAL c1 = double(N+5)/double(4*N);
					P += buffer[j]*c1;
				}
				else
				{
					/*float*/typename PFP::REAL c2 = (3.0+2.0*cos(2.0*M_PI*(double(i-j))/double(N))) /(4.0*N);
					P+= c2*buffer[j];
				}
			}
			map.template setOrbitEmbeddingOnNewCell<VERTEX>(e);
			position[e] = P;
			e = map.phi1(e);
		}
		buffer.clear();
	}
}

inline double sqrt3_K(unsigned int n)
{
	switch(n)
	{
		case 1: return 0.333333 ;
		case 2: return 0.555556 ;
		case 3: return 0.5 ;
		case 4: return 0.444444 ;
		case 5: return 0.410109 ;
		case 6: return 0.388889 ;
		case 7: return 0.375168 ;
		case 8: return 0.365877 ;
		case 9: return 0.359328 ;
		case 10: return 0.354554 ;
		case 11: return 0.350972 ;
		case 12: return 0.348219 ;
		default:
			double t = cos((2.0*M_PI)/double(n)) ;
			return (4.0 - t) / 9.0 ;
	}
}

//template <typename PFP>
//void Sqrt3Subdivision(typename PFP::MAP& map, VertexAttribute<VEC3>& position)
//{
//	typedef typename PFP::VEC3 VEC3 ;
//	typedef typename PFP::REAL REAL ;
//
//	FaceAttribute<VEC3> positionF = map.template getAttribute<VEC3, FACE>("position") ;
//	if(!positionF.isValid())
//		positionF = map.template addAttribute<VEC3, FACE>("position") ;
//	Geometry::computeCentroidFaces<PFP>(map, position, positionF) ;
//
//	computeDual<PFP>(map);
//
//	VertexAttribute<VEC3> tmp = position ;
//	position = positionF ;
//	positionF = tmp ;
//
//	CellMarker m(map, VERTEX) ;
//	m.markAll() ;
//
//	trianguleFaces<PFP>(map, position, positionF);
//
//	for(Dart d = map.begin(); d != map.end(); map.next(d))
//	{
//		if(!m.isMarked(d))
//		{
//			m.mark(d) ;
//			VEC3 P = position[d] ;
//			VEC3 newP(0) ;
//			unsigned int val = 0 ;
//			Dart vit = d ;
//			do
//			{
//				newP += position[map.phi2(vit)] ;
//				++val ;
//				vit = map.phi2_1(vit) ;
//			} while(vit != d) ;
//			REAL K = sqrt3_K(val) ;
//			newP *= REAL(3) ;
//			newP -= REAL(val) * P ;
//			newP *= K / REAL(2 * val) ;
//			newP += (REAL(1) - K) * P ;
//			position[d] = newP ;
//		}
//	}
//}

template <typename PFP>
void computeDual(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;

	// Face Attribute -> after dual new Vertex Attribute
	FaceAttribute<VEC3, MAP> positionF  = map.template getAttribute<VEC3, FACE>("position") ;
	if(!positionF.isValid())
		positionF = map.template addAttribute<VEC3, FACE>("position") ;

	// Compute Centroid for the faces
	Algo::Surface::Geometry::computeCentroidFaces<PFP>(map, position, positionF) ;

	// Compute the Dual mesh
	map.computeDual();
	position = positionF ;
}


template <typename PFP>
void computeBoundaryConstraintDual(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	// Face Attribute -> after dual new Vertex Attribute
	FaceAttribute<VEC3, MAP> positionF  = map.template getAttribute<VEC3, FACE>("position") ;
	if(!positionF.isValid())
		positionF = map.template addAttribute<VEC3, FACE>("position") ;

	//Triangule boundary faces & compute for each new face the centroid
	std::vector<Dart> boundsDart;
	DartMarkerStore<MAP> mf(map);
	for(Dart dit = map.begin() ; dit != map.end() ; map.next(dit))
	{
		if(!mf.isMarked(dit) && map.template isBoundaryMarked<2>(dit))
		{
			boundsDart.push_back(dit);
			Dart db = dit;
			Dart d1 = map.phi1(db);
			Dart dprev = map.phi_1(db);
			map.splitFace(db, d1) ;
			map.cutEdge(map.phi_1(db)) ;

			positionF[dit] = (position[dit] + position[map.phi2(dit)]) * REAL(0.5);
			mf.template markOrbit<FACE>(dit);

			Dart x = map.phi2(map.phi_1(db)) ;
			Dart dd = map.phi1(map.phi1(map.phi1(x)));
			while(dd != x)
			{
				Dart next = map.phi1(dd) ;
				Dart prev = map.phi_1(dd);
				map.splitFace(dd, map.phi1(x)) ;
				positionF[prev] = (position[prev] + position[map.phi1(prev)]) * REAL(0.5);
				mf.template markOrbit<FACE>(prev);
				dd = next ;
			}

			positionF[dprev] = (position[dprev] + position[map.phi1(dprev)]) * REAL(0.5);
			mf.template markOrbit<FACE>(dprev);
		}
	}

	// Compute Centroid for the other faces
	Algo::Surface::Geometry::computeCentroidFaces<PFP>(map, position, positionF) ;

	// Fill the holes
	for(Dart dit = map.begin() ; dit != map.end() ; map.next(dit))
	{
		if(mf.isMarked(dit) && map.template isBoundaryMarked<2>(dit))
		{
			map.fillHole(dit);
			mf.template unmarkOrbit<FACE>(dit);
		}
	}

	// Compute the Dual mesh
	map.computeDual();
	position = positionF ;

	// Create the new border with the old boundary edges
	for(std::vector<Dart>::iterator it = boundsDart.begin() ; it != boundsDart.end() ; ++it)
	{
		map.createHole(map.phi2(map.phi1(*it)));
	}
}

template <typename PFP>
void computeBoundaryConstraintKeepingOldVerticesDual(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	// Face Attribute -> after dual new Vertex Attribute
	FaceAttribute<VEC3, MAP> positionF  = map.template getAttribute<VEC3, FACE>("position") ;
	if(!positionF.isValid())
		positionF = map.template addAttribute<VEC3, FACE>("position") ;

	//Triangule boundary faces & compute for each new face the centroid
	std::vector<Dart> boundsDart;
	DartMarkerStore<MAP> mf(map);
	for(Dart dit = map.begin() ; dit != map.end() ; map.next(dit))
	{
		if(!mf.isMarked(dit) && map.template isBoundaryMarked<2>(dit))
		{
			boundsDart.push_back(dit);
			Dart db = dit;
			Dart d1 = map.phi1(db);
			Dart dprev = map.phi_1(db);
			map.splitFace(db, d1) ;
			map.cutEdge(map.phi_1(db)) ;

			positionF[dit] = (position[dit] + position[map.phi2(dit)]) * REAL(0.5);
			mf.template markOrbit<FACE>(dit);

			Dart x = map.phi2(map.phi_1(db)) ;
			Dart dd = map.phi1(map.phi1(map.phi1(x)));
			while(dd != x)
			{
				Dart next = map.phi1(dd) ;
				Dart prev = map.phi_1(dd);
				map.splitFace(dd, map.phi1(x)) ;
				positionF[prev] = (position[prev] + position[map.phi1(prev)]) * REAL(0.5);
				mf.template markOrbit<FACE>(prev);
				dd = next ;
			}

			positionF[dprev] = (position[dprev] + position[map.phi1(dprev)]) * REAL(0.5);
			mf.template markOrbit<FACE>(dprev);
		}
	}

	// Compute Centroid for the other faces
	Algo::Surface::Geometry::computeCentroidFaces<PFP>(map, position, positionF) ;

	// Fill the holes
	for(Dart dit = map.begin() ; dit != map.end() ; map.next(dit))
	{
		if(mf.isMarked(dit) && map.template isBoundaryMarked<2>(dit))
		{
			map.fillHole(dit);
			mf.template unmarkOrbit<FACE>(dit);
		}
	}

	// Compute the Dual mesh
	map.computeDual();

	//Saving old position VertexAttribute to a FaceAttribute
	FaceAttribute<VEC3, MAP> temp;
	temp = position;
	position = positionF ;
	positionF = temp;

	// Create the new border with the old boundary edges
	for(std::vector<Dart>::iterator it = boundsDart.begin() ; it != boundsDart.end() ; ++it)
	{
		map.createHole(map.phi2(map.phi1(*it)));
	}

	// Manage old vertices with new FaceAttribute
	for(Dart dit = map.begin() ; dit != map.end() ; map.next(dit))
	{
		if(!mf.isMarked(dit) && map.template isBoundaryMarked<2>(dit))
		{
			Dart nd = map.cutEdge(dit);
			position[nd] = positionF[map.phi2(dit)];
			mf.template markOrbit<EDGE>(dit);
			mf.template markOrbit<EDGE>(nd);
		}
	}
}

} // namespace Modelisation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
