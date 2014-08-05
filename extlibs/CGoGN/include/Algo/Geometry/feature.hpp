/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2011, IGG Team, LSIIT, University of Strasbourg           *
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
* Web site: http://cgogn.u-strasbg.fr/                                         *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#include "Geometry/basic.h"
#include "Algo/Geometry/normal.h"
#include "Topology/generic/traversor/traversorCell.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP>
void featureEdgeDetection(
	typename PFP::MAP& map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	CellMarker<typename PFP::MAP, EDGE>& featureEdge)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	featureEdge.unmarkAll() ;

	FaceAttribute<VEC3, MAP> fNormal = map.template getAttribute<VEC3, FACE>("normal") ;
	if(!fNormal.isValid())
		fNormal = map.template addAttribute<VEC3, FACE>("normal") ;
	Algo::Surface::Geometry::computeNormalFaces<PFP>(map, position, fNormal) ;

	TraversorE<MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
	{
		if(!map.isBoundaryEdge(d) && Geom::angle(fNormal[d], fNormal[map.phi2(d)]) > M_PI / REAL(6))
			featureEdge.mark(d) ;
	}

//	map.removeAttribute(fNormal) ;
}

template <typename PFP>
void computeFaceGradient(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_normal,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& scalar,
	const FaceAttribute<typename PFP::REAL, typename PFP::MAP>& area,
	FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_gradient,
	unsigned int thread)
{
	TraversorF<typename PFP::MAP> trav(map, thread);
	for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		face_gradient[d] = faceGradient<PFP>(map, d, position, face_normal, scalar, area) ;
}

template <typename PFP>
typename PFP::VEC3 faceGradient(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_normal,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& scalar,
	const FaceAttribute<typename PFP::REAL, typename PFP::MAP>& face_area)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;

	Traversor2FV<typename PFP::MAP> t(map, d) ;

	Dart it = t.begin() ;
	VEC3 pos1 = position[it] ;
	REAL k1 = scalar[it] ;

	it = t.next() ;
	VEC3 pos2 = position[it] ;
	REAL k2 = scalar[it] ;

	it = t.next() ;
	VEC3 pos3 = position[it] ;
	REAL k3 = scalar[it] ;

	VEC3 n = face_normal[d] ;
	REAL a = face_area[d] ;
	VEC3 G = k1 * ( ( n ^ ( pos3 - pos2 ) ) / ( 2 * a ) ) +
			 k2 * ( ( n ^ ( pos1 - pos3 ) ) / ( 2 * a ) ) +
			 k3 * ( ( n ^ ( pos2 - pos1 ) ) / ( 2 * a ) ) ;

	G.normalize() ;
	return G ;
}

template <typename PFP>
void computeVertexGradient(
	typename PFP::MAP& map,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_gradient,
	const FaceAttribute<typename PFP::REAL, typename PFP::MAP>& face_area,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& vertex_gradient,
	unsigned int thread)
{
	TraversorV<typename PFP::MAP> trav(map, thread);
	for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		vertex_gradient[d] = vertexGradient<PFP>(map, d, face_gradient, face_area) ;
}

template <typename PFP>
typename PFP::VEC3 vertexGradient(
	typename PFP::MAP& map,
	Dart d,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& face_gradient,
	const FaceAttribute<typename PFP::REAL, typename PFP::MAP>& face_area)
{
	typename PFP::VEC3 G(0) ;
	typename PFP::REAL A(0) ;
	Traversor2VF<typename PFP::MAP> t(map, d) ;
	for (Dart d = t.begin(); d != t.end(); d = t.next())
	{
		G += face_area[d] * face_gradient[d] ;
		A += face_area[d] ;
	}
	G /= A ;

	G.normalize() ;
	return G ;
}

//template <typename PFP>
//typename PFP::REAL extremality(
//	typename PFP::MAP& map,
//	Dart d,
//	const typename PFP::VEC3& K,
//	const FaceAttribute<typename PFP::VEC3>& face_gradient,
//	const FaceAttribute<typename PFP::REAL>& face_area)
//{
//	typedef typename PFP::REAL REAL ;
//
//	REAL a = 0 ;
//	REAL e = 0 ;
//
//	Traversor2VF<typename PFP::MAP> trav(map, d) ;
//	for (Dart d2 = trav.begin(); d2 != trav.end(); d2 = trav.next())
//	{
//		a += face_area[d2] ;
//		e += face_area[d2] * ( face_gradient[d2] * K ) ;
//	}
//
//	return (e / a) ;
//}

template <typename PFP>
void computeTriangleType(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& K,
	CellMarker<typename PFP::MAP, FACE>& regularMarker,
	unsigned int thread)
{
	TraversorF<typename PFP::MAP> trav(map, thread);
	for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		if(isTriangleRegular<PFP>(map, d, K))
			regularMarker.mark(d) ;
	}
}

template <typename PFP>
bool mutuallyPositive(typename PFP::VEC3& v1, typename PFP::VEC3& v2, typename PFP::VEC3& v3)
{
	typename PFP::REAL v1v2 = v1 * v2 ;
	typename PFP::REAL v1v3 = v1 * v3 ;
	typename PFP::REAL v2v3 = v2 * v3 ;
	if(v1v2 > 0 && v1v3 > 0 && v2v3 > 0)
		return true ;
	return false ;
}

template <typename PFP>
bool isTriangleRegular(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& K)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;

	Dart v1 = d ;
	Dart v2 = map.phi1(v1) ;
	Dart v3 = map.phi1(v2) ;

	VEC3 K1 = K[v1] ;
	VEC3 K2 = K[v2] ;
	VEC3 K3 = K[v3] ;

//	VEC3 K1n = typename VEC3::DATA_TYPE(-1) * K1 ;
	VEC3 K2n = typename VEC3::DATA_TYPE(-1) * K2 ;
	VEC3 K3n = typename VEC3::DATA_TYPE(-1) * K3 ;

	if(mutuallyPositive<PFP>(K1, K2, K3))
		return true ;
	if(mutuallyPositive<PFP>(K1, K2, K3n))
		return true ;
	if(mutuallyPositive<PFP>(K1, K2n, K3))
		return true ;
	if(mutuallyPositive<PFP>(K1, K2n, K3n))
		return true ;

//	if(mutuallyPositive<PFP>(K1n, K2, K3))
//		return true ;
//	if(mutuallyPositive<PFP>(K1n, K2, K3n))
//		return true ;
//	if(mutuallyPositive<PFP>(K1n, K2n, K3))
//		return true ;
//	if(mutuallyPositive<PFP>(K1n, K2n, K3n))
//		return true ;

	return false ;
}

template <typename PFP>
void initRidgeSegments(
	typename PFP::MAP& map,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments,
	unsigned int thread)
{
	TraversorF<typename PFP::MAP> trav(map, thread);
	for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		ridge_segments[d].type = EMPTY ;
}

template <typename PFP>
void computeRidgeLines(
	typename PFP::MAP& map,
	CellMarker<typename PFP::MAP, FACE>& regularMarker,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& K,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& vertex_gradient,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& k,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& k2,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments,
	unsigned int thread)
{
	TraversorF<typename PFP::MAP> trav(map, thread);
	for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		if (regularMarker.isMarked(d))
			ridgeLines<PFP>(map, d, position, K, vertex_gradient, k, k2, ridge_segments) ;
	}
}

template <typename PFP>
void ridgeLines(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& K,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& vertex_gradient,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& k,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& k2,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;

	Dart v1 = d ;
	Dart v2 = map.phi1(v1) ;
	Dart v3 = map.phi1(v2) ;

	VEC3 Kv1 = K[v1] ;
	VEC3 Kv2 = K[v2] ;
	VEC3 Kv3 = K[v3] ;

	if((Kv1 * Kv2) < 0)
		Kv2 *= -1 ;
	if((Kv1 * Kv3) < 0)
		Kv3 *= -1 ;

	assert(mutuallyPositive<PFP>(Kv1, Kv2, Kv3)) ;

	// Calcul coefficient extremalite

//	REAL e1 = extremality<PFP>(map, v1, Kv1, face_gradient, face_area) ;
//	REAL e2 = extremality<PFP>(map, v2, Kv2, face_gradient, face_area) ;
//	REAL e3 = extremality<PFP>(map, v3, Kv3, face_gradient, face_area) ;

	REAL e1 = vertex_gradient[v1] * Kv1 ;
	REAL e2 = vertex_gradient[v2] * Kv2 ;
	REAL e3 = vertex_gradient[v3] * Kv3 ;

	// Extraction des zeros

	bool p1set = false ;
	bool p2set = false ;

	assert(ridge_segments[d].type == EMPTY) ;

	if( (e1 < 0 && e2 > 0) || (e1 > 0 && e2 < 0) )
	{
		REAL alpha = fabs(e1) / ( fabs(e1) + fabs(e2) ) ;
		ridge_segments[d].p1.d = v1 ;
		ridge_segments[d].p1.w = alpha ;
		p1set = true ;
	}
	if( (e2 < 0 && e3 > 0) || (e2 > 0 && e3 < 0) )
	{
		REAL alpha = fabs(e2) / ( fabs(e2) + fabs(e3) ) ;
		if(!p1set)
		{
			ridge_segments[d].p1.d = v2 ;
			ridge_segments[d].p1.w = alpha ;
			p1set = true ;
		}
		else
		{
			ridge_segments[d].p2.d = v2 ;
			ridge_segments[d].p2.w = alpha ;
			p2set = true ;
			ridge_segments[d].type = SEGMENT ;
		}
	}
	if( (e3 < 0 && e1 > 0) || (e3 > 0 && e1 < 0) )
	{
		REAL alpha = fabs(e3) / ( fabs(e1) + fabs(e3) ) ;
		if(p1set && !p2set)
		{
			ridge_segments[d].p2.d = v3 ;
			ridge_segments[d].p2.w = alpha ;
			p2set = true ;
			ridge_segments[d].type = SEGMENT ;
		}
	}

	if(p1set && p2set)
	{
		VEC3 n = Algo::Surface::Geometry::faceNormal<PFP>(map, d, position) ;
		REAL a = Algo::Surface::Geometry::convexFaceArea<PFP>(map, d, position) ;

		VEC3 p1 = position[v1] ;
		VEC3 p2 = position[v2] ;
		VEC3 p3 = position[v3] ;

		VEC3 eg = e1 * ( ( n ^ ( p3 - p2 ) ) / ( 2 * a ) ) +
				  e2 * ( ( n ^ ( p1 - p3 ) ) / ( 2 * a ) ) +
				  e3 * ( ( n ^ ( p2 - p1 ) ) / ( 2 * a ) ) ;
		eg.normalize() ;

		VEC3 Ktotal = Kv1 + Kv2 + Kv3 ;
		REAL ktotal = k[v1] + k[v2] + k[v3] ;
		ktotal = ktotal > 0 ? ktotal : -1.0 * ktotal ;
		REAL k2total = k2[v1] + k2[v2] + k2[v3] ;
		k2total = k2total > 0 ? k2total : -1.0 * k2total ;

		if( ( (eg * Ktotal) < 0 ) && ( ktotal > k2total ) )
			ridge_segments[d].type = FEATURE ;
	}
}

template <typename PFP>
void computeSingularTriangle(
	typename PFP::MAP& map,
	CellMarker<typename PFP::MAP, FACE>& regularMarker,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments,
	unsigned int thread)
{
	TraversorF<typename PFP::MAP> trav(map, thread);
	for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		if (! regularMarker.isMarked(d))
			singularTriangle<PFP>(map, d, regularMarker, ridge_segments) ;
	}
}

template <typename PFP>
void singularTriangle(
	typename PFP::MAP& map,
	Dart d,
	CellMarker<typename PFP::MAP, FACE>& regularMarker,
	FaceAttribute<ridgeSegment, typename PFP::MAP>& ridge_segments)
{
	int nbPoint = 0 ;

	Traversor2FFaE<typename PFP::MAP> f(map, d) ;
	for (Dart d2 = f.begin(); d2 != f.end(); d2 = f.next())
	{
		if(regularMarker.isMarked(d2) and ridge_segments[d2].type == SEGMENT)
		{
			if(isEdgeInTriangle<PFP>(map, ridge_segments[d2].p1.d, d))
			{
				if(nbPoint == 0)
				{
					ridge_segments[d].p1.d = map.phi2(ridge_segments[d2].p1.d) ;
					ridge_segments[d].p1.w = 1.0 - ridge_segments[d2].p1.w ;
				}
				else
				{
					ridge_segments[d].type ++ ;
					ridge_segments[d].p2.d = map.phi2(ridge_segments[d2].p1.d) ;
					ridge_segments[d].p2.w = 1.0 - ridge_segments[d2].p1.w ;
				}
				nbPoint ++ ;
			}
			else if(isEdgeInTriangle<PFP>(map, ridge_segments[d2].p1.d, d))
			{
				if(nbPoint == 0)
				{
					ridge_segments[d].p1.d = map.phi2(ridge_segments[d2].p2.d) ;
					ridge_segments[d].p1.w = 1.0 - ridge_segments[d2].p2.w ;
				}
				else
				{
					ridge_segments[d].type ++ ;
					ridge_segments[d].p2.d = map.phi2(ridge_segments[d2].p2.d) ;
					ridge_segments[d].p2.w = 1.0 - ridge_segments[d2].p2.w ;
				}
				nbPoint ++ ;
			}
		}
	}
}

template <typename PFP>
bool isEdgeInTriangle(typename PFP::MAP& map, Dart edge, Dart triangle)
{
	bool inTriangle = false ;

	Traversor2FE<typename PFP::MAP> t(map, triangle) ;
	for (Dart e = t.begin(); e != t.end(); e = t.next())
	{
		if(map.phi2(e) == edge)
			inTriangle = true ;
	}

	return inTriangle ;
}

template <typename PFP>
std::vector<typename PFP::VEC3> occludingContoursDetection(
	typename PFP::MAP& map,
	const typename PFP::VEC3& cameraPosition,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal)
{
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	std::vector<VEC3> occludingContours ;

	TraversorF<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
	{
		VEC3 p1 = position[d] ;
		VEC3 p2 = position[map.phi1(d)] ;
		VEC3 p3 = position[map.phi_1(d)] ;

		REAL dp1 = (p1 - cameraPosition) * normal[d] ;
		REAL dp2 = (p2 - cameraPosition) * normal[map.phi1(d)] ;
		REAL dp3 = (p3 - cameraPosition) * normal[map.phi_1(d)] ;

		if(dp1 < 0 && dp2 > 0)
		{
			REAL alpha = -dp1 / (-dp1 + dp2) ;
			occludingContours.push_back(alpha * p1 + (1 - alpha) * p2) ;
		}
		if(dp2 < 0 && dp1 > 0)
		{
			REAL alpha = dp1 / (dp1 - dp2) ;
			occludingContours.push_back(alpha * p1 + (1 - alpha) * p2) ;
		}
		if(dp1 < 0 && dp3 > 0)
		{
			REAL alpha = -dp1 / (-dp1 + dp3) ;
			occludingContours.push_back(alpha * p1 + (1 - alpha) * p3) ;
		}
		if(dp3 < 0 && dp1 > 0)
		{
			REAL alpha = dp1 / (dp1 - dp3) ;
			occludingContours.push_back(alpha * p1 + (1 - alpha) * p3) ;
		}
		if(dp2 < 0 && dp3 > 0)
		{
			REAL alpha = -dp2 / (-dp2 + dp3) ;
			occludingContours.push_back(alpha * p2 + (1 - alpha) * p3) ;
		}
		if(dp3 < 0 && dp2 > 0)
		{
			REAL alpha = dp2 / (dp2 - dp3) ;
			occludingContours.push_back(alpha * p2 + (1 - alpha) * p3) ;
		}
	}

	return occludingContours ;
}

} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
