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

#include "Geometry/basic.h"
#include "Geometry/intersection.h"
#include "Geometry/inclusion.h"

#include "Algo/Geometry/normal.h"
#include "Algo/Geometry/centroid.h"

#include <limits>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP>
bool intersectionLineConvexFace(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& P, const typename PFP::VEC3& Dir, typename PFP::VEC3& Inter)
{
	typedef typename PFP::VEC3 VEC3 ;

	const float SMALL_NUM = std::numeric_limits<typename PFP::REAL>::min() * 5.0f;

	Dart d = f.dart;

	VEC3 p1 = position[d];
	VEC3 n = faceNormal<PFP>(map, d, position);
	VEC3 w0 = P - p1;
    float a = -(n*w0);
    float b = n*Dir;

    if (fabs(b) < SMALL_NUM)
		return false;

	float r = a / b;
	Inter = P + r * Dir;           // intersect point of ray and plane

    // is I inside the face?
	VEC3 p2 = position[map.phi1(d)];
	VEC3 v = p2 - p1 ;
	VEC3 vInter = Inter - p1;
	float dirV = v * vInter;
	if(fabs(dirV) < SMALL_NUM) // on an edge
		return true;

	Dart it = map.phi1(d);
	while(it != d)
	{
		p1 = p2;
		p2 = position[map.phi1(it)];
		v = p2 - p1;
		vInter = Inter - p1;
		float dirD = v * vInter;

		if(fabs(dirD) < SMALL_NUM) // on an edge
			return true;
		if((dirV > SMALL_NUM && dirD < SMALL_NUM) || (dirV < SMALL_NUM && dirD > SMALL_NUM)) //exterior of the face
			return false;
		it = map.phi1(it) ;
	}

    return true;
}

template <typename PFP>
bool intersectionSegmentConvexFace(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& PA, const typename PFP::VEC3& PB, typename PFP::VEC3& Inter)
{
	typedef typename PFP::VEC3 VEC3 ;

	VEC3 dir = PB - PA;
	if (intersectionLineConvexFace(map, f, position, PA, dir, Inter))
	{
		VEC3 dirA = PA - Inter;
		VEC3 dirB = PB - Inter;

		if (dirA * dirB < 0)
			return true;
	}
	return false;
}

template <typename PFP>
bool areTrianglesInIntersection(typename PFP::MAP& map, Face t1, Face t2, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	typedef typename PFP::VEC3 VEC3 ;

	Dart tri1 = t1.dart;
	Dart tri2 = t2.dart;

	//get vertices position
	VEC3 tris1[3];
	VEC3 tris2[3];
	for (unsigned int i = 0; i < 3; ++i)
	{
		tris1[i] = position[tri1];
		tris2[i] = position[tri2];
		tri1 = map.phi1(tri1);
		tri2 = map.phi1(tri2);
	}

// 	gmtl::Vec3f nTri1,nTri2;
//	float offset1,offset2;
// 	CGoGN::Algo::Geometry::trianglePlane<PFP>(map,tri1,nTri1,offset1);
// 	CGoGN::Algo::Geometry::trianglePlane<PFP>(map,tri2,nTri2,offset2);
//
// 	Orientation3D oP[2][3];
// 	oP[0][0] = testOrientation3D(tris2[0],offset1,nTri1);
// 	oP[0][1] = testOrientation3D(tris2[1],offset1,nTri1);
// 	oP[0][2] = testOrientation3D(tris2[2],offset1,nTri1);
//
// 	if(oP[0][0]==oP[0][1] && oP[0][1]==oP[0][2]) {
// 		if(oP[0][0]==ON) { //coplanar triangles
// 			return isSegmentInTriangle2D(tris1[0],tris1[1],tris2[0],tris2[1],triS2[2],nTri2)
// 			||  isSegmentInTriangle2D(tris1[1],tris1[2],tris2[0],tris2[1],triS2[2],nTri2)
// 			|| isSegmentInTriangle2D(tris1[2],tris1[0],tris2[0],tris2[1],triS2[2],nTri2);
// 		}
// 		else
// 			return false;
// 	}
//
// 	oP[1][0] = testOrientation3D(tris1[0],offset2,nTri2);
// 	oP[1][1] = testOrientation3D(tris1[1],offset2,nTri2);
// 	oP[1][2] = testOrientation3D(tris1[2],offset2,nTri2);
//
// 	if(oP[1][0]==oP[1][1] && oP[1][1]==oP[1][2])
// 		return false;
//
// 	//search segment of tri 1 in plane of tri 2
// 	gmtl::Point3f inter1,inter2;
// 	bool found = false;
// 	for(unsigned int i=0;i<3 && !found;++i) {
// 		//test if the first point is the one opposite to the two others
// 		if(oP[0][i]!=oP[0][(1+i)%3] && oP[0][(1+i)%3]==oP[0][(2+i)%3]) {
// 			found=true;
// 			//search collision points with the two edges
// 			float offset= gmtl::dot(tris2[0],nTri2);
// 			gmtl::Planef pl(nTri2,offset);
//
// 			gmtl::Vec3f dir1(oP[0][(1+i)%3]);
// 			dir1 -= oP[0][i];
//
// 			gmtl::Vec3f dir2(oP[0][(2+i)%3]);
// 			dir2 -= oP[0][i];
//
// 			inter1 = gmtl::intersect(pl,gmtl::Ray(oP[0][i],dir1));
// 			inter2 = gmtl::intersect(pl,gmtl::Ray(oP[0][i],dir2));
// 		}
// 	}
//
// 	return isSegmentInTriangle2D(inter1,inter2,tris2[0],tris2[1],triS2[2],nTri2);

	//compute face normal
	VEC3 normale1 = faceNormal<PFP>(map, t1, position);
	VEC3 bary1 = faceCentroid<PFP>(map, t1, position);

	int pos = 0;
	int neg = 0;
	//test position of points relative to first tri
	for (unsigned int i = 0; i < 3 ; ++i)
	{
		VEC3 nTest = bary1 - tris2[i];
		float scal = nTest * normale1;
		if (scal < 0)
			++neg;
		if (scal > 0)
			++pos;
	}

	//if all pos or neg then no intersection
	if (neg == 3 || pos == 3)
		return false;

	//same for the second triangle
	VEC3 normale2 = faceNormal<PFP>(map, t2, position);
	VEC3 bary2 = faceCentroid<PFP>(map, t2, position);
	pos = 0;
	neg = 0;
	for (unsigned int i = 0; i < 3 ; ++i)
	{
		VEC3 nTest = bary2 - tris1[i];
		float scal = nTest * normale2;
		if (scal<0)
			++neg;
		if (scal>0)
			++pos;
	}

	if (neg == 3 || pos == 3)
		return false;

	bool intersection = false;

	for (unsigned int i = 0; i < 3 && !intersection; ++i)
	{
		VEC3 inter;
		intersection = Geom::intersectionSegmentTriangle(tris1[i], tris1[(i+1)%3], tris2[0], tris2[1], tris2[2], inter);
	}

	if (intersection)
		return true;

	for (unsigned int i = 0; i < 3 && !intersection; ++i)
	{
		VEC3 inter;
		intersection = Geom::intersectionSegmentTriangle(tris2[i], tris2[(i+1)%3], tris1[0], tris1[1], tris1[2], inter);
	}

	return intersection;
}

template <typename PFP>
bool intersectionSphereEdge(typename PFP::MAP& map, typename PFP::VEC3& center, typename PFP::REAL radius, Edge e, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, typename PFP::REAL& alpha)
{
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	const VEC3& p1 = position[e.dart];
	const VEC3& p2 = position[map.phi1(e.dart)];
	if(Geom::isPointInSphere(p1, center, radius) && !Geom::isPointInSphere(p2, center, radius))
	{
		VEC3 p = p1 - center;
		VEC3 qminusp = p2 - center - p;
		REAL s = p * qminusp;
		REAL n2 = qminusp.norm2();
		alpha = (- s + sqrt(s*s + n2 * (radius*radius - p.norm2()))) / n2;
		return true ;
	}
	return false ;
}

} // namespace Geometry

} // Surface

} // namespace Algo

} // namespace CGoGN
