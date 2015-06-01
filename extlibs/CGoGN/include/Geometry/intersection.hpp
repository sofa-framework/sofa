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

#include "Geometry/distances.h"

namespace CGoGN
{

namespace Geom
{

template <typename VEC3>
Intersection intersectionLinePlane(const VEC3& P, const VEC3& Dir, const VEC3& PlaneP, const VEC3& NormP, VEC3& Inter)
{
	float b = NormP * Dir ;

#define PRECISION 1e-6
	if (fabs(b) < PRECISION)		//ray parallel to triangle
	{
		VEC3 v = PlaneP - P;
		float c = NormP * v;
		if (fabs(c) < PRECISION )
			return EDGE_INTERSECTION;

		return NO_INTERSECTION;
	}
#undef PRECISION

	float a = NormP * (PlaneP - P);

	Inter = P + (a / b) * Dir;

	return FACE_INTERSECTION;
}

template <typename VEC3, typename PLANE>
Intersection intersectionLinePlane(const VEC3& P, const VEC3& Dir, const PLANE& Plane, VEC3& Inter)
{
	return intersectionLinePlane(P, Dir, Plane.normal()*Plane.d(), Plane.normal(), Inter);
}

template <typename VEC3>
Intersection intersectionRayTriangleOpt(const VEC3& P, const VEC3& Dir, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, VEC3& Inter)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 u = Ta - P ;
	VEC3 v = Tb - P ;
	VEC3 w = Tc - P ;

	T x = tripleProduct(Dir, u, v) ;
	T y = tripleProduct(Dir, v, w) ;
	T z = tripleProduct(Dir, w, u) ;

	unsigned int np = 0 ;
	unsigned int nn = 0 ;
	unsigned int nz = 0 ;

	if (x > T(0))
		++np ;
	else if (x < T(0))
		++nn ;
	else
		++nz ;

	if (y > T(0))
		++np ;
	else if (y < T(0))
		++nn ;
	else
		++nz ;

	if (z > T(0))
		++np ;
	else if (z < T(0))
		++nn ;
	else
		++nz ;

	if ((np != 0) && (nn != 0)) return NO_INTERSECTION ;

	T sum = x + y + z ;
	T alpha = y / sum ;
	T beta = z / sum ;
	T gamma = T(1) - alpha - beta ;
	Inter = Ta * alpha + Tb * beta + Tc * gamma ;

	return Intersection(FACE_INTERSECTION - nz) ;
}

template <typename VEC3>
Intersection intersectionRayTriangleOpt(const VEC3& P, const VEC3& Dir, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 u = Ta - P ;
	VEC3 v = Tb - P ;
	VEC3 w = Tc - P ;

	T x = tripleProduct(Dir, u, v) ;
	T y = tripleProduct(Dir, v, w) ;
	T z = tripleProduct(Dir, w, u) ;

	unsigned int np = 0 ;
	unsigned int nn = 0 ;
	unsigned int nz = 0 ;

	if (x > T(0))
		++np ;
	else if (x < T(0))
		++nn ;
	else
		++nz ;

	if (y > T(0))
		++np ;
	else if (y < T(0))
		++nn ;
	else
		++nz ;

	if (z > T(0))
		++np ;
	else if (z < T(0))
		++nn ;
	else
		++nz ;

	if ((np != 0) && (nn != 0)) return NO_INTERSECTION ;

	return Intersection(FACE_INTERSECTION - nz) ;
}

template <typename VEC3>
Intersection intersectionRayTriangle(const VEC3& P, const VEC3& Dir, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, VEC3& Inter)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 u = Ta - P ;
	VEC3 v = Tb - P ;
	VEC3 w = Tc - P ;
	T x = tripleProduct(Dir, u, v) ;
	T y = tripleProduct(Dir, v, w) ;
	T z = tripleProduct(Dir, w, u) ;
//	std::cout << x << " / "<< y << " / "<< z << std::endl;
	if((x < T(0) && y < T(0) && z < T(0)) || (x > T(0) && y > T(0) && z > T(0)))
	{
		T sum = x + y + z ;
		T alpha = y / sum ;
		T beta = z / sum ;
		T gamma = T(1) - alpha - beta ;
		Inter = Ta * alpha + Tb * beta + Tc * gamma ;
		return FACE_INTERSECTION ;
	}
	if(
		( x == T(0) && y > T(0) && z > T(0) ) || ( x == T(0) && y < T(0) && z < T(0) ) || // intersection on [Ta,Tb]
		( x > T(0) && y == T(0) && z > T(0) ) || ( x < T(0) && y == T(0) && z < T(0) ) || // intersection on [Tb,Tc]
		( x > T(0) && y > T(0) && z == T(0) ) || ( x < T(0) && y < T(0) && z == T(0) )	  // intersection on [Tc,Ta]
	)
		return EDGE_INTERSECTION ;
	if(
		( x == T(0) && y == T(0) && z != T(0) ) || // intersection on Tb
		( x == T(0) && y != T(0) && z == T(0) ) || // intersection on Ta
		( x != T(0) && y == T(0) && z == T(0) )	   // intersection on Tc
	)
		return VERTEX_INTERSECTION ;
	return NO_INTERSECTION ;
}


template <typename VEC3>
Intersection intersectionLineTriangle(const VEC3& P, const VEC3& Dir, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, VEC3& Inter)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 u = Tb - Ta ;
	VEC3 v = Tc - Ta ;
	VEC3 n = u ^ v ;

	VEC3 w0 = P - Ta ;
    T a = -(n * w0) ;
    T b = (n * Dir) ;

#define PRECISION 1e-20
    if(fabs(b) < PRECISION)			//ray parallel to triangle
			return NO_INTERSECTION ;
#undef PRECISION

	T r = a / b ;
	Inter = P + r * Dir ;			// intersect point of ray and plane

    // is I inside T?
	T uu = u.norm2() ;
	T uv = u * v ;
	T vv = v.norm2() ;
	VEC3 w = Inter - Ta ;
	T wu = w * u ;
	T wv = w * v ;
	T D = (uv * uv) - (uu * vv) ;

    // get and test parametric coords
	T s = ((uv * wv) - (vv * wu)) / D ;
	if(s < T(0) || s > T(1))
		return NO_INTERSECTION ;
	T t = ((uv * wu) - (uu * wv)) / D ;
	if(t < T(0) || (s + t) > T(1))
        return NO_INTERSECTION ;

	if((s == T(0) || s == T(1)))
		if(t == T(0) || t == T(1))
			return VERTEX_INTERSECTION ;
		else
			return EDGE_INTERSECTION ;
	else if(t == T(0) || t == T(1))
			return EDGE_INTERSECTION ;

    return FACE_INTERSECTION ;
}

template <typename VEC3>
Intersection intersectionLineTriangle2D(const VEC3& P, const VEC3& Dir, const VEC3& Ta,  const VEC3& Tb, const VEC3& Tc, VEC3& Inter)
{
	Inclusion inc = isPointInTriangle(P,Ta,Tb,Tc) ;
	VEC3 N = Ta ^ Tb ;

	switch(inc)
	{
		case FACE_INCLUSION :
			Inter = P ;
			return FACE_INTERSECTION ;
		case EDGE_INCLUSION :
			Inter = P ;
			return EDGE_INTERSECTION ;
		case VERTEX_INCLUSION :
			Inter = P ;
			return VERTEX_INTERSECTION ;
		default : //NO_INCLUSION : test if ray enters the triangle
			VEC3 P2 = P + Dir ;
			Orientation2D oA = testOrientation2D(Ta, P, P2, N) ;
			Orientation2D oB = testOrientation2D(Tb, P, P2, N) ;
			Orientation2D oC = testOrientation2D(Tc, P, P2, N) ;

			if(oA == oB && oB == oC)
				return NO_INTERSECTION ;

			Orientation2D oPBC = testOrientation2D(P,Tb,Tc,N) ;
			if(oPBC == LEFT)  // same side of A, test edge AC and AB
			{
				if(oA == LEFT)
				{
					if(oB == ALIGNED)
					{
						Inter = Tb ;
						return VERTEX_INTERSECTION ;
					}
					//inter with AB
//					CGoGNout << __FILE__ << " TODO compute edge coplanar intersection AB" << CGoGNendl ;
					return EDGE_INTERSECTION ;
				}
				if(oA == ALIGNED)
				{
					Inter = Ta ;
					return VERTEX_INTERSECTION ;
				}
				if(oC == ALIGNED)
				{
					Inter = Tc ;
					return VERTEX_INTERSECTION ;
				}
				//inter with AC
//				CGoGNout << __FILE__ << " TODO compute edge coplanar intersection AC" << CGoGNendl ;
				return EDGE_INTERSECTION ;
			}
			if(oPBC == RIGHT) // same side of BC, test this edge
			{
				if(oB == ALIGNED)
				{
					Inter = Tb ;
					return VERTEX_INTERSECTION ;
				}
				if(oC == ALIGNED)
				{
					Inter = Tc ;
					return VERTEX_INTERSECTION ;
				}
				//inter with BC
//				CGoGNout << __FILE__ << " TODO compute edge coplanar intersection BC" << CGoGNendl ;
				return EDGE_INTERSECTION ;
			}

			//aligned with BC
			//possibly colliding with edge AB or AC
			Orientation2D oPAB = testOrientation2D(P,Ta,Tb,N) ;
			if(oPAB == RIGHT) //possibly colliding with edge AB
			{
				if(oA == ALIGNED)
				{
					Inter = Ta ;
					return VERTEX_INTERSECTION ;
				}
				//inter with AB
//				CGoGNout << __FILE__ << " TODO compute edge coplanar intersection AB" << CGoGNendl ;
				return EDGE_INTERSECTION ;
			}
			if(oPAB == ALIGNED)
			{
				Inter = Tb ;
				return VERTEX_INTERSECTION ;
			}
			//possibly colliding with edge AC
			else if(oC == ALIGNED)
			{
				Inter = Tc ;
				return VERTEX_INTERSECTION ;
			}
			else if(oA == ALIGNED)
			{
				Inter = Ta ;
				return VERTEX_INTERSECTION ;
			}
			//inter with AC
//			CGoGNout << __FILE__ << " TODO compute edge coplanar intersection AC" << CGoGNendl ;
			return EDGE_INTERSECTION ;
	}
}

template <typename VEC3>
Intersection intersectionSegmentTriangle(const VEC3& PA, const VEC3& PB, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, VEC3& Inter)
{
        typedef double T ;
        const T precision = std::numeric_limits<T>::min();

	VEC3 u = Tb - Ta ;
	VEC3 v = Tc - Ta ;
	VEC3 Dir = PB - PA ;

        VEC3 n = u.cross(v) ;

	VEC3 w0 = PA - Ta ;
    float a = -(n * w0) ;
    float b = (n * Dir) ;

    if(fabs(b) < precision)			//ray parallel to triangle
		return NO_INTERSECTION ;

	//compute intersection
	T r = a / b ;

	if((r < -precision) || (r > (T(1) + precision)))
		return NO_INTERSECTION;

	Inter = PA + r * Dir;			// intersect point of ray and plane

    // is I inside T?
	T uu = u.norm2() ;
	T uv = u * v ;
	T vv = v.norm2() ;
	VEC3 w = Inter - Ta ;
	T wu = w * u ;
	T wv = w * v ;
	T D = (uv * uv) - (uu * vv) ;

    // get and test parametric coords
	T s = ((uv * wv) - (vv * wu)) / D ;

	if(s <= precision)
		s = 0.0f;

	if(s < T(0) || s > T(1))
		return NO_INTERSECTION ;

	T t = ((uv * wu) - (uu * wv)) / D ;

	if(t <= precision)
		t = 0.0f;

	if(t < T(0) || (s + t) > T(1))
        return NO_INTERSECTION ;

	if((s == T(0) || s == T(1)))
		if(t == T(0) || t == T(1))
			return VERTEX_INTERSECTION ;
		else
			return EDGE_INTERSECTION ;
	else if(t == T(0) || t == T(1))
			return EDGE_INTERSECTION ;

    return FACE_INTERSECTION ;
}

// template <typename VEC3>
// Intersection intersectionSegmentSegment2D(const VEC3& PA, const VEC3& PB, const VEC3& PC,  const VEC3& PD, VEC3& Inter) 
// {
// 	CGoGNout << __FILE__ << " " << __LINE__ << " to write intersectionSegmentSegment2D" << CGoGNendl;
// 	return NO_INTERSECTION;
// }

template <typename VEC3, typename PLANE3D>
Intersection intersectionPlaneRay(const PLANE3D& pl, const VEC3& p1, const VEC3& dir, VEC3& Inter)
{
	typename VEC3::DATA_TYPE denom = pl.normal()*dir;

	if (denom == 0)
	{
		if (pl.distance(p1) == 0)
		{
			Inter = p1 ;
			return FACE_INTERSECTION ;
		}
		else
			return NO_INTERSECTION ;
	}

	typename VEC3::DATA_TYPE isect = (pl.normal() * (pl.normal() * -1.0f * pl.d() - p1)) / denom ;

	Inter = p1 + dir * isect ;

	if (0.0f <= isect)
	{
		return FACE_INTERSECTION ;
	}

	return NO_INTERSECTION;
}

template <typename VEC3>
Intersection intersection2DSegmentSegment(const VEC3& PA, const VEC3& PB, const VEC3& QA, const VEC3& QB, VEC3& Inter)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 vp1p2 = PB - PA;
	VEC3 vq1q2 = QB - QA;
	VEC3 vp1q1 = QA - PA;
	T delta = vp1p2[0] * vq1q2[1] - vp1p2[1] * vq1q2[0] ;
	T coeff = vp1q1[0] * vq1q2[1] - vp1q1[1] * vq1q2[0] ;

	if (delta == 0) //parallel
	{
		//test if collinear
		if (coeff == 0)
		{
			//collinear
			//TODO : check if there is a common point between the two edges
			Inter = QA;
			return EDGE_INTERSECTION;
		}
		else
			return NO_INTERSECTION;
	}
	else
		Inter = VEC3((PA[0] * delta + vp1p2[0] * coeff) / delta, (PA[1] * delta + vp1p2[1] * coeff) / delta, (PA[2] * delta + vp1p2[2] * coeff) / delta) ;

	//test if inter point is outside the edges
	if(
		(Inter[0] < PA[0] && Inter[0] < PB[0]) || (Inter[0] > PA[0] && Inter[0] > PB[0]) ||
		(Inter[0] < QA[0] && Inter[0] < QB[0]) || (Inter[0] > QA[0] && Inter[0] > QB[0]) ||
		(Inter[1] < PA[1] && Inter[1] < PB[1]) || (Inter[1] > PA[1] && Inter[1] > PB[1]) ||
		(Inter[1] < QA[1] && Inter[1] < QB[1]) || (Inter[1] > QA[1] && Inter[1] > QB[1])
	)
		return NO_INTERSECTION;

	if(Geom::arePointsEquals(PA, Inter) || Geom::arePointsEquals(PB, Inter) || Geom::arePointsEquals(QA, Inter) || Geom::arePointsEquals(QB, Inter))
		return VERTEX_INTERSECTION;

	return EDGE_INTERSECTION;
}

template <typename VEC3>
Intersection intersectionSegmentPlan(const VEC3& PA, const VEC3& PB, const VEC3& PlaneP, const VEC3& NormP)//, VEC3& Inter)
{
//	typename VEC3::DATA_TYPE panp = NormP * PA;
//	typename VEC3::DATA_TYPE pbnp = NormP * PB;

//	if(panp == 0 || pbnp == 0)
//		return VERTEX_INTERSECTION;
//	else if((panp < 0 && pbnp > 0) || (panp > 0 && pbnp < 0))
//		return EDGE_INTERSECTION;
//	else
//		return NO_INTERSECTION;

#define EPSILON 1e-12

	typename VEC3::DATA_TYPE panp = NormP * (PA-PlaneP);
	typename VEC3::DATA_TYPE pbnp = NormP * (PB-PlaneP);

	if(abs(panp) < EPSILON || abs(pbnp) < EPSILON)
		return VERTEX_INTERSECTION;
//	else if((panp < 0 && pbnp > 0) || (panp > 0 && pbnp < 0))
	else if (panp*pbnp < 0)
			return EDGE_INTERSECTION;
	else
		return NO_INTERSECTION;
#undef EPSILON
}

template <typename VEC3>
Intersection intersectionSegmentPlan(const VEC3& PA, const VEC3& PB, const VEC3& PlaneP, const VEC3& NormP, VEC3& Inter)
{
#define EPSILON 1e-12

	typename VEC3::DATA_TYPE panp = NormP * (PA-PlaneP);
	typename VEC3::DATA_TYPE pbnp = NormP * (PB-PlaneP);

	if(abs(panp) < EPSILON)
	{
		Inter = PA;
		return VERTEX_INTERSECTION;
	}
	else if(abs(pbnp) < EPSILON)
	{
		Inter = PB;
		return VERTEX_INTERSECTION;
	}
	else if (panp*pbnp < 0)
	{
		Inter = (abs(panp)*PB + abs(pbnp)*PA)/(abs(panp)+abs(pbnp)) ;
		return EDGE_INTERSECTION;
	}
	else
		return NO_INTERSECTION;
#undef EPSILON
}



template <typename VEC3>
Intersection intersectionTrianglePlan(const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, const VEC3& PlaneP, const VEC3& NormP) //, VEC3& Inter) ;
{
	if((intersectionSegmentPlan<VEC3>(Ta,Tb,PlaneP, NormP) == EDGE_INTERSECTION)
			|| (intersectionSegmentPlan<VEC3>(Ta,Tc,PlaneP, NormP) == EDGE_INTERSECTION)
			|| (intersectionSegmentPlan<VEC3>(Tb,Tc,PlaneP, NormP)  == EDGE_INTERSECTION))
	{
		return FACE_INTERSECTION;
	}
	else if((intersectionSegmentPlan<VEC3>(Ta,Tb,PlaneP, NormP) == VERTEX_INTERSECTION)
			|| (intersectionSegmentPlan<VEC3>(Ta,Tc,PlaneP, NormP) == VERTEX_INTERSECTION)
			|| (intersectionSegmentPlan<VEC3>(Tb,Tc,PlaneP, NormP)  == VERTEX_INTERSECTION))
	{
		return VERTEX_INTERSECTION;
	}
	else
	{
		return NO_INTERSECTION;
	}
}

template <typename VEC3>
Intersection intersectionSegmentHalfPlan(const VEC3& PA, const VEC3& PB,
		const VEC3& P, const VEC3& DirP, const VEC3& OrientP)//, VEC3& Inter)
{
	VEC3 NormP = (DirP-P) ^ (OrientP-P) ;
	NormP.normalize() ;

	//intersection SegmentPlan
	Intersection inter = intersectionSegmentPlan(PA,PB,P,NormP);
	if(inter == EDGE_INTERSECTION)
	{
		//and one of the two points must be in the right side of the line
		return intersectionSegmentPlan(PA,PB, P, OrientP);
	}
	else
	{
		return inter;
	}



}

template <typename VEC3>
Intersection intersectionTriangleHalfPlan(const VEC3& Ta, const VEC3& Tb, const VEC3& Tc,
		const VEC3& P, const VEC3& DirP, const VEC3& OrientP) //, VEC3& Inter)
{
	if((intersectionSegmentHalfPlan<VEC3>(Ta,Tb,P, DirP, OrientP) == EDGE_INTERSECTION)
			|| (intersectionSegmentHalfPlan<VEC3>(Ta,Tc,P, DirP, OrientP) == EDGE_INTERSECTION)
			|| (intersectionSegmentHalfPlan<VEC3>(Tb,Tc,P, DirP, OrientP)  == EDGE_INTERSECTION))
	{
		return FACE_INTERSECTION;
	}
	else if((intersectionSegmentHalfPlan<VEC3>(Ta,Tb,P, DirP, OrientP) == VERTEX_INTERSECTION)
			|| (intersectionSegmentHalfPlan<VEC3>(Ta,Tc,P, DirP, OrientP) == VERTEX_INTERSECTION)
			|| (intersectionSegmentHalfPlan<VEC3>(Tb,Tb,P, DirP, OrientP)  == VERTEX_INTERSECTION))
	{
		return FACE_INTERSECTION;
	}
	else
	{
		return NO_INTERSECTION;
	}
}

template <typename VEC3>
bool interLineSeg(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2,
				  const VEC3& P, const VEC3& Q, VEC3& inter)
{
#define EPSILON (1.0e-5)
	typedef typename VEC3::DATA_TYPE T ;

	T dist = Geom::distancePoint2TrianglePlane(AB-A,A,P,Q);

//	std::cout << "dist "<<  dist << std::endl;

	if (dist>EPSILON)
		return false;

	VEC3 AP = P - A ;
	VEC3 PQ = Q - P ;
	T X = AB * PQ ;
	T beta = ( AB2 * (AP*PQ) - X * (AP*AB) ) / ( X*X - AB2 * PQ.norm2() ) ;

//	std::cout << "beta "<<  beta << std::endl;

	if ((beta<0.0) || (beta>1.0))
		return false;

	inter = beta*Q +(1.0-beta)*P;
	return true;
#undef EPSILON
}

}

}
