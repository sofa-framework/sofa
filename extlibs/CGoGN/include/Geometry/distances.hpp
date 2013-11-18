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

namespace CGoGN
{

namespace Geom
{

template <typename VEC3>
inline typename VEC3::DATA_TYPE squaredDistancePoint2TrianglePlane(const VEC3& P, const VEC3& A, const VEC3& B, const VEC3& C)
{
    VEC3 vAB = B - A ;
    VEC3 vAC = C - A ;
	VEC3 N = vAB ^ vAC ;

	typename VEC3::DATA_TYPE x = (P * N) - (A * N) ;
	return (x * x) / (N.norm2());
}

template <typename VEC3>
inline typename VEC3::DATA_TYPE distancePoint2TrianglePlane(const VEC3& P, const VEC3& A, const VEC3& B, const VEC3& C)
{
	Geom::Plane3D<typename VEC3::DATA_TYPE> plane(A,B,C) ;
	return plane.distance(P) ;
}

template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistancePoint2Triangle(const VEC3& P, const VEC3& A, const VEC3& B, const VEC3& C)
{
    VEC3 vPA = A - P ;
    VEC3 vAB = B - A ;
    VEC3 vAC = C - A ;
    double fA00 = vAB.norm2() ;
    double fA01 = vAB * vAC ;
    double fA11 = vAC.norm2() ;
    double fB0 = vPA * vAB ;
    double fB1 = vPA * vAC ;
    double fC = vPA.norm2() ;
    double fDet = fabs(fA00*fA11-fA01*fA01);
    double fS = fA01*fB1-fA11*fB0;
    double fT = fA01*fB0-fA00*fB1;
    double fSqrDistance;

    if (fS + fT <= fDet)
    {
        if (fS < 0.0f)
        {
            if (fT < 0.0f)  // region 4
            {
                if (fB0 < 0.0f)
                {
                    fT = 0.0f;
                    if (-fB0 >= fA00) { fS = 1.0f; fSqrDistance = fA00+(2.0f)*fB0+fC; }
                    else { fS = -fB0/fA00; fSqrDistance = fB0*fS+fC; }
                }
                else
                {
                    fS = 0.0f;
                    if (fB1 >= 0.0f) { fT = 0.0f; fSqrDistance = fC; }
                    else if (-fB1 >= fA11) { fT = 1.0f; fSqrDistance = fA11+(2.0f)*fB1+fC; }
                    else { fT = -fB1/fA11; fSqrDistance = fB1*fT+fC; }
                }
            }
            else  // region 3
            {
                fS = 0.0f;
                if (fB1 >= 0.0f) { fT = 0.0f; fSqrDistance = fC; }
                else if (-fB1 >= fA11) { fT = 1.0f; fSqrDistance = fA11+(2.0f)*fB1+fC; }
                else { fT = -fB1/fA11; fSqrDistance = fB1*fT+fC; }
            }
        }
        else if (fT < 0.0f)  // region 5
        {
            fT = 0.0f;
            if (fB0 >= 0.0f) { fS = 0.0f; fSqrDistance = fC; }
            else if (-fB0 >= fA00) { fS = 1.0f; fSqrDistance = fA00+(2.0f)*fB0+fC; }
            else { fS = -fB0/fA00; fSqrDistance = fB0*fS+fC; }
        }
        else  // region 0
        {
            // minimum at interior point
            double fInvDet = (1.0f)/fDet;
            fS *= fInvDet;
            fT *= fInvDet;
            fSqrDistance = fS*(fA00*fS+fA01*fT+(2.0f)*fB0) + fT*(fA01*fS+fA11*fT+(2.0f)*fB1)+fC;
        }
    }
    else
    {
        double fTmp0, fTmp1, fNumer, fDenom;

        if (fS < 0.0f)  // region 2
        {
            fTmp0 = fA01 + fB0;
            fTmp1 = fA11 + fB1;
            if (fTmp1 > fTmp0)
            {
                fNumer = fTmp1 - fTmp0;
                fDenom = fA00-2.0f*fA01+fA11;
                if (fNumer >= fDenom) { fS = 1.0f; fT = 0.0f; fSqrDistance = fA00+(2.0f)*fB0+fC; }
                else { fS = fNumer/fDenom; fT = 1.0f - fS; fSqrDistance = fS*(fA00*fS+fA01*fT+2.0f*fB0) + fT*(fA01*fS+fA11*fT+(2.0f)*fB1)+fC; }
            }
            else
            {
                fS = 0.0f;
                if (fTmp1 <= 0.0f) { fT = 1.0f; fSqrDistance = fA11+(2.0f)*fB1+fC; }
                else if (fB1 >= 0.0f) { fT = 0.0f; fSqrDistance = fC; }
                else { fT = -fB1/fA11; fSqrDistance = fB1*fT+fC; }
            }
        }
        else if (fT < 0.0f)  // region 6
        {
            fTmp0 = fA01 + fB1;
            fTmp1 = fA00 + fB0;
            if (fTmp1 > fTmp0)
            {
                fNumer = fTmp1 - fTmp0;
                fDenom = fA00-(2.0f)*fA01+fA11;
                if (fNumer >= fDenom) { fT = 1.0f; fS = 0.0f; fSqrDistance = fA11+(2.0f)*fB1+fC; }
                else { fT = fNumer/fDenom; fS = 1.0f - fT; fSqrDistance = fS*(fA00*fS+fA01*fT+(2.0f)*fB0) + fT*(fA01*fS+fA11*fT+(2.0f)*fB1)+fC; }
            }
            else
            {
                fT = 0.0f;
                if (fTmp1 <= 0.0f) { fS = 1.0f; fSqrDistance = fA00+(2.0f)*fB0+fC; }
                else if (fB0 >= 0.0f) { fS = 0.0f; fSqrDistance = fC; }
                else { fS = -fB0/fA00; fSqrDistance = fB0*fS+fC; }
            }
        }
        else  // region 1
        {
            fNumer = fA11 + fB1 - fA01 - fB0;
            if (fNumer <= 0.0f) { fS = 0.0f; fT = 1.0f; fSqrDistance = fA11+(2.0f)*fB1+fC; }
            else
            {
                fDenom = fA00-2.0f*fA01+fA11;
                if (fNumer >= fDenom) { fS = 1.0f; fT = 0.0f; fSqrDistance = fA00+(2.0f)*fB0+fC; }
                else { fS = fNumer/fDenom; fT = 1.0f - fS; fSqrDistance = fS*(fA00*fS+fA01*fT+(2.0f)*fB0) + fT*(fA01*fS+fA11*fT+(2.0f)*fB1)+fC; }
            }
        }
    }

    // account for numerical round-off error
    if (fSqrDistance < 0.0f)
        fSqrDistance = 0.0f;

    return fSqrDistance;
}

template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceLine2Point(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2, const VEC3& P)
{
	VEC3 V = A - P ;
	VEC3 W = AB ^ V ;
	return W.norm2() / AB2 ;
}

template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceLine2Point(const VEC3& A, const VEC3& B, const VEC3& P)
{
	VEC3 AB = B - A ;
	return squaredDistanceLine2Point(A, AB, AB.norm2(), P) ;
}

template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceLine2Line(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2, const VEC3& P, const VEC3& Q)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 PQ = Q - P ;
	VEC3 temp = AB ^ PQ ;
	T den = temp.norm2() ;

	if(den > 0)	// droites non //
	{
		T num = (P - A) * temp ;
		return (num*num) / den ;
	}
	else		// droites //
	{
		VEC3 AP = P - A ;
		VEC3 W = AB ^ AP ;
		T num = W.norm2() ;
		return num / AB2 ;
	}
}

template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceLine2Seg(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2, const VEC3& P, const VEC3& Q)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 PQ = Q - P ;
	T X = AB * PQ ;
	VEC3 AP = P - A ;

	T beta = ( AB2 * (AP*PQ) - X * (AP*AB) ) / ( X*X - AB2 * PQ.norm2() ) ;

	if(beta < T(0))
	{
		VEC3 W = AB ^ AP ;
		return W.norm2() / AB2 ;
	}

	if(beta > T(1))
	{
		VEC3 AQ = Q - A ;
		VEC3 W = AB ^ AQ ;
		return W.norm2() / AB2 ;
	}

	VEC3 temp = AB ^ PQ ;
	T num = AP * temp ;
	T den = temp.norm2() ;

	return (num*num) / den ;
}

template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceSeg2Point(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2, const VEC3& P)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 AP = P - A ;

	// position of projection of P on [A,B]
	T t = (AP * AB) / AB2 ;

	// before A, distance is PA
	if(t <= T(0))
		return AP.norm2() ;

	// after B, distantce is PB
	if(t >= T(1))
	{
		VEC3 BP = P - (AB + A) ;
		return BP.norm2() ;
	}

	// between A & B, distance is projection on (AB)
	VEC3 X = AB ^ AP ;
	return X.norm2() / AB2 ;
}


template <typename VEC3>
bool lineLineClosestPoints(const VEC3& P1, const VEC3& V1, const VEC3& P2, const VEC3& V2, VEC3& Q1, VEC3& Q2)
{
	   Geom::Vec3f P12 = P1 - P2;

	   float d1343 = P12 * V2;
	   float d4321 = V2*V1;
	   float d1321 = P12*V1;
	   float d4343 = V2*V2;
	   float d2121 = V1*V1;

	   float denom = d2121 * d4343 - d4321 * d4321;

	   if (fabs(denom) < 0.0000001)
	      return false;

	   float numer = d1343 * d4321 - d1321 * d4343;

	   float mua = numer / denom;
	   float mub = (d1343 + d4321 * mua) / d4343;

	   Q1 = P1 + mua*V1;
	   Q2 = P2 + mub*V2;
	   return true;
}


}

}
