/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef MAPPINGJACOBIAN_H
#define MAPPINGJACOBIAN_H

#include <Flexible/types/AffineTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{
namespace component
{
namespace mapping
{
	typedef defaulttype::Mat<3, 3, SReal> Mat3x3;
	typedef defaulttype::Mat<12, 3, SReal> Mat12x3;
	typedef defaulttype::Mat<12, 6, SReal> Mat12x6;

    typedef defaulttype::Mat<6, 3, SReal> Mat6x3;
    typedef defaulttype::Mat<6, 6, SReal> Mat6x6;

    //============================================================
    // Jacobian of RigidToAffineMultiMapping
    //============================================================
	// Computation of J1 = df(r,s)/dr
	template<class CoordIn1, class CoordIn2, class CoordOut>
    inline void computeFrameJacobianR(const CoordIn1& /*rigid*/, const CoordIn2& /*scale*/, const CoordOut& affine, Mat12x6& res)
	{
		// variable
		// Sw = Skew matrix, R = Rotation matrix, S = scale matrix, A = Affine matrix, R_dot = R*Sw
		CoordOut A(affine);

		// Computation of the Jacobian
		// dt/dt
        res[0][0] = (SReal)1.;  res[0][1] = 0;			res[0][2] = 0;
        res[1][0] = 0;			res[1][1] = (SReal)1.;  res[1][2] = 0;
		res[2][0] = 0;			res[2][1] = 0;			res[2][2] = (SReal)1.;

		// dt/dw
		res[0][3] = 0;			res[0][4] = 0;			res[0][5] = 0;
		res[1][3] = 0;			res[1][4] = 0;			res[1][5] = 0;
		res[2][3] = 0;			res[2][4] = 0;			res[2][5] = 0;

		// dR_dot/dt
		res[3][0] = 0;			res[3][1] = 0;			res[3][2] = 0;
		res[4][0] = 0;			res[4][1] = 0;			res[4][2] = 0;
		res[5][0] = 0;			res[5][1] = 0;			res[5][2] = 0;
		res[6][0] = 0;			res[6][1] = 0;			res[6][2] = 0;
		res[7][0] = 0;			res[7][1] = 0;			res[7][2] = 0;
		res[8][0] = 0;			res[8][1] = 0;			res[8][2] = 0;
		res[9][0] = 0;			res[9][1] = 0;			res[9][2] = 0;
		res[10][0] = 0;			res[10][1] = 0;			res[10][2] = 0;
		res[11][0] = 0;			res[11][1] = 0;			res[11][2] = 0;

		// dR_dot/dw
		res[3][3] = 0;			res[3][4] = A[9];		res[3][5] = -A[6];
		res[4][3] = 0;			res[4][4] = A[10];		res[4][5] = -A[7];
		res[5][3] = 0;			res[5][4] = A[11];		res[5][5] = -A[8];
		//-------------------------------------------------------------------
		res[6][3] = -A[9];		res[6][4] = 0;			res[6][5] = A[3];
		res[7][3] = -A[10];		res[7][4] = 0;			res[7][5] = A[4];
		res[8][3] = -A[11];		res[8][4] = 0;			res[8][5] = A[5];
		//-------------------------------------------------------------------
		res[9][3] = A[6];		res[9][4] = -A[3];		res[9][5] = 0;
		res[10][3] = A[7];		res[10][4] = -A[4];		res[10][5] = 0;
		res[11][3] = A[8];		res[11][4] = -A[5];		res[11][5] = 0;
		return;
	}

	// Computation of J2 = df(r,s)/ds
	template<class CoordIn1, class CoordIn2, class CoordOut>
    inline void computeFrameJacobianS(const CoordIn1& rigid, const CoordIn2& /*scale*/, CoordOut& /*output*/, Mat12x3& res)
	{
		// variable
		Mat3x3 R; // Sw = Skew matrix, R = Rotation matrix, S = scale matrix, A = Affine matrix, R_dot = R*Sw
		// Computation of rotation and scale matrices
        defaulttype::RigidTypes::Quat q = rigid.getOrientation();
		// -- rotation
		q.toMatrix(R);

		// Computation of the Jacobian
		// dt/ds
        res[0][0] = 0;      res[0][1] = 0;          res[0][2] = 0;
        res[1][0] = 0;		res[1][1] = 0;          res[1][2] = 0;
        res[2][0] = 0;		res[2][1] = 0;          res[2][2] = 0;

		// dR/ds
        res[3][0] = R[0][0];res[3][1] = 0;          res[3][2] = 0;
        res[4][0] = 0;		res[4][1] = R[0][1];    res[4][2] = 0;
        res[5][0] = 0;		res[5][1] = 0;          res[5][2] = R[0][2];
		//-------------------------------------------------------------------
        res[6][0] = R[1][0];res[6][1] = 0;          res[6][2] = 0;
        res[7][0] = 0;		res[7][1] = R[1][1];    res[7][2] = 0;
        res[8][0] = 0;		res[8][1] = 0;          res[8][2] = R[1][2];
		//-------------------------------------------------------------------
        res[9][0] = R[2][0];res[9][1] = 0;          res[9][2] = 0;
        res[10][0] = 0;		res[10][1] = R[2][1];   res[10][2] = 0;
        res[11][0] = 0;		res[11][1] = 0;         res[11][2] = R[2][2];

		return;
	}

    //============================================================
    // Jacobian of RigidScaleToRigidMultiMapping
    //============================================================
    template<class CoordIn1, class CoordIn2, class CoordOut>
    inline void computeFrameRigidJacobianR(const CoordIn1& rigid, const CoordIn2& scale, const CoordOut& relativeCoordinate, const CoordOut& /*affine*/, Mat6x6& res)
    {
        // Variabless
        Mat3x3 R, S;
        // Get important components
        defaulttype::Rigid3Types::Quat q = rigid.getOrientation();
        // Conversion of the rigid quaternion into a rotation matrix
        q.toMatrix(R);
        // Conversion of the scale into a 3x3 matrix
        for (unsigned int i = 0; i < 3; ++i) S[i][i] = scale[i];
        // Computation of the new position
        CoordIn2 t0_up = (R*S)*relativeCoordinate.getCenter();

        // Computation of the Jacobian
        // df1/dt
        res[0][0] = (SReal)1;   res[0][1] = 0;          res[0][2] = 0;
        res[1][0] = 0;          res[1][1] = (SReal)1;   res[1][2] = 0;
        res[2][0] = 0;          res[2][1] = 0;          res[2][2] = (SReal)1;

        // df1_dot/dw
        res[0][3] = 0;			res[0][4] = t0_up[2];	res[0][5] =-t0_up[1];
        res[1][3] =-t0_up[2];   res[1][4] = 0;			res[1][5] = t0_up[0];
        res[2][3] = t0_up[1];   res[2][4] =-t0_up[0];   res[2][5] = 0;

        // df2/dt
        res[3][0] = 0;			res[3][1] = 0;			res[3][2] = 0;
        res[4][0] = 0;			res[4][1] = 0;			res[4][2] = 0;
        res[5][0] = 0;			res[5][1] = 0;			res[5][2] = 0;

        // df2_dot/dw
        res[3][3] = (SReal)1;   res[3][4] = 0;          res[3][5] = 0;
        res[4][3] = 0;			res[4][4] = (SReal)1;   res[4][5] = 0;
        res[5][3] = 0;			res[5][4] = 0;          res[5][5] = (SReal)1;
    }

    template<class CoordIn1, class CoordIn2, class CoordOut>
    inline void computeFrameRigidJacobianS(const CoordIn1& rigid, const CoordIn2& /*scale*/, const CoordOut& relativeCoordinate, const CoordOut& /*affine*/, Mat6x3& res)
    {
        // Variabless
        Mat3x3 R;
        // Get important components
        defaulttype::Rigid3Types::Quat q = rigid.getOrientation();
        // Conversion of the rigid quaternion into a rotation matrix
        q.toMatrix(R);
        CoordIn2 t0(relativeCoordinate.getCenter());
        // Computation of the Jacobian
        // df1/ds
        res[0][0] = R[0][0]*t0[0];  res[0][1] = R[0][1]*t0[1];  res[0][2] = R[0][2]*t0[2];
        res[1][0] = R[1][0]*t0[0];  res[1][1] = R[1][1]*t0[1];  res[1][2] = R[1][2]*t0[2];
        res[2][0] = R[2][0]*t0[0];  res[2][1] = R[2][1]*t0[1];  res[2][2] = R[2][2]*t0[2];
        // df2/ds
        res[3][0] = 0;              res[3][1] = 0;              res[3][2] = 0;
        res[4][0] = 0;              res[4][1] = 0;              res[4][2] = 0;
        res[5][0] = 0;              res[5][1] = 0;              res[5][2] = 0;
    }

}// mapping
}// component
}// sofa

#endif // MAPPINGJACOBIAN_H
