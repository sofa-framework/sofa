/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <SofaSimpleFem/TriangleFEMUtils.h>

namespace sofa::component::forcefield
{


////////////// small displacements method





////////////// large displacements method

template<class DataTypes>
void TriangleFEMUtils<DataTypes>::computeRotationLarge(Transformation& r, const Coord& pA, const Coord& pB, const Coord& pC)
{
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    /// first vector on first edge
    /// second vector in the plane of the two first edges
    /// third vector orthogonal to first and second
    const Coord edgex = (pB - pA).normalized();
    Coord edgey = pC - pA;
    const Coord edgez = cross(edgex, edgey).normalized();
    edgey = cross(edgez, edgex); //edgey is unit vector because edgez and edgex are orthogonal unit vectors

    r[0][0] = edgex[0];
    r[0][1] = edgex[1];
    r[0][2] = edgex[2];
    r[1][0] = edgey[0];
    r[1][1] = edgey[1];
    r[1][2] = edgey[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];
}


template<class DataTypes>
void TriangleFEMUtils<DataTypes>::computeStrainDisplacementGlobal(StrainDisplacement& J, SReal& area, const Coord& pA, const Coord& pB, const Coord& pC)
{
    Coord ab_cross_ac = cross(pB - pA, pC - pA);
    Real determinant = ab_cross_ac.norm();
    area = determinant * 0.5f;

    Real x13 = (pA[0] - pC[0]) / determinant;
    Real x21 = (pB[0] - pA[0]) / determinant;
    Real x32 = (pC[0] - pB[0]) / determinant;
    Real y12 = (pA[1] - pB[1]) / determinant;
    Real y23 = (pB[1] - pC[1]) / determinant;
    Real y31 = (pC[1] - pA[1]) / determinant;

    J[0][0] = y23;
    J[0][1] = 0;
    J[0][2] = x32;

    J[1][0] = 0;
    J[1][1] = x32;
    J[1][2] = y23;

    J[2][0] = y31;
    J[2][1] = 0;
    J[2][2] = x13;

    J[3][0] = 0;
    J[3][1] = x13;
    J[3][2] = y31;

    J[4][0] = y12;
    J[4][1] = 0;
    J[4][2] = x21;

    J[5][0] = 0;
    J[5][1] = x21;
    J[5][2] = y12;
}


template<class DataTypes>
void TriangleFEMUtils<DataTypes>::computeStrainDisplacementLocal(StrainDisplacement& J, SReal& area, const Coord& pB, const Coord& pC)
{
    // local computation taking into account that a = [0, 0, 0]
    Real determinant = pB[0] * pC[1]; // b = [x, 0, 0], c = [y, y, 0]
    area = determinant*0.5f;

    /* The following formulation is actually equivalent:
      Let
      | alpha1 alpha2 alpha3 |                      | 1 xa ya |
      | beta1  beta2  beta3  | = be the inverse of  | 1 xb yb |
      | gamma1 gamma2 gamma3 |                      | 1 xc yc |
      The strain-displacement matrix is:
      | beta1  0       beta2  0        beta3  0      |
      | 0      gamma1  0      gamma2   0      gamma3 | / (2*A)
      | gamma1 beta1   gamma2 beta2    gamma3 beta3  |
      where A is the area of the triangle and 2*A is the determinant of the matrix with the xa,ya,xb...
      Since a0=a1=b1=0, the matrix is triangular and its inverse is:
      |  1              0              0  |
      | -1/xb           1/xb           0  |
      | -(1-xc/xb)/yc  -xc/(xb*yc)   1/yc |
      our strain-displacement matrix is:
      | -1/xb           0             1/xb         0            0     0    |
      | 0              -(1-xc/xb)/yc  0            -xc/(xb*yc)  0     1/yc |
      | -(1-xc/xb)/yc  -1/xb          -xc/(xb*yc)  1/xb         1/yc  0    |
      */

  //    Real beta1  = -1/pB[0]; = -1 / pB[0] * 1 / 2*A = -1 /(pB[0] * (pB[0] * pC[1]))
  //    Real beta2  =  1/pB[0];
  //    Real gamma1 = (pC[0]/pB[0]-1)/pC[1];
  //    Real gamma2 = -pC[0]/(pB[0]*pC[1]);
  //    Real gamma3 = 1/pC[1];

  //    // The transpose of the strain-displacement matrix is thus:
  //    J[0][0] = J[1][2] = beta1;
  //    J[0][1] = J[1][0] = 0;
  //    J[0][2] = J[1][1] = gamma1;

  //    J[2][0] = J[3][2] = beta2;
  //    J[2][1] = J[3][0] = 0;
  //    J[2][2] = J[3][1] = gamma2;

  //    J[4][0] = J[5][2] = 0;
  //    J[4][1] = J[5][0] = 0;
  //    J[4][2] = J[5][1] = gamma3;

    J[0][0] = J[1][2] = -pC[1] / determinant;
    J[0][2] = J[1][1] = (pC[0] - pB[0]) / determinant;
    J[2][0] = J[3][2] = pC[1] / determinant;
    J[2][2] = J[3][1] = -pC[0] / determinant;
    J[4][0] = J[5][2] = 0;
    J[4][2] = J[5][1] = pB[0] / determinant;
    J[1][0] = J[3][0] = J[5][0] = J[0][1] = J[2][1] = J[4][1] = 0;
}


// --------------------------------------------------------------------------------------------------------
// --- Strain = StrainDisplacement * Displacement = JtD = Bd
// --------------------------------------------------------------------------------------------------------
template<class DataTypes>
void TriangleFEMUtils<DataTypes>::computeStrain(type::Vec<3, Real>& strain, const StrainDisplacement& J, const Displacement& D, bool fullMethod)
{
    type::Mat<3, 6, Real> Jt;
    Jt.transpose(J);

    if (fullMethod) // _anisotropicMaterial or SMALL case
    {
        strain = Jt * D;
    }
    else
    {        
        // Optimisations: The following values are 0 (per StrainDisplacement )
        // \        0        1        2        3        4        5
        // 0    Jt[0][0]     0    Jt[0][2]     0        0        0
        // 1        0    Jt[1][1]     0     Jt[1][3]    0     Jt[1][5]
        // 2    Jt[2][0] Jt[2][1] Jt[2][2]  Jt[2][3]  Jt[2][4]   0

        // Strain = StrainDisplacement (Jt) * Displacement (D)
        strain[0] = Jt[0][0] * D[0] + Jt[0][2] * D[2];
        strain[1] = Jt[1][1] * D[1] + Jt[1][3] * D[3] + Jt[1][5] * D[5];
        strain[2] = Jt[2][0] * D[0] + Jt[2][1] * D[1] + Jt[2][2] * D[2] + Jt[2][3] * D[3] + Jt[2][4] * D[4];
    }
}


// --------------------------------------------------------------------------------------------------------
// --- Stress = K * Strain = KJtD = KBd
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangleFEMUtils<DataTypes>::computeStress(type::Vec<3, Real>& stress, const MaterialStiffness& K, const type::Vec<3, Real>& strain, bool fullMethod)
{
    if (fullMethod) // _anisotropicMaterial or SMALL case
    {
        stress = K * strain;
    }
    else
    {
        // Optimisations: The following values are 0 (per MaterialStiffnesses )
        // \       0        1        2
        // 0   K[0][0]    K[0][1]    0
        // 1   K[1][0]    K[1][1]    0 
        // 2       0        0      K[2][2]

        // stress = MaterialStiffnesses (K) * strain
        stress[0] = K[0][0] * strain[0] + K[0][1] * strain[1];
        stress[1] = K[1][0] * strain[0] + K[1][1] * strain[1];
        stress[2] = K[2][2] * strain[2];
    }
}


} //namespace sofa::component::forcefield
