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
#include <sofa/component/solidmechanics/fem/elastic/TriangleFEMUtils.h>

namespace sofa::component::solidmechanics::fem::elastic
{


////////////// small displacements method

// ---------------------------------------------------------------------------------------------------------------
// ---	Compute displacement vector D as the difference between current position 'p' and initial position
// 
// Notes: Displacement is computed in local frame. Current position coordinates in this frame:
//  deforme_a = pA - pA = 0
//  deforme_b = pB - pA = pAB
//  deforme_c = pB - pA = pAC
// ---------------------------------------------------------------------------------------------------------------
template<class DataTypes>
constexpr void TriangleFEMUtils<DataTypes>::computeDisplacementSmall(Displacement& D, const type::fixed_array<Coord, 3>& rotatedInitCoord, const Coord& pAB, const Coord& pAC) const
{
    D[0] = 0;
    D[1] = 0;
    D[2] = rotatedInitCoord[1][0] - pAB[0];
    D[3] = rotatedInitCoord[1][1] - pAB[1];
    D[4] = rotatedInitCoord[2][0] - pAC[0];
    D[5] = rotatedInitCoord[2][1] - pAC[1];
}


////////////// large displacements method


// ---------------------------------------------------------------------------------------------------------------
// --- Compute rotation matrix to change to base composed of [pA, pB, pC]
// first vector on first edge
// second vector in the plane of the two first edges
// third vector orthogonal to first and second
// ---------------------------------------------------------------------------------------------------------------
template<class DataTypes>
constexpr void TriangleFEMUtils<DataTypes>::computeRotationLarge(Transformation& r, const Coord& pA, const Coord& pB, const Coord& pC) const
{
    const Coord edgex = (pB - pA).normalized();
    Coord edgey = pC - pA;
    const Coord edgez = cross(edgex, edgey).normalized();
    edgey = cross(edgez, edgex); //edgey is unit vector because edgez and edgex are orthogonal unit vectors

    r(0,0) = edgex[0];
    r(0,1) = edgex[1];
    r(0,2) = edgex[2];
    r(1,0) = edgey[0];
    r(1,1) = edgey[1]; 
    r(1,2) = edgey[2];
    r(2,0) = edgez[0];
    r(2,1) = edgez[1];
    r(2,2) = edgez[2];
}



// -------------------------------------------------------------------------------------------------------------
// --- Compute displacement vector D as the difference between current position 'p' and initial position
// --- expressed in the co-rotational frame of reference
// -------------------------------------------------------------------------------------------------------------
template<class DataTypes>
constexpr void TriangleFEMUtils<DataTypes>::computeDisplacementLarge(Displacement& D, const Transformation& R_0_2, const type::fixed_array<Coord, 3>& rotatedInitCoord, const Coord& pA, const Coord& pB, const Coord& pC) const
{
    // positions of the deformed and displaced triangle in its local frame
    const Coord deforme_b = R_0_2 * (pB - pA);
    const Coord deforme_c = R_0_2 * (pC - pA);

    // displacements in the local frame
    D[0] = 0;
    D[1] = 0;
    D[2] = rotatedInitCoord[1][0] - deforme_b[0];
    D[3] = 0;
    D[4] = rotatedInitCoord[2][0] - deforme_c[0];
    D[5] = rotatedInitCoord[2][1] - deforme_c[1];
}



// --------------------------------------------------------------------------------------
// ---	Compute force as: F = J * stress
// Notes: Optimisations: The following values are 0 (per computeStrainDisplacement )
// \       0        1        2
// 0   J(0,0)      0      J(0,2)
// 1       0     J(1,1)   J(1,2)
// 2   J(2,0)      0      J(2,2)
// 3       0     J(3,1)   J(3,2)
// 4       0        0      J(4,2)
// 5       0     J(5,1)     0
// --------------------------------------------------------------------------------------
template<class DataTypes>
constexpr void TriangleFEMUtils<DataTypes>::computeForceLarge(Displacement& F, const StrainDisplacement& J, const type::Vec<3, Real>& stress) const
{
    F[0] = J(0,0) * stress[0] + J(0,2) * stress[2];
    F[1] = J(1,1) * stress[1] + J(1,2) * stress[2];
    F[2] = J(2,0) * stress[0] + J(2,2) * stress[2];
    F[3] = J(3,1) * stress[1] + J(3,2) * stress[2];
    F[4] = J(4,2) * stress[2];
    F[5] = J(5,1) * stress[1];
}



// ------------------------------------------------------------------------------------------------------------
// --- Compute the strain-displacement matrix where (pA, pB, pC) are the coordinates of the 3 nodes of a triangle
// ------------------------------------------------------------------------------------------------------------
template<class DataTypes>
constexpr void TriangleFEMUtils<DataTypes>::computeStrainDisplacementGlobal(StrainDisplacement& J, const Coord& pA, const Coord& pB, const Coord& pC) const
{
    const Coord ab_cross_ac = cross(pB - pA, pC - pA);
    const Real determinant = ab_cross_ac.norm();
    
    if (fabs(determinant) < std::numeric_limits<Real>::epsilon())
    {
        msg_error("TriangleFEMUtils") << "Null determinant in computeStrainDisplacementGlobal: " << determinant;
        throw std::logic_error("Division by zero exception in computeStrainDisplacementGlobal ");
    }

    const Real invDet = 1 / determinant;

    const Real x13 = (pA[0] - pC[0]) * invDet;
    const Real x21 = (pB[0] - pA[0]) * invDet;
    const Real x32 = (pC[0] - pB[0]) * invDet;
    const Real y12 = (pA[1] - pB[1]) * invDet;
    const Real y23 = (pB[1] - pC[1]) * invDet;
    const Real y31 = (pC[1] - pA[1]) * invDet;

    J(0,0) = y23;
    J(0,1) = 0;
    J(0,2) = x32;

    J(1,0) = 0;
    J(1,1) = x32;
    J(1,2) = y23;

    J(2,0) = y31;
    J(2,1) = 0;
    J(2,2) = x13;

    J(3,0) = 0;
    J(3,1) = x13;
    J(3,2) = y31;

    J(4,0) = y12;
    J(4,1) = 0;
    J(4,2) = x21;

    J(5,0) = 0;
    J(5,1) = x21;
    J(5,2) = y12;
}



// --------------------------------------------------------------------------------------------------------------------------
// --- Compute the strain-displacement matrix where (pB, pC) are the coordinates of the 2 nodes of a triangle in local space
// Notes: The following formulation is actually equivalent:
// Let
// | alpha1 alpha2 alpha3 |                      | 1 xa ya |
// | beta1  beta2  beta3  | = be the inverse of  | 1 xb yb |
// | gamma1 gamma2 gamma3 |                      | 1 xc yc |
//
// The strain - displacement matrix is :
// | beta1  0       beta2  0        beta3  0      |
// | 0      gamma1  0      gamma2   0      gamma3 | / (2 * A)
// | gamma1 beta1   gamma2 beta2    gamma3 beta3  |
// where A is the area of the triangle and 2 * A is the determinant of the matrix with the xa, ya, xb...
//
// Since a0 = a1 = b1 = 0, the matrix is triangular and its inverse is :
// | 1                0              0    |
// | -1/xb            1/xb           0    |
// | -(1-xc/xb)/yc    -xc/(xb*yc)    1/yc |
//
// Our strain - displacement matrix is :
// | -1/xb           0             1/xb         0             0         0    |
// | 0              -(1-xc/xb)/yc  0            -xc/(xb*yc)   0         1/yc |
// | -(1-xc/xb)/yc  -1/xb          -xc/(xb*yc)  1/xb          1/yc      0    |
// 
// Then:
//  Real beta1  = -1/b[0]
//  Real beta2  =  1/b[0]
//  Real gamma1 = (c[0]/b[0]-1)/c[1]
//  Real gamma2 = -c[0]/(b[0]*c[1])
//  Real gamma3 = 1/c[1]
//
// The transpose of the strain-displacement matrix is thus:
//  J(0,0) = J(1,2) = beta1
//  J(0,1) = J(1,0) = 0
//  J(0,2) = J(1,1) = gamma1
//
//  J(2,0) = J(3,2) = beta2
//  J(2,1) = J(3,0) = 0
//  J(2,2) = J(3,1) = gamma2
//
//  J(4,0) = J(5,2) = 0
//  J(4,1) = J(5,0) = 0
//  J(4,2) = J(5,1) = gamma3
// --------------------------------------------------------------------------------------------------------------------------
template<class DataTypes>
constexpr void TriangleFEMUtils<DataTypes>::computeStrainDisplacementLocal(StrainDisplacement& J, const Coord& pB, const Coord& pC) const
{
    // local computation taking into account that a = [0, 0, 0], b = [x, 0, 0], c = [y, y, 0]
    const Real determinant = pB[0] * pC[1];
    
    if (fabs(determinant) < std::numeric_limits<Real>::epsilon())
    {
        msg_error("TriangleFEMUtils") << "Null determinant in computeStrainDisplacementLocal: " << determinant;
        throw std::logic_error("Division by zero exception in computeStrainDisplacementLocal");
    }
    const Real invDet = 1 / determinant;

    J(0,0) = J(1,2) = -pC[1] * invDet;
    J(0,2) = J(1,1) = (pC[0] - pB[0]) * invDet;
    J(2,0) = J(3,2) = pC[1] * invDet;
    J(2,2) = J(3,1) = -pC[0] * invDet;
    J(4,0) = J(5,2) = 0;
    J(4,2) = J(5,1) = pB[0] * invDet;
    J(1,0) = J(3,0) = J(5,0) = J(0,1) = J(2,1) = J(4,1) = 0;
}



// --------------------------------------------------------------------------------------------------------
// --- Strain = StrainDisplacement (Jt) * Displacement (D) = JtD = Bd
// Notes: Optimisations (@param fullMethod = false): The following values are 0 (per StrainDisplacement )
// | \        0        1        2        3        4        5      |
// | 0    Jt(0,0)     0    Jt(0,2)     0        0        0      |
// | 1        0    Jt(1,1)     0     Jt(1,3)    0     Jt(1,5)  |
// | 2    Jt(2,0) Jt(2,1) Jt(2,2)  Jt(2,3)  Jt(2,4)   0      |
// --------------------------------------------------------------------------------------------------------
template<class DataTypes>
constexpr void TriangleFEMUtils<DataTypes>::computeStrain(type::Vec<3, Real>& strain, const StrainDisplacement& J, const Displacement& D, bool fullMethod) const
{    
    if (fullMethod) // _anisotropicMaterial or SMALL case
    {
        strain = J.multTranspose(D);
    }
    else
    {
        // Use directly J to avoid computing Jt
        strain[0] = J(0,0) * D[0] + J(2,0) * D[2];
        strain[1] = J(1,1) * D[1] + J(3,1) * D[3] + J(5,1) * D[5];
        strain[2] = J(0,2) * D[0] + J(1,2) * D[1] + J(2,2) * D[2] + J(3,2) * D[3] + J(4,2) * D[4];
    }
}



// --------------------------------------------------------------------------------------------------------
// --- Stress = MaterialStiffnesses (K) * Strain = KJtD = KBd
// Notes: Optimisations (@param fullMethod = false): The following values are 0 (per MaterialStiffnesses )
// | \       0        1        2     |
// | 0   K(0,0)    K(0,1)    0     |
// | 1   K(1,0)    K(1,1)    0     |
// | 2       0        0      K(2,2) |
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
constexpr void TriangleFEMUtils<DataTypes>::computeStress(type::Vec<3, Real>& stress, const MaterialStiffness& K, const type::Vec<3, Real>& strain, bool fullMethod) const
{
    if (fullMethod) // _anisotropicMaterial or SMALL case
    {
        stress = K * strain;
    }
    else
    {
        stress[0] = K(0,0) * strain[0] + K(0,1) * strain[1];
        stress[1] = K(1,0) * strain[0] + K(1,1) * strain[1];
        stress[2] = K(2,2) * strain[2];
    }
}


} //namespace sofa::component::solidmechanics::fem::elastic
