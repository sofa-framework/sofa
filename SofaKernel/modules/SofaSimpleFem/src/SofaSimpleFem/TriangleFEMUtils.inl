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
void TriangleFEMUtils<DataTypes>::computeStrainDisplacementLarge(StrainDisplacement& J, Index elementIndex, Coord a, Coord b, Coord c)
{

}



} //namespace sofa::component::forcefield
