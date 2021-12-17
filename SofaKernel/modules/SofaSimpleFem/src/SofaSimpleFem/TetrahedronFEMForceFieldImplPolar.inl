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

#include <SofaSimpleFem/TetrahedronFEMForceFieldImplPolar.h>

namespace sofa::component::forcefield
{

template <class DataTypes>
void TetrahedronFEMForceFieldImplPolar<DataTypes>::computeRotation(Transformation& rotation, const VecCoord& positions, const Index a, const Index b, const Index c, const Index d, unsigned elementIndex) const
{
    Transformation A;
    A[0] = positions[b]-positions[a];
    A[1] = positions[c]-positions[a];
    A[2] = positions[d]-positions[a];

    helper::Decompose<Real>::polarDecomposition( A, rotation );
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplPolar<DataTypes>
::computeDisplacement(Displacement& displacement, const type::fixed_array<type::VecNoInit<3, Real>, 4>& deformed_element,
    const type::fixed_array<Coord, 4>& rotated_initial_element) const
{
    displacement[ 0] = rotated_initial_element[0][0] - deformed_element[0][0];
    displacement[ 1] = rotated_initial_element[0][1] - deformed_element[0][1];
    displacement[ 2] = rotated_initial_element[0][2] - deformed_element[0][2];
    displacement[ 3] = rotated_initial_element[1][0] - deformed_element[1][0];
    displacement[ 4] = rotated_initial_element[1][1] - deformed_element[1][1];
    displacement[ 5] = rotated_initial_element[1][2] - deformed_element[1][2];
    displacement[ 6] = rotated_initial_element[2][0] - deformed_element[2][0];
    displacement[ 7] = rotated_initial_element[2][1] - deformed_element[2][1];
    displacement[ 8] = rotated_initial_element[2][2] - deformed_element[2][2];
    displacement[ 9] = rotated_initial_element[3][0] - deformed_element[3][0];
    displacement[10] = rotated_initial_element[3][1] - deformed_element[3][1];
    displacement[11] = rotated_initial_element[3][2] - deformed_element[3][2];
}

template <class DataTypes>
auto TetrahedronFEMForceFieldImplPolar<DataTypes>::computeRotatedInitialElement(
    const Transformation& rotation,
    const Element& element,
    const VecCoord& initialPoints) const
-> type::fixed_array<Coord,4>
{
    const Index a = element[0];
    const Index b = element[1];
    const Index c = element[2];
    const Index d = element[3];

    return
    {
        rotation * initialPoints[a],
        rotation * initialPoints[b],
        rotation * initialPoints[c],
        rotation * initialPoints[d]
    };
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplPolar<
DataTypes>::computeDeformedElement(type::fixed_array<type::VecNoInit<3, Real>,4>& deforme, const Transformation& rotation, const Coord& xa, const Coord& xb, const Coord& xc,
    const Coord& xd) const
{
    deforme[0] = rotation * xa;
    deforme[1] = rotation * xb;
    deforme[2] = rotation * xc;
    deforme[3] = rotation * xd;
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplPolar<DataTypes>::addForceAssembled(const FiniteElementArrays& finite_element_arrays)
{
    static bool first = true;
    if (first)
    {
        msg_error("TetrahedronFEMForceFieldImplPolar") << "Computing the assembled matrix is not supported with the polar method";
        first = false;
    }
}
}
