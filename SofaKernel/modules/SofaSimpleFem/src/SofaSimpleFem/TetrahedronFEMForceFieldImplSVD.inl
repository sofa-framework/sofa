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

#include <SofaSimpleFem/TetrahedronFEMForceFieldImplSVD.h>

namespace sofa::component::forcefield
{

template <class DataTypes>
void TetrahedronFEMForceFieldImplSVD<DataTypes>::init(const FiniteElementArrays& finiteElementArrays)
{
    m_initialTransformation.clear();
    m_initialTransformation.reserve(finiteElementArrays.elements->size());

    Inherit::init(finiteElementArrays);

    const auto& positions = *finiteElementArrays.initialPoints;
    for (const auto& element : *finiteElementArrays.elements)
    {
        const Index a = element[0];
        const Index b = element[1];
        const Index c = element[2];
        const Index d = element[3];

        Transformation A;
        A[0] = positions[b]-positions[a];
        A[1] = positions[c]-positions[a];
        A[2] = positions[d]-positions[a];

        Transformation A_inverted;
        const bool canInvert = A_inverted.invert(A);
        assert(canInvert);
        SOFA_UNUSED(canInvert);

        m_initialTransformation.emplace_back(std::move(A_inverted));
    }
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplSVD<DataTypes>
::computeRotation(Transformation& rotation, const VecCoord& positions, const Index a, const Index b, const Index c, const Index d, unsigned elementIndex) const
{
    Transformation A;
    A[0] = positions[b]-positions[a];
    A[1] = positions[c]-positions[a];
    A[2] = positions[d]-positions[a];

    const auto F = A * this->m_initialTransformation[elementIndex];

    rotation.clear();

    if(type::determinant(F) < 1e-6 ) // inverted or too flat element -> SVD decomposition + handle degenerated cases
    {
        helper::Decompose<Real>::polarDecomposition_stable( F, rotation );
        rotation = rotation.multTransposed( this->m_initialRotations[elementIndex] );
    }
    else // not inverted & not degenerated -> classical polar
    {
        helper::Decompose<Real>::polarDecomposition( A, rotation );
    }
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplSVD<DataTypes>::computeInitialRotation(Transformation& rotation,
    const VecCoord& positions, Index a, Index b, Index c, Index d, unsigned elementIndex) const
{
    SOFA_UNUSED(elementIndex);

    Transformation A;
    A[0] = positions[b]-positions[a];
    A[1] = positions[c]-positions[a];
    A[2] = positions[d]-positions[a];

    helper::Decompose<Real>::polarDecomposition_stable( A, rotation );
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplSVD<DataTypes>::addForceAssembled(const FiniteElementArrays& finite_element_arrays)
{
    static bool first = true;
    if (first)
    {
        msg_error("TetrahedronFEMForceFieldImplSVD") << "Computing the assembled matrix is not supported with the SVD method";
        first = false;
    }
}
}
