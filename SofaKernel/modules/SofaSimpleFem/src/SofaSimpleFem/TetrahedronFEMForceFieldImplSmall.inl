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
#include <SofaSimpleFem/TetrahedronFEMForceFieldImplSmall.h>

namespace sofa::component::forcefield
{

template <class DataTypes>
void TetrahedronFEMForceFieldImplSmall<DataTypes>::init(const FiniteElementArrays& finiteElementArrays)
{
    unsigned int elementIndex {};
    const auto& initialPoints = *finiteElementArrays.initialPoints;
    auto& strainDisplacements = *finiteElementArrays.strainDisplacements;

    for (const auto element : *finiteElementArrays.elements)
    {
        const Index a = element[0];
        const Index b = element[1];
        const Index c = element[2];
        const Index d = element[3];

        strainDisplacements[elementIndex++] = Inherit::computeStrainDisplacement(
            initialPoints[a], initialPoints[b], initialPoints[c], initialPoints[d] );
        // msg_info("tetra") << "strain displacement " << elementIndex-1 << ": " << strainDisplacements[elementIndex-1];
    }
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplSmall<DataTypes>::addForce(const FiniteElementArrays& finiteElementArrays)
{
    if (!this->m_assemble)
    {
        unsigned int elementIndex {};
        for (const auto& element : *finiteElementArrays.elements)
        {
            addForceElementElastic(finiteElementArrays, element, elementIndex++);
        }
    }
    else
    {
        addForceAssembled(finiteElementArrays);
    }
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplSmall<DataTypes>::addDForce(const FiniteElementArrays& finiteElementArrays,
    Real kFactor)
{
    const auto& dx = *finiteElementArrays.dx;

    unsigned int elementIndex {};
    for (const auto& element : *finiteElementArrays.elements)
    {
        Displacement displacement;
        for (unsigned int node = 0; node < element.static_size; ++node)
        {
            const Index nodeId = element[node];
            const auto& dxNode = dx[nodeId];
            for (unsigned int i = 0; i < Deriv::total_size; ++i)
            {
                displacement[node * Deriv::total_size + i] = dxNode[i];
            }
        }

        Force F;
        Inherit::computeForce(F,
            displacement,
            (*finiteElementArrays.plasticStrains)[elementIndex],
            (*finiteElementArrays.materialsStiffnesses)[elementIndex],
            (*finiteElementArrays.strainDisplacements)[elementIndex],
            this->m_plasticity);
        F *= - kFactor;

        Inherit::copyForceFromElementToGlobal(*finiteElementArrays.dForce, F, element);

        ++elementIndex;
    }
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplSmall<DataTypes>::addForceAssembled(const FiniteElementArrays& finiteElementArrays)
{
    unsigned int elementIndex {};
    if (!this->m_plasticity.hasPlasticity())
    {
        Inherit::initStiffnesses(finiteElementArrays);

        for (const auto element : *finiteElementArrays.elements)
        {
            addForceElementAssembled(finiteElementArrays, element, elementIndex++);
        }
    }
    else
    {
        static bool first = true;
        if (first)
        {
            msg_error("TetrahedronFEMForceFieldImpl") << "Assembled matrix with plasticity is not supported";
            first = false;
        }
    }
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplSmall<DataTypes>::addForceElementElastic(
    const FiniteElementArrays& finiteElementArrays, const Element& element, const unsigned elementIndex)
{
    const auto displacement = computeDisplacement(
        *finiteElementArrays.initialPoints,
        *finiteElementArrays.positions,
        element);

    Force F;
    Inherit::computeForce(F,
        displacement,
        (*finiteElementArrays.plasticStrains)[elementIndex],
        (*finiteElementArrays.materialsStiffnesses)[elementIndex],
        (*finiteElementArrays.strainDisplacements)[elementIndex],
        this->m_plasticity);

    Inherit::copyForceFromElementToGlobal(*finiteElementArrays.force, F, element);
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplSmall<DataTypes>::addForceElementAssembled(
    const FiniteElementArrays& finiteElementArrays, const Element& element, const unsigned elementIndex)
{
    const auto displacement = computeDisplacement(
        *finiteElementArrays.initialPoints,
        *finiteElementArrays.positions,
        element);

    const auto JKJt = this->assembleStiffnessMatrix(finiteElementArrays, element, elementIndex);

    const auto F = JKJt * displacement;

    Inherit::copyForceFromElementToGlobal(*finiteElementArrays.force, F, element);
}

template <class DataTypes>
typename TetrahedronFEMForceFieldImpl<DataTypes>::Displacement TetrahedronFEMForceFieldImplSmall<DataTypes>::
computeDisplacement(const VecCoord& initialPoints, const VecCoord& positions, const Element& element)
{
    const Index a = element[0];
    const Index b = element[1];
    const Index c = element[2];
    const Index d = element[3];

    return computeDisplacement(
        initialPoints[a], initialPoints[b], initialPoints[c], initialPoints[d],
        positions[a], positions[b], positions[c], positions[d]);
}

template <class DataTypes>
typename TetrahedronFEMForceFieldImpl<DataTypes>::Displacement TetrahedronFEMForceFieldImplSmall<DataTypes>::
computeDisplacement(const Coord& a_0, const Coord& b_0, const Coord& c_0, const Coord& d_0, const Coord& a,
    const Coord& b, const Coord& c, const Coord& d)
{
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] =  b_0[0] - a_0[0] - b[0] + a[0];
    D[4] =  b_0[1] - a_0[1] - b[1] + a[1];
    D[5] =  b_0[2] - a_0[2] - b[2] + a[2];
    D[6] =  c_0[0] - a_0[0] - c[0] + a[0];
    D[7] =  c_0[1] - a_0[1] - c[1] + a[1];
    D[8] =  c_0[2] - a_0[2] - c[2] + a[2];
    D[9] =  d_0[0] - a_0[0] - d[0] + a[0];
    D[10] = d_0[1] - a_0[1] - d[1] + a[1];
    D[11] = d_0[2] - a_0[2] - d[2] + a[2];
    return D;
}

}
