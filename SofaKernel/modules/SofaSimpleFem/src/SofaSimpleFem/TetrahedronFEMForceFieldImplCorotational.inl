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

#include <SofaSimpleFem/TetrahedronFEMForceFieldImplCorotational.h>

namespace sofa::component::forcefield
{

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::init(const FiniteElementArrays& finiteElementArrays)
{
    const auto& initialPoints = *finiteElementArrays.initialPoints;

    m_initialRotations.clear();
    m_initialRotations.reserve(finiteElementArrays.elements->size());

    m_rotations.clear();
    m_rotations.reserve(finiteElementArrays.elements->size());

    m_rotatedInitialElements.clear();
    m_rotatedInitialElements.reserve(finiteElementArrays.elements->size());

    auto& strainDisplacements = *finiteElementArrays.strainDisplacements;
    strainDisplacements.resize(finiteElementArrays.elements->size());

    unsigned int elementIndex {};
    for (const auto element : *finiteElementArrays.elements)
    {
        initElement(initialPoints, strainDisplacements[elementIndex], element, elementIndex);
        ++elementIndex;
    }
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::initElement(const VecCoord& initialPoints, StrainDisplacement& strainDisplacement, const Element& element, unsigned int elementIndex)
{
    const Index a = element[0];
    const Index b = element[1];
    const Index c = element[2];
    const Index d = element[3];

    Transformation R_0_1;
    computeInitialRotation(R_0_1, initialPoints, a, b, c, d, elementIndex);

    Transformation R_0_1_T;
    R_0_1_T.transpose(R_0_1);

    m_initialRotations.push_back(R_0_1_T);
    m_rotations.push_back(R_0_1_T);

    const auto rotatedElement = computeRotatedInitialElement(R_0_1, element, initialPoints);
    // msg_info("tetra") << "rotated initial element " << elementIndex << ": " << rotatedElement;

    m_rotatedInitialElements.push_back(rotatedElement);

    strainDisplacement = Inherit::computeStrainDisplacement(
        rotatedElement[0], rotatedElement[1], rotatedElement[2], rotatedElement[3] );

    // msg_info("tetra") << "strain displacement " << elementIndex << ": " << strainDisplacement;
}

template <class DataTypes>
auto TetrahedronFEMForceFieldImplCorotational<DataTypes>::computeRotatedInitialElement(
    const Transformation& rotation,
    const Element& element,
    const VecCoord& initialPoints) const
-> type::fixed_array<Coord,4>
{
    const Index a = element[0];
    const Index b = element[1];
    const Index c = element[2];
    const Index d = element[3];

    const auto firstNodeRotated = rotation * initialPoints[a];
    return
    {
        Coord(0,0,0),
          rotation * initialPoints[b] - firstNodeRotated,
          rotation * initialPoints[c] - firstNodeRotated,
          rotation * initialPoints[d] - firstNodeRotated
    };
}

template <class DataTypes>
typename TetrahedronFEMForceFieldImpl<DataTypes>::StiffnessMatrix TetrahedronFEMForceFieldImplCorotational<DataTypes>::
computeStiffnessMatrix(const MaterialStiffness& K, const StrainDisplacement& J, const unsigned elementIndex) const
{
    const auto JKJt = Inherit::computeStiffnessMatrix(K, J, elementIndex);

    const auto& Rot = m_rotations[elementIndex];

    type::MatNoInit<12, 12, Real> RR, RRt;
    RR.clear();
    RRt.clear();

    for(int i=0; i<3; ++i)
    {
        for(int j=0; j<3; ++j)
        {
            RR[i][j] = RR[i+3][j+3] = RR[i+6][j+6] = RR[i+9][j+9] = Rot[i][j];
            RRt[i][j]= RRt[i+3][j+3]= RRt[i+6][j+6]= RRt[i+9][j+9]= Rot[j][i];
        }
    }

    return RR * JKJt * RRt;
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::addForce(const FiniteElementArrays& finiteElementArrays)
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
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::computeInitialRotation(Transformation& rotation,
    const VecCoord& positions, Index a, Index b, Index c, Index d, unsigned elementIndex) const
{
    return computeRotation(rotation, positions, a, b, c, d, elementIndex);
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::addForceElementElastic(
    const FiniteElementArrays& finiteElementArrays, const Element& element, const unsigned elementIndex)
{
    const Index a = element[0];
    const Index b = element[1];
    const Index c = element[2];
    const Index d = element[3];

    const auto& p = *finiteElementArrays.positions;

    Transformation R_0_2;
    computeRotation(R_0_2, p, a, b, c, d, elementIndex);

    auto& rotation = m_rotations[elementIndex];

    rotation.transpose(R_0_2);

    type::fixed_array<type::VecNoInit<3, Real>,4> deforme;
    computeDeformedElement(deforme, R_0_2, p[a], p[b], p[c], p[d]);

    Displacement displacement;
    computeDisplacement(displacement,
        deforme, m_rotatedInitialElements[elementIndex]);

    Force F;
    Inherit::computeForce(F,
        displacement,
        (*finiteElementArrays.materialsStiffnesses)[elementIndex],
        (*finiteElementArrays.strainDisplacements)[elementIndex]);

    Inherit::copyForceFromElementToGlobal(*finiteElementArrays.force, F, element, rotation);
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::addForceAssembled(
    const FiniteElementArrays& finiteElementArrays)
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
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::addForceElementAssembled(
    const FiniteElementArrays& finiteElementArrays, const Element& element, const unsigned elementIndex)
{
    const Index a = element[0];
    const Index b = element[1];
    const Index c = element[2];
    const Index d = element[3];

    const auto& p = *finiteElementArrays.positions;

    Transformation R_0_2;
    computeRotation(R_0_2, p, a, b, c, d, elementIndex);

    auto& rotation = m_rotations[elementIndex];

    rotation.transpose(R_0_2);

    type::fixed_array<type::VecNoInit<3, Real>,4> deforme;
    computeDeformedElement(deforme, R_0_2, p[a], p[b], p[c], p[d]);

    Displacement displacement;
    computeDisplacement(displacement,
        deforme, m_rotatedInitialElements[elementIndex]);

    (*finiteElementArrays.strainDisplacements)[elementIndex][6][0] = 0;
    (*finiteElementArrays.strainDisplacements)[elementIndex][9][0] = 0;
    (*finiteElementArrays.strainDisplacements)[elementIndex][10][1] = 0;

    this->assembleStiffnessMatrix(finiteElementArrays, element, elementIndex);

    Force F;
    Inherit::computeForce(F,
        displacement,
        (*finiteElementArrays.materialsStiffnesses)[elementIndex],
        (*finiteElementArrays.strainDisplacements)[elementIndex]);

    Inherit::copyForceFromElementToGlobal(*finiteElementArrays.force, F, element, rotation);
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::computeDisplacement(Displacement& displacement, const VecDeriv& dx, const Index a, const Index b, const Index c, const Index d, const Transformation& rotation)
{
    displacement[0]  = rotation[0][0] * dx[a][0] + rotation[1][0] * dx[a][1] + rotation[2][0] * dx[a][2];
    displacement[1]  = rotation[0][1] * dx[a][0] + rotation[1][1] * dx[a][1] + rotation[2][1] * dx[a][2];
    displacement[2]  = rotation[0][2] * dx[a][0] + rotation[1][2] * dx[a][1] + rotation[2][2] * dx[a][2];

    displacement[3]  = rotation[0][0] * dx[b][0] + rotation[1][0] * dx[b][1] + rotation[2][0] * dx[b][2];
    displacement[4]  = rotation[0][1] * dx[b][0] + rotation[1][1] * dx[b][1] + rotation[2][1] * dx[b][2];
    displacement[5]  = rotation[0][2] * dx[b][0] + rotation[1][2] * dx[b][1] + rotation[2][2] * dx[b][2];

    displacement[6]  = rotation[0][0] * dx[c][0] + rotation[1][0] * dx[c][1] + rotation[2][0] * dx[c][2];
    displacement[7]  = rotation[0][1] * dx[c][0] + rotation[1][1] * dx[c][1] + rotation[2][1] * dx[c][2];
    displacement[8]  = rotation[0][2] * dx[c][0] + rotation[1][2] * dx[c][1] + rotation[2][2] * dx[c][2];

    displacement[9]  = rotation[0][0] * dx[d][0] + rotation[1][0] * dx[d][1] + rotation[2][0] * dx[d][2];
    displacement[10] = rotation[0][1] * dx[d][0] + rotation[1][1] * dx[d][1] + rotation[2][1] * dx[d][2];
    displacement[11] = rotation[0][2] * dx[d][0] + rotation[1][2] * dx[d][1] + rotation[2][2] * dx[d][2];
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::addDForceElement(
    const FiniteElementArrays& finiteElementArrays,
    const Real kFactor,
    const VecDeriv& dx,
    const unsigned elementIndex,
    const Element& element)
{
    const Index a = element[0];
    const Index b = element[1];
    const Index c = element[2];
    const Index d = element[3];

    Displacement displacement;

    const auto& rotation = m_rotations[elementIndex];

    // rotate by rotation transposed (compute R^T * D)
    computeDisplacement(displacement, dx, a, b, c, d, rotation);

    Force F;

    // compute J * K * J_T * (R^T * D)
    // Another application of R is required on the left
    Inherit::computeForce(F,
        displacement, // = R^T * D
        (*finiteElementArrays.plasticStrains)[elementIndex],
        (*finiteElementArrays.materialsStiffnesses)[elementIndex],
        (*finiteElementArrays.strainDisplacements)[elementIndex],
        this->m_plasticity);

    F *= - kFactor;

    // copy R*F, therefore the final computation is F = -kFactor * R * J * K * J^T * R^T * D
    Inherit::copyForceFromElementToGlobal(*finiteElementArrays.dForce, F, element, rotation);
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::addDForce(const FiniteElementArrays& finiteElementArrays,
    Real kFactor)
{
    const auto& dx = *finiteElementArrays.dx;

    unsigned int elementIndex {};
    for (const auto& element : *finiteElementArrays.elements)
    {
        addDForceElement(finiteElementArrays, kFactor, dx, elementIndex++, element);
    }

}

template <class DataTypes>
typename TetrahedronFEMForceFieldImpl<DataTypes>::Displacement TetrahedronFEMForceFieldImplCorotational<DataTypes>::
getDisplacement(const VecCoord& positions, const Element& element, const unsigned elementIndex) const
{
    const Index a = element[0];
    const Index b = element[1];
    const Index c = element[2];
    const Index d = element[3];

    const auto& p = positions;
    Transformation R_0_2;
    computeRotation(R_0_2, p, a, b, c, d, elementIndex);

    type::fixed_array<type::VecNoInit<3, Real>,4> deforme;
    computeDeformedElement(deforme, R_0_2, p[a], p[b], p[c], p[d]);

    Displacement displacement;
    computeDisplacement(displacement,
        deforme, m_rotatedInitialElements[elementIndex]);

    return displacement;
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>
::computeDeformedElement(type::fixed_array<type::VecNoInit<3, Real>,4>& deforme, const Transformation& rotation, const Coord& xa, const Coord& xb, const Coord& xc, const Coord& xd) const
{
    deforme[0] = rotation * xa;
    deforme[1] = rotation * xb;
    deforme[2] = rotation * xc;
    deforme[3] = rotation * xd;

    deforme[1][0] -= deforme[0][0];
    deforme[2][0] -= deforme[0][0];
    deforme[2][1] -= deforme[0][1];
    deforme[3] -= deforme[0];
}

template <class DataTypes>
void TetrahedronFEMForceFieldImplCorotational<DataTypes>::
computeDisplacement(Displacement& displacement, const type::fixed_array<type::VecNoInit<3, Real>, 4>& deformedElement, const type::fixed_array<Coord,4>& rotatedInitialElement) const
{
    displacement[ 0] = 0;
    displacement[ 1] = 0;
    displacement[ 2] = 0;
    displacement[ 3] = rotatedInitialElement[1][0] - deformedElement[1][0];
    displacement[ 4] = 0;
    displacement[ 5] = 0;
    displacement[ 6] = rotatedInitialElement[2][0] - deformedElement[2][0];
    displacement[ 7] = rotatedInitialElement[2][1] - deformedElement[2][1];
    displacement[ 8] = 0;
    displacement[ 9] = rotatedInitialElement[3][0] - deformedElement[3][0];
    displacement[10] = rotatedInitialElement[3][1] - deformedElement[3][1];
    displacement[11] = rotatedInitialElement[3][2] - deformedElement[3][2];
}

}
