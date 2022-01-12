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

#include <SofaSimpleFem/TetrahedronFEMForceFieldImpl.h>
namespace sofa::component::forcefield
{

template<class DataTypes>
class TetrahedronFEMForceFieldImplCorotational : public TetrahedronFEMForceFieldImpl<DataTypes>
{
public:

    using Inherit = TetrahedronFEMForceFieldImpl<DataTypes>;
    using typename Inherit::Real;
    using typename Inherit::Coord;
    using typename Inherit::Deriv;
    using typename Inherit::VecCoord;
    using typename Inherit::VecDeriv;
    using typename Inherit::Element;
    using typename Inherit::VecElement;
    using typename Inherit::Displacement;
    using typename Inherit::MaterialStiffness;
    using typename Inherit::StrainDisplacement;
    using typename Inherit::VecStrainDisplacement;
    using typename Inherit::Force;
    using typename Inherit::FiniteElementArrays;
    using typename Inherit::Transformation;
    using typename Inherit::StiffnessMatrix;

    void init(const FiniteElementArrays& finiteElementArrays) override;
    void addForce(const FiniteElementArrays& finiteElementArrays) override;
    void addDForce(const FiniteElementArrays& finiteElementArrays, Real kFactor) override;

    typename Inherit::Displacement getDisplacement(const VecCoord& positions, const Element& element, unsigned elementIndex) const override;

    typename Inherit::Transformation getRotation(const unsigned int elementIndex) const override
    {
        return m_rotations[elementIndex];
    }

protected:

    type::vector<Transformation> m_initialRotations;
    type::vector<Transformation> m_rotations;
    std::vector<type::fixed_array<Coord,4> > m_rotatedInitialElements;   ///< The initials positions in its frame

    void initElement(const VecCoord& initialPoints, StrainDisplacement& strainDisplacement, const Element& element, unsigned int elementIndex);

    virtual void computeRotation(Transformation& rotation, const VecCoord& positions, Index a, Index b, Index c, Index d, unsigned elementIndex) const = 0;
    virtual void computeInitialRotation(Transformation& rotation, const VecCoord& positions, Index a, Index b, Index c, Index d, unsigned elementIndex) const;

    void addForceElementElastic(const FiniteElementArrays& finiteElementArrays, const Element& element, const unsigned int elementIndex);

    virtual void addForceAssembled(const FiniteElementArrays& finiteElementArrays);
    void addForceElementAssembled(const FiniteElementArrays& finiteElementArrays, const Element& element, const unsigned int elementIndex);

    static void computeDisplacement(Displacement& displacement, const VecDeriv& dx, Index a, Index b, Index c, Index d,
                                    const Transformation& rotation);

    void addDForceElement(const FiniteElementArrays& finiteElementArrays, Real kFactor, const VecDeriv& dx, unsigned elementIndex,
                          const Element& element);

    virtual void
    computeDeformedElement(type::fixed_array<type::VecNoInit<3, Real>,4>& deforme, const Transformation& rotation, const Coord& xa, const Coord& xb, const Coord& xc, const Coord& xd) const;

    virtual void computeDisplacement(Displacement& displacement, const type::fixed_array<type::VecNoInit<3, Real>,4>& deformedElement,
        const type::fixed_array<Coord,4>& rotatedInitialElement) const;

    virtual type::fixed_array<Coord,4>
    computeRotatedInitialElement(const Transformation& rotation, const Element& element, const VecCoord& initialPoints) const;

    typename Inherit::StiffnessMatrix computeStiffnessMatrix(const MaterialStiffness &K, const StrainDisplacement &J, unsigned elementIndex) const override;
};

}

#include <SofaSimpleFem/TetrahedronFEMForceFieldImplCorotational.inl>
