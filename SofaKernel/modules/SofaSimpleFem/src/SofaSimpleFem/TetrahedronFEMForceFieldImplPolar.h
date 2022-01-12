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

template<class DataTypes>
class TetrahedronFEMForceFieldImplPolar : public TetrahedronFEMForceFieldImplCorotational<DataTypes>
{
protected:

    using Inherit = TetrahedronFEMForceFieldImplCorotational<DataTypes>;
    using typename Inherit::Transformation;
    using typename Inherit::Coord;
    using typename Inherit::VecCoord;
    using typename Inherit::Real;
    using typename Inherit::Displacement;
    using typename Inherit::Element;
    using typename Inherit::FiniteElementArrays;

    void computeRotation(Transformation& rotation, const VecCoord& positions, Index a, Index b, Index c, Index d, unsigned elementIndex) const override;

    void computeDisplacement(Displacement& displacement, const type::fixed_array<type::VecNoInit<3, Real>,4>& deformedElement,
            const type::fixed_array<Coord,4>& rotatedInitialElement) const override;

    type::fixed_array<Coord,4>
    computeRotatedInitialElement(const Transformation& rotation, const Element& element, const VecCoord& initialPoints) const override;

    void computeDeformedElement(type::fixed_array<type::VecNoInit<3, Real>,4>& deforme,
        const Transformation& rotation, const Coord& xa, const Coord& xb, const Coord& xc, const Coord& xd) const override;

    void addForceAssembled(const FiniteElementArrays& finiteElementArrays) override;

};



}

#include <SofaSimpleFem/TetrahedronFEMForceFieldImplPolar.inl>
