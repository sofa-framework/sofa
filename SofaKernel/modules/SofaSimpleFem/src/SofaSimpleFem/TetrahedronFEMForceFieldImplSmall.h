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
class TetrahedronFEMForceFieldImplSmall : public TetrahedronFEMForceFieldImpl<DataTypes>
{
public:

    using Inherit = TetrahedronFEMForceFieldImpl<DataTypes>;
    using Inherit::Real;
    using Inherit::Coord;
    using Inherit::Deriv;
    using Inherit::VecCoord;
    using Inherit::Element;
    using Inherit::VecElement;
    using Inherit::Displacement;
    using Inherit::MaterialStiffness;
    using Inherit::VecStrainDisplacement;
    using Inherit::Force;
    using Inherit::FiniteElementArrays;
    using Inherit::Transformation;
    using Inherit::StiffnessMatrix;

    void init(const FiniteElementArrays& finiteElementArrays) override;
    void addForce(const FiniteElementArrays& finiteElementArrays) override;
    void addDForce(const FiniteElementArrays& finiteElementArrays, Real kFactor) override;

protected:

    void addForceAssembled(const FiniteElementArrays& finiteElementArrays);

    void addForceElementElastic(const FiniteElementArrays& finiteElementArrays, const Element& element, const unsigned int elementIndex);

    void addForceElementAssembled(const FiniteElementArrays& finiteElementArrays, const Element& element, const unsigned int elementIndex);

    static Displacement computeDisplacement(const VecCoord& initialPoints, const VecCoord& positions, const Element& element);

    static Displacement computeDisplacement(
        const Coord& a_0, const Coord& b_0, const Coord& c_0, const Coord& d_0,
        const Coord& a  , const Coord& b  , const Coord& c  , const Coord& d
    );
};

}

#include <SofaSimpleFem/TetrahedronFEMForceFieldImplSmall.inl>
