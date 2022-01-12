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
class TetrahedronFEMForceFieldImplLarge : public TetrahedronFEMForceFieldImplCorotational<DataTypes>
{
public:
    using Inherit = TetrahedronFEMForceFieldImplCorotational<DataTypes>;
    using typename Inherit::Transformation;
    using typename Inherit::Coord;
    using typename Inherit::VecCoord;

protected:

    void computeRotation(Transformation& rotation, const VecCoord& positions, Index a, Index b, Index c, Index d, unsigned elementIndex) const override;
};

}

#include <SofaSimpleFem/TetrahedronFEMForceFieldImplLarge.inl>
