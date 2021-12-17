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

#include <SofaSimpleFem/TetrahedronFEMForceFieldImplLarge.h>

namespace sofa::component::forcefield
{

template <class DataTypes>
void TetrahedronFEMForceFieldImplLarge<DataTypes>::
computeRotation(Transformation& rotation, const VecCoord& positions, const Index a, const Index b, const Index c, const Index d, unsigned elementIndex) const
{
    SOFA_UNUSED(d);
    SOFA_UNUSED(elementIndex);

    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    const Coord edgex = (positions[b]-positions[a]).normalized();
          Coord edgey = positions[c]-positions[a];
    const Coord edgez = cross( edgex, edgey ).normalized();
                edgey = cross( edgez, edgex );

    rotation[0][0] = edgex[0];
    rotation[0][1] = edgex[1];
    rotation[0][2] = edgex[2];
    rotation[1][0] = edgey[0];
    rotation[1][1] = edgey[1];
    rotation[1][2] = edgey[2];
    rotation[2][0] = edgez[0];
    rotation[2][1] = edgez[1];
    rotation[2][2] = edgez[2];

    // TODO handle degenerated cases like in the SVD method
}

}