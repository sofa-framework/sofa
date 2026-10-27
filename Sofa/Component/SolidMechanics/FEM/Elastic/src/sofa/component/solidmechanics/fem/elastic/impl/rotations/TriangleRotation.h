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

#include <sofa/helper/SelectableItem.h>
#include <sofa/type/Mat.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes>
struct TriangleRotation
{
    using RotationMatrix = sofa::type::Mat<DataTypes::spatial_dimensions, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>>;

    template<std::size_t NumberOfNodesInElement>
    void computeRotation(RotationMatrix& rotationMatrix, const RotationMatrix& initialRotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesPosition,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesRestPosition)
    {
        SOFA_UNUSED(nodesRestPosition);

        RotationMatrix currentRotation(sofa::type::NOINIT);
        computeRotationFrom3Points(currentRotation, {nodesPosition[0], nodesPosition[1], nodesPosition[2]});

        rotationMatrix = currentRotation.multTranspose(initialRotationMatrix);
    }

    template<std::size_t NumberOfNodesInElement, class JacobianBlock>
    bool computeRotationJacobian(std::array<JacobianBlock, NumberOfNodesInElement>& jacobian,
        const RotationMatrix& rotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesPosition,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesRestPosition)
    {
        SOFA_UNUSED(nodesRestPosition);

        RotationMatrix currentFrame(sofa::type::NOINIT);
        computeRotationFrom3Points(currentFrame, {nodesPosition[0], nodesPosition[1], nodesPosition[2]});
        const RotationMatrix restFrame = currentFrame * rotationMatrix;

        const auto a = nodesPosition[1] - nodesPosition[0];
        const auto b = nodesPosition[2] - nodesPosition[0];
        const auto lengthA = a.norm();
        const auto b1 = b * currentFrame[0];
        const auto b2 = b * currentFrame[1];

        const RotationMatrix Ga {{0, 0, -b1 / (lengthA * b2)}, {0, 0, -1 / lengthA}, {0, 1 / lengthA, 0}};
        const RotationMatrix Gb {{0, 0, 1 / b2}, {0, 0, 0}, {0, 0, 0}};

        jacobian = {};
        jacobian[0] = restFrame.multTranspose((-Ga - Gb) * restFrame);
        jacobian[1] = restFrame.multTranspose(Ga * restFrame);
        jacobian[2] = restFrame.multTranspose(Gb * restFrame);
        return true;
    }

    static constexpr sofa::helper::Item getItem()
    {
        return {"triangle", "Compute the rotation based on the Gram-Schmidt frame alignment"};
    }

private:

    void computeRotationFrom3Points(RotationMatrix& rotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, 3>& nodesPosition)
    {
        using Coord = sofa::Coord_t<DataTypes>;

        const Coord xAxis = (nodesPosition[1] - nodesPosition[0]).normalized();
        Coord yAxis = nodesPosition[2] - nodesPosition[0];
        const Coord zAxis = cross( xAxis, yAxis ).normalized();
        yAxis = cross( zAxis, xAxis ); //yAxis is a unit vector because zAxis and xAxis are orthogonal unit vectors

        rotationMatrix[0] = xAxis;
        rotationMatrix[1] = yAxis;
        rotationMatrix[2] = zAxis;
    }
};

}
