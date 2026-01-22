#pragma once

#include <sofa/component/solidmechanics/fem/elastic/impl/MatrixTools.h>
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
