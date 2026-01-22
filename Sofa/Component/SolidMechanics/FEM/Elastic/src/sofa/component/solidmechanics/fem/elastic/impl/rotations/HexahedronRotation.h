#pragma once

#include <sofa/core/trait/DataTypes.h>
#include <sofa/helper/SelectableItem.h>
#include <sofa/type/Mat.h>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class HexahedronRotation
 * @brief Computes the orientation matrix for 8-node hexahedral elements
 *
 * This rotation method calculates the element's orientation relative to its rest configuration
 * by computing an orthonormal basis from two average edges of the hexahedron. The resulting
 * rotation matrix is then used to transform the initial rotation matrix.
 *
 * @tparam DataTypes The data type used throughout the simulation (e.g., sofa::defaulttype::Vec3Types for 3D)
 */
template <class DataTypes>
struct HexahedronRotation
{
    using RotationMatrix = sofa::type::Mat<DataTypes::spatial_dimensions, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>>;

    /**
     * Computes the rotation matrix for the hexahedral element
     *
     * @param rotationMatrix Output: Current rotation matrix (relative to initial configuration)
     * @param initialRotationMatrix The initial rotation matrix (from rest configuration)
     * @param nodesPosition Current positions of all nodes
     * @param nodesRestPosition Rest positions of all nodes (initial configuration)
     */
    template<std::size_t NumberOfNodesInElement>
    void computeRotation(RotationMatrix& rotationMatrix, const RotationMatrix& initialRotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesPosition,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesRestPosition)
    {
        SOFA_UNUSED(nodesRestPosition);

        RotationMatrix currentRotation(sofa::type::NOINIT);
        computeOrientationFromHexahedron(currentRotation, nodesPosition);

        rotationMatrix = currentRotation.multTranspose(initialRotationMatrix);
    }

    static constexpr sofa::helper::Item getItem()
    {
        return {"hexahedron", "Compute the rotation based on two average edges in the hexahedron"};
    }

private:

    /**
     * Computes the orthonormal basis for the hexahedron
     *
     * @param rotationMatrix Output: Orthonormal basis matrix (x-axis, y-axis, z-axis)
     * @param hexahedronNodes Current positions of all 8 nodes
     */
    template<std::size_t NumberOfNodesInElement>
    void computeOrientationFromHexahedron(RotationMatrix& rotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& hexahedronNodes)
    {
        using Coord = sofa::Coord_t<DataTypes>;

        // Compute average edge vectors
        Coord xAxis = (hexahedronNodes[1] - hexahedronNodes[0] +
                      hexahedronNodes[2] - hexahedronNodes[3] +
                      hexahedronNodes[5] - hexahedronNodes[4] +
                      hexahedronNodes[6] - hexahedronNodes[7]) * 0.25;

        Coord yAxis = (hexahedronNodes[3] - hexahedronNodes[0] +
                      hexahedronNodes[2] - hexahedronNodes[1] +
                      hexahedronNodes[7] - hexahedronNodes[4] +
                      hexahedronNodes[6] - hexahedronNodes[5]) * 0.25;

        // Normalize the x-axis
        xAxis.normalize();

        // Compute z-axis as cross product of x and y
        Coord zAxis = cross(xAxis, yAxis).normalized();

        // Recompute y-axis to be orthogonal to x and z
        yAxis = cross(zAxis, xAxis);

        // Set the orthonormal basis matrix
        rotationMatrix[0] = xAxis;
        rotationMatrix[1] = yAxis;
        rotationMatrix[2] = zAxis;
    }
};

}
