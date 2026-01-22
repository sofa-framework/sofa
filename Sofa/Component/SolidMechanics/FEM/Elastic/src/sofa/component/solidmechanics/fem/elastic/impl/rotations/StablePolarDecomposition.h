#pragma once

#include <sofa/core/trait/DataTypes.h>
#include <sofa/helper/SelectableItem.h>
#include <sofa/helper/decompose.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes>
struct StablePolarDecomposition
{
    using RotationMatrix = sofa::type::Mat<DataTypes::spatial_dimensions, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>>;

    template<std::size_t NumberOfNodesInElement>
    sofa::Coord_t<DataTypes> computeCentroid(const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodes)
    {
        sofa::Coord_t<DataTypes> centroid;
        for (const auto node : nodes)
        {
            centroid += node;
        }
        centroid /= static_cast<sofa::Real_t<DataTypes>>(NumberOfNodesInElement);
        return centroid;
    }

    template<std::size_t NumberOfNodesInElement>
    void computeRotation(RotationMatrix& rotationMatrix, const RotationMatrix& initialRotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesPosition,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesRestPosition)
    {
        SOFA_UNUSED(initialRotationMatrix);

        const auto t = computeCentroid(nodesPosition);
        const auto t0 = computeCentroid(nodesRestPosition);

        sofa::type::Mat<NumberOfNodesInElement, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>> P(sofa::type::NOINIT);
        sofa::type::Mat<NumberOfNodesInElement, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>> Q(sofa::type::NOINIT);

        for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
        {
            P[j] = nodesPosition[j] - t;
            Q[j] = nodesRestPosition[j] - t0;
        }

        const auto H = P.multTranspose(Q);

        sofa::helper::Decompose<sofa::Real_t<DataTypes>>::polarDecomposition_stable(H, rotationMatrix);
    }

    static constexpr sofa::helper::Item getItem()
    {
        return {"stable_polar", "Stable polar decomposition"};
    }
};

}
