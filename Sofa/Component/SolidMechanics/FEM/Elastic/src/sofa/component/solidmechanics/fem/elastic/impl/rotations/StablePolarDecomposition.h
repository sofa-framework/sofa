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

#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/PolarDecomposition.h>
#include <sofa/core/trait/DataTypes.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/helper/SelectableItem.h>
#include <sofa/helper/decompose.h>

#include <algorithm>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType, Fit FitType>
struct StablePolarDecomposition
{
    using RotationMatrix = sofa::type::Mat<DataTypes::spatial_dimensions, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>>;

    /// The remainder of the dimensions of an embedded element within a higher-dimensional space.
    // TODO: edges in 3D are not handled.
    static constexpr sofa::Size Codimension =
        DataTypes::spatial_dimensions - sofa::fem::FiniteElement<ElementType, DataTypes>::TopologicalDimension;

    template<std::size_t NumberOfNodesInElement>
    sofa::Coord_t<DataTypes> computeCentroid(const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodes)
    {
        sofa::Coord_t<DataTypes> centroid;
        for (const auto& node : nodes)
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
        RotationMatrix F = H;

        // The H of an embedded element is singular, as it holds nothing along the element normal.
        // Adding n n0^T completes it, so that R turns the rest normal n0 into the current normal n.
        if constexpr (Codimension == 1)
        {
            const auto M = Q.multTranspose(Q);
            const auto cofactorM = sofa::type::cofactor(M);
            // The unit normal of the element at rest. Choose the greatest one for accuracy.
            const auto n0 = std::max_element(cofactorM.begin(), cofactorM.end(),
                [](const auto& a, const auto& b) { return a.norm2() < b.norm2(); })->normalized();
            const auto n = (sofa::type::cofactor(H) * n0).normalized();

            if constexpr (FitType == Fit::Affine)
            {
                // M is scaled by its trace so that small elements do not test as singular. Polar
                // ignores the scale. M has nothing along n0, which n0 n0^T fills in.
                F = H * (M / sofa::type::trace(M) + sofa::type::dyad(n0, n0)).inverted();
            }
            F += sofa::type::dyad(n, n0);
        }
        else if constexpr (Codimension > 1)
        {
            // Not handled: H is decomposed as it is.
        }
        else if constexpr (FitType == Fit::Affine)
        {
            const auto M = Q.multTranspose(Q);
            // M is scaled by its trace so that small elements do not test as singular. Polar ignores the scale.
            F = H * (M / sofa::type::trace(M)).inverted();
        }

        sofa::helper::Decompose<sofa::Real_t<DataTypes>>::polarDecomposition_stable(F, rotationMatrix);
    }

    static constexpr sofa::helper::Item getItem()
    {
        if constexpr (FitType == Fit::Rigid)
        {
            return {"stable_polar", "Stable polar decomposition of the rigid fit"};
        }
        else
        {
            return {"stable_polar_affine", "Stable polar decomposition of the affine fit"};
        }
    }
};

template <class DataTypes, class ElementType>
using RigidFitStablePolarDecomposition = StablePolarDecomposition<DataTypes, ElementType, Fit::Rigid>;

template <class DataTypes, class ElementType>
using AffineFitStablePolarDecomposition = StablePolarDecomposition<DataTypes, ElementType, Fit::Affine>;

}
