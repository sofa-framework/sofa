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

#include <sofa/core/trait/DataTypes.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/helper/SelectableItem.h>
#include <sofa/helper/decompose.h>

#include <algorithm>

namespace sofa::component::solidmechanics::fem::elastic
{

/// How the motion of the nodes is fitted before the polar decomposition.
enum class Fit { Rigid, Affine };

/// How the polar decomposition is computed: 
/// Standard calls Decompose::polarDecomposition
/// Stable calls Decompose::polarDecomposition_stable
enum class Algorithm { Standard, Stable };

template <class DataTypes, class ElementType, Fit FitType, Algorithm AlgorithmType>
struct PolarDecomposition
{
    using RotationMatrix = sofa::type::Mat<DataTypes::spatial_dimensions, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>>;
    using StretchMatrix = RotationMatrix;

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

        const auto F = [&P, &Q]() -> RotationMatrix
        {
            const auto H = P.multTranspose(Q);

            if constexpr (Codimension == 1)
            {
                // The H of an embedded element is singular. It holds nothing along the element
                // normal. It is completed by mapping the current normal to the rest normal.
                const auto M = Q.multTranspose(Q);
                const auto cofactorM = sofa::type::cofactor(M);
                // The unit normal of the element at rest. Choose the greatest one for accuracy.
                const auto n0 = std::max_element(cofactorM.begin(), cofactorM.end(),
                    [](const auto& a, const auto& b) { return a.norm2() < b.norm2(); })->normalized();
                // The current normal of the element.
                const auto n = (sofa::type::cofactor(H) * n0).normalized();

                if constexpr (FitType == Fit::Affine)
                {
                    // M is scaled by its trace so that small elements do not test as singular.
                    // Polar ignores the scale. M has nothing along n0, which n0 n0^T fills in.
                    return H * (M / sofa::type::trace(M) + sofa::type::dyad(n0, n0)).inverted() + sofa::type::dyad(n, n0);
                }
                else
                {
                    return H + sofa::type::dyad(n, n0);
                }
            }
            else if constexpr (Codimension > 1)
            {
                // Not handled: H is decomposed as it is.
                return H;
            }
            else if constexpr (FitType == Fit::Affine)
            {
                const auto M = Q.multTranspose(Q);
                return H * (M / sofa::type::trace(M)).inverted();
            }
            else
            {
                return H;
            }
        }();

        if constexpr (AlgorithmType == Algorithm::Standard)
        {
            sofa::helper::Decompose<sofa::Real_t<DataTypes>>::polarDecomposition(F, rotationMatrix);
        }
        else
        {
            sofa::helper::Decompose<sofa::Real_t<DataTypes>>::polarDecomposition_stable(F, rotationMatrix);
        }
    }

    /// Computes the jacobian of the rotation matrix
    template<std::size_t NumberOfNodesInElement, class JacobianBlock>
    bool computeRotationJacobian(std::array<JacobianBlock, NumberOfNodesInElement>& jacobian,
        const RotationMatrix& rotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesPosition,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesRestPosition)
    {
        if constexpr (DataTypes::spatial_dimensions == 1)
        {
            // In 1D, there is no rotation.
            return true;
        }
        else if constexpr (Codimension > 0)
        {
            // TODO: embedded elements.
            jacobian = {};
            return false;
        }
        else
        {
            const auto t = computeCentroid(nodesPosition);
            const auto t0 = computeCentroid(nodesRestPosition);

            sofa::type::Mat<NumberOfNodesInElement, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>> P(sofa::type::NOINIT);
            sofa::type::Mat<NumberOfNodesInElement, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>> Q(sofa::type::NOINIT);

            for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
            {
                P[j] = nodesPosition[j] - t;
                Q[j] = nodesRestPosition[j] - t0;
            }

            const auto M = Q.multTranspose(Q);
            const auto traceM = sofa::type::trace(M);

            const auto w = [&Q, &M, traceM]()
            {
                if constexpr (FitType == Fit::Affine)
                {
                    return Q * (M / traceM).inverted() / traceM;
                }
                else
                {
                    return Q / traceM;
                }
            }();

            const auto F = P.multTranspose(w);
            const StretchMatrix S = rotationMatrix.multTranspose(F);

            sofa::type::Mat<JacobianBlock::nbLines, JacobianBlock::nbLines, sofa::Real_t<DataTypes>> A;
            if constexpr (DataTypes::spatial_dimensions == 3)
            {
                A = StretchMatrix::Identity() * sofa::type::trace(S) - S;
            }
            else
            {
                A(0, 0) = sofa::type::trace(S);
            }

            decltype(A) inverseA(sofa::type::NOINIT);
            if (!inverseA.invert(A))
            {
                jacobian = {};
                return false;
            }

            for (sofa::Size k = 0; k < NumberOfNodesInElement; ++k)
            {
                jacobian[k] = inverseA * sofa::type::crossProductMatrix(w[k]);
            }
            return true;
        }
    }

    static constexpr sofa::helper::Item getItem()
    {
        if constexpr (AlgorithmType == Algorithm::Standard)
        {
            if constexpr (FitType == Fit::Rigid)
            {
                return {"polar", "Polar decomposition of the rigid fit"};
            }
            else
            {
                return {"polar_affine", "Polar decomposition of the affine fit"};
            }
        }
        else
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
    }
};

template <class DataTypes, class ElementType>
using RigidFitPolarDecomposition = PolarDecomposition<DataTypes, ElementType, Fit::Rigid, Algorithm::Standard>;

template <class DataTypes, class ElementType>
using AffineFitPolarDecomposition = PolarDecomposition<DataTypes, ElementType, Fit::Affine, Algorithm::Standard>;

template <class DataTypes, class ElementType>
using RigidFitStablePolarDecomposition = PolarDecomposition<DataTypes, ElementType, Fit::Rigid, Algorithm::Stable>;

template <class DataTypes, class ElementType>
using AffineFitStablePolarDecomposition = PolarDecomposition<DataTypes, ElementType, Fit::Affine, Algorithm::Stable>;

}
