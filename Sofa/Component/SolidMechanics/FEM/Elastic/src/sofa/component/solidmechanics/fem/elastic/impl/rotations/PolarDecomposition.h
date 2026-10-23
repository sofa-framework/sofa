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
    // TODO: embedded elements (Codimension > 0) are not handled.
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

            if constexpr (Codimension > 0)
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
        else if constexpr (Codimension > 0 || FitType == Fit::Affine)
        {
            // TODO: the affine fit and embedded elements. Without a jacobian, the tangent treats the
            // rotation as constant.
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

            const auto H = P.multTranspose(Q);

            // S and the offsets q_k are divided by the trace of M so that small elements do not test A as
            // singular. The jacobian is unchanged.
            const auto inverseTraceM = static_cast<sofa::Real_t<DataTypes>>(1) / sofa::type::trace(Q.multTranspose(Q));
            const StretchMatrix S = rotationMatrix.multTranspose(H) * inverseTraceM;

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

            for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
            {
                jacobian[j] = inverseA * sofa::type::crossProductMatrix(Q[j] * inverseTraceM);
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
