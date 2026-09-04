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

#include <sofa/type/Mat.h>

#include <sofa/type/IdentityMatrix.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

/**
 * @struct DeformationGradientTag
 * @brief Tag for deformation gradient initialization
 */
inline struct DeformationGradientTag {} deformationGradient;

/**
 * @struct RightCauchyGreenTensorTag
 * @brief Tag for right Cauchy-Green tensor initialization
 */
inline struct RightCauchyGreenTensorTag {} rightCauchyGreenTensor;

/**
 * @class Strain
 * @brief Class to compute and manage strain measures in elasticity simulations
 *
 * This class handles the computation of strain measures (deformation gradient, right Cauchy-Green tensor, Green-Lagrange tensor, and strain invariants)
 * using an optional pattern for lazy computation. It supports initialization from either the deformation gradient (F) or the right Cauchy-Green tensor (C).
 *
 * @tparam DataTypes The data type of the simulation. Must provide:
 *   - `spatial_dimensions`: Spatial dimension of the problem (2 or 3)
 *   - `Real_t`: Scalar type for numerical operations
 *
 * @note The class follows the following invariant rules:
 *   - When initialized with F: Right Cauchy-Green tensor (C) is computed as C = F^T * F
 *   - When initialized with C: Deformation gradient (F) is not computed
 *   - All computed values are stored in `std::optional` to avoid unnecessary computations
 *   - Invariants are defined as:
 *      I1 = tr(C)
 *      I2 = (I1^2 - tr(C^2)) / 2
 *      I3 = det(C) = (det(F))^2
 */
template <class DataTypes>
struct Strain
{
    /**
     * @brief Spatial dimension of the problem (2 or 3)
     */
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    using Real = sofa::Real_t<DataTypes>;

    /**
     * @brief Type for deformation gradient tensor (spatial x spatial)
     */
    using DeformationGradient = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

    /**
     * @brief Type for right Cauchy-Green tensor (spatial x spatial)
     */
    using RightCauchyGreenTensor = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

    /**
     * @brief Type for Green-Lagrange tensor (spatial x spatial)
     */
    using GreenLagrangeTensor = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

    static constexpr sofa::type::IdentityMatrix identity {};

    /**
     * @brief Constructor initializing from deformation gradient
     *
     * @param tag Tag for deformation gradient initialization
     * @param F Deformation gradient tensor (F)
     */
    constexpr Strain(DeformationGradientTag, const DeformationGradient& F) : m_deformationGradient(F) {}

    /**
     * @brief Constructor initializing from right Cauchy-Green tensor
     *
     * @param tag Tag for right Cauchy-Green tensor initialization
     * @param C Right Cauchy-Green tensor (C)
     */
    constexpr Strain(RightCauchyGreenTensorTag, const RightCauchyGreenTensor& C) : m_rightCauchyGreenTensor(C) {}

    const DeformationGradient& deformationGradient() const
    {
        if (!m_deformationGradient.has_value())
        {
            throw std::logic_error("Cannot access deformation gradient. It has not been provided");
        }
        return *m_deformationGradient;
    }

    void setDeformationGradient(const DeformationGradient& F)
    {
        reset();
        m_deformationGradient = F;
    }

    Real getDeterminantDeformationGradient()
    {
        if (!m_determinantF)
        {
            computeDeterminant();
        }
        return *m_determinantF;
    }

    const RightCauchyGreenTensor& getRightCauchyGreenTensor()
    {
        if (!m_rightCauchyGreenTensor)
        {
            computeRightCauchyGreenTensor();
        }
        return *m_rightCauchyGreenTensor;
    }

    void setRightCauchyGreenTensor(const RightCauchyGreenTensor& C)
    {
        reset();
        m_rightCauchyGreenTensor = C;
    }

    const GreenLagrangeTensor& getGreenLagrangeTensor()
    {
        if (!m_greenLagrangeTensor)
        {
            computeGreenLagrangeTensor();
        }
        return *m_greenLagrangeTensor;
    }

    Real getInvariant1()
    {
        if (!m_invariant1)
        {
            computeInvariant1();
        }
        return *m_invariant1;
    }

    Real getInvariant2()
    {
        if (!m_invariant2)
        {
            computeInvariant2();
        }
        return *m_invariant2;
    }

    Real getInvariant3()
    {
        if (!m_invariant3)
        {
            computeInvariant3();
        }
        return *m_invariant3;
    }

    /**
     * @brief Reset all computed values
     *
     * Clears all optional values (deformation gradient, C, invariants, tensors)
     */
    void reset()
    {
        m_deformationGradient.reset();
        m_rightCauchyGreenTensor.reset();
        m_determinantF.reset();
        m_greenLagrangeTensor.reset();
        m_invariant1.reset();
        m_invariant2.reset();
        m_invariant3.reset();
    }

protected:

    std::optional<DeformationGradient> m_deformationGradient;
    std::optional<RightCauchyGreenTensor> m_rightCauchyGreenTensor;

    std::optional<Real> m_determinantF;
    std::optional<GreenLagrangeTensor> m_greenLagrangeTensor;
    std::optional<Real> m_invariant1;
    std::optional<Real> m_invariant2;
    std::optional<Real> m_invariant3;

    void computeDeterminant()
    {
        if (m_deformationGradient.has_value())
        {
            m_determinantF = sofa::type::determinant(*m_deformationGradient);
        }
        else if (m_rightCauchyGreenTensor.has_value())
        {
            m_determinantF = sqrt(sofa::type::determinant(*m_rightCauchyGreenTensor));
        }
        else
        {
            throw std::logic_error("Cannot compute determinant. Neither deformation gradient nor right Cauchy-Green tensor is provided");
        }
    }

    void computeRightCauchyGreenTensor()
    {
        if (m_deformationGradient.has_value())
        {
            const auto& F = *m_deformationGradient;
            auto& C = m_rightCauchyGreenTensor.emplace();
            for (sofa::Size i = 0; i < spatial_dimensions; ++i)
            {
                for (sofa::Size j = 0; j < spatial_dimensions; ++j)
                {
                    for (sofa::Size k = 0; k < spatial_dimensions; ++k)
                    {
                        C(i, j) += F(k, i) * F(k, j);
                    }
                }
            }
        }
        else
        {
            throw std::logic_error("Cannot compute right Cauchy-Green tensor. Deformation gradient is not provided");
        }
    }

    void computeGreenLagrangeTensor()
    {
        const auto& C = getRightCauchyGreenTensor();
        m_greenLagrangeTensor = static_cast<Real>(0.5) * (C - identity);
    }

    void computeInvariant1()
    {
        const auto& C = getRightCauchyGreenTensor();
        m_invariant1 = sofa::type::trace(C);
    }

    void computeInvariant2()
    {
        const auto I1 = getInvariant1();
        const auto& C = getRightCauchyGreenTensor();
        const auto trC2 = sofa::type::trace(C * C);
        m_invariant2 = static_cast<Real>(0.5) * (I1 * I1 - trC2);
    }

    void computeInvariant3()
    {
        const auto J = getDeterminantDeformationGradient();
        m_invariant3 = J * J;
    }
};

}
