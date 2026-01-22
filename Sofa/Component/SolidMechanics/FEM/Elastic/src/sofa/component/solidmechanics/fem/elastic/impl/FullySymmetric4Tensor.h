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

#include <sofa/component/solidmechanics/fem/elastic/impl/KroneckerDelta.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/SymmetricTensor.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/VoigtNotation.h>
#include <sofa/core/trait/DataTypes.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/type/MatSym.h>

#include <iomanip>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * A class to represent a fully symmetric 4th rank tensor.
 *
 * It's a tensor having both minor and major symmetries. Given the indices i,j,k,l, a fully symmetric
 * tensor C has the following properties:
 * C(i,j,k,l) = C(k,l,i,j) (major symmetry)
 * C(i,j,k,l) = C(j,i,k,l) (minor symmetry)
 * C(i,j,k,l) = C(i,j,l,k) (minor symmetry)
 */
template <class DataTypes>
class FullySymmetric4Tensor
{
private:
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size NumberOfIndependentElements = symmetric_tensor::NumberOfIndependentElements<spatial_dimensions>;
    using Real = sofa::Real_t<DataTypes>;

public:
    FullySymmetric4Tensor() = default;

    template<class Callable>
    explicit FullySymmetric4Tensor(Callable callable) : m_matrix(sofa::type::NOINIT)
    {
        fill(callable);
    }

    template<class Callable>
    void fill(Callable callable)
    {
        SCOPED_TIMER_TR("fillFullySymmetric4Tensor");

#ifndef NDEBUG
        checkSymmetry(callable);
#endif

        for (sofa::Size a = 0; a < NumberOfIndependentElements; ++a)
        {
            const auto [i, j] = toTensorIndices<DataTypes>(a);
            for (sofa::Size b = a; b < NumberOfIndependentElements; ++b) // the Voigt representation is symmetric, that is why b starts at a
            {
                const auto [k, l] = toTensorIndices<DataTypes>(b);
                m_matrix(a, b) = callable(i, j, k, l);
            }
        }
    }

    Real& operator()(sofa::Size i, sofa::Size j, sofa::Size k, sofa::Size l)
    {
        const auto a = tensorToVoigtIndex<DataTypes>(i, j);
        const auto b = tensorToVoigtIndex<DataTypes>(k, l);
        return m_matrix(a, b);
    }

    Real operator()(sofa::Size i, sofa::Size j, sofa::Size k, sofa::Size l) const
    {
        const auto a = tensorToVoigtIndex<DataTypes>(i, j);
        const auto b = tensorToVoigtIndex<DataTypes>(k, l);
        return m_matrix(a, b);
    }

    const sofa::type::MatSym<NumberOfIndependentElements, Real>& toVoigtMatSym() const
    {
        return m_matrix;
    }

private:
    sofa::type::MatSym<NumberOfIndependentElements, Real> m_matrix;

    template<class Callable>
    static void checkSymmetry(Callable callable)
    {
        for (sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            for (sofa::Size j = 0; j < spatial_dimensions; ++j)
            {
                for (sofa::Size k = 0; k < spatial_dimensions; ++k)
                {
                    for (sofa::Size l = 0; l < spatial_dimensions; ++l)
                    {
                        const auto ijkl = callable(i, j, k, l);
                        const auto jikl = callable(i, j, k, l);
                        const auto klij = callable(k, l, i, j);
                        const auto ijlk = callable(i, j, l, k);
                        constexpr auto max_precision{std::numeric_limits<long double>::digits10 + 1};
                        msg_error_when(std::abs(ijkl - klij) > 1e-6, "ElasticityTensor") << "No major symmetry (ij) <-> (kl) " << std::setprecision(max_precision) << ijkl << " != " << klij;
                        msg_error_when(std::abs(ijkl - jikl) > 1e-6, "ElasticityTensor") << "No minor symmetry i <-> j " << std::setprecision(max_precision) << ijkl << " != " << jikl;
                        msg_error_when(std::abs(ijkl - ijlk) > 1e-6, "ElasticityTensor") << "No minor symmetry k <-> l " << std::setprecision(max_precision) << ijkl << " != " << ijlk;
                    }
                }
            }
        }
    }
};


/**
 * @brief Creates an isotropic elasticity tensor for given material properties.
 *
 * This function constructs and returns an elasticity tensor for an isotropic material
 * characterized by its Young's modulus and Poisson's ratio. It computes the tensor
 * using the Lamé parameters, which are derived from the given material properties.
 *
 * @param youngModulus The Young's modulus of the material, representing its stiffness.
 * @param poissonRatio The Poisson's ratio of the material, describing its deformation behavior.
 * @return The isotropic elasticity tensor represented as an object of
 * `ElasticityTensor<DataTypes>`.
 */
template <class DataTypes>
FullySymmetric4Tensor<DataTypes> makeIsotropicElasticityTensor(sofa::Real_t<DataTypes> mu, sofa::Real_t<DataTypes> lambda)
{
    using Real = sofa::Real_t<DataTypes>;

    return FullySymmetric4Tensor<DataTypes>{
        [mu, lambda](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
        {
            return mu * (kroneckerDelta<Real>(i, k) * kroneckerDelta<Real>(j, l) + kroneckerDelta<Real>(i, l) * kroneckerDelta<Real>(j, k)) +
                        lambda * kroneckerDelta<Real>(i, j) * kroneckerDelta<Real>(k, l);
        }};
}

}
