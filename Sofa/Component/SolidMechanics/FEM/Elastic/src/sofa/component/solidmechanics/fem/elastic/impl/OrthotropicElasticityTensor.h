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
#include <sofa/component/solidmechanics/fem/elastic/impl/LameParameters.h>
#include <sofa/type/FullySymmetric4Tensor.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/KroneckerDelta.h>

namespace sofa::component::solidmechanics::fem::elastic
{


/**
 * @brief Creates an isotropic elasticity tensor for given material properties.
 *
 * This function constructs and returns an elasticity tensor for an isotropic material
 * characterized by its Young's modulus and Poisson's ratio. It computes the tensor
 * using the Lamé parameters, which are derived from the given material properties.
 *
 * @param mu Lamé's first parameter
 * @param lambda Lamé's second parameter
 * @return The isotropic elasticity tensor
 */
template <sofa::Size D, class real>
auto makeIsotropicElasticityTensor(LameMu<real> mu, LameLambda<real> lambda)
{
    return sofa::type::FullySymmetric4Tensor<D, real>{
        [mu = mu.get(), lambda = lambda.get()](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
        {
            return mu * (kroneckerDelta<real>(i, k) * kroneckerDelta<real>(j, l) + kroneckerDelta<real>(i, l) * kroneckerDelta<real>(j, k)) +
                        lambda * kroneckerDelta<real>(i, j) * kroneckerDelta<real>(k, l);
        }};
}


template <sofa::Size N, class real>
constexpr sofa::type::Vec<N, real> orthotropicElasticityTensorProduct(const sofa::type::Mat<N, N, real>& tensor, const sofa::type::Vec<N, real>& v)
{
    return tensor * v;
}

/**
 * Specialization for 3D where the operations containing known zeros in the elasticity tensor are
 * omitted.
 *
 * WARNING: this specialization is only valid for a give Voigt index mapping. If this mapping happens
 * to change, this specialization will need to be updated accordingly.
 */
template <class real>
constexpr sofa::type::Vec<6, real> orthotropicElasticityTensorProduct(const sofa::type::Mat<6, 6, real>& tensor, const sofa::type::Vec<6, real>& v)
{
    sofa::type::Vec<6, real> result { sofa::type::NOINIT };

    result[0] = tensor(0, 0) * v[0] + tensor(0, 1) * v[1] + tensor(0, 2) * v[2];
    result[1] = tensor(1, 0) * v[0] + tensor(1, 1) * v[1] + tensor(1, 2) * v[2];
    result[2] = tensor(2, 0) * v[0] + tensor(2, 1) * v[1] + tensor(2, 2) * v[2];
    result[3] = tensor(3, 3) * v[3];
    result[4] = tensor(4, 4) * v[4];
    result[5] = tensor(5, 5) * v[5];

    return result;
}

/**
 * @class OrthotropicElasticityTensor
 * @brief Represents an elasticity tensor for orthotropic materials.
 *
 * This class is a wrapper on a matrix (6x6 in 3D) with a specialization for the matrix-vector
 * product, speeding up computations compared to a regular product. It uses the known shape of an
 * orthotropic material to optimize the matrix-vector product omitting the operations containing
 * zero coefficients.
 */
template <std::size_t D, class real>
struct OrthotropicElasticityTensor
{
    using Real = real;
    static constexpr auto NbIndependentElements = sofa::type::NumberOfIndependentElements<D>;

    explicit OrthotropicElasticityTensor(const sofa::type::Mat<NbIndependentElements, NbIndependentElements, Real>& mat) : C(mat) {}
    OrthotropicElasticityTensor() = default;

    void set(const sofa::type::Mat<NbIndependentElements, NbIndependentElements, Real>& mat)
    {
        C = mat;
    }

    void set(const sofa::type::FullySymmetric4Tensor<D, Real>& tensor)
    {
        C = tensor.toVoigtMatSym().toMat();
    }

    sofa::type::Vec<NbIndependentElements, Real> operator*(const sofa::type::Vec<NbIndependentElements, Real>& v) const
    {
        return orthotropicElasticityTensorProduct(C, v);
    }

    template<sofa::Size nbColumns>
    sofa::type::Mat<NbIndependentElements, nbColumns, Real> operator*(const sofa::type::Mat<NbIndependentElements, nbColumns, Real>& v) const noexcept
    {
        return C * v;
    }

    Real operator()(sofa::Size i, sofa::Size j) const
    {
        return C(i, j);
    }

    Real& operator()(sofa::Size i, sofa::Size j)
    {
        return C(i, j);
    }

    const sofa::type::Mat<NbIndependentElements, NbIndependentElements, Real>& toMat() const
    {
        return C;
    }

private:
    sofa::type::Mat<NbIndependentElements, NbIndependentElements, Real> C;

};

}
