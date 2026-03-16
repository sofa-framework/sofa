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
#include <sofa/component/solidmechanics/fem/elastic/impl/FullySymmetric4Tensor.h>
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


template <sofa::Size N, class real>
constexpr sofa::type::Vec<N, real> isotropicElasticityTensorProduct(const sofa::type::Mat<N, N, real>& tensor, const sofa::type::Vec<N, real>& v)
{
    return tensor * v;
}

/**
 * Specialization for 3D where the operations containing known zeros in the elasticity tensor are
 * omitted.
 */
template <class real>
constexpr sofa::type::Vec<6, real> isotropicElasticityTensorProduct(const sofa::type::Mat<6, 6, real>& tensor, const sofa::type::Vec<6, real>& v)
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

template<class DataType>
struct IsotropicElasticityTensor
{
    static constexpr auto spatial_dimensions = DataType::spatial_dimensions;
    static constexpr auto NbIndependentElements = sofa::type::NumberOfIndependentElements<spatial_dimensions>;

    explicit IsotropicElasticityTensor(const sofa::type::Mat<NbIndependentElements, NbIndependentElements, sofa::Real_t<DataType>>& mat) : C(mat) {}
    IsotropicElasticityTensor() = default;

    sofa::type::Vec<NbIndependentElements, sofa::Real_t<DataType>> operator*(const sofa::type::Vec<NbIndependentElements, sofa::Real_t<DataType>>& v) const
    {
        return isotropicElasticityTensorProduct(C, v);
    }

    template<sofa::Size nbColumns>
    sofa::type::Mat<NbIndependentElements, nbColumns, sofa::Real_t<DataType>> operator*(const sofa::type::Mat<NbIndependentElements, nbColumns, sofa::Real_t<DataType>>& v) const noexcept
    {
        return C * v;
    }

    sofa::Real_t<DataType> operator()(sofa::Size i, sofa::Size j) const
    {
        return C(i, j);
    }

    sofa::Real_t<DataType>& operator()(sofa::Size i, sofa::Size j)
    {
        return C(i, j);
    }

    const sofa::type::Mat<NbIndependentElements, NbIndependentElements, sofa::Real_t<DataType>>& toMat() const
    {
        return C;
    }

private:
    sofa::type::Mat<NbIndependentElements, NbIndependentElements, sofa::Real_t<DataType>> C;

};

}
