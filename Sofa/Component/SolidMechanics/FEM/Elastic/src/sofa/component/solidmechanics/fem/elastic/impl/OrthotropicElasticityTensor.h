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

}
