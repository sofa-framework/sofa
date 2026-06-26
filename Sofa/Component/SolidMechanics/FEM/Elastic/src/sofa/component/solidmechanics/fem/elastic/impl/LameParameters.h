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

#include <utility>

#include <sofa/type/StrongType.h>
#include <sofa/config.h>

namespace sofa::component::solidmechanics::fem::elastic
{
template<class real>
using YoungModulus = sofa::type::StrongType<real, struct YoungModulusTag, sofa::type::functionality::Arithmetic>;

template<class real>
using PoissonRatio = sofa::type::StrongType<real, struct PoissonRatioTag, sofa::type::functionality::Arithmetic>;

template<class real>
using LameLambda = sofa::type::StrongType<real, struct LameLambdaTag, sofa::type::functionality::Arithmetic>;

template<class real>
using LameMu = sofa::type::StrongType<real, struct LameMuTag, sofa::type::functionality::Arithmetic>;

/**
 * @brief Converts Young's modulus and Poisson's ratio to Lamé parameters.
 *
 * This function calculates and returns the two Lamé parameters, μ (shear modulus)
 * and λ, derived from the given Young’s modulus and Poisson’s ratio of a material.
 * These parameters are fundamental in describing isotropic elastic behavior.
 *
 * @param youngModulus The Young's modulus of the material, representing its stiffness.
 * @param poissonRatio The Poisson's ratio of the material, describing its deformation behavior.
 * @return A pair containing the calculated Lamé parameters:
 *         - First: μ (shear modulus),
 *         - Second: λ.
 */
template <std::size_t spatial_dimensions, class real>
void toLameParameters(
    //input
    YoungModulus<real> youngModulus, PoissonRatio<real> poissonRatio,
    //output
    LameLambda<real>& lambda, LameMu<real>& mu)
{
    mu.get() = youngModulus.get() / (2 * (1 + poissonRatio.get()));
    lambda.get() = youngModulus.get() * poissonRatio.get() / ((1 + poissonRatio.get()) * (1 - (spatial_dimensions - 1) * poissonRatio.get()));
}

}
