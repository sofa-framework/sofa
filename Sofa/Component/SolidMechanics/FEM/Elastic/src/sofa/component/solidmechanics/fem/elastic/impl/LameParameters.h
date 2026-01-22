#pragma once

#include <utility>

#include <sofa/core/trait/DataTypes.h>
#include <sofa/config.h>

namespace sofa::component::solidmechanics::fem::elastic
{
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
template<class DataTypes>
std::pair<sofa::Real_t<DataTypes>, sofa::Real_t<DataTypes>>
toLameParameters(sofa::Real_t<DataTypes> youngModulus, sofa::Real_t<DataTypes> poissonRatio)
{
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    const auto mu = youngModulus / (2 * (1 + poissonRatio));
    const auto lambda = youngModulus * poissonRatio / ((1 + poissonRatio) * (1 - (spatial_dimensions - 1) * poissonRatio));
    return std::make_pair(mu, lambda);
}
}
