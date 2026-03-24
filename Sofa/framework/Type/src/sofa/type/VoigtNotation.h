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

#include <sofa/type/MatSym.h>
#include <array>

namespace sofa::type
{

constexpr std::array voigt3d {
    std::make_pair(0, 0),
    std::make_pair(1, 1),
    std::make_pair(2, 2),
    std::make_pair(1, 2),
    std::make_pair(0, 2),
    std::make_pair(0, 1)
};

constexpr std::array voigt2d {
    std::make_pair(0, 0),
    std::make_pair(1, 1),
    std::make_pair(0, 1)
};

template<std::size_t V>
struct InvalidDimension;

/**
 * Converts a Voigt index to the corresponding tensor indices (i, j) for a symmetric tensor.
 *
 * The Voigt index is a 0-indexed integer that uniquely identifies an independent component of a symmetric tensor.
 *
 * @tparam spatial_dimensions The number of space dimensions (e.g. 3 for 3D, 2 for 2D)
 * @param voigtIndex The Voigt index (0 <= voigtIndex < N, where N is the number of independent components)
 * @return A pair of tensor indices (i, j) such that the symmetric tensor component (i, j) corresponds to the Voigt index
 * @pre voigtIndex must be less than symmetric_tensor::NumberOfIndependentElements<DataTypes::spatial_dimensions>
 * @note For 3D: (0,0) -> 0, (1,1) -> 1, (2,2) -> 2, (1,2) -> 3, (0,2) -> 4, (0,1) -> 5
 *       For 2D: (0,0) -> 0, (1,1) -> 1, (0,1) -> 2
 */
template<std::size_t spatial_dimensions>
constexpr auto toTensorIndices(std::size_t voigtIndex)
{
    if constexpr (spatial_dimensions == 3)
    {
        return voigt3d[voigtIndex];
    }
    else if constexpr (spatial_dimensions == 2)
    {
        return voigt2d[voigtIndex];
    }
    else if constexpr (spatial_dimensions == 1)
    {
        SOFA_UNUSED(voigtIndex);
        return std::make_pair(0, 0);
    }
    else
    {
        //InvalidDimension is incomplete. It triggers an error showing the value of spatial_dimensions for debugging.
        return InvalidDimension<spatial_dimensions>{};
    }
}

/**
 * Converts tensor indices (i, j) to the corresponding Voigt index for a symmetric tensor.
 *
 * This function handles symmetric tensors by mapping (i, j) to an index in the Voigt convention.
 *
 * @tparam spatial_dimensions The number of space dimensions (e.g. 3 for 3D, 2 for 2D)
 * @param i First tensor index
 * @param j Second tensor index
 * @return Voigt index (0 <= index < N, where N is the number of independent components)
 * @pre i and j must be valid spatial dimensions (0 <= i, j < DataTypes::spatial_dimensions)
 * @note For 3D: (0,0)->0, (1,1)->1, (2,2)->2, (1,2)->3, (0,2)->4, (0,1)->5
 *       For 2D: (0,0)->0, (1,1)->1, (0,1)->2
 */
template<std::size_t spatial_dimensions>
constexpr std::size_t tensorToVoigtIndex(std::size_t i, std::size_t j)
{
    assert(i < spatial_dimensions);
    assert(j < spatial_dimensions);
    if (i == j)
        return i;
    return sofa::type::NumberOfIndependentElements<spatial_dimensions> - i - j;
}

static_assert(tensorToVoigtIndex<3>(0,0) == 0);
static_assert(tensorToVoigtIndex<3>(0,1) == 5);
static_assert(tensorToVoigtIndex<3>(0,2) == 4);
static_assert(tensorToVoigtIndex<3>(1,0) == 5);
static_assert(tensorToVoigtIndex<3>(1,1) == 1);
static_assert(tensorToVoigtIndex<3>(1,2) == 3);
static_assert(tensorToVoigtIndex<3>(2,0) == 4);
static_assert(tensorToVoigtIndex<3>(2,1) == 3);
static_assert(tensorToVoigtIndex<3>(2,2) == 2);

static_assert(tensorToVoigtIndex<2>(0,0) == 0);
static_assert(tensorToVoigtIndex<2>(0,1) == 2);
static_assert(tensorToVoigtIndex<2>(1,0) == 2);
static_assert(tensorToVoigtIndex<2>(1,1) == 1);

static_assert(tensorToVoigtIndex<1>(0,0) == 0);


}
