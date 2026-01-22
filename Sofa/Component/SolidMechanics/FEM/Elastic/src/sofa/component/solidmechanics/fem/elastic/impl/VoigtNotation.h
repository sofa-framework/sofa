#pragma once

#include <sofa/defaulttype/VecTypes.h>

#include <array>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * Converts a Voigt index to the corresponding tensor indices (i, j) for a symmetric tensor.
 *
 * The Voigt index is a 0-indexed integer that uniquely identifies an independent component of a symmetric tensor.
 *
 * @tparam DataTypes The data type (e.g., sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types)
 * @param voigtIndex The Voigt index (0 <= voigtIndex < N, where N is the number of independent components)
 * @return A pair of tensor indices (i, j) such that the symmetric tensor component (i, j) corresponds to the Voigt index
 * @pre voigtIndex must be less than symmetric_tensor::NumberOfIndependentElements<DataTypes::spatial_dimensions>
 * @note For 3D: (0,0) -> 0, (1,1) -> 1, (2,2) -> 2, (1,2) -> 3, (0,2) -> 4, (0,1) -> 5
 *       For 2D: (0,0) -> 0, (1,1) -> 1, (0,1) -> 2
 */
template<class DataTypes>
constexpr auto toTensorIndices(std::size_t voigtIndex)
{
    assert(voigtIndex < symmetric_tensor::NumberOfIndependentElements<DataTypes::spatial_dimensions>);
    if constexpr (DataTypes::spatial_dimensions == 3)
    {
        constexpr std::array voigt3d {
            std::make_pair(0, 0),
            std::make_pair(1, 1),
            std::make_pair(2, 2),
            std::make_pair(1, 2),
            std::make_pair(0, 2),
            std::make_pair(0, 1)
        };
        assert(voigtIndex < voigt3d.size());
        return voigt3d[voigtIndex];
    }
    else if constexpr (DataTypes::spatial_dimensions == 2)
    {
        constexpr std::array voigt2d {
            std::make_pair(0, 0),
            std::make_pair(1, 1),
            std::make_pair(0, 1)
        };
        assert(voigtIndex < voigt2d.size());
        return voigt2d[voigtIndex];
    }
    else
    {
        return std::make_pair(0, 0);
    }
}

/**
 * Converts tensor indices (i, j) to the corresponding Voigt index for a symmetric tensor.
 *
 * This function handles symmetric tensors by mapping (i, j) to an index in the Voigt convention.
 *
 * @tparam DataTypes The data type (e.g., sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types)
 * @param i First tensor index
 * @param j Second tensor index
 * @return Voigt index (0 <= index < N, where N is the number of independent components)
 * @pre i and j must be valid spatial dimensions (0 <= i, j < DataTypes::spatial_dimensions)
 * @note For 3D: (0,0)->0, (1,1)->1, (2,2)->2, (1,2)->3, (0,2)->4, (0,1)->5
 *       For 2D: (0,0)->0, (1,1)->1, (0,1)->2
 */
template<class DataTypes>
constexpr std::size_t tensorToVoigtIndex(std::size_t i, std::size_t j)
{
    assert(i < DataTypes::spatial_dimensions);
    assert(j < DataTypes::spatial_dimensions);
    if (i == j)
        return i;
    return symmetric_tensor::NumberOfIndependentElements<DataTypes::spatial_dimensions> - i - j;
}

static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec3Types>(0,0) == 0);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec3Types>(0,1) == 5);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec3Types>(0,2) == 4);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec3Types>(1,0) == 5);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec3Types>(1,1) == 1);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec3Types>(1,2) == 3);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec3Types>(2,0) == 4);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec3Types>(2,1) == 3);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec3Types>(2,2) == 2);

static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec2Types>(0,0) == 0);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec2Types>(0,1) == 2);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec2Types>(1,0) == 2);
static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec2Types>(1,1) == 1);

static_assert(tensorToVoigtIndex<sofa::defaulttype::Vec1Types>(0,0) == 0);


}
