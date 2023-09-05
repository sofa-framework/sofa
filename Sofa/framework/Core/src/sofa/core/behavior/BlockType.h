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

#include <sofa/core/config.h>
#include <sofa/core/behavior/fwd.h>

namespace sofa::core::behavior
{

enum class BlocData
{
    SCALAR,
    MAT33
};

enum class BlocPrecision
{
    FLOAT,
    DOUBLE
};

template <class T>
BlocPrecision getBlockPrecision()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return BlocPrecision::FLOAT;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return BlocPrecision::DOUBLE;
    }
    else
    {
        return getBlockPrecision<typename T::Real>();
    }
}

struct BlockType
{
    BlocData blocData;
    BlocPrecision blocPrecision;

    bool operator<(const BlockType& blocType) const
    {
        return
            100 * static_cast<int>(blocData) + static_cast<int>(blocPrecision)
            <
            100 * static_cast<int>(blocType.blocData) + static_cast<int>(blocType.blocPrecision);
    }

    bool operator==(const BlockType& blocType) const
    {
        return blocData == blocType.blocData && blocPrecision == blocType.blocPrecision;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BlockType& s )
    {
        out << s.toString();
        return out;
    }

    [[nodiscard]]
    std::string toString() const
    {
        static const std::map<BlocData, std::string> blocDataStringMap{
            {BlocData::SCALAR, "scalar"},
            {BlocData::MAT33, "mat33"},
        };
        static const std::map<BlocPrecision, std::string> blocPrecisionStringMap{
            {BlocPrecision::FLOAT, "float"},
            {BlocPrecision::DOUBLE, "double"},
        };
        const auto blockDataIt  = blocDataStringMap.find(blocData);
        const auto blockPrecisionIt  = blocPrecisionStringMap.find(blocPrecision);
        if (blockDataIt != blocDataStringMap.end() && blockPrecisionIt != blocPrecisionStringMap.end())
        {
            return blockDataIt->second + "_" + blockPrecisionIt->second;
        }
        return "Unknown bloc type: cannot convert bloc type to string";
    }

};

template<class TReal>
std::enable_if_t<std::is_floating_point_v<TReal>, BlockType>
getScalarBlocType()
{
    return {BlocData::SCALAR, getBlockPrecision<TReal>()};
}

template<class TReal>
std::enable_if_t<std::is_floating_point_v<TReal>, BlockType>
getMat33BlocType()
{
    return {BlocData::MAT33, getBlockPrecision<TReal>()};
}

} //namespace sofa::core::behavior
