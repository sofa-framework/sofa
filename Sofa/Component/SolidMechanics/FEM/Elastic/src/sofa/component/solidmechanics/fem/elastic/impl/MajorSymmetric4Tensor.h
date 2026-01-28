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
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/type/MatSym.h>

namespace elasticity
{

/**
 * A class to represent a major symmetric 4th rank tensor.
 *
 * Given the indices i,j,k,l, a major symmetric tensor C has the following properties:
 * C(i,j,k,l) = C(k,l,i,j) (major symmetry)
 */
template <class DataTypes>
class MajorSymmetric4Tensor
{
private:
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size spatial_dimension_square = spatial_dimensions * spatial_dimensions;
    using Real = sofa::Real_t<DataTypes>;

public:
    MajorSymmetric4Tensor() = delete;

    template<class Callable>
    MajorSymmetric4Tensor(Callable callable) : m_matrix(sofa::type::NOINIT)
    {
        fill(callable);
    }

    template<class Callable>
    void fill(Callable callable)
    {
        SCOPED_TIMER_TR("fillMajorSymmetric4Tensor");
        for (sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            for (sofa::Size j = 0; j < spatial_dimensions; ++j)
            {
                const auto ij = i * spatial_dimensions + j;
                for (sofa::Size k = 0; k < spatial_dimensions; ++k)
                {
                    for (sofa::Size l = 0; l < spatial_dimensions; ++l)
                    {
                        const auto kl = k * spatial_dimensions + l;
                        if (kl <= ij)
                        {
                            m_matrix(ij, kl) = callable(i, j, k, l);
                        }
                    }
                }
            }
        }
    }

    Real& operator()(sofa::Size i, sofa::Size j, sofa::Size k, sofa::Size l)
    {
        const auto a = i * spatial_dimensions + j;
        const auto b = k * spatial_dimensions + l;
        return m_matrix(a, b);
    }

    Real operator()(sofa::Size i, sofa::Size j, sofa::Size k, sofa::Size l) const
    {
        const auto a = i * spatial_dimensions + j;
        const auto b = k * spatial_dimensions + l;
        return m_matrix(a, b);
    }

private:
    sofa::type::MatSym<spatial_dimension_square, Real> m_matrix;
};

}
