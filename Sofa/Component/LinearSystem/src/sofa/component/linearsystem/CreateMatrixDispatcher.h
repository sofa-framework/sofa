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

#include <sofa/component/linearsystem/matrixaccumulators/BaseAssemblingMatrixAccumulator.h>

namespace sofa::component::linearsystem
{

/**
 * Helper class used in matrix assembly, containing some factory methods that
 * must be overriden to define the type of local matrices.
 */
template <core::matrixaccumulator::Contribution c>
struct CreateMatrixDispatcher
{
    virtual ~CreateMatrixDispatcher() = default;

    /**
     * \brief Create a local matrix object for non-mapped components
     * \param withIndexChecking If true, the returned local matrix checks the
     * indices before inserting the contributions into the matrix.
     * \return A pointer to a local matrix
     */
    [[nodiscard]] typename BaseAssemblingMatrixAccumulator<c>::SPtr createLocalMatrix(const bool withIndexChecking) const
    {
        if (withIndexChecking)
        {
            return createLocalMatrixWithIndexChecking();
        }
        return createLocalMatrix();
    }

    /**
     * \brief Create a local matrix object for mapped components
     * \return A pointer to a local matrix
     */
    [[nodiscard]] virtual typename BaseAssemblingMatrixAccumulator<c>::SPtr createLocalMappedMatrix() = 0;

protected:

    [[nodiscard]] virtual typename BaseAssemblingMatrixAccumulator<c>::SPtr createLocalMatrix() const = 0;
    [[nodiscard]] virtual typename BaseAssemblingMatrixAccumulator<c>::SPtr createLocalMatrixWithIndexChecking() const = 0;

};

}
