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

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa::core::behavior
{

/**
 *  \brief Abstract base class for ordering methods in sparse linear solvers
 *
 */
class SOFA_CORE_API BaseOrderingMethod : virtual public objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseOrderingMethod, objectmodel::BaseObject)

    ~BaseOrderingMethod() override;

    struct SparseMatrixPattern
    {
        int matrixSize;
        int numberOfNonZeros;
        int* rowBegin;
        int* colsIndex;
    };

    virtual void computePermutation(
        const SparseMatrixPattern& inPattern,
        int* outPermutation,
        int* outInversePermutation) = 0;

    static void computeInverseFromPermutation(int matrixSize, const int* inPermutation, int* outInversePermutation);
};

}
