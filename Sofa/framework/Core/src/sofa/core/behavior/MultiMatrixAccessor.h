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
#include <sofa/core/fwd.h>
#include <sofa/defaulttype/fwd.h>

namespace sofa::core::behavior
{

/// Abstract class allowing to find the part of the mechanical matrix
/// associated with a given MechanicalState or pair of MechanicalState
/// (for interaction forces and mappings).
///
class SOFA_CORE_API MultiMatrixAccessor
{
public:
    using Index = sofa::SignedIndex ;

    virtual ~MultiMatrixAccessor();

    /// Simple structure holding a reference to the submatrix related to one MechanicalState
    class MatrixRef
    {
    public:
        linearalgebra::BaseMatrix* matrix;
        unsigned int offset;
        MatrixRef() : matrix(nullptr), offset(0) {}
        constexpr MatrixRef(const MatrixRef & other) = default;
        linearalgebra::BaseMatrix* operator->() const { return matrix; }
        bool operator!() const { return matrix == nullptr; }
        operator bool() const { return matrix != nullptr; }
        void operator =(const MatrixRef& b) {offset = b.offset; matrix = b.matrix;}
    };

    /// Simple structure holding a reference to the submatrix related to the interactions between two MechanicalStates
    class InteractionMatrixRef
    {
    public:
        linearalgebra::BaseMatrix* matrix;
        unsigned int offRow, offCol;
        InteractionMatrixRef() : matrix(nullptr), offRow(0), offCol(0) {}
        constexpr InteractionMatrixRef(const InteractionMatrixRef & other) = default;
        linearalgebra::BaseMatrix* operator->() const { return matrix; }
        bool operator!() const { return matrix == nullptr; }
        operator bool() const { return matrix != nullptr; }
        void operator =(const InteractionMatrixRef& b) {offRow = b.offRow; offCol = b.offCol; matrix = b.matrix;}
    };

    virtual void addMechanicalState(const BaseMechanicalState* mstate);
    virtual void addMechanicalMapping(BaseMapping* mapping);
    virtual void addMappedMechanicalState(const BaseMechanicalState* mstate);

    virtual Index getGlobalDimension() const = 0;
    virtual int getGlobalOffset(const BaseMechanicalState* mstate) const = 0;

    virtual MatrixRef getMatrix(const BaseMechanicalState* mstate) const = 0;
    virtual InteractionMatrixRef getMatrix(const BaseMechanicalState* mstate1, const BaseMechanicalState* mstate2) const = 0;

};

} // namespace sofa::core::behavior
