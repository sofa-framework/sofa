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

#include <sofa/core/MultiVecId.h>
#include <sofa/core/behavior/MechanicalMatrix.h>

namespace sofa::core::behavior
{

/// Helper class providing a high-level view of underlying linear system matrices.
///
/// It is used to convert math-like operations to call to computation methods.
template<class Parent>
class MultiMatrix
{
public:
    typedef sofa::core::VecId VecId;

    /// Copy-constructor is forbidden
    MultiMatrix(const MultiMatrix<Parent>&) = delete;

protected:
    /// Solver who is using this matrix
    Parent* parent { nullptr };

public:

    explicit MultiMatrix(Parent* parent) : parent(parent)
    {
    }

    ~MultiMatrix() = default;

    /// m = 0
    void clear()
    {
        parent->m_resetSystem();
    }

    /// m = 0
    void reset()
    {
        parent->m_resetSystem();
    }

    void setSystemMBKMatrix(const MechanicalMatrix& m)
    {
        parent->m_setSystemMBKMatrix(m.getMFact(), m.getBFact(), m.getKFact());
    }

    void solve(MultiVecDerivId solution, MultiVecDerivId rh)
    {
        parent->m_setSystemRHVector(rh);
        parent->m_setSystemLHVector(solution);
        parent->m_solveSystem();
    }

    friend std::ostream& operator << (std::ostream& out, const MultiMatrix<Parent>& m )
    {
        m.parent->m_print(out);
        return out;
    }
};

} /// namespace sofa::core::behavior
