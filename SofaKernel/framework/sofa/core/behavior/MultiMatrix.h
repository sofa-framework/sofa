/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_BEHAVIOR_MULTIMATRIX_H
#define SOFA_CORE_BEHAVIOR_MULTIMATRIX_H

#include <sofa/core/MultiVecId.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/// Helper class allowing to construct mechanical expressions
///
class SOFA_CORE_API MechanicalMatrix
{
protected:
    enum { MFACT = 0, BFACT = 1, KFACT = 2 };
    defaulttype::Vec<3,SReal> factors;
public:
    MechanicalMatrix(SReal m, SReal b, SReal k) : factors(m,b,k) {}
    explicit MechanicalMatrix(const defaulttype::Vec<3,SReal>& f) : factors(f) {}

    static const MechanicalMatrix M;
    static const MechanicalMatrix B;
    static const MechanicalMatrix K;

    SReal getMFact() const { return factors[MFACT]; }
    SReal getBFact() const { return factors[BFACT]; }
    SReal getKFact() const { return factors[KFACT]; }

    MechanicalMatrix operator + (const MechanicalMatrix& m2) const { return MechanicalMatrix(factors + m2.factors); }
    MechanicalMatrix operator - (const MechanicalMatrix& m2) const { return MechanicalMatrix(factors - m2.factors); }
    MechanicalMatrix operator - () const { return MechanicalMatrix(- factors); }
    MechanicalMatrix operator * (SReal f) const { return MechanicalMatrix(factors * f); }
    //friend MechanicalMatrix operator * (SReal f, const MechanicalMatrix& m1) { return MechanicalMatrix(m1.factors * f); }
    MechanicalMatrix operator / (SReal f) const { return MechanicalMatrix(factors / f); }
    friend std::ostream& operator << (std::ostream& out, const MechanicalMatrix& m )
    {
        out << '(';
        bool first = true;
        for (unsigned int i=0; i<m.factors.size(); ++i)
        {
            SReal f = m.factors[i];
            if (f!=0.0)
            {
                if (!first) out << ' ';
                if (f == -1.0) out << '-';
                else if (f < 0) out << f << ' ';
                else
                {
                    if (!first) out << '+';
                    if (f != 1.0) out << f << ' ';
                }
                out << ("MBK")[i];
                first = false;
            }
        }
        out << ')';
        return out;
    }
};

/// Helper class providing a high-level view of underlying linear system matrices.
///
/// It is used to convert math-like operations to call to computation methods.
template<class Parent>
class MultiMatrix
{
public:
    typedef sofa::core::VecId VecId;

protected:
    /// Solver who is using this matrix
    Parent* parent;

    /// Copy-constructor is forbidden
    MultiMatrix(const MultiMatrix<Parent>&);

public:

    MultiMatrix(Parent* parent) : parent(parent)
    {
    }

    ~MultiMatrix()
    {
    }

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

    /// m = m*M+b*B+k*K
    void operator=(const MechanicalMatrix& m)
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

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
