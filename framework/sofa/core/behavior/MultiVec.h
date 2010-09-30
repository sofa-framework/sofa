/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_MULTIVEC_H
#define SOFA_CORE_BEHAVIOR_MULTIVEC_H

#include <sofa/core/MultiVecId.h>
#include <sofa/core/behavior/BaseVectorOperations.h>
namespace sofa
{

namespace core
{

namespace behavior
{

/// Helper class providing a high-level view of underlying state vectors.
///
/// It is used to convert math-like operations to call to computation methods.
template<VecType vtype>
class TMultiVec
{
public:

    typedef TMultiVecId<vtype, V_WRITE> MyMultiVecId;

protected:
    /// Solver who is using this vector
    BaseVectorOperation* vop;

    /// Identifier of this vector
    MyMultiVecId v

    /// Flag indicating if this vector was dynamically allocated
    bool dynamic;

private:
    /// Copy-constructor is forbidden
    TMultiVec(const MultiVec<vtype>& ) {}

public:
    /// Refers to a state vector with the given ID (VecId::position(), VecId::velocity(), etc).
    TMultiVec( BaseVectorOperation* vop, MyMultiVecId v) : vop(vop), v(v), dynamic(false)
    {}

    /// Allocate a new temporary vector with the given type (sofa::core::V_COORD or sofa::core::V_DERIV).
    TMultiVec( BaseVectorOperation* vop, core::VecType t) : vop(vop), v(vop->v_alloc(t)), dynamic(true)
    {}

    ~TMultiVec()
    {
        if (dynamic) vop->v_free(v);
    }

    /// Automatic conversion to the underlying VecId
    operator MyMultiVecId()
    {
        return v;
    }

    /// v = 0
    void clear()
    {
        vop->v_clear(v);
    }

    /// v = a
    void eq(VecId a)
    {
        vop->v_eq(v, a);
    }

    /// v += a*f
    void peq(VecId a, double f=1.0)
    {
        vop->v_peq(v, a, f);
    }

    /// v *= f
    void teq(double f)
    {
        vop->v_teq(v, f);
    }

    /// v = a+b*f
    void eq(VecId a, VecId b, double f=1.0)
    {
        vop->v_op(v, a, b, f);
    }

    /// \return v.a
    double dot(MyMultiVecId a)
    {
        vop->v_dot(v, a);
        return vop->finish();
    }

    /// nullify values below given threshold
    void threshold( double threshold )
    {
        vop->v_threshold(v, threshold);
    }

    /// \return sqrt(v.v)
    double norm()
    {
        vop->v_dot(v, v);
        return sqrt( parent->finish() );
    }

    /// v = a
    void operator=(MyMultiVecId a)
    {
        eq(a);
    }

    /// v = a
    void operator=(const TMultiVec<vtype>& a)
    {
        eq(a.v);
    }

    /// v += a
    void operator+=(MyMultiVecId a)
    {
        peq(a);
    }

    /// v -= a
    void operator-=(MyMultiVecId a)
    {
        peq(a,-1);
    }

    /// v *= f
    void operator*=(double f)
    {
        teq(f);
    }

    /// v /= f
    void operator/=(double f)
    {
        teq(1.0/f);
    }

    /// return the scalar product dot(v,a)
    double operator*(MyMultiVecId a)
    {
        return dot(a);
    }

    friend std::ostream& operator << (std::ostream& out, const TMultiVec<vtype>& mv )
    {
        mv.vop->print(mv.v,out);
        return out;
    }
};

typedef TMultiVec<V_COORD> MultiVecCoord;
typedef TMultiVec<V_DERIV> MultiVecDeriv;
typedef TMultiVec<V_MATDERIV> MultiVecMatrixDeriv;

} // namespace behavior

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BEHAVIOR_MULTIVEC_H

