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
    typedef TMultiVecId<vtype, V_READ> ConstMyMultiVecId;
    typedef TMultiVecId<V_ALL, V_WRITE> AllMultiVecId;
    typedef TMultiVecId<V_ALL, V_READ> ConstAllMultiVecId;

protected:
    /// Solver who is using this vector
    BaseVectorOperations* vop;

    /// Identifier of this vector
    MyMultiVecId v;

    /// Flag indicating if this vector was dynamically allocated
    bool dynamic;

private:
    /// Copy-constructor is forbidden
    TMultiVec(const TMultiVec<vtype>& ) {}

public:

    /// Refers to a state vector with the given ID (VecId::position(), VecId::velocity(), etc).
    TMultiVec( BaseVectorOperations* vop, MyMultiVecId v) : vop(vop), v(v), dynamic(false)
    {}

    /// Refers to a not yet allocated state vector
    TMultiVec() : vop(NULL), v(MyMultiVecId::null()), dynamic(false)
    {}

    /// Allocate a new temporary vector with the given type (sofa::core::V_COORD or sofa::core::V_DERIV).
    TMultiVec( BaseVectorOperations* vop, bool dynamic=true) : vop(vop), v(MyMultiVecId::null()), dynamic(dynamic)
    {
        static_assert(vtype == V_COORD || vtype == V_DERIV, "");
        vop->v_alloc( v );
    }


    ~TMultiVec()
    {
        if (dynamic) vop->v_free(v);
    }

    /// Automatic conversion to the underlying VecId
    operator MyMultiVecId() {  return v;  }
    operator ConstMyMultiVecId() {  return v;  }
    operator AllMultiVecId() {  return v;  }
    operator ConstAllMultiVecId() {  return v;  }

    const MyMultiVecId& id() const { return v; }
    MyMultiVecId& id() { return v; }

    BaseVectorOperations* ops() { return vop; }
    void setOps(BaseVectorOperations* op) { vop = op; }

    /// allocates vector for every newly appeared mechanical states (initializing them to 0 and does not modify already allocated mechanical states)
    /// \param interactionForceField set to true, also allocate external mechanical states linked by an InteractionForceField (TODO remove this option by seeing external mmstates as abstract null vectors)
    void realloc( BaseVectorOperations* _vop, bool interactionForceField=false, bool propagate=false )
    {
        vop = _vop;
        vop->v_realloc(v, interactionForceField, propagate);
    }

    /// v = 0
    void clear()
    {
        vop->v_clear(v);
    }

    /// v = a
    void eq(MyMultiVecId a)
    {
        vop->v_eq(v, a);
    }

    /// v = a*f
    void eq(MyMultiVecId a, SReal f)
    {
        vop->v_eq(v, a, f);
    }

    /// v += a*f
    void peq(AllMultiVecId a, SReal f=1.0)
    {
        vop->v_peq(v, a, f);
    }

    /// v *= f
    void teq(SReal f)
    {
        vop->v_teq(v, f);
    }

    /// v = a+b*f
    void eq(AllMultiVecId a, AllMultiVecId b, SReal f=1.0)
    {
        vop->v_op(v, a, b, f);
    }

    /// \return v.a
    SReal dot(MyMultiVecId a)
    {
        vop->v_dot(v, a);
        return vop->finish();
    }

    /// nullify values below given threshold
    void threshold( SReal threshold )
    {
        vop->v_threshold(v, threshold);
    }

    /// \return sqrt(v.v)
    SReal norm()
    {
        vop->v_dot(v, v);
        return sqrt( vop->finish() );
    }

    /** Compute the norm of a vector.
     * The type of norm is set by parameter l. Use 0 for the infinite norm.
     * Note that the 2-norm is more efficiently computed using the square root of the dot product.
     */
    SReal norm(unsigned l)
    {
        vop->v_norm(v, l);
        return (SReal)vop->finish();
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
    void operator*=(SReal f)
    {
        teq(f);
    }

    /// v /= f
    void operator/=(SReal f)
    {
        teq(1.0/f);
    }

    /// return the scalar product dot(v,a)
    SReal operator*(MyMultiVecId a)
    {
        return dot(a);
    }

    friend std::ostream& operator << (std::ostream& out, const TMultiVec<vtype>& mv )
    {
        mv.vop->print(mv.v,out);
        return out;
    }

    size_t size() const
    {
        return vop->v_size(v);
    }
};

typedef TMultiVec<V_COORD> MultiVecCoord;
typedef TMultiVec<V_DERIV> MultiVecDeriv;
typedef TMultiVec<V_MATDERIV> MultiVecMatrixDeriv;



} // namespace behavior

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BEHAVIOR_MULTIVEC_H

