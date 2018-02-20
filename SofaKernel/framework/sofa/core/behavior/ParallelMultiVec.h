/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SMP_CORE_PARALLELMULTIVEC_H
#define SOFA_SMP_CORE_PARALLELMULTIVEC_H

#include <sofa/defaulttype/SharedTypes.h>
#include <sofa/core/behavior/MultiVec.h>

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
class TParallelMultiVec: public TMultiVec<vtype>
{
public:

    typedef TMultiVecId<vtype, V_WRITE> MyMultiVecId;
    typedef TMultiVecId<vtype, V_READ> ConstMyMultiVecId;
    typedef TMultiVecId<V_ALL, V_WRITE> AllMultiVecId;
    typedef TMultiVecId<V_ALL, V_READ> ConstAllMultiVecId;

private:
    /// Copy-constructor is forbidden
    TParallelMultiVec(const TParallelMultiVec<vtype>& ) {}

public:
    /// Refers to a state vector with the given ID (core::VecId::position(), core::VecId::velocity(), etc).
    TParallelMultiVec( sofa::core::behavior::BaseVectorOperations* vop, MyMultiVecId v) : sofa::core::behavior::TMultiVec<vtype>(vop, v)
    {}

    /// Allocate a new temporary vector with the given type (sofa::core::V_COORD or sofa::core::V_DERIV).
    TParallelMultiVec( sofa::core::behavior::BaseVectorOperations* vop ) : sofa::core::behavior::TMultiVec<vtype>(vop)
    {}

    ~TParallelMultiVec()
    {
        if (this->dynamic) this->vop->v_free(this->v);
    }


    void peq(AllMultiVecId a,Shared<double> &fSh, double f=1.0)
    {
        this->vop->v_peq(this->v, a, fSh,f);
    }
    void peq(AllMultiVecId a, double f=1.0)
    {
        this->vop->v_peq(this->v, a, f);
    }
    void meq(AllMultiVecId a,Shared<double> &fSh)
    {
        this->vop->v_meq(this->v, a, fSh);
    }
    void dot(Shared<double> &r, MyMultiVecId a)
    {
        this->vop->v_dot(r,this->v, a);
    }
    void print()
    {
        this->vop->print(this->v,std::cerr);
    }

    operator MyMultiVecId()	{	return this->v ; }
    operator ConstMyMultiVecId() { return this->v ; }
    operator AllMultiVecId()	{	return this->v ; }
    operator ConstAllMultiVecId() { return this->v ; }
};

typedef TParallelMultiVec<V_COORD> ParallelMultiVecCoord;
typedef TParallelMultiVec<V_DERIV> ParallelMultiVecDeriv;
typedef TParallelMultiVec<V_MATDERIV> ParallelMultiVecMatrixDeriv;

} // namespace behavior

} // namespace core

} // namespace sofa


#endif
