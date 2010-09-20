/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_SMP_CORE_PARALLELMULTIVECTOR_H
#define SOFA_SMP_CORE_PARALLELMULTIVECTOR_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/SharedTypes.h>
#include <sofa/core/behavior/MultiVector.h>
namespace sofa
{

namespace core
{

namespace behavior
{
using namespace sofa::core::behavior;

/// Helper class providing a high-level view of underlying state vectors.
///
/// It is used to convert math-like operations to call to computation methods.
template<class Parent>
class ParallelMultiVector: public  sofa::core::behavior::MultiVector<Parent>
{
public:
    typedef BaseMechanicalState::VecId VecId;



public:
    /// Refers to a state vector with the given ID (VecId::position(), VecId::velocity(), etc).
    ParallelMultiVector(Parent* parent, VecId v) : sofa::core::behavior::MultiVector<Parent>(parent,v)
    {}

    /// Allocate a new temporary vector with the given type (sofa::core::V_COORD or sofa::core::V_DERIV).
    ParallelMultiVector(Parent* parent, VecId::Type t, const char* name="") : sofa::core::behavior::MultiVector<Parent>(parent,t/*, name*/)
    {}

    ~ParallelMultiVector()
    {
        if (this->dynamic) this->parent->v_free(this->v);


    }


    void peq(VecId a,Shared<double> &fSh, double f=1.0)
    {
        this->parent->v_peq(this->v, a, fSh,f);
    }
    void peq(VecId a, double f=1.0)
    {
        this->parent->v_peq(this->v, a, f);
    }
    void meq(VecId a,Shared<double> &fSh)
    {
        this->parent->v_meq(this->v, a, fSh);
    }
    void dot(Shared<double> &r,VecId a)

    {

        this->parent->v_dot(r,this->v, a);

    }
    void print()
    {
        this->parent->print(this->v,std::cerr);
    }
    operator VecId()
    {
        return this->v;
    }
};



} // namespace core

} // namespace sofa
}


#endif
