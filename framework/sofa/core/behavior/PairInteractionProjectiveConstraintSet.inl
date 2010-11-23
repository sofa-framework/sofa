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
#ifndef SOFA_CORE_BEHAVIOR_PAIRINTERACTIONPROJECTIVECONSTRAINTSET_INL
#define SOFA_CORE_BEHAVIOR_PAIRINTERACTIONPROJECTIVECONSTRAINTSET_INL

#include <sofa/core/behavior/PairInteractionProjectiveConstraintSet.h>


namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
PairInteractionProjectiveConstraintSet<DataTypes>::PairInteractionProjectiveConstraintSet(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : endTime( initData(&endTime,(double)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
    , object1( initData(&object1, "object1", "First Object to Constraint"))
    , object2( initData(&object2, "object2", "Second Object to Constraint"))
    , mstate1(mm1), mstate2(mm2)
{
}

template<class DataTypes>
PairInteractionProjectiveConstraintSet<DataTypes>::~PairInteractionProjectiveConstraintSet()
{
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::init()
{
    BaseInteractionProjectiveConstraintSet::init();
    if (mstate1 == NULL || mstate2 == NULL)
    {
        mstate1 = mstate2 = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    }

    this->mask1 = &mstate1->forceMask;
    this->mask2 = &mstate2->forceMask;
}

template<class DataTypes>
bool PairInteractionProjectiveConstraintSet<DataTypes>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectJacobianMatrix(MultiMatrixDerivId /*cId*/, const MechanicalParams* /*mparams*/)
{
    serr << "NOT IMPLEMENTED YET" << sendl;
}

#ifdef SOFA_SMP
template<class DataTypes>
struct PairConstraintProjectResponseTask
{
    void operator()(PairInteractionProjectiveConstraintSet<DataTypes>  *c, Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > dx1,Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > dx2, const MechanicalParams* mparams)
    {
        c->projectResponse(dx1.access(), dx2.access(), mparams);
    }
};

template<class DataTypes>
struct PairConstraintProjectVelocityTask
{
    void operator()(PairInteractionProjectiveConstraintSet<DataTypes>  *c, Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > v1, Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > v2, const MechanicalParams* mparams)
    {
        c->projectVelocity(v1.access(), v2.access(), mparams);
    }
};

template<class DataTypes>
struct PairConstraintProjectPositionTask
{
    void operator()(PairInteractionProjectiveConstraintSet<DataTypes>  *c, Shared_rw< objectmodel::Data< typename DataTypes::VecCoord> > x1, Shared_rw< objectmodel::Data< typename DataTypes::VecCoord> > x2, const MechanicalParams* mparams)
    {
        c->projectPosition(x1.access(), x2.access(), mparams);
    }
};
#endif /* SOFA_SMP */

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectResponse(MultiVecDerivId dxId, const MechanicalParams* mparams)
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
    {
        this->mask1 = &mstate1->forceMask;
        this->mask2 = &mstate2->forceMask;
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<PairConstraintProjectResponseTask<DataTypes> >(this, **defaulttype::getShared(*dxId[mstate1].write()), **defaulttype::getShared(*dxId[mstate2].write()), mparams);
        else
#endif /* SOFA_SMP */
            projectResponse(*dxId[mstate1].write(), *dxId[mstate2].write(), mparams);
    }
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectVelocity(MultiVecDerivId vId, const MechanicalParams* mparams)
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
    {
        this->mask1 = &mstate1->forceMask;
        this->mask2 = &mstate2->forceMask;
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<PairConstraintProjectVelocityTask<DataTypes> >(this, **defaulttype::getShared(*vId[mstate1].write()), **defaulttype::getShared(*vId[mstate2].write()), mparams);
        else
#endif /* SOFA_SMP */
            projectVelocity(*vId[mstate1].write(), *vId[mstate2].write(), mparams);
    }
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectPosition(MultiVecCoordId xId, const MechanicalParams* mparams)
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
    {
        this->mask1 = &mstate1->forceMask;
        this->mask2 = &mstate2->forceMask;
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<PairConstraintProjectPositionTask<DataTypes> >(this, **defaulttype::getShared(*xId[mstate1].write()), **defaulttype::getShared(*xId[mstate2].write()), mparams);
        else
#endif /* SOFA_SMP */
            projectPosition(*xId[mstate1].write(), *xId[mstate2].write(), mparams);
    }
}

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
