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
#ifndef SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_INL
#define SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_INL

#include <sofa/core/behavior/ProjectiveConstraintSet.h>

namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
ProjectiveConstraintSet<DataTypes>::ProjectiveConstraintSet(MechanicalState<DataTypes> *mm)
    : endTime( initData(&endTime,(Real)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
    , mstate(mm)
{
}

template<class DataTypes>
ProjectiveConstraintSet<DataTypes>::~ProjectiveConstraintSet()
{
}

template <class DataTypes>
bool ProjectiveConstraintSet<DataTypes>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::init()
{
    BaseProjectiveConstraintSet::init();
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectJacobianMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, MultiMatrixDerivId cId)
{
    if (!isActive())
        return;

    if (mstate)
    {
        projectJacobianMatrix(mparams /* PARAMS FIRST */, *cId[mstate].write());
    }
}

#ifdef SOFA_SMP
template<class T>
struct projectResponseTask
{
    void operator()(const MechanicalParams* mparams /* PARAMS FIRST */, void *c, Shared_rw< objectmodel::Data< typename T::VecDeriv> > dx)
    {
        ((T *)c)->T::projectResponse(mparams /* PARAMS FIRST */, dx.access());
    }
};

template<class T>
struct projectVelocityTask
{
    void operator()(const MechanicalParams* mparams /* PARAMS FIRST */, void *c, Shared_rw< objectmodel::Data< typename T::VecDeriv> > v)
    {
        ((T *)c)->T::projectVelocity(mparams /* PARAMS FIRST */, v.access());
    }
};

template<class T>
struct projectPositionTask
{
    void operator()(const MechanicalParams* mparams /* PARAMS FIRST */, void *c, Shared_rw< objectmodel::Data< typename T::VecCoord> > x)
    {
        ((T *)c)->T::projectPosition(mparams /* PARAMS FIRST */, x.access());
    }
};

template<class DataTypes>
struct projectResponseTask<ProjectiveConstraintSet< DataTypes > >
{
    void operator()(const MechanicalParams* mparams /* PARAMS FIRST */, ProjectiveConstraintSet<DataTypes>  *c, Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > dx)
    {
        c->projectResponse(mparams /* PARAMS FIRST */, dx.access());
    }
};

template<class DataTypes>
struct projectVelocityTask<ProjectiveConstraintSet< DataTypes > >
{
    void operator()(const MechanicalParams* mparams /* PARAMS FIRST */, ProjectiveConstraintSet<DataTypes>  *c, Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > v)
    {
        c->projectVelocity(mparams /* PARAMS FIRST */, v.access());
    }
};

template<class DataTypes>
struct projectPositionTask<ProjectiveConstraintSet< DataTypes > >
{
    void operator()(const MechanicalParams* mparams /* PARAMS FIRST */, ProjectiveConstraintSet<DataTypes>  *c, Shared_rw< objectmodel::Data< typename DataTypes::VecCoord> > x)
    {
        c->projectPosition(mparams /* PARAMS FIRST */, x.access());
    }
};
#endif /* SOFA_SMP */

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectResponse(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId dxId)
{
    if (!isActive())
        return;

    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<projectResponseTask<ProjectiveConstraintSet< DataTypes > > >(mparams /* PARAMS FIRST */, this,
                    **defaulttype::getShared(*dxId[mstate].write()));
        else
#endif /* SOFA_SMP */
            projectResponse(mparams /* PARAMS FIRST */, *dxId[mstate].write());
    }
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectVelocity(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId vId)
{
    if (!isActive())
        return;

    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<projectVelocityTask<ProjectiveConstraintSet< DataTypes > > >(mparams /* PARAMS FIRST */, this,
                    **defaulttype::getShared(*vId[mstate].write()));
        else
#endif /* SOFA_SMP */
            projectVelocity(mparams /* PARAMS FIRST */, *vId[mstate].write());
    }
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectPosition(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecCoordId xId)
{
    if (!isActive())
        return;

    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<projectPositionTask<ProjectiveConstraintSet< DataTypes > > >(mparams /* PARAMS FIRST */, this,
                    **defaulttype::getShared(*xId[mstate].write()));
        else
#endif /* SOFA_SMP */
            projectPosition(mparams /* PARAMS FIRST */, *xId[mstate].write());
    }
}

#ifdef SOFA_SMP

// TODO
// template<class DataTypes>
// void ProjectiveConstraintSet<DataTypes>::projectFreeVelocity()
// {
// 	if( !isActive() ) return;
// 	if (mstate)
// 		Task<projectVelocityTask<ProjectiveConstraintSet< DataTypes > > >(this,**mstate->getVfree());
// }
//
// template<class DataTypes>
// void ProjectiveConstraintSet<DataTypes>::projectFreePosition()
// {
// 	if( !isActive() ) return;
// 	if (mstate)
// 		Task<projectPositionTask<ProjectiveConstraintSet< DataTypes > > >(this,**mstate->getXfree());
// }

#endif /* SOFA_SMP */

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
