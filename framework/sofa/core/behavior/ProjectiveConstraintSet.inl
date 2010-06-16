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
void ProjectiveConstraintSet<DataTypes>::projectJacobianMatrix()
{
    if( !isActive() ) return;
    if (mstate)
    {
        VecConst *C=mstate->getC();
        typedef typename VecConst::iterator VecConstIterator;
        for (VecConstIterator it=C->begin(); it!=C->end(); ++it) projectResponse(*it);
    }
}

#ifndef SOFA_SMP
template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectResponse()
{
    if( !isActive() ) return;
    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
        projectResponse(*mstate->getDx());
    }
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectVelocity()
{
    if( !isActive() ) return;
    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
        projectVelocity(*mstate->getV());
    }
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectPosition()
{
    if( !isActive() ) return;
    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
        projectPosition(*mstate->getX());
    }
}
template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectFreeVelocity()
{
    if( !isActive() ) return;
    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
        projectVelocity(*mstate->getVfree());
    }
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectFreePosition()
{
    if( !isActive() ) return;
    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
        projectPosition(*mstate->getXfree());
    }
}

#endif

#ifdef SOFA_SMP

template<class T>
struct projectResponseTask
{
    void operator()(  void *c, Shared_rw< typename T::VecDeriv> dx)
    {
        ((T *)c)->T::projectResponse(dx.access());
    }
};

template<class T>
struct projectVelocityTask
{
    void operator()(  void *c, Shared_rw< typename T::VecDeriv> v)
    {
        ((T *)c)->T::projectVelocity(v.access());
    }
};

template<class T>
struct projectPositionTask
{
    void operator()(  void *c, Shared_rw< typename T::VecCoord> x)
    {
        ((T *)c)->T::projectPosition(x.access());
    }
};




template<class DataTypes>
struct projectResponseTask<ProjectiveConstraintSet< DataTypes > >
{
    void operator()(   ProjectiveConstraintSet<DataTypes>  *c, Shared_rw< typename DataTypes::VecDeriv> dx)
    {
        c->projectResponse(dx.access());


    }
};

template<class DataTypes>
struct projectVelocityTask<ProjectiveConstraintSet< DataTypes > >
{
    void operator()(   ProjectiveConstraintSet<DataTypes>  *c, Shared_rw< typename DataTypes::VecDeriv> v)
    {
        c->projectVelocity(v.access());


    }
};

template<class DataTypes>
struct projectPositionTask<ProjectiveConstraintSet< DataTypes > >
{
    void operator()(   ProjectiveConstraintSet<DataTypes>  *c, Shared_rw< typename DataTypes::VecCoord> x)
    {
        c->projectPosition(x.access());


    }
};

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectResponse()
{
    if( !isActive() ) return;
    if (mstate)
        Task<projectResponseTask<ProjectiveConstraintSet< DataTypes > > >(this,**mstate->getDx());

}
template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectVelocity()
{
    if( !isActive() ) return;
    if (mstate)
        Task<projectVelocityTask<ProjectiveConstraintSet< DataTypes > > >(this,**mstate->getV());
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectPosition()
{
    if( !isActive() ) return;
    if (mstate)
        Task<projectPositionTask<ProjectiveConstraintSet< DataTypes > > >(this,**mstate->getX());
}
template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectFreeVelocity()
{
    if( !isActive() ) return;
    if (mstate)
        Task<projectVelocityTask<ProjectiveConstraintSet< DataTypes > > >(this,**mstate->getVfree());
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectFreePosition()
{
    if( !isActive() ) return;
    if (mstate)
        Task<projectPositionTask<ProjectiveConstraintSet< DataTypes > > >(this,**mstate->getXfree());
}
#endif
} // namespace behavior

} // namespace core

} // namespace sofa

#endif
