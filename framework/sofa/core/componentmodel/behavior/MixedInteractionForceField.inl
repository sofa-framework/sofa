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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MIXEDINTERACTIONFORCEFIELD_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MIXEDINTERACTIONFORCEFIELD_INL

#include "MixedInteractionForceField.h"

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes1, class DataTypes2>
MixedInteractionForceField<DataTypes1, DataTypes2>::MixedInteractionForceField(MechanicalState<DataTypes1> *mm1, MechanicalState<DataTypes2> *mm2)
    : mstate1(mm1), mstate2(mm2), mask1(NULL), mask2(NULL)
{
}

template<class DataTypes1, class DataTypes2>
MixedInteractionForceField<DataTypes1, DataTypes2>::~MixedInteractionForceField()
{
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::init()
{
    InteractionForceField::init();
    this->mask1 = &mstate1->forceMask;
    this->mask2 = &mstate2->forceMask;
}
#ifdef SOFA_SMP



template<class DataTypes1, class DataTypes2>
struct ParallelMixedInteractionForceFieldAddForce
{
    void	operator()(MixedInteractionForceField<DataTypes1, DataTypes2> *ff,
            Shared_rw<typename DataTypes1::VecDeriv> _f1,Shared_rw<typename DataTypes2::VecDeriv> _f2,
            Shared_r<typename DataTypes1::VecCoord> _x1,Shared_r<typename DataTypes2::VecCoord> _x2,
            Shared_r<typename DataTypes1::VecDeriv> _v1,Shared_r<typename DataTypes2::VecDeriv> _v2)
    {
        typename DataTypes1::VecDeriv &f1= _f1.access();
        typename DataTypes2::VecDeriv &f2= _f2.access();
        const typename DataTypes1::VecCoord &x1= _x1.read();
        const typename DataTypes2::VecCoord &x2= _x2.read();
        ff->setValidGPUDForce(true);
        ff->setValidCPUDForce(false);

        if(0&&x1.size()!=f1.size())
        {
            f1.resize(x1.size());
//          f1.zero();
        }
        if(0&&x2.size()!=f2.size())
        {
            f2.resize(x2.size());

            // f2.zero();
        }
        ff->addForce(f1,f2,x1,x2,_v1.read(),_v2.read());
    }
};




template<class DataTypes1, class DataTypes2>
struct ParallelMixedInteractionForceFieldAddDForce
{
    void	operator()(MixedInteractionForceField<DataTypes1, DataTypes2> *ff,
            Shared_rw<typename DataTypes1::VecDeriv> _df1,Shared_rw<typename DataTypes2::VecDeriv> _df2,
            Shared_r<typename DataTypes1::VecDeriv> _dx1,Shared_r<typename DataTypes2::VecDeriv> _dx2
            ,double /*kFactor*/, double bFactor)
    {
        typename DataTypes1::VecDeriv &df1= _df1.access();
        typename DataTypes2::VecDeriv &df2= _df2.access();
        const typename DataTypes1::VecDeriv &dx1= _dx1.read();
        const typename DataTypes2::VecDeriv &dx2= _dx2.read();
        if(!ff->isValidGPUDForce())
        {
            ff->copyDForceToGPU();
            ff->setValidGPUDForce(true);
            //ff->setValidCPUDForce(false);

        }
        if(0&&dx1.size()!=df1.size())
        {
            df1.resize(dx1.size());
            // df1.zero();
        }
        if(0&&dx2.size()!=df2.size())
        {
            df2.resize(dx2.size());
            //df2.zero();
        }
        ff->addDForce(df1,df2,dx1,dx2,1.0,bFactor);
    }

};





template<class DataTypes1, class DataTypes2>
struct ParallelMixedInteractionForceFieldAddForceCPU
{
    void	operator()(MixedInteractionForceField<DataTypes1, DataTypes2> *ff,
            Shared_rw<typename DataTypes1::VecDeriv> _f1,Shared_rw<typename DataTypes2::VecDeriv> _f2,
            Shared_r<typename DataTypes1::VecCoord> _x1,Shared_r<typename DataTypes2::VecCoord> _x2,
            Shared_r<typename DataTypes1::VecDeriv> _v1,Shared_r<typename DataTypes2::VecDeriv> _v2)
    {
        typename DataTypes1::VecDeriv &f1= _f1.access();
        typename DataTypes2::VecDeriv &f2= _f2.access();
        const typename DataTypes1::VecCoord &x1= _x1.read();
        const typename DataTypes2::VecCoord &x2= _x2.read();
        ff->setValidGPUDForce(false);
        ff->setValidCPUDForce(true);

        if(0&&x1.size()!=f1.size())
        {
            f1.resize(x1.size());
        }
        if(0&&x2.size()!=f2.size())
        {
            f2.resize(x2.size());
        }
        ff->addForceCPU(f1,f2,x1,x2,_v1.read(),_v2.read());
    }

};




template<class DataTypes1, class DataTypes2>
struct ParallelMixedInteractionForceFieldAddDForceCPU
{
    void	operator()(MixedInteractionForceField<DataTypes1, DataTypes2> *ff,
            Shared_rw<typename DataTypes1::VecDeriv> _df1,Shared_rw<typename DataTypes2::VecDeriv> _df2,
            Shared_r<typename DataTypes1::VecDeriv> _dx1,Shared_r<typename DataTypes2::VecDeriv> _dx2
            ,double /*kFactor*/, double bFactor)
    {
        typename DataTypes1::VecDeriv &df1= _df1.access();
        typename DataTypes2::VecDeriv &df2= _df2.access();
        const typename DataTypes1::VecDeriv &dx1= _dx1.read();
        const typename DataTypes2::VecDeriv &dx2= _dx2.read();
        if(!ff->isValidCPUDForce())
        {
            ff->copyDForceToCPU();
            //ff->setValidGPUDForce(false);
            ff->setValidCPUDForce(true);
        }
        if(0&&dx1.size()!=df1.size())
        {
            df1.resize(dx1.size());
            // df1.zero();
        }
        if(0&&dx2.size()!=df2.size())
        {
            df2.resize(dx2.size());
            //df2.zero();
        }



        ff->addDForceCPU(df1,df2,dx1,dx2,1.0,bFactor);
    }

};
template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForce(double kFactor, double bFactor)
{
    if (mstate1 && mstate2)
    {
        VecDeriv1& df1 =*mstate1->getF();
        VecDeriv2& df2 = *mstate2->getF();
        Task<ParallelMixedInteractionForceFieldAddDForceCPU<DataTypes1, DataTypes2>,ParallelMixedInteractionForceFieldAddDForce< DataTypes1, DataTypes2> >(this,*df1,*df2,**mstate1->getDx(),**mstate2->getDx(),kFactor,bFactor);
    }
}
template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForceV(double kFactor, double bFactor)
{
    if (mstate1 && mstate2)
    {
        VecDeriv1& df1 =*mstate1->getF();
        VecDeriv2& df2 = *mstate2->getF();
        Task<ParallelMixedInteractionForceFieldAddDForceCPU<DataTypes1, DataTypes2>,ParallelMixedInteractionForceFieldAddDForce<DataTypes1, DataTypes2> >(this,*df1,*df2,**mstate1->getV(),**mstate2->getV(),kFactor,bFactor);
    }
}
template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addForce()
{

    if (mstate1 && mstate2)
    {
        VecDeriv1& f1 =*mstate1->getF();
        VecDeriv2& f2 = *mstate2->getF();

        Task<ParallelMixedInteractionForceFieldAddForceCPU< DataTypes1, DataTypes2> ,ParallelMixedInteractionForceFieldAddForce< DataTypes1, DataTypes2> >(this,*f1,*f2,**mstate1->getX(),**mstate2->getX()
                ,**mstate1->getV(),**mstate2->getV());
    }
}


#else
template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addForce()
{
    if (mstate1 && mstate2)
    {
        mstate1->forceMask.setInUse(this->useMask());
        mstate2->forceMask.setInUse(this->useMask());
        addForce(*mstate1->getF(), *mstate2->getF(),
                *mstate1->getX(), *mstate2->getX(),
                *mstate1->getV(), *mstate2->getV());
    }
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForce(double kFactor, double bFactor)
{
    if (mstate1 && mstate2)
        addDForce(*mstate1->getF(),  *mstate2->getF(),
                *mstate1->getDx(), *mstate2->getDx(),
                kFactor, bFactor);
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForceV(double kFactor, double bFactor)
{
    if (mstate1 && mstate2)
        addDForce(*mstate1->getF(),  *mstate2->getF(),
                *mstate1->getV(), *mstate2->getV(),
                kFactor, bFactor);
}
#endif

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForce(VecDeriv1& /*df1*/, VecDeriv2& /*df2*/, const VecDeriv1& /*dx1*/, const VecDeriv2& /*dx2*/)
{
    serr << "ERROR("<<getClassName()<<"): addDForce not implemented." << sendl;
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForce(VecDeriv1& df1, VecDeriv2& df2, const VecDeriv1& dx1, const VecDeriv2& dx2, double kFactor, double /*bFactor*/)
{
    if (kFactor == 1.0)
        addDForce(df1, df2, dx1, dx2);
    else if (kFactor != 0.0)
    {
        BaseMechanicalState::VecId vtmp1(BaseMechanicalState::VecId::V_DERIV,BaseMechanicalState::VecId::V_FIRST_DYNAMIC_INDEX);
        mstate1->vAvail(vtmp1);
        mstate1->vAlloc(vtmp1);
        BaseMechanicalState::VecId vdx1(BaseMechanicalState::VecId::V_DERIV,0);
        /// @TODO: Add a better way to get the current VecId of dx
        for (vdx1.index=0; vdx1.index<vtmp1.index; ++vdx1.index)
            if (mstate1->getVecDeriv(vdx1.index) == &dx1)
                break;
        mstate1->vOp(vtmp1,BaseMechanicalState::VecId::null(),vdx1,kFactor);

        BaseMechanicalState::VecId vtmp2(BaseMechanicalState::VecId::V_DERIV,BaseMechanicalState::VecId::V_FIRST_DYNAMIC_INDEX);
        mstate2->vAvail(vtmp2);
        mstate2->vAlloc(vtmp2);
        BaseMechanicalState::VecId vdx2(BaseMechanicalState::VecId::V_DERIV,0);
        /// @TODO: Add a better way to get the current VecId of dx
        for (vdx2.index=0; vdx2.index<vtmp2.index; ++vdx2.index)
            if (mstate2->getVecDeriv(vdx2.index) == &dx2)
                break;
        mstate2->vOp(vtmp2,BaseMechanicalState::VecId::null(),vdx2,kFactor);

        addDForce(df1, df2, *mstate1->getVecDeriv(vtmp1.index), *mstate2->getVecDeriv(vtmp2.index));

        mstate1->vFree(vtmp1);
        mstate2->vFree(vtmp2);
    }
}

template<class DataTypes1, class DataTypes2>
double MixedInteractionForceField<DataTypes1, DataTypes2>::getPotentialEnergy() const
{
    if (mstate1 && mstate2)
        return getPotentialEnergy(*mstate1->getX(), *mstate2->getX());
    else return 0;
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
