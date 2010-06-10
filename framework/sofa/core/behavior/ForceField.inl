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
#ifndef SOFA_CORE_BEHAVIOR_FORCEFIELD_INL
#define SOFA_CORE_BEHAVIOR_FORCEFIELD_INL

#include <sofa/core/behavior/ForceField.h>
#include <iostream>
namespace sofa
{

namespace core
{

namespace behavior
{


template<class DataTypes>
ForceField<DataTypes>::ForceField(MechanicalState<DataTypes> *mm)
    : mstate(mm)
{
}

template<class DataTypes>
ForceField<DataTypes>::~ForceField()
{
}

template<class DataTypes>
void ForceField<DataTypes>::init()
{
    //serr<<"ForceField<DataTypes>::init() "<<getName()<<" start"<<sendl;
    BaseForceField::init();
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    //serr<<"ForceField<DataTypes>::init() "<<getName()<<" done"<<sendl;
}

#ifndef SOFA_SMP
template<class DataTypes>
void ForceField<DataTypes>::addForce()
{
    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
        addForce(*mstate->getF(), *mstate->getX(), *mstate->getV());
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addDForce(double kFactor, double bFactor)
{
    if (mstate)
        addDForce(*mstate->getF(), *mstate->getDx(), kFactor, bFactor);
}

template<class DataTypes>
void ForceField<DataTypes>::addDForceV(double kFactor, double bFactor)
{
    if (mstate)
        addDForce(*mstate->getF(), *mstate->getV(), kFactor, bFactor);
}
#endif
template<class DataTypes>
void ForceField<DataTypes>::addDForce(VecDeriv& /*df*/, const VecDeriv& /*dx*/)
{
    serr << "ERROR("<<getClassName()<<"): addDForce not implemented." << sendl;
}

template<class DataTypes>
void ForceField<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx, double kFactor, double /*bFactor*/)
{
    if (kFactor == 1.0)
        addDForce(df, dx);
    else if (kFactor != 0.0)
    {
        BaseMechanicalState::VecId vtmp(BaseMechanicalState::VecId::V_DERIV,BaseMechanicalState::VecId::V_FIRST_DYNAMIC_INDEX);
        mstate->vAvail(vtmp);
        mstate->vAlloc(vtmp);
        BaseMechanicalState::VecId vdx(BaseMechanicalState::VecId::V_DERIV,0);
        /// @TODO: Add a better way to get the current VecId of dx
        for (vdx.index=0; vdx.index<vtmp.index; ++vdx.index)
            if (mstate->getVecDeriv(vdx.index) == &dx)
                break;
        mstate->vOp(vtmp,BaseMechanicalState::VecId::null(),vdx,kFactor);
        addDForce(df, *mstate->getVecDeriv(vtmp.index));
        mstate->vFree(vtmp);
    }
}

template<class DataTypes>
double ForceField<DataTypes>::getPotentialEnergy() const
{
    if (mstate)
        return getPotentialEnergy(*mstate->getX());
    else return 0;
}

template<class DataTypes>
void ForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, double /*kFact*/, unsigned int &/*offset*/)
{
    serr << "addKToMatrix not implemented by " << this->getClassName() << sendl;
}

template<class DataTypes>
void ForceField<DataTypes>::addBToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, double /*bFact*/, unsigned int &/*offset*/)
{

}

#ifdef SOFA_SMP
template<class DataTypes>
struct ParallelForceFieldAddForceCPU
{
    void	operator()(ForceField< DataTypes > *ff,Shared_rw< typename DataTypes::VecDeriv> _f,Shared_r< typename DataTypes::VecCoord> _x,Shared_r< typename DataTypes::VecDeriv> _v)
    {
        ff->addForce(_f.access(),_x.read(),_v.read());
    }
};

template<class DataTypes>
struct ParallelForceFieldAddDForceCPU
{
    void	operator()(ForceField< DataTypes > *ff,Shared_rw<typename DataTypes::VecDeriv> _df,Shared_r<typename  DataTypes::VecDeriv> _dx,double kFactor, double bFactor)
    {
        ff->addDForce(_df.access(),_dx.read(),kFactor,bFactor);
    }
};

template<class DataTypes>
struct ParallelForceFieldAddForce
{
    void    operator()(ForceField< DataTypes > *ff,Shared_rw< typename DataTypes::VecDeriv> _f,Shared_r< typename DataTypes::VecCoord> _x,Shared_r< typename DataTypes::VecDeriv> _v)
    {
        ff->addForce(_f.access(),_x.read(),_v.read());
    }
};

template<class DataTypes>
struct ParallelForceFieldAddDForce
{
    void    operator()(ForceField< DataTypes >*ff,Shared_rw<typename DataTypes::VecDeriv> _df,Shared_r<typename  DataTypes::VecDeriv> _dx,double kFactor, double bFactor)
    {
        ff->addDForce(_df.access(),_dx.read(),kFactor,bFactor);
    }
};

template<class DataTypes>
void ForceField< DataTypes >::addForce()
{
    Task<ParallelForceFieldAddForceCPU< DataTypes  > ,ParallelForceFieldAddForce< DataTypes  > >(this,**mstate->getF(), **mstate->getX(), **mstate->getV());
}

template<class DataTypes>
void ForceField< DataTypes >::addDForce(double kFactor, double bFactor)
{
    Task<ParallelForceFieldAddDForceCPU< DataTypes  >,ParallelForceFieldAddDForce< DataTypes  > >(this,**mstate->getF(), **mstate->getDx(),kFactor,bFactor);
}

template<class DataTypes>
void ForceField< DataTypes >::addDForceV(double kFactor, double bFactor)
{
    Task<ParallelForceFieldAddDForceCPU< DataTypes  > ,ParallelForceFieldAddDForce< DataTypes  > >(this,**mstate->getF(), **mstate->getV(),kFactor,bFactor);
}
#endif
} // namespace behavior

} // namespace core

} // namespace sofa

#endif
