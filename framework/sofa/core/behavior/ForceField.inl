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
#ifdef SOFA_SMP
#include <sofa/defaulttype/SharedTypes.h>
#endif
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
    BaseForceField::init();

    if (!mstate)
        mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
}

#ifdef SOFA_SMP
template<class DataTypes>
struct ParallelForceFieldAddForce
{
    void operator()(const MechanicalParams *mparams /* PARAMS FIRST */, ForceField< DataTypes > *ff,Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv > > _f,Shared_r< objectmodel::Data< typename DataTypes::VecCoord > > _x,Shared_r< objectmodel::Data< typename DataTypes::VecDeriv> > _v)
    {
        ff->addForce(mparams /* PARAMS FIRST */, _f.access(),_x.read(),_v.read());
    }
};

template<class DataTypes>
struct ParallelForceFieldAddDForce
{
    void operator()(const MechanicalParams *mparams /* PARAMS FIRST */, ForceField< DataTypes >*ff,Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > _df,Shared_r<objectmodel::Data< typename DataTypes::VecDeriv> > _dx)
    {
        ff->addDForce(mparams /* PARAMS FIRST */, _df.access(),_dx.read());
    }
};
#endif /* SOFA_SMP */


template<class DataTypes>
void ForceField<DataTypes>::addForce(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId fId )
{
    if (mparams)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            // Task<ParallelForceFieldAddForce< DataTypes > >(mparams /* PARAMS FIRST */, this, sofa::defaulttype::getShared(*fId[mstate].write()),
            // 	defaulttype::getShared(*mparams->readX(mstate)), defaulttype::getShared(*mparams->readV(mstate)));
            Task<ParallelForceFieldAddForce< DataTypes > >(mparams /* PARAMS FIRST */, this, **defaulttype::getShared(*fId[mstate].write()),
                    **defaulttype::getShared(*mparams->readX(mstate)), **defaulttype::getShared(*mparams->readV(mstate)));
        else
#endif /* SOFA_SMP */
            addForce(mparams /* PARAMS FIRST */, *fId[mstate].write() , *mparams->readX(mstate), *mparams->readV(mstate));
    }
}
#ifndef SOFA_DEPRECATE_OLD_API
template<class DataTypes>
void ForceField<DataTypes>::addForce(const MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv &  f, const DataVecCoord &  x , const DataVecDeriv & v )
{
    if (mstate)
    {
        mstate->forceMask.setInUse(this->useMask());
        addForce( *f.beginEdit(mparams) , x.getValue(mparams), v.getValue(mparams));
        f.endEdit(mparams);
    }
}
template<class DataTypes>
void ForceField<DataTypes>::addForce(VecDeriv& , const VecCoord& , const VecDeriv& )
{
    serr << "ERROR("<<getClassName()<<"): addForce(VecDeriv& , const VecCoord& , const VecDeriv& ) not implemented." << sendl;
}
#endif



template<class DataTypes>
void ForceField<DataTypes>::addDForce(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId dfId )
{
    if (mparams)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<ParallelForceFieldAddDForce< DataTypes > >(mparams /* PARAMS FIRST */, this, **defaulttype::getShared(*dfId[mstate].write()), **defaulttype::getShared(*mparams->readDx(mstate)));
        else
#endif /* SOFA_SMP */

            mparams->setKFactorUsed(false);

        addDForce(mparams /* PARAMS FIRST */, *dfId[mstate].write(), *mparams->readDx(mstate));

        if (!mparams->getKFactorUsed())
            serr << "WARNING " << getClassName() << " (in ForceField<DataTypes>::addDForce): please use mparams->kFactor() in addDForce" << sendl;
    }
}

#ifndef SOFA_DEPRECATE_OLD_API
template<class DataTypes>
void ForceField<DataTypes>::addDForce(const MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv & df, const DataVecDeriv & dx )
{
    if (mstate)
    {
        addDForce( *df.beginEdit(mparams) , dx.getValue(mparams), mparams->kFactor() ,mparams->bFactor());
        df.endEdit(mparams);
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx, double kFactor, double /*bFactor*/)
{
    if (kFactor == 1.0)
        addDForce(df, dx);
    else if (kFactor != 0.0)
    {
        VecDerivId vtmp( VecDerivId::V_FIRST_DYNAMIC_INDEX);
        mstate->vAvail(core::ExecParams::defaultInstance(), vtmp);
        mstate->vAlloc(core::ExecParams::defaultInstance(), vtmp);
        VecDerivId vdx(0);
        /// @TODO: Add a better way to get the current VecId of dx
        for (vdx.index=0; vdx.index<vtmp.index; ++vdx.index)
        {
            const Data<VecDeriv> *d_vdx = mstate->read(ConstVecDerivId(vdx));
            if (d_vdx)
            {
                if (&d_vdx->getValue() == &dx)
                    break;
            }
        }

        mstate->vOp(core::ExecParams::defaultInstance(), vtmp, VecId::null(), vdx, kFactor);
        //addDForce(df, *mstate->getVecDeriv(vtmp.index));
        addDForce(df, mstate->read(ConstVecDerivId(vtmp))->getValue());

        mstate->vFree(core::ExecParams::defaultInstance(), vtmp);
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addDForce(VecDeriv& , const VecDeriv& )
{
    serr << "ERROR("<<getClassName()<<"): addDForce(VecDeriv& , const VecDeriv& ) not implemented." << sendl;
}
#endif

template<class DataTypes>
double ForceField<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getPotentialEnergy(mparams /* PARAMS FIRST */, *mparams->readX(mstate));
    return 0;
}

#ifndef SOFA_DEPRECATE_OLD_API
template<class DataTypes>
double ForceField<DataTypes>::getPotentialEnergy( const MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x ) const
{
    serr << "ERROR("<<getClassName()<<"): getPotentialEnergy(const MechanicalParams* /* PARAMS FIRST */, const DataVecCoord&) not implemented." << sendl;
    return getPotentialEnergy(x.getValue(mparams));
}
template<class DataTypes>
double ForceField<DataTypes>::getPotentialEnergy(const VecCoord&) const
{
    serr << "ERROR("<<getClassName()<<"): getPotentialEnergy(const VecCoord&) not implemented." << sendl;
    return 0.0;
}
#endif


template<class DataTypes>
void ForceField<DataTypes>::addKToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addKToMatrix(r.matrix, mparams->kFactor(), r.offset);
}
template<class DataTypes>
void ForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, double /*kFact*/, unsigned int &/*offset*/)
{
    serr << "ERROR("<<getClassName()<<"): addKToMatrix not implemented." << sendl;
}



template<class DataTypes>
void ForceField<DataTypes>::addBToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addBToMatrix(r.matrix, mparams->kFactor() , r.offset);
}
template<class DataTypes>
void ForceField<DataTypes>::addBToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, double /*bFact*/, unsigned int &/*offset*/)
{
//    serr << "ERROR("<<getClassName()<<"): addBToMatrix not implemented." << sendl;
}


} // namespace behavior

} // namespace core

} // namespace sofa

#endif
