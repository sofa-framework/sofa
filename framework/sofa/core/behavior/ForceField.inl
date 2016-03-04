/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
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
    : BaseForceField()
    , mstate(initLink("mstate", "MechanicalState used by this ForceField"), mm)
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

    if (!mstate.get())
        mstate.set(dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState()));
}

#ifdef SOFA_SMP
template<class DataTypes>
struct ParallelForceFieldAddForce
{
    void operator()(const MechanicalParams *mparams, ForceField< DataTypes > *ff,Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv > > _f,Shared_r< objectmodel::Data< typename DataTypes::VecCoord > > _x,Shared_r< objectmodel::Data< typename DataTypes::VecDeriv> > _v)
    {
        ff->addForce(mparams, _f.access(),_x.read(),_v.read());
    }
};

template<class DataTypes>
struct ParallelForceFieldAddDForce
{
    void operator()(const MechanicalParams *mparams, ForceField< DataTypes >*ff,Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > _df,Shared_r<objectmodel::Data< typename DataTypes::VecDeriv> > _dx)
    {
        ff->addDForce(mparams, _df.access(),_dx.read());
    }
};
#endif /* SOFA_SMP */


template<class DataTypes>
void ForceField<DataTypes>::addForce(const MechanicalParams* mparams, MultiVecDerivId fId )
{
    if (mparams)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            // Task<ParallelForceFieldAddForce< DataTypes > >(mparams, this, sofa::defaulttype::getShared(*fId[mstate.get(mparams)].write()),
            // 	defaulttype::getShared(*mparams->readX(mstate)), defaulttype::getShared(*mparams->readV(mstate)));
            Task<ParallelForceFieldAddForce< DataTypes > >(mparams, this, **defaulttype::getShared(*fId[mstate.get(mparams)].write()),
                    **defaulttype::getShared(*mparams->readX(mstate)), **defaulttype::getShared(*mparams->readV(mstate)));
        else
#endif /* SOFA_SMP */
            addForce(mparams, *fId[mstate.get(mparams)].write() , *mparams->readX(mstate), *mparams->readV(mstate));
            updateForceMask();
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId )
{
    if (mparams)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<ParallelForceFieldAddDForce< DataTypes > >(mparams, this, **defaulttype::getShared(*dfId[mstate.get(mparams)].write()), **defaulttype::getShared(*mparams->readDx(mstate)));
        else
#endif /* SOFA_SMP */

#ifndef NDEBUG
            mparams->setKFactorUsed(false);
#endif

        addDForce(mparams, *dfId[mstate.get(mparams)].write(), *mparams->readDx(mstate.get(mparams)));

#ifndef NDEBUG
        if (!mparams->getKFactorUsed())
            serr << "WARNING " << getClassName() << " (in ForceField<DataTypes>::addDForce): please use mparams->kFactor() in addDForce" << sendl;
#endif
    }
}

template<class DataTypes>
SReal ForceField<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getPotentialEnergy(mparams, *mparams->readX(mstate));
    return 0;
}

template<class DataTypes>
void ForceField<DataTypes>::addKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addKToMatrix(r.matrix, mparams->kFactorIncludingRayleighDamping(rayleighStiffness.getValue()), r.offset);
    else serr<<"ERROR("<<getClassName()<<"): addKToMatrix found no valid matrix accessor." << sendl;
}

template<class DataTypes>
void ForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, SReal /*kFact*/, unsigned int &/*offset*/)
{
    static int i=0;
    if (i < 10) {
        serr << "ERROR("<<getClassName()<<"): addKToMatrix not implemented." << sendl;
        i++;
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addSubKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> & subMatrixIndex )
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r) addSubKToMatrix(r.matrix, subMatrixIndex, mparams->kFactorIncludingRayleighDamping(rayleighStiffness.getValue()), r.offset);
    else serr<<"ERROR("<<getClassName()<<"): addKToMatrix found no valid matrix accessor." << sendl;
}

template<class DataTypes>
void ForceField<DataTypes>::addSubKToMatrix(sofa::defaulttype::BaseMatrix * mat, const helper::vector<unsigned> & /*subMatrixIndex*/, SReal kFact, unsigned int & offset)
{
    addKToMatrix(mat,kFact,offset);
}




template<class DataTypes>
void ForceField<DataTypes>::addBToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addBToMatrix(r.matrix, mparams->bFactor() , r.offset);
}
template<class DataTypes>
void ForceField<DataTypes>::addBToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, SReal /*bFact*/, unsigned int &/*offset*/)
{
//    static int i=0;
//    if (i < 10) {
//        serr << "ERROR("<<getClassName()<<"): addBToMatrix not implemented." << sendl;
//        i++;
//    }
}

template<class DataTypes>
void ForceField<DataTypes>::addSubBToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> & subMatrixIndex)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r) addSubBToMatrix(r.matrix, subMatrixIndex, mparams->bFactor() , r.offset);
}

template<class DataTypes>
void ForceField<DataTypes>::addSubBToMatrix(sofa::defaulttype::BaseMatrix * mat, const helper::vector<unsigned> & /*subMatrixIndex*/, SReal bFact, unsigned int & offset)
{
    addBToMatrix(mat,bFact,offset);
}


template<class DataTypes>
void ForceField<DataTypes>::updateForceMask()
{
    // the default implementation adds every dofs to the mask
    // this sould be overloaded by each forcefield to only add the implicated dofs subset to the mask
    mstate->forceMask.assign( mstate->getSize(), true );
}


} // namespace behavior

} // namespace core

} // namespace sofa

#endif
