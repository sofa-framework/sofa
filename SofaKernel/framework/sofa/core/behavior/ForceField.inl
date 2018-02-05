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




template<class DataTypes>
void ForceField<DataTypes>::addForce(const MechanicalParams* mparams, MultiVecDerivId fId )
{
    if (mparams)
    {
            addForce(mparams, *fId[mstate.get(mparams)].write() , *mparams->readX(mstate), *mparams->readV(mstate));
            updateForceMask();
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId )
{
    if (mparams)
    {

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
void ForceField<DataTypes>::addClambda(const MechanicalParams* mparams, MultiVecDerivId resId, MultiVecDerivId lambdaId, SReal cFactor )
{
    if (mparams)
    {
        addClambda(mparams, *resId[mstate.get(mparams)].write(), *lambdaId[mstate.get(mparams)].read(), cFactor);
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addClambda(const MechanicalParams* /*mparams*/, DataVecDeriv& /*df*/, const DataVecDeriv& /*lambda*/, SReal /*cFactor*/ )
{
    serr<<"function 'addClambda' is not implemented"<<sendl;
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
    else serr<<"addKToMatrix found no valid matrix accessor." << sendl;
}

template<class DataTypes>
void ForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, SReal /*kFact*/, unsigned int &/*offset*/)
{
    static int i=0;
    if (i < 10) {
        serr << "addKToMatrix not implemented." << sendl;
        i++;
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addSubKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> & subMatrixIndex )
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r) addSubKToMatrix(r.matrix, subMatrixIndex, mparams->kFactorIncludingRayleighDamping(rayleighStiffness.getValue()), r.offset);
    else serr<<"addKToMatrix found no valid matrix accessor." << sendl;
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
//        serr << "addBToMatrix not implemented." << sendl;
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
