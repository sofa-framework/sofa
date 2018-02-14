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
#ifndef SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONFORCEFIELD_INL
#define SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONFORCEFIELD_INL

#include "MixedInteractionForceField.h"

namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes1, class DataTypes2>
MixedInteractionForceField<DataTypes1, DataTypes2>::MixedInteractionForceField(MechanicalState<DataTypes1> *mm1, MechanicalState<DataTypes2> *mm2)
    : mstate1(initLink("object1", "First object in interaction"), mm1)
    , mstate2(initLink("object2", "Second object in interaction"), mm2)
{
    if (!mm1)
        mstate1.setPath("@./"); // default to state of the current node
    if (!mm2)
        mstate2.setPath("@./"); // default to state of the current node
}

template<class DataTypes1, class DataTypes2>
MixedInteractionForceField<DataTypes1, DataTypes2>::~MixedInteractionForceField()
{
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::init()
{
    BaseInteractionForceField::init();

    if (mstate1.get() == NULL || mstate2.get() == NULL)
    {
        serr<< "Init of MixedInteractionForceField " << getContext()->getName() << " failed!" << sendl;
        //getContext()->removeObject(this);
        return;
    }
}




template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addForce(const MechanicalParams* mparams, MultiVecDerivId fId )
{

    if (mstate1 && mstate2)
    {

            addForce( mparams, *fId[mstate1.get(mparams)].write()   , *fId[mstate2.get(mparams)].write()   ,
                    *mparams->readX(mstate1), *mparams->readX(mstate2),
                    *mparams->readV(mstate1), *mparams->readV(mstate2) );

        updateForceMask();
    }
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId )
{
    if (mstate1 && mstate2)
    {
            addDForce( mparams, *dfId[mstate1.get(mparams)].write()    , *dfId[mstate2.get(mparams)].write()   ,
                    *mparams->readDx(mstate1) , *mparams->readDx(mstate2) );
    }
}



template<class DataTypes1, class DataTypes2>
SReal MixedInteractionForceField<DataTypes1, DataTypes2>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (mstate1 && mstate2)
        return getPotentialEnergy(mparams, *mparams->readX(mstate1),*mparams->readX(mstate2));
    else return 0;
}

/*
template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addForce(const MechanicalParams* mparams, DataVecDeriv1& f1, DataVecDeriv2& f2, const DataVecCoord1& x1, const DataVecCoord2& x2, const DataVecDeriv1& v1, const DataVecDeriv2& v2 )
{
    addForce( *f1.beginEdit(mparams) , *f2.beginEdit(mparams),
			  x1.getValue(mparams)   , x2.getValue(mparams)  ,
			  v1.getValue(mparams)   , v2.getValue(mparams) );
	f1.endEdit(mparams); f2.endEdit(mparams);
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addForce(VecDeriv1& , VecDeriv2& , const VecCoord1& , const VecCoord2& , const VecDeriv1& , const VecDeriv2& )
{
    serr << "ERROR("<<getClassName()<<"): addForce not implemented." << sendl;
}
*/

/*
template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForce(const MechanicalParams* mparams, DataVecDeriv1& df1, DataVecDeriv2& df2, const DataVecDeriv1& dx1, const DataVecDeriv2& dx2)
{
    	addDForce(*df1.beginEdit(mparams), *df2.beginEdit(mparams), dx1.getValue(mparams), dx2.getValue(mparams),mparams->kFactor(),mparams->bFactor());
    	df1.endEdit(mparams); df2.endEdit(mparams);
}
template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForce(VecDeriv1& df1, VecDeriv2& df2, const VecDeriv1& dx1, const VecDeriv2& dx2, SReal kFactor, SReal )
{
    if (kFactor == 1.0)
        addDForce(df1, df2, dx1, dx2);
    else if (kFactor != 0.0)
    {
        VecDerivId vtmp1(VecDerivId::V_FIRST_DYNAMIC_INDEX);
        mstate1->vAvail(vtmp1);
        mstate1->vAlloc(vtmp1);
        VecDerivId vdx1(0);
        /// @TODO: Add a better way to get the current VecId of dx
        for (vdx1.index=0;vdx1.index<vtmp1.index;++vdx1.index)
             if (&mstate1->read(ConstVecDerivId(vdx1))->getValue() == &dx1)
		break;
        mstate1->vOp(vtmp1,VecId::null(),vdx1,kFactor);

        VecDerivId vtmp2(VecDerivId::V_FIRST_DYNAMIC_INDEX);
		mstate2->vAvail(vtmp2);
		mstate2->vAlloc(vtmp2);
		VecId vdx2(sofa::core::V_DERIV,0);
		/// @TODO: Add a better way to get the current VecId of dx
		for (vdx2.index=0;vdx2.index<vtmp2.index;++vdx2.index)
				if (&mstate2->read(ConstVecDerivId(vdx2))->getValue() == &dx2)
			break;
		mstate2->vOp(vtmp2,VecId::null(),vdx2,kFactor);

			//addDForce(df1, df2, *mstate1->getVecDeriv(vtmp1.index), *mstate2->getVecDeriv(vtmp2.index));
			addDForce(df1, df2, mstate1->read(ConstVecDerivId(vtmp1))->getValue(), mstate2->read(ConstVecDerivId(vtmp2))->getValue());

		mstate1->vFree(vtmp1);
		mstate2->vFree(vtmp2);
    }
}
template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForce(VecDeriv1& , VecDeriv2& , const VecDeriv1& , const VecDeriv2& )
{
    serr << "ERROR("<<getClassName()<<"): addDForce not implemented." << sendl;
}
*/

/*
template<class DataTypes1, class DataTypes2>
SReal MixedInteractionForceField<DataTypes1, DataTypes2>::getPotentialEnergy(const MechanicalParams* mparams, const DataVecCoord1& x1, const DataVecCoord2& x2) const
{
	return getPotentialEnergy( x1.getValue(mparams) , x2.getValue(mparams) );
}

template<class DataTypes1, class DataTypes2>
SReal MixedInteractionForceField<DataTypes1, DataTypes2>::getPotentialEnergy(const VecCoord1& , const VecCoord2& ) const
{
    serr << "ERROR("<<getClassName()<<"): getPotentialEnergy(const VecCoord1& , const VecCoord2&) not implemented." << sendl;
    return 0.0;
}
*/


template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::updateForceMask()
{
    // the default implementation adds every dofs to the mask
    // this sould be overloaded by each forcefield to only add the implicated dofs subset to the mask
    mstate1->forceMask.assign( mstate1->getSize(), true );
    mstate2->forceMask.assign( mstate2->getSize(), true );
}




} // namespace behavior

} // namespace core

} // namespace sofa

#endif
