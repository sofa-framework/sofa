/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_CORE_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL
#define SOFA_CORE_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL

#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <iostream>

namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
PairInteractionForceField<DataTypes>::PairInteractionForceField(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : mstate1(initLink("object1", "First object in interaction"), mm1)
    , mstate2(initLink("object2", "Second object in interaction"), mm2)
{
    if (!mm1)
        mstate1.setPath("@./"); // default to state of the current node
    if (!mm2)
        mstate2.setPath("@./"); // default to state of the current node
}

template<class DataTypes>
PairInteractionForceField<DataTypes>::~PairInteractionForceField()
{
}


template<class DataTypes>
void PairInteractionForceField<DataTypes>::init()
{

    BaseInteractionForceField::init();

    if (mstate1.get() == NULL || mstate2.get() == NULL)
    {
        serr<< "Init of PairInteractionForceField " << getContext()->getName() << " failed!" << sendl;
        //getContext()->removeObject(this);
        return;
    }
}

#ifdef SOFA_SMP
template <class DataTypes>
struct ParallelPairInteractionForceFieldAddForce
{
    void	operator()(const MechanicalParams* mparams, PairInteractionForceField<DataTypes> *ff,
            Shared_rw<objectmodel::Data< typename DataTypes::VecDeriv> > _f1,Shared_rw<objectmodel::Data< typename DataTypes::VecDeriv> > _f2,
            Shared_r<objectmodel::Data< typename DataTypes::VecCoord> > _x1,Shared_r<objectmodel::Data< typename DataTypes::VecCoord> > _x2,
            Shared_r<objectmodel::Data< typename DataTypes::VecDeriv> > _v1,Shared_r<objectmodel::Data< typename DataTypes::VecDeriv> > _v2)
    {
        helper::WriteAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > f1= _f1.access();
        helper::WriteAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > f2= _f2.access();
        helper::ReadAccessor< objectmodel::Data<typename DataTypes::VecCoord> > x1= _x1.read();
        helper::ReadAccessor< objectmodel::Data<typename DataTypes::VecCoord> > x2 = _x2.read();
        if(0&&x1.size()!=f1.size())
        {
            f1.resize(x1.size());
        }
        if(0&&x2.size()!=f2.size())
        {
            f2.resize(x2.size());
        }
        ff->addForce(mparams, _f1.access(),_f2.access(),_x1.read(),_x2.read(),_v1.read(),_v2.read());
    }

    void	operator()(const MechanicalParams *mparams, PairInteractionForceField<DataTypes> *ff,
            Shared_rw<objectmodel::Data< typename DataTypes::VecDeriv> > _f1,
            Shared_r<objectmodel::Data< typename DataTypes::VecCoord> > _x1,
            Shared_r<objectmodel::Data< typename DataTypes::VecDeriv> > _v1)
    {
        helper::WriteAccessor< objectmodel::Data< typename DataTypes::VecDeriv > > f1= _f1.access();

        helper::ReadAccessor< objectmodel::Data< typename DataTypes::VecCoord> > x1= _x1.read();
        helper::ReadAccessor< objectmodel::Data< typename DataTypes::VecDeriv> > v1= _v1.read();
        if(0&&x1.size()!=f1.size())
        {
            f1.resize(x1.size());
        }
        ff->addForce(mparams, _f1.access(),_f1.access(),_x1.read(),_x1.read(),_v1.read(),_v1.read());
    }

};


template <class DataTypes>
struct ParallelPairInteractionForceFieldAddDForce
{
    void	operator()(const MechanicalParams* mparams, PairInteractionForceField<DataTypes> *ff,
            Shared_rw<objectmodel::Data< typename DataTypes::VecDeriv> > _df1,Shared_rw<objectmodel::Data< typename DataTypes::VecDeriv> > _df2,
            Shared_r<objectmodel::Data< typename DataTypes::VecDeriv> > _dx1,Shared_r<objectmodel::Data< typename DataTypes::VecDeriv> > _dx2)
    {
        helper::WriteAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > df1= _df1.access();
        helper::WriteAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > df2= _df2.access();
        helper::ReadAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > dx1 = _dx1.read();
        helper::ReadAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > dx2 = _dx2.read();

        if(0&&dx1.size()!=df1.size())
        {
            df1.resize(dx1.size());
        }
        if(0&&dx2.size()!=df2.size())
        {
            df2.resize(dx2.size());
        }
        // mparams->setKFactor(1.0);
        ff->addDForce(mparams, _df1.access(),_df2.access(),_dx1.read(),_dx2.read());
    }

    void	operator()(const MechanicalParams* mparams, PairInteractionForceField<DataTypes> *ff,Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > _df1, Shared_r< objectmodel::Data< typename DataTypes::VecDeriv> > _dx1)
    {
        helper::WriteAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > df1= _df1.access();
        helper::ReadAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > dx1= _dx1.read();

        if(0&&dx1.size()!=df1.size())
        {
            df1.resize(dx1.size());
        }
        // mparams->setKFactor(1.0);
        ff->addDForce(mparams, _df1.access(),_df1.access(),_dx1.read(),_dx1.read());
    }

}; // ParallelPairInteractionForceFieldAddDForce
#endif /* SOFA_SMP */

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addForce(const MechanicalParams* mparams, MultiVecDerivId fId )
{
    if (mstate1 && mstate2)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
        {
            if (mstate1 == mstate2)
                Task<ParallelPairInteractionForceFieldAddForce< DataTypes > >(mparams, this,
                        **defaulttype::getShared(*fId[mstate1.get(mparams)].write()),
                        **defaulttype::getShared(*mparams->readX(mstate1)), **defaulttype::getShared(*mparams->readV(mstate1)));
            else
                Task<ParallelPairInteractionForceFieldAddForce< DataTypes > >(mparams, this,
                        **defaulttype::getShared(*fId[mstate1.get(mparams)].write()), **defaulttype::getShared(*fId[mstate2.get(mparams)].write()),
                        **defaulttype::getShared(*mparams->readX(mstate1)), **defaulttype::getShared(*mparams->readX(mstate2)),
                        **defaulttype::getShared(*mparams->readV(mstate1)), **defaulttype::getShared(*mparams->readV(mstate2)));
        }
        else
#endif /* SOFA_SMP */
            addForce( mparams, *fId[mstate1.get(mparams)].write()   , *fId[mstate2.get(mparams)].write()   ,
                    *mparams->readX(mstate1), *mparams->readX(mstate2),
                    *mparams->readV(mstate1), *mparams->readV(mstate2) );

        updateForceMask();
    }
    else
        serr<<"PairInteractionForceField<DataTypes>::addForce(const MechanicalParams* /*mparams*/, MultiVecDerivId /*fId*/ ), mstate missing"<<sendl;
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId )
{
    if (mstate1 && mstate2)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
        {
            if (mstate1 == mstate2)
                Task<ParallelPairInteractionForceFieldAddDForce< DataTypes > >(mparams, this,
                        **defaulttype::getShared(*dfId[mstate1.get(mparams)].write()),
                        **defaulttype::getShared(*mparams->readDx(mstate1)));
            else
                Task<ParallelPairInteractionForceFieldAddDForce< DataTypes > >(mparams, this,
                        **defaulttype::getShared(*dfId[mstate1.get(mparams)].write()), **defaulttype::getShared(*dfId[mstate2.get(mparams)].write()),
                        **defaulttype::getShared(*mparams->readDx(mstate1)), **defaulttype::getShared(*mparams->readDx(mstate2)));
        }
        else
#endif /* SOFA_SMP */
            addDForce(
                mparams, *dfId[mstate1.get(mparams)].write()    , *dfId[mstate2.get(mparams)].write()   ,
                *mparams->readDx(mstate1) , *mparams->readDx(mstate2) );
    }
    else
        serr<<"PairInteractionForceField<DataTypes>::addDForce(const MechanicalParams* /*mparams*/, MultiVecDerivId /*fId*/ ), mstate missing"<<sendl;
}

/*
template<class DataTypes>
void PairInteractionForceField<DataTypes>::addForce(const MechanicalParams* mparams, DataVecDeriv& f1, DataVecDeriv& f2, const DataVecCoord& x1, const DataVecCoord& x2, const DataVecDeriv& v1, const DataVecDeriv& v2 )
{
    addForce( *f1.beginEdit(mparams) , *f2.beginEdit(mparams),
			  x1.getValue(mparams)   , x2.getValue(mparams)  ,
			  v1.getValue(mparams)   , v2.getValue(mparams) );
	f1.endEdit(mparams); f2.endEdit(mparams);
}
template<class DataTypes>
void PairInteractionForceField<DataTypes>::addForce(VecDeriv& , VecDeriv& , const VecCoord& , const VecCoord& , const VecDeriv& , const VecDeriv& )
{
    serr << "ERROR("<<getClassName()<<"): addForce not implemented." << sendl;
}
*/


/*
template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(const MechanicalParams* mparams, DataVecDeriv& df1, DataVecDeriv& df2, const DataVecDeriv& dx1, const DataVecDeriv& dx2)
{
	addDForce(*df1.beginEdit(mparams), *df2.beginEdit(mparams), dx1.getValue(mparams), dx2.getValue(mparams),mparams->kFactor(),mparams->bFactor());
	df1.endEdit(mparams); df2.endEdit(mparams);
}
template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, SReal kFactor, SReal)
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
            if (&mstate1->read(VecDerivId(vdx1))->getValue() == &dx1)
		break;
        VecDeriv* dx1scaled = mstate1->write(vtmp1)->beginEdit();
        dx1scaled->resize(dx1.size());
        mstate1->vOp(vtmp1,VecId::null(),vdx1,kFactor);
        //sout << "dx1 = "<<dx1<<sendl;
        //sout << "dx1*"<<kFactor<<" = "<<dx1scaled<<sendl;
        VecDerivId vtmp2(VecDerivId::V_FIRST_DYNAMIC_INDEX);
        mstate2->vAvail(vtmp2);
        mstate2->vAlloc(vtmp2);
        VecDerivId vdx2(0);
        /// @TODO: Add a better way to get the current VecId of dx
        for (vdx2.index=0;vdx2.index<vtmp2.index;++vdx2.index)

            if (&mstate2->read(VecDerivId(vdx2))->getValue() == &dx2)
		break;
        VecDeriv* dx2scaled = mstate2->write(vtmp2)->beginEdit();
        dx2scaled->resize(dx2.size());
        mstate2->vOp(vtmp2,VecId::null(),vdx2,kFactor);
        //sout << "dx2 = "<<dx2<<sendl;
        //sout << "dx2*"<<kFactor<<" = "<<dx2scaled<<sendl;

        addDForce(df1, df2, *dx1scaled, *dx2scaled);

        mstate1->write(vtmp1)->endEdit();
        mstate2->write(vtmp2)->endEdit();

		mstate1->vFree(vtmp1);
		mstate2->vFree(vtmp2);
    }
}
template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(VecDeriv& , VecDeriv&, const VecDeriv&, const VecDeriv& )
{
    serr << "ERROR("<<getClassName()<<"): addDForce not implemented." << sendl;
}

*/



template<class DataTypes>
SReal PairInteractionForceField<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (mstate1 && mstate2)
        return getPotentialEnergy(mparams, *mparams->readX(mstate1),*mparams->readX(mstate2));
    else return 0.0;
}

/*
template<class DataTypes>
SReal PairInteractionForceField<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams, const DataVecCoord& x1, const DataVecCoord& x2 ) const
{
	return getPotentialEnergy( x1.getValue(mparams) , x2.getValue(mparams) );
}
template<class DataTypes>
SReal PairInteractionForceField<DataTypes>::getPotentialEnergy(const VecCoord& , const VecCoord& ) const
{
    serr << "ERROR("<<getClassName()<<"): getPotentialEnergy(const VecCoord1& , const VecCoord2&) not implemented." << sendl;
    return 0.0;
}
*/


template<class DataTypes>
void PairInteractionForceField<DataTypes>::updateForceMask()
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
