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
    : _object1(initData(&_object1, "object1", "First object in interaction")),
      _object2(initData(&_object2, "object2", "Second object in interaction")),
      mstate1(mm1), mstate2(mm2), mask1(NULL), mask2(NULL)
{

    if(mm1==0||mm2==0)
        return;
}

template<class DataTypes>
PairInteractionForceField<DataTypes>::~PairInteractionForceField()
{
    if(mstate1==0||mstate2==0)
        return;
}


template<class DataTypes>
BaseMechanicalState*  PairInteractionForceField<DataTypes>::getMState(sofa::core::objectmodel::BaseContext* context, std::string path)
{
    std::string::size_type pos_slash = path.find("/");

    sofa::core::objectmodel::BaseNode* currentNode = dynamic_cast< sofa::core::objectmodel::BaseNode *>(context);
    if (pos_slash == std::string::npos)
    {
        if (path.empty())
        {
            BaseMechanicalState *result;
            context->get(result, sofa::core::objectmodel::BaseContext::SearchDown);
            return result;
        }
        sofa::helper::vector< sofa::core::objectmodel::BaseNode* > list_child = currentNode->getChildren();

        for (unsigned int i=0; i< list_child.size(); ++i)
        {
            if (list_child[i]->getName() == path)
            {
                sofa::core::objectmodel::BaseContext *c = list_child[i]->getContext();
                BaseMechanicalState *result;
                c->get(result, sofa::core::objectmodel::BaseContext::SearchDown);
                return result;
            }
        }
    }
    else
    {
        std::string name_expected = path.substr(0,pos_slash);
        path = path.substr(pos_slash+1);
        sofa::helper::vector< sofa::core::objectmodel::BaseNode* > list_child = currentNode->getChildren();

        for (unsigned int i=0; i< list_child.size(); ++i)
        {
            if (list_child[i]->getName() == name_expected)
                return getMState(list_child[i]->getContext(), path);
        }
    }
    return NULL;
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::init()
{

    BaseInteractionForceField::init();
    if (mstate1 == NULL || mstate2 == NULL)
    {
        std::string path_object1 = _object1.getValue();
        std::string path_object2 = _object2.getValue();

        mstate1 =  dynamic_cast< MechanicalState<DataTypes>* >( getMState(getContext(), path_object1));
        mstate2 =  dynamic_cast< MechanicalState<DataTypes>* >( getMState(getContext(), path_object2));
        if (mstate1 == NULL || mstate2 == NULL)
        {
            serr<< "Init of PairInteractionForceField " << getContext()->getName() << " failed!" << sendl;
            getContext()->removeObject(this);
            return;
        }
    }
    else
    {
        //Interaction created by passing Mechanical State directly, need to find the name of the path to be able to save the scene eventually

        /*
           if (mstate1->getContext() != getContext())
           {
             sofa::core::objectmodel::BaseContext *context = NULL;
             sofa::core::objectmodel::BaseNode*    currentNode = dynamic_cast< sofa::core::objectmodel::BaseNode *>(mstate1->getContext());

             std::string object_name=currentNode->getPathName();
             if (context != NULL) _object1.setValue(object_name);
           }


           if (mstate2->getContext() != getContext())
           {
             sofa::core::objectmodel::BaseContext *context = NULL;
             sofa::core::objectmodel::BaseNode*    currentNode = dynamic_cast< sofa::core::objectmodel::BaseNode *>(mstate2->getContext());

             std::string object_name=currentNode->getPathName();
             if (context != NULL) _object2.setValue(object_name);
           }
        */
    }

    this->mask1 = &mstate1->forceMask;
    this->mask2 = &mstate2->forceMask;
}

#ifdef SOFA_SMP
template <class DataTypes>
struct ParallelPairInteractionForceFieldAddForce
{
    void	operator()(const MechanicalParams* mparams /* PARAMS FIRST */, PairInteractionForceField<DataTypes> *ff,
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
        ff->addForce(mparams /* PARAMS FIRST */, _f1.access(),_f2.access(),_x1.read(),_x2.read(),_v1.read(),_v2.read());
    }

    void	operator()(const MechanicalParams *mparams /* PARAMS FIRST */, PairInteractionForceField<DataTypes> *ff,
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
        ff->addForce(mparams /* PARAMS FIRST */, _f1.access(),_f1.access(),_x1.read(),_x1.read(),_v1.read(),_v1.read());
    }

};


template <class DataTypes>
struct ParallelPairInteractionForceFieldAddDForce
{
    void	operator()(const MechanicalParams* mparams /* PARAMS FIRST */, PairInteractionForceField<DataTypes> *ff,
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
        ff->addDForce(mparams /* PARAMS FIRST */, _df1.access(),_df2.access(),_dx1.read(),_dx2.read());
    }

    void	operator()(const MechanicalParams* mparams /* PARAMS FIRST */, PairInteractionForceField<DataTypes> *ff,Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > _df1, Shared_r< objectmodel::Data< typename DataTypes::VecDeriv> > _dx1)
    {
        helper::WriteAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > df1= _df1.access();
        helper::ReadAccessor< objectmodel::Data<typename DataTypes::VecDeriv> > dx1= _dx1.read();

        if(0&&dx1.size()!=df1.size())
        {
            df1.resize(dx1.size());
        }
        // mparams->setKFactor(1.0);
        ff->addDForce(mparams /* PARAMS FIRST */, _df1.access(),_df1.access(),_dx1.read(),_dx1.read());
    }

}; // ParallelPairInteractionForceFieldAddDForce
#endif /* SOFA_SMP */

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addForce(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId fId )
{
    if (mstate1 && mstate2)
    {
        mstate1->forceMask.setInUse(this->useMask());
        mstate2->forceMask.setInUse(this->useMask());

#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
        {
            if (mstate1 == mstate2)
                Task<ParallelPairInteractionForceFieldAddForce< DataTypes > >(mparams /* PARAMS FIRST */, this,
                        **defaulttype::getShared(*fId[mstate1].write()),
                        **defaulttype::getShared(*mparams->readX(mstate1)), **defaulttype::getShared(*mparams->readV(mstate1)));
            else
                Task<ParallelPairInteractionForceFieldAddForce< DataTypes > >(mparams /* PARAMS FIRST */, this,
                        **defaulttype::getShared(*fId[mstate1].write()), **defaulttype::getShared(*fId[mstate2].write()),
                        **defaulttype::getShared(*mparams->readX(mstate1)), **defaulttype::getShared(*mparams->readX(mstate2)),
                        **defaulttype::getShared(*mparams->readV(mstate1)), **defaulttype::getShared(*mparams->readV(mstate2)));
        }
        else
#endif /* SOFA_SMP */
            addForce( mparams /* PARAMS FIRST */, *fId[mstate1].write()   , *fId[mstate2].write()   ,
                    *mparams->readX(mstate1), *mparams->readX(mstate2),
                    *mparams->readV(mstate1), *mparams->readV(mstate2) );
    }
    else
        serr<<"PairInteractionForceField<DataTypes>::addForce(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, MultiVecDerivId /*fId*/ ), mstate missing"<<sendl;
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId dfId )
{
    if (mstate1 && mstate2)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
        {
            if (mstate1 == mstate2)
                Task<ParallelPairInteractionForceFieldAddDForce< DataTypes > >(mparams /* PARAMS FIRST */, this,
                        **defaulttype::getShared(*dfId[mstate1].write()),
                        **defaulttype::getShared(*mparams->readDx(mstate1)));
            else
                Task<ParallelPairInteractionForceFieldAddDForce< DataTypes > >(mparams /* PARAMS FIRST */, this,
                        **defaulttype::getShared(*dfId[mstate1].write()), **defaulttype::getShared(*dfId[mstate2].write()),
                        **defaulttype::getShared(*mparams->readDx(mstate1)), **defaulttype::getShared(*mparams->readDx(mstate2)));
        }
        else
#endif /* SOFA_SMP */
            addDForce(
                mparams /* PARAMS FIRST */, *dfId[mstate1].write()    , *dfId[mstate2].write()   ,
                *mparams->readDx(mstate1) , *mparams->readDx(mstate2) );
    }
    else
        serr<<"PairInteractionForceField<DataTypes>::addDForce(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, MultiVecDerivId /*fId*/ ), mstate missing"<<sendl;
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
void PairInteractionForceField<DataTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double)
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
double PairInteractionForceField<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (mstate1 && mstate2)
        return getPotentialEnergy(mparams /* PARAMS FIRST */, *mparams->readX(mstate1),*mparams->readX(mstate2));
    else return 0.0;
}

/*
template<class DataTypes>
double PairInteractionForceField<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams, const DataVecCoord& x1, const DataVecCoord& x2 ) const
{
	return getPotentialEnergy( x1.getValue(mparams) , x2.getValue(mparams) );
}
template<class DataTypes>
double PairInteractionForceField<DataTypes>::getPotentialEnergy(const VecCoord& , const VecCoord& ) const
{
    serr << "ERROR("<<getClassName()<<"): getPotentialEnergy(const VecCoord1& , const VecCoord2&) not implemented." << sendl;
    return 0.0;
}
*/









} // namespace behavior

} // namespace core

} // namespace sofa

#endif
