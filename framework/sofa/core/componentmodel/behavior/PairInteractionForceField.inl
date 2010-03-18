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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL

#include <sofa/core/objectmodel/DataPtr.h>
#include "PairInteractionForceField.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <iostream>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes>
PairInteractionForceField<DataTypes>::PairInteractionForceField(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : _object1(initData(&_object1, "object1", "First object in interaction")),
      _object2(initData(&_object2, "object2", "Second object in interaction")),
      mstate1(mm1), mstate2(mm2), mask1(NULL), mask2(NULL)
{
}

template<class DataTypes>
PairInteractionForceField<DataTypes>::~PairInteractionForceField()
{
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

    InteractionForceField::init();
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
    }

    this->mask1 = &mstate1->forceMask;
    this->mask2 = &mstate2->forceMask;
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addForce()
{
    if (mstate1 && mstate2)
    {
        mstate1->forceMask.setInUse(this->useMask());
        mstate2->forceMask.setInUse(this->useMask());
        addForce(*mstate1->getF(), *mstate2->getF(),
                *mstate1->getX(), *mstate2->getX(),
                *mstate1->getV(), *mstate2->getV());
    }
    else serr<<"PairInteractionForceField<DataTypes>::addForce(), mstate missing"<<sendl;
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(double kFactor, double bFactor)
{
    if (mstate1 && mstate2)
        addDForce(*mstate1->getF(),  *mstate2->getF(),
                *mstate1->getDx(), *mstate2->getDx(),
                kFactor, bFactor);
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForceV(double kFactor, double bFactor)
{
    if (mstate1 && mstate2)
        addDForce(*mstate1->getF(),  *mstate2->getF(),
                *mstate1->getV(), *mstate2->getV(),
                kFactor, bFactor);
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(VecDeriv& /*df1*/, VecDeriv& /*df2*/, const VecDeriv& /*dx1*/, const VecDeriv& /*dx2*/)
{
    serr << "ERROR("<<getClassName()<<"): addDForce not implemented." << sendl;
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double /*bFactor*/)
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
        VecDeriv& dx1scaled = *mstate1->getVecDeriv(vtmp1.index);
        dx1scaled.resize(dx1.size());
        mstate1->vOp(vtmp1,BaseMechanicalState::VecId::null(),vdx1,kFactor);
        //sout << "dx1 = "<<dx1<<sendl;
        //sout << "dx1*"<<kFactor<<" = "<<dx1scaled<<sendl;
        BaseMechanicalState::VecId vtmp2(BaseMechanicalState::VecId::V_DERIV,BaseMechanicalState::VecId::V_FIRST_DYNAMIC_INDEX);
        mstate2->vAvail(vtmp2);
        mstate2->vAlloc(vtmp2);
        BaseMechanicalState::VecId vdx2(BaseMechanicalState::VecId::V_DERIV,0);
        /// @TODO: Add a better way to get the current VecId of dx
        for (vdx2.index=0; vdx2.index<vtmp2.index; ++vdx2.index)
            if (mstate2->getVecDeriv(vdx2.index) == &dx2)
                break;
        VecDeriv& dx2scaled = *mstate2->getVecDeriv(vtmp2.index);
        dx2scaled.resize(dx2.size());
        mstate2->vOp(vtmp2,BaseMechanicalState::VecId::null(),vdx2,kFactor);
        //sout << "dx2 = "<<dx2<<sendl;
        //sout << "dx2*"<<kFactor<<" = "<<dx2scaled<<sendl;

        addDForce(df1, df2, dx1scaled, dx2scaled);

        mstate1->vFree(vtmp1);
        mstate2->vFree(vtmp2);
    }
}

template<class DataTypes>
double PairInteractionForceField<DataTypes>::getPotentialEnergy() const
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
