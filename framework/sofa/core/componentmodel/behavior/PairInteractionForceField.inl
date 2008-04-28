/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL

#include <sofa/core/objectmodel/DataPtr.h>
#include "PairInteractionForceField.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseNode.h>

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
      mstate1(mm1), mstate2(mm2)
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
            std::cerr<< "Init of PairInteractionForceField " << getContext()->getName() << " failed!\n";
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

            std::string object_name=currentNode->getContext()->getName();
            currentNode = dynamic_cast< sofa::core::objectmodel::BaseNode *>(currentNode->getParent());
            while (currentNode != NULL)
            {
                context = currentNode->getContext();
                if (context == this->getContext()) break;
                object_name = context->getName() + "/" + object_name;
                currentNode = dynamic_cast< sofa::core::objectmodel::BaseNode *>(currentNode->getParent());
            }
            if (context != NULL) _object1.setValue(object_name);
        }


        if (mstate2->getContext() != getContext())
        {
            sofa::core::objectmodel::BaseContext *context = NULL;
            sofa::core::objectmodel::BaseNode*    currentNode = dynamic_cast< sofa::core::objectmodel::BaseNode *>(mstate2->getContext());

            std::string object_name=currentNode->getContext()->getName();
            currentNode = dynamic_cast< sofa::core::objectmodel::BaseNode *>(currentNode->getParent());
            while (currentNode != NULL)
            {
                context = currentNode->getContext();
                if (context == this->getContext()) break;
                object_name = context->getName() + "/" + object_name;
                currentNode = dynamic_cast< sofa::core::objectmodel::BaseNode *>(currentNode->getParent());
            }
            if (context != NULL) _object2.setValue(object_name);
        }
    }

}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addForce()
{
    if (mstate1 && mstate2)
        addForce(*mstate1->getF(), *mstate2->getF(),
                *mstate1->getX(), *mstate2->getX(),
                *mstate1->getV(), *mstate2->getV());
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce()
{
    if (mstate1 && mstate2)
        addDForce(*mstate1->getF(),  *mstate2->getF(),
                *mstate1->getDx(), *mstate2->getDx());
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForceV()
{
    if (mstate1 && mstate2)
        addDForce(*mstate1->getF(),  *mstate2->getF(),
                *mstate1->getV(), *mstate2->getV());
}


template<class DataTypes>
sofa::defaulttype::Vector3::value_type PairInteractionForceField<DataTypes>::getPotentialEnergy()
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
