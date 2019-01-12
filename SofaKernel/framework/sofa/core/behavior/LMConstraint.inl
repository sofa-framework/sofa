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
#ifndef SOFA_CORE_BEHAVIOR_LMCONSTRAINT_INL
#define SOFA_CORE_BEHAVIOR_LMCONSTRAINT_INL

#include <sofa/core/behavior/LMConstraint.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/objectmodel/BaseNode.h>


namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes1,class DataTypes2>
LMConstraint<DataTypes1,DataTypes2>::~LMConstraint()
{
}


template<class DataTypes1,class DataTypes2>
void LMConstraint<DataTypes1,DataTypes2>::init()
{
    using sofa::core::objectmodel::BaseContext;
    using sofa::core::objectmodel::BaseNode;

    BaseLMConstraint::init();

    if (constrainedObject1 != NULL && constrainedObject2 != NULL)
    {
        //Constraint created by passing Mechanical State directly, need to find the name of the path to be able to save the scene eventually

        if (constrainedObject1->getContext() != getContext())
        {
            BaseContext *context = NULL;
            BaseNode *currentNode = constrainedObject1->getContext()->toBaseNode();

            std::string constrainedObject_name = currentNode->getPathName();
            if (context != NULL)
                this->pathObject1.setValue(constrainedObject_name);
        }

        if (constrainedObject2->getContext() != getContext())
        {
            BaseContext *context = NULL;
            BaseNode *currentNode = constrainedObject2->getContext()->toBaseNode();

            std::string constrainedObject_name = currentNode->getPathName();
            if (context != NULL)
                this->pathObject2.setValue(constrainedObject_name);
        }

        simulatedObject1 = constrainedObject1;

        while (simulatedObject1)
        {
            core::BaseMapping* mapping;
            simulatedObject1->getContext()->get(mapping);
            if (!mapping)
                break;
            if(!mapping->isMechanical())
                break;
            simulatedObject1 = mapping->getMechFrom()[0];
        }

        simulatedObject2 = constrainedObject2;

        while (simulatedObject2)
        {
            core::BaseMapping* mapping;
            simulatedObject2->getContext()->get(mapping);
            if (!mapping)
                break;
            if(!mapping->isMechanical())
                break;
            simulatedObject2 = mapping->getMechFrom()[0];
        }
    }
}


} // namespace behavior

} // namespace core

} // namespace sofa

#endif
