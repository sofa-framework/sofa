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
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseContext.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

BaseObjectTasks::~BaseObjectTasks()
{}

Iterative::IterativePartition* BaseObject::prepareTask()
{
    Iterative::IterativePartition *p=NULL;
    sofa::core::objectmodel::Context *context=dynamic_cast<sofa::core::objectmodel::Context *>(this->getContext());
    if(this->getPartition())
    {
        p=this->getPartition();
    }
    else if(context&&context->is_partition())
    {
        p=context->getPartition();
    }
    return p;
}

} // namespace objectmodel

} // namespace core

} // namespace sofa

