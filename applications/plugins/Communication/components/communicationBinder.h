/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMMUNICATIONBINDER_H
#define SOFA_COMMUNICATIONBINDER_H

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/core/objectmodel/Event.h>
using sofa::core::objectmodel::Event;

#include <sofa/simulation/AnimateBeginEvent.h>

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;

#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace communication
{
template <class DataTypes>
class CommunicationBinder : public BaseObject
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(CommunicationBinder, DataTypes), BaseObject);

    Data<DataTypes> m_data;

    CommunicationBinder() ;
    virtual ~CommunicationBinder() ;

    virtual void init();
    virtual void handleEvent(Event *);

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }


};

} /// communication
} /// component
} /// sofa

#endif // SOFA_COMMUNICATIONBINDER_H
