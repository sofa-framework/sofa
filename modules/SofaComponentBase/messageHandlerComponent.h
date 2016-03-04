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
*                               SOFA :: Modules                               *
*                                                                             *
* This component is open-source                                               *
*                                                                             *
* Authors: Damien Marchal                                                     *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/
#ifndef SOFA_MESSAGEHANDLERCOMPONENT_H
#define SOFA_MESSAGEHANDLERCOMPONENT_H

#include "config.h"

#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>

#include <string>

namespace sofa
{
namespace helper
{
namespace logging
{
    class MessageHandler ;
}
}
}

namespace sofa
{
namespace component
{
namespace logging
{

/// A sofa component to add a MessageHandler to the main logger
class SOFA_COMPONENT_BASE_API MessageHandlerComponent : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(MessageHandlerComponent, core::objectmodel::BaseObject);

    MessageHandlerComponent() ;
    virtual ~MessageHandlerComponent(){}

    /// Inherited from BaseObject.
    /// Parse the given description to assign values to this object's fields and
    /// potentially other parameters.
    virtual void parse ( core::objectmodel::BaseObjectDescription* arg ) ;

    Data<std::string>        d_type       ;
    bool                m_isValid    ;

    bool isValid(){ return m_isValid; }
};


/// A sofa component to add a FileMessageHandlerComponent to the main logger
class SOFA_COMPONENT_BASE_API FileMessageHandlerComponent : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(FileMessageHandlerComponent, core::objectmodel::BaseObject) ;

    FileMessageHandlerComponent() ;
    virtual ~FileMessageHandlerComponent() ;

    /// Inherited from BaseObject.
    /// Parse the given description to assign values to this object's fields and
    /// potentially other parameters.
    virtual void parse ( core::objectmodel::BaseObjectDescription* arg ) ;

    Data<std::string>        d_filename        ;
    helper::logging::MessageHandler*     m_handler         ;


    bool                m_isValid    ;
    bool isValid(){ return m_isValid; }
};

}
}
}
#endif // SOFA_MESSAGEHANDLERCOMPONENT_H
