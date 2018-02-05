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
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/logging/PerComponentLoggingMessageHandler.h>

namespace sofa
{
namespace helper
{
namespace logging
{
namespace percomponentloggingmessagehandler
{

void PerComponentLoggingMessageHandler::process(Message& m)
{
    SofaComponentInfo* nfo = dynamic_cast<SofaComponentInfo*>( m.componentInfo().get() ) ;
    if(nfo != nullptr)
    {
        nfo->m_component->addMessage( m ) ;
    }
}

PerComponentLoggingMessageHandler::PerComponentLoggingMessageHandler()
{
}

PerComponentLoggingMessageHandler& MainPerComponentLoggingMessageHandler::getInstance()
{
    static PerComponentLoggingMessageHandler s_instance;
    return s_instance;
}


} // percomponentloggingmessagehandler
} // logging
} // helper
} // sofa

