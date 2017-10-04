/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-20ll6 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Contributors:                                                               *
*       - damien.marchal@univ-lille1.fr                                       *
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

