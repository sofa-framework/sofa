/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/helper/logging/RoutingMessageHandler.h>


namespace sofa::helper::logging::routingmessagehandler
{

void RoutingMessageHandler::process(Message& m)
{
    for(const auto& f : m_filters)
    {
        if(f.first(m))
        {
            f.second->process(m) ;
        }
    }
}

RoutingMessageHandler::RoutingMessageHandler()
{
}

void RoutingMessageHandler::setAFilter(FilterFunction f, MessageHandler* handler)
{
    m_filters.push_back(std::make_pair(f, handler));
}

void RoutingMessageHandler::removeAllFilters()
{
    m_filters.clear();
}

RoutingMessageHandler& MainRoutingMessageHandler::getInstance()
{
    static RoutingMessageHandler s_instance;
    return s_instance;
}

void MainRoutingMessageHandler::setAFilter(FilterFunction filter,
                                           MessageHandler* handler)
{
    getInstance().setAFilter(filter, handler);
}

void MainRoutingMessageHandler::removeAllFilters()
{
    getInstance().removeAllFilters() ;
}


}
