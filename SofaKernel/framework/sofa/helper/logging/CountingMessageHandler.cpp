/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <cassert>
#include <sofa/helper/logging/CountingMessageHandler.h>

namespace sofa
{
namespace helper
{
namespace logging
{
namespace countingmessagehandler
{

void CountingMessageHandler::process(Message& m)
{
    assert(m.type()<m_countMatching.size() && "If this happens this means that the code initializing m_countMatching is broken.") ;

    m_countMatching[m.type()]++ ;
}

void CountingMessageHandler::reset(){
    for(unsigned int i=0;i<m_countMatching.size();i++){
        m_countMatching[i] = 0 ;
    }
}

CountingMessageHandler::CountingMessageHandler() {
    for(unsigned int i=Message::Info;i<Message::TypeCount;i++){
        m_countMatching.push_back(0) ;
    }
}

int CountingMessageHandler::getMessageCountFor(const Message::Type& type) const {
    assert(type < m_countMatching.size() && "If this happens this means that the code initializing m_countMatching is broken.") ;
    return m_countMatching[type] ;
}


sofa::helper::logging::CountingMessageHandler& MainCountingMessageHandler::getInstance()
{
    static sofa::helper::logging::CountingMessageHandler s_instance;
    return s_instance;
}

void MainCountingMessageHandler::reset(){
    getInstance().reset() ;
}

int MainCountingMessageHandler::getMessageCountFor(const Message::Type &type)
{
    return getInstance().getMessageCountFor(type) ;
}


} /// namespace countingmessagehandler
} /// namespace logging
} /// namespace helper
} /// namespace sofa
