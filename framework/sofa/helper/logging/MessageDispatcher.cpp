/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*          Bruno Carrez                                                       *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/
#include <sstream>
using std::ostringstream ;

#include <iostream>
using std::endl ;

#include <vector>
using std::vector ;

#include <string>
using std::string ;

#include <algorithm>
using std::remove ;

#include "MessageDispatcher.h"

#include "MessageHandler.h"
#include "ConsoleMessageHandler.h"


namespace sofa
{

namespace helper
{

namespace logging
{

// keep a track of messages.
vector<MessageHandler*> s_m_handlers ;

// todo(damien): currently the message list and id are shared by all the message handler.
int s_m_lastAllocatedID = -1 ; // keep a count of the last id allocated
int s_m_lastErrorId = -1 ;     // keep count of the last error message received.
int s_m_lastWarningId = -1 ;
int s_m_lastInfoId = -1 ;

sofa::helper::logging::Message& operator<<=(MessageDispatcher &d, sofa::helper::logging::Message& m){
    s_m_lastAllocatedID++ ;

    m.setId(s_m_lastAllocatedID) ;

    if(m.type()==Message::Error)
        s_m_lastErrorId = s_m_lastAllocatedID ;
    else if(m.type()==Message::Warning)
        s_m_lastWarningId = s_m_lastAllocatedID ;
    else if(m.type()==Message::Info)
        s_m_lastInfoId = s_m_lastAllocatedID ;

    d.process(m);
    return m;
}

MessageDispatcher::MessageDispatcher()
{
    // add a default handler
    addHandler(new sofa::helper::logging::ConsoleMessageHandler());
}


int MessageDispatcher::getLastMessageId() {
    return s_m_lastAllocatedID ;
}

int MessageDispatcher::getLastErrorId(){
    return s_m_lastErrorId ;
}

int MessageDispatcher::getLastWarningId(){
    return s_m_lastWarningId ;
}

int MessageDispatcher::getLastInfoId(){
    return s_m_lastInfoId ;
}

int MessageDispatcher::addHandler(MessageHandler* o){
    if( std::find(s_m_handlers.begin(), s_m_handlers.end(), o) == s_m_handlers.end()){
        s_m_handlers.push_back(o) ;
        return s_m_handlers.size()-1 ;
    }
    return -1;
}

int MessageDispatcher::rmHandler(MessageHandler* o){
    s_m_handlers.erase(remove(s_m_handlers.begin(), s_m_handlers.end(), o), s_m_handlers.end());
    return s_m_handlers.size()-1 ;
}

void MessageDispatcher::clearHandlers(bool del){
    if(del)
        for(unsigned int i = 0;i<s_m_handlers.size();i++)
            delete (s_m_handlers[i]) ;
    s_m_handlers.clear() ;
}

void MessageDispatcher::process(sofa::helper::logging::Message& m){
    for(unsigned int i=0;i<s_m_handlers.size();i++)
        s_m_handlers[i]->process(m) ;
}

} // logging
} // helper
} // sofa

// THE main MessageDipatcher...
SOFA_HELPER_API sofa::helper::logging::MessageDispatcher gMessageDispatcher;



