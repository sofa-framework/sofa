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

Nop Nop::s_nop;

////////////////////// THE UNDERLYING OBJECT ///////////////////////////////////
class SOFA_HELPER_API MessageDispatcherImpl
{
public:
    MessageDispatcherImpl();

    LoggerStream log(Message::Class mclass, Message::Type type,
                     const std::string& sender = "", FileInfo fileInfo = FileInfo()) {
        return LoggerStream(*this, mclass, type, sender, fileInfo);
    }

    LoggerStream log(Message::Class mclass, Message::Type type,
                     const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) {
        return LoggerStream(*this, mclass, type, sender, fileInfo);
    }

    int addHandler(MessageHandler* o) ;
    int rmHandler(MessageHandler* o) ;
    void clearHandlers() ;
    ConsoleMessageHandler* getDefaultMessageHandler();

//    int getLastMessageId() ;
//    int getLastErrorId() ;
//    int getLastWarningId() ;
//    int getLastInfoId() ;

    void process(sofa::helper::logging::Message &m);

private:
    vector<MessageHandler*> m_handlers ;

//    int m_lastAllocatedID ; // keep a count of the last id allocated
//    int m_lastErrorId ;     // keep count of the last error message received.
//    int m_lastWarningId ;
//    int m_lastInfoId ;


//    friend Message& operator<<=(MessageDispatcherImpl& d, Message& m) ;
};


//sofa::helper::logging::Message& operator<<=(MessageDispatcherImpl &d, sofa::helper::logging::Message& m){
//    d.m_lastAllocatedID++ ;

//    m.setId(d.m_lastAllocatedID) ;

//    if(m.type()==Message::Error)
//        d.m_lastErrorId = d.m_lastAllocatedID ;
//    else if(m.type()==Message::Warning)
//        d.m_lastWarningId = d.m_lastAllocatedID ;
//    else if(m.type()==Message::Info)
//        d.m_lastInfoId = d.m_lastAllocatedID ;

//    d.process(m);
//    return m;
//}

MessageDispatcherImpl::MessageDispatcherImpl()
{
//    m_lastAllocatedID = -1 ; // keep a count of the last id allocated
//    m_lastErrorId = -1 ;     // keep count of the last error message received.
//    m_lastWarningId = -1 ;
//    m_lastInfoId = -1 ;

    // add a default handler
    addHandler(getDefaultMessageHandler());
}


//int MessageDispatcherImpl::getLastMessageId() {
//    return m_lastAllocatedID ;
//}

//int MessageDispatcherImpl::getLastErrorId(){
//    return m_lastErrorId ;
//}

//int MessageDispatcherImpl::getLastWarningId(){
//    return m_lastWarningId ;
//}

//int MessageDispatcherImpl::getLastInfoId(){
//    return m_lastInfoId ;
//}

int MessageDispatcherImpl::addHandler(MessageHandler* o){
    if( std::find(m_handlers.begin(), m_handlers.end(), o) == m_handlers.end()){
        m_handlers.push_back(o) ;
        return m_handlers.size()-1 ;
    }
    return -1;
}

int MessageDispatcherImpl::rmHandler(MessageHandler* o){
    m_handlers.erase(remove(m_handlers.begin(), m_handlers.end(), o), m_handlers.end());
    return m_handlers.size()-1 ;
}

void MessageDispatcherImpl::clearHandlers(){
    m_handlers.clear() ;
}

void MessageDispatcherImpl::process(sofa::helper::logging::Message& m){
    for(unsigned int i=0;i<m_handlers.size();i++)
        m_handlers[i]->process(m) ;
}

ConsoleMessageHandler s_defaultMessageHandler;

ConsoleMessageHandler* MessageDispatcherImpl::getDefaultMessageHandler()
{
    return &s_defaultMessageHandler;
}


LoggerStream::~LoggerStream()
{
    if ( !m_message.empty() ) m_dispatcher.process(m_message);
}


} // logging
} // helper
} // sofa


/////////////////////////// UNIQUE NAMESPACE //////////////////////////////////
namespace sofa
{
namespace helper
{
namespace logging
{
namespace unique
{

// THE main MessageDipatcher...
sofa::helper::logging::MessageDispatcherImpl gMessageDispatcher;

int MessageDispatcher::addHandler(MessageHandler* o){
    return gMessageDispatcher.addHandler(o) ;
}

int MessageDispatcher::rmHandler(MessageHandler* o){
    return gMessageDispatcher.rmHandler(o) ;
}

void MessageDispatcher::clearHandlers(){
    gMessageDispatcher.clearHandlers() ;
}

ConsoleMessageHandler* MessageDispatcher::getDefaultMessageHandler(){
    return gMessageDispatcher.getDefaultMessageHandler() ;
}

//int MessageDispatcher::getLastMessageId(){
//    return gMessageDispatcher.getLastMessageId() ;
//}

//int MessageDispatcher::getLastErrorId(){
//    return gMessageDispatcher.getLastErrorId() ;
//}

//int MessageDispatcher::getLastWarningId(){
//    return gMessageDispatcher.getLastWarningId() ;
//}

//int MessageDispatcher::getLastInfoId(){
//    return gMessageDispatcher.getLastInfoId() ;
//}

LoggerStream MessageDispatcher::info(Message::Class mclass, const std::string& sender, FileInfo fileInfo) {
    return gMessageDispatcher.log(mclass, Message::Info, sender, fileInfo);
}

LoggerStream MessageDispatcher::info(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo) {
    return gMessageDispatcher.log(mclass, Message::Info, sender, fileInfo);
}

LoggerStream MessageDispatcher::warning(Message::Class mclass, const std::string& sender, FileInfo fileInfo) {
    return gMessageDispatcher.log(mclass, Message::Warning, sender, fileInfo);
}

LoggerStream MessageDispatcher::warning(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo) {
    return gMessageDispatcher.log(mclass, Message::Warning, sender, fileInfo);
}

LoggerStream MessageDispatcher::error(Message::Class mclass, const std::string& sender, FileInfo fileInfo) {
    return gMessageDispatcher.log(mclass, Message::Error, sender, fileInfo);
}

LoggerStream MessageDispatcher::error(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo) {
    return gMessageDispatcher.log(mclass, Message::Error, sender, fileInfo);
}

LoggerStream MessageDispatcher::fatal(Message::Class mclass, const std::string& sender, FileInfo fileInfo) {
    return gMessageDispatcher.log(mclass, Message::Fatal, sender, fileInfo);
}

LoggerStream MessageDispatcher::fatal(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo) {
    return gMessageDispatcher.log(mclass, Message::Fatal, sender, fileInfo);
}

}
}
}
}




