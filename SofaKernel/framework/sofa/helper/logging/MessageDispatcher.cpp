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

#include <sofa/helper/logging/DefaultStyleMessageFormatter.h>
using sofa::helper::logging::DefaultStyleMessageFormatter;

#include <mutex>
using std::lock_guard ;
using std::mutex;

namespace sofa
{

namespace helper
{

namespace logging
{

#if(SOFA_WITH_THREADING==1)
   #define MUTEX_IF_THREADING lock_guard<mutex> guard(getMainInstance()->getMutex()) ;
#else
   #define MUTEX_IF_THREADING
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////
/// Threading issues...
///     a mutex is serializing the access to the message API.
/// Memory management:
///     object are passed to the message info.
///     some of them are duplicated
///     other get a weak reference

std::vector<MessageHandler*> getDefaultMessageHandlers(){
    std::vector<MessageHandler*> messageHandlers;
    static ConsoleMessageHandler s_consoleMessageHandler(&DefaultStyleMessageFormatter::getInstance());
    messageHandlers.push_back(&s_consoleMessageHandler);
    return messageHandlers;
}

class MessageDispatcherImpl
{
public:
    mutex m_mutex ;
    mutex& getMutex()
    {
        return m_mutex;
    }

    std::vector<MessageHandler*> m_messageHandlers = getDefaultMessageHandlers();

    std::vector<MessageHandler*>& getHandlers()
    {
        return m_messageHandlers ;
    }

    int addHandler(MessageHandler* o)
    {
        if( std::find(m_messageHandlers.begin(), m_messageHandlers.end(), o) == m_messageHandlers.end())
        {
            m_messageHandlers.push_back(o) ;
            return (int)(m_messageHandlers.size()-1);
        }
        return -1;
    }

    int rmHandler(MessageHandler* o)
    {
        m_messageHandlers.erase(remove(m_messageHandlers.begin(), m_messageHandlers.end(), o), m_messageHandlers.end());
        return (int)(m_messageHandlers.size()-1);
    }

    void clearHandlers()
    {
        m_messageHandlers.clear() ;
    }

    void process(sofa::helper::logging::Message& m)
    {
        for( size_t i=0 ; i<m_messageHandlers.size() ; i++ ){
            m_messageHandlers[i]->process(m) ;
        }
    }
};


MessageDispatcherImpl* s_messagedispatcher = nullptr ;

MessageDispatcherImpl* getMainInstance(){
    if(s_messagedispatcher==nullptr){
        s_messagedispatcher = new MessageDispatcherImpl();
    }
    return s_messagedispatcher;
}

std::vector<MessageHandler*>& MessageDispatcher::getHandlers()
{
    MUTEX_IF_THREADING ;
    return getMainInstance()->getHandlers();
}

int MessageDispatcher::addHandler(MessageHandler* o){
    MUTEX_IF_THREADING ;
    return getMainInstance()->addHandler(o);
}

int MessageDispatcher::rmHandler(MessageHandler* o){
    MUTEX_IF_THREADING ;
    return getMainInstance()->rmHandler(o);
}

void MessageDispatcher::clearHandlers(){
    MUTEX_IF_THREADING ;
    getMainInstance()->clearHandlers();
}

void MessageDispatcher::process(sofa::helper::logging::Message& m){
    MUTEX_IF_THREADING ;
    getMainInstance()->process(m);
}


MessageDispatcher::LoggerStream MessageDispatcher::log(Message::Class mclass, Message::Type type,
                                                       const ComponentInfo::SPtr& cinfo, const  FileInfo::SPtr& fileInfo) {
    return MessageDispatcher::LoggerStream(mclass, type, cinfo, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::info(Message::Class mclass,
                                                        const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Info, cinfo, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::deprecated(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Deprecated, cinfo, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::warning(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Warning, cinfo, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::error(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Error, cinfo, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::fatal(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Fatal, cinfo, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::advice(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Advice, cinfo, fileInfo);
}

MessageDispatcher::LoggerStream::LoggerStream(Message::Class mclass, Message::Type type,
             const ComponentInfo::SPtr& cInfo, const FileInfo::SPtr& fileInfo)
    : m_message( mclass
                 , type
                 , cInfo
                 , fileInfo )
{
}

MessageDispatcher::LoggerStream::~LoggerStream()
{
    if ( !m_message.empty() ) MessageDispatcher::process(m_message);
}


} // logging
} // helper
} // sofa


