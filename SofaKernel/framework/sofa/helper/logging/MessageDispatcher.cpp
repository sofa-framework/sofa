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

#include <sofa/core/objectmodel/Base.h>

#include <sofa/helper/logging/RichConsoleStyleMessageFormatter.h>
using sofa::helper::logging::RichConsoleStyleMessageFormatter;

#include <mutex>
using std::lock_guard ;
using std::mutex;

namespace sofa
{

namespace helper
{

namespace logging
{

mutex s_dispatchermutex ;

static std::vector<MessageHandler*> setDefaultMessageHandler()
{
    /// This static function of the dispatcher has to be protected against concurent access
    /// This is done by a mutex and a scoped_guard as in Use the http://en.cppreference.com/w/cpp/thread/lock_guard
    //lock_guard<mutex> guard(s_dispatchermutex) ;

    std::vector<MessageHandler*> messageHandlers;
    static ConsoleMessageHandler s_consoleMessageHandler(new RichConsoleStyleMessageFormatter());
    messageHandlers.push_back(&s_consoleMessageHandler);
    return messageHandlers;
}


std::vector<MessageHandler*>& MessageDispatcher::getHandlers()
{
    /// This static function of the dispatcher has to be protected against concurent access
    /// This is done by a mutex and a scoped_guard as in Use the http://en.cppreference.com/w/cpp/thread/lock_guard
    //lock_guard<mutex> guard(s_dispatchermutex) ;

    static std::vector<MessageHandler*> s_handlers = setDefaultMessageHandler();
    return s_handlers;
}

int MessageDispatcher::addHandler(MessageHandler* o){
    /// This static function of the dispatcher has to be protected against concurent access
    /// This is done by a mutex and a scoped_guard as in Use the http://en.cppreference.com/w/cpp/thread/lock_guard
    //lock_guard<mutex> guard(s_dispatchermutex) ;

    std::vector<MessageHandler*>& handlers = getHandlers();
    if( std::find(handlers.begin(), handlers.end(), o) == handlers.end()){
        handlers.push_back(o) ;
        return (int)(handlers.size()-1);
    }
    return -1;
}

int MessageDispatcher::rmHandler(MessageHandler* o){
    /// This static function of the dispatcher has to be protected against concurent access
    /// This is done by a mutex and a scoped_guard as in Use the http://en.cppreference.com/w/cpp/thread/lock_guard
    //lock_guard<mutex> guard(s_dispatchermutex) ;

    std::vector<MessageHandler*>& handlers = getHandlers();
    handlers.erase(remove(handlers.begin(), handlers.end(), o), handlers.end());
    return (int)(handlers.size()-1);
}

void MessageDispatcher::clearHandlers(){
    /// This static function of the dispatcher has to be protected against concurent access
    /// This is done by a mutex and a scoped_guard as in Use the http://en.cppreference.com/w/cpp/thread/lock_guard
    //lock_guard<mutex> guard(s_dispatchermutex) ;

    std::vector<MessageHandler*>& handlers = getHandlers();
    handlers.clear() ;
}

void MessageDispatcher::process(sofa::helper::logging::Message& m){
    /// This static function of the dispatcher has to be protected against concurent access
    /// This is done by a mutex and a scoped_guard as in Use the http://en.cppreference.com/w/cpp/thread/lock_guard
    //lock_guard<mutex> guard(s_dispatchermutex) ;

    std::vector<MessageHandler*>& handlers = getHandlers();
    for( size_t i=0 ; i<handlers.size() ; i++ )
        handlers[i]->process(m) ;
}


MessageDispatcher::LoggerStream MessageDispatcher::log(Message::Class mclass, Message::Type type, const std::string& sender, const FileInfo::SPtr& fileInfo) {
    return MessageDispatcher::LoggerStream( mclass, type, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::log(Message::Class mclass, Message::Type type, const sofa::core::objectmodel::Base* sender, const  FileInfo::SPtr& fileInfo) {
    return MessageDispatcher::LoggerStream(mclass, type, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::info(Message::Class mclass, const std::string& sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Info, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::info(Message::Class mclass, const sofa::core::objectmodel::Base* sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Info, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::deprecated(Message::Class mclass, const std::string& sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Deprecated, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::deprecated(Message::Class mclass, const sofa::core::objectmodel::Base* sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Deprecated, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::warning(Message::Class mclass, const std::string& sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Warning, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::warning(Message::Class mclass, const sofa::core::objectmodel::Base* sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Warning, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::error(Message::Class mclass, const std::string& sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Error, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::error(Message::Class mclass, const sofa::core::objectmodel::Base* sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Error, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::fatal(Message::Class mclass, const std::string& sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Fatal, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::fatal(Message::Class mclass, const sofa::core::objectmodel::Base* sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Fatal, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::advice(Message::Class mclass, const std::string& sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Advice, sender, fileInfo);
}

MessageDispatcher::LoggerStream MessageDispatcher::advice(Message::Class mclass, const sofa::core::objectmodel::Base* sender, const FileInfo::SPtr& fileInfo) {
    return log(mclass, Message::Advice, sender, fileInfo);
}


MessageDispatcher::LoggerStream::LoggerStream(Message::Class mclass, Message::Type type,
             const sofa::core::objectmodel::Base* sender, const FileInfo::SPtr& fileInfo)
    : m_message( mclass
                 , type
                 , sender->getClassName()
                 // temporary, until Base object reference kept in the message itself ->
                 // (mattn) not sure it is a good idea, the Message could be kept after the Base is deleted
                 // TODO(dmarchal): by converting this to a string we are not able anymore to
                 // display rich information like the component name or location in the scene tree while these
                 // information are really usefull. I have to make small experiment to see what could be a nice
                 // approach without too much overhead.
                 , fileInfo
                 , ComponentInfo::SPtr(new ComponentInfo(sender->getName(), "")) )
{
}

MessageDispatcher::LoggerStream::~LoggerStream()
{
    if ( !m_message.empty() ) MessageDispatcher::process(m_message);
}


} // logging
} // helper
} // sofa


