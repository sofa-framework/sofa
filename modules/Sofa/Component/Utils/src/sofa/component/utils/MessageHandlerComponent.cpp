/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
* in the logging.h file.
******************************************************************************/
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

//#include <sofa/helper/logging/LoggerMessageHandler.h>
//using sofa::helper::logging::LoggerMessageHandler ;

#include <sofa/helper/logging/ConsoleMessageHandler.h>
using sofa::helper::logging::ConsoleMessageHandler ;

#include <sofa/helper/logging/FileMessageHandler.h>
using sofa::helper::logging::FileMessageHandler ;

#include <sofa/helper/logging/ClangStyleMessageFormatter.h>
using sofa::helper::logging::ClangStyleMessageFormatter;

#include <sofa/helper/logging/DefaultStyleMessageFormatter.h>
using sofa::helper::logging::DefaultStyleMessageFormatter;

#include <sofa/helper/logging/ClangMessageHandler.h>
using sofa::helper::logging::ClangMessageHandler ;

#include <sofa/core/logging/RichConsoleStyleMessageFormatter.h>
using sofa::helper::logging::RichConsoleStyleMessageFormatter ;

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::MessageDispatcher;

#include <sofa/component/utils/MessageHandlerComponent.h>

using std::string;

namespace sofa
{
namespace component
{
namespace utils
{
namespace logging
{

MessageHandlerComponent::MessageHandlerComponent() :
    d_type(initData(&d_type, "handler", "Type of the message handler to use among "
                                        "[sofa, clang\
                                        //, log\
                                        , silent]. "))
{
    m_isValid = false ;
}

void MessageHandlerComponent::parse ( core::objectmodel::BaseObjectDescription* arg )
{
    BaseObject::parse(arg) ;

    const char* type=arg->getAttribute("handler") ;
    if(type==NULL){
        msg_info(this) << "The 'handler' attribute is missing. The default sofa style will be used. "
                          "To suppress this message you need to specify the 'handler' attribute. "
                          "eg: handler='silent' ";
        return ;
    }

    string stype(type) ;

    if(stype=="sofa"){
        MessageDispatcher::addHandler(new ConsoleMessageHandler()) ;
    }else if(stype=="clang"){
        MessageDispatcher::addHandler(new ClangMessageHandler()) ;
//    }else if(stype=="log"){
//        MessageDispatcher::addHandler(new LoggerMessageHandler()) ;
    }else if(stype=="rich"){
         MessageDispatcher::addHandler(new ConsoleMessageHandler(new RichConsoleStyleMessageFormatter())) ;
    }else if(stype=="silent"){
        MessageDispatcher::clearHandlers() ;
    }else{
        msg_info(this) << " the following handler '" << stype << "' is not a supported. "
                          "The default sofa style will be used. "
                          "To supress this message you need to specify a valid attribute "
                          "among [clang, log, silent, sofa]." ;
        return ;
    }

    msg_info(this) << " Adding a new message handler of type: " << stype ;

    m_isValid = true ;
}

SOFA_DECL_CLASS(MessageHandlerComponent)

int MessageHandlerComponentClass = RegisterObject("This object controls the way Sofa print's "
                                                  "info/warning/error/fatal messages. ")
        .add< MessageHandlerComponent >()
        ;



////////////////////////// FileMessageHandlerComponent ////////////////////////////////////
FileMessageHandlerComponent::FileMessageHandlerComponent() :
    d_filename(initData(&d_filename, "filename", "Name of the file into which the message will be saved in."))
{
    m_isValid = false ;
}

FileMessageHandlerComponent::~FileMessageHandlerComponent()
{
    MessageDispatcher::rmHandler(m_handler) ;

    delete m_handler ;
}

void FileMessageHandlerComponent::parse ( core::objectmodel::BaseObjectDescription* arg )
{
    BaseObject::parse(arg) ;

    const char* type=arg->getAttribute("filename") ;
    if(type==NULL){
        msg_warning(this) << "Name of the log file is missing. "
                             "To suppress this message you need to add the specify the filename eg:"
                             "  filename='nameOfTheLogFile.log' ";
        return ;
    }

    FileMessageHandler *handler = new FileMessageHandler(type, &DefaultStyleMessageFormatter::getInstance()) ;
    if(handler==NULL){
        msg_fatal(this) << "Unable to allocate memory. This is a fatal error. To fix"
                           "this you may free more RAM before running Sofa." ;
        return ;
    }

    if(!handler->isValid()){
        msg_warning(this) << " Unable to open the file named '" << type << "'. "
                             " Logs will not be written. ";
        return ;
    }

    msg_info(this) << " Logging messages into the file " << type << "." ;

    m_handler = handler;
    MessageDispatcher::addHandler(m_handler) ;

    m_isValid = true ;
}

SOFA_DECL_CLASS(FileMessageHandlerComponent)

int FileMessageHandlerComponentClass = RegisterObject("This component dump all the messages into"
                                                      "a file.")
        .add< FileMessageHandlerComponent >()
        ;
}
}
}
}


