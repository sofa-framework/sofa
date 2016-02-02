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
#ifndef MESSAGEDISPATCHER_H
#define MESSAGEDISPATCHER_H

#include <sofa/helper/helper.h>
#include "Message.h"

namespace sofa
{

namespace core
{
namespace objectmodel
{
class Base; // forward declaration
} // namespace objectmodel
} // namespace core


namespace helper
{

namespace logging
{

class SOFA_HELPER_API Nop
{
public:
    static inline Nop& getAnInstance(){ return s_nop; }

    //todo(damien): This function breaks the semantic because it returns a Nop instead of a message...
    template<typename T> inline Nop& operator<<(const T /*v*/){ return *this; }

    static Nop s_nop;
};


namespace unique{
    class MessageDispatcher ;
}


class MessageHandler ;
class ConsoleMessageHandler;
class MessageDispatcherImpl ;

class SOFA_HELPER_API LoggerStream
{

    friend class MessageDispatcherImpl;
    friend class MessageDispatcher;


public:
    // We need the copy constructor to return LoggerStreams by value, but it
    // should be optimized away any decent compiler.

    LoggerStream(const LoggerStream& s)
        : m_message( s.m_message )
        , m_dispatcher(s.m_dispatcher) {}

    LoggerStream(MessageDispatcherImpl& dispatcher, Message::Class mclass, Message::Type type,
                 const std::string& sender, FileInfo fileInfo)
        : m_message( mclass, type, sender, fileInfo )
        , m_dispatcher(dispatcher)
    {
    }

    LoggerStream(MessageDispatcherImpl& dispatcher, Message::Class mclass, Message::Type type,
                 const sofa::core::objectmodel::Base* sender, FileInfo fileInfo);

    ~LoggerStream() ;

    template<class T>
    LoggerStream& operator<<(const T &x)
    {
        m_message << x;
        return *this;
    }


private:
    Message m_message;
    MessageDispatcherImpl& m_dispatcher;
};


////////////////////// THE STATIC FACADE ///////////////////////////////////
namespace unique{
    class SOFA_HELPER_API MessageDispatcher{

    public:
        static int addHandler(MessageHandler* o) ;
        static int rmHandler(MessageHandler* o) ;
        static void clearHandlers() ;
        static ConsoleMessageHandler* getDefaultMessageHandler();

//        static int getLastMessageId() ;
//        static int getLastErrorId() ;
//        static int getLastWarningId() ;
//        static int getLastInfoId() ;

        static LoggerStream info(Message::Class mclass, const std::string& sender = "", FileInfo fileInfo = FileInfo()) ;
        static LoggerStream info(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) ;
        static LoggerStream warning(Message::Class mclass, const std::string& sender = "", FileInfo fileInfo = FileInfo()) ;
        static LoggerStream warning(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) ;
        static LoggerStream error(Message::Class mclass, const std::string& sender = "", FileInfo fileInfo = FileInfo()) ;
        static LoggerStream error(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) ;
        static LoggerStream fatal(Message::Class mclass, const std::string& sender = "", FileInfo fileInfo = FileInfo()) ;
        static LoggerStream fatal(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) ;
    };
}

} // logging
} // helper
} // sofa

#endif // MESSAGEDISPATCHER_H
