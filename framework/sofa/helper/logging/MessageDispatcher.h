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
*          Matthieu Nesme                                                     *
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
#include <vector>

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


// forward declaration
class MessageHandler;


/// static interface to manage the list of MessageHandlers
/// that process the Messages
class SOFA_HELPER_API MessageDispatcher
{

public:

        /// a utility interface to automatically process a Message
        /// at the end of scope of the LoggerStream variable
        /// (processed by all the handlers of the MessageDispatcher)
        class SOFA_HELPER_API LoggerStream
        {

        public:

            LoggerStream(const LoggerStream& s)
                : m_message( s.m_message )
            {}

            LoggerStream(Message::Class mclass, Message::Type type,
                         const std::string& sender, FileInfo fileInfo)
                : m_message( mclass, type, sender, fileInfo )
            {
            }

            LoggerStream(Message::Class mclass, Message::Type type,
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

        };

        /// @internal to be able to redirect Messages to nowhere
        class SOFA_HELPER_API NullLoggerStream
        {
        public:
            template<typename T> inline const NullLoggerStream& operator<<(const T& /*v*/) const { return *this; }
        private:
            NullLoggerStream(){}
            NullLoggerStream(const NullLoggerStream&);
            ~NullLoggerStream(){}
        protected:
            friend class MessageDispatcher;
            static const NullLoggerStream& getInstance(){ static const NullLoggerStream s_nop; return s_nop; }
        };




        static int addHandler(MessageHandler* o) ; ///< to add a MessageHandler
        static int rmHandler(MessageHandler* o) ; ///< to remove a MessageHandler
        static void clearHandlers() ; ///< to remove every MessageHandlers
        static std::vector<MessageHandler*>& getHandlers(); ///< the list of MessageHandlers

        static LoggerStream info(Message::Class mclass, const std::string& sender = "", FileInfo fileInfo = FileInfo()) ;
        static LoggerStream info(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) ;
        static LoggerStream deprecated(Message::Class mclass, const std::string& sender = "", FileInfo fileInfo = FileInfo()) ;
        static LoggerStream deprecated(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) ;
        static LoggerStream warning(Message::Class mclass, const std::string& sender = "", FileInfo fileInfo = FileInfo()) ;
        static LoggerStream warning(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) ;
        static LoggerStream error(Message::Class mclass, const std::string& sender = "", FileInfo fileInfo = FileInfo()) ;
        static LoggerStream error(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) ;
        static LoggerStream fatal(Message::Class mclass, const std::string& sender = "", FileInfo fileInfo = FileInfo()) ;
        static LoggerStream fatal(Message::Class mclass, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo()) ;
        static const NullLoggerStream& null() { return NullLoggerStream::getInstance(); }
        static MessageDispatcher::LoggerStream log(Message::Class mclass, Message::Type type, const std::string& sender = "", FileInfo fileInfo = FileInfo());
        static MessageDispatcher::LoggerStream log(Message::Class mclass, Message::Type type, const sofa::core::objectmodel::Base* sender, FileInfo fileInfo = FileInfo());

        /// Process the Message by all the Message handlers.
        /// Called in the destructor of LoggerStream
        /// and can be called manually on a hand-made (possibly predefined) Message
        static void process(sofa::helper::logging::Message& m);

    private:

        // static interface
        MessageDispatcher();
        MessageDispatcher(const MessageDispatcher&);
        ~MessageDispatcher();

};


} // logging
} // helper
} // sofa

#endif // MESSAGEDISPATCHER_H
