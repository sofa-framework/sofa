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
#ifndef MESSAGEDISPATCHER_H
#define MESSAGEDISPATCHER_H

#include <sofa/helper/helper.h>
#include "Message.h"
#include <vector>
#include <sofa/helper/system/SofaOStream.h>

namespace sofa
{
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
                         const ComponentInfo::SPtr& sender, const FileInfo::SPtr& fileInfo) ;

            ~LoggerStream() ;

            template<class T>
            LoggerStream& operator<<(const T &x)
            {
                m_message << x;
                return *this;
            }

            Message getMessage()const { return m_message; }

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

        static LoggerStream info(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo = EmptyFileInfo) ;
        static LoggerStream deprecated(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo = EmptyFileInfo) ;
        static LoggerStream warning(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo = EmptyFileInfo) ;
        static LoggerStream error(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo = EmptyFileInfo) ;
        static LoggerStream fatal(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo = EmptyFileInfo) ;
        static LoggerStream advice(Message::Class mclass, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo = EmptyFileInfo) ;

        static const NullLoggerStream& null() { return NullLoggerStream::getInstance(); }
        static MessageDispatcher::LoggerStream log(Message::Class mclass, Message::Type type, const ComponentInfo::SPtr& cinfo, const FileInfo::SPtr& fileInfo = EmptyFileInfo);

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
