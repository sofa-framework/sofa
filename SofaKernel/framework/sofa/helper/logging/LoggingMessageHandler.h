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
* in the messaging.h file.
******************************************************************************/
#ifndef LOGGINMESSAGEHANDLER_H
#define LOGGINMESSAGEHANDLER_H

#include <sofa/helper/logging/MessageHandler.h>
#include <sofa/helper/logging/Message.h>
#include <vector>

namespace sofa
{
namespace helper
{
namespace logging
{

/// I use a per-file namespace so that I can employ the 'using' keywords without
/// fearing it will leack names into the global namespace.
/// When closing this namespace selected objects from this per-file namespace
/// are then imported into their parent namespace for ease of use.
namespace loggingmessagehandler
{
using std::vector ;

///
/// \brief The LoggingMessageHandler class saves a copy of the messages in a buffer.
///
/// This class is a MessageHandler that can be added to in a MessageDispatcher.
/// Once set the class can start copying the messages passing through
/// the MessageDispatcher in a buffer.
///
/// Dedicated function allows to activate/deactive the saving of the messages as well
/// as to clear the content of the buffer. The class keeps tracks of its
/// number of activation/deactivation. Message logging is activated as long
/// as there is not a paired number of activation/deactivation.
///
/// User interested in having a singleton of this class should have a look
/// at \see MainLoggingMessageHandler.
///
/// You can also see the \see LogMessage for example of use
///
class SOFA_HELPER_API LoggingMessageHandler : public MessageHandler
{
public:
    LoggingMessageHandler() ;
    virtual ~LoggingMessageHandler() {}

    //TODO(dmarchal): there is several defect in the design of this class:
    //   - no maximum buffer size (we should implement a circular buffer approach)
    //   - but if we do the index returned by activate and deactivate will required to be
    //     invalidaded maybe via a kind of listening pattern.
    //   - counting the number of activate/deactivate is a weak form of tracking the
    //     client of this class.
    void reset() ;
    int activate() ;
    int deactivate() ;
    const vector<Message>& getMessages() const ;

    /// Inherited from MessageHandler
    virtual void process(Message& m) ;

private:
    int             m_activationCount    {0};
    vector<Message> m_messages ;
} ;

///
/// \brief The MainLoggingMessageHandler class contains a singleton to CountingMessageHandler
/// and offer static version of CountingMessageHandler API
///
/// \see LoggingMessageHandler
///
class SOFA_HELPER_API MainLoggingMessageHandler
{
public:
    static LoggingMessageHandler& getInstance() ;
    static int activate() ;
    static int deactivate() ;
    static const vector<Message>& getMessages() ;
};

class SOFA_HELPER_API LogMessage
{
public:
    LogMessage() {
        m_firstMessage = MainLoggingMessageHandler::activate() ;
    }

    ~LogMessage() {}

    //TODO(dmarchal): Thread safetines issue !!
    std::vector<Message>::const_iterator begin()
    {
        const std::vector<Message>& messages = MainLoggingMessageHandler::getMessages() ;

        assert(m_firstMessage<=messages.size()) ;
        return messages.begin()+m_firstMessage ;
    }

    std::vector<Message>::const_iterator end()
    {
        const std::vector<Message>& messages = MainLoggingMessageHandler::getMessages() ;
        return messages.end() ;
    }

    int size()
    {
        const std::vector<Message>& messages = MainLoggingMessageHandler::getMessages() ;

        return messages.end()-(messages.begin()+m_firstMessage);
    }

private:
    unsigned int m_firstMessage      {0} ;
};


} // loggingmessagehandler

using loggingmessagehandler::LoggingMessageHandler ;
using loggingmessagehandler::MainLoggingMessageHandler ;
using loggingmessagehandler::LogMessage ;

} // logging
} // helper
} // sofa

#endif // TESTMESSAGEHANDLER_H

