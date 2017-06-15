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
#ifndef ROUTINGMESSAGEHANDLER_H
#define ROUTINGMESSAGEHANDLER_H

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
namespace routingmessagehandler
{
using std::vector ;

typedef bool (*FilterFunction) (Message&) ;

///
/// \brief The RoutingMessageHandler class saves a copy of the messages in a buffer.
///
/// This class is a MessageHandler to implement complex routing rules.
///
/// User interested in having a singleton of this class should have a look
/// at \see MainRoutingMessageHandler.
///
///
class SOFA_HELPER_API RoutingMessageHandler : public MessageHandler
{
public:
    RoutingMessageHandler() ;
    virtual ~RoutingMessageHandler() {}

    /// All the message of the given class will be routed to this handler
    void setAFilter(FilterFunction,
                         MessageHandler* handler) ;

    /// Remove all the filter but don't delete the associated memory.
    void removeAllFilters() ;

    /// Inherited from MessageHandler
    virtual void process(Message& m) ;

private:
    std::vector<std::pair<FilterFunction, MessageHandler*> > m_filters;
} ;

///
/// \brief The MainRoutingMessageHandler class contains a singleton to RoutingMessageHandler
/// and offer static version of RoutingMessageHandler API
///
/// \see RoutingMessageHandler
///
class SOFA_HELPER_API MainRoutingMessageHandler
{
public:
    static RoutingMessageHandler& getInstance() ;

    /// All the message of the given class will be routed to this handler
    static void setAFilter(FilterFunction,
                                MessageHandler* handler) ;

    static void removeAllFilters() ;
};

} // loggingmessagehandler

using routingmessagehandler::RoutingMessageHandler ;
using routingmessagehandler::MainRoutingMessageHandler ;

} // logging
} // helper
} // sofa

#endif // TESTMESSAGEHANDLER_H

