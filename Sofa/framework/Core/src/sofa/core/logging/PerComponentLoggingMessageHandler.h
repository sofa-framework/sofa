/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef PERCOMPONENTLOGGINGMESSAGEHANDLER_H
#define PERCOMPONENTLOGGINGMESSAGEHANDLER_H
#include <sofa/core/config.h>
#include <sofa/helper/logging/MessageHandler.h>

namespace sofa::helper::logging
{

/// I use a per-file namespace so that I can employ the 'using' keywords without
/// fearing it will leack names into the global namespace.
/// When closing this namespace selected objects from this per-file namespace
/// are then imported into their parent namespace for ease of use.
namespace percomponentloggingmessagehandler
{
using std::vector ;

///
/// \brief The RoutingMessageHandler class saves a copy of the messages in a buffer.
///
/// This class is a MessageHandler that can be added to in a MessageDispatcher.
/// Once set the class can start copying the messages passing through
/// the MessageDispatcher in a buffer.
///
///
/// User interested in having a singleton of this class should have a look
/// at \see MainRoutingMessageHandler.
///
///
class SOFA_CORE_API PerComponentLoggingMessageHandler : public MessageHandler
{
public:
    PerComponentLoggingMessageHandler() ;
    ~PerComponentLoggingMessageHandler() override {}

    /// Inherited from MessageHandler
    void process(Message& m) override ;
} ;

///
/// \brief The MainPerComponentLoggingMessageHandler class contains a singleton to PerComponentLoggingMessageHandler
/// and offer static version of PerComponentLoggingMessageHandler API
///
/// \see PerComponentLoggingMessageHandler
///
class SOFA_CORE_API MainPerComponentLoggingMessageHandler
{
public:
    static PerComponentLoggingMessageHandler& getInstance() ;
};

} // namespace percomponentloggingmessagehandler

using percomponentloggingmessagehandler::PerComponentLoggingMessageHandler ;
using percomponentloggingmessagehandler::MainPerComponentLoggingMessageHandler ;
} // sofa::helper::logging

#endif // PERCOMPONENTLOGGINGMESSAGEHANDLER_H

