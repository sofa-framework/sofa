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
#ifndef COUNTINGMESSAGEHANDLER_H
#define COUNTINGMESSAGEHANDLER_H

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
namespace countingmessagehandler
{

///
/// \brief The CountingMessageHandler class count the messages by types
///
/// This class is a MessageHandler that can be added to in a MessageDispatcher.
/// Once set the class will start counting the messages passing through
/// the MessageDispatcher according to their each Message::Type.
///
/// It is possible to query the number of a specific Message::Type
/// using the getMessageCountFor function.
///
/// User interested in having a singleton of this class should have a look
/// at \see MainCountingMessageHandler.
///
class SOFA_HELPER_API CountingMessageHandler : public MessageHandler
{
public:
    CountingMessageHandler() ;
    virtual ~CountingMessageHandler(){}

    void reset() ;
    int getMessageCountFor(const Message::Type& type) const ;

    /// Inherited from MessageHandler
    virtual void process(Message& m) ;
private:
    std::vector<int> m_countMatching ;
} ;

///
/// \brief The MainCountingMessageHandler class contains a singleton to CountingMessageHandler
/// and offer static version of CountingMessageHandler API
///
/// \see CountingMessageHandler
///
class SOFA_HELPER_API MainCountingMessageHandler
{
public:
    static CountingMessageHandler& getInstance() ;
    static void reset() ;
    static int getMessageCountFor(const Message::Type &type) ;
};

} /// namespace countingmessagehandler

/// Importing the per-file names into the 'library namespace'
using countingmessagehandler::MainCountingMessageHandler ;
using countingmessagehandler::CountingMessageHandler ;

} /// namespace logging
} /// namespace helper
} /// namespace sofa

#endif // COUNTINGMESSAGEHANDLER_H
