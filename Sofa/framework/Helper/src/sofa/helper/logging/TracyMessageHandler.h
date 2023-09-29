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
#pragma once

#include <sofa/helper/logging/MessageHandler.h>
#include <sofa/helper/config.h>

namespace sofa::helper::logging
{

class MessageFormatter;

/// Send the message to the Tracy profiler
class SOFA_HELPER_API TracyMessageHandler : public MessageHandler
{
public:
    /// Create a new ConsoleMessageHandler. By default the handler is using the
    /// DefaultStyleMessageFormatter object to format the message.
    TracyMessageHandler(MessageFormatter* formatter = nullptr);
    void process(Message &m) override ;
    void setMessageFormatter( MessageFormatter* formatter );

private:
    MessageFormatter *m_formatter { nullptr };

};

///
/// \brief The MainTracyMessageHandler class contains a singleton to TracyMessageHandler
/// and offer static version of TracyMessageHandler API
///
/// \see TracyMessageHandler
///
class SOFA_HELPER_API MainTracyMessageHandler
{
public:
    static TracyMessageHandler& getInstance() ;
};
} // namespace sofa::helper::logging

