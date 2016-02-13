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
* Authors: Matthieu Nesme                                                     *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/
#ifndef TESTMESSAGEHANDLER_H
#define TESTMESSAGEHANDLER_H

#include <sofa/helper/logging/MessageHandler.h>
#include <sofa/helper/logging/Message.h>
#include "InitPlugin_test.h"
#include <gtest/gtest.h>

namespace sofa
{

namespace helper
{

namespace logging
{


/// each ERROR and FATAL message raises a gtest error
class SOFA_TestPlugin_API TestMessageHandler : public MessageHandler
{
public:

    /// raises a gtest error as soon as message is an error
    /// iff the handler is active (see setActive)
    virtual void process(Message &m)
    {
        if( active && m.type()>=Message::Error )
            ADD_FAILURE() << std::endl;
    }

    // singleton
    static TestMessageHandler& getInstance()
    {
        static TestMessageHandler s_instance;
        return s_instance;
    }

    /// raising a gtest error can be temporarily deactivated
    /// indeed, sometimes, testing that a error message is raised is mandatory
    /// and should not raise a gtest error
    static void setActive( bool a ) { getInstance().active = a; }

private:

    /// true by default
    bool active;

    // private default constructor for singleton
    TestMessageHandler() : active(true) {}
};


/// the TestMessageHandler is deactivated in the scope of a ScopedDeactivatedTestMessageHandler variable
struct SOFA_TestPlugin_API ScopedDeactivatedTestMessageHandler
{
    ScopedDeactivatedTestMessageHandler() { TestMessageHandler::setActive(false); }
    ~ScopedDeactivatedTestMessageHandler() { TestMessageHandler::setActive(true); }
};




} // logging
} // helper
} // sofa

#endif // TESTMESSAGEHANDLER_H

