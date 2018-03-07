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
/******************************************************************************
 * Contributors:
 *    - damien.marchal@univ-lille1.fr
 ******************************************************************************/
#ifndef TESTMESSAGEHANDLER_H
#define TESTMESSAGEHANDLER_H

#include <sofa/helper/vector.h>
#include <sofa/helper/logging/CountingMessageHandler.h>
#include <sofa/helper/logging/LoggingMessageHandler.h>
#include <sofa/helper/logging/MessageHandler.h>
#include <sofa/helper/logging/Message.h>
#include <gtest/gtest.h>


////////////////////////////////////////////////////////////////////////////////////////////////////
/// This file is providing an API to combine gtest and msg_* API.
/// The underlying idea is to be able to test sofa's message.
///
/// The API is composed of two macro:
///    - EXPECT_MSG_EMIT(...);
///    - EXPECT_MSG_NOEMIT(...);
///
/// The first one generates a gtest failure when a message of a given type is not emitted. So
/// You need to use it express that the good behavior from the object is to rise a message.
///
/// The second one generates a gtest failure when a message of a given type is emitted.
///
/// Examples of use:
///     for(BaseLoader* b : objectlist)
///     {
///         EXPECT_MESSAGE_NOEMIT(Warning);
///         EXPECT_MESSAGE_EMIT(Error);
///         b->load("Invalid file");
///     }
///
/// To work the API need to a specific handler to be install in the messaging system.
/// This means that we need to install the message handler using. This is not done automatically
/// To not add something with a linear time complexity in the process.
///
/// Example of installation:
///     MessageDispatcher::addHandler( MainGtestMessageHandler::getInstance() ) ;
///
/// NB: This is done automatically if you are inhering from Sofa_test.
///
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace sofa
{

namespace helper
{

namespace logging
{
/// Forward declaration of private classes.
class GtestMessageFrame;

/// Rise a gtest failure when a message of type Message:type is emitted.
/// Better use the macro:
///    EXPECT_MSG_NOEMIT(Error) as a more 'good looking' version of
///
/// sofa::helper::logging::MessageAsTestFailure failure(sofa::helper::logging::Message::Error, __FILE__, __LINE__);
class SOFA_HELPER_API MessageAsTestFailure
{
public:
    MessageAsTestFailure(Message::Type t,
                           const char* filename="unknown", int lineno=0) ;

    virtual ~MessageAsTestFailure() ;

private:
    GtestMessageFrame* m_frame ;
};

/// Rise a gtest failure during the object destruction when the expected message have not
/// been received.
/// Better use the macro:
///    EXPECT_MSG_EMIT(Error) as a more 'good looking' version of
///
/// sofa::helper::logging::ExpectMessage failure(sofa::helper::logging::Message::Error, __FILE__, __LINE__);
class SOFA_HELPER_API ExpectMessage
{
public:
    ExpectMessage(Message::Type t,
                   const char* filename="unknown", int lineno=0) ;

    virtual ~ExpectMessage() ;

private:
    GtestMessageFrame* m_frame ;
};


/// Locally hide the fact that a message is expected.
/// Better use the macro:
///    IGNORE_MSG(Error) as a more 'good looking' version of
///
/// sofa::helper::logging::IgnoreMessage ignore(sofa::helper::logging::Message::Error);
class SOFA_HELPER_API IgnoreMessage
{
public:
    IgnoreMessage(Message::Type t) ;
    virtual ~IgnoreMessage() ;

private:
    GtestMessageFrame* m_frame ;
};

/// Inherited from MessageHandler, this handler must be installed to have the testing subsystem
/// working. By default it is added in Sofa_test but if you are not inheriting from Sofa_test
/// you have to install it manually.
class SOFA_HELPER_API MainGtestMessageHandler
{
public:
    static MessageHandler* getInstance() ;
};

////////////////////////////// MACROS TO EASE THE USERS ////////////////////////////////////////////
/// Using the raw objects is very verbose as it lead to code like this:
///    ExpectMessage error(Message::Error, __FILE__, __LINE__) ;
///
/// which obfuscate the readability of the source code and force developper to import
/// ExpectMessage and Message into the current namespace.
///
/// For that reason we are provide a set of macro that can be used in the following way:
///    EXPECT_MSG_EMIT(Error) or EXPECT_MSG_NOEMIT instead of calling the ExpectMessage
///    and MessageAsTestFailure objcet.
///
/// The macros are mimicking the way gtest macros are working... and thus as any macro are really
/// hard to understand and error prone.
////////////////////////////////////////////////////////////////////////////////////////////////////
#define IGNORE_MSG(t) sofa::helper::logging::IgnoreMessage EXPECT_MSG_EVALUATOR(__hiddenscopevarI_, __LINE__) ( sofa::helper::logging::Message::t)

///From http://en.cppreference.com/w/cpp/preprocessor/replace
#define EXPECT_MSG_PASTER(x,y) x ## _ ## y
#define EXPECT_MSG_EVALUATOR(x,y)  EXPECT_MSG_PASTER(x,y)

///TAKE FROM http://stackoverflow.com/questions/3046889/optional-parameters-with-c-macros
#define FUNC_CHOOSER(_f1, _f2, _f3, ...) _f3
#define FUNC_RECOMPOSER(argsWithParentheses) FUNC_CHOOSER argsWithParentheses

#define EXPECT_MSG_EMIT2(a,b) \
    sofa::helper::logging::ExpectMessage EXPECT_MSG_EVALUATOR(__hiddenscopevarA_, __LINE__) ( sofa::helper::logging::Message::a, __FILE__, __LINE__ ); \
    sofa::helper::logging::ExpectMessage EXPECT_MSG_EVALUATOR(__hiddenscopevarB_, __LINE__) ( sofa::helper::logging::Message::b, __FILE__, __LINE__ )

#define EXPECT_MSG_EMIT1(t)   sofa::helper::logging::ExpectMessage EXPECT_MSG_EVALUATOR(__hiddenscopevarT_, __LINE__) ( sofa::helper::logging::Message::t, __FILE__, __LINE__ )
#define EXPECT_MSG_EMIT0

#define EXPECT_MSG_EMIT_CHOOSE_FROM_ARG_COUNT(...) FUNC_RECOMPOSER((__VA_ARGS__, EXPECT_MSG_EMIT2, EXPECT_MSG_EMIT1, ))
#define EXPECT_MSG_EMIT_NO_ARG_EXPANDER() ,,EXPECT_MSG_EMIT0
#define EXPECT_MSG_EMIT_CHOOSER(...) EXPECT_MSG_EMIT_CHOOSE_FROM_ARG_COUNT(EXPECT_MSG_EMIT_NO_ARG_EXPANDER __VA_ARGS__ ())

#define EXPECT_MSG_EMIT(...) EXPECT_MSG_EMIT_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

#define EXPECT_MSG_NOEMIT2(a,b) \
       sofa::helper::logging::MessageAsTestFailure EXPECT_MSG_EVALUATOR(__hiddenscopevarA_, __LINE__) ( sofa::helper::logging::Message::a, __FILE__, __LINE__ ); \
       sofa::helper::logging::MessageAsTestFailure EXPECT_MSG_EVALUATOR(__hiddenscopevarB_, __LINE__) ( sofa::helper::logging::Message::b, __FILE__, __LINE__ )

#define EXPECT_MSG_NOEMIT1(t)   sofa::helper::logging::MessageAsTestFailure EXPECT_MSG_EVALUATOR(__hiddenscopevarT_, __LINE__)( sofa::helper::logging::Message::t, __FILE__, __LINE__ )
#define EXPECT_MSG_NOEMIT0

#define EXPECT_MSG_NOEMIT_CHOOSE_FROM_ARG_COUNT(...) FUNC_RECOMPOSER((__VA_ARGS__, EXPECT_MSG_NOEMIT2, EXPECT_MSG_NOEMIT1, ))
#define EXPECT_MSG_NOEMIT_NO_ARG_EXPANDER() ,,EXPECT_MSG_NOEMIT0
#define EXPECT_MSG_NOEMIT_CHOOSER(...) EXPECT_MSG_NOEMIT_CHOOSE_FROM_ARG_COUNT(EXPECT_MSG_NOEMIT_NO_ARG_EXPANDER __VA_ARGS__ ())

#define EXPECT_MSG_NOEMIT(...) EXPECT_MSG_NOEMIT_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

} // logging
} // helper

} // sofa

#endif // TESTMESSAGEHANDLER_H

