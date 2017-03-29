/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef TESTMESSAGEHANDLER_H
#define TESTMESSAGEHANDLER_H

#include <sofa/helper/vector.h>
#include <sofa/helper/logging/CountingMessageHandler.h>
#include <sofa/helper/logging/LoggingMessageHandler.h>
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
/// Forward declaration of private classes.

class GtestMessageFrame;

/// Rise a gtest failure when a message of type Message:type is emitted.
/// Better use the macro:
///    EXPECT_MSG_NOEMIT(Error) as a more 'good looking' version of
///
/// sofa::helper::logging::MessageAsTestFailure failure(sofa::helper::logging::Message::Error, __FILE__, __LINE__);
class SOFA_TestPlugin_API MesssageAsTestFailure
{
public:
    MesssageAsTestFailure(Message::Type t,
                           const char* filename="unknown", int lineno=0) ;

    virtual ~MesssageAsTestFailure() ;

private:
    GtestMessageFrame* m_frame ;
};

/// Rise a gtest failure during the object destruction when the expected message have not
/// been received.
/// Better use the macro:
///    EXPECT_MSG_EMIT(Error) as a more 'good looking' version of
///
/// sofa::helper::logging::ExpectMessage failure(sofa::helper::logging::Message::Error, __FILE__, __LINE__);
class SOFA_TestPlugin_API ExpectMessage
{
public:
    ExpectMessage(Message::Type t,
                   const char* filename="unknown", int lineno=0) ;

    virtual ~ExpectMessage() ;

private:
    GtestMessageFrame* m_frame ;
};

/// Inherited from MessageHandler, this handler must be installed to have the testing subsystem
/// working. By default it is added in Sofa_test but if you are not inheriting from Sofa_test
/// you have to install it manually.
class SOFA_TestPlugin_API MainGtestMessageHandler
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
       sofa::helper::logging::MesssageAsTestFailure EXPECT_MSG_EVALUATOR(__hiddenscopevarA_, __LINE__) ( sofa::helper::logging::Message::a, __FILE__, __LINE__ ); \
       sofa::helper::logging::MesssageAsTestFailure EXPECT_MSG_EVALUATOR(__hiddenscopevarB_, __LINE__) ( sofa::helper::logging::Message::b, __FILE__, __LINE__ )

#define EXPECT_MSG_NOEMIT1(t)   sofa::helper::logging::MesssageAsTestFailure EXPECT_MSG_EVALUATOR(__hiddenscopevarT_, __LINE__)( sofa::helper::logging::Message::t, __FILE__, __LINE__ )
#define EXPECT_MSG_NOEMIT0

#define EXPECT_MSG_NOEMIT_CHOOSE_FROM_ARG_COUNT(...) FUNC_RECOMPOSER((__VA_ARGS__, EXPECT_MSG_NOEMIT2, EXPECT_MSG_NOEMIT1, ))
#define EXPECT_MSG_NOEMIT_NO_ARG_EXPANDER() ,,EXPECT_MSG_NOEMIT0
#define EXPECT_MSG_NOEMIT_CHOOSER(...) EXPECT_MSG_NOEMIT_CHOOSE_FROM_ARG_COUNT(EXPECT_MSG_NOEMIT_NO_ARG_EXPANDER __VA_ARGS__ ())

#define EXPECT_MSG_NOEMIT(...) EXPECT_MSG_NOEMIT_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

} // logging
} // helper

} // sofa

#endif // TESTMESSAGEHANDLER_H

