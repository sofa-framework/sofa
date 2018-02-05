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
/***********************************************************************
 * Not so Simple Messaging API
 *
 *
 * Eg:
 * If you add in your source code:
 *      msg_info(this) << "This is an error messaging from somewhere in my sofa"
 *                        " component saying that he cannot read a file with "
 *                        " name [" << filename << "]".
 *
 * This should print (using the Sofa pretty formatter) something like:
 *      [INFO] (name(component)): This is an error messae from somewhere in my sofa
 *                                component saying that he cannot read a file with
 *                                a name [/home/path/to/my/file.txt]
 *                        source: /home/path/to/mysourceode.cpp:354:0
 *
 * Eg:
 *  msg_info() << "This is the same as the version with 'this' ;
 *
 * Eg:
 *  msg_info("EmitterBasedOnString") << "An emitter's name" ;
 *
 *
 * The API is composed of two objects:
 *      - MessageHandler
 *      - Message
 *
 * a messaging contains:
 *      - a textual description (not formatted).
 *      - some specifier about the messaging type
 *      - a (possibly empty) backtrace
 *      - the source file and line from where the messaging was emitted.
 *
 * for any usefull purpose, a messaging needs to be send to
 * a MessageHandler either using an object or using the singleton/static
 * part of the API.
 *
 * The default messaging is adding the messaging to a list and gives
 * them a number. This number can then be used to retrieve the messaging
 * in the messaging list. Once added, the messaging also call the
 * virtual prettyPrintMessage() for effective displaying the messaging.
 *
 * It is possible, by inheriting and overriding the prettyPrintMessage
 * function to implement different policy.
 * Currently we are implementing
 *      SofaStyleHandler to format the messaging in a sofa way
 *                      (eg: [WARN] (object<classType>): messaging)
 *                      and print it on std::cout
 *      ClangStyleHandler: format the messaging in a clang style
 *                      to easy integration of the messaging in the IDE
 *                      (clicking on the messaging to go to the source code).
 *
 *  This is the place to go to implement other policy like:
 *      open a QTDialogBox, log the messaging into a file, report the
 *      messaging into the serr/sout of sofa.
 **********************************************************************/
#ifndef MESSAGING_H
#define MESSAGING_H

#include "MessageDispatcher.h"

#define msgendl "  \n"

#define nmsg_info(emitter)       sofa::helper::logging::MessageDispatcher::null()
#define nmsg_deprecated(emitter) sofa::helper::logging::MessageDispatcher::null()
#define nmsg_advice(emitter) sofa::helper::logging::MessageDispatcher::null()
#define nmsg_warning(emitter)    sofa::helper::logging::MessageDispatcher::null()
#define nmsg_error(emitter)      sofa::helper::logging::MessageDispatcher::null()
#define nmsg_fatal(emitter)      sofa::helper::logging::MessageDispatcher::null()

//todo(bruno&damien): the first quick&dirty version should be improved to preserve the semantic between
// the version compiled with WITH_SOFA_DEVTOOLS enabled and the other.
#ifdef SOFA_WITH_DEVTOOLS
#define olddmsg_info(emitter)       sofa::helper::logging::MessageDispatcher::info(sofa::helper::logging::Message::Dev, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define olddmsg_deprecated(emitter) sofa::helper::logging::MessageDispatcher::deprecated(sofa::helper::logging::Message::Dev, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define olddmsg_warning(emitter)    sofa::helper::logging::MessageDispatcher::warning(sofa::helper::logging::Message::Dev, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define olddmsg_error(emitter)      sofa::helper::logging::MessageDispatcher::error(sofa::helper::logging::Message::Dev, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define olddmsg_fatal(emitter)      sofa::helper::logging::MessageDispatcher::fatal(sofa::helper::logging::Message::Dev, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define olddmsg_advice(emitter)      sofa::helper::logging::MessageDispatcher::advice(sofa::helper::logging::Message::Dev, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#else
#define olddmsg_info(emitter)       nmsg_info(emitter)
#define olddmsg_deprecated(emitter) nmsg_deprecated(emitter)
#define olddmsg_warning(emitter)    nmsg_warning(emitter)
#define olddmsg_error(emitter)      nmsg_error(emitter)
#define olddmsg_fatal(emitter)      nmsg_fatal(emitter)
#define olddmsg_advice(emitter)     nmsg_advice(emitter)
#endif // NDEBUG

#define oldmsg_info(emitter)       sofa::helper::logging::MessageDispatcher::info(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define oldmsg_deprecated(emitter) sofa::helper::logging::MessageDispatcher::deprecated(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define oldmsg_warning(emitter)    sofa::helper::logging::MessageDispatcher::warning(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define oldmsg_error(emitter)      sofa::helper::logging::MessageDispatcher::error(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define oldmsg_fatal(emitter)      sofa::helper::logging::MessageDispatcher::fatal(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define oldmsg_advice(emitter)      sofa::helper::logging::MessageDispatcher::advice(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)

///#define msg_info_when(cond)          if(cond) sofa::helper::logging::MessageDispatcher::info(sofa::helper::logging::Message::Runtime, this, SOFA_FILE_INFO)
///#define msg_deprecated_when(cond)    if(cond) sofa::helper::logging::MessageDispatcher::deprecated(sofa::helper::logging::Message::Runtime, this, SOFA_FILE_INFO)
///#define msg_warning_when(cond)       if(cond) sofa::helper::logging::MessageDispatcher::warning(sofa::helper::logging::Message::Runtime, this, SOFA_FILE_INFO)
///#define msg_error_when(cond)         if(cond) sofa::helper::logging::MessageDispatcher::error(sofa::helper::logging::Message::Runtime, this, SOFA_FILE_INFO)
///#define msg_fatal_when(cond)         if(cond) sofa::helper::logging::MessageDispatcher::fatal(sofa::helper::logging::Message::Runtime, this, SOFA_FILE_INFO)
///#define msg_advice_when(cond)        if(cond) sofa::helper::logging::MessageDispatcher::advice(sofa::helper::logging::Message::Runtime, this, SOFA_FILE_INFO)

#define logmsg_info(emitter)       sofa::helper::logging::MessageDispatcher::info(sofa::helper::logging::Message::Log, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define logmsg_deprecated(emitter) sofa::helper::logging::MessageDispatcher::deprecated(sofa::helper::logging::Message::Log, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define logmsg_warning(emitter)    sofa::helper::logging::MessageDispatcher::warning(sofa::helper::logging::Message::Log, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define logmsg_error(emitter)      sofa::helper::logging::MessageDispatcher::error(sofa::helper::logging::Message::Log, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define logmsg_fatal(emitter)      sofa::helper::logging::MessageDispatcher::fatal(sofa::helper::logging::Message::Log, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)
#define logmsg_advice(emitter)      sofa::helper::logging::MessageDispatcher::advice(sofa::helper::logging::Message::Log, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO)

#define msg_info_withfile(emitter, file, line)       sofa::helper::logging::MessageDispatcher::info(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO_COPIED_FROM(file,line))
#define msg_deprecated_withfile(emitter, file, line) sofa::helper::logging::MessageDispatcher::deprecated(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO_COPIED_FROM(file,line))
#define msg_warning_withfile(emitter, file,line)    sofa::helper::logging::MessageDispatcher::warning(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO_COPIED_FROM(file,line))
#define msg_error_withfile(emitter, file,line)      sofa::helper::logging::MessageDispatcher::error(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO_COPIED_FROM(file,line))
#define msg_fatal_withfile(emitter, file,line)      sofa::helper::logging::MessageDispatcher::fatal(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO_COPIED_FROM(file,line))
#define msg_advice_withfile(emitter, file,line)      sofa::helper::logging::MessageDispatcher::advice(sofa::helper::logging::Message::Runtime, sofa::helper::logging::getComponentInfo(emitter), SOFA_FILE_INFO_COPIED_FROM(file,line))

#define FILEINFO(filename, line) sofa::helper::logging::FileInfo(filename, line)

#define FILEINFO(filename, line) sofa::helper::logging::FileInfo(filename, line)

/// THESE MACRO BEASTS ARE FOR AUTOMATIC DETECTION OF MACRO NO or ONE ARGUMENTS
#define TWO_FUNC_CHOOSER(_f1, _f2 ,...) _f2
#define TWO_FUNC_RECOMPOSER(argsWithParentheses) TWO_FUNC_CHOOSER argsWithParentheses

/// THE INFO BEAST
#define MSGINFO_1(x) if( sofa::helper::logging::notMuted(x) ) oldmsg_info(x)
#define MSGINFO_0()  if( sofa::helper::logging::notMuted(this) ) oldmsg_info(this)

#define MSGINFO_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, MSGINFO_1, ))
#define MSGINFO_NO_ARG_EXPANDER() ,MSGINFO_0
#define MSGINFO_CHOOSER(...) MSGINFO_CHOOSE_FROM_ARG_COUNT(MSGINFO_NO_ARG_EXPANDER __VA_ARGS__ ())

#define msg_info(...) MSGINFO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define msg_info_when(cond, ...) if(cond) msg_info(__VA_ARGS__)

/// THE WARNING BEAST
#define MSGWARNING_1(x) oldmsg_warning(x)
#define MSGWARNING_0()  oldmsg_warning(this)

#define MSGWARNING_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, MSGWARNING_1, ))
#define MSGWARNING_NO_ARG_EXPANDER() ,MSGWARNING_0
#define MSGWARNING_CHOOSER(...) MSGWARNING_CHOOSE_FROM_ARG_COUNT(MSGWARNING_NO_ARG_EXPANDER __VA_ARGS__ ())

#define msg_warning(...) MSGWARNING_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define msg_warning_when(cond, ...) if((cond)) msg_warning(__VA_ARGS__)


/// THE ERROR BEAST
#define MSGERROR_1(x) oldmsg_error(x)
#define MSGERROR_0()  oldmsg_error(this)

#define MSGERROR_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, MSGERROR_1, ))
#define MSGERROR_NO_ARG_EXPANDER() ,MSGERROR_0
#define MSGERROR_CHOOSER(...) MSGERROR_CHOOSE_FROM_ARG_COUNT(MSGERROR_NO_ARG_EXPANDER __VA_ARGS__ ())

#define msg_error(...) MSGERROR_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define msg_error_when(cond, ...) if((cond)) msg_error(__VA_ARGS__)


/// THE FATAL BEAST
#define MSGFATAL_1(x) oldmsg_fatal(x)
#define MSGFATAL_0()  oldmsg_fatal(this)

#define MSGFATAL_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, MSGFATAL_1, ))
#define MSGFATAL_NO_ARG_EXPANDER() ,MSGFATAL_0
#define MSGFATAL_CHOOSER(...) MSGFATAL_CHOOSE_FROM_ARG_COUNT(MSGFATAL_NO_ARG_EXPANDER __VA_ARGS__ ())

#define msg_fatal(...) MSGFATAL_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define msg_fatal_when(cond, ...) if((cond)) msg_fatal(__VA_ARGS__)


/// THE DEPRECATED BEAST
#define MSGDEPRECATED_1(x) oldmsg_deprecated(x)
#define MSGDEPRECATED_0()  oldmsg_deprecated(this)

#define MSGDEPRECATED_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, MSGDEPRECATED_1, ))
#define MSGDEPRECATED_NO_ARG_EXPANDER() ,MSGDEPRECATED_0
#define MSGDEPRECATED_CHOOSER(...) MSGDEPRECATED_CHOOSE_FROM_ARG_COUNT(MSGDEPRECATED_NO_ARG_EXPANDER __VA_ARGS__ ())

#define msg_deprecated(...) MSGDEPRECATED_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define msg_deprecated_when(cond, ...) if((cond)) msg_deprecated(__VA_ARGS__)


/// THE ADVICE BEAST
#define MSGADVICE_1(x) if( sofa::helper::logging::notMuted(x) ) oldmsg_advice(x)
#define MSGADVICE_0()  if( sofa::helper::logging::notMuted(this) ) oldmsg_advice(this)

#define MSGADVICE_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, MSGADVICE_1, ))
#define MSGADVICE_NO_ARG_EXPANDER() ,MSGADVICE_0
#define MSGADVICE_CHOOSER(...) MSGADVICE_CHOOSE_FROM_ARG_COUNT(MSGADVICE_NO_ARG_EXPANDER __VA_ARGS__ ())

#define msg_advice(...) MSGADVICE_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define msg_advice_when(cond, ...) if((cond)) msg_advice(__VA_ARGS__)


////////////////////////////////// DMSG
/// THESE MACRO BEASTS ARE FOR AUTOMATIC DETECTION OF MACRO NO or ONE ARGUMENTS

/// THE INFO BEAST
#define DMSGINFO_1(x) if( sofa::helper::logging::notMuted(x) ) olddmsg_info(x)
#define DMSGINFO_0()  if( sofa::helper::logging::notMuted(this) ) olddmsg_info(this)

#define DMSGINFO_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, DMSGINFO_1, ))
#define DMSGINFO_NO_ARG_EXPANDER() ,DMSGINFO_0
#define DMSGINFO_CHOOSER(...) DMSGINFO_CHOOSE_FROM_ARG_COUNT(DMSGINFO_NO_ARG_EXPANDER __VA_ARGS__ ())

#define dmsg_info(...) DMSGINFO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define dmsg_info_when(cond, ...) if(cond) dmsg_info(__VA_ARGS__)

/// THE WARNING BEAST
#define DMSGWARNING_1(x) olddmsg_warning(x)
#define DMSGWARNING_0()  olddmsg_warning(this)

#define DMSGWARNING_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, DMSGWARNING_1, ))
#define DMSGWARNING_NO_ARG_EXPANDER() ,DMSGWARNING_0
#define DMSGWARNING_CHOOSER(...) DMSGWARNING_CHOOSE_FROM_ARG_COUNT(DMSGWARNING_NO_ARG_EXPANDER __VA_ARGS__ ())

#define dmsg_warning(...) DMSGWARNING_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define dmsg_warning_when(cond, ...) if((cond)) dmsg_warning(__VA_ARGS__)


/// THE ERROR BEAST
#define DMSGERROR_1(x) olddmsg_error(x)
#define DMSGERROR_0()  olddmsg_error(this)

#define DMSGERROR_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, DMSGERROR_1, ))
#define DMSGERROR_NO_ARG_EXPANDER() ,DMSGERROR_0
#define DMSGERROR_CHOOSER(...) DMSGERROR_CHOOSE_FROM_ARG_COUNT(DMSGERROR_NO_ARG_EXPANDER __VA_ARGS__ ())

#define dmsg_error(...) DMSGERROR_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define dmsg_error_when(cond, ...) if((cond)) dmsg_error(__VA_ARGS__)


/// THE FATAL BEAST
#define DMSGFATAL_1(x) olddmsg_fatal(x)
#define DMSGFATAL_0()  olddmsg_fatal(this)

#define DMSGFATAL_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, DMSGFATAL_1, ))
#define DMSGFATAL_NO_ARG_EXPANDER() ,DMSGFATAL_0
#define DMSGFATAL_CHOOSER(...) DMSGFATAL_CHOOSE_FROM_ARG_COUNT(DMSGFATAL_NO_ARG_EXPANDER __VA_ARGS__ ())

#define dmsg_fatal(...) DMSGFATAL_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define dmsg_fatal_when(cond, ...) if((cond)) dmsg_fatal(__VA_ARGS__)


/// THE DEPRECATED BEAST
#define DMSGDEPRECATED_1(x) olddmsg_deprecated(x)
#define DMSGDEPRECATED_0()  olddmsg_deprecated(this)

#define DMSGDEPRECATED_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, DMSGDEPRECATED_1, ))
#define DMSGDEPRECATED_NO_ARG_EXPANDER() ,DMSGDEPRECATED_0
#define DMSGDEPRECATED_CHOOSER(...) DMSGDEPRECATED_CHOOSE_FROM_ARG_COUNT(DMSGDEPRECATED_NO_ARG_EXPANDER __VA_ARGS__ ())

#define dmsg_deprecated(...) DMSGDEPRECATED_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define dmsg_deprecated_when(cond, ...) if((cond)) dmsg_deprecated(__VA_ARGS__)


/// THE ADVICE BEAST
#define DMSGADVICE_1(x) if( sofa::helper::logging::notMuted(x) ) olddmsg_advice(x)
#define DMSGADVICE_0()  if( sofa::helper::logging::notMuted(this) ) olddmsg_advice(this)

#define DMSGADVICE_CHOOSE_FROM_ARG_COUNT(...) TWO_FUNC_RECOMPOSER((__VA_ARGS__, DMSGADVICE_1, ))
#define DMSGADVICE_NO_ARG_EXPANDER() ,DMSGADVICE_0
#define DMSGADVICE_CHOOSER(...) DMSGADVICE_CHOOSE_FROM_ARG_COUNT(DMSGADVICE_NO_ARG_EXPANDER __VA_ARGS__ ())

#define dmsg_advice(...) DMSGADVICE_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define dmsg_advice_when(cond, ...) if((cond)) dmsg_advice(__VA_ARGS__)

#define MSG_REGISTER_CLASS(classType, nameName) \
    namespace sofa {         \
    namespace helper {       \
    namespace logging {      \
        inline bool notMuted(const classType* ){ return true; }         \
        inline ComponentInfo::SPtr getComponentInfo(const classType* )  \
        {                                                               \
            return ComponentInfo::SPtr(new ComponentInfo(nameName)) ;   \
        }                                                               \
    } \
    } \
    }

#endif // MESSAGING_H
