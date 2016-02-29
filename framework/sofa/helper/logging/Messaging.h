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
* Authors: Damien Marchal                                                     *
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

#define nmsg_info(emitter)       sofa::helper::logging::MessageDispatcher::null()
#define nmsg_deprecated(emitter) sofa::helper::logging::MessageDispatcher::null()
#define nmsg_warning(emitter)    sofa::helper::logging::MessageDispatcher::null()
#define nmsg_error(emitter)      sofa::helper::logging::MessageDispatcher::null()
#define nmsg_fatal(emitter)      sofa::helper::logging::MessageDispatcher::null()

//todo(bruno&damien): the first quick&dirty version should be improved to preserve the semantic between
// the version compiled with WITH_SOFA_DEVTOOLS enabled and the other.
#ifdef WITH_SOFA_DEVTOOLS
#define dmsg_info(emitter)       sofa::helper::logging::MessageDispatcher::info(sofa::helper::logging::Message::Dev, emitter, SOFA_FILE_INFO)
#define dmsg_deprecated(emitter) sofa::helper::logging::MessageDispatcher::deprecated(sofa::helper::logging::Message::Dev, emitter, SOFA_FILE_INFO)
#define dmsg_warning(emitter)    sofa::helper::logging::MessageDispatcher::warning(sofa::helper::logging::Message::Dev, emitter, SOFA_FILE_INFO)
#define dmsg_error(emitter)      sofa::helper::logging::MessageDispatcher::error(sofa::helper::logging::Message::Dev, emitter, SOFA_FILE_INFO)
#define dmsg_fatal(emitter)      sofa::helper::logging::MessageDispatcher::fatal(sofa::helper::logging::Message::Dev, emitter, SOFA_FILE_INFO)
#else
#define dmsg_info(emitter)       nmsg_info(emitter)
#define dmsg_deprecated(emitter) nmsg_deprecated(emitter)
#define dmsg_warning(emitter)    nmsg_warning(emitter)
#define dmsg_error(emitter)      nmsg_error(emitter)
#define dmsg_fatal(emitter)      nmsg_fatal(emitter)
#endif // NDEBUG

#define msg_info(emitter)       sofa::helper::logging::MessageDispatcher::info(sofa::helper::logging::Message::Runtime, emitter, SOFA_FILE_INFO)
#define msg_deprecated(emitter) sofa::helper::logging::MessageDispatcher::deprecated(sofa::helper::logging::Message::Runtime, emitter, SOFA_FILE_INFO)
#define msg_warning(emitter)    sofa::helper::logging::MessageDispatcher::warning(sofa::helper::logging::Message::Runtime, emitter, SOFA_FILE_INFO)
#define msg_error(emitter)      sofa::helper::logging::MessageDispatcher::error(sofa::helper::logging::Message::Runtime, emitter, SOFA_FILE_INFO)
#define msg_fatal(emitter)      sofa::helper::logging::MessageDispatcher::fatal(sofa::helper::logging::Message::Runtime, emitter, SOFA_FILE_INFO)

#endif // MESSAGING_H
