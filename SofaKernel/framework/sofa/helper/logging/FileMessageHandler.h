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
#ifndef FILEMESSAGEHANDLER_H
#define FILEMESSAGEHANDLER_H

#include "MessageHandler.h"
#include <fstream>

namespace sofa
{

namespace helper
{

namespace logging
{

class MessageFormatter;

/// A message handle that saves the content message passing by in a file.
/// The formatting can be customize by passing a different MessageFormatter in the constructor.
/// Example of use:
///     MessageDispatcher::addHandler(new FileMessageHandler("myfile.log"));
class SOFA_HELPER_API FileMessageHandler : public MessageHandler
{
public:
    FileMessageHandler(const char* filename,MessageFormatter* formatter = 0);

    virtual ~FileMessageHandler();
    virtual void process(Message& m) ;

    bool isValid(); // is output file ok ?

private:
    std::ofstream       m_outFile;
    MessageFormatter    *m_formatter;
};


} // logging
} // helper
} // sofa

#endif // FILEMESSAGEHANDLER_H
