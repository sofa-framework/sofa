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

#include <sofa/helper/logging/Message.h>
#include <sofa/helper/logging/MessageFormatter.h>
#include <sofa/helper/logging/DefaultStyleMessageFormatter.h>
#include <sofa/helper/logging/FileMessageHandler.h>
#include <sofa/helper/logging/Messaging.h>


namespace sofa::helper::logging
{

FileMessageHandler::FileMessageHandler(const char* filename,MessageFormatter *formatter)
{
    m_formatter = (formatter==nullptr?&DefaultStyleMessageFormatter::getInstance():formatter);
    m_outFile.open(filename,std::ios_base::out | std::ios_base::trunc);
    if (!m_outFile.is_open())
        msg_error("FileMessageHandler") << "Could not open output log file: " << filename;
}

FileMessageHandler::~FileMessageHandler()
{
    if (m_outFile.is_open())
        m_outFile.close();
}

void FileMessageHandler::process(Message& m)
{
    if (m_outFile.is_open())
    {
        // TODO: formatter ?
        m_formatter->formatMessage(m,m_outFile);
        m_outFile << std::endl;
        m_outFile.flush() ;
    }
}

bool FileMessageHandler::isValid()
{
    return m_outFile.is_open();
}


}

