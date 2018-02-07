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
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/
#ifndef EXCEPTIONMESSAGEHANDLER_H
#define EXCEPTIONMESSAGEHANDLER_H

#include "MessageHandler.h"
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace logging
{


/// Raise an exception each time a error message is processed.
/// Useful for automatic examples that only detects crashes!
class SOFA_HELPER_API ExceptionMessageHandler : public MessageHandler
{
public:

    /// the exception raised by an error Message
    struct SOFA_HELPER_API ErrorMessageException: public std::exception
    {
        ErrorMessageException(/*could take some parameters to get Message's infos*/){}

        virtual const char* what() const throw()
        {
            return "An error Message has been written.";
        }
    };



    virtual void process(Message &m);

};


} // logging
} // helper
} // sofa

#endif // EXCEPTIONMESSAGEHANDLER_H
