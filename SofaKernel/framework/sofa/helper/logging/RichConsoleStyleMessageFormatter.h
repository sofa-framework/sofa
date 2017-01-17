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
* Contributors:                                                               *
*    - damien.marchal@univ-lille1.fr                                          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/
#ifndef RICHCONSOLESTYLEMESSAGEFORMATTER_H
#define RICHCONSOLESTYLEMESSAGEFORMATTER_H
#include <sstream>
#include <string>
#include "Message.h"
#include "MessageFormatter.h"
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace logging
{

namespace richconsolestylemessageformater
{

///
/// \brief The RichConsoleStyleMessageFormatter class
///
///  The class implement a message formatter dedicated to console pretty printing on a console
///  Among other thing it feature formatting using a markdown like syntax:
///     - color rendering, 'italics' or *italics*
///     - alignement and wrapping for long message that are then much easier to read.
///     - automatic reading of the console number of column for prettier display.
///
///
class SOFA_HELPER_API RichConsoleStyleMessageFormatter : public MessageFormatter
{
public:
    virtual void formatMessage(const Message& m,std::ostream& out);

    RichConsoleStyleMessageFormatter();
};

/// Singleton based façade to RichConsoleStyleMessageFormatter.
class SOFA_HELPER_API MainRichConsoleStyleMessageFormatter
{
public:
    static void formatMessage(const Message& m,std::ostream& out)
    {
        static RichConsoleStyleMessageFormatter formatter ;
        formatter.formatMessage(m, out) ;
    }
};


} // richconsolestylemessageformater

using richconsolestylemessageformater::MainRichConsoleStyleMessageFormatter ;
using richconsolestylemessageformater::RichConsoleStyleMessageFormatter ;

} // logging
} // helper
} // sofa

#endif // DEFAULTSTYLEMESSAGEFORMATTER_H
