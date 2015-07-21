/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*          Bruno Carrez                                                       *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/
#ifndef MESSAGEDISPATCHER_H
#define MESSAGEDISPATCHER_H

#include <sofa/helper/helper.h>
#include "Message.h"

namespace sofa
{

namespace helper
{

namespace messaging
{

class MessageHandler ;

class SOFA_HELPER_API MessageDispatcher
{
public:
    // handlers stuff is not static
    int addHandler(MessageHandler* o) ;
    int rmHandler(MessageHandler* o) ;
    void clearHandlers(bool deleteExistingOnes=true) ;

    // message IDs can stay static, they won't interfere if several dispatchers coexist
    static int getLastMessageId() ;
    static int getLastErrorId() ;
    static int getLastWarningId() ;
    static int getLastInfoId() ;

private:
    void process(sofa::helper::messaging::Message &m);

    friend Message& operator<<=(MessageDispatcher& d, Message& m) ;
};

Message& operator+=(MessageDispatcher& d, Message& m) ;
Message& operator<<=(MessageDispatcher& d, Message& m) ;


class Nop
{
public:
    static inline Nop& getAnInstance(){ return s_nop; }

    //todo(damien): This function breaks the semantic because it returns a Nop instead of a message...
    template<typename T> inline Nop& operator<<(const T /*v*/){ return *this; }

    static Nop s_nop;
};


} // messaging
} // helper
} // sofa

#endif // MESSAGEDISPATCHER_H
