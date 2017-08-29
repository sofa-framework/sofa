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
#ifndef SOFA_COMMUNICATIONSUBSCRIBER_H
#define SOFA_COMMUNICATIONSUBSCRIBER_H

#include <Communication/config.h>
#include <Communication/components/serverCommunication.h>

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/Link.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject;

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <regex>

namespace sofa
{

namespace component
{

namespace communication
{

class SOFA_COMMUNICATION_API CommunicationSubscriber: public BaseObject
{

public:

    typedef BaseObject Inherited;
    SOFA_CLASS(CommunicationSubscriber, Inherited);

    CommunicationSubscriber() ;
    virtual ~CommunicationSubscriber() ;

    unsigned int getArgumentSize();
    std::string getArgumentName(unsigned int);
    std::vector<std::string> getArgumentList();
    std::string getSubject();
    BaseObject* getSource();

    ////////////////////////// Inherited from BaseObject ////////////////////
    virtual void init() override;
    /////////////////////////////////////////////////////////////////////////

    Data<std::string>                                                                       d_subject;
    Data<std::string>                                                                       d_argumentsName;
    SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK>             l_source;
    SingleLink<CommunicationSubscriber,  ServerCommunication, BaseLink::FLAG_DOUBLELINK>    l_communication;

protected:
    std::vector<std::string> m_argumentsNameList;

};

} /// namespace communication
} /// namespace component
} /// namespace sofa

#endif // SOFA_COMMUNICATIONSUBSCRIBER_H
