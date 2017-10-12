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
#include <Communication/components/CommunicationSubscriber.h>

namespace sofa
{

namespace component
{

namespace communication
{

CommunicationSubscriber::CommunicationSubscriber()
    : d_subject(initData(&d_subject, (std::string)"", "subject", "ServerCommunication will parse and analyse only subscribed subject (default="")"))
    , d_argumentsName(initData(&d_argumentsName, (std::string)"x", "arguments", "Arguments name will be used to store data(default=x)"))
    , l_communication(initLink("communication","Required link, subscriber will define which kind of subject the communication have to listen"))
    , l_source(initLink("source","Required link, subscriber will define which BaseObject will be used for read/write data"))
{
}

CommunicationSubscriber::~CommunicationSubscriber()
{
}

void CommunicationSubscriber::init()
{
    // TODO attention python tableau
    std::regex regex{R"([\s]+)"};
    std::string string = d_argumentsName.getValueString();
    std::sregex_token_iterator it{string.begin(), string.end(), regex, -1};
    m_argumentsNameList = std::vector<std::string>{it, {}};
    if (l_communication)
        l_communication->addSubscriber(this);
    else
        msg_error() << "Communication link has not been set for " << this->getName() << ", this subscriber is useless until a link has been provide";
}

std::vector<std::string> CommunicationSubscriber::getArgumentList()
{
    return m_argumentsNameList;
}

unsigned int CommunicationSubscriber::getArgumentSize()
{
    return m_argumentsNameList.size();
}

std::string CommunicationSubscriber::getArgumentName(unsigned int index)
{
     return m_argumentsNameList.at(index);
}

std::string CommunicationSubscriber::getSubject()
{
    return d_subject.getValueString();
}

SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK> CommunicationSubscriber::getSource()
{
    return l_source;
}

} /// communication

} /// component

} /// sofa
