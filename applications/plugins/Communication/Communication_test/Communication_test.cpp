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
#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test ;

#include <sofa/defaulttype/Vec.h>
using sofa::defaulttype::Vec3f;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;
using sofa::core::ExecParams;

#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base;

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseData;

#include <sofa/helper/vectorData.h>
using sofa::helper::vectorData;
using sofa::helper::WriteAccessorVector;
using sofa::helper::WriteAccessor;
using sofa::helper::ReadAccessor;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

// COMMUNICATION PART
#include <Communication/components/serverCommunication.h>
#include <Communication/components/CommunicationSubscriber.h>
using sofa::component::communication::ServerCommunication;
using sofa::component::communication::CommunicationSubscriber;

// TIMEOUT
#include <iostream>
#include <future>
#include <thread>
#include <chrono>

namespace sofa
{
namespace component
{
namespace communication
{

using sofa::defaulttype::Vector3 ;

class Communication_test : public Sofa_test<>
{
    /// TODO
};

} // communication
} // component
} // sofa
