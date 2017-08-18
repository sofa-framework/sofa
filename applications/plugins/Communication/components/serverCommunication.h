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
#ifndef SOFA_SERVERCOMMUNICATION_H
#define SOFA_SERVERCOMMUNICATION_H

#include <Communication/config.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseData;

#include <sofa/helper/vectorData.h>
using sofa::helper::vectorData;
using sofa::helper::WriteAccessorVector;
using sofa::helper::WriteAccessor;
using sofa::helper::ReadAccessor;

#include <sofa/helper/OptionsGroup.h>
using sofa::helper::OptionsGroup;

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
using sofa::core::objectmodel::Event;


using sofa::defaulttype::Vec3d;
using sofa::defaulttype::Vec3f;
using sofa::defaulttype::Vec1d;
using sofa::defaulttype::Vec1f;
using sofa::defaulttype::Vec;
using sofa::helper::vector;

#include <sofa/defaulttype/RigidTypes.h>
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid3fTypes;


#include <pthread.h>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <mutex>
#include <cmath>

#define BENCHMARK false

namespace sofa
{

namespace component
{

namespace communication
{

template <class DataTypes>
class SOFA_COMMUNICATION_API ServerCommunication : public BaseObject
{

public:

    typedef BaseObject Inherited;
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(ServerCommunication, DataTypes), Inherited);

    ServerCommunication() ;
    virtual ~ServerCommunication() ;

    ////////////////////////// Inherited from BaseObject ////////////////////
    virtual void init() override;
    virtual void handleEvent(Event *) override;
    /////////////////////////////////////////////////////////////////////////

    //////////////////////////////// Inherited from Base /////////////////////////////////
    virtual std::string getTemplateName() const {return templateName(this);}
    static std::string templateName(const ServerCommunication<DataTypes>* = NULL);
    /////////////////////////////////////////////////////////////////////////////////

    Data<helper::OptionsGroup>  d_job;
    Data<std::string>           d_adress;
    Data<int>                   d_port;
    Data<double>                d_refreshRate;
    Data<unsigned int>          d_nbDataField;
    vectorData<DataTypes>       d_data;
    vectorData<DataTypes>       d_data_copy;
    pthread_t m_thread;
    bool m_running = true;
#if BENCHMARK
    timeval t1, t2;
#endif

protected:
    pthread_mutex_t mutex;

    virtual void openCommunication();
    virtual void closeCommunication();
    static void* thread_launcher(void*);
    virtual void sendData() =0;
    virtual void receiveData() =0;

};

} /// namespace communication
} /// namespace component
} /// namespace sofa

#endif // SOFA_SERVERCOMMUNICATION_H
