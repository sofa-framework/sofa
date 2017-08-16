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
#include <Communication/components/serverCommunication.h>

using sofa::core::RegisterObject ;

namespace sofa
{

namespace component
{

namespace communication
{

template <class DataTypes>
ServerCommunication<DataTypes>::ServerCommunication()
    : d_job(initData(&d_job, OptionsGroup(2,"receiver","sender"), "job", "If unspecified, the default value is receiver"))
    , d_adress(initData(&d_adress, (std::string)"127.0.0.1", "adress", "Scale for object display. (default=localhost)"))
    , d_port(initData(&d_port, (int)(6000), "port", "Port to listen (default=6000)"))
    , d_refreshRate(initData(&d_refreshRate, (double)(30.0), "refreshRate", "Refres rate aka frequency (default=30), only used by sender"))
    , d_nbDataField(initData(&d_nbDataField, (unsigned int)1, "nbData",
                             "Number of field 'data' the user want to send or receive.\n"
                             "Default value is 1."))
    , d_data(this, "data", "Data to send or receive.")
    , d_data_copy(this, "datac", "Data to send or receive.")
{
    d_data.resize(d_nbDataField.getValue());
    d_data_copy.resize(d_nbDataField.getValue());
    pthread_mutex_init(&mutex, NULL);

#if BENCHMARK
    gettimeofday(&t2, NULL);
    gettimeofday(&t1, NULL);
#endif
}

template <class DataTypes>
ServerCommunication<DataTypes>::~ServerCommunication()
{
    m_senderRunning = false;
    pthread_join(m_thread, NULL);
}

template <class DataTypes>
void * ServerCommunication<DataTypes>::thread_launcher(void *voidArgs)
{
    ServerCommunication *args = (ServerCommunication*)voidArgs;
    args->openCommunication();
    return NULL;
}

template <class DataTypes>
void ServerCommunication<DataTypes>::init()
{
    d_data.resize(d_nbDataField.getValue());
    d_data_copy.resize(d_nbDataField.getValue());
    f_listening = true;
    pthread_create(&m_thread, NULL, &ServerCommunication::thread_launcher, this);
}

template <class DataTypes>
void ServerCommunication<DataTypes>::handleEvent(Event* event)
{
    std::cout << "Event  " << std::endl;
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
#if BENCHMARK
        // Uncorrect results if frequency == 1hz, due to tv_usec precision
        gettimeofday(&t1, NULL);
        if(d_refreshRate.getValue() <= 1.0)
            std::cout << "Animation Loop frequency : " << fabs((t1.tv_sec - t2.tv_sec)) << " s or " << fabs(1.0 / ((t1.tv_sec - t2.tv_sec))) << " hz"<< std::endl;
        else
            std::cout << "Animation Loop frequency : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
        gettimeofday(&t2, NULL);
#endif
    }
}

template <class DataTypes>
void ServerCommunication<DataTypes>::openCommunication()
{
    if (d_job.getValueString().compare("receiver") == 0)
    {
        receiveData();
    }
    else if (d_job.getValueString().compare("sender") == 0)
    {
        while (m_senderRunning)
            sendData();
    }
}

} /// communication

} /// component

} /// sofa
