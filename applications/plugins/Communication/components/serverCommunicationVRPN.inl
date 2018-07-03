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
#include "serverCommunicationVRPN.h"
#include "Communication/components/CommunicationSubscriber.h"

#include <vrpn_Text.h>
#include <vrpn_Connection.h>

namespace sofa
{

namespace component
{

namespace communication
{

ServerCommunicationVRPN::ServerCommunicationVRPN()
    : Inherited()
{
}

ServerCommunicationVRPN::~ServerCommunicationVRPN()
{
    this->m_running = false;

    if(isVerbose())
        msg_info(this) << "waiting for timeout";

    Inherited::closeCommunication();
}

ServerCommunicationVRPN::VRPNDataFactory* ServerCommunicationVRPN::getFactoryInstance()
{
    static VRPNDataFactory* s_localfactory = nullptr ;
    if(s_localfactory==nullptr)
        s_localfactory = new ServerCommunicationVRPN::VRPNDataFactory() ;
    return s_localfactory ;
}

void ServerCommunicationVRPN::initTypeFactory()
{
    getFactoryInstance()->registerCreator("VRPNfloat", new DataCreator<float>());
    getFactoryInstance()->registerCreator("VRPNdouble", new DataCreator<double>());
    getFactoryInstance()->registerCreator("int", new DataCreator<int>());
    getFactoryInstance()->registerCreator("VRPNstring", new DataCreator<std::string>());

    getFactoryInstance()->registerCreator("matrixVRPNfloat", new DataCreator<FullMatrix<float>>());
    getFactoryInstance()->registerCreator("matrixVRPNdouble", new DataCreator<FullMatrix<double>>());
    getFactoryInstance()->registerCreator("matrixVRPNint", new DataCreator<FullMatrix<int>>());
}

std::string ServerCommunicationVRPN::defaultDataType()
{
    return "VRPNstring";
}

/******************************************************************************
*                                                                             *
* SEND PART                                                                   *
*                                                                             *
******************************************************************************/

void ServerCommunicationVRPN::sendData()
{
    std::string address = d_address.getValueString();
    m_connection = vrpn_create_server_connection();
    std::map<std::string, CommunicationSubscriber*> subscribersMap = getSubscribers();
    if (subscribersMap.size() == 0)
    {
        if (isVerbose())
            msg_info(this) << "Server Communication VRPN does not have Subscriber";
        return;
    }
    for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
    {
        CommunicationSubscriber* subscriber = it->second;
        std::string strTestText = subscriber->getSubject()+"@"+address;
        const char *device = strTestText.c_str();
        vrpn_text_sender = new vrpn_Text_Sender(device, m_connection);
        vrpn_analog_server = new vrpn_Analog_Server(device, m_connection);
        vrpn_tracker_server = new vrpn_Tracker_Server(device, m_connection);
    }

    while (!m_connection->connected() && this->m_running)
        m_connection->mainloop();

    while (m_connection->connected() && this->m_running)
    {

        for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
        {
            CommunicationSubscriber* subscriber = it->second;
            ArgumentList argumentList = subscriber->getArgumentList();

            try
            {
                for (ArgumentList::iterator itArgument = argumentList.begin(); itArgument != argumentList.end(); itArgument++ )
                    createVRPNMessage(subscriber, *itArgument);
            } catch(const std::exception& e) {
                if (isVerbose())
                    msg_info("ServerCommunicationVRPN") << e.what();
            }
            m_connection->mainloop(&delay);
        }
    }
}

void ServerCommunicationVRPN::createVRPNMessage(CommunicationSubscriber* subscriber, std::string argument)
{
    std::stringstream messageStr;
    BaseData* data = fetchDataFromSenderBuffer(subscriber, argument);
    if (!data)
        throw std::invalid_argument("data is null");
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const void* valueVoidPtr = data->getValueVoidPtr();

    delay.tv_sec = 0L;
    delay.tv_usec = 0L;
    angle += 0.001f;

    if (!typeinfo->Container())
    {
        if (vrpn_text_sender)
            vrpn_text_sender->send_message(data->getValueString().c_str(), vrpn_TEXT_NORMAL);

        if (vrpn_analog_server)
        {
            vrpn_analog_server->setNumChannels(1);
            double *channels = vrpn_analog_server->channels();
            static int done = 0;
            if (!done)
            {
                //converting string to double*
                channels[0] = stod (data->getValueString());
                done = 1;
            }
            else
                channels[0] += stod (data->getValueString());

            vrpn_analog_server->report_changes();
        }

        if(vrpn_tracker_server)
        {   //Position of Tracker
            pos[0] = sin ( angle );
            pos[1] = stof (data->getValueString());
            pos[2] = stof (data->getValueString());

            //Orientation of Tracker
            for(int i=0; i<4; i++)
            {
                d_quat[i] = stof (data->getValueString());
            }

            vrpn_tracker_server->report_pose(sensor, delay, pos, d_quat, class_of_service);
        }

    }

    delete data;
}

/******************************************************************************
*                                                                             *
* RECEIVE PART                                                                *
*                                                                             *
******************************************************************************/

void ServerCommunicationVRPN::receiveData()
{
    std::string address = d_address.getValueString();
    std::vector<vrpn_BaseClass*> receivers;

    std::map<std::string, CommunicationSubscriber*> subscribersMap = getSubscribers();
    if (subscribersMap.size() == 0)
    {
        if (isVerbose())
            msg_info(this) << "Server Communication VRPN does not have Subscriber";
        return;
    }
    for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
    {
        CommunicationSubscriber* subscriber = it->second;

        //Taking a string in convertng it into char *
        std::string str = subscriber->getSubject()+"@"+address;
        const char *device = str.c_str();

        //Recieving Text via VRPN
        vrpn_Text_Receiver *vrpnText = new vrpn_Text_Receiver(device);
        vrpnText->register_message_handler( (void*) this, processTextMessage );
        receivers.push_back(vrpnText);

        //Receiving Analog via VRPN
        vrpn_Analog_Remote *vrpnAnalog = new vrpn_Analog_Remote(device);
        vrpnAnalog->register_change_handler( (void*) this, processAnalogMessage);
        receivers.push_back(vrpnAnalog);

        //Receiving Tracker via VRPN
        vrpn_Tracker_Remote *vrpnTracker = new vrpn_Tracker_Remote(device);
        vrpnTracker->register_change_handler( (void*) this, processTrackerMessage);
        receivers.push_back(vrpnTracker);
    }

    while(this->m_running)
    {
        for(auto rec : receivers )
        {
            rec->mainloop();
        }
    }
}

void VRPN_CALLBACK ServerCommunicationVRPN::processTextMessage(void *userdata, const vrpn_TEXTCB t)
{
    ServerCommunicationVRPN* instance = static_cast<ServerCommunicationVRPN*>(userdata);
    std::map<std::string, CommunicationSubscriber*> subscribersMap = instance->getSubscribers();
    ArgumentList messageStream;

    if (t.type == vrpn_TEXT_NORMAL)
    {
        std::string message = "VRPNString:";
        message.append("'");
        message.append(t.message);
        message.append("'");
        messageStream.push_back(message);
        std::cout  << " : Text message: " << message << std::endl;
        for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
        {
            CommunicationSubscriber* subscriber = it->second;
            instance->saveDatasToReceivedBuffer(subscriber->getSubject(), messageStream, -1, -1);
        }
    }
}

void VRPN_CALLBACK ServerCommunicationVRPN::processAnalogMessage(void *userdata, const vrpn_ANALOGCB a)
{
    int nbChannels = a.num_channel;

    ServerCommunicationVRPN* instance = static_cast<ServerCommunicationVRPN*>(userdata);
    std::map<std::string, CommunicationSubscriber*> subscribersMap = instance->getSubscribers();

    ArgumentList analogStream;

    if (a.num_channel>1)
    {
        int row=0, col=0;
        try
        {   // Matrix will have a single row but the number of columns depends on number of channels
            row = 1;
            col = a.num_channel;
            if (row < 0 || col < 0)
                return;
        }
        catch(std::invalid_argument& e)
        {
            msg_warning(instance) << "no available conversion for: " << e.what();
            return;
        }
        catch(std::out_of_range& e)
        {
            msg_warning(instance) << "out of range : " << e.what();
            return;
        }

        for (int i = 0; i < a.num_channel; i++)
        {
            std::string stream = "VRPNdouble:";
            stream.append(std::to_string(a.channel[i]));
            analogStream.push_back(stream);
        }

        if(analogStream.size() == 0)
        {
            msg_error(instance) << "argument list size is empty";
            return;
        }

        if((unsigned int)row*col != analogStream.size())
        {
            msg_error(instance) << "argument list size is != row/cols; " << analogStream.size() << " instead of " << row*col;
            return;
        }

        for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
        {
            CommunicationSubscriber* subscriber = it->second;
            instance->saveDatasToReceivedBuffer(subscriber->getSubject(), analogStream, row, col);
        }
    }

    else
    {
        std::string stream = "VRPNdouble:";
        stream.append(std::to_string(a.channel[0]));
        analogStream.push_back(stream);
        std::cout << stream << std::endl;

        for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
        {
            CommunicationSubscriber* subscriber = it->second;
            instance->saveDatasToReceivedBuffer(subscriber->getSubject(), analogStream, -1, -1);
        }
    }
}

void VRPN_CALLBACK ServerCommunicationVRPN::processTrackerMessage(void *userdata, const vrpn_TRACKERCB z)
{
    ServerCommunicationVRPN* instance = static_cast<ServerCommunicationVRPN*>(userdata);
    std::map<std::string, CommunicationSubscriber*> subscribersMap = instance->getSubscribers();
    ArgumentList trackerStream;
    int row, col;

    try
    {   // Matrix will have a single row but the number of columns will be 3.
        row = 1;
        col = 3;
        if (row < 0 || col < 0)
            return;
    }
    catch(std::invalid_argument& e)
    {
        msg_warning(instance) << "no available conversion for: " << e.what();
        return;
    }
    catch(std::out_of_range& e)
    {
        msg_warning(instance) << "out of range : " << e.what();
        return;
    }

    for (int i = 0; i < col; i++)
    {
        std::string stream = "VRPNdouble:";
        stream.append(std::to_string(z.pos[i]));
        trackerStream.push_back(stream);
    }

    if(trackerStream.size() == 0)
    {
        msg_error(instance) << "argument list size is empty";
        return;
    }

    if((unsigned int)row*col != trackerStream.size())
    {
        msg_error(instance) << "argument list size is != row/cols; " << trackerStream.size() << " instead of " << row*col;
        return;
    }

    for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
    {
        CommunicationSubscriber* subscriber = it->second;
        instance->saveDatasToReceivedBuffer(subscriber->getSubject(), trackerStream, row, col);
    }
}

std::string ServerCommunicationVRPN::getArgumentValue(std::string value)
{
    std::string stringData = value;
    std::string returnValue;
    size_t pos = stringData.find(":"); // That's how VRPN messages could be. Type:value
    stringData.erase(0, pos+1);
    std::remove_copy(stringData.begin(), stringData.end(), std::back_inserter(returnValue), '\'');
    return returnValue;
}

std::string ServerCommunicationVRPN::getArgumentType(std::string value)
{
    std::string stringType = value;
    size_t pos = stringType.find(":"); // That's how VRPN messages could be. Type:value
    if (pos == std::string::npos)
        return "VRPNString";
    stringType.erase(pos, stringType.size()-1);
    return stringType;
}
}   //communication
}   //component
}   //sofa
