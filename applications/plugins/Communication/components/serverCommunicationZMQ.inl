#ifndef SOFA_CONTROLLER_ServerCommunicationZMQ_INL
#define SOFA_CONTROLLER_ServerCommunicationZMQ_INL

#include "serverCommunicationZMQ.h"

namespace sofa
{

namespace component
{

namespace communication
{

template<class DataTypes>
ServerCommunicationZMQ<DataTypes>::ServerCommunicationZMQ()
    : Inherited()
    , d_pattern(initData(&d_pattern, OptionsGroup(2,"publish/subscribe","request/reply"), "pattern",
                                      "Pattern used for communication. \n"
                                      "publish/subscribe: Messages sent are distributed in a fan out fashion to all connected peers. Never blocks.\n"
                                      "request/reply: Message sent are waiting for reply. Allows only an alternating sequence of send\reply calls.\n"
                                      "Default is publish/subscribe. WARNING: the pattern should be the same for both sender and receiver to be effective."))

{
}


template<class DataTypes>
ServerCommunicationZMQ<DataTypes>::~ServerCommunicationZMQ()
{
    m_socket->close();
    delete m_socket;
    Inherited::closeCommunication();
}

template<class DataTypes>
void ServerCommunicationZMQ<DataTypes>::sendData()
{
    std::string adress = "tcp://*:";
    std::string port = this->d_port.getValueString();
    adress.insert(adress.length(), port);

    if(this->d_pattern.getValue().getSelectedItem() == "publish/subscribe")
        this->m_socket = new zmq::socket_t(this->m_context, ZMQ_PUB);
    else
        this->m_socket = new zmq::socket_t(this->m_context, ZMQ_REP);

    this->m_socket->bind(adress.c_str());
    while (this->m_running)
    {
        if(this->d_pattern.getValue().getSelectedItem() == "request/reply")
            receiveRequest();

        std::string messageStr;
        convertDataToMessage(messageStr);

        zmq::message_t message(messageStr.length());

        memcpy(message.data(), messageStr.c_str(), messageStr.length());

        bool status = m_socket->send(message);
        if(!status)
            msg_warning(this) << "Problem with communication";
    }
}


template<class DataTypes>
void ServerCommunicationZMQ<DataTypes>::receiveData()
{
    std::string IP ="localhost";
    if(this->d_adress.isSet())
        IP = this->d_adress.getValue();

    std::string adress = "tcp://"+IP+":";
    std::string port = this->d_port.getValueString();
    adress.insert(adress.length(), port);

    if(this->d_pattern.getValue().getSelectedItem() == "publish/subscribe")
    {
        m_socket = new zmq::socket_t(m_context, ZMQ_SUB);

        // Should drop sent messages if exceed HWM.
        // WARNING: does not work, uncomment if problem solved
        //uint64_t HWM = d_HWM.getValue();
        //m_socket->setsockopt(ZMQ_RCVHWM, &HWM, sizeof(HWM));

        m_socket->connect(adress.c_str());
        m_socket->setsockopt(ZMQ_SUBSCRIBE, "", 0); // Arg2: publisher name - Arg3: size of publisher name
    }
    else
    {
        m_socket = new zmq::socket_t(m_context, ZMQ_REQ);
        m_socket->connect(adress.c_str());
    }

    while (this->m_running)
    {
        if(this->d_pattern.getValue().getSelectedItem() == "request/reply")
            sendRequest();

        zmq::message_t message;
        bool status = m_socket->recv(&message);
        if(status)
        {
            char messageChar[message.size()];
            memcpy(&messageChar, message.data(), message.size());

            std::stringstream stream;
            unsigned int nbDataFieldReceived = 0;
            for(unsigned int i=0; i<message.size(); i++)
            {
                if(messageChar[i]==' ' || i==message.size()-1)
                    nbDataFieldReceived++;

                if(messageChar[i]==',')
                    messageChar[i]='.';

                stream << messageChar[i];
            }

            convertStringStreamToData(&stream);
            checkDataSize(nbDataFieldReceived);
        }
        else
            msg_warning(this) << "Problem with communication";
    }
}

template<class DataTypes>
void ServerCommunicationZMQ<DataTypes>::convertDataToMessage(std::string& messageStr)
{
    for(unsigned int i=0; i<this->d_data_copy.size(); i++)
    {
        ReadAccessor<Data<DataTypes>> data = this->d_data_copy[i];
        messageStr += std::to_string(data) + " ";
    }
}

template<class DataTypes>
void ServerCommunicationZMQ<DataTypes>::convertStringStreamToData(std::stringstream* stream)
{
    for (unsigned int i= 0; i<this->d_data_copy.size(); i++)
    {
        WriteAccessor<Data<DataTypes>> data = this->d_data_copy[i];
        (*stream) >> data;
    }
}

template<class DataTypes>
void ServerCommunicationZMQ<DataTypes>::checkDataSize(const unsigned int& nbDataFieldReceived)
{
    if(nbDataFieldReceived!= this->d_nbDataField.getValue())
        msg_warning(this) << "Something wrong with the size of data received. Please check template.";
}

template<class DataTypes>
void ServerCommunicationZMQ<DataTypes>::sendRequest()
{
    zmq::message_t request;
    m_socket->send(request);
}

template<class DataTypes>
void ServerCommunicationZMQ<DataTypes>::receiveRequest()
{
    zmq::message_t request;
    m_socket->recv(&request);
}


}   /// namespace communication
}   /// namespace component
}   /// namespace sofa


#endif // SOFA_CONTROLLER_ServerCommunicationZMQ_INL

