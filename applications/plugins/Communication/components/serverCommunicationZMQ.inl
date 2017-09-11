#ifndef SOFA_CONTROLLER_ServerCommunicationZMQ_INL
#define SOFA_CONTROLLER_ServerCommunicationZMQ_INL

#include "serverCommunicationZMQ.h"
#include <Communication/components/CommunicationSubscriber.h>

namespace sofa
{

namespace component
{

namespace communication
{

ServerCommunicationZMQ::ServerCommunicationZMQ()
    : Inherited()
    , d_pattern(initData(&d_pattern, OptionsGroup(2,"publish/subscribe","request/reply"), "pattern",
                         "Pattern used for communication. \n"
                         "publish/subscribe: Messages sent are distributed in a fan out fashion to all connected peers. Never blocks.\n"
                         "request/reply: Message sent are waiting for reply. Allows only an alternating sequence of send\reply calls.\n"
                         "Default is publish/subscribe. WARNING: the pattern should be the same for both sender and receiver to be effective."))
{
}

ServerCommunicationZMQ::~ServerCommunicationZMQ()
{
    m_socket->close();
    delete m_socket;
    Inherited::closeCommunication();
}

ServerCommunicationZMQ::ZMQDataFactory* ServerCommunicationZMQ::getFactoryInstance(){
    static ZMQDataFactory* s_localfactory = nullptr ;
    if(s_localfactory==nullptr)
        s_localfactory = new ServerCommunicationZMQ::ZMQDataFactory() ;
    return s_localfactory ;
}

void ServerCommunicationZMQ::initTypeFactory()
{
    getFactoryInstance()->registerCreator("float", new DataCreator<float>());
    getFactoryInstance()->registerCreator("double", new DataCreator<double>());
    getFactoryInstance()->registerCreator("int", new DataCreator<int>());
    getFactoryInstance()->registerCreator("string", new DataCreator<std::string>());
    //TODO Correct value for matrix
    getFactoryInstance()->registerCreator("matrixfloat", new DataCreator<helper::vector<float>>());
    getFactoryInstance()->registerCreator("matrixint", new DataCreator<helper::vector<int>>());
}

void ServerCommunicationZMQ::sendData()
{
    std::string address = "tcp://*:";
    std::string port = this->d_port.getValueString();
    address.insert(address.length(), port);

    if(this->d_pattern.getValue().getSelectedItem() == "publish/subscribe")
        this->m_socket = new zmq::socket_t(this->m_context, ZMQ_PUB);
    else
        this->m_socket = new zmq::socket_t(this->m_context, ZMQ_REP);

    this->m_socket->bind(address.c_str());
    while (this->m_running)
    {
        if(this->d_pattern.getValue().getSelectedItem() == "request/reply")
            receiveRequest();

        std::map<std::string, CommunicationSubscriber*> subscribersMap = getSubscribers();
        std::string messageStr;

        for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
        {
            CommunicationSubscriber* subscriber = it->second;
            std::vector<std::string> argumentList = subscriber->getArgumentList();
            messageStr += subscriber->getSubject() + " ";

            for (std::vector<std::string>::iterator itArgument = argumentList.begin(); itArgument != argumentList.end(); itArgument++ )
                messageStr += dataToString(subscriber, *itArgument);

            zmq::message_t message(messageStr.length());
            memcpy(message.data(), messageStr.c_str(), messageStr.length());

            bool status = m_socket->send(message);
            if(!status)
                msg_warning(this) << "Problem with communication";
            messageStr.clear();
            std::this_thread::sleep_for(std::chrono::microseconds(int(1000000.0/(double)this->d_refreshRate.getValue())));
        }
    }
}

void ServerCommunicationZMQ::receiveData()
{
    std::string IP ="localhost";
    if(d_address.isSet())
        IP = d_address.getValue();

    std::string address = "tcp://"+IP+":";
    std::string port = d_port.getValueString();
    address.insert(address.length(), port);

    if(d_pattern.getValue().getSelectedItem() == "publish/subscribe")
    {
        m_socket = new zmq::socket_t(m_context, ZMQ_SUB);
        m_socket->connect(address.c_str());
        m_socket->setsockopt(ZMQ_SUBSCRIBE, "", 0); // Arg2: publisher name - Arg3: size of publisher name
    }
    else
    {
        m_socket = new zmq::socket_t(m_context, ZMQ_REQ);
        m_socket->connect(address.c_str());
    }


    while (this->m_running)
    {
        if(this->d_pattern.getValue().getSelectedItem() == "request/reply")
            sendRequest();

        zmq::message_t message;
        bool status = this->m_socket->recv(&message);
        if(status)
        {

            char* tmp = (char*)malloc(sizeof(char) * message.size());
            memcpy(tmp, message.data(), message.size());

            stringToData(std::string(tmp));
        }
        else
            msg_warning(this) << "Problem with communication";

    }
}

std::string ServerCommunicationZMQ::dataToString(CommunicationSubscriber* subscriber, std::string argument)
{
    std::string messageStr = "empty";
    SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK> source = subscriber->getSource();
    MapData dataMap = source->getDataAliases();
    MapData::const_iterator itData = dataMap.find(argument);

    // handle no argument
    if (itData != dataMap.end())
    {
        BaseData* data = itData->second;
        const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
        messageStr.clear();

        if (typeinfo->Container())
        {
            int rowWidth = typeinfo->size();
            int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();
            messageStr += "matrix int::" + std::to_string(rowWidth) + " int::" + std::to_string(nbRows) + " ";
            if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
            {
                msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
                messageStr += "type::unknow " +(data->getValueString()) + " ";
            }

            if (typeinfo->Text())
            {
                messageStr += "type:string ";
            }
            else if (typeinfo->Scalar())
            {
                messageStr += "type:float ";
            }
            else if (typeinfo->Integer())
            {
                messageStr += "type:int ";
            }
        }
        else
        {
            if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
            {
                msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type=" << data->getValueTypeString() << " for data "<<data->getName()<<" ; returning string value" ;
                messageStr += "unknow:";
            }
            if (typeinfo->Text())
            {
                messageStr += "string:";
            }
            else if (typeinfo->Scalar())
            {
                messageStr += "float:";
            }
            else if (typeinfo->Integer())
            {
                messageStr += "int:";
            }
        }
        messageStr += (data->getValueString()) + " ";
    }
    return messageStr;
}

std::vector<std::string> stringToArgumentList(std::string dataString)
{
    std::regex rgx("\\s+");
    std::sregex_token_iterator iter(dataString.begin(), dataString.end(), rgx, -1);
    std::sregex_token_iterator end;
    std::vector<std::string> listArguments;

    for ( ; iter != end; ++iter)
        listArguments.push_back(*iter);
    return listArguments;
}


std::string getArgumentValue(std::string value)
{
    std::string stringData = value;
    size_t pos = stringData.find(":"); // That's how ZMQ messages could be. Type:value
    stringData.erase(0, pos+1);
    return stringData;
}

std::string getArgumentType(std::string value)
{
    std::string stringType = value;
    size_t pos = stringType.find(":"); // That's how ZMQ messages could be. Type:value
    stringType.erase(pos, stringType.size()-1);
    return stringType;
}

void ServerCommunicationZMQ::stringToData(std::string dataString)
{
    BaseData* data;
    std::string subject;
    CommunicationSubscriber * subscriber;
    std::vector<std::string> argumentList = stringToArgumentList(dataString);
    SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK> source;

    if (argumentList.empty())
        return;

    std::vector<std::string>::iterator it = argumentList.begin();
    subject = *it;
    subscriber = getSubscriberFor(subject);
    if (!subscriber)
        return;
    source = subscriber->getSource();

    std::string firstArg = getArgumentValue(*(++it));

    if (firstArg.compare("matrix") == 0)
    {
        int row = 0, col = 0;
        if (argumentList.size() >= 3+1) // +1 due to subject count in
        {
            try
            {
                row = std::stoi(getArgumentValue(*(++it)));
                col = std::stoi(getArgumentValue(*(++it)));
                if (row < 0 || col < 0)
                    return;
            } catch(std::invalid_argument& e){
                msg_warning() << "no available conversion for: " << getArgumentValue(*(++it));
                return;
            } catch(std::out_of_range& e){
                msg_warning() << "out of range : " << getArgumentValue(*(++it));
                return;
            }
        }
        data = fetchData(source, "matrix" + getArgumentType(*(++it)), subscriber->getArgumentName(0));
        if (!data)
            return;
        std::stringstream stream;
        for ( it ; it != argumentList.end(); it++)
            stream << getArgumentValue(*it) << " ";
        data->read(stream.str());
    }
    else
    {
        if (!isSubscribedTo(subject, argumentList.size()))
            return;
        int i = 0;
        for ( it ; it != argumentList.end(); it++)
        {
            data = fetchData(source, getArgumentType(*it), subscriber->getArgumentName(i));
            if (!data)
                continue;
            data->read(getArgumentValue(*it));
            i++;
        }
    }
}

void ServerCommunicationZMQ::sendRequest()
{
    zmq::message_t request;
    m_socket->send(request);
}

void ServerCommunicationZMQ::receiveRequest()
{
    zmq::message_t request;
    m_socket->recv(&request);
}


}   /// namespace communication
}   /// namespace component
}   /// namespace sofa


#endif // SOFA_CONTROLLER_ServerCommunicationZMQ_INL
