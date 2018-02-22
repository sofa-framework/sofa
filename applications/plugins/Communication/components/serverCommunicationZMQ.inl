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

    m_timeout = 1000;
}

ServerCommunicationZMQ::~ServerCommunicationZMQ()
{
    this->m_running = false;

    if(isVerbose())
        msg_info(this) << "waiting for timeout";

    // this sleep ensure we does not delete/close the socket before its timeout happens
    std::this_thread::sleep_for(std::chrono::milliseconds(m_timeout * 2));

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

    getFactoryInstance()->registerCreator("matrixfloat", new DataCreator<FullMatrix<float>>());
    getFactoryInstance()->registerCreator("matrixdouble", new DataCreator<FullMatrix<double>>());
    getFactoryInstance()->registerCreator("matrixint", new DataCreator<FullMatrix<int>>());
}

std::string ServerCommunicationZMQ::defaultDataType()
{
    return "string";
}

int ServerCommunicationZMQ::getTimeout() const
{
    return m_timeout;
}

void ServerCommunicationZMQ::setTimeout(int timeout)
{
    m_timeout = timeout;
}

/******************************************************************************
*                                                                             *
* SEND PART                                                                   *
*                                                                             *
******************************************************************************/

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
        std::map<std::string, CommunicationSubscriber*> subscribersMap = getSubscribers();
        if (subscribersMap.size() == 0)
            continue;

        std::string messageStr;

        if(this->d_pattern.getValue().getSelectedItem() == "request/reply")
            receiveRequest();

        for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
        {
            CommunicationSubscriber* subscriber = it->second;
            ArgumentList argumentList = subscriber->getArgumentList();
            messageStr += subscriber->getSubject() + " ";

            try
            {
                for (ArgumentList::iterator itArgument = argumentList.begin(); itArgument != argumentList.end(); itArgument++ )
                    messageStr += createZMQMessage(subscriber, *itArgument);

                zmq::message_t message(messageStr.length());
                memcpy(message.data(), messageStr.c_str(), messageStr.length());

                bool status = m_socket->send(message);
                if(!status)
                    msg_warning(this) << "Problem with communication";
            } catch(const std::exception& e) {
                if (isVerbose())
                    msg_info("ServerCommunicationZMQ") << e.what();
            }
            messageStr.clear();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(int(1000000.0/(double)this->d_refreshRate.getValue())));
    }
}

std::string ServerCommunicationZMQ::createZMQMessage(CommunicationSubscriber* subscriber, std::string argument)
{
    std::stringstream messageStr;
    BaseData* data = fetchDataFromSenderBuffer(subscriber, argument);
    if (!data)
        throw std::invalid_argument("data is null");
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const void* valueVoidPtr = data->getValueVoidPtr();

    if (typeinfo->Container())
    {
        int nbRows = typeinfo->size();
        int nbCols  = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();
        messageStr << "matrix int:" << std::to_string(nbRows) << " int:" << std::to_string(nbCols) << " ";

        if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
        {
            msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
            messageStr << "string:'" << (data->getValueString()) << "' ";
        }
        else if (typeinfo->Text())
            for (int i=0; i < nbRows; i++)
                for (int j=0; j<nbCols; j++)
                    messageStr << "string:" << typeinfo->getTextValue(valueVoidPtr,(i*nbCols) + j).c_str() << " ";
        else if (typeinfo->Scalar())
            for (int i=0; i < nbRows; i++)
                for (int j=0; j<nbCols; j++)
                    messageStr << "float:" << (float)typeinfo->getScalarValue(valueVoidPtr,(i*nbCols) + j) << " ";
        else if (typeinfo->Integer())
            for (int i=0; i < nbRows; i++)
                for (int j=0; j<nbCols; j++)
                    messageStr << "int:" << (int)typeinfo->getIntegerValue(valueVoidPtr,(i*nbCols) + j) << " ";
    }
    else
    {
        if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
        {
            msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type=" << data->getValueTypeString() << " for data "<<data->getName()<<" ; returning string value" ;
            messageStr << "string:'" << (data->getValueString()) << "' ";
        }
        if (typeinfo->Text())
        {
            messageStr << "string:'" << (data->getValueString()) << "' ";
        }
        else if (typeinfo->Scalar())
        {
            messageStr << "float:" << (data->getValueString()) << " ";
        }
        else if (typeinfo->Integer())
        {
            messageStr << "int:" << (data->getValueString()) << " ";
        }
    }
    delete data;
    return messageStr.str();
}

/******************************************************************************
*                                                                             *
* RECEIVE PART                                                                *
*                                                                             *
******************************************************************************/

void ServerCommunicationZMQ::receiveData()
{
    std::string ip ="localhost";
    if(d_address.isSet())
        ip = d_address.getValue();

    std::string address = "tcp://"+ip+":";
    std::string port = d_port.getValueString();
    address.insert(address.length(), port);

    if(d_pattern.getValue().getSelectedItem() == "publish/subscribe")
    {
        m_socket = new zmq::socket_t(m_context, ZMQ_SUB);
        m_socket->connect(address.c_str());
        m_socket->setsockopt(ZMQ_SUBSCRIBE, "", 0);
        m_socket->setsockopt(ZMQ_RCVTIMEO, &m_timeout, sizeof(m_timeout));
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
        zmq::message_t reply;
        bool status = this->m_socket->recv(&reply);
        if(status)
        {
            std::string rpl = std::string(static_cast<char*>(reply.data()), reply.size());
            processMessage(rpl);
        }
        else
            if(isVerbose())
                msg_warning(this) << "Timeout or problem with communication";
    }
}

void ServerCommunicationZMQ::processMessage(std::string dataString)
{
    std::string subject;
    ArgumentList onlyArgumentList;
    ArgumentList argumentList = stringToArgumentList(dataString);

    if (argumentList.empty() || argumentList.size() < 2) // < 2 = subject only
        return;

    std::vector<std::string>::iterator it = argumentList.begin();
    subject = *it;
    if (!getSubscriberFor(subject))
        return;

    std::string firstArg = getArgumentValue(*(++it));
    if (firstArg.compare("matrix") == 0)
    {
        int row = 0, col = 0;
        if (argumentList.size() >= 3+1) // +1 due to subject count in
        {
            for (it = argumentList.begin()+4; it != argumentList.end();it++)
                onlyArgumentList.push_back(*it);
            try
            {
                row = std::stoi(getArgumentValue(argumentList.at(2)));
                col = std::stoi(getArgumentValue(argumentList.at(3)));
                if (row < 0 || col < 0)
                    return;
            } catch(std::invalid_argument& e){
                msg_warning() << "no available conversion for: " << e.what();
                return;
            } catch(std::out_of_range& e){
                msg_warning() << "out of range : " << e.what();
                return;
            }
        } else
            msg_warning() << subject << " is matrix, but message size is not correct. Should be : /subject matrix rows cols value value value... ";

        ++it; // needed for the accessing the correct first argument

        if(onlyArgumentList.size() == 0)
        {
            msg_error() << "argument list size is empty";
            return;
        }

        if((unsigned int)row*col != onlyArgumentList.size())
        {
            msg_error() << "argument list size is != row/cols; " << onlyArgumentList.size() << " instead of " << row*col;
            return;
        }

        saveDatasToReceivedBuffer(subject, onlyArgumentList, row, col);
    }
    else
    {
        for (it = argumentList.begin()+1; it != argumentList.end();it++)
            onlyArgumentList.push_back(*it);
        saveDatasToReceivedBuffer(subject, onlyArgumentList, -1, -1);
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

/******************************************************************************
*                                                                             *
* MESSAGE CONVERTION PART                                                     *
*                                                                             *
******************************************************************************/

ArgumentList ServerCommunicationZMQ::stringToArgumentList(std::string dataString)
{
    std::regex rgx("\\s+");
    std::sregex_token_iterator iter(dataString.begin(), dataString.end(), rgx, -1);
    std::sregex_token_iterator end;
    ArgumentList listArguments;
    while (iter != end)
    {
        std::string tmp = *iter;
        if (tmp.find("string:'") != std::string::npos && tmp.find_last_of("'") != tmp.length()-1)
        {
            bool stop = false;
            std::string concat = tmp;
            iter++;
            while (iter != end && !stop)
            {
                std::string anotherTmp = *iter;
                if (anotherTmp.find("string:'") != std::string::npos)
                    stop = true;
                else
                {
                    concat.append(" ");
                    concat.append(anotherTmp);
                    iter++;
                }
            }
            listArguments.push_back(concat);
        }
        else
        {
            iter++;
            listArguments.push_back(tmp);
        }
    }
    return listArguments;
}

std::string ServerCommunicationZMQ::getArgumentValue(std::string value)
{
    std::string stringData = value;
    std::string returnValue;
    size_t pos = stringData.find(":"); // That's how ZMQ messages could be. Type:value
    stringData.erase(0, pos+1);
    std::remove_copy(stringData.begin(), stringData.end(), std::back_inserter(returnValue), '\'');
    return returnValue;
}

std::string ServerCommunicationZMQ::getArgumentType(std::string value)
{
    std::string stringType = value;
    size_t pos = stringType.find(":"); // That's how ZMQ messages could be. Type:value
    if (pos == std::string::npos)
        return "string";
    stringType.erase(pos, stringType.size()-1);
    return stringType;
}

}   /// namespace communication
}   /// namespace component
}   /// namespace sofa


#endif // SOFA_CONTROLLER_ServerCommunicationZMQ_INL
