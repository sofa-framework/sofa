#ifndef SOFA_CONTROLLER_ServerCommunicationQSerial_INL
#define SOFA_CONTROLLER_ServerCommunicationQSerial_INL

#include "serverCommunicationQSerial.h"
#include <Communication/components/CommunicationSubscriber.h>

namespace sofa
{

namespace component
{

namespace communication
{

ServerCommunicationQSerial::ServerCommunicationQSerial()
    : Inherited()
{
    serial = new QSerialPort(this);
}

ServerCommunicationQSerial::~ServerCommunicationQSerial()
{
    this->m_running = false;

    if(isVerbose())
        msg_info(this) << "waiting for timeout";

    serial->close();
    Inherited::closeCommunication();
}

ServerCommunicationQSerial::QSerialDataFactory* ServerCommunicationQSerial::getFactoryInstance(){
    static QSerialDataFactory* s_localfactory = nullptr ;
    if(s_localfactory==nullptr)
        s_localfactory = new ServerCommunicationQSerial::QSerialDataFactory() ;
    return s_localfactory ;
}

void ServerCommunicationQSerial::initTypeFactory()
{
    getFactoryInstance()->registerCreator("QSerialfloat", new DataCreator<float>());
    getFactoryInstance()->registerCreator("QSerialdouble", new DataCreator<double>());
    getFactoryInstance()->registerCreator("QSerialint", new DataCreator<int>());
    getFactoryInstance()->registerCreator("QSerialstring", new DataCreator<std::string>());

    getFactoryInstance()->registerCreator("matrixfloat", new DataCreator<FullMatrix<float>>());
    getFactoryInstance()->registerCreator("matrixdouble", new DataCreator<FullMatrix<double>>());
    getFactoryInstance()->registerCreator("matrixint", new DataCreator<FullMatrix<int>>());
}

std::string ServerCommunicationQSerial::defaultDataType()
{
    return "QSerialstring";
}

/******************************************************************************
*                                                                             *
* SEND PART                                                                   *
*                                                                             *
******************************************************************************/

void ServerCommunicationQSerial::sendData()
{
    std::string port = this->d_port.getValueString();

    while (this->m_running)
    {
        connect(serial, SIGNAL(readyRead()), this, SLOT(readData()));
        std::map<std::string, CommunicationSubscriber*> subscribersMap = getSubscribers();
        if (subscribersMap.size() == 0)
            continue;

        std::string messageStr;

        for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
        {
            CommunicationSubscriber* subscriber = it->second;
            ArgumentList argumentList = subscriber->getArgumentList();
            messageStr += subscriber->getSubject() + " ";

            serial->setPort(port);
            serial->setBaudRate(QSerialPort::Baud9600);
            serial->setDataBits(QSerialPort::Data8);
            if(serial->open(QIODevice::ReadWrite))
            {
                try
                {
                    for (ArgumentList::iterator itArgument = argumentList.begin(); itArgument != argumentList.end(); itArgument++ )
                        messageStr += createQSerialMessage(subscriber, *itArgument);

                    bool status = serial->write(messageStr);
                    if(!status)
                        msg_warning(this) << "Problem with communication";
                } catch(const std::exception& e) {
                    if (isVerbose())
                        msg_info("ServerCommunicationQSerial") << e.what();
                }
            }
            messageStr.clear();
        }
    }
}

std::string ServerCommunicationQSerial::createQSerialMessage(CommunicationSubscriber *subscriber, std::string argument)
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
        messageStr << "matrixint:" << std::to_string(nbRows) << " QSerialint:" << std::to_string(nbCols) << " ";

        if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
        {
            msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
            messageStr << "QSerialstring:'" << (data->getValueString()) << "' ";
        }
        else if (typeinfo->Text())
            for (int i=0; i < nbRows; i++)
                for (int j=0; j<nbCols; j++)
                    messageStr << "QSerialstring:" << typeinfo->getTextValue(valueVoidPtr,(i*nbCols) + j).c_str() << " ";
        else if (typeinfo->Scalar())
            for (int i=0; i < nbRows; i++)
                for (int j=0; j<nbCols; j++)
                    messageStr << "QSerialfloat:" << (float)typeinfo->getScalarValue(valueVoidPtr,(i*nbCols) + j) << " ";
        else if (typeinfo->Integer())
            for (int i=0; i < nbRows; i++)
                for (int j=0; j<nbCols; j++)
                    messageStr << "QSerialint:" << (int)typeinfo->getIntegerValue(valueVoidPtr,(i*nbCols) + j) << " ";
    }
    else
    {
        if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
        {
            msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type=" << data->getValueTypeString() << " for data "<<data->getName()<<" ; returning string value" ;
            messageStr << "QSerialstring:'" << (data->getValueString()) << "' ";
        }
        if (typeinfo->Text())
        {
            messageStr << "QSerialstring:'" << (data->getValueString()) << "' ";
        }
        else if (typeinfo->Scalar())
        {
            messageStr << "QSerialfloat:" << (data->getValueString()) << " ";
        }
        else if (typeinfo->Integer())
        {
            messageStr << "QSerialint:" << (data->getValueString()) << " ";
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

void ServerCommunicationQSerial::receiveData()
{
    std::string port = d_port.getValueString();
    while(m_running)
    {
        serial->setPort(port);
        serial->setBaudRate(QSerialPort::Baud9600);
        serial->setDataBits(QSerialPort::Data8);
        if (serial->open(QIODevice::ReadOnly))
        {
            serial->setDataTerminalReady(true);
            //                    serial->setRequestToSend(true);
            if(serial->waitForReadyRead(2000))
            {
                QByteArray byteArray = serial->readAll();
                if(!byteArray.isEmpty())
                {
                    //QByteArray byteArray = serial->readAll();
                    processMessage(byteArray);
                }
                else
                    if(isVerbose())
                        msg_warning(this) << "Timeout or problem with communication";
            }
            serial->close();
        }
    }
}

void ServerCommunicationQSerial::processMessage(QByteArray msg)
{
    std::string dataString = msg.toStdString();
    std::cout << "received " << dataString << std::endl;

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

/******************************************************************************
*                                                                             *
* MESSAGE CONVERTION PART                                                     *
*                                                                             *
******************************************************************************/

ArgumentList ServerCommunicationQSerial::stringToArgumentList(std::string dataString)
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

std::string ServerCommunicationQSerial::getArgumentValue(std::string value)
{
    std::string stringData = value;
    std::string returnValue;
    size_t pos = stringData.find(":"); // That's how QSerial messages could be. Type:value
    stringData.erase(0, pos+1);
    std::remove_copy(stringData.begin(), stringData.end(), std::back_inserter(returnValue), '\'');
    return returnValue;
}

std::string ServerCommunicationQSerial::getArgumentType(std::string value)
{
    std::string stringType = value;
    size_t pos = stringType.find(":"); // That's how QSerial messages could be. Type:value
    if (pos == std::string::npos)
        return "string";
    stringType.erase(pos, stringType.size()-1);
    return stringType;
}

}   /// namespace communication
}   /// namespace component
}   /// namespace sofa


#endif // SOFA_CONTROLLER_ServerCommunicationQSerial_INL
