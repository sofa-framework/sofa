#ifndef SOFA_SERVERCOMMUNICATIONQSerial_H
#define SOFA_SERVERCOMMUNICATIONQSerial_H
#define WIN32_LEAN_AND_MEAN
#include "serverCommunication.h"

#include <QObject>
#include <QDebug>
#include <QCoreApplication>
#include <QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>
#include <QtSerialPort/QtSerialPort>

#include <iostream>
#include <thread>
namespace sofa
{

namespace component
{

namespace communication
{

class SOFA_COMMUNICATION_API ServerCommunicationQSerial : public ServerCommunication
{
public:

    typedef ServerCommunication Inherited;
    SOFA_CLASS(ServerCommunicationQSerial, Inherited);

    ServerCommunicationQSerial();
    virtual ~ServerCommunicationQSerial();

    ArgumentList stringToArgumentList(std::string dataString);

    //////////////////////////////// Factory QSerial type /////////////////////////////////
    typedef CommunicationDataFactory QSerialDataFactory;
    QSerialDataFactory* getFactoryInstance();
    virtual void initTypeFactory() override;
    /////////////////////////////////////////////////////////////////////////////////

    virtual std::string getArgumentType(std::string value) override;
    virtual std::string getArgumentValue(std::string value) override;

    //Data<helper::OptionsGroup>  d_pattern;


    //int getTimeout() const;
    //void setTimeout(int timeout);

protected:

    std::thread m_thread;
    bool m_running = true;

    //////////////////////////////// Inherited from ServerCommunication /////////////////////////////////
    virtual void sendData() override;
    virtual void receiveData() override;
//    virtual std::string defaultDataType() override;
//    /////////////////////////////////////////////////////////////////////////////////

//    void sendRequest();
//    void receiveRequest();

//    std::string createQSerialMessage(CommunicationSubscriber* subscriber, std::string argument);
    void processMessage(QByteArray msg);

};

}   /// namespace communication
}   /// namespace component
}   /// namespace sofa

#endif // SOFA_SERVERCOMMUNICATIONQSerial_H

