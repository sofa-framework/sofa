#ifndef SOFA_SERVERCOMMUNICATIONZMQ_H
#define SOFA_SERVERCOMMUNICATIONZMQ_H

#include "serverCommunication.h"

#include <zmq.hpp>

namespace sofa
{

namespace component
{

namespace communication
{

class SOFA_COMMUNICATION_API ServerCommunicationZMQ : public ServerCommunication
{
public:

    typedef ServerCommunication Inherited;
    SOFA_CLASS(ServerCommunicationZMQ, Inherited);

    ServerCommunicationZMQ();
    virtual ~ServerCommunicationZMQ();

    //////////////////////////////// Factory OSC type /////////////////////////////////
    typedef CommunicationDataFactory ZMQDataFactory;
    ZMQDataFactory* getFactoryInstance();
    virtual void initTypeFactory() override;
    /////////////////////////////////////////////////////////////////////////////////

    Data<helper::OptionsGroup>  d_pattern;


protected:

    zmq::context_t     m_context{1};
    zmq::socket_t      *m_socket;

    //////////////////////////////// Inherited from ServerCommunication /////////////////////////////////
    virtual void sendData();
    virtual void receiveData();
    /////////////////////////////////////////////////////////////////////////////////

    void sendRequest();
    void receiveRequest();

    std::string dataToString(CommunicationSubscriber* subscriber, std::string argument);
    void stringToData(std::string dataString);

};

}   /// namespace communication
}   /// namespace component
}   /// namespace sofa

#endif // SOFA_SERVERCOMMUNICATIONZMQ_H

