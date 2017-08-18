#ifndef SOFA_SERVERCOMMUNICATIONZMQ_H
#define SOFA_SERVERCOMMUNICATIONZMQ_H

#include "serverCommunication.h"

#include <sofa/core/objectmodel/BaseObjectDescription.h>
using sofa::core::objectmodel::BaseObjectDescription;

#include <zmq.hpp>

namespace sofa
{

namespace component
{

namespace communication
{

template< class DataTypes >
class SOFA_COMMUNICATION_API ServerCommunicationZMQ : public ServerCommunication<DataTypes>
{
public:

    typedef ServerCommunication<DataTypes> Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ServerCommunicationZMQ, DataTypes), Inherited);

    ServerCommunicationZMQ    ();
    virtual ~ServerCommunicationZMQ();

    //////////////////////////////// Inherited from Base /////////////////////////////////
    virtual std::string getTemplateName() const {return templateName(this);}
    static std::string templateName(const ServerCommunicationZMQ<DataTypes>* = NULL);
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

    // Factoring for templates
    void convertDataToMessage(std::string& messageStr);
    void convertStringStreamToData(std::stringstream *stream);
    void checkDataSize(const unsigned int& nbDataFieldReceived);

};  //class ServerCommunicationZMQ

}   /// namespace communication
}   /// namespace component
}   /// namespace sofa

#endif // SOFA_SERVERCOMMUNICATIONZMQ_H

