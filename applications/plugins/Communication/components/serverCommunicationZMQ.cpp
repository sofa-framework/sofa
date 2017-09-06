#ifndef SOFA_SERVERCOMMUNICATIONZMQ_CPP
#define SOFA_SERVERCOMMUNICATIONZMQ_CPP

#include "serverCommunicationZMQ.inl"

namespace sofa
{

namespace component
{

namespace communication
{

SOFA_DECL_CLASS(ServerCommunicationZMQ)

int ServerCommunicationZMQClass = sofa::core::RegisterObject("Comunication component using ZMQprotocol").add<ServerCommunicationZMQ>();

}   //namespace controller
}   //namespace component
}   //namespace sofa


#endif // SOFA_CONTROLLER_ServerCommunicationZMQ_CPP

