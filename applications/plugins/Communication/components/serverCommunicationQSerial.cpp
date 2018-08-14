#ifndef SOFA_SERVERCOMMUNICATIONQSerial_CPP
#define SOFA_SERVERCOMMUNICATIONQSerial_CPP

#include "serverCommunicationQSerial.inl"

namespace sofa
{

namespace component
{

namespace communication
{

SOFA_DECL_CLASS(ServerCommunicationQSerial)

int ServerCommunicationQSerialClass = sofa::core::RegisterObject("Comunication component using QSerial protocol").add<ServerCommunicationQSerial>();

}   //namespace controller
}   //namespace component
}   //namespace sofa


#endif // SOFA_CONTROLLER_ServerCommunicationQSerial_CPP
