#ifndef SOFA_SERVERCOMMUNICATIONVRPN_CPP
#define SOFA_SERVERCOMMUNICATIONVRPN_CPP

#include "serverCommunicationVRPN.inl"

using sofa::core::RegisterObject ;

namespace sofa
{

namespace component
{

namespace communication
{

SOFA_DECL_CLASS(ServerCommunicationVRPN)

int ServerCommunicationVRPNClass = sofa::core::RegisterObject("Communication component using VRPN protocol").add<ServerCommunicationVRPN>();

}   //namespace controller
}   //namespace component
}   //namespace sofa


#endif // SOFA_CONTROLLER_ServerCommunicationVRPN_CPP
