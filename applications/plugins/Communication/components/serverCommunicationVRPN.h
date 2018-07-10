#include "serverCommunication.h"

#include<vrpn_BaseClass.h>
#include<vrpn_Analog.h>
#include<vrpn_Text.h>
#include<vrpn_Button.h>
#include<vrpn_Tracker.h>
#include<vrpn_Connection.h>
#include<vrpn_Configure.h>

struct VrpnSenders
{
    vrpn_Text_Sender *vrpn_text_sender;
    vrpn_Analog_Server *vrpn_analog_server;
//    vrpn_Button_Server *vrpn_button_server;
//    vrpn_Tracker_Server *vrpn_tracker_server;
};

namespace sofa
{

namespace component
{

namespace communication
{

class SOFA_COMMUNICATION_API ServerCommunicationVRPN : public ServerCommunication
{
public:

    typedef ServerCommunication Inherited;
    SOFA_CLASS(ServerCommunicationVRPN, Inherited);

    ServerCommunicationVRPN();
    virtual ~ServerCommunicationVRPN();

    //////////////////////////////// Factory VRPN type /////////////////////////////////
    typedef CommunicationDataFactory VRPNDataFactory;
    VRPNDataFactory* getFactoryInstance();
    virtual void initTypeFactory() override;
    /////////////////////////////////////////////////////////////////////////////////

    virtual std::string getArgumentType(std::string value) override;
    virtual std::string getArgumentValue(std::string value) override;

protected:

    vrpn_Connection *m_connection;
    std::map<CommunicationSubscriber*, VrpnSenders> senders;

    //////////////////////////////// Inherited from ServerCommunication /////////////////////////////////
    virtual void sendData() override;
    virtual void receiveData() override;
    virtual std::string defaultDataType() override;
    /////////////////////////////////////////////////////////////////////////////////

    static void VRPN_CALLBACK processTextMessage(void *userdata, const vrpn_TEXTCB t);
    static void VRPN_CALLBACK processAnalogMessage(void *userdata, const vrpn_ANALOGCB a);
    // static void VRPN_CALLBACK processTrackerMessage(void* userdata, const vrpn_TRACKERCB r);
    void createVRPNMessage(CommunicationSubscriber *subscriber, std::string argument);
};

}   /// namespace communication
}   /// namespace component
}   /// namespace sofa
