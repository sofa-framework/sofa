#include "osc/OscOutboundPacketStream.h"
#include "ip/UdpSocket.h"

#include <iostream>
#include <unistd.h>

#define ADDRESS "127.0.0.1"
#define PORT 6000

#define OUTPUT_BUFFER_SIZE 8192

int main()
{
    std::cout << "Communication OSC color demo" << std::endl;
    UdpTransmitSocket transmitSocket( IpEndpointName( ADDRESS, PORT ) );
    while (1)
    {
        char buffer[OUTPUT_BUFFER_SIZE];
        osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );

        p << osc::BeginBundleImmediate;
        std::string messageName = "/colorLight";
        p << osc::BeginMessage(messageName.c_str());

        std::string str;
        str += std::to_string(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1.0)));
        str += " ";
        str += std::to_string(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1.0)));
        str += " ";
        str += std::to_string(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1.0)));
        str += " ";
        str += std::to_string(1.0);
        p << str.c_str();

        /// ANOTHER WAY TO SEND COLOR, using matrix style. Color is a vector, then you can use matrix syle
        /// matrix style is more consistent than using string
        //        p << "matrix";
        //        p << 3 ;
        //        p << 1 ;
        //        p << static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/MAX));
        //        p << static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/MAX));
        //        p << static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/MAX));


        p << osc::EndMessage;
        p << osc::EndBundle;
        transmitSocket.Send( p.Data(), p.Size() );

        usleep(100000);
    }

    return 0;
}

