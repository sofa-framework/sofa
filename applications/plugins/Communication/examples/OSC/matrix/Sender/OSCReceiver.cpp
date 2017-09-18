#include <iostream>

#include "osc/OscReceivedElements.h"
#include "osc/OscPrintReceivedElements.h"

#include "ip/UdpSocket.h"
#include "ip/PacketListener.h"

#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <cmath>

timeval t1, t2;

class OscDumpPacketListener : public PacketListener{
public:
    virtual void ProcessPacket( const char *data, int size,
            const IpEndpointName& remoteEndpoint )
    {
        (void) remoteEndpoint; // suppress unused parameter warning

        // Uncorrect results if frequency == 1hz, due to tv_usec precision
        gettimeofday(&t1, NULL);
            std::cout << "Animation Loop frequency : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
        gettimeofday(&t2, NULL);
        std::cout << osc::ReceivedPacket( data, size );
    }
};

int main(int argc, char* argv[])
{
    if( argc >= 2 && std::strcmp( argv[1], "-h" ) == 0 ){
        std::cout << "usage: OscDump [port]\n";
        return 0;
    }

    gettimeofday(&t2, NULL);
    gettimeofday(&t1, NULL);
    int port = 6000;

    if( argc >= 2 )
        port = std::atoi( argv[1] );

    OscDumpPacketListener listener;
    UdpListeningReceiveSocket s(
            IpEndpointName( IpEndpointName::ANY_ADDRESS, port ),
            &listener );

    std::cout << "listening for input on port " << port << "...\n";
    std::cout << "press ctrl-c to end\n";

    s.RunUntilSigInt();

    std::cout << "finishing.\n";

    return 0;
}


