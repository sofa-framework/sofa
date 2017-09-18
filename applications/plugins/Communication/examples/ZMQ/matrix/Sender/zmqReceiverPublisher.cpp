#include <zmq.hpp>
#include <string>
#include <iostream>
#include <unistd.h>

int main ()
{
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_SUB);
    socket.connect ("tcp://localhost:6000");
    socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    while (1)
    {
        zmq::message_t reply;
        socket.recv (&reply);
        std::cout << "Received : " << (char*)reply.data() << std::endl;
    }
    return 0;
}

