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
        char* tmp = (char*)malloc(sizeof(char) * reply.size());
        memcpy(tmp, reply.data(), reply.size());
        std::cout << "Received  " << tmp << std::endl;
    }
    return 0;
}

