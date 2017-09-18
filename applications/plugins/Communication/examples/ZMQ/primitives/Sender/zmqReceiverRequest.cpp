#include <zmq.hpp>
#include <string>
#include <iostream>
#include <unistd.h>

int main ()
{
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REQ);
    socket.connect ("tcp://localhost:6000");

    while (1)
    {
        zmq::message_t request (5);
        memcpy (request.data (), "", 5);
        std::cout << "Request" << std::endl;
        socket.send (request);

        zmq::message_t reply;
        socket.recv (&reply);
        std::cout << "Reply : " << (char*)reply.data() << std::endl;

        usleep(100000);
    }
    return 0;
}

