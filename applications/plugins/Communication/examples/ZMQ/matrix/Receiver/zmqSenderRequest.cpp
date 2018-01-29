#include <zmq.hpp>
#include <string>
#include <iostream>
#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>

#define sleep(n)    Sleep(n)
#endif
#define MAX 1

int main () {
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    socket.bind ("tcp://*:6000");

    while (true) {
        zmq::message_t request;
        socket.recv (&request);
        std::cout << "Request" << std::endl;

        std::string mesg = "/colorLight ";
        mesg += "matrix int:3 int:1 ";
        mesg += "float:" + std::to_string(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1.0))) + " ";
        mesg += "float:" + std::to_string(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1.0))) + " ";
        mesg += "float:" + std::to_string(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1.0))) + " ";


        zmq::message_t reply (mesg.size());
        memcpy (reply.data (), mesg.c_str(), mesg.size());
        socket.send (reply);
        std::cout << "Reply" << std::endl;
        usleep(10000);
    }
    return 0;
}

