#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <sys/types.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/time.h>
#endif
#include <vrpn_Connection.h>
#include "vrpn_Text.h"
#include "vrpn_Analog.h"

#define MAX 1024

int main ()
{
	char msg[MAX];
	vrpn_Connection *sc = vrpn_create_server_connection();
	vrpn_Text_Sender *s = new vrpn_Text_Sender("testing@localhost", sc);

	while (1) {
		while (!sc->connected()) {  // wait until we've got a connection
			sc->mainloop();
			std::cout<<"WAITING CONNECTION"<<std::endl;
			sleep(1);
		}
		while (sc->connected()) {
			std::string msg= "SENDER WORKING";
			s->send_message(msg.c_str(), vrpn_TEXT_NORMAL);
			sc->mainloop();
			std::cout<<std::endl;
		}
	}
}