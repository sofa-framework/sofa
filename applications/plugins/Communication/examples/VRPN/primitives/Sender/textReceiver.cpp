#include "vrpn_Analog.h"
#include <vrpn_Text.h> 

#include <iostream>
using namespace std;


void VRPN_CALLBACK handle_text(void *userdata, const vrpn_TEXTCB t)
{
    const char *name = (const char *)userdata;

    // Warnings and errors are printed by the system text printer.
    if (t.type == vrpn_TEXT_NORMAL) {
        printf("%s: Text message: %s\n", name, t.message);
    }
}

int main(int argc, char* argv[])
{   
    //To recieve text from server
    vrpn_Text_Receiver *text = new vrpn_Text_Receiver("testing@localhost");

    text->register_message_handler( 0, handle_text );

    while(1)
    {
        text->mainloop();
    }

    return 0;
}