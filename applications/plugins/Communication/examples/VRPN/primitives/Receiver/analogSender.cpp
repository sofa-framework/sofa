#include <vrpn_Connection.h>
#include <vrpn_Analog.h>

#include <stdio.h>  // fprintf()
#include <stdlib.h>  // exit()

#include <math.h>  // sin(), cos() for debugging

// Server program that opens up a port and reports over it
// any changes to the array of doubles exposed as
// vrpn_Analog_Server::channels().  The size of this array is
// defined in vrpn_Analog.h;  currently 128.
//
// init_* and do_* are sample routines - put whatever you need there.
//

void do_audio_throughput_magic (double * channels) {
  static int done = 0;

#if 1
  if (!done) {
    channels[0] = 0.0;
    done = 1;
  } else
    channels[0] += 0.5;
#else
  struct timeval now;
  vrpn_gettimeofday(&now, NULL);
  channels[0] = sin(((double) now.tv_usec) / 1000000L);
#endif
}

int main (int argc, char ** argv) {

  vrpn_Connection * c;
  vrpn_Analog_Server * ats;

  struct timeval delay;

  c = vrpn_create_server_connection();
  ats = new vrpn_Analog_Server ("testing@localhost", c);
  ats->setNumChannels(1);

  delay.tv_sec = 0L;
  delay.tv_usec = 0L;

  while (1) {
    do_audio_throughput_magic(ats->channels());
    ats->report_changes();
    c->mainloop(&delay);
    fprintf(stderr, "while():  a = %.2g\n", ats->channels()[0]);
  }
}
