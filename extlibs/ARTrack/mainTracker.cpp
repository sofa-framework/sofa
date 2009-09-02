/* dtracklib: functions to receive and process DTrack UDP packets (ASCII protocol)
 * Copyright (C) 2000-2006, Advanced Realtime Tracking GmbH
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include <iostream>

using namespace std;
#ifndef _WIN32
#define OS_UNIX  // for Unix (Linux, Irix)
#else
#define OS_WIN   // for Windows (NT 4.0, 2000, XP)
#endif

// --------------------------------------------------------------------------

#include "dtracklib.h"

#ifdef OS_UNIX
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <unistd.h>
#endif
#ifdef OS_WIN
#include <windows.h>
#include <winsock.h>
#endif

#define DTRACKLIB_ERR_NONE       0
#define DTRACKLIB_ERR_TIMEOUT    1  
#define DTRACKLIB_ERR_UDP        2  
#define DTRACKLIB_ERR_PARSE      3  

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <string.h>
#include <time.h>

static void set_noerror(dtracklib_type* handle);
static void set_timeout(dtracklib_type* handle);
static void set_udperror(dtracklib_type* handle);
static void set_parseerror(dtracklib_type* handle);

static char* string_nextline(char* str, char* start, unsigned long len);
static char* string_get_ul(char* str, unsigned long* ul);
static char* string_get_d(char* str, double* d);
static char* string_get_f(char* str, float* f);
static char* string_get_block(char* str, const char* fmt, unsigned long* uldat, float* fdat);

static int udp_init(unsigned short port);
static int udp_exit(int sock);
static int udp_receive(int sock, void *buffer, int maxlen, unsigned long tout_us);
static int udp_send(int sock, void* buffer, int len, unsigned long ipaddr, unsigned short port, unsigned long tout_us);

static unsigned long udp_inet_atoh(char* s);

//static dtracklib_glove_type glove[MAX_NGLOVE];
//static int nglove;
//static dtracklib_marker_type marker[MAX_NMARKER];
//static int nmarker;
/*************************************************************************************/
/** fonctions reseaux **/

dtracklib_type* dtracklib_init(unsigned short udpport, char* remote_ip, unsigned short remote_port,int udpbufsize, unsigned long udptimeout_us){
	dtracklib_type* handle;

	// creat buffer for handle:

	handle = (dtracklib_type *)malloc(sizeof(dtracklib_type));

	if(!handle){return NULL;}

	set_noerror(handle);

	// creat UDP socket:

	handle->d_udpsock = udp_init(udpport);

	if(handle->d_udpsock < 0){
		free(handle);
		return NULL;
	}

	handle->d_udptimeout_us = udptimeout_us;

	// creat UDP buffer:
	handle->d_udpbufsize = udpbufsize;

	handle->d_udpbuf = (char *)malloc(udpbufsize);

	if(!handle->d_udpbuf){
		udp_exit(handle->d_udpsock);
		free(handle);
		return NULL;
	}

	// DTrack remote control parameters:
	if(remote_ip != NULL && remote_port != 0){
		handle->d_remote_ip = udp_inet_atoh(remote_ip);
		handle->d_remote_port = remote_port;
	}else{
		handle->d_remote_ip = 0;
		handle->d_remote_port = 0;
	}
	return handle;
}

// Leave the library properly:
//
// handle (i): dtracklib handle

void dtracklib_exit(dtracklib_type* handle){
	int sock;

	if(!handle){
		return;
	}

	sock = handle->d_udpsock;

	// release buffers:
	if(handle->d_udpbuf){
		free(handle->d_udpbuf);
	}

	if(handle){
		free(handle);
	}

	// release UDP socket:
	if(sock > 0){
		udp_exit(sock);
	}
}

// Check last receive/send error:
//
// handle (i): dtracklib handle
//
// return value (o): boolean (0 no, 1 yes)

int dtracklib_timeout(dtracklib_type* handle){       // 'timeout'
	return (handle->d_lasterror == DTRACKLIB_ERR_TIMEOUT) ? 1 : 0;
}

int dtracklib_udperror(dtracklib_type* handle){      // 'udp error'
	return (handle->d_lasterror == DTRACKLIB_ERR_UDP) ? 1 : 0;
}

int dtracklib_parseerror(dtracklib_type* handle){    // 'parse error'
	return (handle->d_lasterror == DTRACKLIB_ERR_PARSE) ? 1 : 0;
}

// Set last receive/send error:
//
// handle (i): dtracklib handle

static void set_noerror(dtracklib_type* handle){     // 'no error'
	handle->d_lasterror = DTRACKLIB_ERR_NONE;
}

static void set_timeout(dtracklib_type* handle){     // 'timeout'
	handle->d_lasterror = DTRACKLIB_ERR_TIMEOUT;
}

static void set_udperror(dtracklib_type* handle){    // 'udp error'
	handle->d_lasterror = DTRACKLIB_ERR_UDP;
}

static void set_parseerror(dtracklib_type* handle){  // 'parse error'
	handle->d_lasterror = DTRACKLIB_ERR_PARSE;
}


// --------------------------------------------------------------------------
// Receive and process one DTrack data packet (UDP; ASCII protocol):
//   (all pointers can be set to NULL, if its information is not wanted)
//
// handle (i): dtracklib handle
//
// framenr (o): frame counter
// timestamp (o): timestamp (-1, if information not available in packet)
//
// nbodycal (o): number of calibrated bodies (-1, if information not available in packet)
// nbody (o): number of tracked bodies
// body (o): array containing 6d data
// max_nbody (i): maximum number of bodies in array body (0 if information is not wanted)
//
// nflystick (o): number of calibrated flysticks
// flystick (o): array containing 6df data
// max_nflystick (i): maximum number of flysticks in array flystick (0 if information is not wanted)
//
// nmeatool (o): number of calibrated measurement tools
// meatool (o): array containing 6dmt data
// max_nmeatool (i): maximum number of measurement tools in array (0 if information is not wanted)
//
// nmarker (o): number of tracked single markers
// marker (o): array containing 3d data
// max_nmarker (i): maximum number of marker in array marker (0 if information is not wanted)
//
// nglove (o): number of tracked Fingertracking hands
// glove (o): array containing gl data
// max_nglove (o): maximum number of Fingertracking hands in array (0 if information is not wanted)
//
// return value (o): receiving was successfull (1 yes, 0 no)

int dtracklib_receive(dtracklib_type* handle,
					  unsigned long* framenr, double* timestamp,
					  int* nbodycal, int* nbody, dtracklib_body_type* body, int max_nbody,
					  int* nflystick, dtracklib_flystick_type* flystick, int max_nflystick,
					  int* nmeatool, dtracklib_meatool_type* meatool, int max_nmeatool,
					  int* nmarker, dtracklib_marker_type* marker, int max_nmarker,
					  int* nglove, dtracklib_glove_type* glove, int max_nglove){

						  char* s;
						  int i, j, len, n;
						  unsigned long ul, ularr[3];
						  float farr[6];

						  if(!handle){
							  return 0;
						  }

						  // Defaults:

						  if(framenr){
							  *framenr = 0;
						  }
						  if(timestamp){
							  *timestamp = -1;   // i.e. not available
						  }
						  if(nbodycal){
							  *nbodycal = -1;    // i.e. not available
						  }
						  if(nbody){
							  *nbody = 0;
						  }
						  if(nflystick){
							  *nflystick = 0;
						  }
						  if(nmeatool){
							  *nmeatool = 0;
						  }
						  if(nmarker){
							  *nmarker = 0;
						  }
						  if(nglove){
							  *nglove = 0;
						  }

						  // Receive UDP packet:
						  len = udp_receive(handle->d_udpsock, handle->d_udpbuf, handle->d_udpbufsize-1, handle->d_udptimeout_us);

						  if(len == -1){
							  set_timeout(handle);
							  return 0;
						  }
						  if(len <= 0){
							  set_udperror(handle);
							  return 0;
						  }

						  s = handle->d_udpbuf;
						  s[len] = '\0';

						  // Process lines:

						  set_parseerror(handle);

						  do{
							  // Line for frame counter:
							  if(!strncmp(s, "fr ", 3)){
								  s += 3;

								  if(framenr){
									  if(!(s = string_get_ul(s, framenr))){       // get frame counter
										  *framenr = 0;
										  return 0;
									  }
								  }
								  continue;
							  }

							  // Line for timestamp:

							  if(!strncmp(s, "ts ", 3)){
								  s += 3;

								  if(timestamp){
									  if(!(s = string_get_d(s, timestamp))){      // get timestamp
										  *timestamp = 0;
										  return 0;
									  }
								  }
								  continue;
							  }

							  // Line for additional information about number of calibrated bodies:

							  if(!strncmp(s, "6dcal ", 6)){
								  s += 6;

								  if(nbodycal){
									  if(!(s = string_get_ul(s, &ul))){            // get number of bodies
										  return 0;
									  }

									  *nbodycal = (int )ul;
								  }
								  continue;
							  }

							  // Line for 6d data:
							  if(!strncmp(s, "6d ", 3)){
								  s += 3;

								  if(nbody && body && max_nbody > 0){
									  if(!(s = string_get_ul(s, &ul))){            // get number of bodies
										  return 0;
									  }

									  *nbody = n = (int )ul;
									  if(n > max_nbody){
										  n = max_nbody;
									  }

									  for(i=0; i<n; i++){                          // get data of body
										  if(!(s = string_get_block(s, "uf", &body[i].id, &body[i].quality))){
											  return 0;
										  }

										  if(!(s = string_get_block(s, "ffffff", NULL, farr))){
											  return 0;
										  }
										  for(j=0; j<3; j++){
											  body[i].loc[j] = farr[j];
											  body[i].ang[j] = farr[j+3];
										  }

										  if(!(s = string_get_block(s, "fffffffff", NULL, body[i].rot))){
											  return 0;
										  }
									  }
								  }

								  continue;
							  }

							  // Line for flystick data:
							  if(!strncmp(s, "6df ", 4)){
								  s += 4;

								  if(nflystick && flystick && max_nflystick > 0){
									  if(!(s = string_get_ul(s, &ul))){            // get number of flysticks
										  return 0;
									  }

									  *nflystick = n = (int )ul;
									  if(n > max_nflystick){
										  n = max_nflystick;
									  }

									  for(i=0; i<n; i++){                          // get data of body
										  if(!(s = string_get_block(s, "ufu", ularr, &flystick[i].quality))){
											  return 0;
										  }

										  flystick[i].id = ularr[0];
										  flystick[i].bt = ularr[1];

										  if(!(s = string_get_block(s, "ffffff", NULL, farr))){
											  return 0;
										  }
										  for(j=0; j<3; j++){
											  flystick[i].loc[j] = farr[j];
											  flystick[i].ang[j] = farr[j+3];
										  }

										  if(!(s = string_get_block(s, "fffffffff", NULL, flystick[i].rot))){
											  return 0;
										  }
									  }
								  }

								  continue;
							  }

							  // Line for measurement tool data:
							  if(!strncmp(s, "6dmt ", 5)){
								  s += 5;

								  if(nmeatool && meatool && max_nmeatool > 0){
									  if(!(s = string_get_ul(s, &ul))){            // get number of measurement tools
										  return 0;
									  }

									  *nmeatool = n = (int )ul;
									  if(n > max_nmeatool){
										  n = max_nmeatool;
									  }

									  for(i=0; i<n; i++){                          // get data of body
										  if(!(s = string_get_block(s, "ufu", ularr, &meatool[i].quality))){
											  return 0;
										  }
										  meatool[i].id = ularr[0];
										  meatool[i].bt = ularr[1];

										  if(!(s = string_get_block(s, "fff", NULL, meatool[i].loc))){
											  return 0;
										  }

										  if(!(s = string_get_block(s, "fffffffff", NULL, meatool[i].rot))){
											  return 0;
										  }
									  }
								  }

								  continue;
							  }

							  // Line for single markers:
							  if(!strncmp(s, "3d ", 3)){
								  s += 3;

								  if(nmarker && marker && max_nmarker > 0){
									  if(!(s = string_get_ul(s, &ul))){            // get number of markers
										  return 0;
									  }

									  *nmarker = n = (int )ul;
									  if(n > max_nmarker){
										  n = max_nmarker;
									  }

									  for(i=0; i<n; i++){                          // get marker data
										  if(!(s = string_get_block(s, "uf", &marker[i].id, &marker[i].quality))){
											  return 0;
										  }

										  if(!(s = string_get_block(s, "fff", NULL, marker[i].loc))){
											  return 0;
										  }
									  }
								  }

								  continue;
							  }

							  // Line for A.R.T. Fingertracking hands:
							  if(!strncmp(s, "gl ", 3)){
								  s += 3;

								  if(nglove && glove && max_nglove > 0){
									  if(!(s = string_get_ul(s, &ul))){            // get number of Fingertracking hands
										  return 0;
									  }

									  *nglove = n = (int )ul;
									  if(n > max_nglove){
										  n = max_nglove;
									  }

									  for(i=0; i<n; i++){                          // get data for a hand
										  if(!(s = string_get_block(s, "ufuu", ularr, &glove[i].quality))){
											  return 0;
										  }
										  glove[i].id = ularr[0];
										  glove[i].lr = (int )ularr[1];
										  glove[i].nfinger = (int )ularr[2];

										  if(!(s = string_get_block(s, "fff", NULL, glove[i].loc))){
											  return 0;
										  }

										  if(!(s = string_get_block(s, "fffffffff", NULL, glove[i].rot))){
											  return 0;
										  }

										  for(j=0; j<glove[i].nfinger; j++){        // get data for a finger
											  if(!(s = string_get_block(s, "fff", NULL, glove[i].finger[j].loc))){
												  return 0;
											  }

											  if(!(s = string_get_block(s, "fffffffff", NULL, glove[i].finger[j].rot))){
												  return 0;
											  }

											  if(!(s = string_get_block(s, "ffffff", NULL, farr))){
												  return 0;
											  }

											  glove[i].finger[j].radiustip = farr[0];
											  glove[i].finger[j].lengthphalanx[0] = farr[1];
											  glove[i].finger[j].lengthphalanx[1] = farr[3];
											  glove[i].finger[j].lengthphalanx[2] = farr[5];
											  glove[i].finger[j].anglephalanx[0] = farr[2];
											  glove[i].finger[j].anglephalanx[1] = farr[4];
										  }
									  }
								  }

								  continue;
							  }

							  // ignore unknown line identifiers (could be valid in future DTracks)

						  }while((s = string_nextline(handle->d_udpbuf, s, handle->d_udpbufsize)));

						  set_noerror(handle);
						  return 1;
}

// ---------------------------------------------------------------------------------------------------
// Send one remote control command (UDP; ASCII protocol) to DTrack:
//
// handle (i): dtracklib handle
//
// cmd (i): command code
// val (i): additional value (if needed)
//
// return value (o): sending was successfull (1 yes, 0 no)

int dtracklib_send(dtracklib_type* handle, unsigned short cmd, int val){
	char cmdstr[100];

	if(!handle){
		return 0;
	}

	if(!handle->d_remote_ip || !handle->d_remote_port){
		return 0;
	}

	// process command code:

	switch(cmd){
		case DTRACKLIB_CMD_CAMERAS_OFF:
			strcpy(cmdstr, "dtrack 10 0");
			break;

		case DTRACKLIB_CMD_CAMERAS_ON:
			strcpy(cmdstr, "dtrack 10 1");
			break;

		case DTRACKLIB_CMD_CAMERAS_AND_CALC_ON:
			strcpy(cmdstr, "dtrack 10 3");
			break;

		case DTRACKLIB_CMD_SEND_DATA:
			strcpy(cmdstr, "dtrack 31");
			break;

		case DTRACKLIB_CMD_STOP_DATA:
			strcpy(cmdstr, "dtrack 32");
			break;

		case DTRACKLIB_CMD_SEND_N_DATA:
			sprintf(cmdstr, "dtrack 33 %d", val);
			break;

		default:
			return 0;
	}

	// send UDP packet:
	if(udp_send(handle->d_udpsock, cmdstr, (int )strlen(cmdstr) + 1,
		handle->d_remote_ip, handle->d_remote_port, handle->d_udptimeout_us))
	{
		set_udperror(handle);
		return 0;
	}

	if(cmd == DTRACKLIB_CMD_CAMERAS_AND_CALC_ON){
#ifdef OS_UNIX
		sleep(1);     // some delay (actually only necessary for older DTrack versions...)
#endif
#ifdef OS_WIN
		Sleep(1000);  // some delay (actually only necessary for older DTrack versions...)
#endif
	}

	set_noerror(handle);
	return 1;
}

// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// Parsing DTrack data:

// Search next line in buffer:
// str (i): buffer (total)
// start (i): start position within buffer
// len (i): buffer length in bytes
// return (i): begin of line, NULL if no new line in buffer

static char* string_nextline(char* str, char* start, unsigned long len){
	char* s = start;
	char* se = str + len;
	int crlffound = 0;

	while(s < se){
		if(*s == '\r' || *s == '\n'){  // crlf
			crlffound = 1;
		}else{
			if(crlffound){              // begin of new line found
				return (*s) ? s : NULL;  // first character is '\0': end of buffer
			}
		}
		s++;
	}
	return NULL;                      // no new line found in buffer
}


// Read next 'unsigned long' value from string:
// str (i): string
// ul (o): read value
// return value (o): pointer behind read value in str; NULL in case of error

static char* string_get_ul(char* str, unsigned long* ul){
	char* s;

	*ul = strtoul(str, &s, 0);
	return (s == str) ? NULL : s;
}


// Read next 'double' value from string:
// str (i): string
// d (o): read value
// return value (o): pointer behind read value in str; NULL in case of error

static char* string_get_d(char* str, double* d){
	char* s;

	*d = strtod(str, &s);
	return (s == str) ? NULL : s;
}


// Read next 'float' value from string:
// str (i): string
// f (o): read value
// return value (o): pointer behind read value in str; NULL in case of error

static char* string_get_f(char* str, float* f){
	char* s;

	*f = (float )strtod(str, &s);   // strtof() only available in GNU-C
	return (s == str) ? NULL : s;
}


// Process next block '[...]' in string:
// str (i): string
// fmt (i): format string ('u' for 'unsigned long', 'f' for 'float')
// uldat (o): array for 'unsigned long' values (long enough due to fmt)
// fdat (o): array for 'float' values (long enough due to fmt)
// return value (o): pointer behind read value in str; NULL in case of error

static char* string_get_block(char* str, const char* fmt, unsigned long* uldat, float* fdat){
	char* strend;
	int index_ul, index_f;

	if(!(str = strchr(str, '['))){       // search begin of block
		return NULL;
	}
	if(!(strend = strchr(str, ']'))){    // search end of block
		return NULL;
	}

	str++;                               // remove delimiters
	*strend = '\0';

	index_ul = index_f = 0;

	while(*fmt){
		switch(*fmt++){
			case 'u':
				if(!(str = string_get_ul(str, &uldat[index_ul++]))){
					*strend = ']';
					return NULL;
				}
				break;

			case 'f':
				if(!(str = string_get_f(str, &fdat[index_f++]))){
					*strend = ']';
					return NULL;
				}
				break;

			default:    // unknown format character
				*strend = ']';
				return NULL;
		}
	}

	// ignore additional data inside the block

	*strend = ']';
	return strend + 1;
}

// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// Handling UDP data:

// Initialize UDP socket:
// port (i): port number
// return value (o): socket number, <0 if error

static int udp_init(unsigned short port){
	int sock;
	struct sockaddr_in name;

	// initialize socket dll (only Windows):

#ifdef OS_WIN
	{
		WORD vreq;
		WSADATA wsa;

		vreq = MAKEWORD(2, 0);

		if(WSAStartup(vreq, &wsa) != 0){
			return -1;
		}
	}
#endif

	// create socket:
	sock = (int )socket(PF_INET, SOCK_DGRAM, 0);

	if(sock < 0){
#ifdef OS_WIN
		WSACleanup();
#endif
		return -2;
	}

	// name socket:

	name.sin_family = AF_INET;
	name.sin_port = htons(port);
	name.sin_addr.s_addr = htonl(INADDR_ANY);

	if(bind(sock, (struct sockaddr *) &name, sizeof(name)) < 0){
		udp_exit(sock);
		return -3;
	}
	return sock;
}


// Deinitialize UDP socket:
// sock (i): socket number
// return value (o): 0 ok, -1 error

static int udp_exit(int sock){
	int err;

#ifdef OS_UNIX
	err = close(sock);
#endif

#ifdef OS_WIN
	err = closesocket(sock);
	WSACleanup();
#endif
	if(err < 0){
		return -1;
	}
	return 0;
}


// Receive UDP data:
//   - tries to receive packets, as long as data are available
// sock (i): socket number
// buffer (o): buffer for UDP data
// maxlen (i): length of buffer
// tout_us (i): timeout in us (micro sec)
// return value (o): number of received bytes, <0 if error/timeout occured
static int udp_receive(int sock, void *buffer, int maxlen, unsigned long tout_us){
	int nbytes, err;
	fd_set set;
	struct timeval tout;

	// waiting for data:

	FD_ZERO(&set);
	FD_SET(sock, &set);

	tout.tv_sec = tout_us / 1000000;
	tout.tv_usec = tout_us % 1000000;

	switch((err = select(FD_SETSIZE, &set, NULL, NULL, &tout))){
		case 1:
			break;        // data available
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}

	// receiving packet:

	while(1){

		// receive one packet:
		nbytes = recv(sock, (char *)buffer, maxlen, 0);

		if(nbytes < 0){  // receive error
			return -3;
		}

		// check, if more data available: if so, receive another packet
		FD_ZERO(&set);
		FD_SET(sock, &set);

		tout.tv_sec = 0;   // no timeout
		tout.tv_usec = 0;

		if(select(FD_SETSIZE, &set, NULL, NULL, &tout) != 1){
			// no more data available: check length of received packet and return
			if(nbytes >= maxlen){   // buffer overflow
				return -4;
			}
			return nbytes;
		}
	}
}

// Send UDP data:
// sock (i): socket number
// buffer (i): buffer for UDP data
// len (i): length of buffer
// ipaddr (i): IP address to send to
// port (i): port number to send to
// tout_us (i): timeout in us (micro sec)
// return value (o): 0 if ok, <0 if error/timeout occured
static int udp_send(int sock, void* buffer, int len, unsigned long ipaddr, unsigned short port, unsigned long tout_us){
	fd_set set;
	struct timeval tout;
	int nbytes, err;
	struct sockaddr_in addr;

	// building address:
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(ipaddr);
	addr.sin_port = htons(port);

	// waiting to send data:

	FD_ZERO(&set);
	FD_SET(sock, &set);

	tout.tv_sec = tout_us / 1000000;
	tout.tv_usec = tout_us % 1000000;

	switch((err = select(FD_SETSIZE, NULL, &set, NULL, &tout))){
		case 1:
			break;
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}

	// sending data:
	nbytes = sendto(sock, (char* )buffer, len, 0, (struct sockaddr* )&addr, (size_t )sizeof(struct sockaddr_in));

	if(nbytes < len){  // send error
		return -3;
	}
	return 0;
}

static unsigned long udp_inet_atoh(char* s){
	int i, a[4];
	char* s1;
	unsigned long ret;

	s1 = s;
	while(*s1){
		if(*s1 == '.'){
			*s1 = ' ';
		}
		s1++;
	}

	if(sscanf(s, "%d %d %d %d", &a[0], &a[1], &a[2], &a[3]) != 4){
		return 0;
	}

	ret = 0;
	for(i=0; i<4; i++){
		if(a[i] < 0 || a[i] > 255){
			return 0;
		}

		ret = (ret << 8) | (unsigned char)a[i];
	}
	return ret;
}
