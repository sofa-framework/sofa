/* dtracklib: C header file, A.R.T. GmbH 25.8.00-4.10.06
 *
 * dtracklib: functions to receive and process DTrack UDP packets (ASCII protocol)
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
 *
 * Version v1.2.2
 *
 * $Id: dtracklib.h,v 1.10.2.1 2006/10/04 13:37:07 kurt Exp $
 */

#ifndef _ART_DTRACKLIB_H
#define _ART_DTRACKLIB_H

#ifdef __cplusplus
extern "C"{
#endif

// --------------------------------------------------------------------------
// Data types and macros

// Body data (6DOF):

typedef struct{
	unsigned long id;     // id number
	float quality;        // quality (0 <= qu <= 1)
	
	float loc[3];         // location (in mm)
	float ang[3];         // orientation angles (eta, theta, phi; in deg)
	float rot[9];         // rotation matrix (column-wise)
} dtracklib_body_type;

// A.R.T. FlyStick data (6DOF + buttons):

typedef struct{
	unsigned long id;     // id number
	float quality;        // quality (0 <= qu <= 1, no tracking if -1)
	unsigned long bt;     // pressed buttons (binary coded)
	
	float loc[3];         // location (in mm)
	float ang[3];         // orientation angles (eta, theta, phi; in deg)
	float rot[9];         // rotation matrix (column-wise)
} dtracklib_flystick_type;

// Measurement tool data (6DOF + buttons):

typedef struct{
	unsigned long id;     // id number
	float quality;        // quality (0 <= qu <= 1, no tracking if -1)
	unsigned long bt;     // pressed buttons (binary coded)
	
	float loc[3];         // location (in mm)
	float rot[9];         // rotation matrix (column-wise)
} dtracklib_meatool_type;

// Single marker data (3DOF):

typedef struct{
	unsigned long id;     // id number
	float quality;        // quality (0 <= qu <= 1)
	
	float loc[3];         // location (in mm)
} dtracklib_marker_type;

// A.R.T. Fingertracking hand (6DOF + fingers):

typedef struct{
	unsigned long id;     // id number
	float quality;        // quality (0 <= qu <= 1)
	int lr;               // left (0) or right (1) hand
	int nfinger;          // number of fingers

	float loc[3];         // back of the hand: location (in mm) : position du gant (point d origine de la direction)
	float rot[9];         // back of the hand: rotation matrix (column-wise) 

	struct{
		float loc[3];            // finger: location (in mm)
		float rot[9];            // finger: rotation matrix (column-wise) 
		
		float radiustip;         // finger: radius of tip
		float lengthphalanx[3];  // finger: length of phalanxes; order: outermost, middle, innermost
		float anglephalanx[2];   // finger: angle between phalanxes
	} finger[5];                // order: thumb, index finger, ... (index: point d arriv�e de la direction)
} dtracklib_glove_type;


// DTrack remote commands:

#define DTRACKLIB_CMD_CAMERAS_OFF           0x1000
#define DTRACKLIB_CMD_CAMERAS_ON            0x1001
#define DTRACKLIB_CMD_CAMERAS_AND_CALC_ON   0x1003

#define DTRACKLIB_CMD_SEND_DATA             0x3100
#define DTRACKLIB_CMD_STOP_DATA             0x3200
#define DTRACKLIB_CMD_SEND_N_DATA           0x3300


// Type for dtracklib handles (don't change elements!):

typedef struct{
	int d_udpsock;                  // socket number for UDP
	unsigned long d_udptimeout_us;  // timeout for receiving and sending

	int d_udpbufsize;               // size of UDP buffer
	char* d_udpbuf;                 // UDP buffer

	unsigned long d_remote_ip;      // DTrack remote command access: IP address
	unsigned short d_remote_port;   // DTrack remote command access: port number

	int d_lasterror;                // last receive/send error
} dtracklib_type;


// --------------------------------------------------------------------------
// Library routines

// Initialize the library:
//
// udpport (i): UDP port number to receive data from DTrack
//
// remote_ip (i): DTrack remote control: ip address of DTrack PC (NULL if not used)
// remote_port (i): port number of DTrack remote control (0 if not used)
//
// udpbufsize (i): size of buffer for UDP packets (in bytes)
// udptimeout_us (i): UDP timeout (receiving and sending) in us (micro sec)
//
// return value (o): dtracklib handle; NULL if error occured

dtracklib_type* dtracklib_init(
	unsigned short udpport, char* remote_ip, unsigned short remote_port,
	int udpbufsize, unsigned long udptimeout_us
);

// Leave the library properly:
//
// handle (i): dtracklib handle

void dtracklib_exit(dtracklib_type* handle);


// Check last receive/send error:
//
// handle (i): dtracklib handle
//
// return value (o): boolean (0 no, 1 yes)

int dtracklib_timeout(dtracklib_type* handle);     // 'timeout'
int dtracklib_udperror(dtracklib_type* handle);    // 'udp error'
int dtracklib_parseerror(dtracklib_type* handle);  // 'parse error'


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
	int* nglove, dtracklib_glove_type* glove, int max_nglove
);


// Send one remote control command (UDP; ASCII protocol) to DTrack:
//
// handle (i): dtracklib handle
//
// cmd (i): command code
// val (i): additional value (if needed)
//
// return value (o): sending was successfull (1 yes, 0 no)

int dtracklib_send(dtracklib_type* handle, unsigned short cmd, int val);


// ---------------------------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // _ART_DTRACKLIB_H


