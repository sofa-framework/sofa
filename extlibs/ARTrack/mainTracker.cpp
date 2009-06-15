/******************************************************************************/
/**                                                                          **/ 
/**                      DAVID LOPES BARATA                                  **/
/**                      ROMAIN MEUNIER                                      **/
/**                                                                          **/
/**                                                                          **/
/**                                                                          **/
/**                             PEINTURE 3D                                  **/
/**                                                                          **/ 
/**                                                                          **/
/**                                                                          **/
/**           Version du 26/07/07 10h45                                      **/
/**                                                                          **/
/******************************************************************************/

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

#include <GL/glut.h>
#include <GL/glu.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <string.h>
#include<time.h>

#ifndef GLUT_WHEEL_UP
#define GLUT_WHEEL_DOWN     4
#define GLUT_WHEEL_UP       3
#endif

#define nb_Points           100000
#define BUFSIZE 512

/* prototypes pour les fonctions reseaux **/
static void set_noerror(dtracklib_type* handle);
static void set_timeout(dtracklib_type* handle);
static void set_udperror(dtracklib_type* handle);
static void set_parseerror(dtracklib_type* handle);

static char* string_nextline(char* str, char* start, unsigned long len);
static char* string_get_ul(char* str, unsigned long* ul);
static char* string_get_d(char* str, double* d);
static char* string_get_f(char* str, float* f);
static char* string_get_block(char* str, char* fmt, unsigned long* uldat, float* fdat);

static int udp_init(unsigned short port);
static int udp_exit(int sock);
static int udp_receive(int sock, void *buffer, int maxlen, unsigned long tout_us);
static int udp_send(int sock, void* buffer, int len, unsigned long ipaddr, unsigned short port, unsigned long tout_us);

static unsigned long udp_inet_atoh(char* s);

GLfloat ROUGE[]={1.0f,0.0f,0.0f,1.0f};
GLfloat VERT[]={0.0f,1.0f,0.0f,1.0f};
GLfloat BLEU[]={0.0f,0.0f,1.0f,1.0f};
GLfloat NOIR[]={0.0f,0.0f,0.0f,1.0f};
GLfloat BLANC[]={1.0f,1.0f,1.0f,1.0f};
GLfloat JAUNE[]={1.0f,1.0f,0.0f,1.0f};
GLfloat ROSE[]={1.0f,0.0f,1.0f,1.0f};
GLfloat GRIS[]={0.4f,0.4f,0.4f,1.0f};
GLfloat ORANGE[]={1.0f,0.4f,0.0f,1.0f};
GLfloat VIOLET[]={0.6f,0.0f,0.6f,1.0f};
GLfloat BLEU_CLAIR[]={0.4f,0.7f,1.0f,1.0f};
GLfloat VERT_FONCE[]={0.0f,0.3f,0.0f,1.0f};
GLfloat JAUNE_PALE[]={1.0f,1.0f,0.5f,1.0f};
GLfloat MARON[]={0.7f,0.4f,0.2f,1.0f};
GLfloat BLEU_GRIS[]={0.5f,0.5f,1.0f,1.0f};
GLfloat VIOLET_PALE[]={0.7f,0.5f,0.9f,1.0f};
GLfloat BORDEAUX[]={0.6f,0.0f,0.0f,1.0f};
GLfloat ROUGE_PALE[]={1.0f,0.5f,0.5f,1.0f};
GLfloat GRIS_CLAIR[]={0.6f,0.6f,0.6f,1.0f};
GLfloat BLEU_FONCE[]={0.0f,0.0f,0.4f,1.0f};
GLfloat JAUNE_FONCE[]={0.5f,0.5f,0.0f,1.0f};
GLfloat VIOLET_FONCE[]={0.3f,0.0f,0.3f,1.0f};
GLfloat ORANGE_JAUNE[]={1.0f,0.7f,0.0f,1.0f};
GLfloat BLEU_CIEL[]={0.08f,1.0f,1.0f,1.0f};
GLfloat VERT_PALE[]={0.49f,1.0f,0.41f,1.0f};
GLfloat ROSE_PALE[]={0.95f,0.69f,1.0f,1.0f};

const GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { -1.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };

int Max_pts_par_trait;           
float positionnement,rapport_X;
int test_si_point_suiv;

float Xmax_GL,Ymin_GL,Zmax_GL_moitie; /** constantes correspondant aux valeurs maximum respectivement sur les X ,Y et Z de la fenetre openGL**/

float largeur;	//width of the brush
int var_largeur;
float seuil_clic;
int cpt_tours=0;
bool clic_temp=false;
int repere_gant;
float K=.01;

FILE * config;
/* variables pour les boutons */
int couleur=0,test_menu_couleur=0; int boule_couleur_appuye[5][5]; int cpt_boule_couleur;
int epaisseur=0,test_menu_epaisseur=0; int trait_appuye[4]; int cpt_trait;
int reset=0;
/* couleur courante (par d�faut blanc)*/

typedef struct Point{
	float X; 
	float Y;
	float Z;       
	int relie_prec; /** booleen qui indique ,lors du trac, si l' utilisateur a lev� le pinceau entre 2 points */
} Point;

typedef struct{
	GLubyte	*imageData;									
	GLuint	bpp;											
	GLuint	width;											
	GLuint	height;										
	GLuint	texID;										
} TextureImage;												

TextureImage textures[25];
TextureImage text_menu[3],text_select[2],text_menus[1],text_boutons[3];
TextureImage text_temp[150];
int num_text=3;					//what is the current texture (with is associated color in fact)
int max_text_temp=0;
int recharge_peinture=1;

Point ZERO;
Point X1={1.0f,0.0f,0.0f,0};  
Point Y1={0.0f,1.0f,0.0f,0};  
Point Z1={0.0f,0.0f,1.0f,0};     
Point P_inter_menu;
Point P_inter_menu_coul_et_ep;

Point Tab_Points[nb_Points];

int taille_tab_p=0; //total of point. One point at one position inside the Points array is the at the same position
//in the array holding the textures
int cpt_pts=0;       //HOW MANY POINTS IN THE CURRENT SWOOP (to know if we have more point in this swoop than the maximum number)
int cpt_tab_coul=0;

/* pour chaque point on stocke son num de sa texture couleur , l'�paisseur et s'il doit apparaitre en fonction de recharge peinture*/
float Tab_coul[nb_Points][3];

int dessiner=0;
float xprec,yprec;
float xprec_reel;

GLfloat trX,trY,trZ;

GLUquadricObj	*obj;
GLfloat width,height,rap;

int deplacement_z;
// bouton poussoir
int cpt_bouton_poussoir1;
int cpt_bouton_poussoir2;
int cpt_bouton_poussoir3;

// position de la paume main et de l index en positions dans le repere absolu et dans le repere openGL
Point paume_main,i,paume_main_GL,index_GL;

int tab_cpt_pts[100];
int T=0;	//??? number to know how much paint can be drawn in one swoop of the brush (pas s�r)
int test_T=0;

bool anim_contour=1;

float trandom(){
	int a;
	float b;

	a=rand()%101; 
	b=(float)a/100;

	return b;
}

float gli[9];

dtracklib_type* handle;
static unsigned long framenr;
static double timestamp;
static dtracklib_glove_type glove[MAX_NGLOVE];
static int nglove;
static dtracklib_marker_type marker[MAX_NMARKER];
static int nmarker;
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

static char* string_get_block(char* str, char* fmt, unsigned long* uldat, float* fdat){
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

/*********************************************************************/
/** fonctions openGL    **/

/**
 * Load the TGA texture from file
 */
bool LoadTGA(TextureImage *texture, char *filename){    
	GLubyte		TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};	
	GLubyte		TGAcompare[12];								
	GLubyte		header[6];								
	GLuint		bytesPerPixel;							
	GLuint		imageSize;								
	GLuint		temp;									
	GLuint		type=GL_RGBA;							

	FILE *file = fopen(filename, "rb");		

	if(	file==NULL ||									
		fread(TGAcompare,1,sizeof(TGAcompare),file)!=sizeof(TGAcompare) ||
		memcmp(TGAheader,TGAcompare,sizeof(TGAheader))!=0				||	
		fread(header,1,sizeof(header),file)!=sizeof(header)){
		if (file == NULL)								
			return false;								
		else{
			fclose(file);									
			return false;								
		}
	}

	texture->width  = header[1] * 256 + header[0];		
	texture->height = header[3] * 256 + header[2];		

	if(	texture->width	<=0	|| texture->height	<=0	||(header[4]!=24 && header[4]!=32)){
		fclose(file);										
		return false;										
	}

	texture->bpp	= header[4];							
	bytesPerPixel	= texture->bpp/8;						
	imageSize		= texture->width*texture->height*bytesPerPixel;	

	texture->imageData=(GLubyte *)malloc(imageSize);	

	if(	texture->imageData==NULL || fread(texture->imageData, 1, imageSize, file)!=imageSize){
			if(texture->imageData!=NULL)					
				free(texture->imageData);						

			fclose(file);									
			return false;										
	}

	for(GLuint i=0; i<int(imageSize); i+=bytesPerPixel){														
		temp=texture->imageData[i];							
		texture->imageData[i] = texture->imageData[i + 2];	
		texture->imageData[i + 2] = temp;					
	}

	fclose (file);										

	glGenTextures(1, &texture[0].texID);					

	glBindTexture(GL_TEXTURE_2D, texture[0].texID);		
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);	
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);	

	if (texture[0].bpp==24){
		type=GL_RGB;									
	}

	glTexImage2D(GL_TEXTURE_2D, 0, type, texture[0].width, texture[0].height, 0, type, GL_UNSIGNED_BYTE, texture[0].imageData);

	return true;										
}

/**
 * Draw the color selection menu on screen
 */
void menu_couleur(){

	bool contour=0;
	glClearDepth(1.0f);										
	glDepthFunc(GL_LEQUAL);									
	glEnable(GL_DEPTH_TEST);									

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
	int i,j;
	glBindTexture(GL_TEXTURE_2D,text_menus[0].texID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_menus[0].width, text_menus[0].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_menus[0].imageData);
	glDisable(GL_LIGHTING);

	glPushMatrix();{  

		glPushMatrix();{
			glScalef(5.0,5.0,3.0);
			glPushMatrix();{

				glTranslatef(-.75,-.75,.3f);
				/** face de devant **/
				glPushMatrix();{
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,1.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(1.5f,1.5f);
						glTexCoord2f(1.0,.0);	glVertex2f(1.5f,.0f); 
					}glEnd();
				}glPopMatrix();
				/** face de derriere **/

				glPushMatrix();{
					glTranslatef(.0,.0,-.4f);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,1.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(1.5f,1.5f);
						glTexCoord2f(1.0,.0);	glVertex2f(1.5f,.0f); 
					}glEnd();
				}glPopMatrix();

				/** face de gauche**/
				glPushMatrix();{
					glTranslatef(1.5f,.0,.0f);
					glRotatef(90,.0,1.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,1.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.4f,1.5f);
						glTexCoord2f(1.0,.0);	glVertex2f(.4f,.0f); 
					}glEnd();
				}glPopMatrix();

				/** face de droite**/
				glPushMatrix();{
					glRotatef(90,.0,1.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,1.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.4f,1.5f);
						glTexCoord2f(1.0,.0);	glVertex2f(.4f,.0f); 
					}glEnd();
				}glPopMatrix();
			}glPopMatrix();
		}glPopMatrix();
		glDisable(GL_TEXTURE_2D);
		glEnable(GL_LIGHTING);
		glPushMatrix();{
			glScalef(5.0,5.0,5.0);
			glTranslatef(-.60f,.90f,.20f);

			for (i=0;i<=4;i++){
				glTranslatef(.0f,-.30f,.0f);
				glPushMatrix();{
					for (j=0;j<=4;j++){
						switch(j) {//display the various colors on the spheres
							case 0: switch (i) {
								case 0 : glColor4fv(ROUGE);break;
								case 1 : glColor4fv(VERT);break;
								case 2 : glColor4fv(BLEU); break;
								case 3 : glColor4fv(BLANC); break;
								case 4 : glColor4fv(JAUNE); break;
							} break;
							case 1: switch (i) {
								case 0 : glColor4fv(BLEU_CLAIR); break;
								case 1 : glColor4fv(VIOLET); break;
								case 2 : glColor4fv(VERT_FONCE); break;
								case 3 : glColor4fv(ROSE); break;
								case 4 : glColor4fv(ORANGE); break;
							}break;
							case 2: switch (i) {
								case 0 : glColor4fv(GRIS); break;
								case 1 : glColor4fv(MARON); break;
								case 2 : glColor4fv(JAUNE_PALE); break;
								case 3 : glColor4fv(BLEU_GRIS); break;
								case 4 : glColor4fv(VIOLET_PALE); break;
							}break;
							case 3: switch (i) {
								case 0 : glColor4fv(BORDEAUX); break;
								case 1 : glColor4fv(ROUGE_PALE); break;
								case 2 : glColor4fv(GRIS_CLAIR); break;
								case 3 : glColor4fv(BLEU_FONCE); break;
								case 4 : glColor4fv(JAUNE_FONCE); break;
							}break;
							case 4: switch (i) {
								case 0 : glColor4fv(VIOLET_FONCE); break;
								case 1 : glColor4fv(ORANGE_JAUNE); break;
								case 2 : glColor4fv(BLEU_CIEL); break;
								case 3 : glColor4fv(VERT_PALE); break;
								case 4 : glColor4fv(ROSE_PALE); break;
							}break;
						}

						glPushMatrix();{
							if(boule_couleur_appuye[i][j]){
								if(cpt_boule_couleur<15){
									glTranslatef(0.0f,0.0f,-0.07f);
									cpt_boule_couleur++;
								}else{
									if(cpt_boule_couleur<25) cpt_boule_couleur++; 
									else{
										cpt_boule_couleur=0;
										boule_couleur_appuye[i][j]=0;
										test_menu_couleur=0;
									}
								}
							}

							glutSolidSphere(.1,50,50);

						}glPopMatrix();
						glTranslatef(.3f,0.0f,0.0f);
					}
				}glPopMatrix();
			}
		}glPopMatrix();
	}glPopMatrix();

	glEnable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_BLEND);
	glutPostRedisplay();    

}

/**
 * Draw the brush width menu on screen
 * The brushes have the color selected by the user on the color selection menu
 */ 
void menu_epaisseur(){

	glClearDepth(1.0f);										
	glDepthFunc(GL_LEQUAL);									
	glEnable(GL_DEPTH_TEST);									

	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	//la couleur de l'objet va �tre (1-alpha_de_l_objet) * couleur du fond et (le_reste * couleur originale) 
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

	int i,temp;
	float taille=0.05;
	glDisable(GL_LIGHTING);
	glBindTexture(GL_TEXTURE_2D,text_menus[0].texID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_menus[0].width, text_menus[0].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_menus[0].imageData);
	glPushMatrix();{
		glScalef(5.0,5.0,3.0);
		glPushMatrix();{

			glTranslatef(-.75,-.75,.3f);
			/** face de devant **/
			glPushMatrix();{
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);	glVertex2f(.0f,1.5f);
					glTexCoord2f(1.0,1.0);	glVertex2f(1.5f,1.5f);
					glTexCoord2f(1.0,.0);	glVertex2f(1.5f,.0f); 
				}glEnd();
			}glPopMatrix();
			/** face de derriere **/
			glPushMatrix();{
				glTranslatef(.0,.0,-.4f);
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);	glVertex2f(.0f,1.5f);
					glTexCoord2f(1.0,1.0);	glVertex2f(1.5f,1.5f);
					glTexCoord2f(1.0,.0);	glVertex2f(1.5f,.0f); 
				}glEnd();
			}glPopMatrix();

			/** face de gauche**/
			glPushMatrix();{
				glTranslatef(1.5f,.0,.0f);
				glRotatef(90,.0,1.0,.0);
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);	glVertex2f(.0f,1.5f);
					glTexCoord2f(1.0,1.0);	glVertex2f(.4f,1.5f);
					glTexCoord2f(1.0,.0);	glVertex2f(.4f,.0f); 
				}glEnd();
			}glPopMatrix();

			/** face de droite**/
			glPushMatrix();{
				glRotatef(90,.0,1.0,.0);
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);	glVertex2f(.0f,1.5f);
					glTexCoord2f(1.0,1.0);	glVertex2f(.4f,1.5f);
					glTexCoord2f(1.0,.0);	glVertex2f(.4f,.0f); 
				}glEnd();
			}glPopMatrix();
		}glPopMatrix();
		glBindTexture(GL_TEXTURE_2D,textures[num_text].texID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textures[num_text].width, textures[num_text].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textures[num_text].imageData);
		glPushMatrix();{
			glTranslatef(-0.55f,.8-0.2-0.05,.32f);
			glBegin(GL_QUADS);{
				glTexCoord2f(1.0,.0);	glVertex2f(.0f,.0f);
				glTexCoord2f(.0,.0);	glVertex2f(.0f,.06f);
				glTexCoord2f(.0,1.0);	glVertex2f(1.1f,.06f);
				glTexCoord2f(1.0,1.0);	glVertex2f(1.1f,.0f); 
			}glEnd();
			if(trait_appuye[0]){ //??? certainly to display a "rotating" selection around the brush
				if(cpt_trait<15) cpt_trait++;
				else{
					cpt_trait=0;
					trait_appuye[0]=0;
					test_menu_epaisseur=0;
				}
			}
			glTranslatef(.0,-.32,.0);
			glBegin(GL_QUADS);{
				glTexCoord2f(1.0,.0);	glVertex2f(.0f,.0f);
				glTexCoord2f(.0,.0);	glVertex2f(.0f,.11f);
				glTexCoord2f(.0,1.0);	glVertex2f(1.1f,.11f);
				glTexCoord2f(1.0,1.0);	glVertex2f(1.1f,.0f); 
			}glEnd();
			if(trait_appuye[1]){//??? certainly to display a "rotating" selection around the brush
				if(cpt_trait<15) cpt_trait++;
				else{
					cpt_trait=0;
					trait_appuye[1]=0;
					test_menu_epaisseur=0;
				}
			}
			glTranslatef(.0,-.38,.0);
			glBegin(GL_QUADS);{
				glTexCoord2f(1.0,.0);	glVertex2f(.0f,.0f);
				glTexCoord2f(.0,.0);	glVertex2f(.0f,.14f);
				glTexCoord2f(.0,1.0);	glVertex2f(1.1f,.14f);
				glTexCoord2f(1.0,1.0);	glVertex2f(1.1f,.0f); 
			}glEnd();
			if(trait_appuye[2]){//??? certainly to display a "rotating" selection around the brush
				if(cpt_trait<15) cpt_trait++;
				else{
					cpt_trait=0;
					trait_appuye[2]=0;
					test_menu_epaisseur=0;
				}
			}
			glTranslatef(.0,-.45,.0);
			glBegin(GL_QUADS);{
				glTexCoord2f(1.0,.0);	glVertex2f(.0f,.0f);
				glTexCoord2f(.0,.0);	glVertex2f(.0f,.18f);
				glTexCoord2f(.0,1.0);	glVertex2f(1.1f,.18f);
				glTexCoord2f(1.0,1.0);	glVertex2f(1.1f,.0f); 
			}glEnd();
			if(trait_appuye[3]){
				if(cpt_trait<15) cpt_trait++;
				else{
					cpt_trait=0;
					trait_appuye[3]=0;
					test_menu_epaisseur=0;
				}
			}
		}glPopMatrix();
		glPushMatrix();{
			if (anim_contour){
				glBindTexture(GL_TEXTURE_2D,text_select[0].texID);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_select[0].width, text_select[0].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_select[0].imageData);
			}else {
				glBindTexture(GL_TEXTURE_2D,text_select[1].texID);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_select[1].width, text_select[1].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_select[1].imageData);
			}
			switch(var_largeur){
				case 0 :
					glPushMatrix();{
						glTranslatef(-0.55f,.8-0.2-0.05-.03,.32f);
						glBegin(GL_QUADS);{
							glTexCoord2f(0.0,.0);	glVertex2f(.0f,.0f);
							glTexCoord2f(.0,1.0);	glVertex2f(.0f,.12f);
							glTexCoord2f(1.0,1.0);	glVertex2f(1.1f,.12f);
							glTexCoord2f(1.0,.0);	glVertex2f(1.1f,.0f); 
						}glEnd();
					}glPopMatrix();
					break;
				case 1 :
					glPushMatrix();{
						glTranslatef(-0.55f,.8-0.2-0.05-.02,.32f);
						glTranslatef(.0,-.32,.0);
						glBegin(GL_QUADS);{
							glTexCoord2f(.0,.0);             glVertex2f(.0f,.0f);
							glTexCoord2f(.0,1.0);            glVertex2f(.0f,.15f);
							glTexCoord2f(1.0,1.0);           glVertex2f(1.1f,.15f);
							glTexCoord2f(1.0,.0);            glVertex2f(1.1f,.0f); 
						}glEnd();
					}glPopMatrix();
					break;
				case 2 :
					glPushMatrix();{
						glTranslatef(-0.55f,.8-0.2-0.05-.025,.32f);
						glTranslatef(.0,-.32,.0);
						glTranslatef(.0,-.38,.0);

						glBegin(GL_QUADS);{
							glTexCoord2f(.0,.0);             glVertex2f(.0f,.0f);
							glTexCoord2f(.0,1.0);            glVertex2f(.0f,.19f);
							glTexCoord2f(1.0,1.0);           glVertex2f(1.1f,.19f);
							glTexCoord2f(1.0,.0);            glVertex2f(1.1f,.0f); 
						}glEnd();
					}glPopMatrix();
					break;
				case 3 :
					glPushMatrix();{
						glTranslatef(-0.55f,.8-0.2-0.05-.025,.32f);
						glTranslatef(.0,-.32,.0);
						glTranslatef(.0,-.38,.0);
						glTranslatef(.0,-.45,.0);
						glBegin(GL_QUADS);{
							glTexCoord2f(.0,.0);             glVertex2f(.0f,.0f);
							glTexCoord2f(.0,1.0);            glVertex2f(.0f,.22f);
							glTexCoord2f(1.0,1.0);           glVertex2f(1.1f,.22f);
							glTexCoord2f(1.0,.0);            glVertex2f(1.1f,.0f); 
						}glEnd();
					}glPopMatrix();
					break;
			}
		}glPopMatrix();

	}glPopMatrix();
	glEnable(GL_BLEND);
	glutPostRedisplay();    
}

/**
 * Draw the back of the hand on screen, at the right coordinates
 */
void position_paume_main(){
	glPushMatrix();{
		glColor4fv(VERT);
		glTranslatef(paume_main_GL.X, paume_main_GL.Y, paume_main_GL.Z);
		glutSolidSphere(0.3,50,50);
	}glPopMatrix();
}

Point intersection_menu (float a, float b,float c,float d)
{
	Point p_inter;
	p_inter.X=0;
	p_inter.Y=0;
	p_inter.Z=0;
	float denom=1/a+0.015-(10532)/(8786*c);

	if ((a!=0)&&(c!=0)&&( denom!=0))
	{
		p_inter.Y=((b/a-10532*d/(8786*c)+29.193121)/denom);
		p_inter.X=(p_inter.Y-b)/a;
		p_inter.Z=(p_inter.Y-d)/c;
	}

	return p_inter;
}

Point intersection_menu_coul_et_ep (float a, float b,float c,float d)
{
	Point p_inter;
	p_inter.X=0;
	p_inter.Y=0;
	p_inter.Z=0;

	if(a!=0)
	{
		p_inter.Z = -11;
		p_inter.Y = c*p_inter.Z + d;
		p_inter.X=(p_inter.Y-b)/a;
	}

	return p_inter;
}

/**
 * Draw a line between two points, here to have the direction between
 * the back of the hand, and the tip of the finger
 */
void direction(Point P1, Point P2){
	float a,b,c,d;
	Point P3;
	glDisable(GL_LIGHTING);
	a = (P2.Y - P1.Y)/(P2.X - P1.X);
	b = P2.Y - a*P2.X;
	c = (P2.Y - P1.Y)/(P2.Z - P1.Z);
	d = P2.Y - c*P2.Z;
	if(P2.X>=P1.X) P3.X = P2.X * 10;
	else P3.X = P2.X * -10;
	P3.Y = a*P3.X + b;
	P3.Z = (P3.Y - d)/c;
	glBegin(GL_LINES);{
		glColor4fv(BLANC); 
		glVertex3f(P1.X, P1.Y, P1.Z);   
		glVertex3f(P3.X, P3.Y, P3.Z);
	}glEnd(); 

	glEnable(GL_LIGHTING);

	//??? calculate intersection with the menus ???
	P_inter_menu = intersection_menu(a,b,c,d); 
	P_inter_menu_coul_et_ep = intersection_menu_coul_et_ep(a,b,c,d);
}

/**
 * Add noise to the textures
 */
void calcul_texture(){
	int i,j;
	int  cpt_calc_tmp=0;

	switch(num_text){
	   case 0 : if(!LoadTGA(&text_temp[max_text_temp],"Data/rouge.tga"))  exit(1); break;
	   case 1 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/vert.tga"))  exit(1);  break;
	   case 2 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/bleu.tga"))  exit(1);  break;
	   case 3 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/blanc.tga"))  exit(1);  break;
	   case 4 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/jaune.tga"))  exit(1);  break;
	   case 5 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/bleu_clair.tga"))  exit(1);  break;
	   case 6 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/violet.tga"))  exit(1);  break;
	   case 7 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/vert_fonce.tga"))  exit(1);  break;
	   case 8 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/rose.tga"))  exit(1);  break;
	   case 9 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/orange.tga"))  exit(1);  break;
	   case 10 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/gris.tga"))  exit(1);  break;
	   case 11 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/maron.tga"))  exit(1);  break;
	   case 12 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/jaune_pale.tga"))  exit(1);  break;
	   case 13 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/bleu_gris.tga"))  exit(1);  break;
	   case 14 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/violet_pale.tga"))  exit(1);  break;
	   case 15 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/bordeaux.tga"))  exit(1);  break;
	   case 16 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/rouge_pale.tga"))  exit(1);  break;
	   case 17 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/gris_clair.tga"))  exit(1);  break;
	   case 18 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/bleu_fonce.tga"))  exit(1);  break;
	   case 19 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/jaune_fonce.tga"))  exit(1);  break;
	   case 20 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/violet_fonce.tga"))  exit(1);  break;
	   case 21 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/orange_jaune.tga"))  exit(1);  break;
	   case 22 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/bleu_ciel.tga"))  exit(1);  break;
	   case 23 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/vert_pale.tga"))  exit(1);  break;
	   case 24 :  if(!LoadTGA(&text_temp[max_text_temp],"Data/rose_pale.tga"))  exit(1);  break;
	}

	i=text_temp[max_text_temp].height*(text_temp[max_text_temp].width-40)*4;
	int max= text_temp[max_text_temp].height;
	/** ajout de bruit sur la fin de  la texture en cours **/
	float att;                                                                                                                
	for (j=0;j<=text_temp[max_text_temp].height*40;j+=1)   
	{
		float a=(float)trandom();

		if (j<=max*5) att=0.90;
		if ((j>max*5)&&(j<=max*10)) att=0.80;
		if ((j>max*10)&&(j<=max*15)) att=0.70;
		if ((j>max*15)&&(j<=max*20)) att=0.60;
		if ((j>max*20)&&(j<=max*25)) att=0.50;
		if ((j>max*25)&&(j<=max*30)) att=0.40;
		if ((j>max*30)&&(j<=max*35)) att=0.30;     
		if ((j>max*35)&&(j<=max*40)) att=0.20;
		if (a>att) text_temp[max_text_temp].imageData[i+3]=0;
		i+=4;    
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_temp[max_text_temp].width, text_temp[max_text_temp].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_temp[max_text_temp].imageData);       
	max_text_temp++;           
}

/**
 * Reset the painting (when the reset button has been clicked)
 */
void init_reset(){
	int i,j;
	//the array of points goes zero
	for(i=0;i<=taille_tab_p;i++){
		Tab_Points[i].X = 0;
		Tab_Points[i].Y = 0;
		Tab_Points[i].Z = 0;  
		Tab_Points[i].relie_prec = 0 ;
	}     
	//the array of textures goes zero
	for(i=0;i<=taille_tab_p;i++){
		Tab_coul[i][0];
		Tab_coul[i][1];   
		Tab_coul[i][2];   
	}
	taille_tab_p=0;
	cpt_tab_coul=0;

	cpt_bouton_poussoir1 = 0;
	cpt_bouton_poussoir2= 0;
	cpt_bouton_poussoir3 = 0;
	trX=-7.5,trY=6.0,trZ=-6.0f;  
	xprec=0,yprec=0;
	xprec_reel=0;

	test_si_point_suiv=0;
	dessiner=0;
	for(i=0;i<5;i++) {for(j=0;j<5;j++) boule_couleur_appuye[i][j]=0;}
	cpt_boule_couleur=0;
	cpt_trait=0;

	max_text_temp=1;
	recharge_peinture=1;

	for(i=0;i<100;i++){
		tab_cpt_pts[i]=0;
	}

	T=0;
}

/**
 * Initialisation of the application :
 * - read values from the config file
 * - load textures from files
 */
void init(){

	char configuration[2048];
	srand(time(NULL));  	
	width=glutGet(GLUT_WINDOW_WIDTH );
	height=glutGet(GLUT_WINDOW_HEIGHT );
	rap=width/height;
	config=fopen("config","r+");
	fgets(configuration,2048,config);
		//[rapports_coord_X] ??
	//[positionnement_X] ??
	//[Max_pts_par_trait] : number of GL points on one continuous paint trace
	//[couleur_initiale]  : starting color of the brush
	//[largeur_pinceau]	  : width size of the brush
	//[seuil_clic(mm)]    : length between the forehand and the backhand under which
	//it's considered that a clic is occuring (the hand is folding in a fist-like motion)
	//[Xmax_GL]			  : ??
	//[Ymin_GL] -12.78    : ??
	//[Zmax_GL_centre]	  : ??
	//[repere_gant]		  : specify if the glove matrix frame has to be displayed on startup
	sscanf(configuration,"%*s %f %*s %f %*s %d %*s %d %*s %f %d %*s %f %*s %f %*s %f %*s %f %*s %d",&rapport_X,&positionnement,&Max_pts_par_trait,&num_text,&largeur,&var_largeur,&seuil_clic,&Xmax_GL,&Ymin_GL,&Zmax_GL_moitie,&repere_gant);

	if(!LoadTGA(&textures[0],"Data/rouge.tga"))  exit(1);
	if(!LoadTGA(&textures[1],"Data/vert.tga"))  exit(1);
	if(!LoadTGA(&textures[2],"Data/bleu.tga"))  exit(1);
	if(!LoadTGA(&textures[3],"Data/blanc.tga"))  exit(1);
	if(!LoadTGA(&textures[4],"Data/jaune.tga"))  exit(1);
	if(!LoadTGA(&textures[5],"Data/bleu_clair.tga"))  exit(1);
	if(!LoadTGA(&textures[6],"Data/violet.tga"))  exit(1);
	if(!LoadTGA(&textures[7],"Data/vert_fonce.tga"))  exit(1);
	if(!LoadTGA(&textures[8],"Data/rose.tga"))  exit(1);
	if(!LoadTGA(&textures[9],"Data/orange.tga"))  exit(1);
	if(!LoadTGA(&textures[10],"Data/gris.tga"))  exit(1);
	if(!LoadTGA(&textures[11],"Data/maron.tga"))  exit(1);
	if(!LoadTGA(&textures[12],"Data/jaune_pale.tga"))  exit(1);
	if(!LoadTGA(&textures[13],"Data/bleu_gris.tga"))  exit(1);
	if(!LoadTGA(&textures[14],"Data/violet_pale.tga"))  exit(1);
	if(!LoadTGA(&textures[15],"Data/bordeaux.tga"))  exit(1);
	if(!LoadTGA(&textures[16],"Data/rouge_pale.tga"))  exit(1);
	if(!LoadTGA(&textures[17],"Data/gris_clair.tga"))  exit(1);
	if(!LoadTGA(&textures[18],"Data/bleu_fonce.tga"))  exit(1);
	if(!LoadTGA(&textures[19],"Data/jaune_fonce.tga"))  exit(1);
	if(!LoadTGA(&textures[20],"Data/violet_fonce.tga"))  exit(1);
	if(!LoadTGA(&textures[21],"Data/orange_jaune.tga"))  exit(1);
	if(!LoadTGA(&textures[22],"Data/bleu_ciel.tga"))  exit(1);
	if(!LoadTGA(&textures[23],"Data/vert_pale.tga"))  exit(1);
	if(!LoadTGA(&textures[24],"Data/rose_pale.tga"))  exit(1);
	if(!LoadTGA(&text_menu[0],"Data/telecommande_front.tga"))  exit(1);                                               
	if(!LoadTGA(&text_menu[1],"Data/telecommande_back.tga"))  exit(1);  
	if(!LoadTGA(&text_menu[2],"Data/telecommande_up.tga"))  exit(1); 
	if(!LoadTGA(&text_select[0],"Data/selection1.tga"))  exit(1); 
	if(!LoadTGA(&text_select[1],"Data/selection2.tga"))  exit(1);
	if(!LoadTGA(&text_menus[0],"Data/metal1.tga"))  exit(1);
	if(!LoadTGA(&text_boutons[0],"Data/bouton1.tga"))  exit(1);
	if(!LoadTGA(&text_boutons[1],"Data/bouton2.tga"))  exit(1); 
	if(!LoadTGA(&text_boutons[2],"Data/bouton3.tga"))  exit(1);     
	init_reset();//it's the first time we start the application, so the "reset" is going to initialise our variables
}

/**
 * GLUT resize
 */
static void resize(int width, int height){
	const float ar = (float) width / (float) height;
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-ar, ar, -1.0, 1.0, 2.0, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

/**
 * The menu box to be displayed on screen, with its buttons
 */
void menu(){

	glPushMatrix();{

		glClearDepth(1.0f);										
		glDepthFunc(GL_LEQUAL);									
		glEnable(GL_DEPTH_TEST);									

		glDisable(GL_LIGHTING);
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_BLEND);
		//la couleur de l'objet va �tre (1-alpha_de_l_objet) * couleur du fond et (le_reste * couleur originale) 
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
		//bind the texture to the menu bx (not the buttons !)
		glBindTexture(GL_TEXTURE_2D,text_menu[0].texID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_menu[0].width, text_menu[0].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_menu[0].imageData);       
		glTranslatef(16.5+2.0,-6.0,-12.0);
		glRotatef(-40,0,1,0);
		glScalef(3,3,1.5);

		//menu 
		glPushMatrix();{
			glScalef(0.09,5.0,1); 
			glutSolidCube(1.0);
		}glPopMatrix(); 

		glPushMatrix();{ 
			glTranslatef(-.05f,-2.5f,.52f);
			/** face de devant **/
			/** fond **/

			glBegin(GL_QUADS);{
				glTexCoord2f(.0,.0);			glVertex2f(.0f,.0f);
				glTexCoord2f(.0,.2763671875);	glVertex2f(.0f,2.0f);
				glTexCoord2f(1.0,.2763671875);	glVertex2f(.6f,2.0f);
				glTexCoord2f(1.0,.0);			glVertex2f(.6f,.0f); 
			}glEnd();
			/** cote gauche **/
			glPushMatrix();{ 
				glTranslatef(.0f,2.0f,.0f);
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.2763671875);		glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);				glVertex2f(.0f,3.2f);
					glTexCoord2f(.234375,1.0);			glVertex2f(.095f,3.2f);
					glTexCoord2f(.234375,.2763671875);	glVertex2f(.095f,.0f); 
				}glEnd();
			}glPopMatrix();
			/** cote droit **/
			glPushMatrix();{ 
				glTranslatef(.505f,2.0f,.0f);
				glBegin(GL_QUADS);{
					glTexCoord2f(.765625,.2763671875);	glVertex2f(.0f,.0f);
					glTexCoord2f(.765625,1.0);			glVertex2f(.0f,3.2f);
					glTexCoord2f(1.0,1.0);				glVertex2f(.095f,3.2f);
					glTexCoord2f(1.0,.2763671875);		glVertex2f(.095f,.0f); 
				}glEnd();
			}glPopMatrix(); 
			/** 1er inter quad**/
			glPushMatrix();{ 
				glTranslatef(.095f,2.9f,.0f);
				glBegin(GL_QUADS);{
					glTexCoord2f(.234375,.4599609375);	glVertex2f(.0f,.0f);
					glTexCoord2f(.234375,.4794921875);	glVertex2f(.0f,.1f);
					glTexCoord2f(.765625,.4794921875);	glVertex2f(0.41,.1f);
					glTexCoord2f(.765625,.4599609375);	glVertex2f(0.41,.0f); 
				}glEnd();
			}glPopMatrix();
			/** 2eme inter quad**/
			glPushMatrix();{ 
				glTranslatef(.095f,3.9f,.0f);
				glBegin(GL_QUADS);{
					glTexCoord2f(.234375,.65911796875);	glVertex2f(.0f,.0f);
					glTexCoord2f(.234375,.67578125);	glVertex2f(.0f,.1f);
					glTexCoord2f(.765625,.67578125);	glVertex2f(0.41,.1f);
					glTexCoord2f(.765625,.65911796875);	glVertex2f(0.41,.0f); 
				}glEnd();
			}glPopMatrix();
			/** 3eme inter quad**/
			glPushMatrix();{ 
				glTranslatef(.095f,4.9f,.0f);
				glBegin(GL_QUADS);{
					glTexCoord2f(.234375,.861328125);	glVertex2f(.0f,.0f);
					glTexCoord2f(.234375,1.0);			glVertex2f(.0f,.3f);
					glTexCoord2f(.765625,1.0);			glVertex2f(0.41,.3f);
					glTexCoord2f(.765625,.861328125);	glVertex2f(0.41,.0f); 
				}glEnd();
			}glPopMatrix(); 

			glBindTexture(GL_TEXTURE_2D,text_menu[1].texID);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_menu[1].width, text_menu[1].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_menu[1].imageData);

			glPushMatrix();{
				/** cote droit **/
				glPushMatrix();{
					glTranslatef(.6f,.0f,.0f);
					glRotatef(90,0.0,1.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,5.2f);
						glTexCoord2f(1.0,1.0);	glVertex2f(1.f,5.2f);
						glTexCoord2f(1.0,.0);	glVertex2f(1.f,.0f); 
					}glEnd();
				}glPopMatrix();                               
				/** cote gauche **/
				glPushMatrix();{
					glRotatef(90,0.0,1.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,5.2f);
						glTexCoord2f(1.0,1.0);	glVertex2f(1.f,5.2f);
						glTexCoord2f(1.0,.0);	glVertex2f(1.f,.0f); 
					}glEnd();
				}glPopMatrix();    
			}glPopMatrix();  
			/** face de derriere **/
			glPushMatrix();{ 
				glTranslatef(.0f,.0f,-1.022f);
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);	glVertex2f(.0f,5.2f);
					glTexCoord2f(1.0,1.0);	glVertex2f(.6f,5.2f);
					glTexCoord2f(1.0,.0);	glVertex2f(.6f,.0f); 
				}glEnd();      
			}glPopMatrix(); 

			/** face du haut **/ 
			glPushMatrix();{
				glTranslatef(.0f,5.2f,-1.0f);
				glRotatef(90,1.0,.0,.0);
				glBindTexture(GL_TEXTURE_2D,text_menu[2].texID);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_menu[2].width, text_menu[2].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_menu[2].imageData);
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);	glVertex2f(.0f,1.f);
					glTexCoord2f(1.0,1.0);	glVertex2f(.6f,1.f);
					glTexCoord2f(1.0,.0);	glVertex2f(.6f,.0f); 
				}glEnd();
			}glPopMatrix();                                  
		}glPopMatrix(); 

		/************ boutons **************/
		//bouton 1
		glBindTexture(GL_TEXTURE_2D,text_boutons[0].texID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_boutons[0].width, text_boutons[0].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_boutons[0].imageData);       
		glColor4fv(BLANC);
		glPushMatrix();{
			if(couleur==1) {glTranslatef(0,0,-0.5); cpt_bouton_poussoir1++;} //if the color button is pressed, we translate it back
			if(cpt_bouton_poussoir1==15) {couleur=0; cpt_bouton_poussoir1=0;} 

			glPushMatrix();{
				glTranslatef(0.095-0.41/8+.005,1.6-.1,0.75);
				glDisable(GL_LIGHTING); 

				/** devant **/   
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
					glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
					glTexCoord2f(1.0,.0);	glVertex2f(.41f,.0f); 
				}glEnd();

				/** derriere **/
				glPushMatrix();{                 
					glTranslatef(.0,.0,-0.5);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
						glTexCoord2f(1.0,.0);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();             

				/** cote gauche **/
				glPushMatrix();{                 
					glRotatef(90,.0,1.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.92);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
						glTexCoord2f(1.0,.92);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();

				/** cote droit **/
				glPushMatrix();{                 
					glTranslatef(0.41,.0,0.);
					glRotatef(90,.0,1.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.92);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
						glTexCoord2f(1.0,.92);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();

				/** bas **/
				glPushMatrix();{                 
					glTranslatef(0.,.0,-0.5);
					glRotatef(90,1.0,.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.92);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.5f);
						glTexCoord2f(1.0,.92);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();                   

				/** haut **/
				glPushMatrix();{                 
					glTranslatef(0.,.9f,-0.5);
					glRotatef(90,1.0,.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.92);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.5f);
						glTexCoord2f(1.0,.92);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();                   

				glEnable(GL_LIGHTING);
			}glPopMatrix();
		}glPopMatrix();



		/** bouton 2 **/
		glBindTexture(GL_TEXTURE_2D,text_boutons[1].texID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_boutons[1].width, text_boutons[1].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_boutons[1].imageData);       

		glPushMatrix();{
			if(epaisseur==1) {glTranslatef(0,0,-0.5); cpt_bouton_poussoir2++;}//if the width button is pressed, we translate it back
			if(cpt_bouton_poussoir2==15) {epaisseur=0; cpt_bouton_poussoir2=0;} 
			glTranslatef(0.,-1.0,0.);
			glPushMatrix();{
				glTranslatef(0.095-0.41/8+.005,1.6-.1,0.75);
				glDisable(GL_LIGHTING); 

				/** devant **/   
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
					glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
					glTexCoord2f(1.0,.0);	glVertex2f(.41f,.0f); 
				}glEnd();

				/** derriere **/
				glPushMatrix();{                 
					glTranslatef(.0,.0,-0.5);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
						glTexCoord2f(1.0,.0);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();             

				/** cote gauche **/
				glPushMatrix();{                 
					glRotatef(90,.0,1.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.98);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
						glTexCoord2f(1.0,.98);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();

				/** cote droit **/
				glPushMatrix();{                 
					glTranslatef(0.41,.0,0.);
					glRotatef(90,.0,1.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.98);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
						glTexCoord2f(1.0,.98);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();

				/** bas **/
				glPushMatrix();{                 
					glTranslatef(0.,.0,-0.5);
					glRotatef(90,1.0,.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.98);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.5f);
						glTexCoord2f(1.0,.98);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();                   

				/** haut **/
				glPushMatrix();{                 
					glTranslatef(0.,.9f,-0.5);
					glRotatef(90,1.0,.0,.0);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.98);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.5f);
						glTexCoord2f(1.0,.98);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();                   

				glEnable(GL_LIGHTING);
			}glPopMatrix();
		}glPopMatrix();

		glBindTexture(GL_TEXTURE_2D,text_boutons[2].texID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_boutons[2].width, text_boutons[2].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_boutons[2].imageData);       
		/** bouton 3 **/
		glPushMatrix();{
			if(reset==1) {glTranslatef(0,0,-0.5); cpt_bouton_poussoir3++;}
			if(cpt_bouton_poussoir3==15) {reset=0; cpt_bouton_poussoir3=0;} 
			glTranslatef(0.,-2.0,0.);
			glPushMatrix();{
				glTranslatef(0.095-0.41/8+.005,1.6-.1,0.75);
				glDisable(GL_LIGHTING); 

				/** devant **/   
				glBegin(GL_QUADS);{
					glTexCoord2f(.0,.0);	glVertex2f(.0f,.0f);
					glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
					glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
					glTexCoord2f(1.0,.0);	glVertex2f(.41f,.0f); 
				}glEnd();

				/** derriere **/
				glPushMatrix();{                 
					glTranslatef(.0,.0,-0.5);
					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.92);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
						glTexCoord2f(1.0,.92);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();             

				/** cote gauche **/
				glPushMatrix();{                 

					glRotatef(90,.0,1.0,.0);

					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.92);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
						glTexCoord2f(1.0,.92);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();

				/** cote droit **/
				glPushMatrix();{                 
					glTranslatef(0.41,.0,0.);
					glRotatef(90,.0,1.0,.0);

					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.92);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.9f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.9f);
						glTexCoord2f(1.0,.92);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();

				/** bas **/
				glPushMatrix();{                 
					glTranslatef(0.,.0,-0.5);
					glRotatef(90,1.0,.0,.0);

					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.92);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.5f);
						glTexCoord2f(1.0,.92);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();                   

				/** haut **/
				glPushMatrix();{                 
					glTranslatef(0.,.9f,-0.5);
					glRotatef(90,1.0,.0,.0);

					glBegin(GL_QUADS);{
						glTexCoord2f(.0,.92);	glVertex2f(.0f,.0f);
						glTexCoord2f(.0,1.0);	glVertex2f(.0f,.5f);
						glTexCoord2f(1.0,1.0);	glVertex2f(.41f,.5f);
						glTexCoord2f(1.0,.92);	glVertex2f(.41f,.0f); 
					}glEnd(); 
				}glPopMatrix();                   

				glEnable(GL_LIGHTING);
			}glPopMatrix();
		}glPopMatrix();

	}glPopMatrix();   
	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);	

}

/**
* Calculate the distance between two points.
* Used to know if the distance between one finger and the back of hand is 
* inferior to a given value (to determine a click)
*/
float calc_dist (Point p1,Point p2){
	float d=0;
	d=pow((p2.X-p1.X),2);
	d+=pow((p2.Y-p1.Y),2);
	d+=pow((p2.Z-p1.Z),2);

	return (sqrt(d));  
}

/**
 * Process the drawing of the scene
 */
void dessin(float x, float y){    

	glClearDepth(1.0f);										
	glDepthFunc(GL_LEQUAL);									
	glEnable(GL_DEPTH_TEST);									

	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);

	float X_temp,Y_temp,Z_temp;
	int i;
	float y_text=0.00,y_text_tmp=0.00;
	float x_decal,y_decal;
	int test_y;
	int var_test=1;
	int cpt_text=1;
	int j=0;
	float a=3;
	int test_2=1;
	int k=1;
	float pas_text= pow( (double) Max_pts_par_trait-1,-1);
	int cpt_points=0;
	int n=0;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
	glPushMatrix();{  

		//we cycle through all the points already paint inside the scene
		for (i=0;i<taille_tab_p;i++){         

			//we bind the texture of the current point
			glBindTexture(GL_TEXTURE_2D,text_temp[cpt_text].texID);

			if  (cpt_points>=n) {y_text_tmp=0; a=1;} //???

			//if three consecutive points are linked together and ???
			if(Tab_Points[i+1].relie_prec==1 && Tab_Points[i+2].relie_prec==1 && Tab_Points[i+3].relie_prec==1 && y_text<=1.0){

				//??? if the Y coordinate of 2 consecutives points is between -.1 and +.1, do ???
				if((Tab_Points[i].Y-Tab_Points[i+1].Y<0.1) && (Tab_Points[i].Y-Tab_Points[i+1].Y>-.1)) test_y = 1;

				//draw the quads holding the textures
				if(test_y==0){
					glPushMatrix();{ 
						glBegin(GL_QUADS);{
							glTexCoord2f(0.0,y_text);  glVertex3f(Tab_Points[i].X,Tab_Points[i].Y,Tab_Points[i].Z);
							glTexCoord2f(0.0,y_text+a/(Max_pts_par_trait-1)+ y_text_tmp); glVertex3f(Tab_Points[i+1].X,Tab_Points[i+1].Y,Tab_Points[i+1].Z);
							glTexCoord2f(1.0,y_text+a/(Max_pts_par_trait-1)+ y_text_tmp); glVertex3f(Tab_Points[i+1].X+Tab_coul[i][1],Tab_Points[i+1].Y,Tab_Points[i+1].Z);
							glTexCoord2f(1.0,y_text); glVertex3f(Tab_Points[i].X+Tab_coul[i][1],Tab_Points[i].Y,Tab_Points[i].Z);
						}glEnd();  
					}glPopMatrix(); 
				}else {
					glPushMatrix();{ 
						glBegin(GL_QUADS);{
							glTexCoord2f(0.0,y_text); glVertex3f(Tab_Points[i].X,Tab_Points[i].Y,Tab_Points[i].Z);
							glTexCoord2f(0.0,y_text+a/(Max_pts_par_trait-1)+ y_text_tmp); glVertex3f(Tab_Points[i+1].X,Tab_Points[i+1].Y,Tab_Points[i+1].Z);
							glTexCoord2f(1.0,y_text+a/(Max_pts_par_trait-1)+ y_text_tmp); glVertex3f(Tab_Points[i+1].X+Tab_coul[i][1],Tab_Points[i+1].Y-.5*Tab_coul[i][1],Tab_Points[i+1].Z);
							glTexCoord2f(1.0,y_text); glVertex3f(Tab_Points[i].X+Tab_coul[i][1],Tab_Points[i].Y-.5*Tab_coul[i][1],Tab_Points[i].Z);
						}glEnd();  
					}glPopMatrix(); 
				}  
				y_text+=pas_text; //???
				if (cpt_points<5) y_text_tmp+=(a-1)*pas_text; // ???
			}
			cpt_points++; //we go on the next point

			if(Tab_Points[i+1].relie_prec==0) {y_text=y_text_tmp=0.00;cpt_text++;a=3;cpt_points=0;}   //??? if the point isn't linked to the next, do ???

		}       
	}glPopMatrix();

	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);	
	glEnable(GL_LIGHTING);
}

/**
* Draw the glove matrix frame (numerical coordinates) on screen
*/
void drawText(void) {

	char  gl0[15]= {""};
	char  gl1[15]= {""};
	char  gl2[15]= {""};
	char  gl3[15]= {""};
	char  gl4[15]= {""};
	char  gl5[15]= {""};
	char  gl6[15]= {""};
	char  gl7[15]= {""};
	char  gl8[15]= {""};
	int i;

	sprintf(gl0,"%.2f",	gli[0]);
	sprintf(gl1,"%.2f",	gli[1]);
	sprintf(gl2,"%.2f",	gli[2]);
	sprintf(gl3,"%.2f",	gli[3] );
	sprintf(gl4,"%.2f",	gli[4]);
	sprintf(gl5,"%.2f",	gli[5]);
	sprintf(gl6,"%.2f",	gli[6]);
	sprintf(gl7,"%.2f",	gli[7]);
	sprintf(gl8,"%.2f",	gli[8]);  

	glColor3f(1.0,1.0,1.0);

	glRasterPos3f(-2.5, -0.6, -6);
	for( i=0; i<(int)strlen( gl0 ); i++ ) glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, gl0[i] ); 

	glRasterPos3f(-2.5+1, -0.6, -6);
	for( i=0; i<(int)strlen( gl1 ); i++ ) glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, gl1[i] ); 

	glRasterPos3f(-2.5+2, -0.6, -6);
	for( i=0; i<(int)strlen( gl2 ); i++ ) glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, gl2[i] );  

	glRasterPos3f(-2.5, -0.6-.5, -6);
	for( i=0; i<(int)strlen( gl3 ); i++ ) glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, gl3[i]  ); 

	glRasterPos3f(-2.5+1, -0.6-.5, -6);
	for( i=0; i<(int)strlen( gl4 ); i++ ) glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, gl4[i] ); 

	glRasterPos3f(-2.5+2, -0.6-.5, -6);
	for( i=0; i<(int)strlen( gl5 ); i++ ) glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, gl5[i] );  

	glRasterPos3f(-2.5, -0.6-1, -6);
	for( i=0; i<(int)strlen( gl6 ); i++ ) glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, gl6[i]  ); 

	glRasterPos3f(-2.5+1, -0.6-1, -6);
	for( i=0; i<(int)strlen( gl7); i++ ) glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, gl7[i]  ); 

	glRasterPos3f(-2.5+2, -0.6-1, -6);
	for( i=0; i<(int)strlen( gl8 ); i++ ) glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, gl8[i] );       
}

/**
* Draw the orientation of the matrix glove on screen
*/
void drawrepglove(){

	glDisable(GL_LIGHTING);

	glPushMatrix();{
		glTranslatef(-6.0,.0,.0);
		glColor3f(1.0,.0f,.0f); 
		glBegin(GL_LINES);{ 
			glVertex3f(12.0f,-6.0f,-6.0f); 
			glVertex3f(12.0+3*gli[0],3*gli[1]-6.0f,3*gli[2]-6.0f);
		}glEnd();
		glPushMatrix();{
			glTranslatef(12.0+3*gli[0],3*gli[1]-6.0f,3*gli[2]-6.0f);
			glutSolidSphere(0.2,50,50);
		}glPopMatrix();
		glColor3f(.0,1.0f,.0f); 
		glBegin(GL_LINES);{ 
			glVertex3f(12.0f,-6.0f,-6.0f); 
			glVertex3f(12.0+3*gli[3],3*gli[4]-6.0,3*gli[5]-6.0);
		}glEnd();
		glPushMatrix();{
			glTranslatef(12.0+3*gli[3],3*gli[4]-6.0,3*gli[5]-6.0);
			glutSolidSphere(0.2,50,50);
		}glPopMatrix();
		glColor3f(.0,.0f,1.0f); 
		glBegin(GL_LINES);{ 
			glVertex3f(12.0f,-6.0f,-6.0f); 
			glVertex3f(12.0+3*gli[6],3*gli[7]-6.0,3*gli[8]-6.0);
		}glEnd();
		glPushMatrix();{
			glTranslatef(12.0+3*gli[6],3*gli[7]-6.0,3*gli[8]-6.0);
			glutSolidSphere(0.2,50,50);
		}glPopMatrix();
	}glPopMatrix();

	glEnable(GL_LIGHTING);
}

/**
* GLUT display
*/
void display(void){

	glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
	glClearColor(0,0,0,0);
	glLoadIdentity();

	glDisable(GL_LIGHTING); 

	//if the glove frame is activated (check the key), we draw the numerical coordinates of the tracker on screen
	if (repere_gant) drawText();
	glEnable(GL_LIGHTING);

	glPushMatrix();{

		glTranslatef(trX,trY,trZ);

		//if the glove frame is activated (check the key), we draw it on screen
		if (repere_gant) drawrepglove();
		menu();

		glPushMatrix();{
			//positionnement = from where we look at the scene
			glTranslatef(positionnement,0,0);
			//if no menu is drawn on screen, we process the drawing of the scene
			if (test_menu_couleur==0 && test_menu_epaisseur==0 )dessin(xprec,yprec); 
		}glPopMatrix();

		//draw the back of the hand on screen
		position_paume_main(); 

		//draw the direction (a line) between the back of the hand and the index
		direction(paume_main_GL,index_GL);

		glPushMatrix();{
			//draw the color menu if it has been clicked
			if (test_menu_couleur) {glTranslatef(7.5f,-6.0f,-12.0f); menu_couleur();}
		}glPopMatrix();     
		glPushMatrix();{
			//draw the brush width menu if it has been clicked
			if (test_menu_epaisseur) {glTranslatef(7.5f,-6.0f,-12.0f); menu_epaisseur();}                             
		}glPopMatrix();

	}glPopMatrix();

	glutSwapBuffers(); 
}


/** solution propos�e : si la distance entre le capteur d un doigt et celui de la main est suffisament petite, alors cela correspond a un clic **/
bool clic (){
	Point centre_main,doigt1;
	float distance_doigt_main;

	centre_main.X=0;
	centre_main.Y=0;
	centre_main.Z=0;
	doigt1.X=glove[0].finger[1].loc[0];
	doigt1.Y=glove[0].finger[1].loc[1];
	doigt1.Z=glove[0].finger[1].loc[2];
	distance_doigt_main=calc_dist(doigt1,centre_main);

	if (distance_doigt_main<=seuil_clic) return true;
	else return false;
}

/**
*	GLUT motion :
* What to do when the mouse is moving.
* When the mouse is moving, we add the point corresponding to the various positions (coordinates)
* of the mouse inside the array holding all the points
*/
void motion (float x, float y, float z){

	//if we're not displaying any menu on screen, and if we just have paint on the brush
	//(if we haven't run out of paint in this swoop, in fact)
	if(dessiner && test_menu_couleur==0 && test_menu_epaisseur==0 && recharge_peinture==1){

		//we add a new point in the array of point, at the coordinates of the mouse
		Tab_Points[taille_tab_p].X=x;
		Tab_Points[taille_tab_p].Y=y;
		Tab_Points[taille_tab_p].Z=z; 
		Tab_Points[taille_tab_p].relie_prec=1; //since we have paint on the brush, this point is linked to the previous
		Tab_coul[taille_tab_p][0]=num_text;		//we specify the current texture (the one selected for this point)
		Tab_coul[taille_tab_p][1]=largeur;		//we specify the current width of brush
		Tab_coul[taille_tab_p][2]=recharge_peinture; //used to specify that this point must be drawn on screen, since
		//there's paint on the brush (recharge_peinture==1)

		printf("Adding a point at the coordinates : %i,%i,%i", x,y,z);

		/* permet de r�cup�rer le Point suivant dans le cas o� il ne faut pas tracer */
		if(test_si_point_suiv==1){
			Tab_Points[taille_tab_p-1].X=x;
			Tab_Points[taille_tab_p-1].Y=y;
			Tab_Points[taille_tab_p-1].Z=z; 
			Tab_coul[taille_tab_p][0]=num_text;
			Tab_coul[taille_tab_p][1]=largeur;
			Tab_coul[taille_tab_p][2]=recharge_peinture;
			test_si_point_suiv=0;
		}

		taille_tab_p++;	//we add one point to the array of points

		if(tab_cpt_pts[T]<=Max_pts_par_trait) tab_cpt_pts[T]++; //??? � quoi sert tab_cpt_pts ?

		cpt_pts++; //one more point in this swwop
		//if we have more point in this swoop than the maximum, we run out of paint
		if(cpt_pts==Max_pts_par_trait)   recharge_peinture=0;  
	}

	glutPostRedisplay();  
}

/**
* GLUT mouse :
* process the "mouse" intersections with the various elements of the menus
* /!\ we need to check at each movement of the mouse if it's not intersecting an element of the scene,
* so we need to do it on the 'idle' function, to process it all the time
*/
void mouse(float x, float y, float z){ 

	//when we click on something
	if (clic()){

		//we're currently drawing
		dessiner=1;

		//put the point of the mouse motion inside the painting
		motion(x,y,z);

		/**
		* if we're interestcing one of the menu, we react accordingly
		*/
		//intersections with the menu box
		if ((P_inter_menu.X!=0)||(P_inter_menu.Y!=0)||(P_inter_menu.Z!=0)){
			/* 1er test pour tous les boutons */                                         
			if ((P_inter_menu.X>=13) && (P_inter_menu.X<=18.7) && (P_inter_menu.Z>=-12.9) && (P_inter_menu.Z<=-8.3) && test_menu_couleur==0 && test_menu_epaisseur==0){
				/* 1er bouton*/
				if ((P_inter_menu.Y<=1.0) &&(P_inter_menu.Y>=-1.6)) {couleur=1;test_menu_couleur=1;}
				/* 2eme bouton*/
				if ((P_inter_menu.Y<=-1.8) &&(P_inter_menu.Y>=-4.5)) {epaisseur=1;test_menu_epaisseur=1;} 
				/* 3eme bouton*/
				if ((P_inter_menu.Y<=-4.7) &&(P_inter_menu.Y>=-7.5)) {reset=1;init_reset();test_T=1;}                                                                  
			}                                       
		}

		// intersections with the color menu
		//Each time, the current texture (num_text) is changed accordingly
		if ((P_inter_menu_coul_et_ep.X!=0)||(P_inter_menu_coul_et_ep.Y!=0)||(P_inter_menu_coul_et_ep.Z!=0)){                                        
			// tests ligne
			if ((P_inter_menu_coul_et_ep.Y<=-2.5) &&(P_inter_menu_coul_et_ep.Y>=-3.5) && test_menu_couleur==1 && test_menu_epaisseur==0){ 
				// tests colonne         
				if ((P_inter_menu_coul_et_ep.X<=5.0) &&(P_inter_menu_coul_et_ep.X>=4.0)) { boule_couleur_appuye[0][0]=1;num_text=0;} 
				if ((P_inter_menu_coul_et_ep.X<=6.5) &&(P_inter_menu_coul_et_ep.X>=5.5)) {boule_couleur_appuye[0][1]=1;num_text=5;} 
				if ((P_inter_menu_coul_et_ep.X<=8.0) &&(P_inter_menu_coul_et_ep.X>=7.0)) {boule_couleur_appuye[0][2]=1;num_text=10;} 
				if ((P_inter_menu_coul_et_ep.X<=9.5) &&(P_inter_menu_coul_et_ep.X>=8.5)) {boule_couleur_appuye[0][3]=1;num_text=15;} 
				if ((P_inter_menu_coul_et_ep.X<=11.0) &&(P_inter_menu_coul_et_ep.X>=10.0)) { boule_couleur_appuye[0][4]=1;num_text=20;} 
			}
			if ((P_inter_menu_coul_et_ep.Y<=-4) &&(P_inter_menu_coul_et_ep.Y>=-5) && test_menu_couleur==1 && test_menu_epaisseur==0){ 
				// tests colonne         
				if ((P_inter_menu_coul_et_ep.X<=5) &&(P_inter_menu_coul_et_ep.X>=4)) {boule_couleur_appuye[1][0]=1;num_text=1;} 
				if ((P_inter_menu_coul_et_ep.X<=6.5) &&(P_inter_menu_coul_et_ep.X>=5.5)) {boule_couleur_appuye[1][1]=1;num_text=6;} 
				if ((P_inter_menu_coul_et_ep.X<=8.0) &&(P_inter_menu_coul_et_ep.X>=7.0)) {boule_couleur_appuye[1][2]=1;num_text=11;} 
				if ((P_inter_menu_coul_et_ep.X<=9.5) &&(P_inter_menu_coul_et_ep.X>=8.5)) {boule_couleur_appuye[1][3]=1;num_text=16;} 
				if ((P_inter_menu_coul_et_ep.X<=11.0) &&(P_inter_menu_coul_et_ep.X>=10.0)) {boule_couleur_appuye[1][4]=1;num_text=21;} 
			}
			if ((P_inter_menu_coul_et_ep.Y<=-5.5) &&(P_inter_menu_coul_et_ep.Y>=-6.5) && test_menu_couleur==1 && test_menu_epaisseur==0){ 
				// tests colonne         
				if ((P_inter_menu_coul_et_ep.X<=5) &&(P_inter_menu_coul_et_ep.X>=4)) {boule_couleur_appuye[2][0]=1;num_text=2;} 
				if ((P_inter_menu_coul_et_ep.X<=6.5) &&(P_inter_menu_coul_et_ep.X>=5.5)) {boule_couleur_appuye[2][1]=1;num_text=7;} 
				if ((P_inter_menu_coul_et_ep.X<=8.0) &&(P_inter_menu_coul_et_ep.X>=7.0)) { boule_couleur_appuye[2][2]=1;num_text=12;} 
				if ((P_inter_menu_coul_et_ep.X<=9.5) &&(P_inter_menu_coul_et_ep.X>=8.5)) { boule_couleur_appuye[2][3]=1;num_text=17;} 
				if ((P_inter_menu_coul_et_ep.X<=11.0) &&(P_inter_menu_coul_et_ep.X>=10.0)) { boule_couleur_appuye[2][4]=1;num_text=22;}
			}
			if ((P_inter_menu_coul_et_ep.Y<=-7.0) &&(P_inter_menu_coul_et_ep.Y>=-8.0) && test_menu_couleur==1 && test_menu_epaisseur==0){ 
				// tests colonne         
				if ((P_inter_menu_coul_et_ep.X<=5) &&(P_inter_menu_coul_et_ep.X>=4)) { boule_couleur_appuye[3][0]=1;num_text=3;} 
				if ((P_inter_menu_coul_et_ep.X<=6.5) &&(P_inter_menu_coul_et_ep.X>=5.5)) { boule_couleur_appuye[3][1]=1;num_text=8;} 
				if ((P_inter_menu_coul_et_ep.X<=8.0) &&(P_inter_menu_coul_et_ep.X>=7.0)) { boule_couleur_appuye[3][2]=1;num_text=13;} 
				if ((P_inter_menu_coul_et_ep.X<=9.5) &&(P_inter_menu_coul_et_ep.X>=8.5)) { boule_couleur_appuye[3][3]=1;num_text=18;} 
				if ((P_inter_menu_coul_et_ep.X<=11.0) &&(P_inter_menu_coul_et_ep.X>=10.0)) { boule_couleur_appuye[3][4]=1;num_text=23;}
			}
			if ((P_inter_menu_coul_et_ep.Y<=-8.5) &&(P_inter_menu_coul_et_ep.Y>=-9.5) && test_menu_couleur==1 && test_menu_epaisseur==0){ 
				// tests colonne         
				if ((P_inter_menu_coul_et_ep.X<=5) &&(P_inter_menu_coul_et_ep.X>=4)) { boule_couleur_appuye[4][0]=1;num_text=4;}
				if ((P_inter_menu_coul_et_ep.X<=6.5) &&(P_inter_menu_coul_et_ep.X>=5.5)) {boule_couleur_appuye[4][1]=1;num_text=9;} 
				if ((P_inter_menu_coul_et_ep.X<=8.0) &&(P_inter_menu_coul_et_ep.X>=7.0)) { boule_couleur_appuye[4][2]=1;num_text=14;} 
				if ((P_inter_menu_coul_et_ep.X<=9.5) &&(P_inter_menu_coul_et_ep.X>=8.5)) { boule_couleur_appuye[4][3]=1;num_text=19;} 
				if ((P_inter_menu_coul_et_ep.X<=11.0) &&(P_inter_menu_coul_et_ep.X>=10.0)) { boule_couleur_appuye[4][4]=1;num_text=24;} 
			}                                         
		}

		//intersections with the brush width menu
		if ((P_inter_menu_coul_et_ep.X!=0)||(P_inter_menu_coul_et_ep.Y!=0)||(P_inter_menu_coul_et_ep.Z!=0)){                                        
			// tests ligne
			if ((P_inter_menu_coul_et_ep.X<=10.0) &&(P_inter_menu_coul_et_ep.X>=5.0) && test_menu_couleur==0 && test_menu_epaisseur==1){ 
				// tests colonne         
				if ((P_inter_menu_coul_et_ep.Y<=-3.0) &&(P_inter_menu_coul_et_ep.Y>=-3.4)) {largeur = 0.2; var_largeur = 0; trait_appuye[0]=1;} 
				if ((P_inter_menu_coul_et_ep.Y<=-4.5) &&(P_inter_menu_coul_et_ep.Y>=-5.0)) {largeur = 0.3; var_largeur = 1; trait_appuye[0]=1;} 
				if ((P_inter_menu_coul_et_ep.Y<=-6.1) &&(P_inter_menu_coul_et_ep.Y>=-6.9)) {largeur = 0.4; var_largeur = 2; trait_appuye[0]=1;} 
				if ((P_inter_menu_coul_et_ep.Y<=-8.0) &&(P_inter_menu_coul_et_ep.Y>=-9.0)) {largeur = 0.5; var_largeur = 3; trait_appuye[0]=1;}                            
			}                                      
		}

		//when we don't intersect any menu and ???, we calculate the "noised" texture, for the brush
		if(test_menu_couleur==0 && test_menu_epaisseur==0 && test_T==0)   calcul_texture();
	}

	//if there's no clic
	if (clic()==false && clic_temp){
		//we're not currently drawing
		dessiner=0;

		//the new point we're adding isn't linked with the previous
		Tab_Points[taille_tab_p].relie_prec=0;

		//for security, we check that the clic hasn't occured while clicking on one of the menu
		if(test_menu_couleur==0 && test_menu_epaisseur==0){
			taille_tab_p++;	//we add one point to the total of points
			test_si_point_suiv=1; //???
			recharge_peinture=1; //since we "de-click", we have now paint on the brush
			cpt_pts=0;	//we have zero points in this swoop
			if(test_T==0) T++; //??
			else test_T=0;
		}
	}  

	//because of the test just above, we need to know if 
	clic_temp=clic(); //??? � tester : voir ce qui se passe si on vire
}

/**
* GLUT keyboardSpec :
* What to do with the special key of the keyboard input
*/
void keyboardSpec (int key, int x, int y){

	switch( key) {
		/** zoom **/
		case GLUT_KEY_UP: trZ+=.5f; break;
		case GLUT_KEY_DOWN: trZ-=.5f; break;
	}

	glutPostRedisplay ();	
}
/**
* GLUT key :
* what to do with the various input keys
*/
void key(unsigned char key, int x, int y){
	//process keys only if no menu is shown on screen (neither the color menu nor the brush width menu)
	if( test_menu_couleur==0 && test_menu_epaisseur==0){
		switch (key){
		case 27: exit(0); break;
		case ' ': ZERO.X=paume_main_GL.X; ZERO.Y=paume_main_GL.Y;ZERO.Z=paume_main_GL.Z; 
			cout << "Calibration done\n";
			break;
		case 'r' : if (repere_gant) repere_gant=0; else repere_gant=1; break;
			cout << "Display glove frame\n";
		}
	}
	glutPostRedisplay();
}

/**
* GLUT idle : 
* While the application is doing nothing, we receive datas from the tracker. The data corresponds
* to the glove position in space
*/
void idle(void){
	float x,y,z;

	/**   r�cup�ration des donne�s issues du gant (position, rotation) : stock�es dans la structure glove**/
	bool ok;
	ok = dtracklib_receive(handle,&framenr, &timestamp,NULL, NULL, 0, 0,NULL, 0, 0,NULL, 0, 0,
		&nmarker, marker, MAX_NMARKER,&nglove, glove, MAX_NGLOVE);

	//errors control
	if(!ok){
		if(dtracklib_timeout(handle)) printf("--- timeout while waiting for udp data\n");
		if(dtracklib_udperror(handle)) printf("--- error while receiving udp data\n");
		if(dtracklib_parseerror(handle)) printf("--- error while parsing udp data\n");	
	}

	//coordinates of the back of the hand
	paume_main.X=glove[0].loc[0];
	paume_main.Y=glove[0].loc[1];
	paume_main.Z=glove[0].loc[2];

	cout << "Coordinates of the back of the hand :\n";
	cout << "X : " << paume_main.X << "\n";
	cout << "Y : " << paume_main.Y << "\n";
	cout << "Z : " << paume_main.Z << "\n";

	//conversion between the coordinates in the absolute space (the room), and the coordinates
	//of the OpenGL scene
	paume_main_GL.X=x=paume_main.X*Xmax_GL/1000 + Xmax_GL/2;
	paume_main_GL.Y=y=-paume_main.Y*Ymin_GL/2000 + Ymin_GL/2;
	paume_main_GL.Z=z=+paume_main.Z*Zmax_GL_moitie/2000 - Zmax_GL_moitie;   

	//??? uses the rotation matrix of the back of the hand to define a point
	//(which will be used later to trace a line between the center of the back of the hand and this point)
	i.X=glove[0].rot[0];
	i.Y=glove[0].rot[1];
	i.Z=glove[0].rot[2];

	//conversion between the coordinates in the absolute space (the room), and the coordinates
	//of the OpenGL scene
	index_GL.X=i.X*Xmax_GL/1000 + Xmax_GL/2;
	index_GL.Y=-i.Y*Ymin_GL/2000 + Ymin_GL/2;
	index_GL.Z=i.Z*Zmax_GL_moitie/2000 - Zmax_GL_moitie;

	//if asked, draw on screen the coordinates of the rotation matrix of the glove
	if (repere_gant)   for (int i=0;i<=8;i++)  gli[i]=glove[0].rot[i];

	//process the mouse movements
	// /!\ we need to check at each movement of the mouse if it's not intersecting an element of the scene,
	//so we need to do it on the 'idle' function, to process it all the time
	mouse(x,y,z);	

	//animation for the dotted line around a selection
	if ((test_menu_epaisseur)||(test_menu_couleur)){
		cpt_tours++;
		if(cpt_tours==15){
			cpt_tours=0;
			if (anim_contour) anim_contour=0;
			else anim_contour=1;
		}  
	}                                
	glutPostRedisplay();
}

int main(int argc, char *argv[]){
	cout << "\nInitialisation of the application...\n";
	/** initialisation de la connexion udp **/
	int port, rport;

	if(argc != 4){
		cout << "\nBad command line :\n";
		cout << "[application] [local port] [remote IP address] [remote port]\n";
		cout << "------------------------------------------------------------\n";
		cout << "[local port]\t\t: local port for communications with the tracker\n";
		cout << "[remote IP address]\t: IP address of the machine controlling the tracker\n";
		cout << "[remote port]\t\t: remote port of the machine controlling the tracker\n";
		return -1;
	}

	port = atoi(argv[1]);
	if(port <=0 || port >= 65536){
		printf("impossible port %d\n", port);
		return -2;
	}

	rport = atoi(argv[3]);
	if(rport <=0 || rport >= 65536){
		printf("impossible port %d\n", rport);
		return -2;
	}

	if(!(handle = dtracklib_init(port, argv[2], rport, UDPBUFSIZE, UDPTIMEOUT))){
		printf("dtracklib init error\n");
		printf("(check that the tracker is opened and started (listening)\n");
		return -3;
	}

	// starting DTrack and calling for data:
	if(!dtracklib_send(handle, DTRACKLIB_CMD_CAMERAS_AND_CALC_ON, 0)){
		printf("dtracklib send command error\n");
		return -4;
	}

	if(!dtracklib_send(handle, DTRACKLIB_CMD_SEND_DATA, 0)){
		printf("dtracklib send command error\n");
		return -4;
	}

	system ("echo veuillez demarrer le gant" );
	system("pause");
	/***************************************************************/
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("Peinture 3D");
	glutFullScreen();

	init();
	glutReshapeFunc(resize);
	glutDisplayFunc(display);
	glutKeyboardFunc(key);
	glutSpecialFunc (keyboardSpec);

	glutIdleFunc(idle);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);

	glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

	glutMainLoop();

	if(!dtracklib_send(handle, DTRACKLIB_CMD_STOP_DATA, 0)){
		printf("dtracklib send command error\n");
		return -4;
	}

	if(!dtracklib_send(handle, DTRACKLIB_CMD_CAMERAS_OFF, 0)){
		printf("dtracklib send command error\n");
		return -4;
	}

	// exit lib:
	dtracklib_exit(handle);

	cout << "\nApplication finished...\n";

	return EXIT_SUCCESS;
}
