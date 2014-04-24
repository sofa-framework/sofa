/****************************************************************************/
/* Zinrimage.h *       Zinrimage                                            */
/*******************                                                        */
/* LIBRARY : Zinrimage                                                      */
/* COMMENT : I/O functions for reading and writing inrimages, gziped 
   inrimages, GIF or RGB images                                             */
/* AUTHOR : J. Montagnat                                                    */
/* SEEALSO :                                                                */
/****************************************************************************/


#ifndef ZINRIMAGE_H
#define ZINRIMAGE_H

//#ifdef WIN32
//  #ifdef EXPORTING
//    #define APIEXPORT __declspec(dllexport)
//  #else
//    #define APIEXPORT __declspec(dllimport)
//  #endif
//#else
//  #define APIEXPORT
//#endif
 #define APIEXPORT

#ifdef __cplusplus
extern "C" {
#endif


#include "ImageIO.h"


typedef enum {
  WT_UNSIGNED_CHAR,  /* Unsigned 8 bits */
  WT_UNSIGNED_SHORT, /* Unsigned 16 bits */
  WT_SIGNED_SHORT,   /* Signed 16 bits */
  WT_UNSIGNED_INT,   /* Unsigned 32 bits */
  WT_SIGNED_INT,     /* Signed 32 bits */
  WT_UNSIGNED_LONG,  /* Unsigned 64 bits */
  WT_SIGNED_LONG,    /* Signed 64 bits */
  WT_FLOAT,          /* Float 32 bits */
  WT_DOUBLE,         /* Float 64 bits */
  WT_RGB,            /* R, G, B, 8 bits each */
  WT_RGBA,           /* R, G, B, A, 8 bits each */
  WT_FLOAT_VECTOR,   /* Vector of 32 bit floats */
  WT_DOUBLE_VECTOR   /* Vector of 64 bit floats */
} WORDTYPE;
/* The different size of words for a 3D image */


typedef struct {
  int ncols;         /* Number of columns (X dimension) */
  int nrows;         /* Number of rows (Y dimension) */
  int nplanes;       /* Number of planes (Z dimension) */
  int vdim;          /* Vector size (3 for WT_RGB, 4 for WT_RGBA, any for
                        WT_FLOAT_VECTOR or WT_DOUBLE_VECTOR and 1 for
                        all other word type) */
  WORDTYPE type;     /* Type of image words */
  void *data;        /* Generic pointer on image data buffer.
		        This pointer has to be casted in proper data type
			depending on the type field */
  void ***array;     /* Generic 3D array pointing on each image element.
		        This pointer has to be casted in proper data type
			depending on the type field */
  void ****varray;   /* Generic 4D array pointing on each image vectorial
			component. This pointer has to be casted in proper
			data type depending on the type field.
		        varray is set to NULL for a scalar image or
		        an interlaced vectorial image. varray is only
		        used for non-interlaced vectorial images load
		        with readNonInterlacedZInrimage. */
  double vx;          /* real voxel size in X dimension */
  double vy;          /* real voxel size in Y dimension */
  double vz;          /* real voxel size in Z dimension */
} inrimage;
/* Inrimage descriptor.<br>
   <br>
   Inrimages are properly allocated using <tt>createInrimage</tt> or<br>
   <tt>readZInrimage</tt> function calls.<br>
   <br>
   Pointers are set as follows:<pre>
   inrimage *inr -> [ int ncols;     ]<br>
                    [ int nrows;     ]<br>
                    [ ....           ]<br>
                    [ void *data ----]------------------------> [ image data ]<br>
                    [ void ***array -]-> [void **  ]                ^  ^  ^<br>
                                         [void **  ]                |  |  |<br>
                                         [void ** -]--> [void * -]-/   |  |<br>
                                         [void **  ]    [void * -]----/   |<br>
                                         [ ...     ]    [void * -]-------/<br>
                                                        [void * ]<br>
                                                        [...    ]<br></pre>
   <br>
   And data on plane p (z=p), row r (y=r) and column c (x=c) of a 16 bits<br>
   image can be acceded through:
   <ul>((short int *)inr->data)[inr->ncols * (p * inr->nrows + y) + c]</ul>
   as well as:
   <ul>((short int ***)inr->array)[p][r][c]</ul>
   <br>
   All buffer are deallocated using <tt>freeInrimage</tt>.<br>
   A change of an inrimage data buffer can be made with <tt>setInrimageData</tt><br>
   that properly frees and reallocates <tt>array</tt> pointers.<br>
   <br>
   Function <tt>initInrimage</tt> allocates an <tt>inrimage</tt> structure<br>
   but lets <tt>data</tt> and <tt>array</t> pointers to NULL<br>.
*/

typedef inrimage INRIMAGE;

typedef inrimage *PTRINRIMAGE;


typedef struct {
  inrimage *inr;          /* Inrimage descriptor */
  ENDIANNESS endianness;  /* Inrimage file endianness */
} INRIMAGE_HEADER;
/* inrimage file descriptor */



/* if epi_kernel has not been included, define EpidaureLib data structures */
#ifndef _epi_kernel_h_

#define TYPEIM_UNKNOWN 0
/* EPILIB: not a known type of thomy images */

#define TYPEIM_BW 1
/* EPILIB:black and white picture */

#define TYPEIM_256 2
/* EPILIB: grey level image one byte encoded */

#define TYPEIM_FLOAT 3
/* EPILIB: a floating point image */

#define TYPEIM_16B 4
/* EPILIB: a 2 byte points image */

#define TYPEIM_S16B 5
/* EPILIB: a signed 2 byte points image */


#define B_FNAME_LENGTH 200
/* EPILIB: fixed file name string length */

typedef struct t_Obj {
  int type;               /* object type */
} t_Obj;
/* EPILIB: generic object */

typedef t_Obj * t_ObjPtr;

typedef t_Obj ** t_ObjHand;


typedef struct t_File{
  int     fd;                   /* unix file descriptor     */
  FILE    *fdt;           	/* ascii file descriptor    */
  int     type;           	/* type of thomy file       */
  char    name[B_FNAME_LENGTH]; /* name of the file         */
  int     mode;     		/* MODE_BIN or MODE_ASCII   */
  int     machine;        	/* machine who produce the file */
} t_File;
/* EPILIB: file description */

typedef t_File * t_FilePtr;


typedef struct t_Image{
  t_Obj obj;		   /* unused now */
  int tx,ty,tz;		   /* size of the image	*/
  char type;		   /* type of the image	*/
  char *buf;		   /* the buffer with the image	*/
  t_File fi;		   /* thomy file descr.	*/
} t_Image;
/* EPILIB: 3D image header */

typedef t_Image * t_ImagePtr;

#endif




APIEXPORT inrimage *readZInrimage(const char * name /* Image name*/);
/* Read an image in given file and return an inrimage descriptor or NULL<br>
   if the image type is not recognized */

APIEXPORT inrimage *readNonInterlacedZInrimage(const char * name /* Image name*/);
/* Read an image in given file and return an inrimage descriptor or NULL<br>
   if the image type is not recognized. If the image is vectorial, the
   image buffer is un-interlaced for a 4D access to vector components
   through the varray 4D pointer of inrimages data structure. */

APIEXPORT int writeZInrimage(const inrimage *inr /* Inrimage descriptor */,
		   const char * /* Image name */);
/* Write the inrimage described by <tt>inr</tt> in given file.<br>
   If the name ends with ".gz" extension, the file is gzipped.<br>
   Return a negative value in case of failure, 0 otherwise   */

APIEXPORT INRIMAGE_HEADER *readZInrimageHeader(const char *name /* Image name */);
/* Read the header from an inrimage file. <br>
   Inrimage <tt>data</tt> and <tt>array</tt> pointers are set to NULL. */



APIEXPORT inrimage *createInrimage(int ncols /* Number of columns (X dimension)*/,
			 int nrows /* Number of rows (Y dimension)*/,
			 int nplanes /* Number of planes (Z dimension)*/,
			 int vdim /* Vector dimension */,
			 WORDTYPE type /* Image voxel word type */);
/* Allocates an inrimage descriptor for a given image type and dimensions.<br>
   Vector dimension is 3 for RGB images, 4 for RGBA images, any value<br>
   given in <tt>vdim</tt> for WT_FLOAT_VECTOR or WT_DOUBLE_VECTOR type<br>
   and 1 for any other image type, regardless of <tt>vdim</tt> value.*/

APIEXPORT inrimage *createNonInterlacedInrimage(
			 int ncols     /* Number of columns (X dimension)*/,
			 int nrows     /* Number of rows (Y dimension)*/,
			 int nplanes   /* Number of planes (Z dimension)*/,
			 int vdim      /* Vector dimension */,
			 WORDTYPE type /* Image voxel word type */);
/* Allocates an inrimage descriptor for a given image type and dimensions.<br>
   Vector dimension is 3 for RGB images, 4 for RGBA images, any value<br>
   given in <tt>vdim</tt> for WT_FLOAT_VECTOR or WT_DOUBLE_VECTOR type<br>
   and 1 for any other image type, regardless of <tt>vdim</tt> value.
   Vectorial images are considered to be stored non interlaced and
   a the image 4D varray field is allocated. */


APIEXPORT inrimage *initInrimage(int ncols /* Number of columns (X dimension)*/,
		       int nrows /* Number of rows (Y dimension)*/,
		       int nplanes /* Number of planes (Z dimension)*/,
		       int vdim /* Vector dimension */,
		       WORDTYPE type /* Image voxel word type */);
/* Creates an inrimage descriptor for a given image type and dimensions<br>
   but does not allocate <tt>data</tt> buffer and <tt>array</tt> pointers. */

APIEXPORT void setInrimageData(inrimage *inr /* Inrimage descriptor */,
		     void *data    /* Data buffer */);
/* Changes the <tt>data</tt> field of an inrimage and set <tt>array</tt><br>
   pointers on new data. */

APIEXPORT void setNonInterlacedInrimageData(inrimage *inr /* Inrimage descriptor */,
				  void *data    /* Data buffer */);
/* Changes the <tt>data</tt> field of an inrimage and set <tt>varray</tt><br>
   pointers on new data. If image is vectorial, data buffer is considered
   to be non-interlaced. */

APIEXPORT void freeInrimage(inrimage *inr /* Inrimage descriptor */);
/* Deallocates an inrimage descriptor and associated buffers */






int writeInrimageHeader(FILE *f /* Openned file descriptor */,
			const inrimage *inr /* Image to write */);
/* Write given inrimage header in an openned file.<br>
   Return a negative value in case of failure, 0 otherwise */

int writeInrimageData(FILE *f /* Openned file descriptor */,
		      const inrimage *inr /* Image to write */);
/* Write given inrimage body in an openned file.<br>
   Return a negative value in case of failure, 0 otherwise  */

unsigned int sizeofWordtype(const WORDTYPE wt /* Word type */);
/* Return the word size of given word type */

unsigned int sizeofImageVoxel(const inrimage *inr /* Inrimage */);
/* Return the image voxel word size */

void ***index3DArray(const void *data /* Data buffer */,
		     unsigned int ncols   /* Number of columns (X dimension) */,
		     unsigned int nrows   /* Number of rows (Y dimension) */,
		     unsigned int nplanes /* Number of planes (Z dimension) */,
		     unsigned int stype   /* Size of data words */);
/* Build a 3D array of pointers on a data buffer knowing the volume<br>
   dimensions and the size of each data word. */

void ****index4DArray(const void *data /* Data buffer */,
		      unsigned int ncols   /* Number of columns (X dimension) */,
		      unsigned int nrows   /* Number of rows (Y dimension) */,
		      unsigned int nplanes /* Number of planes (Z dimension) */,
		      unsigned int vdim    /* Vectorial dimension */,
		      unsigned int stype   /* Size of data words */);
/* Build a 4D array of pointers on a data buffer knowing the volume<br>
   dimensions and the size of each data vector component. Data are
   indexed on vector dimension first, Z, Y, and X coordinate. */





t_ImagePtr readEpidaureLibInrimage(const char *name /* Image name */);
/* Reads an image from given file and returns an epidaurelib data structure. */

t_ImagePtr readEpidaureLibInrimageHeader(const char *name /* Image name */);
/* Reads an inrimage buffer form  given file and returns it in an<br>
   epidaurelib data structure. */

int writeEpidaureLibInrimage(const t_ImagePtr epimg
			     /* Epidaurelib image descriptor */);
/* Writes the inrimage described in the given epidaurelib descriptor.<br>
   Return a negative value in case of failure, 0 otherwise   */

void freeEpidaureLibInrimage(const t_ImagePtr epimg
			     /* Epidaurelib image descriptor */);
/* Free an epidaurelib inrimage. */





inrimage *_image2inrimage(const _image *img /* _image image descriptor */);
/* Convert an _image image desciptor to an inrimage image descriptor. */

_image *inrimage2_image(const inrimage *inr /* inrimage image desciptor */);
/* Convert an inrimage image desciptor to an _image image descriptor. */

inrimage *epidaureLib2Inrimage(const t_ImagePtr epimg
			       /* Epidaurelib image descriptor */);
/* Convert an epidaurelib image desciptor to an inrimage descriptor. */

t_ImagePtr inrimage2EpidaureLib(const inrimage *inr
				/* inrimage descriptor */ );
/* Convert an inrimage desciptor to an epidaurelib image descriptor. */

t_ImagePtr _image2epidaureLib(const _image *img /* _image image descriptor */);
/* Convert an _image image desciptor to an epidaurelib image descriptor. */

_image *epidaureLib2_image(const t_ImagePtr epimg
			   /* Epidaurelib image descriptor */);
/* Convert an epidaurelib _image desciptor to an image image descriptor. */



#ifdef __cplusplus
}
#endif

#endif
