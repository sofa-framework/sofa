/***************************************************************************
 * $Author: hdeling $
 * $Revision: 1.11 $
 * $Log: ImageIO.h,v $
 * Revision 1.11  2000/11/21 13:50:40  hdeling
 * after compiling on MSVC6.0
 *
 * Revision 1.10  2000/07/06 23:13:57  jmontagn
 * fixed minc reading / added minc writing capability
 *
 * Revision 1.8  1999/05/05 12:24:39  jmontagn
 * Modified _writeImageHeader to allow to fix endianness
 *
 * Revision 1.7  1999/02/19 15:27:18  jmontagn
 * doc++ified
 *
 * Revision 1.6  1999/02/19 15:03:20  jmontagn
 * Make direct use of libz.so to read/write compressed inrimages
 * rather than calling gzip through a popen().
 *
 * Revision 1.5  1999/01/14 07:22:54  jmontagn
 * Added capability to read vectorial non-interlaced inrimage data
 *
 * Revision 1.4  1998/10/16 17:17:07  jmontagn
 * Modified TCL commands management.
 * Set static commands declaration.
 * Set documentation in .C files.
 * Get rid of .doc files.
 *
 * Revision 1.3  1998/09/24 20:40:11  jmontagn
 * Added typedef for (de)allocation routines
 *
 * Revision 1.2  1998/09/15 15:23:46  jmontagn
 * Suppressed any malloc/free from C++ code
 *
 * Revision 1.1  1998/09/14 09:08:47  jmontagn
 * Moved ImageIO routines to CVS base
 *
 * Revision 1.8  1998/07/29  08:34:48  jmontagn
 * Added imageType() function
 *
 * Revision 1.7  1998/07/28  11:43:58  aguimond
 * Added analyze image format.
 * Added _openWriteImage().
 * Renamed _openImage() to _openReadImage().
 *
 * Revision 1.6  1998/07/27 09:01:57  jmontagn
 * Added RCS header keywords
 *
 *
 * $Id: ImageIO.h,v 1.11 2000/11/21 13:50:40 hdeling Exp $
 ***************************************************************************/

#ifndef IMAGEIO_H
#define IMAGEIO_H

#include <zlib.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef __APPLE__
	#include <sys/malloc.h>
#else
	#include <malloc.h>
#endif
#include <string.h>
#ifndef WIN32
#include <strings.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

#if (defined _ALPHA_ || (defined _SGI_ && (defined _64_ || defined _64_M4_ || defined _64_M3_)))
#define LONG long int
/* 64 bits integer on 64 bits processor */
#else
#define LONG long long int
/* 64 bits integer on 32 bits processor */
#endif



/** Magic header for inrimages */
#define INR_MAGIC "#INR"
/** Magic header for GIF files */
#define GIF_MAGIC "GIF8"
/** Magic header for RGB files */
#define IRIS_MAGIC 0732
/** Magic header for ANALYZE files written in little endian format */
#define ANALYZE_LE_MAGIC "\000\000\001\134"
/** Magic header for ANALYZE files written in big endian format */
#define ANALYZE_BE_MAGIC "\134\001\000\000"
/** Magic header for MINC (MNI NetCDF) file format */
#ifdef MINC_FILES
#  define MINC_MAGIC "CDF"
#endif

/** file open mode */
typedef enum {
  /** no file open */
  OM_CLOSE,
  /** file is stdin or stdout */
  OM_STD,
  /** file is gzipped */
  OM_GZ,
  /** normal file */
  OM_FILE
} OPEN_MODE;


/** kind of image word */
typedef enum {
  /** fixed type */
  WK_FIXED,
  /** floating point */
  WK_FLOAT,
  /** unknown (uninitialized) */
  WK_UNKNOWN
} WORD_KIND;


/** image word sign */
typedef enum {
  /** signed */
  SGN_SIGNED,
  /** unsigned */
  SGN_UNSIGNED,
  /** unknown (uninitialized or floating point words) */
  SGN_UNKNOWN
} SIGN;


/** endianness */
typedef enum {
  /** Little endian processor */
  END_LITTLE,
  /** Big endian processor */
  END_BIG,
  /** Unknown endianness (unopenned file) */
  END_UNKNOWN
} ENDIANNESS;


/** inrimage vectorial storage mode */
typedef enum {
  /** interlaced vectors (i.e. x1, y1, z1, x2, y2, z2, x3, y3, z3, ...) */
  VM_INTERLACED,
  /** non interlaced vectors (i.e. x1, x2, x3, ..., y1, y2, y3, ..., z1, z2, z3...) */
  VM_NON_INTERLACED,
  /** scalar inrimage */
  VM_SCALAR
} VECTORIAL_MODE;


/** image file format */
typedef enum {
  /** unknown file type */
  IF_UNKNOWN,
  /** inrimage format */
  IF_INR,
  /** gif format */
  IF_GIF,
  /** SGI iris format */
  IF_IRIS,
  /** ANALYZE format order */
  IF_ANALYZE
  /** MINC format */
#ifdef MINC_FILES
  , IF_MINC
#endif
} IMAGE_FORMAT;



/** Image descriptor */
typedef struct {
  /** Image x dimension (number of columns) */
  unsigned int xdim;
  /** Image y dimension (number of rows) */
  unsigned int ydim;
  /** Image z dimension (number of planes) */
  unsigned int zdim;
  /** Image vectorial dimension */
  unsigned int vdim;

  /** Image voxel size in x dimension */
  float vx;
  /** Image voxel size in y dimension */
  float vy;
  /** Image voxel size in z dimension */
  float vz;

  /** Image data buffer */
  void *data;

  /** Image word size (in bytes) */
  unsigned int wdim;
  /** Image format to use for I/0. Should not be set by user */
  IMAGE_FORMAT imageFormat;
  /** Data buffer vectors are interlaced or non interlaced */
  VECTORIAL_MODE vectMode;
  /** Image words kind */
  WORD_KIND wordKind;
  /** Image words sign */
  SIGN sign;

  /** User defined strings array. The user can use any internal purpose string.
      Each string is written at then end of header after a '#' character. */
  char **user;
  /** Number of user defined strings */
  unsigned int nuser;

  /** Image file descriptor */
  gzFile fd;
  /** Kind of image file descriptor */
  OPEN_MODE openMode;
  /** Written words endianness */
  ENDIANNESS endianness;
} _image;



/** Allocates and initializes an image descriptor */
_image *_initImage();

/** Free an image descriptor
    @param im image descriptor */
void _freeImage(_image *im);

/** creates an image descriptor from the given header information
    @param x image x dimension (number of columns)
    @param y image y dimension (number of rows)
    @param z image z dimension (number of planes)
    @param v image vectorial dimension
    @param vx image voxel size in x dimension
    @param vy image voxel size in y dimension
    @param vz image voxel size in z dimension
    @param w image word size in bytes
    @param wk image word kind
    @param sgn image word sign */
_image *_createImage(int x, int y, int z, int v,
		     float vx, float vy, float vz, int w,
		     WORD_KIND wk, SIGN sgn);


/** Reads an image from a file and returns an image descriptor or NULL if<br>
    reading failed.<br>
    Reads from stdin if image name is NULL.
    The image data field points to a xdim * ydim * zdim * vdim buffer
    containing voxels in order:<pre>
    (Z1, Y1, X1, V1) (Z1, Y1, X1, V2), ... , (Z1, Y1, X1, Vt),
    (Z1, Y1, X2, V1) ...         ...       , (Z1, Y1, X2, Vt),
    ...
    (Z1, Y2, X1, V1) ...         ...       , (Z1, Y2, X1, Vt),
    ...
    (Z2, Y1, X1, V1) ...         ...       , (Z2, Y1, X1, Vt),
    ...
                     ...         ...       , (Zl, Ym, Xn, Vt)</pre>
   @param name image file name or NULL for stdin */
_image* _readImage(const char *name);

/** Reads an image from a file and returns an image descriptor or NULL if<br>
    reading failed.<br>
    Reads from stdin if image name is NULL.
    If the image is vectorial, it is uninterlaced, i.e. the image data
    field points to a xdim * ydim * zdim * vdim buffer containing voxels
    in order:<pre>
     (V1, Z1, Y1, X1) (V1, Z1, Y1, X2), ... , (V1, Z1, Y1, Xn),
     (V1, Z1, Y2, X1) ...         ...       , (V1, Z1, Y2, Xn),
     ...
     (V1, Z2, Y1, X1) ...         ...       , (V1, Z2, Y1, Xn),
     ...
     (V2, Z1, Y1, X1) ...         ...       , (V2, Z1, Y1, Xn),
     ...
                      ...         ...       , (Vt, Zl, Ym, Xn)</pre>
   @param name image file name or NULL */
_image* _readNonInterlacedImage(const char *name);

/** Writes given image in file 'name'.<br>
    If name ends with '.gz', file is gzipped.<br>
    If name is NULL, image is sent to stdout.
    @param im image descriptor 
    @param name file name to store image or NULL */
int _writeImage(_image *im, const char *name);

/** Read one slice of given image whose header has already been read.<br>
    File descriptor is let at the beginning of next slice and closed<br>
    when end of file is encountered.<br>
    If data buffer is NULL, it is allocated for one slice only.<br>
    This funtion is dedicated to read huge inrimages.
    @param im image descriptor */
void _getNextSlice(_image *im);



/** Reads header from an image file<br>
    If file is an inrimage, only header is read. Otherwise, whole image<br>
    is read and image file descriptor is closed.<br>
    If name is NULL, header is read from STDIN
    @param name image file name or NULL */
_image* _readImageHeader(const char *name);

/** Reads body from an inrmage whose header has been read by
    _readImageHeader
    @param im image to read */
int _readImageData(_image *im);

/** Reads body from a vectorial inrimage whose header has been read by
    _readImageHeader. The image is uninterlaced
    (see _readNonInterlacedImage for details).
    @param im image descriptor*/
int _readNonInterlacedImageData(_image *im);

/** Reads body from a non-interlaced vectorial inrimage whose header has
    been read by _readImageHeader. The image buffer is interlaced.
    @param im image descriptor */
int _readNonInterlacedFileData(_image *im);

/** Writes the given inrimage header in an already opened file.
    @param im image descriptor
    @param end image data endianness (END_UNKNOWN to use architecture endianness) */
int _writeInrimageHeader(const _image *im,
			 ENDIANNESS end);

/** Writes the given image body in an already opened file.
    @param im image descriptor */
int _writeInrimageData(const _image *im);



/** given an initialized file descriptor and a file name, open file
   from stdout (if name == NULL), a gziped pipe (if file is gziped)
   or a standard file otherwise.
   @param im initialized image descriptor
   @param name image file name */
void _openWriteImage(_image* im, const char *name) ;
   
/** open an image file from stdin (if name == NULL), from a pipe
   (piped with gzip if image was compressed) or from a standard file
   @param im initialized image descriptor
   @param name image file name */
void _openReadImage(_image *im, const char *name);

/** close an image file descriptor that was opened using _openImage
    @param im opened image descriptor */
void _closeImage(_image *im);

/** read header from an opened inrimage file
    @param im opened inrmage descriptor */
int readInrimageHeader(_image *im);


/** return image type in given file
    @param fileName image file name */
IMAGE_FORMAT imageType(const char *fileName);


/** function prototype to allocate memory */
typedef void *(*ALLOCATION_FUNCTION)(size_t);

/** function prototype to free memory */
typedef void (*DEALLOCATION_FUNTION)(void *);


/** set allocation and deallocation routines
    @param alloc new allocation routine
    @param del new deallocation routine */
void setImageIOAllocationRoutines(ALLOCATION_FUNCTION alloc,
				  DEALLOCATION_FUNTION del);



/** call allocation routine */
void *ImageIO_alloc(size_t);
/** call deallocation routine */
void ImageIO_free(void *);

/** replaces fwrite function
    @param im image to write
    @param buf data buffer to write
    @param buffer length */
size_t ImageIO_write(const _image *im, const void *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif
