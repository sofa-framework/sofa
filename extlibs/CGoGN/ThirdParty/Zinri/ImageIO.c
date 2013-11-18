/***************************************************************************
 * $Author: hdeling $
 * $Revision: 1.25 $
 * $Log: ImageIO.c,v $
 * Revision 1.25  2001/01/05 17:02:12  hdeling
 * now saving inrimage as binary files (crucial on Windows NT)
 *
 * Revision 1.24  2000/12/08 14:08:46  hdeling
 * corrected GIF image reading routine
 *
 * Revision 1.23  2000/11/21 16:17:10  hdeling
 * avoid include Bool.h
 *
 * Revision 1.22  2000/11/21 13:42:57  hdeling
 * after compiling on MSVC6.0
 *
 * Revision 1.21  2000/09/01 16:28:17  hdeling
 * bug corrected that created a memory leak when reading an Inrimage
 * header
 *
 * Revision 1.20  2000/07/11 15:14:43  jmontagn
 * fixed minc reading / added minc writing capability
 *
 * Revision 1.17  1999/08/28 16:22:47  jmontagn
 * translated code for g++ 2.95.1 release
 *
 * Revision 1.16  1999/05/19 16:55:29  aguimond
 * Added analyse type in imageType().
 *
 * Revision 1.15  1999/05/15 17:17:00  jmontagn
 * Fixed writing on stdout
 *
 * Revision 1.14  1999/05/10 21:41:15  aguimond
 * corrected bug with length of buffer to switch endian in vectorial images.
 *
 * Revision 1.13  1999/05/05 12:24:36  jmontagn
 * Modified _writeImageHeader to allow to fix endianness
 *
 * Revision 1.12  1999/03/25 11:33:42  aguimond
 * corrected bug when saving noninterlaced images.
 *
 * Revision 1.11  1999/03/15 14:21:36  jmontagn
 * Fixed inrmage reading from stdin
 *
 * Revision 1.10  1999/02/19 15:03:16  jmontagn
 * Make direct use of libz.so to read/write compressed inrimages
 * rather than calling gzip through a popen().
 *
 * Revision 1.9  1999/01/14 07:22:51  jmontagn
 * Added capability to read vectorial non-interlaced inrimage data
 *
 * Revision 1.8  1998/11/26 08:31:54  jmontagn
 * Used ImageIO_alloc and ImageIO_free routines
 *
 * Revision 1.7  1998/10/16 17:16:59  jmontagn
 * Modified TCL commands management.
 * Set static commands declaration.
 * Set documentation in .C files.
 * Get rid of .doc files.
 *
 * Revision 1.6  1998/09/30 09:39:46  jmontagn
 * It seems there is a bug in eg++: when using exception in class
 * constructor initializer (A::A try : B() { ... } catch(...) {...}) and
 * if A declared a destructor, unexpected() is called for any exception
 * thrown in B at the end of ~A. I suppress Inrimage constructor
 * exceptions and I put everything in BaseInrimage with a -DLIBINRIMAGE
 * flag.
 *
 * Revision 1.5  1998/09/24 20:39:39  jmontagn
 * Added typedef for (de)allocation routines
 *
 * Revision 1.4  1998/09/21 14:32:08  jmontagn
 * Improved exception management.
 *
 * Revision 1.3  1998/09/15 15:23:42  jmontagn
 * Suppressed any malloc/free from C++ code
 *
 * Revision 1.2  1998/09/14 09:08:41  jmontagn
 * Moved ImageIO routines to CVS base
 *
 * Revision 1.1.1.1  1998/07/31 20:09:12  jmontagn
 * Initialized libInrimage module
 *
 * Revision 1.6  1998/07/29  08:34:48  jmontagn
 * Added imageType() function
 *
 * Revision 1.5  1998/07/28  11:43:58  aguimond
 * Added analyze image format.
 * Added _openWriteImage().
 * Renamed _openImage() to _openReadImage().
 *
 * Revision 1.4  1998/07/27 09:00:31  jmontagn
 * fixed return value for _writeInrimageData when writting non-interlaced vectorial images
 *
 *
 * $Id: ImageIO.c,v 1.25 2001/01/05 17:02:12 hdeling Exp $
 ***************************************************************************/

#include "ImageIO.h"
#include "gif.h"
#include "iris.h"
#include "analyze.h"
#ifdef MINC_FILES
#  include "mincio.h"
#endif


#if (defined (_ALPHA_) || defined (_LINUX_) || defined(WIN32))
#define ARCHITECTURE_ENDIANNESS END_LITTLE
#else
#define ARCHITECTURE_ENDIANNESS END_BIG
#endif
/* compile time endianness */


#define INR4_MAGIC "#INRIMAGE-4#{"
/* Magic header for inrimages v4 */


typedef struct stringListElementStruct {
  char *string;
  struct stringListElementStruct *next;
} stringListElement;
/* list element with a pointer on a string */

typedef struct {
  stringListElement *begin, *end;
} stringListHead;
/* string list descriptor */

static void addStringElement(stringListHead *strhead,
			     const char *str);
/* add a string element at the tail of given list */

static void concatStringElement(const stringListHead *strhead,
				const char *str);
/* concat given string at the last element of given list */

static char *fgetns(char *str, int n, gzFile f);
/* get a string from a file and discard the ending newline character
   if any */


/* default allocation routine */
static void *(*allocRoutine)(size_t) = 0;
/* default deallocation routine */
static void (*deleteRoutine)(void *) = 0;

void *ImageIO_alloc(size_t s) {
  if(!allocRoutine) allocRoutine = malloc;
  return (*allocRoutine)(s);
}
/* call allocation routine */
void ImageIO_free(void *m) {
  if(!deleteRoutine) deleteRoutine = free;
  (*deleteRoutine)(m);
}
/* call deallocation routine */


size_t ImageIO_write(const _image *im, const void *buf, size_t len) {
  switch(im->openMode) {
  case OM_FILE:
  case OM_STD:
    return fwrite(buf, 1, len, (FILE *) im->fd);
  case OM_GZ:
    return gzwrite(im->fd, (void *) buf, len);
  case OM_CLOSE:
  default:
    return 0;
  }
}



/* set allocation and deallocation routines */
void setImageIOAllocationRoutines(ALLOCATION_FUNCTION alloc,
				  DEALLOCATION_FUNTION del) {
  if(alloc != NULL) allocRoutine = alloc;
  if(del != NULL) deleteRoutine = del;
}


/* Allocates and initializes an image descriptor */
_image *_initImage() {
  _image *im;

  im = (_image *) ImageIO_alloc(sizeof(_image));

  /* default image size is 1*1*1 */
  im->xdim = im->ydim = im->zdim = im->vdim = 1;
  /* default image voxel size is 1.0*1.0*1.0 */
  im->vx = im->vy = im->vz = 1.0;

  /* no data yet */
  im->data = NULL;

  /* no file associated to image */
  im->fd = NULL;
  im->openMode = OM_CLOSE;
  im->endianness = END_UNKNOWN;

  /* no user string */
  im->user = NULL;
  im->nuser = 0;

  /* unknown word kind */
  im->wdim = 0;
  im->wordKind = WK_UNKNOWN;
  im->vectMode = VM_SCALAR;
  im->sign = SGN_UNKNOWN;
  im->imageFormat = IF_UNKNOWN;

  /* return image descriptor */
  return im;
}



/* Free an image descriptor */
void _freeImage(_image *im) {
  unsigned int i;

  /* close image if opened */
  if(im->openMode != OM_CLOSE) _closeImage(im);

  /* free data if any */
  if(im->data) ImageIO_free(im->data);

  /* free user string array if any */
  if(im->nuser > 0) {
    for(i = 0; i < im->nuser; i++)
      ImageIO_free(im->user[i]);
    ImageIO_free(im->user);
  }

  /* free given descriptor */
  ImageIO_free(im);
}




/* creates an image descriptor from the given header information */
_image *_createImage(int x, int y, int z, int v,
		     float vx, float vy, float vz,
		     int w, WORD_KIND wk, SIGN sgn) {
  _image *im;

  im = (_image *) ImageIO_alloc(sizeof(_image));

  /* image size */
  im->xdim = x;
  im->ydim = y;
  im->zdim = z;
  im->vdim = v;
  /* image voxel size */
  im->vx = vx;
  im->vy = vy;
  im->vz = vz;

  /* no file associated to image */
  im->fd = NULL;
  im->openMode = OM_CLOSE;
  im->endianness = END_UNKNOWN;
  im->imageFormat = IF_INR;

  /* no user string */
  im->user = NULL;
  im->nuser = 0;

  /* unknown word kind */
  im->wdim = w;
  im->wordKind = wk;
  im->sign = sgn;

  /* default vectors setting: interlaced */
  if(im->vdim == 1)
    im->vectMode = VM_SCALAR;
  else
    im->vectMode = VM_INTERLACED;

  /* no data yet */
  im->data = ImageIO_alloc(x * y * z * v * w);
  memset((char *) im->data,0, x * y * z * v * w);

  /* return image descriptor */
  return im;
}







/* Reads an image from a file and returns an image descriptor or NULL if
   reading failed.
   Reads from stdin if image name is NULL. */
_image* _readImage(const char *name) {
  _image *im;

  /* read header */
  im = _readImageHeader(name);

  if(im && im->openMode != OM_CLOSE) {
    /* read body */
    if(_readImageData(im) < 0) {
      fprintf(stderr, "_readImage: error: invalid data encountered in \'%s\'\n",
	      name);
      _freeImage(im);
      return NULL;
    }
    _closeImage(im);
  }

  return im;
}


/* Reads an image from a file and returns an image descriptor or NULL if<br>
   reading failed.<br>
   Reads from stdin if image name is NULL.
   If the image is vectorial, it is uninterlaced. */
_image* _readNonInterlacedImage(const char *name) {
  _image *im;

  /* read header */
  im = _readImageHeader(name);

  if(im && im->openMode != OM_CLOSE) {
    /* read scalar image body */
    if(im->vdim == 1) {
      if(_readImageData(im) < 0) {
	fprintf(stderr, "_readImage: error: invalid data encountered in \'%s\'\n",
		name);
	_freeImage(im);
	return NULL;
      }
    }
    /* read vectorial image body */
    else {
      im->vectMode = VM_NON_INTERLACED;
      if(_readNonInterlacedImageData(im) < 0) {
	fprintf(stderr, "_readImage: error: invalid data encountered in \'%s\'\n",
		name);
	_freeImage(im);
	return NULL;
      }
    }
    _closeImage(im);
   }

  return im;
}


/* Write inrimage given in inr in file name. If file name's suffix is
   .gz, the image is gziped. If file name's suffix is .hdr, the image
   is written in ANALYZE format. If file name is NULL, image is written
   on stdout */
int _writeImage(_image *im, const char *name) {
  int r = 1;

  /* open file descriptor */
  _openWriteImage( im, name ) ;

  if(!im->fd) {
     fprintf(stderr, "_writeImage: error: open failed\n");
     return -1;
  }

  switch( im->imageFormat ) {
     case IF_ANALYZE:
	/* write header */
	if(writeAnalyzeHeader(im) < 0) {
	   fprintf(stderr, "_writeImage: error: unable to write header of \'%s\'\n",
		   name);
	   r = -1;
	}
	else
	{
	   if( name != NULL )
	   {
	      int length = strlen(name) ;
	      char* data_filename =(char *) ImageIO_alloc(length+1) ;
	      if( strcmp( name+length-4, ".hdr" ) )
	      {
		 fprintf (stderr,
			  "_writeImage: error: file header extention must be .hdr\n");
		 _freeImage(im);
		 return 0;
	      }
	      
	      strcpy(data_filename,name);
	      strcpy(data_filename+length-3, "img");
	      
	      _closeImage(im);
	      _openWriteImage(im, data_filename);
	      ImageIO_free(data_filename);
	      
	      if(!im->fd) {
		 fprintf(stderr, "readAnalyzeHeader: error: unable to open file \'%s\'\n", data_filename);
		 _freeImage(im);
		 return 0;
	      }
	   }
	   
	   /* write body */
	   if(writeAnalyzeData(im) < 0) {
	      fprintf(stderr, "_writeImage: error: unable to write data of \'%s\'\n",
		      name);
	      r = -1;
	   }
	}
	break ;

     case IF_INR:
     case IF_GIF:
     case IF_IRIS:
     default:
	/* write header */
	if(_writeInrimageHeader(im, END_UNKNOWN) < 0) {
	   fprintf(stderr, "_writeImage: error: unable to write header of \'%s\'\n",
		   name);
	   r = -1;
	}
	
	/* write body */
	else if(_writeInrimageData(im) < 0) {
	   fprintf(stderr, "_writeImage: error: unable to write data of \'%s\'\n",
		   name);
	   r = -1;
	}
	break ;
  }


  /* close file descriptor */
  switch(im->openMode) {
  case OM_STD:
  case OM_CLOSE:
    break;
  case OM_GZ:
    gzclose(im->fd);
    break;
  case OM_FILE:
    fclose((FILE *) im->fd);
    break;
  }

  im->fd = NULL;
  im->openMode = OM_CLOSE;

  return r;
}



/* read header from an image file */
_image *_readImageHeader(const char *name) {
  _image *im;
  char magic[5];
  int res;

  /* open image file */
  im = _initImage();
  _openReadImage(im, name);	

  if(!im->fd) {
    fprintf(stderr, "_readImageHeader: error: unable to open file \'%s\'\n", name);
    _freeImage(im);
    return NULL;
  }

  else {
    if(im->openMode != OM_STD) {
      gzread(im->fd, magic, 4);
      magic[4] = '\0';
      gzseek(im->fd, 0L, SEEK_SET);
    }

    /* openned image is an inrimage */
    if(im->openMode == OM_STD || !strcmp(magic, INR_MAGIC)) {
      if(readInrimageHeader(im) < 0) {
	fprintf(stderr, "_readImageHeader: error: invalid inrimage header encountered in \'%s\'\n", name);
	_freeImage(im);
	return NULL;
      }
      im->imageFormat = IF_INR;
    }

    /* opened image is an ANALYZE */
    else if(!memcmp(magic,ANALYZE_LE_MAGIC,4) ||
	    !memcmp(magic,ANALYZE_BE_MAGIC,4))
    {
       if( !readAnalyzeHeader(im,name) ) {
	  fprintf(stderr, "_readImageHeader: error: invalid ANALYZE header encountered in \'%s\'\n", name);
	  _freeImage(im);
	  return NULL;
       }
       im->imageFormat = IF_ANALYZE;
    }    

#ifdef MINC_FILES
    else if(!memcmp(magic, MINC_MAGIC, 3)) {
      double startx, starty, startz, stepx, stepy, stepz;
      double Xcosine[3], Ycosine[3], Zcosine[3];

      if( !readMincHeader(im, name,
			  &startx, &starty, &startz, &stepx, &stepy, &stepz,
			  Xcosine, Ycosine, Zcosine) ) {
	fprintf(stderr, "_readImageHeader: error: invalid MINC header encountered in \'%s\'\n", name);
	_freeImage(im);
	return NULL;
      }
      im->imageFormat = IF_MINC;
    }
#endif

    else {

      /* opened image is a GIF: full read */
      if(!strcmp(magic, GIF_MAGIC)) {
	 res = readGifImage(im,name);
	 im->imageFormat = IF_GIF;
      }

      /* opened image is an RGB: full read */
      else if((((unsigned char *)magic)[0]<<8) + ((unsigned char *)magic)[1]
	      == IRIS_MAGIC) {
	res = readIrisImage(im);
	im->imageFormat = IF_IRIS;
      }

      /* unknown image type */
      else {
	fprintf(stderr,
		"_readImageHeader: error: unknown image type \'%s\'\n", name);
	_freeImage(im);
	return NULL;
      }

      /* close image file and return */
      if(res > 0) {
	_closeImage(im);
	im->fd = NULL;
	im->openMode = OM_CLOSE;
      }
      else {
	_freeImage(im);
	fprintf(stderr,
		"_readImageHeader: error: invalid file \'%s\' encountered\n", name);
	return NULL;
      }

    }

  }

  return im;
}



/* Read data of an inrimage.
   If im->data is not NULL, assume that the buffer was previously allocated
   Swap bytes depending on the endianness and the current architecture  */
int _readImageData(_image *im) {
  unsigned int size, nread, length;
  unsigned char *ptr1, *ptr2, b[8];

  if(im->openMode != OM_CLOSE) {
    size = im->xdim * im->ydim * im->zdim * im->vdim * im->wdim;

    if(!im->data) {
      im->data = (unsigned char *) ImageIO_alloc(size);
      if(!im->data) return -2;
    }

    nread = gzread(im->fd, im->data, size);
    if(nread != size) return -1;

    /* architecture is big endian and data little endian */
    if(ARCHITECTURE_ENDIANNESS != im->endianness) {
      
      length = size / im->wdim;
      ptr1 = ptr2 = (unsigned char *) im->data;
	
      /* 2 bytes swap */
      if(im->wdim == 2) {
	while(length--) {
	  b[0] = *ptr1++;
	  b[1] = *ptr1++;
	  *ptr2++ = b[1];
	  *ptr2++ = b[0];
	}
      }
      
      /* 4 bytes swap */
      else if(im->wdim == 4) {
	while(length--) {
	  b[0] = *ptr1++;
	  b[1] = *ptr1++;
	  b[2] = *ptr1++;
	  b[3] = *ptr1++;
	  *ptr2++ = b[3];
	  *ptr2++ = b[2];
	  *ptr2++ = b[1];
	  *ptr2++ = b[0];
	}
      }
      
      /* 8 bytes swap */
      else if(im->wdim == 8) {
	while(length--) {
	  b[0] = *ptr1++;
	  b[1] = *ptr1++;
	  b[2] = *ptr1++;
	  b[3] = *ptr1++;
	  b[4] = *ptr1++;
	  b[5] = *ptr1++;
	  b[6] = *ptr1++;
	  b[7] = *ptr1++;
	  *ptr2++ = b[7];
	  *ptr2++ = b[6];
	  *ptr2++ = b[5];
	  *ptr2++ = b[4];
	  *ptr2++ = b[3];
	  *ptr2++ = b[2];
	  *ptr2++ = b[1];
	  *ptr2++ = b[0];
	}
      }
    }

    /* reorder lines */
    if( im->imageFormat == IF_ANALYZE ) {
       int lineSize = im->wdim * im->xdim * im->vdim ;
       char* buf1 = (char *)im->data ;
       char* buf2 = buf1 + lineSize * im->ydim * im->zdim - lineSize ;
       char* swapped = (char *) ImageIO_alloc(lineSize) ;
       
       while( buf1 < buf2 )
       {
	  memcpy( swapped, buf1, lineSize ) ;
	  memcpy( buf1, buf2, lineSize ) ;
	  memcpy( buf2, swapped, lineSize ) ;
	  buf1 += lineSize ;
	  buf2 -= lineSize ;
       }

       ImageIO_free( swapped ) ;
    }
  }

  return 1;
}



/* Read data of a vectorial inrimage, making the resulting buffer non-
   inerlaced.
   If im->data is not NULL, assume that the buffer was previously allocated
   Swap bytes depending on the endianness and the current architecture. */
int _readNonInterlacedImageData(_image *im) {
  unsigned int size, nread, length;
  unsigned char *ptr1, *ptr2, b[8], **vp, *buf;
  unsigned int i, j, k, v, w;

  if(im->vdim == 1) return _readImageData(im);

  if(im->openMode != OM_CLOSE) {
    size = im->xdim * im->ydim * im->zdim * im->vdim * im->wdim;

    if(!im->data) {
      im->data = (unsigned char *) ImageIO_alloc(size);
      if(!im->data) return -2;
    }

    vp = (unsigned char **) ImageIO_alloc(im->vdim * sizeof(unsigned char *));
    buf = (unsigned char *) ImageIO_alloc(im->vdim * im->wdim);
    size = im->xdim * im->ydim * im->zdim * im->wdim;
    for(v = 0; v < im->vdim; v++)
      vp[v] = (unsigned char *) im->data + v * size;

    for(k = 0; k < im->zdim; k++) {
      for(j = 0; j < im->ydim; j++) {
	for(i = 0; i < im->xdim; i++) {
	  nread = gzread(im->fd, buf, im->vdim * im->wdim);
	  if(nread != im->vdim * im->wdim) return -1;
	  for(v = 0; v < im->vdim; v++)
	    for(w = 0; w < im->wdim; w++)
	      *vp[v]++ = *buf++;
	  buf -= im->vdim * im->wdim;
	}
      }
    }

    ImageIO_free(buf);
    ImageIO_free(vp);

    /* architecture is big endian and data little endian */
    if(ARCHITECTURE_ENDIANNESS != im->endianness) {
      
      length = im->xdim * im->ydim * im->zdim * im->vdim;
      ptr1 = ptr2 = (unsigned char *) im->data;
	
      /* 2 bytes swap */
      if(im->wdim == 2) {
	while(length--) {
	  b[0] = *ptr1++;
	  b[1] = *ptr1++;
	  *ptr2++ = b[1];
	  *ptr2++ = b[0];
	}
      }
      
      /* 4 bytes swap */
      else if(im->wdim == 4) {
	while(length--) {
	  b[0] = *ptr1++;
	  b[1] = *ptr1++;
	  b[2] = *ptr1++;
	  b[3] = *ptr1++;
	  *ptr2++ = b[3];
	  *ptr2++ = b[2];
	  *ptr2++ = b[1];
	  *ptr2++ = b[0];
	}
      }
      
      /* 8 bytes swap */
      else if(im->wdim == 8) {
	while(length--) {
	  b[0] = *ptr1++;
	  b[1] = *ptr1++;
	  b[2] = *ptr1++;
	  b[3] = *ptr1++;
	  b[4] = *ptr1++;
	  b[5] = *ptr1++;
	  b[6] = *ptr1++;
	  b[7] = *ptr1++;
	  *ptr2++ = b[7];
	  *ptr2++ = b[6];
	  *ptr2++ = b[5];
	  *ptr2++ = b[4];
	  *ptr2++ = b[3];
	  *ptr2++ = b[2];
	  *ptr2++ = b[1];
	  *ptr2++ = b[0];
	}
      }
    }
    
    /* reorder lines */
    /* no non-interlaced data for ANALYZE. But if ever... */
/*     if( im->imageFormat == IF_ANALYZE ) { */
/*        int v ; */
/*        int vdim = im->vdim ; */
/*        int lineSize = im->wdim * im->xdim ; */
/*        int vsize = lineSize * im->ydim * im->zdim ; */
/*        char* swapped = ImageIO_alloc(lineSize) ; */
/*        for( v = 0 ; v < vdim ; ++v ) */
/*        { */
/* 	  char* buf1 = (char*)im->data + v*vsize ; */
/* 	  char* buf2 = buf1 + vsize - lineSize ; */
	  
/* 	  while( buf1 < buf2 ) */
/* 	  { */
/* 	     memcpy( swapped, buf1, lineSize ) ; */
/* 	     memcpy( buf1, buf2, lineSize ) ; */
/* 	     memcpy( buf2, swapped, lineSize ) ; */
/* 	     buf1 += lineSize ; */
/* 	     buf2 -= lineSize ; */
/* 	  } */

/* 	  ImageIO_free( swapped ) ; */
/*        } */
/*     } */
  }

  return 1;
}


/* Reads body from a non-interlaced vectorial inrimage whose header has
   been read by _readImageHeader. The image buffer is interlaced. */
int _readNonInterlacedFileData(_image *im) {
  unsigned int size, nread, length;
  unsigned char *ptr1, *ptr2, b[8], *vp, *buf;
  unsigned int i, j, k, v, w;

  if(im->vdim == 1) return _readImageData(im);

  if(im->openMode != OM_CLOSE) {
    size = im->xdim * im->ydim * im->zdim * im->vdim * im->wdim;

    if(!im->data) {
      im->data = (unsigned char *) ImageIO_alloc(size);
      if(!im->data) return -2;
    }

    size = im->xdim * im->ydim * im->zdim * im->wdim;
    buf = ptr1 = (unsigned char *) ImageIO_alloc(size);

    for(v = 0; v < im->vdim; v++) {
      buf = ptr1;
      nread = gzread(im->fd, buf, size);
      if(nread != size) return -1;
      vp = (unsigned char *) im->data + (v * im->wdim);
      for(k = 0; k < im->zdim; k++) {
	for(j = 0; j < im->ydim; j++) {
	  for(i = 0; i < im->xdim; i++) {
	    for(w = 0; w < im->wdim; w++) *vp++ = *buf++;
	    vp += (im->vdim - 1) * im->wdim;
	  }
	}
      }
    }

    ImageIO_free(buf);

    /* architecture is big endian and data little endian */
    if(ARCHITECTURE_ENDIANNESS != im->endianness) {
      
      length = size / im->wdim;
      ptr1 = ptr2 = (unsigned char *) im->data;
	
      /* 2 bytes swap */
      if(im->wdim == 2) {
	while(length--) {
	  b[0] = *ptr1++;
	  b[1] = *ptr1++;
	  *ptr2++ = b[1];
	  *ptr2++ = b[0];
	}
      }
      
      /* 4 bytes swap */
      else if(im->wdim == 4) {
	while(length--) {
	  b[0] = *ptr1++;
	  b[1] = *ptr1++;
	  b[2] = *ptr1++;
	  b[3] = *ptr1++;
	  *ptr2++ = b[3];
	  *ptr2++ = b[2];
	  *ptr2++ = b[1];
	  *ptr2++ = b[0];
	}
      }
      
      /* 8 bytes swap */
      else if(im->wdim == 8) {
	while(length--) {
	  b[0] = *ptr1++;
	  b[1] = *ptr1++;
	  b[2] = *ptr1++;
	  b[3] = *ptr1++;
	  b[4] = *ptr1++;
	  b[5] = *ptr1++;
	  b[6] = *ptr1++;
	  b[7] = *ptr1++;
	  *ptr2++ = b[7];
	  *ptr2++ = b[6];
	  *ptr2++ = b[5];
	  *ptr2++ = b[4];
	  *ptr2++ = b[3];
	  *ptr2++ = b[2];
	  *ptr2++ = b[1];
	  *ptr2++ = b[0];
	}
      }
    }
    
    /* reorder lines */
    /* no non-interlaced data for ANALYZE. But if ever... */
/*     if( im->imageFormat == IF_ANALYZE ) { */
/*        int v ; */
/*        int vdim = im->vdim ; */
/*        int lineSize = im->wdim * im->xdim ; */
/*        int vsize = lineSize * im->ydim * im->zdim ; */
/*        char* swapped = ImageIO_alloc(lineSize) ; */
/*        for( v = 0 ; v < vdim ; ++v ) */
/*        { */
/* 	  char* buf1 = (char*)im->data + v*vsize ; */
/* 	  char* buf2 = buf1 + vsize - lineSize ; */
	  
/* 	  while( buf1 < buf2 ) */
/* 	  { */
/* 	     memcpy( swapped, buf1, lineSize ) ; */
/* 	     memcpy( buf1, buf2, lineSize ) ; */
/* 	     memcpy( buf2, swapped, lineSize ) ; */
/* 	     buf1 += lineSize ; */
/* 	     buf2 -= lineSize ; */
/* 	  } */

/* 	  ImageIO_free( swapped ) ; */
/*        } */
/*     } */
  }

  return 1;  
}




/* Writes the given inrimage header in an already opened file.*/
int _writeInrimageHeader(const _image *im, ENDIANNESS end) {
  unsigned int pos, i;
  char type[30], endianness[5], buf[257], scale[20];

  if(im->openMode != OM_CLOSE) {
    /* fix word kind */
    switch(im->wordKind) {

    case WK_FLOAT:
      sprintf(type, "float");
      scale[0] = '\0';
      break;

    case WK_FIXED:
      switch(im->sign) {
      case SGN_SIGNED:
	sprintf(type, "signed fixed");
	break;
      case SGN_UNSIGNED:
	sprintf(type, "unsigned fixed");
	break;
      default:
	return -1;
      }
      sprintf(scale, "SCALE=2**0\n");
      break;
      
    default:
      return -1;
    }
    
    switch(end) {
    case END_LITTLE:
      sprintf(endianness, "decm");
      break;
    case END_BIG:
      sprintf(endianness, "sun");
      break;
    default:
      /* fix architecture endianness */
      if(ARCHITECTURE_ENDIANNESS == END_LITTLE)
	sprintf(endianness, "decm");
      else
	sprintf(endianness, "sun");
      break;
    }

    /* write header information */
    sprintf(buf, "%s\nXDIM=%i\nYDIM=%i\nZDIM=%i\nVDIM=%d\nTYPE=%s\nPIXSIZE=%i bits\n%sCPU=%s\nVX=%f\nVY=%f\nVZ=%f\n",
	    INR4_MAGIC, im->xdim, im->ydim, im->zdim, im->vdim,
	    type, im->wdim*8, scale, endianness, im->vx, im->vy, im->vz);
    pos = strlen(buf);  
    if(ImageIO_write(im, buf, strlen(buf)) == EOF) return -1;
    
    
    /* write user strings */
    for(i = 0; i < im->nuser; i++) {
      pos += strlen(im->user[i]) + 2;
      if(ImageIO_write(im, "#", 1) == EOF) return -1;
      if(ImageIO_write(im, im->user[i], strlen(im->user[i])) == EOF) return -1;
      if(ImageIO_write(im, "\n", 1) == EOF) return -1;
    }

    /* write end of header */
    pos = pos % 256;
    if(pos > 252) {
      for(i = pos; i < 256; i++)
	if(ImageIO_write(im, "\n", 1) != 1) return -1;
      pos = 0;
    }
    buf[0] = '\0';
    for(i = pos; i < 252; i++) strcat(buf, "\n");
    strcat(buf, "##}\n");
    
    if(ImageIO_write(im, buf, strlen(buf)) == EOF) return -1;
    else return 1;
  }

  else return -1;
}



/* Writes the given image body in an already opened file.*/
int _writeInrimageData(const _image *im) {
  unsigned int size, nbv, nwrt, i;
  unsigned int v;
  unsigned char **vp;
  
  if(im->openMode != OM_CLOSE) {

    /* scalar or interlaced vectors */
    if(im->vectMode != VM_NON_INTERLACED) {
      size = im->xdim * im->ydim * im->zdim * im->vdim * im->wdim;
      nwrt = ImageIO_write(im, im->data, size);
      if(nwrt != size) return -1;
      else return 1;
    }

    /* non interlaced vectors: interlace for saving */
    else {
      nbv = im->xdim * im->ydim * im->zdim;
      size = im->xdim * im->ydim * im->zdim * im->wdim;
      vp = (unsigned char **) ImageIO_alloc(im->vdim * sizeof(unsigned char *));
      for(v = 0; v < im->vdim; v++)
	vp[v] = (unsigned char *) im->data + v * size;
      for(i = 0; i < nbv; i++)
	for(v = 0; v < im->vdim; v++) {
	  if(ImageIO_write(im, (const void *) vp[v], im->wdim) != im->wdim)
	    return -1;
	  vp[v] += im->wdim;
	}
      ImageIO_free(vp);
      return 1;
    }
  }
  else return -1;
}



/* Read one slice of given image whose header has already been read.
   File descriptor is let at the beginning of next slice and closed
   when end of file is encountered.
   If data buffer is NULL, it is allocated for one slice only.
   This funtion is dedicated to read huge inrimages. */
void _getNextSlice(_image *im) {
  unsigned int size, nread;
  int i;

  if(im->openMode != OM_CLOSE) {

    size = im->xdim * im->ydim * im->vdim * im->wdim;

    if(!im->data)
      im->data = ImageIO_alloc(size);

    /* read a plane */
    nread = fread(im->data, 1, size, (FILE *)im->fd);

    if(nread != size) {
      fprintf(stderr, "_getNextSlice: error: truncated file\n");
      _closeImage(im);
    }
    else {
      /* test end of file */
      i = fgetc((FILE *)im->fd);
      if(i == EOF) _closeImage(im);
      else ungetc(i, (FILE *)im->fd);
    }
    
    /* reorder lines */
    if( im->imageFormat == IF_ANALYZE ) {
       int lineSize = im->wdim * im->xdim * im->vdim ;
       char* buf1 = (char *)im->data ;
       char* buf2 = buf1 + lineSize * im->ydim - lineSize ;
       char* swapped = (char *) ImageIO_alloc(lineSize) ;
       
       while( buf1 < buf2 )
       {
	  memcpy( swapped, buf1, lineSize ) ;
	  memcpy( buf1, buf2, lineSize ) ;
	  memcpy( buf2, swapped, lineSize ) ;
	  buf1 += lineSize ;
	  buf2 -= lineSize ;
       }

       ImageIO_free( swapped ) ;
    }
  }
}



/* read header of an opened inrimage */
int readInrimageHeader(_image *im) {
  char str[257];
  int n, nusr;
  stringListHead strl = { NULL, NULL };
  stringListElement *oel, *el;

  if(im->openMode != OM_CLOSE) {
    /* read image magic number */
    if(!fgetns(str, 257, im->fd)) return -1;
    if(strcmp(str, INR4_MAGIC)) return -1;

    /* while read line does not begin with '#' or '\n', read line
       and decode field */
    if(!fgetns(str, 257, im->fd)) return -1;
    while(str[0] != '#' && str[0] != '\0') {
      if(!strncmp(str, "XDIM=", 5)) {
	if(sscanf(str+5, "%i", &im->xdim) != 1) return -1;
      }
      else if(!strncmp(str, "YDIM=", 5)) {
	if(sscanf(str+5, "%i", &im->ydim) != 1) return -1;
      }
      else if(!strncmp(str, "ZDIM=", 5)) {
	if(sscanf(str+5, "%i", &im->zdim) != 1) return -1;
      }
      else if(!strncmp(str, "VDIM=", 5)) {
	if(sscanf(str+5, "%i", &im->vdim) != 1) return -1;
	if(im->vdim == 1) im->vectMode = VM_SCALAR;
	else im->vectMode = VM_INTERLACED;
      }
      else if(!strncmp(str, "VX=", 3)) {
	if(sscanf(str+3, "%f", &im->vx) != 1) return -1;
      }
      else if(!strncmp(str, "VY=", 3)) {
	if(sscanf(str+3, "%f", &im->vy) != 1) return -1;
      }
      else if(!strncmp(str, "VZ=", 3)) {
	if(sscanf(str+3, "%f", &im->vz) != 1) return -1;
      }
      else if(!strncmp(str, "TYPE=", 5)) {
	if(!strncmp(str+5, "float", 5)) im->wordKind = WK_FLOAT;
	else {
	  if(!strncmp(str+5, "signed fixed", 12)) {
	    im->wordKind = WK_FIXED;
	    im->sign = SGN_SIGNED;
	  }
	  else if(!strncmp(str+5, "unsigned fixed", 14)) {
	    im->wordKind = WK_FIXED;
	    im->sign = SGN_UNSIGNED;
	  }
	  else return -1;
	}
      }
      else if(!strncmp(str, "PIXSIZE=", 8)) {
	if(sscanf(str+8, "%i %n", &im->wdim, &n) != 1) return -1;
	if(im->wdim != 8 && im->wdim != 16 && im->wdim != 32 &&
	   im->wdim != 64) return -1;
	im->wdim >>= 3;
	if(strncmp(str+8+n, "bits", 4)) return -1;
      }
      else if(!strncmp(str, "SCALE=", 6)) ;
      else if(!strncmp(str, "CPU=", 4)) {
	if(!strncmp(str+4, "decm", 4)) im->endianness = END_LITTLE;
	else if(!strncmp(str+4, "alpha", 5)) im->endianness = END_LITTLE;
	else if(!strncmp(str+4, "pc", 2)) im->endianness = END_LITTLE;
	else if(!strncmp(str+4, "sun", 3)) im->endianness = END_BIG;
	else if(!strncmp(str+4, "sgi", 3)) im->endianness = END_BIG;
	else return -1;
      }

      if(!fgetns(str, 257, im->fd)) return -1;
    }

    /* parse user strings */
    im->nuser = nusr = 0;
    while(str[0] == '#' && strncmp(str, "##}", 3)) {
      addStringElement(&strl, str + 1);
      while(strlen(str) == 256) {
	if(!fgetns(str, 257, im->fd)) return -1;
	concatStringElement(&strl, str);
      }
      nusr++;
      if(!fgetns(str, 257, im->fd)) return -1;      
    }
    
    /* go to end of header */
    while(strncmp(str, "##}", 3)) {
      if(!fgetns(str, 257, im->fd)) return -1;
    }

    /* check header validity */
    if(im->xdim > 0 && im->ydim > 0 && im->zdim > 0 && im->vdim > 0 &&
       im->vx > 0.0 && im->vy > 0.0 && im->vz > 0.0 &&
       (im->wordKind == WK_FLOAT || (im->wordKind == WK_FIXED &&
				     im->sign != SGN_UNKNOWN)) &&
       im->endianness != END_UNKNOWN) {
      if(nusr > 0) {
	im->nuser = nusr;
	im->user = (char **) ImageIO_alloc(im->nuser * sizeof(char *));
	oel = NULL;
	for(el = strl.begin, n = 0; el != NULL; el = oel, n++) {
	  im->user[n] = el->string;
	  oel = el->next;
	  ImageIO_free(el);
	}
      }
      return 1;
    }
    else return -1;

  }
  else return -1;
}


/* given an initialized file descriptor and a file name, open file
   from stdin (if name == NULL), a gziped pipe (if file is gziped)
   or a standard file otherwise. */
void _openReadImage(_image* im, const char *name) {
  if(im->openMode == OM_CLOSE) {

    /* open from stdin */
    if(name == NULL) {
      im->openMode = OM_STD;
      im->fd = gzdopen(fileno(stdin), "rb");
    }

    else {
      im->fd = gzopen(name, "rb");
      if(im->fd) im->openMode = OM_GZ;
    }

  }
}


/* given an initialized file descriptor and a file name, open file
   from stdout (if name == NULL), a gziped pipe (if file is gziped)
   or a standard file otherwise. */
void _openWriteImage(_image* im, const char *name) {
   if(!name) {
      im->openMode = OM_STD;
      im->fd = (gzFile) stdout;
   }
   
   else {
      int i = strlen(name);
      if(!strncmp(name+i-3, ".gz", 3)) {
	 im->openMode = OM_GZ;
	 im->fd = gzopen(name, "wb");
      }
      else {
	 im->openMode = OM_FILE;
	 im->fd = (gzFile) fopen(name, "wb");
      }
   }
}



/* close an image opened using _open{Read|Write}Image */
void _closeImage(_image *im) {
  switch(im->openMode) {
  case OM_CLOSE:
  case OM_STD:
    break;
  case OM_FILE:
    fclose((FILE *) im->fd);
    break;
  case OM_GZ:
    gzclose(im->fd);
    break;
  }

  im->fd = NULL;
  im->openMode = OM_CLOSE;
}





/* add a string element at the tail of given list */
static void addStringElement(stringListHead *strhead, const char *str) {
  stringListElement *el;

  el = (stringListElement *) ImageIO_alloc(sizeof(stringListElement));
  el->string = strdup(str);
  el->next = NULL;
  if(strhead->begin == NULL)
    strhead->begin = strhead->end = el;
  else {
    strhead->end->next = el;
    strhead->end = el;
  }
}

/* concat given string at the last element of given list */
static void concatStringElement(const stringListHead *strhead,
				const char *str) {
  stringListElement *el;

  el = strhead->end;
  el->string = (char *) realloc(el->string,
				strlen(el->string) + strlen(str) + 1);
  strcat(el->string, str);
}

/* get a string from a file and discard the ending newline character
   if any */
static char *fgetns(char *str, int n, gzFile f) {
  char *ret;
  int l;

  ret = gzgets(f, str, n);
  if(!ret) return NULL;
  l = strlen(str);
  if(l > 0 && str[l-1] == '\n') str[l-1] = '\0';
  return ret;
}


/* check the type of image in fileName */
IMAGE_FORMAT imageType(const char *fileName) {
  gzFile f;
  char magic[5];

  if(!fileName) f = gzdopen(fileno(stdin), "rb");
  else f = gzopen(fileName, "r");
  if(!f) return IF_UNKNOWN;

  gzread(f, (void *) magic, 4);
  magic[4] = '\0';
  if(fileName) gzclose(f);
  
  if(!strcmp(magic, INR_MAGIC)) return IF_INR;
  else if(!strcmp(magic, GIF_MAGIC)) return IF_GIF;
  else if(!strcmp(magic, ANALYZE_LE_MAGIC)) return IF_ANALYZE;
  else if(!strcmp(magic, ANALYZE_BE_MAGIC)) return IF_ANALYZE;
  else if((((unsigned char *)magic)[0]<<8) + ((unsigned char *)magic)[1]
	  == IRIS_MAGIC)
    return IF_IRIS;

  return IF_UNKNOWN;
}


