/*
 * xviris.c - load routine for IRIS 'rgb' format pictures
 *
 * LoadIRIS()
 * WriteIRIS()
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ImageIO.h"
#include "iris.h"

typedef unsigned char byte;

#define BPPMASK			0x00ff
#define ITYPE_VERBATIM		0x0000
#define ITYPE_RLE		0x0100
#define ISRLE(type)		(((type) & 0xff00) == ITYPE_RLE)
#define ISVERBATIM(type)	(((type) & 0xff00) == ITYPE_VERBATIM)
#define BPP(type)		((type) & BPPMASK)
#define RLE(bpp)		(ITYPE_RLE | (bpp))
#define VERBATIM(bpp)		(ITYPE_VERBATIM | (bpp))

static int errnum;
#define FERROR(fp) (((void) gzerror(fp, &errnum), errnum != Z_OK) || gzeof(fp))



#define TAGLEN	(5)

#define RINTLUM (79)
#define GINTLUM (156)
#define BINTLUM (21)

#define OFFSET_R	3	/* this is byte order dependent */
#define OFFSET_G	2
#define OFFSET_B	1
#define OFFSET_A	0

#define ILUM(r,g,b)     ((int)(RINTLUM*(r)+GINTLUM*(g)+BINTLUM*(b))>>8)
#define CHANOFFSET(z)	(3-(z))	/* this is byte order dependent */

static byte    *getimagedata  (gzFile, unsigned short, int, int, int);
static void     interleaverow (byte *, byte *, int, int);
static void     expandrow     (byte *, byte *, int);
static void     readtab       (gzFile,unsigned int *, int);
static void     addimgtag     (byte *, int, int);


static unsigned short  getshort      (gzFile);
static unsigned int   getlong       (gzFile);


/******************************************/
static void interleaverow(byte *lptr, byte *cptr, int z, int n)
{
  lptr += z;
  while(n--) {
    *lptr = *cptr++;
    lptr += 4;
  }
}


/******************************************/
static void expandrow(byte *optr, byte *iptr, int z)
{
  byte pixel, count;

  optr += z;
  while (1) {
    pixel = *iptr++;
    if ( !(count = (pixel & 0x7f)) ) return;
    if (pixel & 0x80) {
      while (count>=8) {
	optr[0*4] = iptr[0];
	optr[1*4] = iptr[1];
	optr[2*4] = iptr[2];
	optr[3*4] = iptr[3];
	optr[4*4] = iptr[4];
	optr[5*4] = iptr[5];
	optr[6*4] = iptr[6];
	optr[7*4] = iptr[7];
	optr += 8*4;
	iptr += 8;
	count -= 8;
      }
      while(count--) {
	*optr = *iptr++;
	optr+=4;
      }
    }
    else {
      pixel = *iptr++;
      while(count>=8) {
	optr[0*4] = pixel;
	optr[1*4] = pixel;
	optr[2*4] = pixel;
	optr[3*4] = pixel;
	optr[4*4] = pixel;
	optr[5*4] = pixel;
	optr[6*4] = pixel;
	optr[7*4] = pixel;
	optr += 8*4;
	count -= 8;
      }
      while(count--) {
	*optr = pixel;
	optr+=4;
      }
    }
  }
}

/****************************************************/
static void readtab(gzFile fp, unsigned int *tab, int n)
{
  while (n) {
    *tab++ = getlong(fp);
    n--;
  }
}


/*****************************************************/
static void addimgtag(byte *dptr, int xsize, int ysize)
{
  /* this is used to extract image data from core dumps. 
     I doubt this is necessary...  --jhb */

  dptr    = dptr + (xsize * ysize * 4);
  dptr[0] = 0x12;  dptr[1] = 0x34;  dptr[2] = 0x56;  dptr[3] = 0x78;
  dptr += 4;

  dptr[0] = 0x59;  dptr[1] = 0x49;  dptr[2] = 0x33;  dptr[3] = 0x33;
  dptr += 4;

  dptr[0] = 0x69;  dptr[1] = 0x43;  dptr[2] = 0x42;  dptr[3] = 0x22;
  dptr += 4;

  dptr[0] = (xsize>>24)&0xff;
  dptr[1] = (xsize>>16)&0xff;
  dptr[2] = (xsize>> 8)&0xff;
  dptr[3] = (xsize    )&0xff;
  dptr += 4;

  dptr[0] = (ysize>>24)&0xff;
  dptr[1] = (ysize>>16)&0xff;
  dptr[2] = (ysize>> 8)&0xff;
  dptr[3] = (ysize    )&0xff;
}

/* byte order independent read/write of shorts and longs. */
/*****************************************************/
static unsigned short getshort(gzFile inf)
{
  byte buf[2];
  gzread(inf, buf, (size_t) 2);
  return (buf[0]<<8)+(buf[1]<<0);
}

/*****************************************************/
static unsigned int getlong(gzFile inf)
{
  byte buf[4];
  gzread(inf, buf, (size_t) 4);
  return (((unsigned int) buf[0])<<24) + (((unsigned int) buf[1])<<16)
       + (((unsigned int) buf[2])<<8) + buf[3];
}

/*****************************************************/
int readIrisImage(_image *im) {
  byte   *rawdata, *rptr;
  byte   *pic824,  *bptr, *iptr;
  int     i, j, size;
  unsigned short imagic, type;
  int xsize, ysize, zsize;


  /* read header information from file */
  imagic = getshort(im->fd);
  type   = getshort(im->fd);
  getshort(im->fd);
  xsize  = getshort(im->fd);
  ysize  = getshort(im->fd);
  zsize  = getshort(im->fd);

  if (FERROR(im->fd)) return 0;

  if (imagic != IRIS_MAGIC) return 0;

  rawdata = getimagedata(im->fd, type, xsize, ysize, zsize);
  if (!rawdata) return 0;

  if (FERROR(im->fd)) return 0;   /* probably truncated file */


  /*  t1=texture_alloc();
  (void) strcpy(t1->name,fname);
  t1->type=IRIS;
  
  t1->nrows = ysize;
  t1->ncols = xsize;
    
  t1->image = create_int_array(t1->nrows,t1->ncols);*/

  /* got the raw image data.  Convert to an XV image (1,3 bytes / pix) */
  if (zsize < 3) 
    {  /* grayscale */
      im->xdim = xsize;
      im->ydim = ysize;
      im->zdim = 1;
      im->vdim = 1;
      im->wdim = 1;
      im->wordKind = WK_FIXED;
      im->sign = SGN_UNSIGNED;
      im->data = ImageIO_alloc(xsize * ysize);

      pic824 = (byte *) ImageIO_alloc((size_t) xsize * ysize);
      if (!pic824) exit(-1);
      
      /* copy plane 3 from rawdata into pic824, inverting pic vertically */
      for (i = 0, bptr = pic824; i < (int) ysize; i++) 
	{
	  rptr = rawdata + 3 + ((ysize - 1) - i) * (xsize * 4);
	  for (j = 0; j < (int) xsize; j++, bptr++, rptr += 4)
	    *bptr = *rptr;
	}
      
      
      size = im->xdim * im->ydim;
      for (bptr = pic824, iptr = (unsigned char *) im->data,
	     i = 0; i < size; ++i, ++iptr, ++bptr)
	{
	  *iptr = *bptr;
	}
      
      ImageIO_free(pic824);
    
  }

  else {  /* truecolor */
    im->xdim = xsize;
    im->ydim = ysize;
    im->zdim = zsize / 3;
    im->vdim = 4;
    im->wdim = 1;
    im->wordKind = WK_FIXED;
    im->sign = SGN_UNSIGNED;
    im->data = ImageIO_alloc(xsize * ysize * im->zdim * 4);

    pic824 = (byte *) ImageIO_alloc((size_t) xsize * ysize * 3);
    if (!pic824) 
      exit(1);
    
    /* copy plane 3 from rawdata into pic824, inverting pic vertically */
    for (i = 0, bptr = pic824; i<(int) ysize; i++) 
      {
	rptr = rawdata + ((ysize - 1) - i) * (xsize * 4);

	for (j=0; j<(int) xsize; j++, rptr += 4) {
	  *bptr++ = rptr[3];
	  *bptr++ = rptr[2];
	  *bptr++ = rptr[1];
	}
      }
    
    
    size = im->xdim * im->ydim;
    for (bptr = pic824, iptr = (unsigned char *) im->data, i = 0;
	 i < size; ++i, iptr += 4, bptr += 3)
      {
#if ( defined ( _ALPHA_ ) || defined ( _LINUX_ ))
	iptr[0] = 0xFF;
	iptr[1] = bptr[2];
	iptr[2] = bptr[1];
	iptr[3] = bptr[0];
#else 
	iptr[0] = bptr[0];
	iptr[1] = bptr[1];
	iptr[2] = bptr[2];
	iptr[3] = 0xFF;
#endif
      }
      
      ImageIO_free(pic824);
  }

  ImageIO_free(rawdata);

  return 1;
}     




/****************************************************/
static byte *getimagedata(gzFile fp, unsigned short type, int xsize, int ysize,
			  int zsize)
{
  /* read in a B/W RGB or RGBA iris image file and return a 
     pointer to an array of 4-byte pixels, arranged ABGR, NULL on error */

  byte   *base, *lptr;
  byte   *verdat;
  int     y, z, tablen;
  int     bpp, rle, cur, badorder;
  int     rlebuflen;
  byte *rledat;
  unsigned int *starttab, *lengthtab;


  rle     = ISRLE(type);
  bpp     = BPP(type);

  if (bpp != 1) {
    return (byte *) NULL;
  }

  if (rle) {

    rlebuflen = 2 * xsize + 10;
    tablen    = ysize * zsize;
    starttab  = (unsigned int *) ImageIO_alloc((size_t) tablen * sizeof(int));
    lengthtab = (unsigned int *) ImageIO_alloc((size_t) tablen * sizeof(int));
    rledat    = (byte *)   ImageIO_alloc((size_t) rlebuflen);

    if (!starttab || !lengthtab || !rledat) 
      exit(1);

    gzseek(fp, 512L, SEEK_SET);
    readtab(fp, starttab,  tablen);
    readtab(fp, lengthtab, tablen);

    if (FERROR(fp)) {
      ImageIO_free(starttab);  ImageIO_free(lengthtab);  ImageIO_free(rledat);
      return (byte *) NULL;
    }


    /* check data order */
    cur = 0;
    badorder = 0;
    for (y=0; y<ysize && !badorder; y++) {
      for (z=0; z<zsize && !badorder; z++) {
	if (starttab[y+z*ysize] < cur) badorder = 1;
	else cur = starttab[y+z*ysize];
      }
    }

    gzseek(fp, (int) (512 + 2*tablen*4), 0);
    cur = 512 + 2*tablen*4;

    base = (byte *) ImageIO_alloc((size_t) (xsize*ysize+TAGLEN) * 4);
    if (!base) 
      exit(1);

    addimgtag(base,xsize,ysize);

    if (badorder) {
      for (z=0; z<zsize; z++) {
	lptr = base;
	for (y=0; y<ysize; y++) {
	  if (cur != starttab[y+z*ysize]) {
	    gzseek(fp, (int) starttab[y+z*ysize], 0);
	    cur = starttab[y+z*ysize];
	  }

	  if (lengthtab[y+z*ysize]>rlebuflen) {
	    ImageIO_free(starttab); ImageIO_free(lengthtab); ImageIO_free(rledat); ImageIO_free(base);
	    return (byte *) NULL;
	  }

	  gzread(fp, rledat, (size_t) lengthtab[y+z*ysize]);
	  cur += lengthtab[y+z*ysize];
	  expandrow(lptr,rledat,3-z);
	  lptr += (xsize * 4);
	}
      }
    }
    else {
      lptr = base;
      for (y=0; y<ysize; y++) {
	for (z=0; z<zsize; z++) {
	  if (cur != starttab[y+z*ysize]) {
	    gzseek(fp, (int) starttab[y+z*ysize], 0);
	    cur = starttab[y+z*ysize];
	  }

	  gzread(fp, rledat, (size_t) lengthtab[y+z*ysize]);
	  cur += lengthtab[y+z*ysize];
	  expandrow(lptr,rledat,3-z);
	}
	lptr += (xsize * 4);
      }
    }

    ImageIO_free(starttab);
    ImageIO_free(lengthtab);
    ImageIO_free(rledat);
    return base;
  }      /* end of RLE case */

  else {  /* not RLE */
    verdat = (byte *) ImageIO_alloc((size_t) xsize);
    base   = (byte *) ImageIO_alloc((size_t) (xsize*ysize+TAGLEN) * 4);
    if (!base || !verdat) 
      exit(1);

    addimgtag(base,xsize,ysize);
    
    gzseek(fp,512L,0);

    for (z=0; z<zsize; z++) {
      lptr = base;
      for (y=0; y<ysize; y++) {
	gzread(fp, verdat, (size_t) xsize);
	interleaverow(lptr,verdat,3-z,xsize);
	lptr += (xsize * 4);
      }
    }

    ImageIO_free(verdat);
    return base;
  }
}
















