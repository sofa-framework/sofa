#define EXPORTING 1


#include <Zinrimage.h>

#if (defined _SGI_) || (defined _SOLARIS_)
  extern FILE     *popen(const char *, const char *);
  extern int      pclose(FILE *);
#endif



/* Read an inrimage in fileName */
inrimage *readZInrimage(const char *fileName) {
  _image *img;
  inrimage *inr = NULL;

  img = _readImage(fileName);
  if(img) {
    inr = _image2inrimage(img);
    img->data = NULL;
    _freeImage(img);
    return inr;
  }
  else return NULL;
}


/* read header of inrimage strored in fileName */
INRIMAGE_HEADER *readZInrimageHeader(const char *fileName) {
  INRIMAGE_HEADER *inrh;
  _image *img;

  img = _readImageHeader(fileName);
  if(img) {
    inrh = (INRIMAGE_HEADER *) malloc(sizeof(INRIMAGE_HEADER));
    inrh->inr = _image2inrimage(img);
    if(inrh->inr->data) {
      free(inrh->inr->data);
      free(inrh->inr->array);
      inrh->inr->data = NULL;
      inrh->inr->array = NULL;
    }
    inrh->endianness = img->endianness;
    return inrh;
  }
  else return NULL;
}


/* Write inrimage given in inr in fileName. If fileName's suffix is
   .gz, the image is gziped. If filename is NULL, image is written
   on stdout */
int writeZInrimage(const inrimage *inr, const char *fileName) {
  _image *img;
  int i;

  img = inrimage2_image(inr);
  i = _writeImage(img, fileName);
  img->data = NULL;
  _freeImage(img);

  if(i >= 0) return 0;
  else return i;
}



int writeInrimageHeader(FILE *f, const inrimage *inr) {
  _image *img;
  int i;
  
  img = inrimage2_image(inr);
  img->fd = f;
  img->openMode = OM_FILE;
  i = _writeInrimageHeader(img, END_UNKNOWN);
  img->data = NULL;
  _freeImage(img);

  if(i >= 0) return 0;
  else return i;
}



int writeInrimageData(FILE *f, const inrimage *inr) {
  _image *img;
  int i;

  img = inrimage2_image(inr);
  img->fd = f;
  img->openMode = OM_FILE;
  i = _writeInrimageData(img);
  img->data = NULL;
  _freeImage(img);

  if(i >= 0) return 0;
  else return i;
}



/* read an inrimage and return an epidaurelib t_Image structure */
t_ImagePtr readEpidaureLibInrimage(const char *name) {
  t_ImagePtr img;
  inrimage *inr;

  inr = readZInrimage(name);
  if(inr) {
    img = (t_ImagePtr) malloc(sizeof(t_Image));
    strncpy(img->fi.name, name, B_FNAME_LENGTH);
    img->tx = inr->ncols;
    img->ty = inr->nrows;
    img->tz = inr->nplanes;
    switch(inr->type) {
    case WT_UNSIGNED_CHAR:
      img->type = TYPEIM_256;
      break;
    case WT_UNSIGNED_SHORT:
      img->type = TYPEIM_16B;
      break;
    case WT_SIGNED_SHORT:
      img->type = TYPEIM_S16B;
      break;
    case WT_FLOAT:
      img->type = TYPEIM_FLOAT;
      break;
    case WT_UNSIGNED_INT:
    case WT_SIGNED_INT:
    case WT_UNSIGNED_LONG:
    case WT_SIGNED_LONG:
    case WT_DOUBLE:
    case WT_RGB:
    case WT_RGBA:
    case WT_FLOAT_VECTOR:
    case WT_DOUBLE_VECTOR:
      free(img);
      freeInrimage(inr);
      return NULL;
    }
    img->buf = (char *) inr->data;

    free(inr);
    return img;
  }
  else return NULL;
}

/* read an inrimage header and return an epidaurelib t_Image structure */
t_ImagePtr readEpidaureLibInrimageHeader(const char *name) {
  t_ImagePtr img;
  INRIMAGE_HEADER *inrh;

  inrh = readZInrimageHeader(name);
  if(inrh) {
    img = (t_ImagePtr) malloc(sizeof(t_Image));
    strncpy(img->fi.name, name, B_FNAME_LENGTH);
    img->tx = inrh->inr->ncols;
    img->ty = inrh->inr->nrows;
    img->tz = inrh->inr->nplanes;
    switch(inrh->inr->type) {
    case WT_UNSIGNED_CHAR:
      img->type = TYPEIM_256;
      break;
    case WT_UNSIGNED_SHORT:
      img->type = TYPEIM_16B;
      break;
    case WT_SIGNED_SHORT:
      img->type = TYPEIM_S16B;
      break;
    case WT_FLOAT:
      img->type = TYPEIM_FLOAT;
      break;
    case WT_UNSIGNED_INT:
    case WT_SIGNED_INT:
    case WT_UNSIGNED_LONG:
    case WT_SIGNED_LONG:
    case WT_DOUBLE:
    case WT_RGB:
    case WT_RGBA:
    case WT_FLOAT_VECTOR:
    case WT_DOUBLE_VECTOR:
      free(img);
      free(inrh->inr);
      free(inrh);
      return NULL;
    }
    img->buf = NULL;

    free(inrh->inr);
    free(inrh);
    return img;
  }
  else return NULL;
}


/* write inrimage described by epidaurelib structure img */
int writeEpidaureLibInrimage(const t_ImagePtr img) {
  inrimage inr;

  inr.ncols = img->tx;
  inr.nrows = img->ty;
  inr.nplanes = img->tz;
  inr.vdim = 1;
  switch(img->type) {
  case TYPEIM_256:
    inr.type = WT_UNSIGNED_CHAR;
    break;
  case TYPEIM_16B:
    inr.type = WT_UNSIGNED_SHORT;
    break;
  case TYPEIM_S16B:
    inr.type = WT_SIGNED_SHORT;
    break;
  case TYPEIM_FLOAT:
    inr.type = WT_FLOAT;
    break;
  default:
    return -1;
  }
  inr.data = (void *) img->buf;
  inr.vx = inr.vy = inr.vz = 1.0;
  return writeZInrimage(&inr, img->fi.name);
}


/* free epidaureLib inrimage structure */
void freeEpidaureLibInrimage(const t_ImagePtr img) {
  if(img->buf) free(img->buf);
  free(img);
}


#if 0
/* create a 3 entries array of elements of size bytes */
static char ***create3DArray(int ncols, int nrows, int nplanes, int bytes) {
  char ***p3, **p2, *p1;
  int iz, iy, ix = ncols * bytes, i, j;
  double r;

  i = nplanes * (1 + nrows) * bytes; 
  r = (double) i / (double) bytes;
  j = (int) r * bytes;
  if (i > j) j += bytes;

  p3 = (char ***) malloc(nplanes * nrows * ncols * bytes + j);

  if(p3) {
    p2 = (char **) (p3 + nplanes);
    p1 = (char *) ((char *) p3 + j);
    for(iz = 0; iz < nplanes; iz++) {
      p3[iz] = p2;
      p2 += nrows;
      for(iy = 0; iy < nrows; iy++) {
	p3[iz][iy] = p1;
	p1 += ix;
      }
    }
  }

  return p3;
}
#endif


void ***index3DArray(const void *data, unsigned int ncols, unsigned int nrows,
		     unsigned int nplanes, unsigned int stype) {
  void ***array;
  unsigned char **t1, *t2;
  unsigned int i, j;

  array = (void ***) malloc(nplanes * sizeof(void **) +
			    nrows * nplanes * sizeof(void *));

  for(i = 0, t1 = (unsigned char **) (array) + nplanes; i < nplanes;
      i++, t1 += nrows)
    ((unsigned char ***) array)[i] = t1;

  for(i = 0, t2 = (unsigned char *) data; i < nplanes; i++)
    for(j = 0; j < nrows; j++, t2 += ncols * stype)
      array[i][j] = t2;

  return array;
}


void ****index4DArray(const void *data, unsigned int ncols, unsigned int nrows,
		      unsigned int nplanes, unsigned int vdim,
		      unsigned int stype) {
  void ****array;
  unsigned char ***t1, **t2, *t3;
  unsigned int i, j, k;

  array = (void ****) malloc(vdim * sizeof(void ***) +
			     vdim * nplanes * sizeof(void **) +
			     vdim * nplanes * nrows * sizeof(void *));

  for(i = 0, t1 = (unsigned char ***) (array) + vdim; i < vdim;
      i++, t1 += nplanes)
    ((unsigned char ****) array)[i] = t1;

  for(i = 0, t2 = (unsigned char **) array + vdim * (1 + nplanes);
      i < vdim; i++) {
    for(j = 0; j < nplanes; j++, t2 += nrows) {
      ((unsigned char ****) array)[i][j] = t2;
    }
  }

  for(i = 0, t3 = (unsigned char *) data; i < vdim; i++) {
    for(j = 0; j < nplanes; j++) {
      for(k = 0; k < nrows; k++, t3 += ncols * stype) {
	((unsigned char ****) array)[i][j][k] = t3;
      }
    }
  }

  return array;
}


/* allocate an inrimage structure */
inrimage *initInrimage(int ncols, int nrows, int nplanes, int vdim,
		       WORDTYPE type) {
  inrimage *inr;

  inr = (inrimage *) malloc(sizeof(inrimage));
  inr->ncols = ncols;
  inr->nrows = nrows;
  inr->nplanes = nplanes;
  inr->vx = inr->vy = inr->vz = 1.0;
  switch(type) {
  case WT_RGB:
    inr->vdim = 3;
    break;
  case WT_RGBA:
    inr->vdim = 4;
    break;
  case WT_FLOAT_VECTOR:
  case WT_DOUBLE_VECTOR:
    inr->vdim = vdim;
    break;
  default:
    inr->vdim = 1;
  }
  inr->type = type;
  inr->data = NULL;
  inr->array = NULL;
  inr->varray = NULL;

  return inr;
}



/* allocate an inrimage structure and a buffer */
inrimage *createInrimage(int ncols, int nrows, int nplanes, int vdim,
			 WORDTYPE type) {
  inrimage *inr;
  unsigned int size;

  inr = initInrimage(ncols, nrows, nplanes, vdim, type);
  
  size = ncols * nrows * nplanes * sizeofImageVoxel(inr);
  
  inr->data = malloc(size);
  inr->array = index3DArray(inr->data, inr->ncols, inr->nrows, inr->nplanes,
			    sizeofImageVoxel(inr));
  inr->varray = NULL;
  memset((char *)inr->data,0,size);
/*  bzero((char *)inr->data, size); */
  return inr;
}




/* allocate an inrimage structure and a buffer */
inrimage *createNonInterlacedInrimage(int ncols, int nrows, int nplanes,
				      int vdim, WORDTYPE type) {
  inrimage *inr;
  unsigned int size;

  if(vdim == 1) return createInrimage(ncols, nrows, nplanes, vdim, type);
  else {
    inr = initInrimage(ncols, nrows, nplanes, vdim, type);
  
    size = ncols * nrows * nplanes * sizeofImageVoxel(inr);
  
    inr->data = malloc(size);
    inr->varray = index4DArray(inr->data, inr->ncols, inr->nrows, inr->nplanes,
			       inr->vdim, sizeofWordtype(inr->type));
/*    bzero((char *)inr->data, size); */
    memset((char *)inr->data,0,size);
  }
  return inr;
}



void setInrimageData(inrimage *inr, void *data) {
  if(inr->data) free(inr->data);
  if(inr->array) free(inr->array);
  if(inr->varray) free(inr->varray);
  inr->data = data;
  inr->array = index3DArray(inr->data, inr->ncols, inr->nrows, inr->nplanes,
			    sizeofImageVoxel(inr));
  inr->varray = NULL;
}



void setNonInterlacedInrimageData(inrimage *inr, void *data) {
  if(inr->vdim == 1) setInrimageData(inr, data);
  else {
    if(inr->data) free(inr->data);
    if(inr->array) free(inr->array);
    if(inr->varray) free(inr->varray);
    inr->data = data;
    inr->varray = index4DArray(inr->data, inr->ncols, inr->nrows,
			       inr->nplanes, inr->vdim,
			       sizeofWordtype(inr->type));
    inr->array = NULL;
  }
}



/* free inrimage */
void freeInrimage(inrimage *inr) {
  if(inr->data) free(inr->data);
  if(inr->array) free(inr->array);
  if(inr->varray) free(inr->varray);
  free(inr);
}


unsigned int sizeofWordtype(const WORDTYPE type) {

  switch(type) {
  case WT_UNSIGNED_CHAR:
  case WT_RGB:
  case WT_RGBA:
    return sizeof(unsigned char);    
  case WT_UNSIGNED_SHORT:
    return sizeof(unsigned short int);
  case WT_SIGNED_SHORT:
    return sizeof(short int);
  case WT_UNSIGNED_INT:
    return sizeof(unsigned int);
  case WT_SIGNED_INT:
    return sizeof(int);
  case WT_UNSIGNED_LONG:
    return sizeof(unsigned LONG);
  case WT_SIGNED_LONG:
    return sizeof(LONG);
  case WT_FLOAT:
  case WT_FLOAT_VECTOR:
    return sizeof(float);
  case WT_DOUBLE:
  case WT_DOUBLE_VECTOR:
    return sizeof(double);
  default:
    return 0;
  }

}



unsigned int sizeofImageVoxel(const inrimage *inr) {
  return inr->vdim * sizeofWordtype(inr->type);
}



/* convert from an _image structure to an inrimage structure */
inrimage *_image2inrimage(const _image *img) {
  inrimage *inr;
  WORDTYPE wt;

  /* vectorial type */
  if(img->vdim > 1) {
    if(img->wdim == 1) {
      if(img->vdim == 3) wt = WT_RGB;
      else if(img->vdim == 4) wt = WT_RGBA;
      else return NULL;
    }
    else if(img->wordKind == WK_FLOAT) {
      if(img->wdim == 4) wt = WT_FLOAT_VECTOR;
      else if(img->wdim == 8) wt = WT_DOUBLE_VECTOR;
      else return NULL;
    }
    else return NULL;
  }

  /* scalar type */
  else {
    if(img->wordKind == WK_FIXED) {
      if(img->wdim == 1) wt = WT_UNSIGNED_CHAR;
      else if(img->wdim == 2) {
	if(img->sign == SGN_SIGNED) wt = WT_SIGNED_SHORT;
	else if(img->sign == SGN_UNSIGNED) wt = WT_UNSIGNED_SHORT;
	else return NULL;
      }
      else if(img->wdim == 4) {
	if(img->sign == SGN_SIGNED) wt = WT_SIGNED_INT;
	else if(img->sign == SGN_UNSIGNED) wt = WT_UNSIGNED_INT;
	else return NULL;
      }
      else if(img->wdim == 8) {
	if(img->sign == SGN_SIGNED) wt = WT_SIGNED_LONG;
	else if(img->sign == SGN_UNSIGNED) wt = WT_UNSIGNED_LONG;
	else return NULL;
      }
      else return NULL;
    }
    else if(img->wordKind == WK_FLOAT) {
      if(img->wdim == 4) wt = WT_FLOAT;
      else if(img->wdim == 8) wt = WT_DOUBLE;
      else return NULL;
    }
    else return NULL;
  }

  inr = initInrimage(img->xdim, img->ydim, img->zdim, img->vdim, wt);
  inr->data = img->data;
  inr->vx = img->vx;
  inr->vy = img->vy;
  inr->vz = img->vz;
  if(img->vectMode == VM_NON_INTERLACED) {
    inr->array = NULL;
    inr->varray = index4DArray(inr->data, inr->ncols, inr->nrows, inr->nplanes,
			       inr->vdim, sizeofWordtype(inr->type));
  }
  else {
    inr->array = index3DArray(inr->data, inr->ncols, inr->nrows, inr->nplanes,
			      sizeofImageVoxel(inr));
    inr->varray = NULL;
  }

  return inr;
}


/* convert from an inrimage structure to an _image structure */
_image *inrimage2_image(const inrimage *inr) {
  _image *img;

  img = _initImage();
  img->xdim = inr->ncols;
  img->ydim = inr->nrows;
  img->zdim = inr->nplanes;
  img->vdim = inr->vdim;
  img->vx = (float) inr->vx;
  img->vy = (float) inr->vy;
  img->vz = (float) inr->vz;
  img->data = inr->data;
  switch(inr->type) {
  case WT_UNSIGNED_CHAR:
  case WT_RGB:
  case WT_RGBA:
    img->wdim = 1;
    img->wordKind = WK_FIXED;
    img->sign = SGN_UNSIGNED;
    break;
  case WT_UNSIGNED_SHORT:
    img->wdim = 2;
    img->wordKind = WK_FIXED;
    img->sign = SGN_UNSIGNED;
    break;
  case WT_SIGNED_SHORT:
    img->wdim = 2;
    img->wordKind = WK_FIXED;
    img->sign = SGN_SIGNED;
    break;
  case WT_UNSIGNED_INT:
    img->wdim = 4;
    img->wordKind = WK_FIXED;
    img->sign = SGN_UNSIGNED;
    break;
  case WT_SIGNED_INT:
    img->wdim = 4;
    img->wordKind = WK_FIXED;
    img->sign = SGN_SIGNED;
    break;
  case WT_UNSIGNED_LONG:
    img->wdim = 8;
    img->wordKind = WK_FIXED;
    img->sign = SGN_UNSIGNED;
    break;
  case WT_SIGNED_LONG:
    img->wdim = 8;
    img->wordKind = WK_FIXED;
    img->sign = SGN_SIGNED;
    break;
  case WT_FLOAT:
    img->wdim = 4;
    img->wordKind = WK_FLOAT;
    break;
  case WT_FLOAT_VECTOR:
    img->wdim = 4;
    img->wordKind = WK_FLOAT;
    if(inr->varray) img->vectMode = VM_NON_INTERLACED;
    else img->vectMode = VM_INTERLACED;
    break;
  case WT_DOUBLE:
    img->wdim = 8;
    img->wordKind = WK_FLOAT;
    break;
  case WT_DOUBLE_VECTOR:
    img->wdim = 8;
    img->wordKind = WK_FLOAT;
    if(inr->varray) img->vectMode = VM_NON_INTERLACED;
    else img->vectMode = VM_INTERLACED;
    break;
  }

  return img;
}



/* convert from a t_Image structure to an inrimage structure */
inrimage *epidaureLib2Inrimage(const t_ImagePtr img) {
  inrimage *inr;
  WORDTYPE wt;

  switch(img->type) {
  case TYPEIM_256:
    wt = WT_UNSIGNED_CHAR;
    break;
  case TYPEIM_16B:
    wt = WT_UNSIGNED_SHORT;
    break;
  case TYPEIM_S16B:
    wt = WT_SIGNED_SHORT;
    break;
  case TYPEIM_FLOAT:
    wt = WT_FLOAT;
    break;
  default:
    return NULL;
  }
  
  inr = initInrimage(img->tx, img->ty, img->tz, 1, wt);
  inr->data = (void *) img->buf;
  inr->array = index3DArray(inr->data, inr->ncols, inr->nrows, inr->nplanes,
			    sizeofImageVoxel(inr));
  inr->varray = NULL;

  return inr;
}


/* convert from an inrimage structure to a t_Image structure */
t_ImagePtr inrimage2EpidaureLib(const inrimage *inr) {
  t_ImagePtr img;

  img = (t_ImagePtr) malloc(sizeof(t_Image));
  img->tx = inr->ncols;
  img->ty = inr->nrows;
  img->tz = inr->nplanes;
  switch(inr->type) {
  case WT_UNSIGNED_CHAR:
    img->type = TYPEIM_256;
    break;
  case WT_UNSIGNED_SHORT:
    img->type = TYPEIM_16B;
    break;
  case WT_SIGNED_SHORT:
    img->type = TYPEIM_S16B;
    break;
  case WT_FLOAT:
    img->type = TYPEIM_FLOAT;
    break;
  default:
    free(img);
    return NULL;
  }
  img->buf = (char *) inr->data;
  strcpy(img->fi.name, "NoName.inr");

  return img;
}


/* convert from an _image structure to a t_Image structure */
t_ImagePtr _image2epidaureLib(const _image *img) {
  t_ImagePtr epi;
  int type;

  if(img->wdim == 1) {
    if(img->wordKind == WK_FIXED)
      type = TYPEIM_256;
    else return NULL;
  }
  else if(img->wdim == 2) {
    if(img->wordKind == WK_FIXED) {
      if(img->sign == SGN_SIGNED)
	type = TYPEIM_S16B;
      else if(img->sign == SGN_UNSIGNED)
	type = TYPEIM_16B;
      else return NULL;
    }
    else return NULL;
  }
  else if(img->wdim == 4) {
    if(img->wordKind == WK_FLOAT)
      type = TYPEIM_FLOAT;
    else
      return NULL;
  }
  else return NULL;

  epi = (t_ImagePtr) malloc(sizeof(t_Image));
  epi->tx = img->xdim;
  epi->ty = img->ydim;
  epi->tz = img->zdim;
  epi->type = type;
  epi->buf = (char *) img->data;
  strcpy(epi->fi.name, "NoName.inr");

  return epi;  
}


/* convert from an t_Image structure to an _image structure */
_image *epidaureLib2_image(const t_ImagePtr epi) {
  _image *img;

  img = _initImage();
  img->xdim = epi->tx;
  img->ydim = epi->ty;
  img->zdim = epi->tz;
  img->vdim = 1;
  img->data = (void *) epi->buf;
  switch(epi->type) {
  case TYPEIM_256:
    img->wdim = 1;
    img->wordKind = WK_FIXED;
    img->sign = SGN_UNSIGNED;
    break;
  case TYPEIM_16B:
    img->wdim = 2;
    img->wordKind = WK_FIXED;
    img->sign = SGN_UNSIGNED;
    break;
  case TYPEIM_S16B:
    img->wdim = 2;
    img->wordKind = WK_FIXED;
    img->sign = SGN_SIGNED;
    break;
  case TYPEIM_FLOAT:
    img->wdim = 4;
    img->wordKind = WK_FLOAT;
    break;
  default:
    break;
  }

  return img;  
}




inrimage *readNonInterlacedZInrimage(const char * name) {
  _image *img;
  inrimage *inr = NULL;

  img = _readNonInterlacedImage(name);
  if(img) {
    inr = _image2inrimage(img);
    img->data = NULL;
    _freeImage(img);
    return inr;
  }
  else return NULL;
}

