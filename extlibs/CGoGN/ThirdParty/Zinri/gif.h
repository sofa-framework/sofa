#ifndef GIF_H
#define GIF_H

#include "ImageIO.h"


#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>

int readGifImage(_image *im,const char *name);


#ifdef __cplusplus
}
#endif

#endif
