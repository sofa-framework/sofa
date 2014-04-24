#ifndef ANALYZE_H
#define ANALYZE_H

#include "ImageIO.h"


#ifdef __cplusplus
extern "C" {
#endif


int readAnalyzeHeader(_image *im,const char* name);
int writeAnalyzeHeader( const _image* im ) ;
int writeAnalyzeData( const _image* im ) ;

#ifdef __cplusplus
}
#endif

#endif
