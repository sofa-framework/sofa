/******* COPYRIGHT ************************************************
*                                                                 *
*                         FlowVR Render                           *
*                   Parallel Rendering Library                    *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 2005 by                                           *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU LGPL, please refer to the     *
* COPYING-LIB file for further information.                       *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 * 
*******************************************************************
*                                                                 *
* File: ./include/flowvr/render/noise.h                           *
*                                                                 *
* Contacts:                                                       *
*                                                                 *
******************************************************************/
/* 
  BASED ON:

  Functions to create OpenGL textures containing pre-filtered noise patterns for procedural texturing.

  Pre-filtering the texture with a bicubic filter avoids some of the artifacts associated with linear filtering,
  and allows us to pre-compute the abs() function required for Perlin-style turbulence.

  sgreen@nvidia.com 7/2000
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

inline int rfloor(double r)
{
    static const double FLOATTOINTCONST=(1.5*(1LL<<(52-16)));
    union { double d;
        int i;
    } temp;
    temp.d = r+FLOATTOINTCONST;
    return ((temp.i)>>16);
}

inline int rnear(double r)
{
    static const double FLOATTOINTCONST_0_5=(1.5*(1LL<<(52-16)))+0.5;
    union { double d;
        int i;
    } temp;
    temp.d = r+FLOATTOINTCONST_0_5;
    return ((temp.i)>>16);
}

inline int rceil(double r)
{
  return -rfloor(-r);
}

inline float rabs(float r)
{
  return (r>=0)?r:-r;
}

// clamp x to be between a and b
inline float rclamp(float x, float a, float b)
{
    return (x < a ? a : (x > b ? b : x));
}

inline float rrand()
{
  return (rand() / (float) RAND_MAX);
}

// a nice piecewise-cubic spline function, defined between 0.0<=x<=2.0
// approximates a windowed sinc function - negative lobes (sharpens slightly)
// using a = -0.75 = contraint parameter -0.5<=a<=-1.0 recommended
inline float cubic(float x)
{
  const float a = -0.75;
  double w;

  if ((x >= 0.0) && (x < 1.0)) {
	// Over [0,1) interval
    // (a+2)x^3 - (a+3)x^2 + 1
    w = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  } else if ((x >= 1.0) && (x <= 2.0)) {
	// Over [1,2] interval
    // ax^3 - 5ax^2 + 8ax - 4a
    w = ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
  } else {
    w = 0.0;
  }
  return (float)w;
}

// 1-dimensional cubic interpolation
// not very fast, but easy to understand
// 0<=x<=1
inline float cubicFilter4f(float x, float vm1, float v, float vp1, float vp2)
{
  return vm1 * cubic(1.0f+x) +
		 v   * cubic(x) +
		 vp1 * cubic(1.0f-x) +
		 vp2 * cubic(2.0f-x);
}

inline float cubicFilter4fv(float x, float *v)
{
  return v[0] * cubic(1.0f+x) +
		 v[1] * cubic(x) +
		 v[2] * cubic(1.0f-x) +
		 v[3] * cubic(2.0f-x);
}

/*
  1D cubic interpolator (this is just a faster version of the above)
  assumes x > 0.0
  constraint parameter = -1
*/
inline float cubicFilterFast4fv(float x, float *i)
{
  float x1, x2, x3;
  float h1, h2, h3, h4;
  float result;

  x1 = x + 1.0f;
  x2 = x1 * x1;
  x3 = x2 * x1;
  h4 = -x3 + 5 * x2 - 8 * x1 + 4;
  result = h4 * i[0];

  x1 = x;
  x2 = x1 * x1;
  x3 = x2 * x1;
  h1 = x3 - 2 * x2 + 1;
  result += h1 * i[1];

  x1 = 1.0f - x;
  x2 = x1 * x1;
  x3 = x2 * x1;
  h2 = x3 - 2 * x2 + 1;
  result += h2 * i[2];

  x1 = 2.0f - x;
  x2 = x1 * x1;
  x3 = x2 * x1;
  h3 = -x3 + 5 * x2 - 8 * x1 + 4;
  result += h3 * i[3];

  return result;
}

class Noise2D
{
public:
  float* rdata;
  int rsize;

  Noise2D(int size=64)
  {
    rsize = 1;
    while (rsize < size) rsize<<=1;
    rdata = new float[rsize*rsize];
    for (int y=0;y<rsize;y++)
      for (int x=0;x<rsize;x++)
	rdata[(y*rsize) + x] = rrand();
  }

  ~Noise2D()
  {
    delete[] rdata;
  }

  float get(int x, int y)
  {
    x = x & (rsize - 1);
    y = y & (rsize - 1);
    return rdata[(y*rsize) + x];
  }

  float cubic(float x, float y)
  {
    int ix = rfloor(x);
    float fx = x - ix;
    int iy = rfloor(y);
    float fy = y - iy;

    float r = cubicFilter4f(fy,
			    cubicFilter4f(fx, get(ix-1, iy-1), get(ix, iy-1), get(ix+1, iy-1), get(ix+2, iy-1)),
			    cubicFilter4f(fx, get(ix-1, iy),   get(ix, iy),   get(ix+1, iy),   get(ix+2, iy)),
			    cubicFilter4f(fx, get(ix-1, iy+1), get(ix, iy+1), get(ix+1, iy+1), get(ix+2, iy+1)),
			    cubicFilter4f(fx, get(ix-1, iy+2), get(ix, iy+2), get(ix+1, iy+2), get(ix+2, iy+2)) );

    return rclamp(r, 0.0, 1.0);
  }
};

class Noise3D
{
public:
  float* rdata;
  int rsize;

  Noise3D(int size=64)
  {
    rsize = 1;
    while (rsize < size) rsize<<=1;
    rdata = new float[rsize*rsize*rsize];
    for (int z=0;z<rsize;z++)
      for (int y=0;y<rsize;y++)
	for (int x=0;x<rsize;x++)
	  rdata[((z*rsize+y)*rsize) + x] = rrand();
  }

  ~Noise3D()
  {
    delete[] rdata;
  }

  float get(int x, int y, int z)
  {
    x = x & (rsize - 1);
    y = y & (rsize - 1);
    z = z & (rsize - 1);
    return rdata[((z*rsize+y)*rsize) + x];
  }

  float cubic(float x, float y, float z)
  {
    int ix = rfloor(x);
    float fx = x - ix;
    int iy = rfloor(y);
    float fy = y - iy;
    int iz = rfloor(z);
    float fz = z - iz;

    float xknots[4], yknots[4], zknots[4];

    for (int k = -1; k <= 2; k++) {
      for (int j = -1; j <= 2; j++) {
	for (int i = -1; i <= 2; i++) {
	  xknots[i+1] = get(ix+i, iy+j, iz+k);
	}
	yknots[j+1] = cubicFilterFast4fv(fx, xknots);
      }
      zknots[k+1] = cubicFilterFast4fv(fy, yknots);
    }
    float r = cubicFilterFast4fv(fz, zknots);

    return rclamp(r, 0.0f, 1.0f);
  }
};

