/******* COPYRIGHT ************************************************
*                                                                 *
*                         FlowVR Render                           *
*                   Parallel Rendering Modules                    *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 2005 by                                           *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU GPL, please refer to the      *
* COPYING file for further information.                           *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 * 
*******************************************************************
*                                                                 *
* File: ./src/librender/mesh.cpp                                  *
*                                                                 *
* Contacts:                                                       *
*                                                                 *
******************************************************************/
#include <flowvr/render/mesh.h>
#include <flowvr/render/mesh.inl>
#ifndef MINI_FLOWVR
#include <flowvr/render/chunkwriter.h>
#endif
#include <ftl/type.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace flowvr
{

namespace render
{

/// Brute-force distance computation
float Mesh::calcDist(Vec3f pos) const
{
  if (nbf()<0) return 1e10;
  // Init with first point
  Vec3f dp = pos-getPP(0);
  bool inside = dot(dp,getPN(0))<=0;
  float dist2 = dp.norm2();

  enum { TFACE=0, TEDGE, TPOINT };
  int nearest_prim = TPOINT;
  int nearest_num = 0;

  // Check faces
  for (int i=0;i<nbf();i++)
  {
    dp = pos-getPP(getFP(i)[0]);
    //std::cout << "dp="<<dp<<std::endl;
    Vec3f n = getFN(i);
    //std::cout << "n="<<n<<std::endl;
    //std::cout << "fu="<<getFU(i)<<std::endl;
    //std::cout << "fv="<<getFV(i)<<std::endl;
    float d = dot(dp,n);
    //std::cout << "d="<<d<<std::endl;
    float d2 = d*d;
    if (d2>=dist2) continue; // too far anyway
    float u = dot(dp,getFU(i));
    //std::cout << "u="<<u<<std::endl;
    if (u<0.0f || u>1.0f) continue; // not in triangle
    float v = dot(dp,getFV(i));
    //std::cout << "v="<<v<<std::endl;
    if (v<0.0f || u+v>1.0f) continue; // not in triangle
    // in triangle and nearest point
    dist2 = d2;
    inside = (d<=0.0f);
    nearest_prim = TFACE;
    nearest_num = i;
  }
  // Check edges
  for (unsigned int i=0, num=0;i<edges.size();i++)
  {
    dp = pos-getPP(i);
    for (std::map<int,Edge>::const_iterator it = edges[i].begin(); it != edges[i].end(); it++, num++)
    {
      //if (it->second.f1!=0 && it->second.f2!=0) continue;
      Vec3f dir = getPP(it->first) - getPP(i);
      float a = dot(dp,dir);
      if (a<0.0f) continue;
      float dir2 = dir.norm2();
      if (a>dir2) continue;
      Vec3f p = dir*(a/dir2);
      float d2 = (p-dp).norm2();
      if (d2>=dist2) continue; // too far
      Vec3f n;
      if (it->second.f1>=0) n+=getFN(it->second.f1);
      if (it->second.f2>=0) n+=getFN(it->second.f2);
      // in edge and nearest point
      dist2 = d2;
      inside = dot(dp,n)<=0.0f;
      nearest_prim = TEDGE;
      nearest_num = num;
    }
  }
  // Check points
  for (int i=1;i<nbp();i++)
  {
    //if (i!=getFP(0)[0] && i!=getFP(0)[1] && i!=getFP(0)[2]) continue;
    dp = pos-getPP(i);
    float d2 = dp.norm2();
    if (d2>=dist2) continue; // too far
    // nearest point
    dist2 = d2;
    inside = dot(dp,getPN(i))<=0.0f;
    nearest_prim = TPOINT;
    nearest_num = i;
  }
  // Quick sanity test
  if (inside && !bb.in(pos))
  {
    int i = nearest_num;
    std::cerr << "DIST ERROR: pos "<<pos<<" was detected as INSIDE with dist="<<sqrt(dist2)<<" while outside of mesh bbox "<<bb<<"\n  nearest primitive is ";
    if (nearest_prim==TPOINT) std::cerr <<" POINT "<<i<<": pos="<<getPP(i)<<" n="<<getPN(i);
    else if (nearest_prim==TFACE) std::cerr << "FACE "<<i<<": P="<<getFP(i)<<" n="<<getFN(i)<<" U="<<getFU(i)<<" V="<<getFV(i)
					    << " p0="<<getPP(getFP(i)[0])<< " p1="<<getPP(getFP(i)[1])<< " p2="<<getPP(getFP(i)[2])
					    <<" u="<<dot(pos-getPP(getFP(i)[0]),getFU(i))<<" v="<<dot(pos-getPP(getFP(i)[0]),getFV(i));
    else if (nearest_prim==TEDGE)
    {
      std::cerr << "EDGE "<<i<<": ";
      for (unsigned int i=0, num=0;i<edges.size();i++)
	for (std::map<int,Edge>::const_iterator it = edges[i].begin(); it != edges[i].end(); it++, num++)
	{
	  if ((int)num==nearest_num)
	  {
	    Vec3f dir = getPP(it->first) - getPP(i);
	    float a = dot(pos-getPP(i),dir);
	    Vec3f n;
	    if (it->second.f1>=0) n+=getFN(it->second.f1);
	    if (it->second.f2>=0) n+=getFN(it->second.f2);
	    std::cerr << "P="<<Vec2i(i,it->first)<<" dir="<<dir<<" f=" << it->second.f1 << "," << it->second.f2 << " n="<<n
		      << " p0="<<getPP(i)<< " p1="<<getPP(it->first)
		      <<" a="<<a<<"("<<a/dir.norm2()<<")";
	  }
	}
    }
    else std::cerr << "UNKNOWN "<<i;
    std::cerr << std::endl;
    inside=false;
  }
  return (inside?-sqrt(dist2):sqrt(dist2));
  //return sqrt(dist2)-0.025;
}

void Mesh::calcDistMap(Mat4x4f mat, int nx, int ny, int nz, float maxDist)
{
  if (distmap!=NULL)
  {
    free(distmap);
    distmap = NULL;
  }
  distmap = (DistMap*)malloc(sizeof(DistMap)+nx*ny*nz*sizeof(float));
  distmap->nx = nx;
  distmap->ny = ny;
  distmap->nz = nz;
  distmap->mat = mat;
  calcExtDistMap<float>(mat, distmap->data, nx, ny, nz, maxDist);
/*
  const float *src = distmap->data;
  for (int z=0;z<distmap->nz;z++)
  {
    for (int y=0;y<distmap->ny;y++)
    {
      for (int x=0;x<distmap->nx;x++)
      {
	std::cout << ' ' << *src;
	++src;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
*/
  setAttrib(MESH_DISTMAP,true);
}

void Mesh::calcDistMap(int nx, int ny, int nz, float maxDist)
{
  BBox bb = calcBBox();
  bb.a -= Vec3f(maxDist,maxDist,maxDist);
  bb.b += Vec3f(maxDist,maxDist,maxDist);
  Vec3f d = bb.b-bb.a;

  d[0] /=(nx-1);
  d[1] /=(ny-1);
  d[2] /=(nz-1);

  float maxd = d[0];
  if (d[1]>maxd) maxd = d[1];
  if (d[2]>maxd) maxd = d[2];

  BBox distbb;
  distbb.a = (bb.a+bb.b - Vec3f((float)(nx-1),(float)(ny-1),(float)(nz-1))*maxd)*0.5f;
  distbb.b = distbb.a + Vec3f((float)(nx-1),(float)(ny-1),(float)(nz-1))*maxd;
  std::cout << "Distance field bbox="<<distbb<<std::endl;

  Mat4x4f m;
  m.identity();
  //m(0,0) = d[0]/(nx-1); m(0,3) = bb.a[0]; //+m(0,0)*0.5;
  //m(1,1) = d[1]/(ny-1); m(1,3) = bb.a[1]; //+m(1,1)*0.5;
  //m(2,2) = d[2]/(nz-1); m(2,3) = bb.a[2]; //+m(2,2)*0.5;
  m(0,0) = maxd; m(0,3) = distbb.a[0];
  m(1,1) = maxd; m(1,3) = distbb.a[1];
  m(2,2) = maxd; m(2,3) = distbb.a[2];
  calcDistMap(m,nx,ny,nz,maxDist);
}

void Mesh::calcDistMap(int n, float maxDist)
{
  calcDistMap(n,n,n,maxDist);
}

} // namespace render

} // namespace flowvr
