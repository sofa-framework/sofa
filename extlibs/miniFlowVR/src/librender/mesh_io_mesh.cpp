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
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <flowvr/render/mesh.h>

namespace flowvr
{

namespace render
{

static bool swap = false;

static bool swapread(void* ptr, size_t size, FILE* fp)
{
  //std::cout << "Reading "<<size<<std::endl;
  if (fread(ptr, size, 1, fp)<=0) return false;
  //std::cout << "Reading "<<size<<" OK"<<std::endl;
  if (swap)
  {
    //std::cout << "Swapping "<<size<<std::endl;
    unsigned int* p = (unsigned int*)ptr;
    while (size>=sizeof(unsigned int))
    {
      *p = ftl::dswap(*p);
      ++p;
      size-=sizeof(unsigned int);
    }
  }
  return true;
}


template<class T>
static bool swapread(std::vector<T>& dest, int size, FILE* fp)
{
  dest.resize(size/sizeof(T));
  return swapread(&(dest[0]),size,fp);
}

template<class T>
static bool swapread(std::vector<T>& dest, FILE* fp)
{
  int size=0;
  if (!swapread(&size,sizeof(int),fp)) return false;
  //std::cout << "size = "<<size<<std::endl;
  return swapread(dest, size, fp);
}

template<class T>
static bool swapwrite(const std::vector<T>& src, FILE* fp)
{
  int size=src.size()*sizeof(T);
  if (fwrite(&size,sizeof(int),1,fp)<=0) return false;
  if (fwrite(&(src[0]),size,1,fp)<=0) return false;
  return true;
}

template<class T>
static bool swapread_malloc(T*& dest, bool swapdata, FILE* fp)
{
  int size=0;
  if (!swapread((void *)&size,sizeof(int),fp)) return false;
  if (size<(int)sizeof(T)) dest = NULL;
  else
  {
    dest = (T*)malloc(size);
    if (swapdata)
    {
      if (!swapread((void*)dest, size, fp)) return false;
    }
    else
    {
      if (!swapread((void*)dest, sizeof(T), fp)) return false;
      if (fread(dest+1, size-sizeof(T), 1, fp)<=0) return false;
    }
  }
  return true;
}

template<class T>
static bool swapwrite(const T* src, FILE* fp)
{
  int size=0;
  if (src!=NULL) size = src->size();
  if (fwrite(&size,sizeof(int),1,fp)<=0) return false;
  if (size>0)
  if (fwrite(src,size,1,fp)<=0) return false;
  return true;
}

bool Mesh::loadMesh(const char* filename)
{
#ifdef WIN32
  FILE* fp = fopen(filename,"rb");
#else
  FILE* fp = fopen(filename,"r");
#endif
  if (fp==NULL) return false;
  std::cout<<"Loading Mesh file "<<filename<<std::endl;
  int magic = 0;
  if (fread(&magic, sizeof(int), 1, fp)<=0) return false;
  if (magic == MESH_MAGIC_SWAP)
  {
    std::cout << "File endianness is swapped"<<std::endl;
    swap = true;
    magic = ftl::dswap(magic);
  }
  else swap = false;
  if (magic != MESH_MAGIC)
  {
    std::cerr << "File "<<filename<<" is NOT a valid Mesh file"<<std::endl;
    fclose(fp);
    return false;
  }
  std::cout << "File "<<filename<<" is a valid Mesh file"<<std::endl;
  if(!swapread(&attrib,sizeof(int),fp)) return false;
  std::cout << "Attribs: 0x"<<std::hex<<attrib<<std::dec<<std::endl;
  int nbp=0,nbf=0;
  if(!swapread(&nbp,sizeof(int),fp)) return false;
  if(!swapread(&nbf,sizeof(int),fp)) return false;
  std::cout<<nbp<<" points, "<<nbf<<" faces"<<std::endl;
  if (getAttrib(MESH_POINTS_POSITION))
    if (!swapread(points_p,fp)) return false;
  if (getAttrib(MESH_POINTS_NORMAL))
    if (!swapread(points_n,fp)) return false;
  if (getAttrib(MESH_POINTS_TEXCOORD))
    if (!swapread(points_t,fp)) return false;
  if (getAttrib(MESH_POINTS_COLOR))
    if (!swapread(points_c,fp)) return false;
  if (getAttrib(MESH_POINTS_VALUE))
    if (!swapread(points_v,fp)) return false;
  if (getAttrib(MESH_POINTS_GROUP))
  {
    if (!swapread(groups_p0,fp)) return false;

    // recompute the points_g array
    points_g.resize(nbp);
    int g0 = 0;
    int p0 = 0;
    for (int g=0; g<nbp; g++)
    {
      int p = getGP0(g);
      if (p < 0) continue; // points are linked to position groups and not normal subgroups
      while (p0 < p)
      {
	  PG(p0) = g0;
	  ++p0;
      }
      g0 = g;
    }
    while (p0 < nbp)
    {
      PG(p0) = g0;
      ++p0;
    }
  }
  //std::cout << "Load Faces"<<std::endl;
  if (getAttrib(MESH_FACES))
    if (!swapread(faces_p,fp)) return false;
  //std::cout << "Load DistMap"<<std::endl;
  if (getAttrib(MESH_DISTMAP))
    if (!swapread_malloc(distmap,true,fp)) return false;
  //std::cout << "Load Voxel"<<std::endl;
  if (getAttrib(MESH_VOXEL))
    if (!swapread_malloc(voxel,false,fp)) return false;
  fclose(fp);
  calcNormals(); // recompute normals
  calcBBox();
  return true;
}

bool Mesh::saveMesh(const char* filename) const
{
  FILE* fp = fopen(filename,"w+");
  if (fp==NULL) return false;
  std::cout<<"Writing Mesh file "<<filename<<std::endl;
  std::cout<<nbp()<<" points, "<<nbf()<<" faces"<<std::endl;
  int i = MESH_MAGIC;
  if(fwrite(&i,sizeof(int),1,fp)<=0) return false;
  i = attrib;
  if (fwrite(&i,sizeof(int),1,fp)<=0) return false;
  i = nbp();
  if (fwrite(&i,sizeof(int),1,fp)<=0) return false;
  i = nbf();
  if (fwrite(&i,sizeof(int),1,fp)<=0) return false;
  if (getAttrib(MESH_POINTS_POSITION))
    if (!swapwrite(points_p,fp)) return false;
  if (getAttrib(MESH_POINTS_NORMAL))
    if (!swapwrite(points_n,fp)) return false;
  if (getAttrib(MESH_POINTS_TEXCOORD))
    if (!swapwrite(points_t,fp)) return false;
  if (getAttrib(MESH_POINTS_COLOR))
    if (!swapwrite(points_c,fp)) return false;
  if (getAttrib(MESH_POINTS_VALUE))
    if (!swapwrite(points_v,fp)) return false;
  if (getAttrib(MESH_POINTS_GROUP))
    if (!swapwrite(groups_p0,fp)) return false;
  if (getAttrib(MESH_FACES))
    if (!swapwrite(faces_p,fp)) return false;
  if (getAttrib(MESH_DISTMAP))
    if (!swapwrite(distmap,fp)) return false;
  if (getAttrib(MESH_VOXEL))
    if (!swapwrite(voxel,fp)) return false;
  fclose(fp);
  return true;
}

} // namespace render

} // namespace flowvr
