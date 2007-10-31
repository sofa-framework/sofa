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

bool Mesh::loadOff(const char* filename)
{
  FILE* fp = fopen(filename,"r");
  if (fp==NULL) return false;
  int nbp=0, nbf=0, nbe=0;
  char line[5000];
  
  //fscanf(fp,"%s\n",line);
  fgets(line,sizeof(line),fp);
  bool color = (strchr(line,'C')!=NULL);
  bool normals = (strchr(line,'N')!=NULL);
  bool texcoord = (strchr(line,'T')!=NULL);
  if (line[0]>='0' && line[0]<='9')
  { // no header
    sscanf(line,"%d %d %d",&nbp, &nbf, &nbe);
  }
  else
  {
    fscanf(fp,"%d %d %d\n",&nbp, &nbf, &nbe);
  }
  
  int flags = MESH_POINTS_POSITION|MESH_FACES;
  if (texcoord) flags |= MESH_POINTS_TEXCOORD;
  if (normals) flags |= MESH_POINTS_NORMAL;
  if (color) flags |= MESH_POINTS_COLOR;
  init(nbp, nbf, flags);
  for(int i=0;i<nbp;i++)
  {
    Vec3f p;
    fscanf(fp,"%f %f %f",&p[0], &p[1], &p[2]);
    PP(i) = p;
    if (normals)
    {
      Vec3f pn;
      fscanf(fp,"%f %f %f",&pn[0], &pn[1], &pn[2]);
      PN(i) = pn;
    }
    if (color)
    {
      /// @TODO !!!
    }
    if (texcoord)
    {
      Vec2f pt;
      fscanf(fp,"%f %f",&pt[0], &pt[1]);
      PT(i) = pt;
    }
  }
  int fnum = 0;
  for(int i=0;i<nbf;i++)
  {
    int nv = 0;
    fscanf(fp,"%d",&nv);
    Vec3i f;
    if (0<nv)
      fscanf(fp,"%d",&f[0]);
    if (1<nv)
      fscanf(fp,"%d",&f[1]);
    for (int j=2;j<nv;j++)
    {
      fscanf(fp,"%d",&f[2]);
      FP(fnum++)=f;
      f[1] = f[2];
    }
    // read the rest of the line
    line [sizeof(line)-1]='\0';
    fgets(line, sizeof(line), fp);
  }
  fclose(fp);
  std::cout<<"Loaded file "<<filename<<std::endl;
  std::cout<<this->nbp()<<" final points, "<<this->nbf()<<" triangles."<<std::endl;
  
  // Computing normals
  calcNormals();
  calcBBox();
  
  return true;
}

bool Mesh::saveOff(const char* filename) const
{
  FILE* fp = fopen(filename,"w+");
  if (fp==NULL) return false;
  std::cout<<"Writing Off file "<<filename<<std::endl;
  bool normal   = getAttrib(MESH_POINTS_NORMAL);
  bool texcoord = getAttrib(MESH_POINTS_TEXCOORD);
  //bool group    = getAttrib(MESH_POINTS_GROUP);
  if (texcoord)
    fprintf(fp,"ST");
  if (normal)
    fprintf(fp,"N");
  fprintf(fp,"OFF\n");
  std::cout<<nbp()<<" points, "<<nbf()<<" faces"<<std::endl;
  fprintf(fp,"%d %d %d\n",nbp(),nbf(),0);
  for (int i=0;i<nbp();i++)
  {
    fprintf(fp,"%f %f %f",getPP(i)[0],getPP(i)[1],getPP(i)[2]);
    if (normal)
      fprintf(fp," %f %f %f",getPN(i)[0],getPN(i)[1],getPN(i)[2]);
    if (texcoord)
      fprintf(fp," %f %f",getPT(i)[0],getPT(i)[1]);
    fprintf(fp,"\n");
  }
  for (int i=0;i<nbf();i++)
  {
    fprintf(fp,"3");
    for (int j=0;j<3;j++)
    {
      int p = getFP(i)[j];
      fprintf(fp," %d",p);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
  return true;
}

} // namespace render

} // namespace flowvr
