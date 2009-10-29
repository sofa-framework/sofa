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
  if (!fgets(line,sizeof(line),fp)) { fclose(fp); return false; }
  bool color = (strchr(line,'C')!=NULL);
  bool normals = (strchr(line,'N')!=NULL);
  bool texcoord = (strchr(line,'T')!=NULL);
  if (!(line[0]>='0' && line[0]<='9'))
      if (!fgets(line,sizeof(line),fp)) { fclose(fp); return false; }
  sscanf(line,"%d %d %d",&nbp, &nbf, &nbe);
  
  int flags = MESH_POINTS_POSITION|MESH_FACES;
  if (texcoord) flags |= MESH_POINTS_TEXCOORD;
  if (normals) flags |= MESH_POINTS_NORMAL;
  if (color) flags |= MESH_POINTS_COLOR;
  init(nbp, nbf, flags);
  for(int i=0;i<nbp;i++)
  {
    Vec3f p;
    if (fscanf(fp,"%f %f %f",&p[0], &p[1], &p[2])!=3) { fclose(fp); return false; }
    PP(i) = p;
    if (normals)
    {
      Vec3f pn;
      if (fscanf(fp,"%f %f %f",&pn[0], &pn[1], &pn[2])!=3) { fclose(fp); return false; }
      PN(i) = pn;
    }
    if (color)
    {
      /// @TODO !!!
    }
    if (texcoord)
    {
      Vec2f pt;
      if (fscanf(fp,"%f %f",&pt[0], &pt[1])!=2) { fclose(fp); return false; }
      PT(i) = pt;
    }
  }
  int fnum = 0;
  for(int i=0;i<nbf;i++)
  {
    int nv = 0;
    if (fscanf(fp,"%d",&nv)!=1) { fclose(fp); return false; }
    Vec3i f;
    if (0<nv)
    {
        if (fscanf(fp,"%d",&f[0])!=1) { fclose(fp); return false; }
    }
    if (1<nv)
    {
        if (fscanf(fp,"%d",&f[1])!=1) { fclose(fp); return false; }
    }
    for (int j=2;j<nv;j++)
    {
        if (fscanf(fp,"%d",&f[2])!=1) { fclose(fp); return false; }
      FP(fnum++)=f;
      f[1] = f[2];
    }
    // read the rest of the line
    line [sizeof(line)-1]='\0';
    if (!fgets(line, sizeof(line), fp)) { /*fclose(fp); return false;*/ }
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
