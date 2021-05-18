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

//bool Mesh::loadStl(const char* filename)
//{
//  return false;
//}

bool Mesh::saveStl(const char* filename) const
{
  FILE* fp = fopen(filename,"w+");
  if (fp==NULL) return false;
  std::cout<<"Writing Stl file "<<filename<<std::endl;
  //bool res = saveStl(fp);
  fprintf(fp,"solid mesh\n");
  for (int i=0;i<nbf();i++)
  {
      
    fprintf(fp,"face normal %f %f %f\n",getFN(i)[0],getFN(i)[1],getFN(i)[2]);
    fprintf(fp,"  outer loop\n");
    for (int j=0;j<3;j++)
    {
      int p = getFP(i)[j];
      fprintf(fp,"    vertex %f %f %f\n",getPP(p)[0],getPP(p)[1],getPP(p)[2]);
    }
    fprintf(fp,"endloop\n");
    fprintf(fp,"endfacet\n");
  }
  fclose(fp);
  return true;
}

} // namespace render

} // namespace flowvr
