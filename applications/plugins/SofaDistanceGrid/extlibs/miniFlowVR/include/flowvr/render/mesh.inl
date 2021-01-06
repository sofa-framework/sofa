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
* File: ./include/flowvr/render/mesh.h                            *
*                                                                 *
* Contacts:                                                       *
*                                                                 *
******************************************************************/
#ifndef FLOWVR_RENDER_MESH_INL
#define FLOWVR_RENDER_MESH_INL

namespace flowvr
{

namespace render
{

template<class Real>
void Mesh::calcExtDistMap(Mat4x4f mat, Real* dest, int nx, int ny, int nz, float /*maxDist*/, float fact)
{
  close(); // make sure the mesh is closed
  calcNormals(true);
  calcEdges();
  for (int z=0;z<nz;z++)
  {
    for (int y=0;y<ny;y++)
      for (int x=0;x<nx;x++)
      {
	*dest = calcDist(transform(mat,Vec3f((float)x,(float)y,(float)z)))*fact;
	++dest;
      }
    std::cout << '.' << std::flush;
  }
  std::cout << std::endl;
}

} // namespace render

} // namespace flowvr

#endif
