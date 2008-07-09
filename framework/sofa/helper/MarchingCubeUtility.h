/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef MARCHINGCUBEUTILITY_H
#define MARCHINGCUBEUTILITY_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/io/Mesh.h>
#include <map>


namespace sofa
{

namespace helper
{

using sofa::defaulttype::Vec;
using sofa::defaulttype::Vector3;
using sofa::helper::vector;

class MarchingCubeUtility
{

    typedef unsigned int IdVertex;
    struct GridCell
    {
        float val[8];
        Vector3 pos[8];
    };

public:
    MarchingCubeUtility():sizeVoxel(Vector3(1.0f,1.0f,1.0f)) {};
    MarchingCubeUtility(const Vec<3,int> &_size, const Vec<3,int>  &_gridSize):size(_size), gridSize(_gridSize), sizeVoxel(Vector3(1.0f,1.0f,1.0f)) {};
    ~MarchingCubeUtility() {};

    void setSize     (const Vec<3,int>   &_size)     {size      = _size;     };
    void setGridSize (const Vec<3,int>   &_gridSize) {gridSize  = _gridSize; };
    void setSizeVoxel(const Vec<3,float> &_sizeVoxel) {sizeVoxel = _sizeVoxel;};

    /// given a set of data (size of the data and size of the marching cube beeing defined previously), we construct the surface.
    /// mesh is a vector containing the triangles defined as a sequence of three indices
    /// map_indices gives the correspondance between an indice and a 3d position in space
    void RenderMarchCube( const unsigned char *data,  const float isolevel,
            sofa::helper::vector< IdVertex >   &mesh,
            std::map< IdVertex, Vector3>  &map_indices,
            unsigned int CONVOLUTION_LENGTH=0) const ;

    /// given a set of data (size of the data and size of the marching cube beeing defined previously), we construct a Sofa mesh.
    void createMesh(  const unsigned char *data,  const float isolevel, sofa::helper::io::Mesh &m,
            unsigned int CONVOLUTION_LENGTH=0) const;

    void createMesh( const sofa::helper::vector< IdVertex >   &mesh,
            std::map< IdVertex, Vector3>  &map_indices,
            sofa::helper::io::Mesh &m) const ;

protected:

    inline void VertexInterp(const float isolevel, const Vector3 &p1, const Vector3 &p2, const float valp1, const float valp2, Vector3 &p) const ;
    inline bool testGrid(const float v, const float isolevel) const;

    int Polygonise(const GridCell &grid, const float isolevel,
            sofa::helper::vector< IdVertex > &triangles,
            std::map< Vector3, IdVertex> &map_vertices,
            std::map< IdVertex, Vector3> &map_indices,
            unsigned int &ID) const ;

    bool getVoxel(unsigned int index, const unsigned char *dataVoxels) const
    {
        const int i = index%8;
        return ((dataVoxels[index>>3]&((int)(pow(2.0f, i)))) >> i) == 1;
    };

    void createConvolutionKernel(unsigned int CONVOLUTION_LENGTH, vector< float >  &convolutionKernel) const;
    void applyConvolution(unsigned int CONVOLUTION_LENGTH,unsigned int x, unsigned int y, unsigned int z, const float *original_data, float *data, const vector< float >  &convolutionKernel) const;
    void smoothData( float *data, unsigned int CONVOLUTION_LENGTH) const;

    Vec<3,int> size;
    Vec<3,int> gridSize;
    Vector3 sizeVoxel;

};

extern const int MarchingCubeEdgeTable[256];
extern const int MarchingCubeTriTable[256][16];
}
}
#endif

