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
public:
    typedef unsigned int PointID;
    typedef Vec<3, int> Vec3i;

public:
    MarchingCubeUtility();

    ~MarchingCubeUtility() {};

    void setDataResolution(const Vec3i   &resolution)
    {
        dataResolution = resolution;
    }

    void setDataVoxelSize(const Vector3	&voxelSize)
    {
        dataVoxelSize = voxelSize;
    }

    void setStep(const unsigned int step)
    {
        cubeStep = step;
    }

    void setConvolutionSize(const unsigned int convolutionSize)
    {
        this->convolutionSize = convolutionSize;
    }

    /// given a set of data (size of the data and size of the marching cube beeing defined previously),
    /// we construct the surface.
    /// mesh is a vector containing the triangles defined as a sequence of three indices
    /// map_indices gives the correspondance between an indice and a 3d position in space
    void run( const unsigned char *data,  const float isolevel,
            sofa::helper::vector< PointID > &triangles,
            sofa::helper::vector< Vector3>  &vertices) const;

    /// given a set of data (size of the data and size of the marching cube beeing defined previously),
    /// we construct a Sofa mesh.
    void run(  const unsigned char *data,  const float isolevel, sofa::helper::io::Mesh &m) const;

private:

    struct GridCell
    {
        float val[8];
        Vector3 pos[8];
    };

    inline void vertexInterp(const float isolevel, const Vector3 &p1, const Vector3 &p2,
            const float valp1, const float valp2, Vector3 &p) const ;

    inline bool testGrid(const float v, const float isolevel) const;

    int polygonise(const GridCell &grid, const float isolevel,
            sofa::helper::vector< PointID > &triangles,
            std::map< Vector3, PointID> &map_vertices,
            std::map< PointID, Vector3> &map_indices,
            unsigned int &ID) const ;

    bool getVoxel(unsigned int index, const unsigned char *dataVoxels) const
    {
        const int i = index%8;
        return ((dataVoxels[index>>3]&((int)(pow(2.0f, i)))) >> i) == 1;
    }

    void createGaussianConvolutionKernel(vector< float >  &convolutionKernel) const;

    void applyConvolution(const float* convolutionKernel,
            unsigned int x, unsigned int y, unsigned int z,
            const float *input_data,
            float *output_data) const;

    void smoothData( float *data) const;

private:
    unsigned int	cubeStep;
    unsigned int	convolutionSize;
    Vec3i			dataResolution;
    Vector3			dataVoxelSize;

};

extern const int MarchingCubeEdgeTable[256];
extern const int MarchingCubeTriTable[256][16];

} // namespace helper

} // namespace sofa

#endif

