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

#include <sofa/helper/helper.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/io/Mesh.h>
#include <map>
#include <set>


namespace sofa
{

namespace helper
{
using sofa::defaulttype::Vec;
using sofa::defaulttype::Vector3;
using sofa::helper::vector;
using std::set;

class SOFA_HELPER_API MarchingCubeUtility
{
public:
    typedef unsigned int PointID;
    typedef Vec<3, int> Vec3i;

public:
    MarchingCubeUtility();

    ~MarchingCubeUtility() {};

    void setDataResolution ( const Vec3i   &resolution )
    {
        dataResolution = resolution;
        setBoundingBox( Vec3i( 0, 0, 0), resolution);
    }

    void setDataVoxelSize ( const Vector3 &voxelSize )
    {
        dataVoxelSize = voxelSize;
    }

    void setStep ( const unsigned int step )
    {
        cubeStep = step;
    }

    void setConvolutionSize ( const unsigned int convolutionSize )
    {
        this->convolutionSize = convolutionSize;
    }

    /// Set the bounding box (in the data space) to apply mCube localy.
    void setBoundingBox ( const Vec3i& min, const Vec3i& size )
    {
        this->bbox.min = min;
        this->bbox.max = min + size;
        assert( bbox.min[0] >= 0);
        assert( bbox.min[1] >= 0);
        assert( bbox.min[2] >= 0);
        assert( bbox.max[0] <= dataResolution[0]);
        assert( bbox.max[1] <= dataResolution[1]);
        assert( bbox.max[2] <= dataResolution[2]);
    }

    /// Set the border to localy remesh from real coords
    void setBordersFromRealCoords( const vector<set<Vector3> >& borders)
    {
        this->borders.clear();

        Vector3 resolution = dataResolution.linearProduct(dataVoxelSize);
        resolution = Vector3( 1/resolution[0], 1/resolution[1], 1/resolution[2]);
        for( vector<set<Vector3> >::const_iterator itBorders = borders.begin(); itBorders != borders.end(); itBorders++)
        {
            set<Vec3i> border;
            for( set<Vector3>::const_iterator it = itBorders->begin(); it != itBorders->end(); it++)
            {
                Vec3i cube = ((*it) - dataVoxelSize/2.0).linearProduct( resolution) * 2.0 - Vector3( 1.0f, 1.0f, 1.0f);
                border.insert( cube);
            }
            this->borders.push_back( border);
        }
    }

    /// Propagate the triangulation surface creation from a cell.
    void propagateFrom ( const Vec3i coord,
            const unsigned char *_data, const float isolevel,
            sofa::helper::vector< PointID >& triangles,
            sofa::helper::vector< Vector3 >& vertices,
            helper::vector< helper::vector<unsigned int /*regular grid space index*/> >* triangleIndexInRegularGrid = NULL ) const;

    /// given a set of data (size of the data and size of the marching cube beeing defined previously),
    /// we construct the surface.
    /// mesh is a vector containing the triangles defined as a sequence of three indices
    /// map_indices gives the correspondance between an indice and a 3d position in space
    void run ( const unsigned char *data, const float isolevel,
            sofa::helper::vector< PointID > &triangles,
            sofa::helper::vector< Vector3>  &vertices,
            helper::vector< helper::vector<unsigned int /*regular grid space index*/> > *triangleIndexInRegularGrid = NULL ) const;

//TODO// c null ces graines, ca n'a aucun interet pratique. Virer pour une methode qui propage a partir d'un maillage existant par exemple... => se servir des bordures pour ca et virer les graines ! :)
    /// Same as the previous function but the surfaces are constructed by propagating from seeds.
    /// Faster than previous but it need the precomputation of the seeds.
    void run ( const unsigned char *data, const vector<Vec3i>& seeds,
            const float isolevel,
            sofa::helper::vector< PointID > &triangles,
            sofa::helper::vector< Vector3>  &vertices,
            helper::vector< helper::vector<unsigned int /*regular grid space index*/> > *triangleIndexInRegularGrid = NULL ) const;

    /// given a set of data (size of the data and size of the marching cube beeing defined previously),
    /// we construct a Sofa mesh.
    void run ( const unsigned char *data,  const float isolevel, sofa::helper::io::Mesh &m ) const;

    /// given a set of data, find seeds to run quickly.
    void findSeeds( vector<Vec3i>& seeds, const unsigned char *_data);

    /// Given coords in the scene, find seeds coords.
    void findSeedsFromRealCoords( vector<Vec3i>& mCubeCoords, const vector<Vec3i>& realCoords) const;
private:

    struct GridCell
    {
        float val[8];
        Vector3 pos[8];
    };

    struct BoundingBox
    {
        Vec3i min;
        Vec3i max;
    };

    inline void initCell( GridCell& cell, const Vec3i& coord, const vector< float >& data, const Vector3& gridStep, const Vec3i& dataGridStep) const;

    inline void vertexInterp ( Vector3 &p, const float isolevel, const Vector3 &p1, const Vector3 &p2, const float valp1, const float valp2 ) const ;

    inline bool testGrid ( const float v, const float isolevel ) const;

    inline void updateTriangleInRegularGridVector ( helper::vector< helper::vector<unsigned int /*regular grid space index*/> >& triangleIndexInRegularGrid, const Vec3i& coord, const GridCell& cell, const Vec3i& gridSize, unsigned int nbTriangles) const;

    int polygonise ( const GridCell &grid, int& cubeConf, const float isolevel,
            sofa::helper::vector< PointID > &triangles,
            std::map< Vector3, PointID> &map_vertices,
            sofa::helper::vector< Vector3 > &map_indices ) const ;

    bool getVoxel ( unsigned int index, const unsigned char *dataVoxels ) const
    {
        const int i = index%8;
        return ( ( dataVoxels[index>>3]& ( ( int ) ( pow ( 2.0f, i ) ) ) ) >> i ) == 1;
    }

    void findConnectedVoxels( set<Vec3i>& connectedVoxels, const Vec3i& from, const vector<float>& data);

    void createGaussianConvolutionKernel ( vector< float >  &convolutionKernel ) const;

    void applyConvolution ( const float* convolutionKernel,
            unsigned int x, unsigned int y, unsigned int z,
            const float *input_data,
            float *output_data ) const;

    void smoothData ( float *data ) const;

private:
    unsigned int  cubeStep;
    unsigned int  convolutionSize;
    Vec3i     dataResolution;
    Vector3     dataVoxelSize;
    BoundingBox bbox;
    vector<set<Vec3i> > borders;
};

extern SOFA_HELPER_API const int MarchingCubeEdgeTable[256];
extern SOFA_HELPER_API const int MarchingCubeFaceTable[256];
extern SOFA_HELPER_API const int MarchingCubeTriTable[256][16];

} // namespace helper

} // namespace sofa

#endif

