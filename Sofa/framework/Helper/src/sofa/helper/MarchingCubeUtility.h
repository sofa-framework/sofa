/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/helper/config.h>

#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>
#include <sofa/helper/set.h>
#include <sofa/helper/io/Mesh.h>
#include <map>

namespace sofa::helper
{
using sofa::type::Vec;
using sofa::type::Vec3;
using sofa::type::vector;

class SOFA_HELPER_API MarchingCubeUtility
{
public:
    typedef sofa::Index PointID;
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vec3i, sofa::type::Vec3i);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vec6i, sofa::type::Vec6i);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Real, SReal);

    MarchingCubeUtility();

    ~MarchingCubeUtility() {};

    void setDataResolution ( const type::Vec3i   &resolution )
    {
        dataResolution = resolution;
        setROI( type::Vec3i ( 0, 0, 0 ), resolution );
        setBoundingBox ( type::Vec3i ( 0, 0, 0 ), resolution );
    }

    void setDataVoxelSize ( const type::Vec3 &voxelSize )
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

    /// Set the bounding box from real coords to apply mCube localy.
    void setBoundingBoxFromRealCoords ( const type::Vec3& min, const type::Vec3& max )
    {
        type::Vec3 gridSize = dataVoxelSize * cubeStep;
        gridSize = type::Vec3 ( 1_sreal / gridSize[0], 1_sreal / gridSize[1], 1_sreal / gridSize[2] );

        //          Vec3i bbMin = ( min - verticesTranslation - ( dataVoxelSize/2.0 ) ).linearProduct ( gridSize );
        //          Vec3i bbMax = ( max - verticesTranslation - ( dataVoxelSize/2.0 ) ).linearProduct ( gridSize );
        setBoundingBox(
                    {static_cast<int>(min.x()), static_cast<int>(min.y()), static_cast<int>(min.z())},
                    {static_cast<int>(max.x()), static_cast<int>(max.y()), static_cast<int>(max.z())});
    }

    /// Set the bounding box (in the data space) to apply mCube localy.
    void setROI ( const type::Vec3i& min, const type::Vec3i& max )
    {
        this->roi.min = min;
        this->roi.max = max;
        if ( roi.min[0] < 0 ) roi.min[0] = 0;
        if ( roi.min[1] < 0 ) roi.min[1] = 0;
        if ( roi.min[2] < 0 ) roi.min[2] = 0;
        if ( roi.max[0] > dataResolution[0] )roi.max[0] = dataResolution[0];
        if ( roi.max[1] > dataResolution[1] )roi.max[1] = dataResolution[1];
        if ( roi.max[2] > dataResolution[2] )roi.max[2] = dataResolution[2];
    }

    /// Set the bounding box (in the data space) to apply mCube localy.
    void setBoundingBox ( const type::Vec6i& roi )
    {
        const type::Vec3i _min( roi[0], roi[1], roi[2]);
        const type::Vec3i _max( roi[3], roi[4], roi[5]);
        setBoundingBox( _min, _max);
    }

    /// Set the bounding box (in the data space) to apply mCube localy.
    void setBoundingBox ( const type::Vec3i& min, const type::Vec3i& max )
    {
        this->bbox.min = min;
        this->bbox.max = max;
        if ( bbox.min[0] < 0 ) bbox.min[0] = 0;
        if ( bbox.min[1] < 0 ) bbox.min[1] = 0;
        if ( bbox.min[2] < 0 ) bbox.min[2] = 0;
        if ( bbox.max[0] > dataResolution[0] )bbox.max[0] = dataResolution[0];
        if ( bbox.max[1] > dataResolution[1] )bbox.max[1] = dataResolution[1];
        if ( bbox.max[2] > dataResolution[2] )bbox.max[2] = dataResolution[2];
    }

    /// given a set of data (size of the data and size of the marching cube beeing defined previously),
    /// we construct the surface.
    /// mesh is a vector containing the triangles defined as a sequence of three indices
    /// map_indices gives the correspondance between an indice and a 3d position in space
    void run ( unsigned char *data, const float isolevel,
               sofa::type::vector< PointID > &triangles,
               sofa::type::vector< type::Vec3>  &vertices,
               type::vector< type::vector<unsigned int> > *triangleIndexInRegularGrid = nullptr ) const;

    /// Same as the previous function but the surfaces are constructed by propagating from seeds.
    /// Faster than previous but it need the precomputation of the seeds.
    void run ( unsigned char *_data, const sofa::type::vector< type::Vec3i > & seeds,
               const float isolevel,
               sofa::type::vector< PointID >& mesh,
               sofa::type::vector< type::Vec3>& vertices,
               std::map< type::Vec3, PointID>& map_vertices,
               type::vector< type::vector<unsigned int> >*triangleIndexInRegularGrid,
               bool propagate ) const;

    /// Same as the previous function but the surfaces are constructed by propagating from seeds.
    /// Faster than previous but it need the precomputation of the seeds.
    void run ( unsigned char *data, const vector<type::Vec3i>& seeds,
               const float isolevel,
               sofa::type::vector< PointID > &triangles,
               sofa::type::vector< type::Vec3>  &vertices,
               type::vector< type::vector<unsigned int> > *triangleIndexInRegularGrid = nullptr,
               bool propagate = true ) const;

    /// given a set of data (size of the data and size of the marching cube beeing defined previously),
    /// we construct a Sofa mesh.
    void run ( unsigned char *data,  const float isolevel, sofa::helper::io::Mesh &m ) const;

    /// given a set of data, find seeds to run quickly.
    void findSeeds ( vector<type::Vec3i>& seeds, const float isoValue, unsigned char *_data );

    /// Given coords in the scene, find seeds coords.
    void findSeedsFromRealCoords ( vector<type::Vec3i>& mCubeCoords, const vector<type::Vec3>& realCoords ) const;

    /// Set the offset to add to each new vertex index in the triangles array.
    void setVerticesIndexOffset( unsigned int verticesIndexOffset);

    /// Set the translation to add to each new vertex in the triangles array.
    void setVerticesTranslation( type::Vec3 verticesTranslation);

private:

    struct GridCell
    {
        float val[8];
        type::Vec3 pos[8];
    };

    struct BoundingBox
    {
        type::Vec3i min;
        type::Vec3i max;
    };

    inline void initCell ( GridCell& cell, const type::Vec3i& coord, const unsigned char* data, const type::Vec3& gridStep, const type::Vec3i& dataGridStep ) const;

    inline void vertexInterp ( type::Vec3 &p, const float isolevel, const type::Vec3 &p1, const type::Vec3 &p2, const float valp1, const float valp2 ) const ;

    inline bool testGrid ( const float v, const float isolevel ) const;

    inline void updateTriangleInRegularGridVector ( type::vector< type::vector<unsigned int /*regular grid space index*/> >& triangleIndexInRegularGrid, const type::Vec3i& coord, const GridCell& cell, unsigned int nbTriangles ) const;

    int polygonise ( const GridCell &grid, int& cubeConf, const float isolevel,
                     sofa::type::vector< PointID > &triangles,
                     std::map< type::Vec3, PointID> &map_vertices,
                     sofa::type::vector< type::Vec3 > &map_indices ) const ;

    bool getVoxel ( unsigned int index, const unsigned char *dataVoxels ) const
    {
        const int i = index%8;
        return ( ( dataVoxels[index>>3]& ( ( int ) ( pow ( 2.0f, i ) ) ) ) >> i ) == 1;
    }

    void findConnectedVoxels ( std::set<unsigned int>& connectedVoxels, const float isoValue, const type::Vec3i& from, unsigned char* data );

    void createGaussianConvolutionKernel ( vector< float >  &convolutionKernel ) const;

    void applyConvolution ( const float* convolutionKernel,
                            unsigned int x, unsigned int y, unsigned int z,
                            const unsigned char *input_data,
                            unsigned char *output_data ) const;

    void smoothData ( unsigned char *data ) const;

    /// Propagate the triangulation surface creation from a cell.
    void propagateFrom ( const sofa::type::vector<type::Vec3i>& coord,
                         unsigned char* data, const float isolevel,
                         sofa::type::vector< PointID >& triangles,
                         sofa::type::vector< type::Vec3 >& vertices,
                         std::set<type::Vec3i>& generatedCubes,
                         std::map< type::Vec3, PointID>& map_vertices,
                         type::vector< type::vector<unsigned int> >* triangleIndexInRegularGrid = nullptr,
                         bool propagate = true ) const;

    unsigned int  cubeStep;
    unsigned int  convolutionSize;
    type::Vec3i     dataResolution;
    type::Vec3     dataVoxelSize;
    BoundingBox bbox; //bbox used to remesh
    BoundingBox roi; // Set value to 0 on this limit to always obtain manifold mesh. (Set to dataResolution by default but can be changed for ROI)
    unsigned int verticesIndexOffset;
    type::Vec3 verticesTranslation;
};

extern SOFA_HELPER_API const int MarchingCubeEdgeTable[256];
extern SOFA_HELPER_API const int MarchingCubeFaceTable[256];
extern SOFA_HELPER_API const int MarchingCubeTriTable[256][16];
} // namespace sofa::helper

