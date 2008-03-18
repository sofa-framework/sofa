#ifndef MARCHINGCUBEUTILITY_H
#define MARCHINGCUBEUTILITY_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/io/Mesh.h>
#include <map>


namespace sofa
{

namespace component
{

namespace topology
{

using sofa::defaulttype::Vec3f;
using sofa::defaulttype::Vec;
using sofa::helper::vector;

class MarchingCubeUtility
{

    typedef unsigned int IdVertex;
    struct GridCell
    {
        float val[8];
        Vec3f pos[8];
    };


public:
    MarchingCubeUtility():sizeVoxel(Vec3f(1.0f,1.0f,1.0f)) {};
    MarchingCubeUtility(const Vec<3,int> &_size, const Vec<3,int>  &_gridSize):size(_size), gridSize(_gridSize), sizeVoxel(Vec3f(1.0f,1.0f,1.0f)) {};
    ~MarchingCubeUtility() {};

    void setSize     (const Vec<3,int>   &_size)     {size      = _size;     };
    void setGridSize (const Vec<3,int>   &_gridSize) {gridSize  = _gridSize; };
    void setSizeVoxel(const Vec<3,float> &_sizeVoxel) {sizeVoxel = _sizeVoxel;};

    /// given a set of data (size of the data and size of the marching cube beeing defined previously), we construct the surface.
    /// mesh is a vector containing the triangles defined as a sequence of three indices
    /// map_indices gives the correspondance between an indice and a 3d position in space
    void RenderMarchCube( const unsigned char *data,  const float isolevel,
            sofa::helper::vector< IdVertex >   &mesh,
            std::map< IdVertex, Vec3f>  &map_indices,
            unsigned int CONVOLUTION_LENGTH=0) const ;

    /// given a set of data (size of the data and size of the marching cube beeing defined previously), we construct a Sofa mesh.
    void createMesh(  const unsigned char *data,  const float isolevel, sofa::helper::io::Mesh &m,
            unsigned int CONVOLUTION_LENGTH=0) const;

    void createMesh( const sofa::helper::vector< IdVertex >   &mesh,
            std::map< IdVertex, Vec3f>  &map_indices,
            sofa::helper::io::Mesh &m) const ;

protected:

    inline void VertexInterp(const float isolevel, const Vec3f &p1, const Vec3f &p2, const float valp1, const float valp2, Vec3f &p) const ;
    inline bool testGrid(const float v, const float isolevel) const;

    int Polygonise(const GridCell &grid, const float isolevel,
            sofa::helper::vector< IdVertex > &triangles,
            std::map< Vec3f, IdVertex> &map_vertices,
            std::map< IdVertex, Vec3f> &map_indices,
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
    Vec<3,float> sizeVoxel;

};

}
}
}
#endif

