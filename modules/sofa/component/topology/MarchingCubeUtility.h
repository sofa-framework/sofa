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

class MarchingCubeUtility
{

    typedef unsigned int IdVertex;
    struct GridCell
    {
        float val[8];
        Vec3f pos[8];
    };


public:
    MarchingCubeUtility() {};
    MarchingCubeUtility(const Vec<3,int> &_size, const Vec<3,int>  &_gridsize):size(_size), gridsize(_gridsize) {};
    ~MarchingCubeUtility() {};

    void setSize(const Vec<3,int> &_size) {size = _size;};
    void setGridSize(const Vec<3,int> &_gridsize) {gridsize = _gridsize;};

    /// given a set of data (size of the data and size of the marching cube beeing defined previously), we construct the surface.
    /// mesh is a vector containing the triangles defined as a sequence of three indices
    /// map_indices gives the correspondance between an indice and a 3d position in space
    void RenderMarchCube(const float *data,  const float isolevel,
            sofa::helper::vector< IdVertex >   &mesh,
            std::map< IdVertex, Vec3f>  &map_indices) const ;

    /// given a set of data (size of the data and size of the marching cube beeing defined previously), we construct a Sofa mesh.
    void createMesh( const float *data,  const float isolevel, sofa::helper::io::Mesh &m) const;

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

    Vec<3,int> size;
    Vec<3,int> gridsize;
};

}
}
}
#endif

