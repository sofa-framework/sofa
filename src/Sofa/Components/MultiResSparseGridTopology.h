#ifndef SOFA_COMPONENTS_SPARSEREGULARGRIDTOPOLOGY_H
#define SOFA_COMPONENTS_SPARSEREGULARGRIDTOPOLOGY_H

#include "GridTopology.h"
#include "Common/Vec.h"
#include <Sofa/Components/Common/vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <map>
namespace Sofa
{

namespace Components
{

using namespace Common;
using Common::vector;
typedef Common::Vec3f Vec3;

class MultiResSparseGridTopology : public Core::Topology
{
    int nbPoints;
public:


    /** Make all resolution of sparse grid create with a .vox file

    512  // Image's size x
    512  // Image's size y
    246  // Image's size z
    0.7  // Voxels's size x
    0.7  // Voxels's size y
    2    // Voxels's size z
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 255 0 0 0 0 ........................

    This is a sparse voxel grid:

    -vertexMap contains all vertices without repeating. It consists of Index3D(int i,j,k), vertex(SbVec3f,int)

    -voxelsMap contains all voxels with the indices of each one. It consists of Index3D which represent the voxel center,
    and of int[8] which leads to the 8 vertices of that voxel in vertices array or in the map

    -surfaceSparseGrid is a vector of Index3D, contains centers of the voxels of the surface

     */
    class SparseGrid
    {
    public :
        ///to represent vertices of voxels
        struct Vertex
        {
            Vec3 vertexPosition;
            int index;
            Vertex(float x=0, float y=0, float z=0)
            {
                //position = new float(3);
                vertexPosition[0]=x;
                vertexPosition[1]=y;
                vertexPosition[2]=z;
            };
        };

        ///To represent voxels centers
        struct Index3D
        {
            int i,j,k;
            Index3D(int a=0,int b=0,int c=0):i(a),j(b),k(c)
            {}

            bool operator < (const Index3D& ind ) const
            {
                return ( i<ind.i || i==ind.i && j<ind.j || i==ind.i && j==ind.j && k<ind.k );
            }
        };
        ///to represent the 8 vertices of one voxel
        struct Voxel
        {
            int vertices[8];
            float density;
            Voxel(int v1=0,int v2=0,int v3=0,int v4=0,int v5=0,int v6=0,int v7=0,int v8=0,float d=0.0)
            {
                vertices[0]=v1;
                vertices[1]=v2;
                vertices[2]=v3;
                vertices[3]=v4;
                vertices[4]=v5;
                vertices[5]=v6;
                vertices[6]=v7;
                vertices[7]=v8;
                density = d;
            }
        };

    protected :
        /// End of list for surface voxel
        int FIN_DE_LISTE;
        /// Grid's scale
        float scale;
        /// array of vertices
        vector<Vec3> vertices;
        /// contains centers of the voxels of the surface
        vector<Index3D> surfaceSparseGrid;
        /// number and array of the 6 faces corresponding to each voxels
        int numIndices;
        vector<int> indices;
        /// number and array of the visible faces
        int numSurfaceIndices;
        vector<int> surfaceIndices;
        /// array of density
        float* voxelsDensity;
        /// size of the voxels
        float dimVoxX, dimVoxY, dimVoxZ;
        /// size of the grid
        int dimX,dimY,dimZ;

        /// contains all vertices without repeating
        std::map<Index3D, Vertex> vertexMap;
        /// contains all the voxels, each represented by its centers and its 8 vertices
        std::map<Index3D,Voxel> voxelsMap;

    public:

        SparseGrid(void);//sans rien
        SparseGrid(int);// avec un indice de fin de liste
        SparseGrid(float); // avec uune echelle
        ~SparseGrid(void);
        /// Init filesNo of pgm images, and the by calling the suitable functions, constructs the grid of voxels
        void initFromPNG( char* fileName,int color,int filesNo );
        /// Init vox file, and the by calling the suitable functions, constructs the grid of voxels
        void initFromVOX( char* fileName,int color);
        ///returns the num of the voxels
        unsigned getNumSparseGrid() const
        {
            return voxelsMap.size();
        }
        ///returns the number of vertices of the grid
        unsigned getNumVertices() const
        {
            return vertexMap.size();
        }
        /// returns the size of the indices array for all the voxels
        unsigned getNumFaceIndices() const
        {
            return numIndices;
        }
        /// returns the size of the indices array for faces of the surface
        unsigned getNumSurfaceIndices() const
        {
            return numSurfaceIndices;
        }
        /// returns the array of vertices
        vector< Vec3>& getVertices()
        {
            return vertices;
        }
        const vector<Vec3>& getVertices() const
        {
            return vertices;
        }
        /// returns the array of indices for all the voxels
        vector<int>& getFaceIndices()
        {
            return indices;
        }
        /// return the voxelMap of the voxels
        std::map<Index3D,Voxel>& getSparseGridMap()
        {
            return voxelsMap;
        }
        const std::map<Index3D,Voxel>& getSparseGridMap() const
        {
            return voxelsMap;
        }
        ///return the begin and the end of the map
        std::map<Index3D,Voxel>::iterator getVoxelsMapBegin()
        {
            return voxelsMap.begin();
        }
        std::map<Index3D,Voxel>::iterator getVoxelsMapEnd()
        {
            return voxelsMap.end();
        }
        /// return the vertexMap of the voxels
        std::map<Index3D,Vertex>& getVertexMap()
        {
            return vertexMap;
        }
        const std::map<Index3D,Vertex>& getVertexMap() const
        {
            return vertexMap;
        }
        ///return the begin and the end of the map
        std::map<Index3D,Vertex>::iterator getVertexMapBegin()
        {
            return vertexMap.begin();
        }
        std::map<Index3D,Vertex>::iterator  const getVertexMapEnd()
        {
            return vertexMap.end();
        }

        /// returns the array of density
        float* getSparseGridDensity()
        {
            return voxelsDensity;
        }
        /// returns the array of indices for faces of the surface
        vector<int>& getSurfaceIndices()
        {
            return surfaceIndices;
        }
        /// make the child grid
        int pasResolution(SparseGrid vg);
        /// get voxel size
        float getDimVoxX()
        {
            return dimVoxX;
        }
        float getDimVoxY()
        {
            return dimVoxY;
        }
        float getDimVoxZ()
        {
            return dimVoxZ;
        }
        /// get grid's size
        int getDimX() const
        {
            return dimX;
        }
        int getDimY() const
        {
            return dimY;
        }
        int getDimZ() const
        {
            return dimZ;
        }
        /// get the density of a voxel
        float getDensity(int i,int j,int k);
        /// print the arry of vertices
        void afficherVertices();
        /// print the voxel of the sparse grid map
        void afficherSparseGridMap();
        /// print the vertex of the vertew map
        void afficherVertexMap();
        /// print the arry of indices
        void afficherIndices();


    protected:
        /// Reads filesNo of pgm images, and the by calling the suitable functions, constructs the grid of voxels
        int readFilePNG( char* fileName,int color,int filesNo );
        /// read .voxel file
        int readFileVOX( char* fileName,int color);
        /// gets the pixels of the given color from each image.
        int setPixels( const char *FileName, int color,float plane);
        /// gets the vertices of each voxels and puts them in the map of vertices
        void setVertexMap();
        /// gets the indices necessary to view the grid of voxels, and making the map of voxels
        void setIndicesMap();
        /// gets the surface voxels, and the indices for faces of the surface
        void setSurfaceSparseGrid();
        /// insert a voxel in the sparse grid map wtih his density
        void insertVoxel(int i,int j,int k,float d);
        /// build the rest of the information
        void buildFromSparseGridMap(int allPixels);
        /// gets the vertices of each voxels and puts them in the map of vertices
        void setDensity();
        /// set sparge grid size
        void setDimX(int x)
        {
            dimX = x;
        }
        void setDimY(int y)
        {
            dimY = y;
        }
        void setDimZ(int z)
        {
            dimZ = z;
        }
        /// set voxel size
        void setDimVoxX(float x)
        {
            dimVoxX = x;
        }
        void setDimVoxY(float y)
        {
            dimVoxY = y;
        }
        void setDimVoxZ(float z)
        {
            dimVoxZ = z;
        }
    };


    typedef fixed_array<int, 8> Cube;
    typedef Vec3d Vec3;

    MultiResSparseGridTopology();
    MultiResSparseGridTopology(const char* filevoxel, int resol,float scale);

    /// contain all resolution of the sparse grid
    vector <SparseGrid> vectorSparseGrid;

    /// indices of the current grid
    int resolution;

    ///Size of the grid and its position
    int dimx,dimy,dimz,px,py,pz;

    /// get the postion of the voxel i
    double getPX(int i) const ;
    double getPY(int i) const ;
    double getPZ(int i) const ;

    ///  get the vertex's psotion with its indice or its index
    Vec3 getPoint(int i) const ;
    Vec3 getPoint(int ,int ,int );

    /// ??????
    bool hasPos() const;
    /// number of voxels in the current grid
    int getNbVoxels() const;
    /// number of vertices int th current grid
    int getNbPoints() const;
    /// get the indices of the voxel x y z
    int point(int x, int y, int z);
    /// get the cube of the voxel i for the mapping
    GridTopology::Cube getCube (int i);
    /// get the indice of the cube at the coordinates pos and the barycentric coordinates
    int findCube(const Vec3& pos, double& fx, double &fy, double &fz) const;
    /// get the indice of the nearest cube of the coordinates pos and the barycentric coordinates
    int findNearestCube(const Vec3& pos, double& fx, double &fy, double &fz) const;
    ///find the vertices in the constraint box
    int getIndicesInSpace( std::vector<int>& indices,float xmin,float xmax,float ymin,float ymax,float zmin,float zmax );
    /// set the position (not used)
    void setP0(const Vec3& val)
    {
        p0 = val;
    }
    /// set the interval between voxels
    void setDx(const Vec3& val)
    {
        dx = val;
        inv_dx2 = 1/(dx*dx);
    }
    void setDy(const Vec3& val)
    {
        dy = val;
        inv_dy2 = 1/(dy*dy);
    }
    void setDz(const Vec3& val)
    {
        dz = val;
        inv_dz2 = 1/(dz*dz);
    }
/// get the position (not used)
    const Vec3& getP0() const
    {
        return p0;
    }
    /// get the interval between voxels
    const Vec3& getDx() const
    {
        return dx;
    }
    const Vec3& getDy() const
    {
        return dy;
    }
    const Vec3& getDz() const
    {
        return dz;
    }

protected:
    /// Position of point 0
    Vec3 p0;
    /// Distance between points in the grid. Must be perpendicular to each other
    Vec3 dx,dy,dz;
    double inv_dx2, inv_dy2, inv_dz2;

};

} // namespace Components

} // namespace Sofa

#endif


