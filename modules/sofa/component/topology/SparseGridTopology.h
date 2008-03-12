/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_SPARSEGRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_SPARSEGRIDTOPOLOGY_H

#include <string>


#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/MarchingCubeUtility.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/component/topology/RegularGridTopology.h>

#include <sofa/helper/io/Mesh.h>
namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


/** A sparse grid topology. Like a sparse FFD building from the bounding box of the object. Starting from a RegularGrid, only valid cells containing matter (ie intersecting the original surface mesh or totally inside the object) are considered.
Valid cells are tagged by a Type BOUNDARY or INSIDE
WARNING: the corresponding node in the XML file has to be placed BEFORE the MechanicalObject node, in order to excute its init() before the MechanicalObject one in order to be able to give dofs
   */
class SparseGridTopology : public MeshTopology
{
public:

    typedef Vec3d Vec3;
    typedef double Real;
    typedef fixed_array<Vec3d,8> CubeCorners;
    typedef enum {OUTSIDE,INSIDE,BOUNDARY} Type; ///< each cube has a type depending on its filling ratio



    SparseGridTopology();

// 					static const float WEIGHT[8][8];
    static const float WEIGHT27[8][27];
    static const int cornerIndicesFromFineToCoarse[8][8];

    bool load(const char* filename);
    virtual void init();
    void buildAsFinest(); ///< building from a mesh file
    void buildFromFiner(); ///< building by condensating a finer sparse grid (used if setFinerSparseGrid has initializated _finerSparseGrid before calling init() )


    typedef std::map<Vec3,int> MapBetweenCornerPositionAndIndice;///< a vertex indice for a given vertex position in space

    /// connexion between several coarsened levels
    typedef std::vector<fixed_array<int,8> > HierarchicalCubeMap; ///< a cube indice -> corresponding 8 child indices on the potential _finerSparseGrid
    HierarchicalCubeMap _hierarchicalCubeMap;
    typedef helper::vector<int> InverseHierarchicalCubeMap; ///< a fine cube indice -> corresponding coarser cube indice
    InverseHierarchicalCubeMap _inverseHierarchicalCubeMap;

    typedef std::map<int,float> AHierarchicalPointMap;
// 					typedef helper::vector< std::pair<int,float> >  AHierarchicalPointMap;
    typedef helper::vector< AHierarchicalPointMap > HierarchicalPointMap; ///< a point indice -> corresponding 27 child indices on the potential _finerSparseGrid with corresponding weight
    HierarchicalPointMap _hierarchicalPointMap;
    typedef helper::vector< AHierarchicalPointMap > InverseHierarchicalPointMap; ///< a fine point indice -> corresponding some parent points for interpolation
    InverseHierarchicalPointMap _inverseHierarchicalPointMap;
    typedef helper::vector< int > PointMap;
    PointMap _pointMap; ///< a coarse point indice -> corresponding point in finer level
    PointMap _inversePointMap;  ///< a fine point indice -> corresponding point in coarser level


    enum {UP,DOWN,RIGHT,LEFT,BEFORE,BEHIND,NUM_CONNECTED_NODES};
    typedef helper::vector< helper::fixed_array<int,NUM_CONNECTED_NODES> > NodeAdjacency; ///< a node -> its 6 neighboors
    NodeAdjacency _nodeAdjacency;
    typedef helper::vector< helper::vector<int> >NodeCubesAdjacency; ///< a node -> its 8 neighboor cells
    NodeCubesAdjacency _nodeCubesAdjacency;
    typedef helper::vector< helper::vector<int> >NodeCornersAdjacency; ///< a node -> its 8 corners of neighboor cells
    NodeCornersAdjacency _nodeCornersAdjacency;


    Vec<3, int> getN() const { return n.getValue();}
    int getNx() const { return n.getValue()[0]; }
    int getNy() const { return n.getValue()[1]; }
    int getNz() const { return n.getValue()[2]; }

    void setN(Vec<3,int> _n) {n.setValue(_n);}
    void setNx(int _n) { n.setValue(Vec<3,int>(_n             ,n.getValue()[1],n.getValue()[2])); }
    void setNy(int _n) { n.setValue(Vec<3,int>(n.getValue()[0],_n             ,n.getValue()[2])); }
    void setNz(int _n) { n.setValue(Vec<3,int>(n.getValue()[0],n.getValue()[1],_n)             ); }

    void setMin(Vec3d _min) {min.setValue(_min);}
    void setXmin(double _min) { min.setValue(Vec3d(_min             ,min.getValue()[1],min.getValue()[2])); }
    void setYmin(double _min) { min.setValue(Vec3d(min.getValue()[0],_min             ,min.getValue()[2])); }
    void setZmin(double _min) { min.setValue(Vec3d(min.getValue()[0],min.getValue()[1],_min)             ); }


    void setMax(Vec3d _max) {min.setValue(_max);}

    void setXmax(double _max) { max.setValue(Vec3d(_max             ,max.getValue()[1],max.getValue()[2])); }
    void setYmax(double _max) { max.setValue(Vec3d(max.getValue()[0],_max             ,max.getValue()[2])); }
    void setZmax(double _max) { max.setValue(Vec3d(max.getValue()[0],max.getValue()[1],_max)             ); }

    Vec3d getMin() {return min.getValue();}
    double getXmin() { return min.getValue()[0]; }
    double getYmin() { return min.getValue()[1]; }
    double getZmin() { return min.getValue()[2]; }

    Vec3d getMax() {return max.getValue();}
    double getXmax() { return max.getValue()[0]; }
    double getYmax() { return max.getValue()[1]; }
    double getZmax() { return max.getValue()[2]; }

    bool hasPos()  const { return true; }

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findCube(const Vec3& pos, double& fx, double &fy, double &fz);

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findNearestCube(const Vec3& pos, double& fx, double &fy, double &fz);

    /// return the type of the i-th cube
    virtual Type getType( int i );

    SparseGridTopology* getFinerSparseGrid() const {return _finerSparseGrid;}
    void setFinerSparseGrid( SparseGridTopology* fsp ) {_finerSparseGrid=fsp;}
    SparseGridTopology* getCoarserSparseGrid() const {return _coarserSparseGrid;}
    void setCoarserSparseGrid( SparseGridTopology* csp ) {_coarserSparseGrid=csp;}

    RegularGridTopology _regularGrid; ///< based on a corresponding RegularGrid
    vector< int > _indicesOfRegularCubeInSparseGrid; ///< to redirect an indice of a cube in the regular grid to its indice in the sparse grid

    Vec3 getPointPos( int i ) { return Vec3( seqPoints[i][0],seqPoints[i][1],seqPoints[i][2] ); }

    void getMesh( sofa::helper::io::Mesh &m);

protected:

    /// cutting number in all directions
    Data< Vec<3, int>    > n;
    Data< Vec<3, double> > min;
    Data< Vec<3, double> > max;

    Data< Vec<3, int>  > dim_voxels;
    Data< Vec3d >        size_voxel;

    virtual void updateEdges();
    virtual void updateQuads();
    virtual void updateHexas();

    MarchingCubeUtility                 MC;
    vector< float >                     dataVoxels;
    sofa::helper::vector< unsigned int> mesh_MC;
    std::map< unsigned int, Vec3f >     map_indices;
    bool                                _usingMC;

    sofa::helper::vector<Type> _types; ///< BOUNDARY or FULL filled cells

    /// start from a seed cell (i,j,k) the OUTSIDE filling is propagated to neighboor cells until meet a BOUNDARY cell (this function is called from all border cells of the RegularGrid)
    void propagateFrom( const int i, const int j, const int k,
            RegularGridTopology& regularGrid,
            vector<Type>& regularGridTypes,
            vector<bool>& alreadyTested  ) const;

    void computeBoundingBox(const helper::vector<Vec3>& vertices,
            double& xmin, double& xmax,
            double& ymin, double& ymax,
            double& zmin, double& zmax) const;

    void voxelizeTriangleMesh(helper::io::Mesh* mesh,
            RegularGridTopology& regularGrid,
            vector<Type>& regularGridTypes) const;

    void buildFromTriangleMesh(const std::string& filename);

    void buildFromRegularGridTypes(RegularGridTopology& regularGrid, const vector<Type>& regularGridTypes);

    /** Create a sparse grid from a .voxel file
    	.voxel file format (ascii):
    	512  // num voxels x
    	512  // num voxels y
    	246  // num voxels z
    	0.7  // voxels size x [mm]
    	0.7  // voxels size y [mm]
    	2    // voxels size z [mm]
    	0 0 255 0 0 ... // data
    */
    void buildFromVoxelFile(const std::string& filename);
    void buildFromRawVoxelFile(const std::string& filename);

    void constructCollisionModels(const sofa::helper::vector< sofa::component::topology::MeshTopology * > &list_mesh,
            const sofa::helper::vector< sofa::helper::vector< Vec3d >* >            &list_X,
            const sofa::helper::vector< unsigned int> mesh_MC,
            std::map< unsigned int, Vec3f >     map_indices) const;

    SparseGridTopology* _finerSparseGrid; ///< an eventual finer sparse grid that can be used to built this coarser sparse grid
    SparseGridTopology* _coarserSparseGrid; ///< an eventual coarser sparse grid


    /*	/// to compute valid cubes (intersection between mesh segments and cubes)
    typedef struct segmentForIntersection{
    	Vec3 center;
    	Vec3 dir;
    	Real norm;
    	segmentForIntersection(const Vec3& s0, const Vec3& s1)
    	{
    		center = (s0+s1)*.5;
    		dir = center-s0;
    		norm = dir.norm();
    		dir /= norm;
    	};
    } SegmentForIntersection;
    struct ltSegmentForIntersection // for set of SegmentForIntersection
    {
    	bool operator()(const SegmentForIntersection& s0, const SegmentForIntersection& s1) const
    	{
    		return s0.center < s1.center || s0.norm < s1.norm;
    	}
    };
    typedef struct cubeForIntersection{
    	Vec3 center;
    	fixed_array<Vec3,3> dir;
    	Vec3 norm;
    	cubeForIntersection( const CubeCorners&  corners )
    	{
    		center = (corners[7] + corners[0]) * .5;

    		norm[0] = (center[0] - corners[0][0]);
    		dir[0] = Vec3(1,0,0);

    		norm[1] = (center[1] - corners[0][1]);
    		dir[1] = Vec3(0,1,0);

    		norm[2] = (center[2] - corners[0][2]);
    		dir[2] = Vec3(0,0,1);
    	}
    } CubeForIntersection;
    /// return true if there is an intersection between a SegmentForIntersection and a CubeForIntersection
    bool intersectionSegmentBox( const SegmentForIntersection& seg, const CubeForIntersection& cube  ); */

    bool _alreadyInit;

public :
    virtual const SeqCubes& getHexas();
    virtual int getNbPoints() const;

    virtual int getNbHexas()
    {
        return getHexas().size();
    }

    virtual vector< fixed_array<unsigned int, 4> >& getQuads()
    {

        if (_usingMC) return this->seqQuads;
        else
        {
            seqQuads = MeshTopology::getQuads();
            return this->seqQuads;
        }
    }

    virtual vector< fixed_array<unsigned int, 3> >& getTriangles()
    {
        vector< fixed_array<unsigned int, 3> > &t = *seqTriangles.beginEdit();
        if (_usingMC)
        {
            t.resize(mesh_MC.size()/3);
            for (unsigned int i=0; i<t.size(); ++i)
            {
                t[i]=fixed_array<int, 3>(mesh_MC[3*i]-1,mesh_MC[3*i+1]-1,mesh_MC[3*i+2]-1);
            }
        }
        return t;
    }

    virtual vector< fixed_array<double,3> >& getPoints()
    {
        std::cout << getPoints() << "\n";
        if (_usingMC)
        {
            seqPoints.resize(map_indices.size());
            for (unsigned int i=0; i<seqPoints.size(); ++i)
            {
                Vec3f p=map_indices[i+1];
                seqPoints[i] = fixed_array<double,3>(p[0],p[1],p[2]);
            }
        }
        return this->seqPoints;
    }

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
