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
#include <sofa/config.h>
#include <sofa/component/topology/container/grid/config.h>

#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/component/topology/container/grid/RegularGridTopology.h>
#include <sofa/helper/MarchingCubeUtility.h>
#include <sofa/type/Vec.h>

#include <sofa/helper/io/Mesh.h>
#include <stack>
#include <string>

namespace sofa::core::loader
{
    class VoxelLoader;
}

namespace sofa::component::topology::container::grid
{

class RegularGridTopology;

/** A sparse grid topology. Like a sparse FFD building from the bounding box of the object. Starting from a RegularGrid, only valid cells containing matter (ie intersecting the original surface mesh or totally inside the object) are considered.
 * Valid cells are tagged by a Type BOUNDARY or INSIDE
 * WARNING: the corresponding node in the XML file has to be placed BEFORE the MechanicalObject node, in order to excute its init() before the MechanicalObject one in order to be able to give dofs
 */
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_GRID_API SparseGridTopology : public container::constant::MeshTopology
{
public:
    SOFA_CLASS(SparseGridTopology,MeshTopology);

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vec3i, sofa::type::Vec3i);
    typedef sofa::type::fixed_array<type::Vec3,8> CubeCorners;

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(BoundingBox, sofa::type::BoundingBox);

    typedef enum {OUTSIDE,INSIDE,BOUNDARY} Type; ///< each cube has a type depending on its filling ratio
protected:
    SparseGridTopology(bool _isVirtual=false);

    /// Define using the resolution and the spatial size. The resolution corresponds to the number of points if all the cells were filled.
    SparseGridTopology(type::Vec3i numVertices, type::BoundingBox box, bool _isVirtual=false);
public:
    static const float WEIGHT27[8][27];
    static const Index cornerIndicesFromFineToCoarse[8][8];

    void init() override;

    /// building from a mesh file
    virtual void buildAsFinest();

    /// building by condensating a finer sparse grid (used if setFinerSparseGrid has initializated _finerSparseGrid before calling init() )
    virtual void buildFromFiner();

    /// building eventual virtual finer levels (cf d_nbVirtualFinerLevels)
    virtual void buildVirtualFinerLevels();

    typedef std::map<type::Vec3, Index> MapBetweenCornerPositionAndIndice;///< a vertex indice for a given vertex position in space

    /// connexion between several coarsened levels
    typedef std::vector<type::fixed_array<Index,8> > HierarchicalCubeMap; ///< a cube indice -> corresponding 8 child indices on the potential _finerSparseGrid
    HierarchicalCubeMap _hierarchicalCubeMap;
    typedef type::vector<Index> InverseHierarchicalCubeMap; ///< a fine cube indice -> corresponding coarser cube indice
    InverseHierarchicalCubeMap _inverseHierarchicalCubeMap;

    typedef std::map<Index,float> AHierarchicalPointMap;
    typedef type::vector< AHierarchicalPointMap > HierarchicalPointMap; ///< a point indice -> corresponding 27 child indices on the potential _finerSparseGrid with corresponding weight
    HierarchicalPointMap _hierarchicalPointMap;
    typedef type::vector< AHierarchicalPointMap > InverseHierarchicalPointMap; ///< a fine point indice -> corresponding some parent points for interpolation
    InverseHierarchicalPointMap _inverseHierarchicalPointMap;
    typedef type::vector< Index > PointMap;
    PointMap _pointMap; ///< a coarse point indice -> corresponding point in finer level
    PointMap _inversePointMap;  ///< a fine point indice -> corresponding point in coarser level


    enum {UP,DOWN,RIGHT,LEFT,BEFORE,BEHIND,NUM_CONNECTED_NODES};
    typedef type::vector< type::fixed_array<Index,NUM_CONNECTED_NODES> > NodeAdjacency; ///< a node -> its 6 neighboors
    NodeAdjacency _nodeAdjacency;
    typedef type::vector< type::vector<Index> >NodeCubesAdjacency; ///< a node -> its 8 neighboor cells
    NodeCubesAdjacency _nodeCubesAdjacency;
    typedef type::vector< type::vector<Index> >NodeCornersAdjacency; ///< a node -> its 8 corners of neighboor cells
    NodeCornersAdjacency _nodeCornersAdjacency;


    type::vector< SparseGridTopology::SPtr > _virtualFinerLevels; ///< saving the virtual levels (cf nbVirtualFinerLevels)
    int getNbVirtualFinerLevels() const { return d_nbVirtualFinerLevels.getValue();}
    void setNbVirtualFinerLevels(int n) {d_nbVirtualFinerLevels.setValue(n);}


    /// Resolution
    sofa::type::Vec<3, int> getN() const { return d_n.getValue();}
    int getNx() const { return d_n.getValue()[0]; }
    int getNy() const { return d_n.getValue()[1]; }
    int getNz() const { return d_n.getValue()[2]; }

    void setN(type::Vec3i _n) {d_n.setValue(_n);}
    void setNx(int _n) { d_n.setValue(type::Vec3i(_n             , d_n.getValue()[1], d_n.getValue()[2])); }
    void setNy(int _n) { d_n.setValue(type::Vec3i(d_n.getValue()[0], _n             , d_n.getValue()[2])); }
    void setNz(int _n) { d_n.setValue(type::Vec3i(d_n.getValue()[0], d_n.getValue()[1], _n)             ); }

    void setMin(type::Vec3 val) {d_min.setValue(val);}
    void setXmin(SReal val) { d_min.setValue(type::Vec3(val             , d_min.getValue()[1], d_min.getValue()[2])); }
    void setYmin(SReal val) { d_min.setValue(type::Vec3(d_min.getValue()[0], val             , d_min.getValue()[2])); }
    void setZmin(SReal val) { d_min.setValue(type::Vec3(d_min.getValue()[0], d_min.getValue()[1], val)             ); }

    void setMax(type::Vec3 val) {d_max.setValue(val);}
    void setXmax(SReal val) { d_max.setValue(type::Vec3(val             , d_max.getValue()[1], d_max.getValue()[2])); }
    void setYmax(SReal val) { d_max.setValue(type::Vec3(d_max.getValue()[0], val             , d_max.getValue()[2])); }
    void setZmax(SReal val) { d_max.setValue(type::Vec3(d_max.getValue()[0], d_max.getValue()[1], val)             ); }

    type::Vec3 getMin() {return d_min.getValue();}
    SReal getXmin() { return d_min.getValue()[0]; }
    SReal getYmin() { return d_min.getValue()[1]; }
    SReal getZmin() { return d_min.getValue()[2]; }

    type::Vec3 getMax() {return d_max.getValue();}
    SReal getXmax() { return d_max.getValue()[0]; }
    SReal getYmax() { return d_max.getValue()[1]; }
    SReal getZmax() { return d_max.getValue()[2]; }

    bool hasPos()  const override { return true; }

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual Index findCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz);

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual Index findNearestCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz);

    /// return indices of 6 neighboor cubes
    virtual type::fixed_array<Index,6> findneighboorCubes(Index indice );

    /// return the type of the i-th cube
    virtual Type getType(Index i );

    /// return the stiffness coefficient of the i-th cube
    virtual float getStiffnessCoef(Index elementIdx);
    /// return the mass coefficient of the i-th cube
    virtual float getMassCoef(Index elementIdx);

    SparseGridTopology *getFinerSparseGrid() const {return _finerSparseGrid;}
    void setFinerSparseGrid( SparseGridTopology *fsp ) {_finerSparseGrid=fsp;}
    SparseGridTopology *getCoarserSparseGrid() const {return _coarserSparseGrid;}
    void setCoarserSparseGrid( SparseGridTopology *csp ) {_coarserSparseGrid=csp;}

    void updateMesh();

    sofa::core::sptr<RegularGridTopology> _regularGrid; ///< based on a corresponding RegularGrid
    type::vector< Index > _indicesOfRegularCubeInSparseGrid; ///< to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    type::vector< Index > _indicesOfCubeinRegularGrid; ///< to redirect an indice of a cube in the sparse grid to its indice in the regular grid

    type::Vec3 getPointPos(Index i ) { return type::Vec3(d_seqPoints.getValue()[i][0], d_seqPoints.getValue()[i][1], d_seqPoints.getValue()[i][2] ); }

    void getMesh( sofa::helper::io::Mesh &m);

    void setDimVoxels( int a, int b, int c) { d_dataResolution.setValue(type::Vec3i(a, b, c));}
    void setSizeVoxel( SReal a, SReal b, SReal c) { d_voxelSize.setValue(type::Vec3(a, b, c));}

    bool getVoxel(unsigned int x, unsigned int y, unsigned int z)
    {
        return getVoxel(d_dataResolution.getValue()[0] * d_dataResolution.getValue()[1] * z + d_dataResolution.getValue()[0] * y + x);
    }

    bool getVoxel(unsigned int index) const
    {
        return d_dataVoxels.getValue()[index] == 1;
    }
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<bool> _fillWeighted;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<bool> bOnlyInsideCells;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data< type::vector< unsigned char > >     dataVoxels;


    Data< type::vector< unsigned char > >     d_dataVoxels;

    Data<bool> d_fillWeighted; ///< is quantity of matter inside a cell taken into account?

    Data<bool> d_bOnlyInsideCells; ///< Select only inside cells (exclude boundary cells)


protected:


    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data< sofa::type::Vec< 3, int > >  n;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<type::Vec3> _min;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<type::Vec3> _max;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<SReal> _cellWidth;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<int> _nbVirtualFinerLevels;


    bool isVirtual;
    /// cutting number in all directions
    Data< sofa::type::Vec< 3, int > > d_n;
    Data< type::Vec3 > d_min; ///< Min
    Data< type::Vec3 > d_max; ///< Max
    Data< SReal > d_cellWidth; ///< if > 0 : dimension of each cell in the created grid
    Data< int > d_nbVirtualFinerLevels; ///< create virtual (not in the animation tree) finer sparse grids in order to dispose of finest information (usefull to compute better mechanical properties for example)

public:

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<type::Vec3i> dataResolution;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<type::Vec3> voxelSize;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<unsigned int> marchingCubeStep;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data<unsigned int> convolutionSize;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_GRID()
    Data< type::vector< type::vector<Index> > > facets;

    Data< type::Vec3i >			d_dataResolution; ///< Dimension of the voxel File
    Data< type::Vec3 >         d_voxelSize; ///< Dimension of one voxel
    Data< unsigned int >    d_marchingCubeStep; ///< Step of the Marching Cube algorithm
    Data< unsigned int >    d_convolutionSize; ///< Dimension of the convolution kernel to smooth the voxels. 0 if no smoothing is required.

    Data< type::vector< type::vector<Index> > >d_facets; ///< Input mesh facets

    /** Create the data structure based on resolution, size and filling.
          \param numPoints  Number of points in the x,y,and z directions
          \param box  Volume occupied by the grid
          \param filling Voxel filling: true if the cell is defined, false if the cell is empty. Voxel order is: for(each z){ for(each y){ for(each x) }}}
          */
    void buildFromData( type::Vec3i numPoints, type::BoundingBox box, const type::vector<bool>& filling );

protected:
    virtual void updateEdges();
    virtual void updateQuads();

    sofa::helper::MarchingCubeUtility                 marchingCubes;
    bool                                _usingMC;

    type::vector<Type> _types; ///< BOUNDARY or FULL filled cells

    type::vector< float > _stiffnessCoefs; ///< a stiffness coefficient per hexa (BOUNDARY=.5, FULL=1)
    type::vector< float > _massCoefs; ///< a stiffness coefficient per hexa (BOUNDARY=.5, FULL=1)

    /// start from a seed cell (i,j,k) the OUTSIDE filling is propagated to neighboor cells until meet a BOUNDARY cell (this function is called from all border cells of the RegularGrid)
    void launchPropagationFromSeed(const type::Vec3i& point,
            sofa::core::sptr<RegularGridTopology> regularGrid,
            type::vector<Type>& regularGrdidTypes,
            type::vector<bool>& alreadyTested,
            std::stack<type::Vec3i>& seed) const;

    void propagateFrom(  const type::Vec3i& point,
            sofa::core::sptr<RegularGridTopology> regularGrid,
            type::vector<Type>& regularGridTypes,
            type::vector<bool>& alreadyTested,
            std::stack< sofa::type::Vec<3,int> > &seed) const;

    void computeBoundingBox(const type::vector<type::Vec3>& vertices,
            SReal& xmin, SReal& xmax,
            SReal& ymin, SReal& ymax,
            SReal& zmin, SReal& zmax) const;

    void voxelizeTriangleMesh(helper::io::Mesh* mesh,
            sofa::core::sptr<RegularGridTopology> regularGrid,
            type::vector<Type>& regularGridTypes) const;

    void buildFromTriangleMesh(sofa::helper::io::Mesh* mesh);

    void buildFromRegularGridTypes(sofa::core::sptr<RegularGridTopology> regularGrid, const type::vector<Type>& regularGridTypes);



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
    void buildFromVoxelLoader(sofa::core::loader::VoxelLoader * loader);

    template< class T>
    void constructCollisionModels(const sofa::type::vector< sofa::core::topology::BaseMeshTopology * > &list_mesh,
            const type::vector< Data< type::vector< sofa::type::Vec<3,T> > >* >            &list_X) ;

    SparseGridTopology* _finerSparseGrid; ///< an eventual finer sparse grid that can be used to built this coarser sparse grid
    SparseGridTopology* _coarserSparseGrid; ///< an eventual coarser sparse grid

    void setVoxel(int index, unsigned char value)
    {
        if (value)
        {
            (*d_dataVoxels.beginEdit())[index] = 1;
        }
        else
        {
            (*d_dataVoxels.beginEdit())[index] = 0;
        }
        d_dataVoxels.beginEdit();
    };



    bool _alreadyInit;

public :


    const SeqHexahedra& getHexahedra() override
    {
        if( !_alreadyInit ) init();
        return container::constant::MeshTopology::getHexahedra();
    }

    Size getNbPoints() const override
    {
        if( !_alreadyInit ) const_cast<SparseGridTopology*>(this)->init();
        return container::constant::MeshTopology::getNbPoints();
    }

    /// TODO 2018-07-23 epernod: check why this method is override to return the same result as parent class.
    Size getNbHexahedra() override { return Size(this->getHexahedra().size());}
};

} //namespace sofa::component::topology::container::grid
