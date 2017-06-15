/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_FRAME_MESHGENERATOR_H
#define SOFA_FRAME_MESHGENERATOR_H

#include <sofa/core/DataEngine.h>
#include <sofa/helper/MarchingCubeUtility.h>
#include <sofa/helper/map.h>
#include <sofa/helper/set.h>

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.h>
#include <SofaOpenglVisual/OglAttribute.h>
#include <SofaOpenglVisual/OglVariable.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include "GridMaterial.h"
#include "Blending.h"
#include <SofaBaseTopology/TopologyData.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology;
using namespace sofa::component::visualmodel;
using namespace sofa::core::topology;
using namespace sofa::core;
using namespace cimg_library;

using sofa::helper::MarchingCubeUtility;
using sofa::helper::set;
using std::map;
using sofa::component::material::GridMaterial;
using sofa::component::material::Material3d;

/**
* This class, called MeshGenerator, generate a triangular mesh from a voxel array :
*
* Considering as INPUT:
* - a voxel array (retreived from GridMaterial)
* - a region of interest
* - an isoValue
* - an optional set of seeds.
* this engine maps it to the OUTPUT:
* - triangular topology
* with the marching cubes algorithm
*
* MeshGenerator class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
*
*/

template <class DataTypes>
class MeshGenerator : public core::DataEngine
{
public:
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef defaulttype::Vec<3, int> Vec3i;
    typedef defaulttype::Vec<6, int> Vec6i;
    typedef unsigned int HexaIDInRegularGrid;

    typedef typename core::behavior::MechanicalState<DataTypes> MState;
    typedef typename sofa::component::container::MechanicalObject<DataTypes> MObject;

    typedef GridMaterial<Material3d> GridMat;
    typedef GridMat::voxelType VoxelType;

    typedef GridMat::GCoord GCoord;
    typedef GridMat::SCoord SCoord;

    typedef unsigned int PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;

    SOFA_CLASS(MeshGenerator, core::DataEngine);

    template <class T>
    class ElementSet: public std::set<T>
    {
    public:
        inline friend std::ostream& operator<< ( std::ostream& o, const ElementSet<T>& e )
        {
            sofa::helper::set<T> &t= ( sofa::helper::set<T>& ) e;
            o << "[ " << t << " ]";
            return o;
        }
        inline friend std::istream& operator>> ( std::istream& i, ElementSet<T>& e )
        {
            e.clear();
            std::string sep; i>>sep;
            if ( sep != "[" ) return i;
            T value;
            while ( i >> sep )
            {
                if ( sep == "]" ) break;
                value = ( T ) atoi ( sep.c_str() );
                e.insert ( value );
            }
            return i;
        }
    };

    /** \brief Constructor.
        *
        * Does nothing.
        */
    MeshGenerator();

    /** \brief Destructor.
    *
    * Does nothing.
    */
    virtual ~MeshGenerator();

    /** \brief Initializes the target BaseTopology from the source BaseTopology.
    */
    virtual void init();

    /** \brief Translates the TopologyChange objects from the source to the target.
    *
    * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
    * reflect the effects of the first topology changes on the second topology.
    *
    */
    //virtual void updateTopologicalMappingTopDown();

    virtual void removeVoxels ( const sofa::helper::vector<unsigned int>& removedHexahedraID ); // This method must be called to init the topological changes process

    virtual void update();

    /// return true if the output topology subdivide the input one. (the topology uses the Loc2GlobVec/Glob2LocMap/In2OutMap structs and share the same DOFs)
    virtual bool isTheOutputTopologySubdividingTheInputOne() { return false;}

    /// Return the hexahedra 'fromIndices' corresponding to the triangle index 'toIndex'
    virtual void getFromIndex ( vector<unsigned int>& fromIndices, const unsigned int toIndex ) const;

    /// Return the triangle 'toIndices' corresponding to the hexa index 'fromIndex'
    virtual void getToIndex ( vector<unsigned int>& toIndices, const unsigned int fromIndex ) const;

    virtual void draw();

    void handleEvent ( core::objectmodel::Event * );

    virtual TriangleSetTopologyContainer* getTo() {return _to_topo;};

protected:
    /**** Input ****/
    Data< Vec6i > roi;
    Data< float > mIsoValue;
    Data< vector<Vec<3, int> > > mCubeSeeds;


    /**** Utils ****/
    MarchingCubeUtility marchingCubes;

    // Voxel values and types
    vector<VoxelType> valueData;
    vector<VoxelType> segmentIDData;

    // Connexion maps
    Data< vector< vector< HexaIDInRegularGrid > > > triangleIndexInRegularGrid;
    Data< map< HexaIDInRegularGrid, ElementSet< BaseMeshTopology::TriangleID > > > triangleIDInRegularGrid2IndexInTopo;

    TriangleSetTopologyContainer* _to_topo;
    TriangleSetGeometryAlgorithms<DataTypes>* _to_geomAlgo;
    TriangleSetTopologyModifier *_to_tstm;
    MState* _to_DOFs;

    Data<unsigned int> smoothIterations;
    PointData<sofa::helper::vector<typename DataTypes::Coord> > smoothedMesh0;
    OglFloatAttribute* segmentationID; // segmentation ID for each vertex of the mesh
    OglFloat3Attribute* restPosition; // Rest position for each vertex of the mesh
    OglFloat3Attribute* restNormal; // Rest normal for each vertex of the mesh

    Data<bool> showHexas2Tri;
    Data<bool> showTri2Hexas;
    Data<bool> showRegularGridIndices;
    Data<double> showTextScaleFactor;

public:
    GridMat* gridMat;
    Data< SCoord > voxelSize;
    Data< SCoord > voxelOrigin;
    Data< GCoord > voxelDimension;


    void getHexaCoord( Coord& coord, const unsigned int hexaID) const;

protected:
    /** \brief init the voxels from the GridMaterial.
    */
    void initVoxels();

    /** \brief init the OglAttributes components in the scene graph depending on the marching cube triangulation.
    */
    void initOglAttributes();

    /** \brief remove triangular mesh corresponding to the removed hexahedra.
    *
    * @param removedHexaID ID of the removed hexahedra
    */
    void removeOldMesh ( const sofa::helper::vector<unsigned int>& removedHexahedraID );

    /** \brief localy remesh
    *
    * @param removedHexaID ID in the regular grid of the removed hexahedra.
    */
    void localyRemesh ( const sofa::helper::vector<BaseMeshTopology::HexaID>& removedHexaID );

    /** \brief Complete an existing mesh by localy remeshing.
    *
    * @param triangleIndexInRegularGrid All the triangle indices in the marching cube regular grid.
    * @param vertices Vertices of the mesh. Given vertices are used to create the new mesh part.
    * @param triangles Triangles of the mesh.
    * @param removedTrianglesCoords 3D coords from where the mCube algorithm is propagated.
    * @param bordersPosition 3D coords until where the mCube algorithm is propagated.
    */
    void computeNewMesh ( vector< Vector3 >& vertices, vector< unsigned int>& triangles, const sofa::helper::vector<BaseMeshTopology::HexaID> &removedHexaID );

    /** \brief Add the triangle to the topology base on the old vertices and the new ones.
    *
    * @param vertices The new vertices to add
    * @param triangles The new triangles to add
    */
    void addNewEltsInTopology ( const sofa::helper::vector< Vector3 >& vertices, const sofa::helper::vector< unsigned int>& triangles );

    /** \brief Smooth the given mesh
    *
    * @param oldVertSize The old vertices number before recomputing the new patch
    * @param oldTriSize The old triangles number before recomputing the new patch
    */
    void smoothMesh ( const unsigned int oldVertSize = 0, const unsigned int oldTriSize = 0);

    /** \brief update the OGL Attributes components (in the scene graph) depending on the marching cube triangulation.
    *
    * @param oldVertSize number of vertices before remeshing
    * @param vertices list of the all vertices
    * @param triangles list of all the triangles
    */
    void updateOglAttributes ( const unsigned int oldVertSize = 0, const unsigned int oldTriSize = 0);

    void updateTrianglesInfos( const vector< vector<unsigned int> >& triangleIndexInRegularGrid);

    void dispMaps() const;
};

} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_FRAME_MESHGENERATOR_H
