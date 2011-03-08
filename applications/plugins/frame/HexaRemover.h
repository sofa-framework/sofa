#ifndef SOFA_FRAME_HEXA_REMOVER_H
#define SOFA_FRAME_HEXA_REMOVER_H

#include <sofa/component/topology/DynamicSparseGridGeometryAlgorithms.h>
#include <sofa/component/topology/DynamicSparseGridTopologyContainer.h>
#include "FrameBlendingMapping.h"
#include "MeshGenerater.h"
#include <sofa/gpu/cuda/CudaRasterizer.h>
#include <map>

#include <sofa/helper/set.h>

//#define VERBOSE

namespace sofa
{

namespace component
{

namespace topology
{


template <class DataTypes>
class HexaRemover: public core::objectmodel::BaseObject
{
    typedef sofa::core::topology::BaseMeshTopology MTopology;
    typedef std::pair< Vector3, Vector3 > BoundingBox;


    typedef MechanicalState<DataTypes> MState;
    typedef gpu::cuda::CudaRasterizer<DataTypes> Rasterizer;
    typedef typename Rasterizer::CellCountLayer CellCountLayer;
    typedef typename Rasterizer::Cell Cell;
    typedef typename Rasterizer::LDI LDI;


    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    typedef typename engine::MeshGenerater<DataTypes> MeshGen;
    typedef typename MeshGen::GridMat GridMat;
    typedef typename MeshGen::GCoord GCoord;
    typedef typename MeshGen::SCoord SCoord;

public:
    SOFA_CLASS(HexaRemover,core::objectmodel::BaseObject);

public:
    typedef sofa::defaulttype::Vec<3, int> Vec3i;

    class ContactInfos
    {
    public:
        sofa::helper::set<unsigned int>& getHexaToRemove() {return hexaToRemove;};
        sofa::helper::set<unsigned int>& getParsedHexasMap() {return parsedHexasMap;};
        sofa::helper::set<unsigned int>& getCheckedOnce() {return checkedOnce;};
        void clear() { hexaToRemove.clear(); parsedHexasMap.clear(); checkedOnce.clear();};

        sofa::helper::set<unsigned int> hexaToRemove; // Hexa to remove for each H2TtopoMapping
        sofa::helper::set<unsigned int> parsedHexasMap;
        sofa::helper::set<unsigned int> checkedOnce;
        Vector3 center;
    };

    HexaRemover();
    ~HexaRemover();

    void init ();
    inline bool isTheModelInteresting ( MTopology* model ) const;
    void findTheCollidedVoxels ( unsigned int triangleID, const Vector3& minBBVolume, const Vector3& maxBBVolume );
    bool removeVoxels();
    void clear();
    virtual void draw();

    /// Handle an event
    virtual void handleEvent(core::objectmodel::Event* ev);

    // Debug purpose
    void clearDebugVectors();
    void drawParsedHexas();
    void drawRemovedHexas();
    void drawCollisionTriangles();
    void drawCollisionVolumes();
    void drawBoundingBox( const BoundingBox& bbox);

    sofa::helper::vector<Vector3> removedVoxelsCenters;



    std::string getTemplateName() const
    {
        return templateName(this);
    }
    static std::string templateName(const HexaRemover<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    std::map< const MeshGen*, ContactInfos> contactInfos;

    inline float squared( const float& number) const {return number*number;};
    inline void detectVoxels();
    inline void propagateFrom ( const unsigned int hexa, const Vector3& minBBVolume, const Vector3& maxBBVolume );
    inline void getNeighbors ( const unsigned int hexaID, helper::set<unsigned int>& neighbors ) const;
    inline bool isCrossingAABB ( const Vector3& min1, const Vector3& max1, const Vector3& min2, const Vector3& max2 ) const;
    inline bool isInsideAABB ( const Vector3& min1, const Vector3& max1, const Vector3& point ) const;
    inline bool isCrossingSphere(const Vector3& min1, const Vector3& max1, const Vector3& point, const float& radius ) const;
private:
    Rasterizer* rasterizer;
    MeshGen* meshGen; // TODO delete this one and replace by those in contactInfos
    TriangleSetGeometryAlgorithms<DataTypes>* triGeoAlgo;

    sofa::helper::vector<MTopology*> cuttingModels;

    // Debug purpose
    sofa::helper::set<Coord> parsedHexasCoords;
    sofa::helper::set<Coord> removedHexasCoords;
    sofa::helper::set<Coord> collisionTrianglesCoords;
    std::set<BoundingBox> collisionVolumesCoords[3];
    Data<bool> showElements;
    Data<bool> showVolumes;
};

}

}

}

#endif
