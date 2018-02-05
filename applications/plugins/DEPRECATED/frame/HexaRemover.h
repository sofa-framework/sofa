/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_FRAME_HEXA_REMOVER_H
#define SOFA_FRAME_HEXA_REMOVER_H

#include "Blending.h"
#include "MeshGenerator.h"
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

/** Handle topology changes (not used currently)
  */
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

    typedef typename engine::MeshGenerator<DataTypes> MeshGen;
    typedef typename MeshGen::GridMat GridMat;
    typedef typename MeshGen::GCoord GCoord;
    typedef typename MeshGen::SCoord SCoord;

    typedef SampleData<DataTypes,false> SData; // = collision FrameBlendingMapping
    typedef std::multimap< double, std::multimap<double, std::pair< double, double> > > RasterizedVol; // map< x, map< y, pair< zMin, zMax> > >
    typedef typename defaulttype::Vec3f FPoint;
    typedef typename helper::fixed_array<FPoint, 2> BBox;

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
    void findTheCollidedVoxels ( unsigned int triangleID);
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
    inline void buildCollisionVolumes();
    inline void propagateFrom ( const unsigned int hexa);
    inline void addVolume( RasterizedVol& rasterizedVolume, double x, double y, double zMin, double zMax);
    inline void getNeighbors ( const unsigned int hexaID, helper::set<unsigned int>& neighbors ) const;
    inline bool isCrossingCube( const Vector3& point, const float& radius ) const;
    inline bool isPointInside( const Vector3& point ) const;

private:
    Rasterizer* rasterizer;
    MeshGen* meshGen; // TODO delete this one and replace by those in contactInfos
    SData* sData;
    TriangleSetGeometryAlgorithms<DataTypes>* triGeoAlgo;

    sofa::helper::vector<MTopology*> cuttingModels;

    RasterizedVol collisionVolumes[3]; // for each axis
    std::set<unsigned int> trianglesToParse;

    // Debug purpose
    std::map<unsigned int, Vec3d> voxelMappedCoord; // coords up to date
    sofa::helper::set<Coord> parsedHexasCoords;
    sofa::helper::set<Coord> removedHexasCoords;
    sofa::helper::set<Coord> collisionTrianglesCoords;
    Data<bool> showElements;
    Data<bool> showVolumes;
};

}

}

}

#endif
