#ifndef SOFA_COMPONENT_COLLISION_VOXELGRID_H
#define SOFA_COMPONENT_COLLISION_VOXELGRID_H

#include <sofa/core/componentmodel/collision/BroadPhaseDetection.h>
#include <sofa/component/collision/NarrowPhaseDetection.h>
#include <sofa/core/VisualModel.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/simulation/tree/GNode.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class VoxelGrid;

class GridCell
{
private:
    //Vector3 minVect, maxVect; // minx, miny, minz; maxx, maxy, maxz
    std::vector<core::CollisionElementIterator> collisElems; // elements wich are added at each iteration
    std::vector<core::CollisionElementIterator> collisElemsImmobile[2]; // elements which are added only once

    Vector3 minCell, maxCell;
    int timeStamp;
public:
    // Adding a sphere in a cell of the voxel grid.
    // When adding a sphere, we test if there are collision with the sphere in the cell
    // then we add it in the vector sphere
    void add(VoxelGrid* grid, core::CollisionElementIterator collElem, std::vector<core::CollisionElementIterator> &vectCollis, int phase);
    void eraseAll(int timeStampMethod);
    GridCell();

    void draw(int timeStampMethod);
    void setMinMax(const Vector3 &minimum, const Vector3& maximum);
};

// inherit of VisualModel for debugging, then we can see the voxel grid
class VoxelGrid : public BroadPhaseDetection, public NarrowPhaseDetection, public core::VisualModel
{
private:
    Vector3 nbSubDiv;
    GridCell ***grid;
    bool bDraw;
    Vector3 minVect, maxVect, step;
    void posToIdx (const Vector3& pos, Vector3 &indices);
    simulation::tree::GNode* timeLogger;
    simulation::tree::GNode::ctime_t timeInter;
    friend class GridCell;
public:
    VoxelGrid (Vector3 minVect = Vector3(-20.0, -20.0, -20.0), Vector3 maxVect = Vector3(-20.0, -20.0, -20.0), Vector3 nbSubdivision = Vector3(5.0, 5.0, 5.0), bool draw=false)
    {
        createVoxelGrid (minVect, maxVect, nbSubdivision);
        timeStamp = 0;
        bDraw = draw;
        timeLogger = NULL;
        timeInter = 0;
    }

    ~VoxelGrid () {}

    // Create a voxel grid define by minx, miny, minz, maxx, maxy, maxz and the number of subdivision on x, y, z
    void createVoxelGrid (const Vector3 &min, const Vector3 &max, const Vector3 &nbSubdivision);

    /* for debugging, VisualModel */
    void draw();
    void initTextures() { }
    void update() { }

    void add(core::CollisionModel *cm, int phase);

    void addCollisionModel(core::CollisionModel *cm);
    void addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair);
    void add (core::CollisionModel *cm);

    void clearBroadPhase()
    {
        BroadPhaseDetection::clearBroadPhase();
        timeStamp++;
    }
    void clearNarrowPhase()
    {
        NarrowPhaseDetection::clearNarrowPhase();
        timeStamp++;
    }

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
