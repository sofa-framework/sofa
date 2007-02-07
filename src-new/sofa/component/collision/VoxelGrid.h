#ifndef SOFA_COMPONENTS_VOXELGRID_H
#define SOFA_COMPONENTS_VOXELGRID_H

#include "Collision/BroadPhaseDetection.h"
#include "Collision/NarrowPhaseDetection.h"
#include "Sofa-old/Abstract/VisualModel.h"
#include "Common/Vec.h"
#include "Graph/GNode.h"

#include <vector>

namespace Sofa
{

namespace Components
{

using namespace Common;

class VoxelGrid;

class GridCell
{
private:
    //Vector3 minVect, maxVect; // minx, miny, minz; maxx, maxy, maxz
    std::vector<Abstract::CollisionElementIterator> collisElems; // elements wich are added at each iteration
    std::vector<Abstract::CollisionElementIterator> collisElemsImmobile[2]; // elements which are added only once

    Vector3 minCell, maxCell;
    int timeStamp;
public:
    // Adding a sphere in a cell of the voxel grid.
    // When adding a sphere, we test if there are collision with the sphere in the cell
    // then we add it in the vector sphere
    void add(VoxelGrid* grid, Abstract::CollisionElementIterator collElem, std::vector<Abstract::CollisionElementIterator> &vectCollis, int phase);
    void eraseAll(int timeStampMethod);
    GridCell();

    void draw(int timeStampMethod);
    void setMinMax(const Vector3 &minimum, const Vector3& maximum);
};

// inherit of VisualModel for debugging, then we can see the voxel grid
class VoxelGrid : public Collision::BroadPhaseDetection, public Collision::NarrowPhaseDetection, public Abstract::VisualModel
{
private:
    Vector3 nbSubDiv;
    GridCell ***grid;
    bool bDraw;
    Vector3 minVect, maxVect, step;
    void posToIdx (const Vector3& pos, Vector3 &indices);
    Graph::GNode* timeLogger;
    Graph::GNode::ctime_t timeInter;
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

    void add(Abstract::CollisionModel *cm, int phase);

    void addCollisionModel(Abstract::CollisionModel *cm);
    void addCollisionPair(const std::pair<Abstract::CollisionModel*, Abstract::CollisionModel*>& cmPair);
    void add (Abstract::CollisionModel *cm);

    void clearBroadPhase()
    {
        Collision::BroadPhaseDetection::clearBroadPhase();
        timeStamp++;
    }
    void clearNarrowPhase()
    {
        Collision::NarrowPhaseDetection::clearNarrowPhase();
        timeStamp++;
    }

};

} // namespace Components

} // namespace Sofa

#endif
