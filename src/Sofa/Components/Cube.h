#ifndef SOFA_COMPONENTS_CUBE_H
#define SOFA_COMPONENTS_CUBE_H

#include "Sofa/Abstract/CollisionElement.h"
#include "Common/Vec.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class CubeModel;

class Cube: public Abstract::CollisionElement
{
protected:
    CubeModel* model;
    int index;
public:
    Cube(CubeModel *cubeModel);
    Cube(const Cube &cube, CubeModel *cubeModel);
    Cube(const Cube &cube);
    Cube (int index, CubeModel *cubeModel);

    const Vector3& minVect() const;
    const Vector3& maxVect() const;

    void getBBox(Vector3 &minBBox, Vector3 &maxBBox)
    {
        minBBox = minVect();
        maxBBox = maxVect();
    }

    void getBBox(double* minV, double* maxV)
    {
        minV[0] = minVect()[0];
        minV[1] = minVect()[1];
        minV[2] = minVect()[2];
        maxV[0] = maxVect()[0];
        maxV[1] = maxVect()[1];
        maxV[2] = maxVect()[2];
    }

    Abstract::CollisionModel* getCollisionModel();

    void draw();

    friend class CubeModel;
};

} // namespace Components

} // namespace Sofa

#endif
