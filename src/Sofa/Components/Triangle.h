#ifndef SOFA_COMPONENTS__TRIANGLE_H
#define SOFA_COMPONENTS__TRIANGLE_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Common/Vec3Types.h"

namespace Sofa
{

namespace Components
{

class TriangleModel;

using namespace Common;

class Triangle : public Abstract::CollisionElement
{

protected:
    int index,i1,i2,i3;
    Vector3 normal;
    TriangleModel * model;
    Core::MechanicalModel<Vec3Types>* mmodel;
    Vector3 minBBox, maxBBox;

    void recalcBBox();
    void recalcContinuousBBox(double dt);
    friend class TriangleModel;

public:

    const Vector3& p1() const { return (*mmodel->getX())[i1]; }
    const Vector3& p2() const { return (*mmodel->getX())[i2]; }
    const Vector3& p3() const { return (*mmodel->getX())[i3]; }

    const Vector3& v1() const { return (*mmodel->getV())[i1]; }
    const Vector3& v2() const { return (*mmodel->getV())[i2]; }
    const Vector3& v3() const { return (*mmodel->getV())[i3]; }

    const Vector3& n() const { return normal; }

    int getIndex() const { return index; }

    Triangle(int index, int i1, int i2, int i3, TriangleModel* model);

    void getBBox(double* minVect, double* maxVect);

    /// Test if collisions with another element should be tested.
    /// Reject any self-collisions, including collision with other collision models attached to the same node
    bool canCollideWith(CollisionElement* elem) {return getCollisionModel()->getContext() != elem->getCollisionModel()->getContext();}

    // for debugging only
    void draw (void);

    friend std::ostream& operator<< (std::ostream& os, const Triangle &tri);

    Abstract::CollisionModel* getCollisionModel();
};

} // namespace Components

} // namespace Sofa

#endif
