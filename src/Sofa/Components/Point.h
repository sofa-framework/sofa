#ifndef SOFA_COMPONENTS_POINT_H
#define SOFA_COMPONENTS_POINT_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Common/Vec3Types.h"

namespace Sofa
{

namespace Components
{

class PointModel;

using namespace Common;

class Point : public Abstract::CollisionElement
{
protected:
    int index;
    PointModel * model;
    Core::MechanicalModel<Vec3Types>* mmodel;
    //Vector3 minBBox, maxBBox;

    void recalcBBox() {}
    void recalcContinuousBBox(double /*dt*/) {}
    friend class PointModel;

public:

    const Vector3& p() const { return (*mmodel->getX())[index]; }

    const Vector3& v() const { return (*mmodel->getV())[index]; }

    int getIndex() const { return index; }

    Point(int index, PointModel* model);

    void getBBox(double* minVect, double* maxVect);
    void getContinuousBBox(double* minVect, double* maxVect, double dt);

    /// Test if collisions with another element should be tested.
    /// Reject any self-collisions, including collision with other collision models attached to the same node
    bool canCollideWith(CollisionElement* elem) {return getCollisionModel()->getContext() != elem->getCollisionModel()->getContext();}

    // for debugging only
    void draw (void);

    friend std::ostream& operator<< (std::ostream& os, const Point &tri);

    Abstract::CollisionModel* getCollisionModel();
};

} // namespace Components

} // namespace Sofa

#endif
