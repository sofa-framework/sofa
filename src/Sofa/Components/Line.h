#ifndef SOFA_COMPONENTS_LINE_H
#define SOFA_COMPONENTS_LINE_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Common/Vec3Types.h"

namespace Sofa
{

namespace Components
{

class LineModel;

using namespace Common;

class Line : public Abstract::CollisionElement
{
protected:
    int index,i1,i2;
    LineModel * model;
    Core::MechanicalModel<Vec3Types>* mmodel;
    Vector3 minBBox, maxBBox;

    void recalcBBox();
    void recalcContinuousBBox(double dt);
    friend class LineModel;

public:

    const Vector3& p1() const { return (*mmodel->getX())[i1]; }
    const Vector3& p2() const { return (*mmodel->getX())[i2]; }

    const Vector3& v1() const { return (*mmodel->getV())[i1]; }
    const Vector3& v2() const { return (*mmodel->getV())[i2]; }

    int getIndex() const { return index; }

    Line(int index, int i1, int i2, LineModel* model);

    void getBBox(double* minVect, double* maxVect);

    /// Test if collisions with another element should be tested.
    /// Reject any self-collisions, including collision with other collision models attached to the same node
    bool canCollideWith(CollisionElement* elem) {return getCollisionModel()->getContext() != elem->getCollisionModel()->getContext();}

    // for debugging only
    void draw (void);

    friend std::ostream& operator<< (std::ostream& os, const Line &tri);

    Abstract::CollisionModel* getCollisionModel();
};

} // namespace Components

} // namespace Sofa

#endif
