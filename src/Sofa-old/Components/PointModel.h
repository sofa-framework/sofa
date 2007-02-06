#ifndef SOFA_COMPONENTS_POINTMODEL_H
#define SOFA_COMPONENTS_POINTMODEL_H

#include "Sofa-old/Abstract/CollisionModel.h"
#include "Sofa-old/Abstract/VisualModel.h"
#include "Sofa-old/Core/MechanicalObject.h"
#include "MeshTopology.h"
#include "Common/Vec3Types.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class PointModel;

class Point : public Abstract::TCollisionElementIterator<PointModel>
{
public:
    Point(PointModel* model, int index);

    explicit Point(Abstract::CollisionElementIterator& i);

    const Vector3& p() const;
    const Vector3& v() const;
};

class PointModel : public Abstract::CollisionModel, public Abstract::VisualModel
{
protected:
    bool static_;
public:
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Point Element;
    friend class Point;

    PointModel();

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    bool isStatic() { return static_; }
    void setStatic(bool val=true) { static_ = val; }

    void draw(int index);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }


    Core::MechanicalModel<Vec3Types>* getMechanicalModel() { return mmodel; }

    virtual const char* getTypeName() const { return "Point"; }

protected:

    Core::MechanicalModel<Vec3Types>* mmodel;
};

inline Point::Point(PointModel* model, int index)
    : Abstract::TCollisionElementIterator<PointModel>(model, index)
{}

inline Point::Point(Abstract::CollisionElementIterator& i)
    : Abstract::TCollisionElementIterator<PointModel>(static_cast<PointModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Vector3& Point::p() const { return (*model->mmodel->getX())[index]; }

inline const Vector3& Point::v() const { return (*model->mmodel->getV())[index]; }

} // namespace Components

} // namespace Sofa

#endif
