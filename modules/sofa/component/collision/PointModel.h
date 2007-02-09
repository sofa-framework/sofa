#ifndef SOFA_COMPONENT_COLLISION_POINTMODEL_H
#define SOFA_COMPONENT_COLLISION_POINTMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class PointModel;

class Point : public core::TCollisionElementIterator<PointModel>
{
public:
    Point(PointModel* model, int index);

    explicit Point(core::CollisionElementIterator& i);

    const Vector3& p() const;
    const Vector3& v() const;
};

class PointModel : public core::CollisionModel, public core::VisualModel
{
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

    void draw(int index);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }


    core::componentmodel::behavior::MechanicalState<Vec3Types>* getMechanicalState() { return mstate; }

    //virtual const char* getTypeName() const { return "Point"; }

protected:

    core::componentmodel::behavior::MechanicalState<Vec3Types>* mstate;
};

inline Point::Point(PointModel* model, int index)
    : core::TCollisionElementIterator<PointModel>(model, index)
{}

inline Point::Point(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<PointModel>(static_cast<PointModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Vector3& Point::p() const { return (*model->mstate->getX())[index]; }

inline const Vector3& Point::v() const { return (*model->mstate->getV())[index]; }

} // namespace collision

} // namespace component

} // namespace sofa

#endif
