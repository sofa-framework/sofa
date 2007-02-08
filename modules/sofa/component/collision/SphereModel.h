#ifndef SOFA_COMPONENT_COLLISION_SPHEREMODEL_H
#define SOFA_COMPONENT_COLLISION_SPHEREMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class SphereModel;

class Sphere : public core::TCollisionElementIterator<SphereModel>
{
public:
    Sphere(SphereModel* model, int index);

    explicit Sphere(core::CollisionElementIterator& i);

    const Vector3& center() const;

    const Vector3& v() const;

    double r() const;
};

class SphereModel : public component::MechanicalObject<Vec3Types>, public core::CollisionModel, public core::VisualModel
{
protected:
    std::vector<double> radius;

    double defaultRadius;

    class Loader;

    bool static_;
public:
    typedef Vec3Types DataTypes;
    typedef Sphere Element;
    friend class Sphere;

    SphereModel(double radius = 1.0);

    int addSphere(const Vector3& pos, double radius);
    void setSphere(int index, const Vector3& pos, double radius);

    bool load(const char* filename);

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
};

inline Sphere::Sphere(SphereModel* model, int index)
    : core::TCollisionElementIterator<SphereModel>(model, index)
{}

inline Sphere::Sphere(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<SphereModel>(static_cast<SphereModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Vector3& Sphere::center() const
{
    return (*model->getX())[index];
}

inline const Vector3& Sphere::v() const
{
    return (*model->getV())[index];
}

inline double Sphere::r() const
{
    return model->radius[index];
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
