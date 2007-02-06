#ifndef SOFA_COMPONENTS_SPHEREMODEL_H
#define SOFA_COMPONENTS_SPHEREMODEL_H

#include "Sofa-old/Abstract/CollisionModel.h"
#include "Sofa-old/Abstract/VisualModel.h"
#include "Sofa-old/Core/MechanicalObject.h"
#include "Common/Vec3Types.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class SphereModel;

class Sphere : public Abstract::TCollisionElementIterator<SphereModel>
{
public:
    Sphere(SphereModel* model, int index);

    explicit Sphere(Abstract::CollisionElementIterator& i);

    const Vector3& center() const;

    const Vector3& v() const;

    double r() const;
};

class SphereModel : public Core::MechanicalObject<Vec3Types>, public Abstract::CollisionModel, public Abstract::VisualModel
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
    : Abstract::TCollisionElementIterator<SphereModel>(model, index)
{}

inline Sphere::Sphere(Abstract::CollisionElementIterator& i)
    : Abstract::TCollisionElementIterator<SphereModel>(static_cast<SphereModel*>(i.getCollisionModel()), i.getIndex())
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

} // namespace Components

} // namespace Sofa

#endif
