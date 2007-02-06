#ifndef SOFA_COMPONENTS_RAYMODEL_H
#define SOFA_COMPONENTS_RAYMODEL_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/Vec3Types.h"

#include <set>

namespace Sofa
{

namespace Components
{

using namespace Common;

class RayModel;

class Ray : public Abstract::TCollisionElementIterator<RayModel>
{
public:
    Ray(RayModel* model, int index);

    explicit Ray(Abstract::CollisionElementIterator& i);

    const Vector3& origin() const;
    const Vector3& direction() const;
    double l() const;

    Vector3& origin();
    Vector3& direction();
    double& l();
};

class BaseRayContact;

class RayModel : public Core::MechanicalObject<Vec3Types>, public Abstract::CollisionModel, public Abstract::VisualModel
{
protected:
    std::vector<double> length;

    double defaultLength;

    std::set<BaseRayContact*> contacts;

    bool static_;
public:
    typedef Vec3Types DataTypes;
    typedef Ray Element;
    friend class Ray;

    RayModel(double defaultLength=1);

    int addRay(const Vector3& origin, const Vector3& direction, double length);

    int getNbRay() const { return size; }

    void setNbRay(int n) { resize(2*n); }

    Ray getRay(int index) { return Ray(this, index); }

    virtual void addContact(BaseRayContact* contact) { contacts.insert(contact); }

    virtual void removeContact(BaseRayContact* contact) { contacts.erase(contact); }

    virtual void resize(int size);

    // -- CollisionModel interface

    virtual void computeBoundingTree(int maxDepth);

    bool isStatic() { return static_; }
    void setStatic(bool val=true) { static_ = val; }

    void draw(int index);

    void applyTranslation(double dx, double dy, double dz);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

inline Ray::Ray(RayModel* model, int index)
    : Abstract::TCollisionElementIterator<RayModel>(model, index)
{}

inline Ray::Ray(Abstract::CollisionElementIterator& i)
    : Abstract::TCollisionElementIterator<RayModel>(static_cast<RayModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Vector3& Ray::origin() const
{
    return (*model->getX())[2*index+0];
}

inline const Vector3& Ray::direction() const
{
    return (*model->getX())[2*index+1];
}

inline double Ray::l() const
{
    return model->length[index];
}

inline Vector3& Ray::origin()
{
    return (*model->getX())[2*index+0];
}

inline Vector3& Ray::direction()
{
    return (*model->getX())[2*index+1];
}

inline double& Ray::l()
{
    return model->length[index];
}

} // namespace Components

} // namespace Sofa

#endif
