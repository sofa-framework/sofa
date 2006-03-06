#ifndef SOFA_COMPONENTS_SPHERE_H
#define SOFA_COMPONENTS_SPHERE_H

#include "Sofa/Abstract/CollisionElement.h"
#include "Common/Vec.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class SphereModel;

class Sphere : public Abstract::CollisionElement
{
protected:
    double	radius;
    int		index;
    SphereModel* model;
public:

    Sphere(SphereModel* sph=NULL);
    Sphere(const Sphere &sphere, SphereModel* sph);
    Sphere(const Sphere &sphere);
    Sphere(double r, int idx, SphereModel* sph=NULL);

    void getBBox (Vector3 &minBBox, Vector3 &maxBBox);
    void getBBox (double* minBBox, double* maxBBox);
    bool isSelfCollis (CollisionElement *elem);
    void clear();

    const Vector3& center() const;

    inline const double& x() const { return center()[0]; }
    inline const double& y() const { return center()[1]; }
    inline const double& z() const { return center()[2]; }
    inline const double& r() const { return radius; }

    void draw();

    friend std::ostream& operator<< (std::ostream& os, const Sphere &sph);

    int getIndex() { return index; }

    const Vector3& getPosition() { return center(); }

//	void addForce (const Vector3 &force);

    Abstract::CollisionModel* getCollisionModel();

    friend class SphereModel;
};

} // namespace Components

} // namespace Sofa

#endif
