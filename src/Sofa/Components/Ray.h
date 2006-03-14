#ifndef SOFA_COMPONENTS_RAY_H
#define SOFA_COMPONENTS_RAY_H

#include "Sofa/Abstract/CollisionElement.h"
#include "Common/Vec.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class RayModel;

class Ray : public Abstract::CollisionElement
{
protected:
    RayModel* model;
    int		index;
    double length;
public:

    Ray(RayModel* sph=NULL);
    Ray(const Ray &sphere, RayModel* sph);
    Ray(const Ray &sphere);
    Ray(double l, int idx, RayModel* sph=NULL);

    void getBBox (Vector3 &minBBox, Vector3 &maxBBox);
    void getBBox (double* minBBox, double* maxBBox);
    bool isSelfCollis (CollisionElement *elem);
    void clear();

    const Vector3& origin() const;
    const Vector3& direction() const;

    Vector3& origin();
    Vector3& direction();

    inline const double& x0() const { return origin()[0]; }
    inline const double& y0() const { return origin()[1]; }
    inline const double& z0() const { return origin()[2]; }
    inline const double& dx() const { return direction()[0]; }
    inline const double& dy() const { return direction()[1]; }
    inline const double& dz() const { return direction()[2]; }
    inline const double& l() const { return length; }
    inline double& l() { return length; }

    void draw();

    friend std::ostream& operator<< (std::ostream& os, const Ray &sph);

    int getIndex() { return index; }

    //const Vector3& getPosition() { return center(); }

//	void addForce (const Vector3 &force);

    Abstract::CollisionModel* getCollisionModel();

    friend class RayModel;
};

} // namespace Components

} // namespace Sofa

#endif
