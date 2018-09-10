#include <SofaGeometry/Ray.h>

namespace sofageometry
{

// Create a new ray starting from 'origin' and pointing to 'direction'
Ray::Ray(const Vec3d& origin, const Vec3d& direction)
{
    this->origin = origin;
    this->direction = direction;
}

// Return a point along the ray at a given 'p' distance .
Vec3d Ray::getPoint(const double p) const
{
     return origin + direction * p ;
}

} /// namespace sofageometry
