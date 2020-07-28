#pragma once

#include "Vec.h"

namespace sofa
{    
namespace defaulttype
{

/// Representation of rays.
/// A ray is an infinite line starting at origin and going in some direction.
class Ray
{
public:
    Ray(const Vec3& origin = Vec3(0,0,0), const Vec3& direction = Vec3(0,0,0))
    {
        m_origin = origin;
        m_direction = direction.normalized();
    }

    const Vec3& origin() const { return m_origin; }
    const Vec3& direction() const { return m_direction; }

    Vec3 getPoint(double z) const //< Returns a point at distance units along the ray.
    {
        return m_origin + (m_direction * z);
    }

    void setOrigin(const Vec3& origin) { m_origin = origin; }
    void setDirection(const Vec3& direction) { m_direction = direction.normalized(); }

private:
    Vec3 m_origin;
    Vec3 m_direction;
};
} /// namespace defaulttype   
} /// namespace sofa
