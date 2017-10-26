#include "SphereSurface.h"

namespace sofa
{
namespace component
{
namespace container
{

double SphereSurface::getDistance(defaulttype::Vec3d& Pos, int& domain)
{
    SOFA_UNUSED(domain) ;
    double result = m_radius - sqrt((Pos[0] - m_center[0])*(Pos[0] - m_center[0]) +
            (Pos[1] - m_center[1])*(Pos[1] - m_center[1]) +
            (Pos[2] - m_center[2])*(Pos[2] - m_center[2]));
    return m_inside ? result : -result;
}

double SphereSurface::getDistance(defaulttype::Vec3d& /*Pos*/, double value, double grad_norm, int &domain)
{
    (void)domain;
    if (grad_norm < 0) // use value
        grad_norm = sqrt(m_inside ? m_radius*m_radius - value : value + m_radius*m_radius);
    else grad_norm /= 2;
    return m_inside ? m_radius - grad_norm : grad_norm - m_radius;
}

}
}
}
