#ifndef SOFA_SOFAIMPLICITFIELD_H
#define SOFA_SOFAIMPLICITFIELD_H

#include <SofaImplicitField/components/geometry/SphericalField.h>

namespace sofa
{
namespace component
{
namespace container
{

class SphereSurface : public sofa::component::geometry::SphericalField
{
public:
    SOFA_CLASS(SphereSurface, sofa::component::geometry::SphericalField) ;

    // The following function uses only either value or grad_norm (they are redundant)
    // - value is used is grad_norm < 0
    // - else grad_norm is used: for example, in that case dist = _radius - grad_norm/2 (with _inside=true)
    virtual double getDistance(sofa::defaulttype::Vec3d& pos, int& domain) ;
    virtual double getDistance(sofa::defaulttype::Vec3d& pos, double value, double grad_norm, int &domain) ;
};

} /// container
} /// components
} /// sofa

#endif /// SOFA_SOFAIMPLICITFIELD_H
