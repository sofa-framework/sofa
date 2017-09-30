/// This file is only forwarding to the new location
#include <SofaImplicitField/components/geometry/DiscreteGridField.h>

namespace sofa
{
namespace component
{
namespace container
{

class ImplicitSurfaceContainer : public sofa::component::geometry::DiscreteGridField
{
public:
    bool loadImage( const char *filename )
    {
        return loadGridFromMHD(filename) ;
    }

    virtual double getDistance(defaulttype::Vec3d& pos)
    {
        int domain=-1;
        return getDistance(pos,domain);
    }

    virtual double getDistance(defaulttype::Vec3d& pos, int& domain)
    {
        defaulttype::Vec3d grad;
        double value;
        getValueAndGradient(pos,value,grad,domain);
        return getDistance(pos,value,grad.norm());
    }

    virtual double getDistance(defaulttype::Vec3d& pos, double value, double grad_norm)
    {
        int domain=-1;
        return getDistance(pos, value, grad_norm, domain);
    }

    virtual double getDistance(defaulttype::Vec3d& /*pos*/, double value, double grad_norm, int &domain)
    {
        SOFA_UNUSED(domain);
        /// use Taubin's distance by default
        if (grad_norm < 1e-10) return value < 0 ? double(std::numeric_limits<long>::min()) : double(std::numeric_limits<long>::max());
        return value/grad_norm;
    }
};

}
}
}
