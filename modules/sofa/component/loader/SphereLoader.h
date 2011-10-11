#ifndef SOFA_COMPONENT_LOADER_SPHERELOADER_H
#define SOFA_COMPONENT_LOADER_SPHERELOADER_H

#include <sofa/core/loader/BaseLoader.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>

namespace sofa
{
namespace component
{
namespace loader
{

class SphereLoader : public sofa::core::loader::BaseLoader
{
public:
    SOFA_CLASS(SphereLoader,sofa::core::loader::BaseLoader);
    SphereLoader();
    // Point coordinates in 3D in double.
    Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > positions;
    Data< helper::vector<SReal> > radius;
    virtual bool load();
};

} //loader
} //component
} //sofa

#endif // SOFA_COMPONENT_LOADER_SPHERELOADER_H
