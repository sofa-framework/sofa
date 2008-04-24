#include "ParticleSource.h"
#include "sofa/core/componentmodel/behavior/Constraint.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include "sofa/defaulttype/Vec3Types.h"

namespace sofa
{

namespace component
{

SOFA_DECL_CLASS(ParticleSource)

int ParticleSourceClass = core::RegisterObject("Parametrable particle generator")
#ifndef SOFA_FLOAT
        .add< ParticleSource<defaulttype::Vec3dTypes> >()
        .add< ParticleSource<defaulttype::Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ParticleSource<defaulttype::Vec3fTypes> >()
        .add< ParticleSource<defaulttype::Vec2fTypes> >()
#endif
        ;

}
}
