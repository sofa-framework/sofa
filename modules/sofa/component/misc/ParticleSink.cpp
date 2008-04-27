#include "ParticleSink.h"
#include "sofa/core/componentmodel/behavior/Constraint.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include "sofa/defaulttype/Vec3Types.h"

namespace sofa
{

namespace component
{

SOFA_DECL_CLASS(ParticleSink)

int ParticleSinkClass = core::RegisterObject("Parametrable particle generator")
#ifndef SOFA_FLOAT
        .add< ParticleSink<defaulttype::Vec3dTypes> >()
        .add< ParticleSink<defaulttype::Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ParticleSink<defaulttype::Vec3fTypes> >()
        .add< ParticleSink<defaulttype::Vec2fTypes> >()
#endif
        ;

}
}
