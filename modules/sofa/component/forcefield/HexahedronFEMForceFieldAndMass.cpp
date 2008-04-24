
#include <sofa/component/forcefield/HexahedronFEMForceFieldAndMass.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(HexahedronFEMForceFieldAndMass)

// Register in the Factory
int HexahedronFEMForceFieldAndMassClass = core::RegisterObject("Hexahedral finite elements with mass")
#ifndef SOFA_FLOAT
        .add< HexahedronFEMForceFieldAndMass<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< HexahedronFEMForceFieldAndMass<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class HexahedronFEMForceFieldAndMass<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class HexahedronFEMForceFieldAndMass<Vec3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa

