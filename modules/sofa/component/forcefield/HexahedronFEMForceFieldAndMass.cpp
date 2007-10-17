
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

template class HexahedronFEMForceFieldAndMass<Vec3dTypes>;
template class HexahedronFEMForceFieldAndMass<Vec3fTypes>;


SOFA_DECL_CLASS(HexahedronFEMForceFieldAndMass)

// Register in the Factory
int HexahedronFEMForceFieldAndMassClass = core::RegisterObject("Hexahedral finite elements with mass")
        .add< HexahedronFEMForceFieldAndMass<Vec3dTypes> >()
        .add< HexahedronFEMForceFieldAndMass<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

