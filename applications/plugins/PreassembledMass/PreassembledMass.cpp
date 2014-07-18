#define __PreassembledMass_CPP

#include "PreassembledMass.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;



SOFA_DECL_CLASS(PreassembledMass)


// Register in the Factory
int PreassembledMassClass = core::RegisterObject("Preassembled mass")

        .add< PreassembledMass<Vec3Types> >( true )
        .add< PreassembledMass<Vec1Types> >()
        .add< PreassembledMass<Rigid3Types> >()

        #if SOFA_HAVE_FLEXIBLE
        .add< PreassembledMass<Affine3Types> >()
        #endif
        ;

template class SOFA_PreassembledMass_API PreassembledMass<defaulttype::Vec3Types>; // volume FEM (tetra, hexa)
template class SOFA_PreassembledMass_API PreassembledMass<defaulttype::Vec1Types>; // subspace
template class SOFA_PreassembledMass_API PreassembledMass<defaulttype::Rigid3Types>; // rigid frames
#if SOFA_HAVE_FLEXIBLE
template class SOFA_PreassembledMass_API PreassembledMass<defaulttype::Affine3Types>; // affine frames
#endif

} // namespace mass

} // namespace component

} // namespace sofa

