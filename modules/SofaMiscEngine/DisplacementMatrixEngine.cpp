//#define SOFA_COMPONENT_ENGINE_DisplacementMatrixEngine_CPP

#include "DisplacementMatrixEngine.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS( DisplacementTransformEngine )

int DisplacementTransformEngineClass = core::RegisterObject("Converts a vector of Rigid to a vector of displacement transforms.")
    .add< DisplacementTransformEngine<Rigid3Types,Mat4x4f> >()
    .add< DisplacementTransformEngine<Rigid3Types,Rigid3Types::Coord> >()
;

template class SOFA_MISC_ENGINE_API DisplacementTransformEngine<Rigid3Types,Mat4x4f>;
template class SOFA_MISC_ENGINE_API DisplacementTransformEngine<Rigid3Types,Rigid3Types::Coord>;

template <>
void DisplacementTransformEngine<Rigid3Types,Rigid3Types::Coord>::setInverse( Rigid3Types::Coord& inv, const Coord& x0 )
{
    inv = Rigid3Types::inverse(x0);
}

template <>
void DisplacementTransformEngine<Rigid3Types,Rigid3Types::Coord>::mult( Rigid3Types::Coord& out, const Rigid3Types::Coord& inv, const Coord& x )
{
    out = x;
    out.multRight(inv);
}

template <>
void DisplacementTransformEngine<Rigid3Types,Mat4x4f>::setInverse( Mat4x4f& inv, const Coord& x0 )
{
    Rigid3Types::inverse(x0).toMatrix(inv);
}

template <>
void DisplacementTransformEngine<Rigid3Types,Mat4x4f>::mult( Mat4x4f& out, const Mat4x4f& inv, const Coord& x )
{
    x.toMatrix(out);
    out = out * inv;
}

/////////////////////////////////////////

SOFA_DECL_CLASS( DisplacementMatrixEngine )

int DisplacementMatrixEngineClass = core::RegisterObject("Converts a vector of Rigid to a vector of displacement matrices.")
    .add< DisplacementMatrixEngine<Rigid3Types> >()
;

} // namespace engine

} // namespace component

} // namespace sofa
