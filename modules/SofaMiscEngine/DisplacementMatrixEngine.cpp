/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
