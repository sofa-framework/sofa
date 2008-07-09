/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "UncoupledConstraintCorrection.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace constraint
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(UncoupledConstraintCorrection)

int UncoupledConstraintCorrectionClass = core::RegisterObject("")
#ifndef SOFA_FLOAT
        .add< UncoupledConstraintCorrection<Vec1dTypes> >()
        .add< UncoupledConstraintCorrection<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< UncoupledConstraintCorrection<Vec1fTypes> >()
        .add< UncoupledConstraintCorrection<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class UncoupledConstraintCorrection<Vec1dTypes>;
template class UncoupledConstraintCorrection<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class UncoupledConstraintCorrection<Vec1fTypes>;
template class UncoupledConstraintCorrection<Rigid3fTypes>;
#endif


} // namespace constraint

} // namespace component

} // namespace sofa
