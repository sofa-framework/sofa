/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define FLEXIBLE_StrainTYPES_CPP

#include "../types/StrainTypes.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/State.inl>
#include <SofaBaseMechanics/MechanicalObject.inl>

#include <sofa/defaulttype/TemplatesAliases.h>
using sofa::defaulttype::RegisterTemplateAlias;

namespace sofa
{
namespace component
{
namespace container
{


// ==========================================================================
// Instanciation

using namespace sofa::defaulttype;

int StrainMechanicalObjectClass = core::RegisterObject ( "mechanical state vectors" )
        .add< MechanicalObject<E331dTypes> >()
        .add< MechanicalObject<E321dTypes> >()
        .add< MechanicalObject<E311dTypes> >()
        .add< MechanicalObject<E332dTypes> >()
        .add< MechanicalObject<E333dTypes> >()
        .add< MechanicalObject<E221dTypes> >()
        .add< MechanicalObject<I331dTypes> >()
        .add< MechanicalObject<U331dTypes> >()
        .add< MechanicalObject<U321dTypes> >();

template class SOFA_Flexible_API MechanicalObject<E331dTypes>;
template class SOFA_Flexible_API MechanicalObject<E321dTypes>;
template class SOFA_Flexible_API MechanicalObject<E311dTypes>;
template class SOFA_Flexible_API MechanicalObject<E332dTypes>;
template class SOFA_Flexible_API MechanicalObject<E333dTypes>;
template class SOFA_Flexible_API MechanicalObject<E221dTypes>;
template class SOFA_Flexible_API MechanicalObject<I331dTypes>;
template class SOFA_Flexible_API MechanicalObject<U331dTypes>;
template class SOFA_Flexible_API MechanicalObject<U321dTypes>;

static RegisterTemplateAlias alias1("E331", E332Types::Name() );
static RegisterTemplateAlias alias2("E321", E321Types::Name() );
static RegisterTemplateAlias alias3("E311", E311Types::Name() );
static RegisterTemplateAlias alias4("E332", E332Types::Name() );
static RegisterTemplateAlias alias5("E333", E333Types::Name() );
static RegisterTemplateAlias alias6("E221", E221Types::Name() );
static RegisterTemplateAlias alias7("I331", I331Types::Name() );
static RegisterTemplateAlias alias8("U331", U331Types::Name() );
static RegisterTemplateAlias alias9("U321", U321Types::Name() );

} // namespace container
} // namespace component
} // namespace sofa
