/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define SOFA_COMPONENT_MASS_DIAGONALMASS_CPP

#include <sofa/component/mass/DiagonalMass.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mass
{

using sofa::core::objectmodel::ComponentState ;

using namespace sofa::type;
using namespace sofa::defaulttype;

// Register in the Factory
int DiagonalMassClass = core::RegisterObject("Define a specific mass for each particle")
        .add< DiagonalMass<Vec3Types> >()
        .add< DiagonalMass<Vec2Types, Vec3Types> >()
        .add< DiagonalMass<Vec1Types> >()
        .add< DiagonalMass<Vec1Types, Vec2Types> >()
        .add< DiagonalMass<Vec1Types, Vec3Types> >()
//        .add< DiagonalMass<Rigid3Types> >()
//        .add< DiagonalMass<Rigid2Types> >()
//        .add< DiagonalMass<Rigid2Types, Rigid3Types> >()

        ;

template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec3Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec2Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec2Types, Vec3Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec1Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec1Types, Vec2Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec1Types, Vec3Types>;
//template class SOFA_COMPONENT_MASS_API DiagonalMass<Rigid3Types>;
//template class SOFA_COMPONENT_MASS_API DiagonalMass<Rigid2Types>;
//template class SOFA_COMPONENT_MASS_API DiagonalMass<Rigid2Types, Rigid3Types>;

} // namespace sofa::component::mass
