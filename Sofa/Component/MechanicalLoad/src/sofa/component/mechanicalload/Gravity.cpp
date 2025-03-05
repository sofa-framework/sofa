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
#include <sofa/component/mechanicalload/Gravity.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa::component::mechanicalload
{

using namespace sofa::type;
using namespace core::behavior;

Gravity::Gravity()
    : f_gravity( initData(&f_gravity,Vec3(0,0,0),"gravity","Gravity in the world coordinate system") )
{
}

void Gravity::apply()
{
    getContext()->setGravity( f_gravity.getValue() );
}

void registerGravity(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Gravity in world coordinates.")
        .add< Gravity >());
}

} // namespace sofa::component::mechanicalload
