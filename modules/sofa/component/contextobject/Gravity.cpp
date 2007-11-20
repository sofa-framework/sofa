/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>


namespace sofa
{

namespace component
{

namespace contextobject
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;

Gravity::Gravity()
    : f_gravity( initData(&f_gravity,Vec3(0,0,0),"gravity","Gravity in the world coordinate system") )
{
}

void Gravity::apply()
{
    getContext()->setGravityInWorld( f_gravity.getValue() );
}

SOFA_DECL_CLASS(Gravity)

int GravityClass = core::RegisterObject("Gravity in world coordinates")
        .add< Gravity >()
        ;

} // namespace contextobject

} // namespace component

} // namespace sofa

