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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

using namespace sofa::defaulttype;

//template class Factory<std::string, Contact, std::pair<core::CollisionModel*,core::CollisionModel*> >;

Contact::Factory* Contact::Factory::getInstance()
{
    static Factory instance;
    return &instance;
}

Contact* Contact::Create(const std::string& type, core::CollisionModel* model1, core::CollisionModel* model2, Intersection* intersectionMethod)
{
    return Factory::CreateObject(type,std::make_pair(std::make_pair(model1,model2),intersectionMethod));
}

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

