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
#ifndef SOFA_COMPONENT_COLLISION_DEFAULTCONTACTMANAGER_H
#define SOFA_COMPONENT_COLLISION_DEFAULTCONTACTMANAGER_H

#include <sofa/core/componentmodel/collision/ContactManager.h>
#include <sofa/core/VisualModel.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace collision
{

class DefaultContactManager : public core::componentmodel::collision::ContactManager, public core::VisualModel
{
protected:
    std::map<std::pair<core::CollisionModel*,core::CollisionModel*>,core::componentmodel::collision::Contact*> contactMap;
    ContactVector contactVec;

    void clear();
public:
    DataField<std::string> response;

    DefaultContactManager();
    ~DefaultContactManager();

    void createContacts(DetectionOutputMap& outputs);

    const ContactVector& getContacts() { return contactVec; }

    //virtual const char* getTypeName() const { return "CollisionResponse"; }

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
