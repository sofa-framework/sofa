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

#ifndef SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_INL
#define SOFA_COMPONENT_CONTAINER_ARTICULATEDHIERARCHYCONTAINER_INL

#include <sofa/component/container/ArticulatedHierarchyContainer.h>

namespace sofa
{

namespace component
{

namespace container
{

ArticulatedHierarchyContainer::ArticulationCenter::Articulation::Articulation():
    axis(dataField(&axis, (Vector3) Vector3(1,0,0), "rotationAxis", "Set the rotation axis for the articulation")),
    rotation(dataField(&rotation, (bool) true, "rotation", "Rotation")),
    translation(dataField(&translation, (bool) false, "translation", "Translation")),
    articulationIndex(dataField(&articulationIndex, (int) 0, "articulationIndex", "Articulation index"))
{
}

ArticulatedHierarchyContainer::ArticulationCenter::ArticulationCenter():
    parentIndex(dataField(&parentIndex, "parentIndex", "Parent of the center articulation")),
    childIndex(dataField(&childIndex, "childIndex", "Child of the center articulation")),
    globalPosition(dataField(&globalPosition, "globalPosition", "Global position of the articulation center")),
    posOnParent(dataField(&posOnParent, "posOnParent", "Parent position of the articulation center")),
    posOnChild(dataField(&posOnChild, "posOnChild", "Child position of the articulation center"))
{
}

const vector<ArticulatedHierarchyContainer::ArticulationCenter*> ArticulatedHierarchyContainer::getArticulationCenters()
{
    vector<ArticulationCenter*> articulationCenters;
    GNode* context = dynamic_cast<GNode*>(this->getContext());
    context->getTreeObjects<ArticulationCenter>(&articulationCenters);
    return articulationCenters;
}

const vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> ArticulatedHierarchyContainer::ArticulationCenter::getArticulations()
{
    vector<Articulation*> articulations;
    GNode* context = dynamic_cast<GNode*>(this->getContext());
    context->getTreeObjects<Articulation>(&articulations);
    return articulations;
}

} // namespace container

} // namespace component

} // namespace sofa

#endif
