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
#ifndef SOFA_CORE_COLLISIONMODEL_INL
#define SOFA_CORE_COLLISIONMODEL_INL

#include <sofa/core/CollisionModel.h>

namespace sofa
{
namespace core
{

/// Helper method to get or create the previous model in the hierarchy.
template<class DerivedModel>
DerivedModel* CollisionModel::createPrevious()
{
    CollisionModel::SPtr prev = previous.get();
    typename DerivedModel::SPtr pmodel = sofa::core::objectmodel::SPtr_dynamic_cast<DerivedModel>(prev);
    if (pmodel.get() == NULL)
    {
        int level = 0;
        CollisionModel *cm = getNext();
        CollisionModel* root = this;
        while (cm) { root = cm; cm = cm->getNext(); ++level; }
        pmodel = sofa::core::objectmodel::New<DerivedModel>();
        pmodel->setName("BVLevel",level);
        root->addSlave(pmodel); //->setContext(getContext());
        pmodel->setMoving(isMoving());
        pmodel->setSimulated(isSimulated());
        pmodel->proximity.setValue(proximity.getValue());
        //pmodel->group.setValue(group_old.getValue());
        pmodel->group.beginEdit()->insert(group.getValue().begin(),group.getValue().end());
        pmodel->group.endEdit();
        //previous=pmodel;
        //pmodel->next = this;
        setPrevious(pmodel);
        if (prev)
        {

        }
    }
    return pmodel.get();
}

} // namespace core

} // namespace sofa

#endif
