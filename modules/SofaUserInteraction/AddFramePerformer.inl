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

#include <SofaUserInteraction/AddFramePerformer.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaUserInteraction/MouseInteractor.h>
#include <SofaRigid/SkinningMapping.inl>
#include <sofa/helper/Quater.h>

namespace sofa
{

namespace component
{

namespace collision
{
template <class DataTypes>
void AddFramePerformer<DataTypes>::start()
{
    std::cout << "Frame1" << std::endl;
    BodyPicked picked=this->interactor->getBodyPicked();
    if (!picked.body && !picked.mstate) return;

    vector<FBMapping*> vFBMapping;
    typename DataTypes::Coord point;

    if (picked.body)
    {
        point = picked.point;
        sofa::core::objectmodel::BaseContext* context=  picked.body->getContext();
        context->get<FBMapping>( &vFBMapping, core::objectmodel::BaseContext::SearchRoot);
    }
    else
    {
        core::behavior::MechanicalState<DataTypes>* mstateCollision=dynamic_cast< core::behavior::MechanicalState<DataTypes>*  >(picked.mstate);
        if (!mstateCollision)
        {
            this->interactor->serr << "incompatible MState during Mouse Interaction " << this->interactor->sendl;
            return;
        }
        static_cast<simulation::Node*>(mstateCollision->getContext())->get<FBMapping>( &vFBMapping, core::objectmodel::BaseContext::SearchRoot);
        int index = picked.indexCollisionElement;
        point=(*(mstateCollision->getX()))[index];
    }

    for( typename vector<FBMapping *>::iterator it = vFBMapping.begin(); it != vFBMapping.end(); it++)
        (*it)->insertFrame( point);
}

template <class DataTypes>
void AddFramePerformer<DataTypes>::execute()
{
    std::cout << "Frame2" << std::endl;
};

template <class DataTypes>
AddFramePerformer<DataTypes>::AddFramePerformer(BaseMouseInteractor *i):TInteractionPerformer<DataTypes>(i)
{
}


template <class DataTypes>
AddFramePerformer<DataTypes>::~AddFramePerformer()
{
    //Should remove the frames added
};


}
}
}
