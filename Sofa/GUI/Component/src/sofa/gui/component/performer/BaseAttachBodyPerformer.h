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
#pragma once

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/gui/component/config.h>

#include <sofa/gui/component/performer/InteractionPerformer.h>
#include <sofa/component/collision/response/mapper/BaseContactMapper.h>
#include <sofa/core/visual/DisplayFlags.h>
#include <sofa/core/visual/VisualParams.h>


namespace sofa::gui::component::performer
{
struct BodyPicked;


/**
 * This class is a virtualization of attachment performer used to allow the blind use of either "AttachBodyPerformer" based on springs and "ConstraintAttachBodyPerformer" based on lagrangian
 * constraints. An example of use can be found in the external plugin Sofa.IGTLink in the component "iGTLinkMouseInteractor"
 */
template <class DataTypes>
class BaseAttachBodyPerformer :  public TInteractionPerformer<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef sofa::component::collision::response::mapper::BaseContactMapper< DataTypes >        MouseContactMapper;
    typedef sofa::core::behavior::MechanicalState< DataTypes >         MouseContainer;

    explicit BaseAttachBodyPerformer(BaseMouseInteractor* i);
    virtual ~BaseAttachBodyPerformer();

    virtual void start();
    virtual void draw(const core::visual::VisualParams* vparams);
    virtual void clear();
    virtual void execute();
    sofa::core::objectmodel::BaseObject::SPtr getInteractionObject();

    virtual bool startPartial(const BodyPicked& picked) = 0;


protected:

    sofa::core::objectmodel::BaseObject::SPtr m_interactionObject;
    MouseContactMapper  *m_mapper;
    core::visual::DisplayFlags m_flags;
};
}
