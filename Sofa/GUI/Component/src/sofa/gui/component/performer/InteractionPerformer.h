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
#include <sofa/gui/component/config.h>

#include <sofa/component/setting/MouseButtonSetting.h>
#include <sofa/helper/Factory.h>
#include <sofa/core/fwd.h>

namespace sofa::gui::component::performer
{

class BaseMouseInteractor;
template <class DataTypes>
class MouseInteractor;


class SOFA_GUI_COMPONENT_API InteractionPerformer
{
public:
    typedef helper::Factory<std::string, InteractionPerformer, BaseMouseInteractor*> InteractionPerformerFactory;

    InteractionPerformer(BaseMouseInteractor *i):m_interactor(i),m_freezePerformer(0) {}
    virtual ~InteractionPerformer() {}

    virtual void configure(sofa::component::setting::MouseButtonSetting* /*setting*/) {}

    virtual void start()=0;
    virtual void execute()=0;

    virtual void handleEvent(core::objectmodel::Event * ) {}
    virtual void draw(const core::visual::VisualParams* ) {}

    virtual void setPerformerFreeze() {m_freezePerformer = true;}

    template <class RealObject>
    static RealObject* create( RealObject*, BaseMouseInteractor* interactor)
    {
        return new RealObject(interactor);
    }
    BaseMouseInteractor *m_interactor;
    bool m_freezePerformer;

    SOFA_ATTRIBUTE_DISABLED__NAMING("v24.06", "v24.06", interactor,m_interactor);
    SOFA_ATTRIBUTE_DISABLED__NAMING("v24.06", "v24.06", freezePerformer,m_freezePerformer);

};


template <class DataTypes>
class TInteractionPerformer: public InteractionPerformer
{
public:

    TInteractionPerformer(BaseMouseInteractor *i):InteractionPerformer(i) {}

    template <class RealObject>
    static RealObject* create( RealObject*, BaseMouseInteractor* interactor)
    {
        if (!dynamic_cast< MouseInteractor<DataTypes>* >(interactor)) return nullptr;
        else return new RealObject(interactor);
    }

};

} //namespace sofa::gui::component::performer

namespace sofa::helper
{
extern template class SOFA_GUI_COMPONENT_API Factory<std::string, sofa::gui::component::performer::InteractionPerformer, sofa::gui::component::performer::BaseMouseInteractor*>;
} //namespace sofa::helper
