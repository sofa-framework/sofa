/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_INTERACTIONPERFORMER_H
#define SOFA_COMPONENT_COLLISION_INTERACTIONPERFORMER_H

#include <sofa/SofaGeneral.h>
#include <SofaGraphComponent/MouseButtonSetting.h>
#include <sofa/helper/Factory.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace collision
{

class BaseMouseInteractor;
template <class DataTypes>
class MouseInteractor;


class SOFA_USER_INTERACTION_API InteractionPerformer
{
public:
    typedef helper::Factory<std::string, InteractionPerformer, BaseMouseInteractor*> InteractionPerformerFactory;

    InteractionPerformer(BaseMouseInteractor *i):interactor(i),freezePerformer(0) {}
    virtual ~InteractionPerformer() {}

    virtual void configure(configurationsetting::MouseButtonSetting* /*setting*/) {}

    virtual void start()=0;
    virtual void execute()=0;

    virtual void handleEvent(core::objectmodel::Event * ) {}
    virtual void draw(const core::visual::VisualParams* ) {}

    virtual void setPerformerFreeze() {freezePerformer = true;}

    template <class RealObject>
    static RealObject* create( RealObject*, BaseMouseInteractor* interactor)
    {
        return new RealObject(interactor);
    }
    BaseMouseInteractor *interactor;
    bool freezePerformer;
};


template <class DataTypes>
class TInteractionPerformer: public InteractionPerformer
{
public:

    TInteractionPerformer(BaseMouseInteractor *i):InteractionPerformer(i) {};

    template <class RealObject>
    static RealObject* create( RealObject*, BaseMouseInteractor* interactor)
    {
        if (!dynamic_cast< MouseInteractor<DataTypes>* >(interactor)) return NULL;
        else return new RealObject(interactor);
    }

};

} // namespace collision
} // namespace component

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_INTERACTIONPERFOMER_CPP)
namespace helper
{
//delay load of the specialized Factory class. unique definition reside in the cpp file.
extern template class SOFA_USER_INTERACTION_API Factory<std::string, component::collision::InteractionPerformer, component::collision::BaseMouseInteractor*>;
}
#endif

} // namespace sofa

#endif
