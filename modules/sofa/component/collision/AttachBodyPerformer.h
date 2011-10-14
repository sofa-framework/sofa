/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_COLLISION_ATTACHBODYPERFORMER_H
#define SOFA_COMPONENT_COLLISION_ATTACHBODYPERFORMER_H

#include <sofa/component/collision/InteractionPerformer.h>
#include <sofa/component/collision/BaseContactMapper.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/component/interactionforcefield/SpringForceField.h>
#include <sofa/component/interactionforcefield/StiffSpringForceField.h>
#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace core
{
namespace objectmodel
{
class TagSet;
}
}
namespace component
{
namespace container
{
template< class T > class MechanicalObject;
} // namespace container

namespace collision
{

struct BodyPicked;

class AttachBodyPerformerConfiguration
{
public:
    AttachBodyPerformerConfiguration():stiffness(1000),size(0),showFactorSize(1.0)
    {};
    void setStiffness(double s) {stiffness=s;}
    void setArrowSize(float s) {size=s;}
    void setShowFactorSize(float s) {showFactorSize = s;}
protected:
    SReal stiffness;
    SReal size;
    SReal showFactorSize;
};

template <class DataTypes>
class AttachBodyPerformer: public TInteractionPerformer<DataTypes>, public AttachBodyPerformerConfiguration
{

    typedef sofa::component::collision::BaseContactMapper< DataTypes >        MouseContactMapper;
    typedef sofa::component::container::MechanicalObject< DataTypes >         MouseContainer;
    typedef sofa::core::behavior::BaseForceField              MouseForceField;

protected:
    AttachBodyPerformer(BaseMouseInteractor *i);
    ~AttachBodyPerformer();
public:
    void start();
    void execute();
    void draw(const core::visual::VisualParams* vparams);
    void clear();



protected:
    bool start_partial(const BodyPicked& picked);
    /*
    initialise MouseForceField according to template.
    StiffSpringForceField for Vec3
    JointSpringForceField for Rigid3
    */

    MouseContactMapper  *mapper;
    MouseForceField::SPtr m_forcefield;

    core::visual::DisplayFlags flags;
};



#if defined(WIN32) && !defined(SOFA_COMPONENT_COLLISION_ATTACHBODYPERFORMER_CPP)
#ifndef SOFA_DOUBLE
extern template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Vec3fTypes>;
extern template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Rigid3fTypes>;

#endif
#ifndef SOFA_FLOAT
extern template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Vec3dTypes>;
extern template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Rigid3dTypes>;
#endif
#endif


}
}
}

#endif
