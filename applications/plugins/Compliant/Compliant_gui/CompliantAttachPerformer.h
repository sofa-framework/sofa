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
#ifndef SOFA_COMPONENT_COLLISION_CompliantAttachPerformer_H
#define SOFA_COMPONENT_COLLISION_CompliantAttachPerformer_H

#include <Compliant/Compliant_gui/config.h>
#include <SofaUserInteraction/InteractionPerformer.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaMiscMapping/DistanceFromTargetMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/Node.h>
#include <sofa/gui/MouseOperations.h>

#include <SofaOpenglVisual/OglModel.h>

#include "../misc/CompliantAttachButtonSetting.h"


namespace sofa
{

namespace gui
{
class SOFA_Compliant_gui_API CompliantAttachOperation : public Operation
{
public:
    CompliantAttachOperation(component::configurationsetting::CompliantAttachButtonSetting::SPtr s = core::objectmodel::New<component::configurationsetting::CompliantAttachButtonSetting>()) : Operation(s), setting(s){}
    static std::string getDescription() {return "CompliantAttach";}

protected:
    virtual std::string defaultPerformerType() { return "CompliantAttach"; }

    virtual void setSetting(component::configurationsetting::MouseButtonSetting* s) { Operation::setSetting(s); setting = down_cast<component::configurationsetting::CompliantAttachButtonSetting>(s); }
    component::configurationsetting::CompliantAttachButtonSetting::SPtr setting;
};
}

namespace component
{

namespace collision
{
struct BodyPicked;

/** Mouse interaction using a compliance forcefield, for an object animated using a compliance solver.

  @author Francois Faure, 2012
  */
template <class DataTypes>
class SOFA_Compliant_gui_API CompliantAttachPerformer: public TInteractionPerformer<DataTypes>
{
    typedef typename DataTypes::Real                                  Real;
    typedef component::container::MechanicalObject< DataTypes > Point3dState;

    typedef defaulttype::Vec<DataTypes::spatial_dimensions,Real> MouseVec;
    typedef defaulttype::StdVectorTypes<MouseVec,MouseVec,Real> MouseTypes;
    typedef component::collision::BaseContactMapper< MouseTypes > MouseContactMapper;



    typedef defaulttype::StdVectorTypes< defaulttype::Vec<1,Real>, defaulttype::Vec<1,Real>  >  DataTypes1;


    simulation::Node::SPtr pickedNode;       ///< Node containing the picked MechanicalState
    int pickedParticleIndex;                 ///< Index of the picked particle in the picked state
    simulation::Node::SPtr interactionNode;  ///< Node used to create the interaction components to constrain the picked point
    core::BaseMapping::SPtr mouseMapping;   ///< Mapping from the mouse position to the 3D point on the ray
    MouseContactMapper* mapper;
    Point3dState* mouseState;                  ///< Mouse state container  (position, velocity)

    mapping::BaseDistanceFromTargetMapping* _distanceMapping; ///< computes the distance from the picked point to its target
    core::behavior::BaseMechanicalState* _baseCollisionMState;


    // HACK FOR VISUAL MODEL EXPORT
    visualmodel::OglModel::SPtr _vm;


    void clear();                             ///< release the current interaction

    SReal _compliance;
    bool _isCompliance;
    SReal _arrowSize;
    defaulttype::RGBAColor _color;
    bool _visualmodel;  // to be able to export the mouse spring in obj


public:
    CompliantAttachPerformer(BaseMouseInteractor *i);
    ~CompliantAttachPerformer();
    virtual void configure(configurationsetting::MouseButtonSetting* setting);


    void start();
    void execute();

};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_CompliantAttachPerformer_CPP)
#ifndef SOFA_DOUBLE
extern template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Vec3fTypes>;
extern template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Rigid3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Vec3dTypes>;
extern template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Rigid3dTypes>;
#endif
#endif


}
}
}

#endif
