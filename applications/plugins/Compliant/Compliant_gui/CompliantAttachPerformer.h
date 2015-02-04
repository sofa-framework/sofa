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
#ifndef SOFA_COMPONENT_COLLISION_CompliantAttachPerformer_H
#define SOFA_COMPONENT_COLLISION_CompliantAttachPerformer_H

#include "initCompliant_gui.h"
#include <sofa/component/collision/InteractionPerformer.h>
#include <sofa/component/collision/BaseContactMapper.h>
#include <sofa/component/mapping/DistanceFromTargetMapping.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/gui/MouseOperations.h>

#include <sofa/component/visualmodel/OglModel.h>

#include "../misc/CompliantAttachButtonSetting.h"


namespace sofa
{
using defaulttype::Vec;

namespace gui
{
class SOFA_Compliant_gui_API CompliantAttachOperation : public Operation
{
public:
    CompliantAttachOperation(sofa::component::configurationsetting::CompliantAttachButtonSetting::SPtr s = sofa::core::objectmodel::New<sofa::component::configurationsetting::CompliantAttachButtonSetting>()) : Operation(s), setting(s){}
//    virtual void start() ;
//    virtual void execution() ;
//    virtual void end() ;
//    virtual void endOperation() ;
    static std::string getDescription() {return "CompliantAttach";}

protected:
    virtual std::string defaultPerformerType() { return "CompliantAttach"; }

    virtual void setSetting(sofa::component::configurationsetting::MouseButtonSetting* s) { Operation::setSetting(s); setting = dynamic_cast<sofa::component::configurationsetting::CompliantAttachButtonSetting*>(s); }
    sofa::component::configurationsetting::CompliantAttachButtonSetting::SPtr setting;
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
    typedef defaulttype::StdVectorTypes< Vec<1,Real>, Vec<1,Real>  >  DataTypes1;
    typedef mapping::DistanceFromTargetMapping< DataTypes,DataTypes1 >          DistanceFromTargetMapping31;
    typedef sofa::component::container::MechanicalObject< DataTypes > Point3dState;

    simulation::Node::SPtr pickedNode;       ///< Node containing the picked MechanicalState
    int pickedParticleIndex;                 ///< Index of the picked particle in the picked state
    simulation::Node::SPtr interactionNode;  ///< Node used to create the interaction components to constrain the picked point
    core::BaseMapping::SPtr mouseMapping;   ///< Mapping from the mouse position to the 3D point on the ray
    sofa::component::collision::BaseContactMapper< DataTypes >  *mapper;
    Point3dState* mouseState;                  ///< Mouse state container  (position, velocity)
    typename DistanceFromTargetMapping31::SPtr distanceMapping; ///< computes the distance from the picked point to its target


    // HACK FOR VISUAL MODEL EXPORT
    visualmodel::OglModel::SPtr _vm;


    void clear();                             ///< release the current interaction

    SReal _compliance;
    bool _isCompliance;
    SReal _arrowSize;
    defaulttype::Vec<4,SReal> _color;
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
