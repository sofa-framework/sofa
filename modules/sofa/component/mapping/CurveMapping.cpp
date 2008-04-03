//
// C++ Implementation: CurveMapping
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/component/mapping/CurveMapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;


SOFA_DECL_CLASS(CurveMapping)

// Register in the Factory
int CurveMappingClass = core::RegisterObject("")
        .add< CurveMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Rigid3dTypes> > > >()
        .add< CurveMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Rigid3fTypes> > > >()
        .add< CurveMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Rigid3dTypes> > > >()
        .add< CurveMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Rigid3fTypes> > > >()
        ;

// Mech -> Mech
template class CurveMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Rigid3dTypes> > >;
template class CurveMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Rigid3fTypes> > >;
template class CurveMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Rigid3dTypes> > >;
template class CurveMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Rigid3fTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa
