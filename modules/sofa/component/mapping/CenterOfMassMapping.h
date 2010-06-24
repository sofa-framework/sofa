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
#ifndef SOFA_COMPONENT_MAPPING_CENTEROFMASSMAPPING_H
#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMAPPING_H

#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::core::behavior;

/** mapping computing the center of mass of an object.
	the output of the mapping has to be a single dof.
	Its position is then set from the input DOFs, proportionally to their mass.
	This allow to control an object by setting forces on its center of mass.
 */
template <class BasicMapping>
class CenterOfMassMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CenterOfMassMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::DataTypes InDataTypes;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename OutCoord::value_type Real;

    CenterOfMassMapping ( In* from, Out* to ): Inherit ( from, to )
    {}

    virtual ~CenterOfMassMapping()
    {}

    void init();

    virtual void apply ( typename Out::VecCoord& childPositions, const typename In::VecCoord& parentPositions );

    virtual void applyJ ( typename Out::VecDeriv& childForces, const typename In::VecDeriv& parentForces );

    virtual void applyJT ( typename In::VecDeriv& parentForces, const typename Out::VecDeriv& childForces );

    void draw();


protected :
    ///pointer on the input DOFs mass
    BaseMass * masses;

    /// the total mass of the input object
    double totalMass;

};
using namespace sofa::defaulttype;
#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_CENTEROFMASSMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< MechanicalMapping<MechanicalState<defaulttype::Rigid3dTypes>, MechanicalState<defaulttype::Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< MechanicalMapping<MechanicalState<defaulttype::Rigid2dTypes>, MechanicalState<defaulttype::Vec2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< Mapping< State<defaulttype::Rigid3dTypes>, MappedModel<defaulttype::Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< Mapping< State<defaulttype::Rigid3dTypes>, MappedModel<defaulttype::ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< Mapping< State<defaulttype::Rigid3dTypes>, MappedModel<defaulttype::ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< MechanicalMapping<MechanicalState<defaulttype::Rigid3fTypes>, MechanicalState<defaulttype::Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< MechanicalMapping<MechanicalState<defaulttype::Rigid2fTypes>, MechanicalState<defaulttype::Vec2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< Mapping< State<defaulttype::Rigid3fTypes>, MappedModel<defaulttype::Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< Mapping< State<defaulttype::Rigid3fTypes>, MappedModel<defaulttype::ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< Mapping< State<defaulttype::Rigid3fTypes>, MappedModel<defaulttype::ExtVec3fTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< MechanicalMapping<MechanicalState<defaulttype::Rigid3dTypes>, MechanicalState<defaulttype::Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< MechanicalMapping<MechanicalState<defaulttype::Rigid3fTypes>, MechanicalState<defaulttype::Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< MechanicalMapping<MechanicalState<defaulttype::Rigid2dTypes>, MechanicalState<defaulttype::Vec2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< MechanicalMapping<MechanicalState<defaulttype::Rigid2fTypes>, MechanicalState<defaulttype::Vec2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< Mapping< State<defaulttype::Rigid3dTypes>, MappedModel<defaulttype::Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMapping< Mapping< State<defaulttype::Rigid3fTypes>, MappedModel<defaulttype::Vec3dTypes> > >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
