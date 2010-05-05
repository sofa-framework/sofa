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
#define SOFA_COMPONENT_MAPPING_RIGIDMAPPING_CPP
#include <sofa/component/mapping/RigidMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MappedModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MechanicalMapping.inl>
#include <sofa/core/Mapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(RigidMapping)

using namespace defaulttype;
using namespace core;
using namespace core::behavior;


// Register in the Factory
int RigidMappingClass = core::RegisterObject("Set the positions and velocities of points attached to a rigid parent")
#ifndef SOFA_FLOAT
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Vec2dTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > > >()
// .add< RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Vec2fTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
// .add< RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Vec2fTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Vec2dTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > > >()
#endif
#endif
        ;

/// Template specialization for 2D rigids
/// \TODO Find a clean way not to replicate the code 4 times...
#ifndef SOFA_FLOAT

template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    const VecCoord& pts = this->getPoints();
    Deriv v;
    Real omega;
    v = in[0].getVCenter();
    omega = (Real)in[0].getVOrientation();
    out.resize(pts.size());
    for(unsigned int i=0; i<pts.size(); i++)
    {
        out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
    }
}

template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    const VecCoord& pts = this->getPoints();
    Deriv v;
    Real omega = (Real)0;
    for(unsigned int i=0; i<pts.size(); i++)
    {
        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[0].getVCenter() += v;
    out[0].getVOrientation() += (In::Real)omega;
}

template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in )
{
    out.resize(in.size());
    /// \TODO !!!
}
#endif

#ifndef SOFA_DOUBLE
template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    const VecCoord& pts = this->getPoints();
    Deriv v;
    Real omega;
    v = in[0].getVCenter();
    omega = (Real)in[0].getVOrientation();
    out.resize(pts.size());
    for(unsigned int i=0; i<pts.size(); i++)
    {
        out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
    }
}



template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    const VecCoord& pts = this->getPoints();
    Deriv v;
    Real omega = (Real)0;
    for(unsigned int i=0; i<pts.size(); i++)
    {
        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[0].getVCenter() += v;
    out[0].getVOrientation() += (In::Real)omega;
}

template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in )
{
    out.resize(in.size());
    /// \TODO !!!
}
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE


template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    const VecCoord& pts = this->getPoints();
    Deriv v;
    Real omega;
    v = in[0].getVCenter();
    omega = (Real)in[0].getVOrientation();
    out.resize(pts.size());
    for(unsigned int i=0; i<pts.size(); i++)
    {
        out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
    }
}


template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    const VecCoord& pts = this->getPoints();
    Deriv v;
    Real omega = (Real)0;
    for(unsigned int i=0; i<pts.size(); i++)
    {
        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[0].getVCenter() += v;
    out[0].getVOrientation() += (In::Real)omega;
}

template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in )
{
    out.resize(in.size());
    /// \TODO !!!
}

template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    const VecCoord& pts = this->getPoints();
    Deriv v;
    Real omega;
    v = in[0].getVCenter();
    omega = (Real)in[0].getVOrientation();
    out.resize(pts.size());
    for(unsigned int i=0; i<pts.size(); i++)
    {
        out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
    }
}



template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    const VecCoord& pts = this->getPoints();
    Deriv v;
    Real omega = (Real)0;
    for(unsigned int i=0; i<pts.size(); i++)
    {
        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[0].getVCenter() += v;
    out[0].getVOrientation() += (In::Real)omega;
}


template<>
void RigidMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in )
{
    out.resize(in.size());
    /// \TODO !!!
}

#endif
#endif


#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid2dTypes>, MechanicalState<Vec2dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid2fTypes>, MechanicalState<Vec2fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid2dTypes>, MechanicalState<Vec2fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid2fTypes>, MechanicalState<Vec2dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif


} // namespace mapping

} // namespace component

} // namespace sofa

