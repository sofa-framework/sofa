/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/mapping/RigidMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(RigidMapping)

using namespace defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;


// Register in the Factory
int RigidMappingClass = core::RegisterObject("Set the positions and velocities of points attached to a rigid parent")
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Vec2dTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Vec2fTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Vec2dTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Vec2fTypes> > > >()
        ;

//Michael : why does this was commented?
template class RigidMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
template class RigidMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;

template class RigidMapping< Mapping< State<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
template class RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
template class RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;

template class RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;


/// Template specialization for 2D rigids
/// \TODO Find a clean way not to replicate the code 4 times...

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    Deriv v;
    Real omega;
    v = in[0].getVCenter();
    omega = (Real)in[0].getVOrientation();
    out.resize(points.getValue().size());
    for(unsigned int i=0; i<points.getValue().size(); i++)
    {
        out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
    }
}

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    Deriv v;
    Real omega;
    v = in[0].getVCenter();
    omega = (Real)in[0].getVOrientation();
    out.resize(points.getValue().size());
    for(unsigned int i=0; i<points.getValue().size(); i++)
    {
        out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
    }
}

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    Deriv v;
    Real omega;
    v = in[0].getVCenter();
    omega = (Real)in[0].getVOrientation();
    out.resize(points.getValue().size());
    for(unsigned int i=0; i<points.getValue().size(); i++)
    {
        out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
    }
}

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    Deriv v;
    Real omega;
    v = in[0].getVCenter();
    omega = (Real)in[0].getVOrientation();
    out.resize(points.getValue().size());
    for(unsigned int i=0; i<points.getValue().size(); i++)
    {
        out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
    }
}


template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    Deriv v;
    Real omega = (Real)0;
    for(unsigned int i=0; i<points.getValue().size(); i++)
    {
        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[0].getVCenter() += v;
    out[0].getVOrientation() += (In::Real)omega;
}

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    Deriv v;
    Real omega = (Real)0;
    for(unsigned int i=0; i<points.getValue().size(); i++)
    {
        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[0].getVCenter() += v;
    out[0].getVOrientation() += (In::Real)omega;
}

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    Deriv v;
    Real omega = (Real)0;
    for(unsigned int i=0; i<points.getValue().size(); i++)
    {
        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[0].getVCenter() += v;
    out[0].getVOrientation() += (In::Real)omega;
}

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    Deriv v;
    Real omega = (Real)0;
    for(unsigned int i=0; i<points.getValue().size(); i++)
    {
        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[0].getVCenter() += v;
    out[0].getVOrientation() += (In::Real)omega;
}



template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in )
{
    out.resize(in.size());
    /// \TODO !!!
}

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in )
{
    out.resize(in.size());
    /// \TODO !!!
}

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in )
{
    out.resize(in.size());
    /// \TODO !!!
}

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in )
{
    out.resize(in.size());
    /// \TODO !!!
}

} // namespace mapping

} // namespace component

} // namespace sofa

