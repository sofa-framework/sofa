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
#define SOFA_COMPONENT_MASS_DIAGONALMASS_CPP

#include <sofa/component/mass/DiagonalMass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/topology/Edge.h>

using sofa::core::visual::VisualParams ;
using sofa::core::MechanicalParams ;
using sofa::helper::ReadAccessor ;

namespace sofa::component::mass
{

using sofa::core::objectmodel::ComponentState ;

using namespace sofa::type;
using namespace sofa::defaulttype;

template <class RigidTypes, class GeometricalTypes>
template <class T>
SReal DiagonalMass<RigidTypes, GeometricalTypes>::getPotentialEnergyRigidImpl( const MechanicalParams* mparams,
                                                                        const DataVecCoord& x) const
{
    SOFA_UNUSED(mparams) ;
    SReal e = 0;
    const MassVector &masses= d_vertexMass.getValue();
    const VecCoord& _x = x.getValue();

    // gravity
    const Vec3& g = this->getContext()->getGravity();
    Deriv theGravity;
    RigidTypes::set( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<_x.size(); i++)
    {
        e -= getVCenter(theGravity) * masses[i].mass * _x[i].getCenter();
    }
    return e;
}


template <class RigidTypes, class GeometricalTypes>
template <class T>
void DiagonalMass<RigidTypes, GeometricalTypes>::drawRigid3dImpl(const VisualParams* vparams)
{
    const MassVector &masses= d_vertexMass.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();

    if(masses.size() != x.size()) return;

    Real totalMass=0;
    typename RigidTypes::Vec3 gravityCenter;
    for (unsigned int i=0; i<x.size(); i++)
    {
        if (masses[i].mass == 0) continue;
        const auto& orient = x[i].getOrientation();
        //orient[3] = -orient[3];
        const typename RigidTypes::Vec3& center = x[i].getCenter();
        typename RigidTypes::Vec3 len;
        // The moment of inertia of a box is:
        //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
        //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
        //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
        // So to get lx,ly,lz back we need to do
        //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
        // Note that RigidMass inertiaMatrix is already divided by M
        const double m00 = masses[i].inertiaMatrix[0][0];
        const double m11 = masses[i].inertiaMatrix[1][1];
        const double m22 = masses[i].inertiaMatrix[2][2];
        len[0] = sqrt(m11+m22-m00);
        len[1] = sqrt(m00+m22-m11);
        len[2] = sqrt(m00+m11-m22);

        vparams->drawTool()->drawFrame(center, orient, len*d_showAxisSize.getValue() );

        gravityCenter += (center * masses[i].mass);
        totalMass += masses[i].mass;
    }

    if(d_showCenterOfGravity.getValue())
    {
        gravityCenter /= totalMass;

        vparams->drawTool()->drawCross(gravityCenter, d_showAxisSize.getValue(), sofa::type::RGBAColor::yellow());
    }
}


template <class RigidTypes, class GeometricalTypes>
template <class T>
void DiagonalMass<RigidTypes, GeometricalTypes>::drawRigid2dImpl(const VisualParams* vparams)
{
    const MassVector &masses= d_vertexMass.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i=0; i<x.size(); i++)
    {
        if (masses[i].mass == 0) continue;
        Vec3d len;
        len[0] = len[1] = sqrt(masses[i].inertiaMatrix);
        len[2] = 0;

        Quatd orient(Vec3d(0,0,1), x[i].getOrientation());
        Vec3d center; center = x[i].getCenter();
        vparams->drawTool()->drawFrame(center, orient, len*d_showAxisSize.getValue() );
    }
}

template <class RigidTypes, class GeometricalTypes>
template <class T>
void DiagonalMass<RigidTypes, GeometricalTypes>::initRigidImpl()
{
    if(this->getContext()==nullptr)
    {
        dmsg_error(this) << "Calling the initRigidImpl function is only possible if the object has a valid associated context \n" ;
        this->d_componentState.setValue(ComponentState::Invalid) ;

        //return;
    }

    if(this->mstate == nullptr )
    {
        msg_error(this) << "DiagonalComponent can only be used on node with an associated '<MechanicalObject>' \n"
                           "To remove this warning you can: add a <MechanicalObject> to the node. \n" ;
        this->d_componentState.setValue(ComponentState::Invalid) ;

        //return;
    }

    if (l_topology.empty())
    {
        msg_warning() << "Link \"topology\" to the Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }
        
    if(!l_topology)
    {
        msg_error(this) << "Unable to retreive a valid MeshTopology component in the current context. \n"
                             "The component cannot be initialized and thus is de-activated. \n "
                             "To supress this warning you can add a Topology component in the parent node of'<"<< this->getName() <<">'.\n" ;
        this->d_componentState.setValue(ComponentState::Invalid) ;
    }
    else
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        if (l_geometryState.empty())
        {
            msg_warning() << "Link \"geometryState\" to the MechanicalObject associated with the geometry should be set to ensure right behavior. First container found from the topology context will be used.";
            sofa::core::behavior::BaseMechanicalState::SPtr baseState;
            l_topology->getContext()->get(baseState);

            if (baseState == nullptr)
            {
                msg_error() << "No compatible state associated with the topology has been found.";
                this->d_componentState.setValue(ComponentState::Invalid);
            }
            else
            {
                typename sofa::core::behavior::MechanicalState<GeometricalTypes>::SPtr geometryState = boost::dynamic_pointer_cast<sofa::core::behavior::MechanicalState<GeometricalTypes>>(baseState);
                if (geometryState == nullptr)
                {
                    msg_error() << "A state associated with the topology has been found but is incompatible with the definition of the mass (templates mismatch).";
                    this->d_componentState.setValue(ComponentState::Invalid);
                }
                else
                {
                    l_geometryState.set(geometryState);
                    msg_info() << "Topology is associated with the state: '" << l_geometryState->getPathName() << "'";
                }
            }
        }
    }

    if (!d_fileMass.getValue().empty())
        load(d_fileMass.getFullPath().c_str());
    Inherited::init();
    initTopologyHandlers();

    // Initialize the f_mass vector. The f_mass vector is enlarged to contains
    // as much as value as the 'mstate'. The new entries are initialized with the
    // last value of f_mass.
    if (!this->mstate && d_vertexMass.getValue().size() > 0 && d_vertexMass.getValue().size() < (unsigned)this->mstate->getSize())
    {
        MassVector &masses= *d_vertexMass.beginEdit();
        size_t i = masses.size()-1;
        size_t n = (size_t)this->mstate->getSize();
        while (masses.size() < n)
            masses.push_back(masses[i]);
        d_vertexMass.endEdit();
    }

    this->d_componentState.setValue(ComponentState::Valid) ;
}

template <class RigidTypes, class GeometricalTypes>
template <class T>
type::Vec6 DiagonalMass<RigidTypes, GeometricalTypes>::getMomentumRigid3Impl ( const MechanicalParams*,
                                                                    const DataVecCoord& vx,
                                                                    const DataVecDeriv& vv ) const
{
    ReadAccessor<DataVecDeriv> v = vv;
    ReadAccessor<DataVecCoord> x = vx;

    const MassVector &masses = d_vertexMass.getValue();

    type::Vec6 momentum;

    for ( unsigned int i=0 ; i<v.size() ; i++ )
    {
        typename RigidTypes::Vec3 linearMomentum = v[i].getLinear() * masses[i].mass;
        for( int j=0 ; j<3 ; ++j ) momentum[j] += linearMomentum[j];

        typename RigidTypes::Vec3 angularMomentum = cross( x[i].getCenter(), linearMomentum ) + ( masses[i].inertiaMassMatrix * v[i].getAngular() );
        for( int j=0 ; j<3 ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}

template <class Vec3Types, class GeometricalTypes >
template <class T>
type::Vec6 DiagonalMass<Vec3Types, GeometricalTypes>::getMomentumVec3Impl( const MechanicalParams*,
                                                                const DataVecCoord& vx,
                                                                const DataVecDeriv& vv ) const
{
    ReadAccessor<DataVecDeriv> v = vv;
    ReadAccessor<DataVecCoord> x = vx;

    const MassVector &masses = d_vertexMass.getValue();

    Vec6 momentum;

    for ( unsigned int i=0 ; i<v.size() ; i++ )
    {
        Deriv linearMomentum = v[i] * masses[i];
        for( int j=0 ; j<3 ; ++j ) momentum[j] += linearMomentum[j];

        Deriv angularMomentum = cross( x[i], linearMomentum );
        for( int j=0 ; j<3 ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}


template <>
SReal DiagonalMass<Rigid3Types>::getPotentialEnergy( const MechanicalParams* mparams,
                                                                   const DataVecCoord& x) const
{
    return getPotentialEnergyRigidImpl<Rigid3Types>(mparams, x) ;
}

template <>
SReal DiagonalMass<Rigid2Types>::getPotentialEnergy( const MechanicalParams* mparams,
                                                                   const DataVecCoord& x) const
{
    return getPotentialEnergyRigidImpl<Rigid2Types>(mparams, x) ;
}

template <>
void DiagonalMass<Rigid3Types>::draw(const VisualParams* vparams)
{
    drawRigid3dImpl<Rigid3Types>(vparams) ;
}

template <>
void DiagonalMass<Rigid3Types>::reinit()
{
    Inherited::reinit();
}

template <>
void DiagonalMass<Rigid2Types>::reinit()
{
    Inherited::reinit();
}

template <>
void DiagonalMass<Rigid3Types>::init()
{
    initRigidImpl<Rigid3Types>() ;
}

template <>
void DiagonalMass<Rigid2Types>::init()
{
    initRigidImpl<Rigid2Types>() ;
}

template <>
void DiagonalMass<Rigid2Types>::draw(const VisualParams* vparams)
{
    drawRigid2dImpl<Rigid2Types>(vparams);
}


template <>
type::Vec6 DiagonalMass<Vec3Types>::getMomentum ( const MechanicalParams* mparams,
                                                        const DataVecCoord& vx,
                                                        const DataVecDeriv& vv ) const
{
    return getMomentumVec3Impl<Vec3Types>(mparams, vx, vv) ;
}

template <>
type::Vec6 DiagonalMass<Rigid3Types>::getMomentum ( const MechanicalParams* mparams,
                                                              const DataVecCoord& vx,
                                                              const DataVecDeriv& vv ) const
{
    return getMomentumRigid3Impl<Rigid3Types>(mparams, vx, vv) ;
}

// Register in the Factory
int DiagonalMassClass = core::RegisterObject("Define a specific mass for each particle")
        .add< DiagonalMass<Vec3Types> >()
        .add< DiagonalMass<Vec2Types, Vec3Types> >()
        .add< DiagonalMass<Vec1Types> >()
        .add< DiagonalMass<Vec1Types, Vec2Types> >()
        .add< DiagonalMass<Vec1Types, Vec3Types> >()
        .add< DiagonalMass<Rigid3Types> >()
        .add< DiagonalMass<Rigid2Types> >()
        .add< DiagonalMass<Rigid2Types, Rigid3Types> >()

        ;

template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec3Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec2Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec2Types, Vec3Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec1Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec1Types, Vec2Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Vec1Types, Vec3Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Rigid3Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Rigid2Types>;
template class SOFA_COMPONENT_MASS_API DiagonalMass<Rigid2Types, Rigid3Types>;

} // namespace sofa::component::mass
