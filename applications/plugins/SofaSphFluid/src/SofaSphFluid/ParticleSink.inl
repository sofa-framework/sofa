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
#ifndef SOFA_COMPONENT_MISC_PARTICLESINK_INL
#define SOFA_COMPONENT_MISC_PARTICLESINK_INL

#include <SofaSphFluid/config.h>
#include <SofaSphFluid/ParticleSink.h>

namespace sofa
{

namespace component
{

namespace misc
{


template<class DataTypes>
ParticleSink<DataTypes>::ParticleSink()
    : d_planeNormal(initData(&d_planeNormal, "normal", "plane normal"))
    , d_planeD0(initData(&d_planeD0, (Real)0, "d0", "plane d coef at which particles acceleration is constrained to 0"))
    , d_planeD1(initData(&d_planeD1, (Real)0, "d1", "plane d coef at which particles are removed"))
    , d_showPlane(initData(&d_showPlane, false, "showPlane", "enable/disable drawing of plane"))
    , d_fixed(initData(&d_fixed, "fixed", "indices of fixed particles"))
    , l_topology(initLink("topology", "link to the topology container"))
    , m_topoModifier(nullptr)
{
    this->f_listening.setValue(true);
    Deriv n;
    DataTypes::set(n, 0, 1, 0);
    d_planeNormal.setValue(n);
}

template<class DataTypes>
ParticleSink<DataTypes>::~ParticleSink()
{

}


template<class DataTypes>
void ParticleSink<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();
    if (!this->mstate) return;

    // If no topology set, check if one is valid in the node
    if (l_topology.empty())
    {
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    sofa::core::topology::BaseMeshTopology* _topology = l_topology.get();

    if (_topology) // If topology is found, will check if it is dynamic. Otherwise only mechanical Object should be used.
    {
        _topology->getContext()->get(m_topoModifier);
        if (!m_topoModifier)
        {
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            msg_error() << "Topology component has been found in Node at: " << _topology->getName() << " but this topology is not dynamic. ParticleSink will not be able to remove points.";
            return;
        }

        // Initialize functions and parameters for topology data and handler
        d_fixed.createTopologyHandler(_topology);
    }

    msg_info() << "Normal=" << d_planeNormal.getValue() << " d0=" << d_planeD0.getValue() << " d1=" << d_planeD1.getValue();
}


template<class DataTypes>
void ParticleSink<DataTypes>::animateBegin(double /*dt*/, double time)
{
    if (!this->mstate) 
        return;
    
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecDeriv& v = this->mstate->read(core::ConstVecDerivId::velocity())->getValue();
    int n = int(x.size());
    type::vector<Index> remove;
    for (int i=n-1; i>=0; --i) // always remove points in reverse order
    {
        Real d = x[i]*d_planeNormal.getValue()-d_planeD1.getValue();
        if (d<0)
        {
            msg_info() << "SINK particle "<<i<<" time "<<time<<" position "<<x[i]<<" velocity "<<v[i] ;
            remove.push_back(i);
        }
    }
    if (!remove.empty())
    {
        if (m_topoModifier != nullptr)
        {
            msg_info() << "Remove: " << remove.size() << " out of: " << n <<" particles using PointSetTopologyModifier.";
            m_topoModifier->removePoints(remove);
        }
        else if(statecontainer::MechanicalObject<DataTypes>* object = dynamic_cast<statecontainer::MechanicalObject<DataTypes>*>(this->mstate.get()))
        {
            msg_info() << "Remove "<<remove.size()<<" particles using MechanicalObject.";
            // deleting the vertices
            for (unsigned int i = 0; i < remove.size(); ++i)
            {
                --n;
                object->replaceValue(n, remove[i] );
            }
            // resizing the state vectors
            this->mstate->resize(n);
        }
        else
        {
            msg_error() << "No external object supporting removing points!";
        }
    }
}

template<class DataTypes>
void ParticleSink<DataTypes>::projectResponse(const sofa::core::MechanicalParams* mparams, DataVecDeriv& dx)
{
    // Fixed points are directly done in projectVelocity
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(dx);
}

template<class DataTypes>
void ParticleSink<DataTypes>::projectVelocity(const sofa::core::MechanicalParams* mparams, DataVecDeriv&  v )
{
    SOFA_UNUSED(mparams);

    if (!this->mstate) return;

    VecDeriv& vel = *v.beginEdit();
    helper::ReadAccessor< Data<SetIndexArray> > _fixed = this->d_fixed;
    Deriv v0 = Deriv();
    for (unsigned int s = 0; s<_fixed.size(); s++)
    {
        vel[_fixed[s]] = v0;
    }    
    v.endEdit();
}


template<class DataTypes>
void ParticleSink<DataTypes>::projectPosition(const sofa::core::MechanicalParams* mparams, DataVecCoord& xData)
{
    SOFA_UNUSED(mparams);

    if (!this->mstate) return;

    VecCoord& x = *xData.beginEdit();

    helper::WriteAccessor< Data< SetIndexArray > > _fixed = d_fixed;

    _fixed.clear();
    // constraint the last value
    for (unsigned int i=0; i<x.size(); i++)
    {
        Real d = x[i]*d_planeNormal.getValue()-d_planeD0.getValue();
        if (d<0)
        {
            _fixed.push_back(i);
        }
    }

    xData.endEdit();
}


template<class DataTypes>
void ParticleSink<DataTypes>::projectJacobianMatrix(const sofa::core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(cData);
}


template<class DataTypes>
void ParticleSink<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if(simulation::AnimateBeginEvent::checkEventType(event) )
    {
        simulation::AnimateBeginEvent* ev = static_cast<simulation::AnimateBeginEvent*>(event);
        animateBegin(ev->getDt(), this->getContext()->getTime());
    }
}

template<class DataTypes>
void ParticleSink<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!d_showPlane.getValue())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    type::Vec3d normal; normal = d_planeNormal.getValue();

    // find a first vector inside the plane
    type::Vec3d v1;
    if( 0.0 != normal[0] ) v1 = type::Vec3d(-normal[1]/normal[0], 1.0, 0.0);
    else if ( 0.0 != normal[1] ) v1 = type::Vec3d(1.0, -normal[0]/normal[1],0.0);
    else if ( 0.0 != normal[2] ) v1 = type::Vec3d(1.0, 0.0, -normal[0]/normal[2]);
    v1.normalize();
    // find a second vector inside the plane and orthogonal to the first
    type::Vec3d v2;
    v2 = v1.cross(normal);
    v2.normalize();
    const float size=1.0f;
    type::Vec3d center = normal*d_planeD0.getValue();
    type::Vec3d corners[4];
    corners[0] = center-v1*size-v2*size;
    corners[1] = center+v1*size-v2*size;
    corners[2] = center+v1*size+v2*size;
    corners[3] = center-v1*size+v2*size;

    vparams->drawTool()->disableLighting();
    vparams->drawTool()->setPolygonMode(0, true);

    std::vector<sofa::type::Vec3> vertices;

    vertices.push_back(sofa::type::Vec3(corners[0]));
    vertices.push_back(sofa::type::Vec3(corners[1]));
    vertices.push_back(sofa::type::Vec3(corners[2]));
    vertices.push_back(sofa::type::Vec3(corners[3]));
    vparams->drawTool()->drawQuad(vertices[0],vertices[1],vertices[2],vertices[3], cross((vertices[1] - vertices[0]), (vertices[2] - vertices[0])), sofa::type::RGBAColor(0.0f, 0.5f, 0.2f, 1.0f));
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_PARTICLESINK_INL

