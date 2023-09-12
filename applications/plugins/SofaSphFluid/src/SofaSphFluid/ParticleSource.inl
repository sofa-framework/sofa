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
#ifndef SOFA_COMPONENT_MISC_PARTICLESOURCE_INL
#define SOFA_COMPONENT_MISC_PARTICLESOURCE_INL
#include <SofaSphFluid/config.h>

#include <SofaSphFluid/ParticleSource.h>
#include <sofa/component/topology/container/dynamic/PointSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/PointSetTopologyModifier.h>
#include <sofa/simulation/AnimateBeginEvent.h>

namespace sofa
{

namespace component
{

namespace misc
{

template<class DataTypes>
ParticleSource<DataTypes>::ParticleSource()
    : d_translation(initData(&d_translation, Coord(), "translation", "translation applied to center(s)"))
    , d_scale(initData(&d_scale, (Real)1.0, "scale", "scale applied to center(s)"))
    , d_center(initData(&d_center, "center", "Source center(s)"))
    , d_radius(initData(&d_radius, Coord(), "radius", "Source radius"))
    , d_velocity(initData(&d_velocity, Deriv(), "velocity", "Particle initial velocity"))
    , d_delay(initData(&d_delay, (Real)0.01, "delay", "Delay between particles creation"))
    , d_start(initData(&d_start, (Real)0, "start", "Source starting time"))
    , d_stop(initData(&d_stop, (Real)1e10, "stop", "Source stopping time"))
    , m_numberParticles(0)
    , m_lastparticles(initData(&m_lastparticles, "lastparticles", "lastparticles indices"))
{
    this->f_listening.setValue(true);
    d_center.beginEdit()->push_back(Coord()); d_center.endEdit();
}


template<class DataTypes>
ParticleSource<DataTypes>::~ParticleSource()
{

}


template<class DataTypes>
void ParticleSource<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();
    if (!this->mstate) {
        //sofa::core::objectmodel::ComponentState::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    m_numberParticles = d_center.getValue().size();
    m_lastTime = d_start.getValue() - d_delay.getValue();
    if (m_lastTime < 0.0)
        m_lastTime = 0.0;

    m_maxdist = 0;
    m_lastpos.resize(m_numberParticles);

    sofa::core::topology::BaseMeshTopology* _topology = this->getContext()->getMeshTopology();
    if (_topology != nullptr)
    {
        m_lastparticles.createTopologyHandler(_topology);
        m_lastparticles.setDestructionCallback([this](Index pointIndex, Index& val)
        {
            SOFA_UNUSED(val);
            m_lastpos[pointIndex] = m_lastpos[m_lastpos.size() - 1];
            m_lastpos.pop_back();
        });
    }

    msg_info() << " center = " << d_center.getValue() << msgendl
               << " radius = " << d_radius.getValue() << " m_numberParticles = " << m_numberParticles << msgendl
               << " start = " << d_start.getValue() << " stop = " << d_stop.getValue() << " delay = " << d_delay.getValue();
}


template<class DataTypes>
void ParticleSource<DataTypes>::reset()
{
    this->mstate->resize(1);
    m_lastTime = d_start.getValue() - d_delay.getValue();
    if (m_lastTime < 0.0)
        m_lastTime = 0.0;
    m_maxdist = 0;

    SetIndexArray& _lastparticles = *this->m_lastparticles.beginEdit();
    _lastparticles.clear();
    m_lastpos.clear();
    this->m_lastparticles.endEdit();
}


template<class DataTypes>
void ParticleSource<DataTypes>::projectResponse(const sofa::core::MechanicalParams* mparams, DataVecDeriv& dxData)
{
    SOFA_UNUSED(mparams);

    if (!this->mstate || m_lastparticles.getValue().empty()) {
        return;
    }
    
    double time = this->getContext()->getTime(); // nothing to do yet or anymore
    if (time < d_start.getValue() || time > d_stop.getValue()) {
        return;
    }

    VecDeriv& dx = *dxData.beginEdit();
    const SetIndexArray& _lastparticles = this->m_lastparticles.getValue();
    for (unsigned int s = 0; s<_lastparticles.size(); s++)
    {
        dx[_lastparticles[s]] = Deriv();
    }
    dxData.endEdit();
}


template<class DataTypes>
void ParticleSource<DataTypes>::projectPosition(const sofa::core::MechanicalParams* mparams, DataVecCoord& xData)
{
    SOFA_UNUSED(mparams);

    if (!this->mstate || m_lastparticles.getValue().empty()) {
        return;
    }
    
    double time = this->getContext()->getTime(); // nothing to do yet or anymore
    if (time < d_start.getValue() || time > d_stop.getValue()) {
        return;
    }

    // constraint the most recent particles
    VecCoord& x = *xData.beginEdit();
    Deriv dpos = d_velocity.getValue()*(time - m_lastTime);
    const SetIndexArray& _lastparticles = this->m_lastparticles.getValue();
    msg_info() << "projectPosition: " << _lastparticles;
    for (unsigned int s = 0; s < _lastparticles.size(); s++)
    {
        x[_lastparticles[s]] = m_lastpos[s];
        x[_lastparticles[s]] += dpos; // account for particle initial motion
    }
    xData.endEdit();
}


template<class DataTypes>
void ParticleSource<DataTypes>::projectVelocity(const sofa::core::MechanicalParams* mparams, DataVecDeriv&  vData)
{
    SOFA_UNUSED(mparams);

    if (!this->mstate || m_lastparticles.getValue().empty()) {
        return;
    }

    double time = this->getContext()->getTime(); // nothing to do yet or anymore
    if (time < d_start.getValue() || time > d_stop.getValue()) {
        return;
    }
    
    // constraint the most recent particles with the initial Velocity
    VecDeriv& res = *vData.beginEdit();
    Deriv v0 = d_velocity.getValue();
    const SetIndexArray& _lastparticles = this->m_lastparticles.getValue();
    for (unsigned int s = 0; s<_lastparticles.size(); s++)
    {
        res[_lastparticles[s]] = v0;
    }
    vData.endEdit();
}


template<class DataTypes>
void ParticleSource<DataTypes>::animateBegin(double /*dt*/, double time)
{    
    if (!this->mstate)
        return;

    if (time < d_start.getValue() || time > d_stop.getValue())
    {
        if (time > d_stop.getValue() && time - this->getContext()->getDt() <= d_stop.getValue())
        {
            msg_info() << "Source stopped, current number of particles : " << this->mstate->getSize();
        }
        return;
    }

    size_t i0 = this->mstate->getSize();
    if (i0 == 1) // ignore the first point if it is the only one
    {
        i0 = 0;
        sofa::component::topology::container::dynamic::PointSetTopologyContainer* pointCon;
        this->getContext()->get(pointCon);
        if (pointCon != nullptr)
        {
            // TODO: epernod.... why why why.... still a diff between meca->getSize and topo->GetNbrPoints...
            pointCon->setNbPoints(1);
        }
    }

    size_t nbParticlesToCreate = (int)((time - m_lastTime) / d_delay.getValue());
    if (nbParticlesToCreate > 0)
    {
        msg_info() << "ParticleSource: animate begin time= " << time << " | size: " << i0;
        msg_info() << "nbParticlesToCreate: " << nbParticlesToCreate << " m_maxdist: " << m_maxdist;
        SetIndexArray& _lastparticles = *m_lastparticles.beginEdit(); ///< lastparticles indices

        type::vector< Coord > newX;
        type::vector< Deriv > newV;

        newX.reserve(nbParticlesToCreate * m_numberParticles);
        newV.reserve(nbParticlesToCreate * m_numberParticles);
        const Deriv v0 = d_velocity.getValue();
        for (size_t i = 0; i < nbParticlesToCreate; i++)
        {
            m_lastTime += d_delay.getValue();
            m_maxdist += d_delay.getValue() * d_velocity.getValue().norm() / d_scale.getValue();

            //int lastparticle = i0 + i * N;
            size_t lp0 = _lastparticles.empty() ? 0 : _lastparticles.size() / 2;
            if (lp0 > 0)
            {
                size_t shift = _lastparticles.size() - lp0;
                Deriv dpos = v0 * d_delay.getValue();
                for (size_t s = 0; s < lp0; s++)
                {
                    _lastparticles[s] = _lastparticles[s + shift];
                    m_lastpos[s] = m_lastpos[s + shift] + dpos;
                }
            }

            _lastparticles.resize(lp0);
            m_lastpos.resize(lp0);

            for (size_t s = 0; s < m_numberParticles; s++)
            {
                Coord p = d_center.getValue()[s] * d_scale.getValue() + d_translation.getValue();

                for (unsigned int c = 0; c < p.size(); c++)
                    p[c] += d_radius.getValue()[c] * rrand();

                m_lastpos.push_back(p);
                _lastparticles.push_back((Index)(i0 + newX.size()));
                newX.push_back(p + v0 * (time - m_lastTime)); // account for particle initial motion
                newV.push_back(v0);
            }
        }

        nbParticlesToCreate = newX.size();
        msg_info() << "ParticleSource: Creating " << nbParticlesToCreate << " particles, total " << i0 + nbParticlesToCreate << " particles.";

        if (nbParticlesToCreate <= 0)
            return;

        sofa::component::topology::container::dynamic::PointSetTopologyModifier* pointMod;
        this->getContext()->get(pointMod);
        
        // Particles creation.
        if (pointMod != nullptr)
        {
            size_t n = i0 + nbParticlesToCreate;
            if (n < this->mstate->getSize())
            {
                msg_error() << "Less particle to create than the number of dof in the current mstate: " << n << " vs " << this->mstate->getSize();
                n = 0;
            }
            else
            {
                n -= this->mstate->getSize();
            }
            pointMod->addPoints(n);
        }
        else
        {
            this->mstate->resize(i0 + nbParticlesToCreate);
        }

        helper::WriteAccessor< Data<VecCoord> > x = *this->mstate->write(core::VecCoordId::position());
        helper::WriteAccessor< Data<VecDeriv> > v = *this->mstate->write(core::VecDerivId::velocity());
        for (size_t s = 0; s < nbParticlesToCreate; ++s)
        {
            x[i0 + s] = newX[s];
            v[i0 + s] = newV[s];
        }
    }

    m_lastparticles.endEdit();
}


template<class DataTypes>
void ParticleSource<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        simulation::AnimateBeginEvent* ev = static_cast<simulation::AnimateBeginEvent*>(event);
        animateBegin(ev->getDt(), this->getContext()->getTime());
    }
}


template<class DataTypes>
void ParticleSource<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    if (!this->mstate || m_lastparticles.getValue().empty())
        return;

    double time = this->getContext()->getTime();
    if (time < d_start.getValue() || time > d_stop.getValue())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    Deriv dpos = d_velocity.getValue()*(time - m_lastTime);

    std::vector< sofa::type::Vec3 > pointsInit;
    for (unsigned int s = 0; s < m_lastpos.size(); s++)
    {
        sofa::type::Vec3 point;
        point = DataTypes::getCPos(m_lastpos[s] + dpos);
        pointsInit.push_back(point);
    }
    vparams->drawTool()->drawPoints(pointsInit, 10, sofa::type::RGBAColor(1., 0.5, 0.5, 1.));
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_PARTICLESOURCE_INL

