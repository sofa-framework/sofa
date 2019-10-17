/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
    , d_canHaveEmptyVector(initData(&d_canHaveEmptyVector, (bool)false, "canHaveEmptyVector", ""))
    , m_numberParticles(0)
    , m_lastparticles(initData(&m_lastparticles, "lastparticles", "lastparticles indices"))
{
    this->f_listening.setValue(true);
    d_center.beginEdit()->push_back(Coord()); d_center.endEdit();

    m_pointHandler = new PSPointHandler(this, &m_lastparticles);
}


template<class DataTypes>
ParticleSource<DataTypes>::~ParticleSource()
{
    if (m_pointHandler)
        delete m_pointHandler;
}


template<class DataTypes>
void ParticleSource<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();
    if (!this->mstate) {
        //sofa::core::objectmodel::ComponentState::d_componentstate.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    m_numberParticles = d_center.getValue().size();
    m_lastTime = d_start.getValue() - d_delay.getValue();
    m_maxdist = 0;
    //lastparticle = -1;
    m_lastpos.resize(m_numberParticles);

    sofa::core::topology::BaseMeshTopology* _topology = this->getContext()->getMeshTopology();
    if (_topology != nullptr)
    {
        m_lastparticles.createTopologicalEngine(_topology, m_pointHandler);
        m_lastparticles.registerTopologicalData();
    }

    msg_info() << "ParticleSource: center = " << d_center.getValue()
        << " radius = " << d_radius.getValue() << " delay = " << d_delay.getValue()
        << " start = " << d_start.getValue() << " stop = " << d_stop.getValue();
    /*
    int i0 = mstate->getSize();

    if (!d_canHaveEmptyVector.getValue())
    {
    // ignore the first point if it is the only one
    if (i0 == 1)
    i0 = 0;
    }

    int ntotal = i0 + ((int)((d_stop.getValue() - d_start.getValue() - d_delay.getValue()) / d_delay.getValue())) * N;

    if (ntotal > 0)
    {
    this->mstate->resize(ntotal);
    if (!d_canHaveEmptyVector.getValue())
    this->mstate->resize((i0==0) ? 1 : i0);
    else
    this->mstate->resize(i0);
    }
    */
}


template<class DataTypes>
void ParticleSource<DataTypes>::reset()
{
    this->mstate->resize(1);
    m_lastTime = d_start.getValue() - d_delay.getValue();
    m_maxdist = 0;

    helper::WriteAccessor<Data<VecIndex> > _lastparticles = this->m_lastparticles; ///< lastparticles indices
    _lastparticles.clear();
    m_lastpos.clear();
}


template<class DataTypes>
void ParticleSource<DataTypes>::projectResponse(const sofa::core::MechanicalParams* mparams, DataVecDeriv& dxData)
{
    VecDeriv& dx = *dxData.beginEdit(mparams);
    projectResponseT(dx);
    dxData.endEdit(mparams);
}


template<class DataTypes>
void ParticleSource<DataTypes>::projectPosition(const sofa::core::MechanicalParams* mparams, DataVecCoord& xData)
{
    if (!this->mstate) return;
    if (m_lastparticles.getValue().empty()) return;

    VecCoord& x = *xData.beginEdit(mparams);

    double time = this->getContext()->getTime();
    if (time < d_start.getValue() || time > d_stop.getValue()) return;
    Deriv dpos = d_velocity.getValue()*(time - m_lastTime);

    helper::ReadAccessor<Data<VecIndex> > _lastparticles = this->m_lastparticles; ///< lastparticles indices
                                                                                // constraint the most recent particles
    for (unsigned int s = 0; s < _lastparticles.size(); s++)
    {
        //HACK: TODO understand why these conditions can be reached
        if (s >= m_lastpos.size() || _lastparticles[s] >= x.size()) continue;
        x[_lastparticles[s]] = m_lastpos[s];
        x[_lastparticles[s]] += dpos; // account for particle initial motion
    }
    xData.endEdit(mparams);
}


template<class DataTypes>
void ParticleSource<DataTypes>::projectVelocity(const sofa::core::MechanicalParams* mparams, DataVecDeriv&  vData)
{
    if (!this->mstate) return;
    if (m_lastparticles.getValue().empty()) return;

    VecDeriv& res = *vData.beginEdit(mparams);

    double time = this->getContext()->getTime();
    if (time < d_start.getValue() || time > d_stop.getValue()) return;
    // constraint the most recent particles
    Deriv v0 = d_velocity.getValue();

    helper::ReadAccessor<Data<VecIndex> > _lastparticles = this->m_lastparticles; ///< lastparticles indices
    for (unsigned int s = 0; s<_lastparticles.size(); s++)
    {
        //HACK: TODO understand why these conditions can be reached
        if (_lastparticles[s] >= res.size()) continue;
        res[_lastparticles[s]] = v0;
    }
    vData.endEdit(mparams);
}


template<class DataTypes>
void ParticleSource<DataTypes>::animateBegin(double /*dt*/, double time)
{
    //msg_info() << "ParticleSource: animate begin time="<<time<<sendl;
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

    int i0 = this->mstate->getSize();

    if (!d_canHaveEmptyVector.getValue())
    {
        // ignore the first point if it is the only one
        if (i0 == 1)
            i0 = 0;
    }

    int nbParticlesToCreate = (int)((time - m_lastTime) / d_delay.getValue());

    if (nbParticlesToCreate > 0)
    {
        helper::WriteAccessor<Data<VecIndex> > _lastparticles = this->m_lastparticles; ///< lastparticles indices

        helper::vector< Coord > newX;
        helper::vector< Deriv > newV;

        newX.reserve(nbParticlesToCreate * m_numberParticles);
        newV.reserve(nbParticlesToCreate * m_numberParticles);
        const Deriv v0 = d_velocity.getValue();
        for (int i = 0; i < nbParticlesToCreate; i++)
        {
            m_lastTime += d_delay.getValue();
            m_maxdist += d_delay.getValue() * d_velocity.getValue().norm() / d_scale.getValue();

            //int lastparticle = i0 + i * N;

            int lp0 = _lastparticles.empty() ? 0 : _lastparticles.size() / 2;
            if (lp0 > 0)
            {
                int shift = _lastparticles.size() - lp0;
                Deriv dpos = v0 * d_delay.getValue();
                for (int s = 0; s < lp0; s++)
                {
                    _lastparticles[s] = _lastparticles[s + shift];
                    m_lastpos[s] = m_lastpos[s + shift] + dpos;
                }
            }

            _lastparticles.resize(lp0);
            m_lastpos.resize(lp0);

            for (int s = 0; s < m_numberParticles; s++)
            {
                if (d_center.getValue()[s].norm() > m_maxdist) continue;
                Coord p = d_center.getValue()[s] * d_scale.getValue() + d_translation.getValue();

                for (unsigned int c = 0; c < p.size(); c++)
                    p[c] += d_radius.getValue()[c] * rrand();
                m_lastpos.push_back(p);
                _lastparticles.push_back(i0 + newX.size());
                newX.push_back(p + v0 * (time - m_lastTime)); // account for particle initial motion
                newV.push_back(v0);
            }
        }

        nbParticlesToCreate = newX.size();
        msg_info() << "ParticleSource: Creating " << nbParticlesToCreate << " particles, total " << i0 + nbParticlesToCreate << " particles.";

        sofa::component::topology::PointSetTopologyModifier* pointMod;
        this->getContext()->get(pointMod);

        // Particles creation.
        if (pointMod != nullptr)
        {
            int n = i0 + nbParticlesToCreate - this->mstate->getSize();

            pointMod->addPointsWarning(n);
            pointMod->addPointsProcess(n);
            pointMod->propagateTopologicalChanges();
        }
        else
        {
            this->mstate->resize(i0 + nbParticlesToCreate);
        }

        //VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
        helper::WriteAccessor< Data<VecCoord> > x = *this->mstate->write(core::VecCoordId::position());
        helper::WriteAccessor< Data<VecDeriv> > v = *this->mstate->write(core::VecDerivId::velocity());
        for (int s = 0; s < nbParticlesToCreate; ++s)
        {
            x[i0 + s] = newX[s];
            v[i0 + s] = newV[s];
        }
    }
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
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!this->mstate) return;
    if (m_lastparticles.getValue().empty()) return;
    double time = this->getContext()->getTime();
    if (time < d_start.getValue() || time > d_stop.getValue()) return;
    Deriv dpos = d_velocity.getValue()*(time - m_lastTime);

    std::vector< sofa::defaulttype::Vector3 > points;
    for (unsigned int s = 0; s < m_lastpos.size(); s++)
    {
        sofa::defaulttype::Vector3 point;
        point = DataTypes::getCPos(m_lastpos[s] + dpos);
        points.push_back(point);
    }
    vparams->drawTool()->drawPoints(points, 10, sofa::defaulttype::Vec<4, float>(1, 0.5, 0.5, 1));
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_PARTICLESOURCE_INL

