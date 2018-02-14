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
//
// C++ Interface: ParticleSource
//
// Description:
//
//
// Author: Jeremie Allard, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_MISC_PARTICLESOURCE_H
#define SOFA_COMPONENT_MISC_PARTICLESOURCE_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <SofaBaseTopology/TopologySubsetData.inl>
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <sofa/core/topology/TopologyChange.h>
#include <vector>
#include <iterator>
#include <iostream>
#include <ostream>
#include <algorithm>

namespace sofa
{

namespace component
{

namespace misc
{

template<class TDataTypes>
class ParticleSource : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParticleSource,TDataTypes), SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef helper::vector<Real> VecDensity;

    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalModel;

    Data< Coord > f_translation; ///< translation applied to center(s)
    Data< Real > f_scale; ///< scale applied to center(s)
    Data< helper::vector<Coord> > f_center; ///< Source center(s)
    Data< Coord > f_radius; ///< Source radius
    Data< Deriv > f_velocity; ///< Particle initial velocity
    Data< Real > f_delay; ///< Delay between particles creation
    Data< Real > f_start; ///< Source starting time
    Data< Real > f_stop; ///< Source stopping time
    Data< bool > f_canHaveEmptyVector;

    ParticleSource()
        : f_translation(initData(&f_translation,Coord(),"translation","translation applied to center(s)") )
        , f_scale(initData(&f_scale,(Real)1.0,"scale","scale applied to center(s)") )
        , f_center(initData(&f_center, "center","Source center(s)") )
        , f_radius(initData(&f_radius, Coord(), "radius", "Source radius"))
        , f_velocity(initData(&f_velocity, Deriv(), "velocity", "Particle initial velocity"))
        , f_delay(initData(&f_delay, (Real)0.01, "delay", "Delay between particles creation"))
        , f_start(initData(&f_start, (Real)0, "start", "Source starting time"))
        , f_stop(initData(&f_stop, (Real)1e10, "stop", "Source stopping time"))
        , f_canHaveEmptyVector(initData(&f_canHaveEmptyVector, (bool)false, "canHaveEmptyVector", ""))
        , N(0)
        , lastparticles(initData(&lastparticles, "lastparticles", "lastparticles indices"))
    {
        this->f_listening.setValue(true);
        f_center.beginEdit()->push_back(Coord()); f_center.endEdit();

        pointHandler = new PSPointHandler(this, &lastparticles);
    }

    virtual ~ParticleSource()
    {
        if (pointHandler)
            delete pointHandler;
    }

    int N;
    Real lasttime;
    Real maxdist;
    //int lastparticle;
    typedef typename VecCoord::template rebind<unsigned int>::other VecIndex;
    sofa::component::topology::PointSubsetData< VecIndex > lastparticles; ///< lastparticles indices
    VecCoord lastpos;

    class PSPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, VecIndex >
    {
    public:
        typedef typename ParticleSource<DataTypes>::VecIndex VecIndex;
        typedef VecIndex container_type;
        typedef typename container_type::value_type value_type;

        PSPointHandler(ParticleSource<DataTypes>* _ps, sofa::component::topology::PointSubsetData<VecIndex >* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, VecIndex >(_data), ps(_ps) {}

        void applyDestroyFunction(unsigned int index, value_type& /*T*/)
        {
            dmsg_info("ParticleSource") << "PSRemovalFunction";
            if(ps)
            {
                /*topology::PointSubset::const_iterator it = std::find(ps->lastparticles.begin(),ps->lastparticles.end(), (unsigned int)index);
                 if (it != ps->lastparticles.end())
                 {
                    ps->lastpos.erase( ps->lastpos.begin()+(it-ps->lastparticles.begin()) );
                    //ps->lastparticles.getArray().erase(it);
                     helper::removeValue(ps->lastparticles,(unsigned int)index);
                 }*/
                VecIndex& _lastparticles = *ps->lastparticles.beginEdit();

                unsigned int size = _lastparticles.size();
                for (unsigned int i = 0; i < size; ++i)
                {
                    if ((unsigned int)_lastparticles[i] == index)
                    {
                        if (i < size-1)
                        {
                            _lastparticles[i] = _lastparticles[size-1];
                            ps->lastpos[i] = ps->lastpos[size-1];
                        }
                        _lastparticles.pop_back();
                        ps->lastpos.pop_back();
                        return;
                    }
                }
                ps->lastparticles.endEdit();
            }
        }


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/) {return false;}

    protected:
        ParticleSource<DataTypes> *ps;
    };



    virtual void init() override
    {
        this->core::behavior::ProjectiveConstraintSet<TDataTypes>::init();
        if (!this->mstate) return;
        N = f_center.getValue().size();
        lasttime = f_start.getValue() - f_delay.getValue();
        maxdist = 0;
        //lastparticle = -1;
        lastpos.resize(N);

        sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();

        lastparticles.createTopologicalEngine(topology, pointHandler);
        lastparticles.registerTopologicalData();

        sout << "ParticleSource: center = " << f_center.getValue()
                << " radius = " << f_radius.getValue() << " delay = " << f_delay.getValue()
                << " start = " << f_start.getValue() << " stop = " << f_stop.getValue() << sendl;
        /*
                int i0 = mstate->getSize();

                if (!f_canHaveEmptyVector.getValue())
                {
                    // ignore the first point if it is the only one
                    if (i0 == 1)
                        i0 = 0;
                }

                int ntotal = i0 + ((int)((f_stop.getValue() - f_start.getValue() - f_delay.getValue()) / f_delay.getValue())) * N;

                if (ntotal > 0)
                {
                    this->mstate->resize(ntotal);
                    if (!f_canHaveEmptyVector.getValue())
                        this->mstate->resize((i0==0) ? 1 : i0);
                    else
                        this->mstate->resize(i0);
                }
        */
    }

    virtual void reset() override
    {
        this->mstate->resize(1);
        lasttime = f_start.getValue()-f_delay.getValue();
        maxdist = 0;

        helper::WriteAccessor<Data<VecIndex> > _lastparticles = this->lastparticles; ///< lastparticles indices
        _lastparticles.clear();
        lastpos.clear();
    }

    Real rrand()
    {
        return (Real)(rand()*1.0/RAND_MAX);
    }

    virtual void animateBegin(double /*dt*/, double time)
    {
        //sout << "ParticleSource: animate begin time="<<time<<sendl;
        if (!this->mstate)
            return;

        if (time < f_start.getValue() || time > f_stop.getValue())
        {
            if (time > f_stop.getValue() && time-this->getContext()->getDt() <= f_stop.getValue())
            {
                sout << "Source stopped, current number of particles : " << this->mstate->getSize() << sendl;
            }
            return;
        }

        int i0 = this->mstate->getSize();

        if (!f_canHaveEmptyVector.getValue())
        {
            // ignore the first point if it is the only one
            if (i0 == 1)
                i0 = 0;
        }

        int nbParticlesToCreate = (int)((time - lasttime) / f_delay.getValue());

        if (nbParticlesToCreate > 0)
        {
            helper::WriteAccessor<Data<VecIndex> > _lastparticles = this->lastparticles; ///< lastparticles indices

            helper::vector< Coord > newX;
            helper::vector< Deriv > newV;

            newX.reserve(nbParticlesToCreate * N);
            newV.reserve(nbParticlesToCreate * N);
            const Deriv v0 = f_velocity.getValue();
            for (int i = 0; i < nbParticlesToCreate; i++)
            {
                lasttime += f_delay.getValue();
                maxdist += f_delay.getValue() * f_velocity.getValue().norm() / f_scale.getValue();

                //int lastparticle = i0 + i * N;

                int lp0 = _lastparticles.empty() ? 0 : _lastparticles.size()/2;
                if (lp0 > 0)
                {
                    int shift = _lastparticles.size() - lp0;
                    Deriv dpos = v0*f_delay.getValue();
                    for (int s = 0; s < lp0; s++)
                    {
                        _lastparticles[s] = _lastparticles[s+shift];
                        lastpos[s] = lastpos[s+shift] + dpos;
                    }
                }

                _lastparticles.resize(lp0);
                lastpos.resize(lp0);

                for (int s = 0; s < N; s++)
                {
                    if (f_center.getValue()[s].norm() > maxdist) continue;
                    Coord p = f_center.getValue()[s] * f_scale.getValue() + f_translation.getValue();

                    for (unsigned int c = 0; c < p.size(); c++)
                        p[c] += f_radius.getValue()[c] * rrand();
                    lastpos.push_back(p);
                    _lastparticles.push_back(i0 + newX.size());
                    newX.push_back(p + v0 * (time - lasttime)); // account for particle initial motion
                    newV.push_back(v0);
                }
            }

            nbParticlesToCreate = newX.size();
            sout << "ParticleSource: Creating "<< nbParticlesToCreate << " particles, total " << i0 + nbParticlesToCreate << " particles." << sendl;

            sofa::component::topology::PointSetTopologyModifier* pointMod;
            this->getContext()->get(pointMod);

            // Particles creation.
            if (pointMod != NULL)
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
                x[i0+s] = newX[s];
                v[i0+s] = newV[s];
            }
        }
    }

    /*
    /// Handle topological changes
    void handleTopologyChange()
    {
        sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();
        std::list<const sofa::core::topology::TopologyChange *>::const_iterator itBegin=topology->beginChange();
        std::list<const sofa::core::topology::TopologyChange *>::const_iterator itEnd=topology->endChange();
        if (itBegin != itEnd)
        {
            if (notMuted())
            {
                sout << "ParticleSource: handleTopologyChange()"<< sendl;
                sout << "lastparticles = ";
                std::copy(lastparticles.begin(),lastparticles.end(),std::ostream_iterator<unsigned int>(sout," "));
                sout << sendl;
            }
            int s1 = lastparticles.size();

            lastparticles.handleTopologyEvents(itBegin, itEnd, this->mstate->getSize());

            int s2 = lastparticles.size();
            if (s2 > s1) sout << "ParticleSource: handleTopologyChange(): " << s2-s1 << " points added!" << sendl;
            if (s2 < s1) sout << "ParticleSource: handleTopologyChange(): " << s1-s2 << " points removed!" << sendl;
            if (notMuted())
            {
                sout << "NEW lastparticles = ";
                std::copy(lastparticles.begin(),lastparticles.end(),std::ostream_iterator<unsigned int>(sout," "));
                sout << sendl;
            }
        }
    }
    */

    template <class DataDeriv>
    void projectResponseT(DataDeriv& res) ///< project dx to constrained space
    {
        if (!this->mstate) return;
        if (lastparticles.getValue().empty()) return;
        //sout << "ParticleSource: projectResponse of last particle ("<<lastparticle<<")."<<sendl;
        double time = this->getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;

        helper::ReadAccessor<Data<VecIndex> > _lastparticles = this->lastparticles; ///< lastparticles indices
        // constraint the last value
        for (unsigned int s=0; s<_lastparticles.size(); s++)
        {
            //HACK: TODO understand why these conditions can be reached
            if (_lastparticles[s] >= (unsigned int) this->mstate->getSize()) continue;

            res[_lastparticles[s]] = Deriv();
        }
    }

    using core::behavior::ProjectiveConstraintSet<TDataTypes>::projectResponse;
    void projectResponse(VecDeriv& dx)
    {
        projectResponseT(dx);
    }

    virtual void projectResponse(const sofa::core::MechanicalParams* mparams, DataVecDeriv& dxData) override ///< project dx to constrained space
    {
        VecDeriv& dx = *dxData.beginEdit(mparams);
        projectResponseT(dx);
        dxData.endEdit(mparams);
    }

/*
    void projectResponse(MatrixDerivRowType& dx)
    {
        projectResponseT(dx);
    }
*/

    virtual void projectVelocity(const sofa::core::MechanicalParams* mparams, DataVecDeriv&  vData ) override ///< project dx to constrained space (dx models a velocity) override
    {
        if (!this->mstate) return;
        if (lastparticles.getValue().empty()) return;

        VecDeriv& res = *vData.beginEdit(mparams);

        double time = this->getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        // constraint the most recent particles
        Deriv v0 = f_velocity.getValue();

        helper::ReadAccessor<Data<VecIndex> > _lastparticles = this->lastparticles; ///< lastparticles indices
        for (unsigned int s=0; s<_lastparticles.size(); s++)
        {
            //HACK: TODO understand why these conditions can be reached
            if ( _lastparticles[s] >= res.size() ) continue;
            res[_lastparticles[s]] = v0;
        }
        vData.endEdit(mparams);
    }

    virtual void projectPosition(const sofa::core::MechanicalParams* mparams, DataVecCoord& xData) override ///< project x to constrained space (x models a position) override
    {
        if (!this->mstate) return;
        if (lastparticles.getValue().empty()) return;

        VecCoord& x = *xData.beginEdit(mparams);

        double time = this->getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        Deriv dpos = f_velocity.getValue()*(time - lasttime);

        helper::ReadAccessor<Data<VecIndex> > _lastparticles = this->lastparticles; ///< lastparticles indices
        // constraint the most recent particles
        for (unsigned int s = 0; s < _lastparticles.size(); s++)
        {
            //HACK: TODO understand why these conditions can be reached
            if (s >= lastpos.size() || _lastparticles[s] >= x.size()) continue;
            x[_lastparticles[s]] = lastpos[s];
            x[_lastparticles[s]] += dpos; // account for particle initial motion
        }
        xData.endEdit(mparams);
    }

    virtual void projectJacobianMatrix(const sofa::core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /* cData */) override
    {

    }

    virtual void animateEnd(double /*dt*/, double /*time*/)
    {

    }

    virtual void handleEvent(sofa::core::objectmodel::Event* event) override
    {
        if(simulation::AnimateBeginEvent::checkEventType(event) )
        {
            simulation::AnimateBeginEvent* ev = static_cast<simulation::AnimateBeginEvent*>(event);
            animateBegin(ev->getDt(), this->getContext()->getTime());
        }
        else if(simulation::AnimateEndEvent::checkEventType(event) )
        {
            simulation::AnimateEndEvent* ev = static_cast<simulation::AnimateEndEvent*>(event);
            animateEnd(ev->getDt(), this->getContext()->getTime());
        }
    }


    void draw(const core::visual::VisualParams* vparams) override
    {
        if (!vparams->displayFlags().getShowBehaviorModels()) return;
        if (!this->mstate) return;
        if (lastparticles.getValue().empty()) return;
        double time = this->getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        Deriv dpos = f_velocity.getValue()*(time - lasttime);

        std::vector< sofa::defaulttype::Vector3 > points;
        for (unsigned int s = 0; s < lastpos.size(); s++)
        {
            sofa::defaulttype::Vector3 point;
            point = DataTypes::getCPos(lastpos[s]+dpos);
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, sofa::defaulttype::Vec<4,float>(1,0.5,0.5,1));
    }

protected:
    PSPointHandler* pointHandler;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MISC_PARTICLESOURCE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_SPH_FLUID_API ParticleSource<defaulttype::Vec3dTypes>;
extern template class SOFA_SPH_FLUID_API ParticleSource<defaulttype::Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_SPH_FLUID_API ParticleSource<defaulttype::Vec3fTypes>;
extern template class SOFA_SPH_FLUID_API ParticleSource<defaulttype::Vec2fTypes>;
#endif
#endif

} // namespace misc

} // namespace component

} // namespace sofa

#endif

