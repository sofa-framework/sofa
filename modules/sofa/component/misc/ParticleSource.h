/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/core/componentmodel/behavior/Constraint.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/component/topology/PointSetTopology.h>
#include <vector>
#include <iterator>
#include <iostream>
#include <ostream>
#include <algorithm>

namespace sofa
{

namespace component
{

template<class TDataTypes>
class ParticleSource : public core::componentmodel::behavior::Constraint<TDataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef helper::vector<Real> VecDensity;

    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalModel;

    Data<Coord> f_translation;
    Data<Real> f_scale;
    Data< helper::vector<Coord> > f_center;
    Data<Coord> f_radius;
    Data<Deriv> f_velocity;
    Data<Real> f_delay;
    Data<Real> f_start;
    Data<Real> f_stop;

    ParticleSource()
        : f_translation(initData(&f_translation,Coord(),"translation","translation applied to center(s)") )
        , f_scale(initData(&f_scale,(Real)1.0,"scale","scale applied to center(s)") )
        , f_center(initData(&f_center, "center","Source center(s)") )
        , f_radius(initData(&f_radius, Coord(), "radius", "Source radius"))
        , f_velocity(initData(&f_velocity, Deriv(), "velocity", "Particle initial velocity"))
        , f_delay(initData(&f_delay, (Real)0.01, "delay", "Delay between particles creation"))
        , f_start(initData(&f_start, (Real)0, "start", "Source starting time"))
        , f_stop(initData(&f_stop, (Real)1e10, "stop", "Source stopping time"))
    {
        f_listening.setValue(true);
        f_center.beginEdit()->push_back(Coord()); f_center.endEdit();
    }

    virtual ~ParticleSource()
    {
    }

    int N;
    Real lasttime;
    //int lastparticle;
    topology::PointSubset lastparticles;
    helper::vector<Coord> lastpos;

    virtual void init()
    {
        this->core::componentmodel::behavior::Constraint<TDataTypes>::init();
        if (!this->mstate) return;
        N = f_center.getValue().size();
        lasttime = f_start.getValue()-f_delay.getValue();
        //lastparticle = -1;
        lastpos.resize(N);

        lastparticles.setTestFunction(PSTestNewPointFunction);
        lastparticles.setRemovalFunction(PSRemovalFunction);

        lastparticles.setTestParameter( (void *) this );
        lastparticles.setRemovalParameter( (void *) this );

        std::cout << "ParticleSource: center="<<f_center.getValue()<<" radius="<<f_radius.getValue()<<" delay="<<f_delay.getValue()<<" start="<<f_start.getValue()<<" stop="<<f_stop.getValue()<<std::endl;
    }

    virtual void reset()
    {
        this->mstate->resize(1);
        lasttime = f_start.getValue()-f_delay.getValue();
        lastparticles.clear();
        lastpos.clear();
    }

    Real rrand()
    {
        return (Real)(rand()*1.0/RAND_MAX);
    }

    virtual void animateBegin(double /*dt*/, double time)
    {
        //std::cout << "ParticleSource: animate begin time="<<time<<std::endl;
        if (!this->mstate) return;
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        int i0 = this->mstate->getX()->size();
        if (i0==1) i0=0; // ignore the first point if it is the only one
        int nparticles = (int)((time - lasttime) / f_delay.getValue());
        if (nparticles>0)
        {
            //std::cout << "ParticleSource: Creating "<<nparticles<<" particles, total "<<i0+nparticles<<" particles."<<std::endl;
            topology::PointSetTopology<DataTypes>* t = dynamic_cast<topology::PointSetTopology<DataTypes>*>(this->getContext()->getMainTopology());
            if (t != NULL)
            {
                int n = i0+nparticles*N - this->mstate->getX()->size();
                ((topology::PointSetTopologyModifier<DataTypes>*)t->getTopologyModifier())->addPointsWarning(n);
                ((topology::PointSetTopologyModifier<DataTypes>*)t->getTopologyModifier())->addPointsProcess(n);
                t->propagateTopologicalChanges();
            }
            else
            {
                this->mstate->resize(i0+nparticles*N);
            }
            VecCoord& x = *this->mstate->getX();
            VecDeriv& v = *this->mstate->getV();
            for (int i=0; i<nparticles; i++)
            {
                lasttime += f_delay.getValue();
                int lastparticle = i0+i*N;
                lastparticles.resize(N);
                for (int s=0; s<N; s++)
                {
                    lastpos[s] = f_center.getValue()[s]*f_scale.getValue() + f_translation.getValue();
                    for (unsigned int c = 0; c < lastpos[s].size(); c++)
                        lastpos[s][c] += f_radius.getValue()[c] * rrand();
                    x[lastparticle+s] = lastpos[s];
                    v[lastparticle+s] = f_velocity.getValue();
                    x[lastparticle+s] += v[lastparticle+s]*(time - lasttime); // account for particle initial motion
                    lastparticles[s] = lastparticle+s;
                }
            }
        }
    }

/// Handle topological changes
    void handleTopologyChange()
    {
        sofa::core::componentmodel::topology::BaseTopology *topology = (getContext()->getMainTopology());
        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=topology->firstChange();
        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=topology->lastChange();
        if (itBegin != itEnd)
        {
            if (this->f_printLog.getValue())
            {
                std::cout << "ParticleSource: handleTopologyChange()"<<std::endl;
                std::cout << "lastparticles = ";
                std::copy(lastparticles.begin(),lastparticles.end(),std::ostream_iterator<unsigned int>(std::cout," "));
                std::cout << std::endl;
            }
            int s1 = lastparticles.size();
            lastparticles.handleTopologyEvents(itBegin, itEnd, this->mstate->getSize());
            int s2 = lastparticles.size();
            if (s2 > s1) std::cout << "ParticleSource: handleTopologyChange(): "<<s2-s1<<" points added!"<<std::endl;
            if (s2 < s1) std::cout << "ParticleSource: handleTopologyChange(): "<<s1-s2<<" points removed!"<<std::endl;
            if (this->f_printLog.getValue())
            {
                std::cout << "NEW lastparticles = ";
                std::copy(lastparticles.begin(),lastparticles.end(),std::ostream_iterator<unsigned int>(std::cout," "));
                std::cout << std::endl;
            }
        }
    }

    virtual void projectResponse(VecDeriv& res) ///< project dx to constrained space
    {
        if (!this->mstate) return;
        if (lastparticles.empty()) return;
        //std::cout << "ParticleSource: projectResponse of last particle ("<<lastparticle<<")."<<std::endl;
        double time = getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        // constraint the last value
        for (unsigned int s=0; s<lastparticles.size(); s++)
            res[lastparticles[s]] = Deriv();
    }

    virtual void projectVelocity(VecDeriv& res) ///< project dx to constrained space (dx models a velocity)
    {
        if (!this->mstate) return;
        if (lastparticles.empty()) return;
        double time = getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        // constraint the last value
        for (unsigned int s=0; s<lastparticles.size(); s++)
            res[lastparticles[s]] = f_velocity.getValue();
    }

    virtual void projectPosition(VecCoord& x) ///< project x to constrained space (x models a position)
    {
        if (!this->mstate) return;
        if (lastparticles.empty()) return;
        double time = getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        // constraint the last value
        for (unsigned int s=0; s<lastparticles.size(); s++)
        {
            x[lastparticles[s]] = lastpos[s];
            x[lastparticles[s]] += f_velocity.getValue()*(time - lasttime); // account for particle initial motion
        }
    }

    virtual void animateEnd(double /*dt*/, double /*time*/)
    {

    }

    virtual void handleEvent(sofa::core::objectmodel::Event* event)
    {
        if (simulation::AnimateBeginEvent* ev = dynamic_cast<simulation::AnimateBeginEvent*>(event))
            animateBegin(ev->getDt(), getContext()->getTime());
        if (simulation::AnimateEndEvent* ev = dynamic_cast<simulation::AnimateEndEvent*>(event))
            animateEnd(ev->getDt(), getContext()->getTime());
    }


protected :
    static bool PSTestNewPointFunction(int, void*, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
    {
        return false;
    }

    static void PSRemovalFunction (int index, void* p)
    {
        ParticleSource* ps = (ParticleSource*)p;
        topology::PointSubset::const_iterator it = std::find(ps->lastparticles.begin(),ps->lastparticles.end(), (unsigned int)index);
        if (it != ps->lastparticles.end())
        {
            ps->lastpos.erase( ps->lastpos.begin()+(it-ps->lastparticles.begin()) );
            //ps->lastparticles.getArray().erase(it);
            helper::removeValue(ps->lastparticles,(unsigned int)index);
        }
    }

};


} // namespace component

} // namespace sofa

#endif

