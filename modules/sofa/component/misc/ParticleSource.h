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
#ifndef ParticleSource_H
#define ParticleSource_H

#include <sofa/core/VisualModel.h>
#include "sofa/core/objectmodel/Event.h"
#include "sofa/simulation/tree/AnimateBeginEvent.h"
#include <sofa/simulation/tree/AnimateEndEvent.h>
#include <sofa/simulation/tree/GNode.h>

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
    Data< helper::vector<Coord> > f_center;
    Data<Coord> f_radius;
    Data<Deriv> f_velocity;
    Data<Real> f_delay;
    Data<Real> f_start;
    Data<Real> f_stop;

    ParticleSource()
        : f_translation(initData(&f_translation,Coord(),"translation","translation") )
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
    int lastparticle;
    helper::vector<Coord> lastpos;

    virtual void init()
    {
        this->core::componentmodel::behavior::Constraint<TDataTypes>::init();
        if (!this->mstate) return;
        N = f_center.getValue().size();
        lasttime = f_start.getValue()-f_delay.getValue();
        lastparticle = -1;
        lastpos.resize(N);
        std::cout << "ParticleSource: center="<<f_center.getValue()<<" radius="<<f_radius.getValue()<<" delay="<<f_delay.getValue()<<" start="<<f_start.getValue()<<" stop="<<f_stop.getValue()<<std::endl;
    }

    virtual void reset()
    {
        lasttime = f_start.getValue()-f_delay.getValue();
        lastparticle = -1;
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
            this->mstate->resize(i0+nparticles*N);
            VecCoord& x = *this->mstate->getX();
            VecDeriv& v = *this->mstate->getV();
            for (int i=0; i<nparticles; i++)
            {
                lasttime += f_delay.getValue();
                lastparticle = i0+i*N;
                for (int s=0; s<N; s++)
                {
                    lastpos[s] = f_center.getValue()[s] + f_translation.getValue();
                    for (unsigned int c = 0; c < lastpos[s].size(); c++)
                        lastpos[s][c] += f_radius.getValue()[c] * rrand();
                    x[lastparticle+s] = lastpos[s];
                    v[lastparticle+s] = f_velocity.getValue();
                    x[lastparticle+s] += v[lastparticle+s]*(time - lasttime); // account for particle initial motion
                }
            }
        }
    }

    virtual void projectResponse(VecDeriv& res) ///< project dx to constrained space
    {
        if (!this->mstate) return;
        if (lastparticle == -1) return;
        //std::cout << "ParticleSource: projectResponse of last particle ("<<lastparticle<<")."<<std::endl;
        double time = getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        // constraint the last value
        for (int s=0; s<N; s++)
            res[lastparticle+s] = Deriv();
    }

    virtual void projectVelocity(VecDeriv& res) ///< project dx to constrained space (dx models a velocity)
    {
        if (!this->mstate) return;
        if (lastparticle == -1) return;
        double time = getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        // constraint the last value
        for (int s=0; s<N; s++)
            res[lastparticle+s] = f_velocity.getValue();
    }

    virtual void projectPosition(VecCoord& x) ///< project x to constrained space (x models a position)
    {
        if (!this->mstate) return;
        if (lastparticle == -1) return;
        double time = getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;
        // constraint the last value
        for (int s=0; s<N; s++)
        {
            x[lastparticle+s] = lastpos[s];
            x[lastparticle+s] += f_velocity.getValue()*(time - lasttime); // account for particle initial motion
        }
    }

    virtual void animateEnd(double /*dt*/, double /*time*/)
    {

    }

    virtual void handleEvent(sofa::core::objectmodel::Event* event)
    {
        if (simulation::tree::AnimateBeginEvent* ev = dynamic_cast<simulation::tree::AnimateBeginEvent*>(event))
            animateBegin(ev->getDt(), getContext()->getTime());
        if (simulation::tree::AnimateEndEvent* ev = dynamic_cast<simulation::tree::AnimateEndEvent*>(event))
            animateEnd(ev->getDt(), getContext()->getTime());
    }
};


}

}

#endif

