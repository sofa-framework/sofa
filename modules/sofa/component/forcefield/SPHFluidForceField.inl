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
#ifndef SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_INL

#include <sofa/component/forcefield/SPHFluidForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/component/container/SpatialGridContainer.inl>
#include <sofa/helper/system/config.h>
#include <sofa/helper/gl/template.h>
#include <math.h>
#include <iostream>

#ifdef WIN32
#include <windows.h>
#endif

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
SPHFluidForceField<DataTypes>::SPHFluidForceField()
    : particleRadius	(initData(&particleRadius		,Real(1)		, "radius", "Radius of a Particle")),
                    particleMass		(initData(&particleMass			,Real(1)		, "mass", "Mass of a Particle")),
                    pressureStiffness	(initData(&pressureStiffness	,Real(100)		, "pressure", "Pressure")),
                    density0			(initData(&density0				,Real(1)		, "density", "Density")),
                    viscosity			(initData(&viscosity			,Real(0.001f)	, "viscosity", "Viscosity")),
                    surfaceTension	(initData(&surfaceTension		,Real(0)		, "surfaceTension", "Surface Tension")),
                    newDensity		(initData(&newDensity			,false			, "newDensity", "Use new and more stable density computation")),
                    pressureExponent	(initData(&pressureExponent		,1				, "pressureExponent", "Exponent of density variation in pressure expression")),
                    usePCISPH			(initData(&usePCISPH			,false			, "usePCISPH", "Use Predictive-Corrective Incompressible SPH")),
                    grid(NULL)
{
}


template<class DataTypes>
void SPHFluidForceField<DataTypes>::init()
{
    this->Inherit::init();
    this->getContext()->get(grid); //new Grid(particleRadius.getValue());
    if (grid==NULL)
        serr<<"SpatialGridContainer not found by SPHFluidForceField, slow O(n2) method will be used !!!" << sendl;
    int n = (*this->mstate->getX()).size();
    particles.resize(n);
    for (int i=0; i<n; i++)
    {
        particles[i].neighbors.clear();
#ifdef SOFA_DEBUG_SPATIALGRIDCONTAINER
        particles[i].neighbors2.clear();
#endif
        particles[i].density = density0.getValue();
        particles[i].pressure = 0;
        particles[i].normal.clear();
        particles[i].curvature = 0;
    }

    lastTime = (Real)this->getContext()->getTime();
}


template<class DataTypes>
void SPHFluidForceField<DataTypes>::addForce(DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v, const core::MechanicalParams* /* mparams */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    const Real h = particleRadius.getValue();
    const Real h2 = h*h;
    const Real m = particleMass.getValue();
    const Real m2 = m*m;
    const bool newDensity = this->newDensity.getValue();
    const Real d0 = density0.getValue();
    const int pE = pressureExponent.getValue();
    const Real k = pressureStiffness.getValue(); // /(pE); //*(Real)pow(d0,pE-1));
    const bool PCISPH = usePCISPH.getValue();
    const Real time = (Real)this->getContext()->getTime();
    const Real dt = (Real)this->getContext()->getDt();
    const Real dt2 = dt*dt;
    const Real betaPCISPH = dt2*m2*2/(d0*d0);
    lastTime = time;

    //const Vec3d localg = this->getContext()->getLocalGravity();
    //Deriv g;
    //DataTypes::set ( g, localg[0], localg[1], localg[2]);
    //const Deriv mg = g * mass;
    const int n = x.size();

    VecCoord vec;
    vec.resize(n);
    if(iterParticles.size()>1) vec = iterParticles[1];
    else vec = x;
    iterParticles.clear();
    iterParticles.push_back(vec);

    // Precompute constants for smoothing kernels
    const Real     CWd =     constWd(h);
    const Real CgradWd = constGradWd(h);
    //const Real  ClaplacianWd =  constLaplacianWd(h);
    //const Real     CWp =     constWp(h);
    const Real CgradWp = constGradWp(h);
    //const Real  ClaplacianWp =  constLaplacianWp(h);
    //const Real     CWv =     constWv(h);
    //const Real CgradWv = constGradWv(h);
    const Real  ClaplacianWv =  constLaplacianWv(h);
    //const Real     CWc =     constWc(h);
    const Real CgradWc = constGradWc(h);
    const Real  ClaplacianWc =  constLaplacianWc(h);

    // Initialization
    f.resize(n);
    dforces.clear();
    int n0 = particles.size();
    particles.resize(n);
    PCIParticles.resize(n);
    for (int i=0; i<n; i++)
    {
        particles[i].neighbors.clear();
#ifdef SOFA_DEBUG_SPATIALGRIDCONTAINER
        particles[i].neighbors2.clear();
#endif
        //particles[i].density = 0;
        particles[i].pressure = 0;
        particles[i].normal.clear();
        particles[i].curvature = 0;
    }
    if (newDensity)
    {
        for (int i=n0; i<n; i++)
            particles[i].density = d0;
    }
    else
    {
        for (int i=0; i<n; i++)
            particles[i].density = 0;
    }

    // First compute the neighbors
    // This is an O(n2) step, except if a hash-grid is used to optimize it
    if (grid == NULL)
    {
        for (int i=0; i<n; i++)
        {
            const Coord& ri = x[i];
            for (int j=i+1; j<n; j++)
            {
                const Coord& rj = x[j];
                Real r2 = (rj-ri).norm2();
                if (r2 < h2)
                {
                    Real r_h = (Real)sqrt(r2/h2);
                    particles[i].neighbors.push_back(std::make_pair(j,r_h));
                    //particles[j].neighbors.push_back(std::make_pair(i,r_h));
                }
            }
        }
    }
    else
    {
        grid->updateGrid(x);
        grid->findNeighbors(this, h);
#ifdef SOFA_DEBUG_SPATIALGRIDCONTAINER
        // Check grid
        for (int i=0; i<n; i++)
        {
            const Coord& ri = x[i];
            for (int j=i+1; j<n; j++)
            {
                const Coord& rj = x[j];
                Real r2 = (rj-ri).norm2();
                if (r2 < h2)
                {
                    Real r_h = (Real)sqrt(r2/h2);
                    particles[i].neighbors2.push_back(std::make_pair(j,r_h));
                }
            }
        }
        for (int i=0; i<n; i++)
        {
            if (particles[i].neighbors.size() != particles[i].neighbors2.size())
            {
                serr << "particle "<<i<<" "<< x[i] <<" : "<<particles[i].neighbors.size()<<" neighbors on grid, "<< particles[i].neighbors2.size() << " neighbors on bruteforce."<<sendl;
                serr << "grid-only neighbors:";
                for (unsigned int j=0; j<particles[i].neighbors.size(); j++)
                {
                    int index = particles[i].neighbors[j].first;
                    unsigned int j2 = 0;
                    while (j2 < particles[i].neighbors2.size() && particles[i].neighbors2[j2].first != index)
                        ++j2;
                    if (j2 == particles[i].neighbors2.size())
                        serr << " "<< x[index] << "<"<< particles[i].neighbors[j].first<<","<<particles[i].neighbors[j].second<<">";
                }
                serr << ""<<sendl;
                serr << "bruteforce-only neighbors:";
                for (unsigned int j=0; j<particles[i].neighbors2.size(); j++)
                {
                    int index = particles[i].neighbors2[j].first;
                    unsigned int j2 = 0;
                    while (j2 < particles[i].neighbors.size() && particles[i].neighbors[j2].first != index)
                        ++j2;
                    if (j2 == particles[i].neighbors.size())
                        serr << " "<< x[index] << "<"<< particles[i].neighbors2[j].first<<","<<particles[i].neighbors2[j].second<<">";
                }
                serr << ""<<sendl;
            }
        }
#endif
    }

    // Compute density and pressure
    //if(!PCISPH)
    {
        if (newDensity)
        {
            for (int i=0; i<n0; i++)
            {
                Particle& Pi = particles[i];
                for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
                {
                    const int j = it->first;
                    const Real r_h = it->second;
                    Particle& Pj = particles[j];
                    Real d = dt*m*(gradWd(x[i]-x[j],r_h,CgradWd)*(v[i]-v[j]));
                    Pi.density += d;
                    Pj.density += d;
                }
                Pi.pressure = k*(Real)pow(Pi.density - d0, pE);
            }
        }
        else
        {
            for (int i=0; i<n; i++)
            {
                Particle& Pi = particles[i];
                Real density = Pi.density;

                density += m*Wd(0,CWd); // density from current particle

                for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
                {
                    const int j = it->first;
                    const Real r_h = it->second;
                    Particle& Pj = particles[j];
                    Real d = m*Wd(r_h,CWd);
                    //				Real d = m*GetMonaghanKernel((x[i]-x[j]).norm(),h);
                    density += d;
                    Pj.density += d;

                }
                Pi.density = density;
                if(!PCISPH)
                    Pi.pressure = k*(density - d0);

            }
        }
    }

    // Compute surface normal and curvature
    if (surfaceTension.getValue() > 0)
    {
        for (int i=0; i<n; i++)
        {
            Particle& Pi = particles[i];
            for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
            {
                const int j = it->first;
                const Real r_h = it->second;
                Particle& Pj = particles[j];
                Deriv n = gradWc(x[i]-x[j],r_h,CgradWc) * (m / Pj.density - m / Pi.density);
                Pi.normal += n;
                Pj.normal -= n;
                Real c = laplacianWc(r_h,ClaplacianWc) * (m / Pj.density - m / Pi.density);
                Pi.curvature += c;
                Pj.curvature -= c;
            }
        }
    }

    //if(!PCISPH)
    {
        // Compute the forces
        for (int i=0; i<n; i++)
        {
            Particle& Pi = particles[i];
            // Gravity
            //f[i] += g*(m*Pi.density);

            for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
            {
                const int j = it->first;
                const Real r_h = it->second;
                Particle& Pj = particles[j];
                // Pressure

                if(!PCISPH)
                {
                    Deriv fpressure = gradWp(x[i]-x[j],r_h,CgradWp) * ( - m2 * (Pi.pressure / (Pi.density*Pi.density) + Pj.pressure / (Pj.density*Pj.density)) );
                    f[i] += fpressure;
                    f[j] -= fpressure;
                }

                // Viscosity
                if(!PCISPH)
                {
                    Deriv fviscosity = ( v[j] - v[i] ) * ( m2 * viscosity.getValue() / (Pi.density * Pj.density) * laplacianWv(r_h,ClaplacianWv) );
                    f[i] += fviscosity;
                    f[j] -= fviscosity;
                }
                else
                {
                    Deriv fviscosity = ( v[j] - v[i] ) * ( m2 * viscosity.getValue() / (Pi.density * Pj.density) * GetMonaghanLap((x[j]-x[i]).norm(),h) );
                    f[i] += fviscosity;
                    f[j] -= fviscosity;
                }
            }

            if (surfaceTension.getValue() > 0)
            {
                Real n = Pi.normal.norm();
                if (n > 0.000001)
                {
                    Deriv fsurface = Pi.normal * ( - m * surfaceTension.getValue() * Pi.curvature / n );
                    f[i] += fsurface;
                }
            }

        }

    }



    if(PCISPH)
    {

        Real max_predicted_density_variation=0;
        int iteration=0;
        for (int i=0; i<n; i++)
        {
            PredictedParticle& Piv = PCIParticles[i];
            Particle& Pi = particles[i];

            //predict velocity and position
            Piv.predicted_density = 0;
            Pi.pressure= 0;
        }



        while(((max_predicted_density_variation>d0*0.01) || (iteration<4)) /*&& iteration <2000*/)
        {
            printf("while((max_predicted_density_variation>d0/100)||(iteration<4)) --> ((%f>%f) || (%d<n)\n",max_predicted_density_variation,d0*0.01,iteration);


            max_predicted_density_variation=0;

            for (int i=0; i<n; i++)
            {

                PredictedParticle& Piv = PCIParticles[i];

                //predict velocity and position
                Piv.predicted_velocity = v[i] + (Piv.pressure_force*dt)/m;
                Piv.predicted_position = x[i] + Piv.predicted_velocity*dt;

                //initialize
                Piv.predicted_density = 0;
                Deriv d;
                Piv.sum_gradWd = d;
                Piv.sum_gradWdWd =0;
                Piv.pressure_force =  f[i];
            }
            Real distance_min=100;
            Real force_max=0;
            for (int i=0; i<n; i++)
            {
                PredictedParticle& Piv = PCIParticles[i];

                Particle& Pi = particles[i];

                //computation of sums GetMonaghanKernel
                Piv.predicted_density += m*Wd(0,CWd);
                //				Piv.predicted_density += m*GetMonaghanKernel(0,h);

                const Coord& ri = Piv.predicted_position;

                Piv.neighborhood = 0;
                for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
                {
                    const int j = it->first;
                    PredictedParticle& Pjv = PCIParticles[j];
                    const Coord& rj = Pjv.predicted_position;
                    Real r2 = (rj-ri).norm2();
                    Real r  = sqrt(r2);
                    Real r_h = (Real)sqrt(r2/h2);

                    if(r2<h2)
                    {
                        Piv.neighborhood++;

                        Real vWd = Wd(r_h,CWd);
                        //						Real vWd = GetMonaghanKernel(r,h);

                        Deriv vgradWd = gradWd(rj-ri,r_h,CgradWd);//value_laplacianWc;
                        //						Deriv vgradWd = (rj-ri)*GetMonaghanGrad(r,h);
                        if(r<distance_min)distance_min=r;
                        //std::cout << "distance: " << r << " - Wd: " << vWd << " - gradWd: " << vgradWd << std::endl;

                        //predict density
                        Piv.predicted_density += m*vWd;
                        Pjv.predicted_density += m*vWd;

                        Piv.sum_gradWd += vgradWd;
                        Piv.sum_gradWdWd += vgradWd*vgradWd;
                        Pjv.sum_gradWd += vgradWd;
                        Pjv.sum_gradWdWd += vgradWd*vgradWd;
                    }

                }
                //std::cout << "predicted density: " << Piv.predicted_density << std::endl;
                if(Piv.neighborhood)
                {
                    //predict density_variation
                    Piv.predicted_density_variation = Piv.predicted_density - d0;
                    if(Piv.predicted_density_variation<0)Piv.predicted_density_variation=0;
                    //std::cout << "predicted variation: " << Piv.predicted_density_variation << std::endl;

                    if(Piv.predicted_density_variation>max_predicted_density_variation)max_predicted_density_variation=Piv.predicted_density_variation;

                    //udpate pressure
                    Real sigma = (Real)-1.0/(betaPCISPH*(-Piv.sum_gradWd*Piv.sum_gradWd-Piv.sum_gradWdWd));
                    Piv.pressure_variation = sigma*Piv.predicted_density_variation;

                    Pi.pressure += Piv.pressure_variation;
                    //std::cout << "sigma: " << sigma << " - pressure_variation: " << Piv.pressure_variation << " - pressure: " << Pi.pressure << std::endl;
                }
                Pi.density = Piv.predicted_density;
            }
            for (int i=0; i<n; i++)
            {
                PredictedParticle& Piv = PCIParticles[i];
                Particle& Pi = particles[i];
                for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
                {
                    const int j = it->first;
                    PredictedParticle& Pjv = PCIParticles[j];
                    Particle& Pj = particles[j];
                    //	Real r_h = (Real)sqrt((x[i]-x[j]).norm2()/h2);
                    Real r = (x[i]-x[j]).norm();
                    //					Deriv fpressure = gradWp(x[i]-x[j],r_h,CgradWd) * ( - m2 * (Pi.pressure / (Pi.density*Pi.density) + Pj.pressure / (Pj.density*Pj.density)) );
                    //if(fpressure.norm2()>55000){fpressure = ((x[i]-x[j])/(x[i]-x[j]).norm())*55000;}
                    Deriv fpressure = (x[i]-x[j]) * GetMonaghanGrad(r,h) * ( - m2 * (Pi.pressure / (Pi.density*Pi.density) + Pj.pressure / (Pj.density*Pj.density)) );
                    //std::cout << "fpressure "<< fpressure << "="<< Pi.pressure <<"/"<< (Pi.density*Pi.density) << "+" << Pj.pressure <<"/"<< (Pj.density*Pj.density) << std::endl;
                    Piv.pressure_force += fpressure;
                    Pjv.pressure_force -= fpressure;
                    if(fpressure.norm()>force_max)force_max=fpressure.norm();
                }

                //std::cout << "distance_min" << distance_min << " force_max:" << force_max << "distance engeandrée " << force_max*dt*dt << std::endl;

                if(distance_min < 0.65)
                {
#ifdef WIN32
                    Sleep(2);
#else
                    sleep(2);
#endif
                }
            }




            VecCoord vec;


            vec.resize(n);
            if(iteration==0)iterParticles.push_back(x);

            for (int i=0; i<n; i++)
            {
                PredictedParticle& Piv = PCIParticles[i];
                vec[i]= Piv.predicted_position;
            }
            iterParticles.push_back(vec);

            iteration++;



        }


        for (int i=0; i<n; i++)
        {

            PredictedParticle& Piv = PCIParticles[i];
            // Particle& Pi = particles[i];

            //std::cout << "force" << f[i] << " + " << Piv.pressure_force << "=" << f[i] +Piv.pressure_force << std::endl;
            f[i] = Piv.pressure_force;
            //			f[i] += Piv.pressure_force;


            /*	for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
            {
            const int j = it->first;
            const Real r_h = it->second;
            Real r = (x[j]-x[i]).norm();
            Particle& Pj = particles[j];

            // Viscosity
            Deriv fviscosity = ( v[j] - v[i] ) * ( m2 * viscosity.getValue() / (Pi.density * Pj.density) * GetMonaghanLap(r,h));
            //Deriv fviscosity = ( v[j] - v[i] ) * ( m2 * viscosity.getValue() / (Pi.density * Pj.density) * laplacianWv(r_h,ClaplacianWv) );
            f[i] += fviscosity;
            f[j] -= fviscosity;
            }
            if (surfaceTension.getValue() > 0)
            {
            Real n = Pi.normal.norm();
            if (n > 0.000001)
            {
            Deriv fsurface = Pi.normal * ( - m * surfaceTension.getValue() * Pi.curvature / n );
            f[i] += fsurface;
            }
            }*/

        }
        //printf("z\n");

    }
    d_f.endEdit();
}

template<class DataTypes>
void SPHFluidForceField<DataTypes>::addDForce(DataVecDeriv& d_df, const DataVecDeriv& d_dx, const core::MechanicalParams* mparams)
{
    VecDeriv& f1 = *d_df.beginEdit();
    const VecDeriv& dx1 = d_dx.getValue();
    double kFactor = mparams->kFactor();

    const VecCoord& p1 = *this->mstate->getX();
    f1.resize(dx1.size());
    for (unsigned int i=0; i<this->dforces.size(); i++)
    {
        const DForce& df = this->dforces[i];
        const unsigned int ia = df.a;
        const unsigned int ib = df.b;
        const Deriv u = p1[ib]-p1[ia];
        const Deriv du = dx1[ib]-dx1[ia];
        const Deriv dforce = u * (df.df * (du*u) * kFactor);
        f1[ia] += dforce * kFactor;
        f1[ib] -= dforce * kFactor;
    }

    d_df.endEdit();
}

template<class DataTypes>
void SPHFluidForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields()) return;
    //if (grid != NULL)
    //	grid->draw();
    const VecCoord& x = *this->mstate->getX();
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(0);
    glColor3f(0,1,1);
    glLineWidth(1);
    glBegin(GL_LINES);
    for (unsigned int i=0; i<particles.size(); i++)
    {
        Particle& Pi = particles[i];
#ifdef SOFA_DEBUG_SPATIALGRIDCONTAINER
        // Check grid
        if (Pi.neighbors.size() != Pi.neighbors2.size())
        {
            glColor4f(1,0,0,1);
            for (unsigned int j=0; j<Pi.neighbors.size(); j++)
            {
                int index = Pi.neighbors[j].first;
                unsigned int j2 = 0;
                while (j2 < Pi.neighbors2.size() && Pi.neighbors2[j2].first != index)
                    ++j2;
                if (j2 == Pi.neighbors2.size())
                {
                    helper::gl::glVertexT(x[i]);
                    helper::gl::glVertexT(x[index]);
                }
            }
            glColor4f(1,0,1,1);
            for (unsigned int j=0; j<Pi.neighbors2.size(); j++)
            {
                int index = Pi.neighbors2[j].first;
                unsigned int j2 = 0;
                while (j2 < Pi.neighbors.size() && Pi.neighbors[j2].first != index)
                    ++j2;
                if (j2 == Pi.neighbors.size())
                {
                    helper::gl::glVertexT(x[i]);
                    helper::gl::glVertexT(x[index]);
                }
            }
        }
#else
        for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
        {
            const int j = it->first;
            const float r_h = (float)it->second;
            float f = r_h*2;
            if (f < 1)
            {
                glColor4f(0,1-f,f,1-r_h);
            }
            else
            {
                glColor4f(f-1,0,2-f,1-r_h);
            }
            helper::gl::glVertexT(x[i]);
            helper::gl::glVertexT(x[j]);
        }
#endif
    }
    glEnd();
    glDisable(GL_BLEND);
    glDepthMask(1);
    glPointSize(5);
    glBegin(GL_POINTS);
    for (unsigned int i=0; i<particles.size(); i++)
    {
        Particle& Pi = particles[i];
        float f = (float)(Pi.density / density0.getValue());
        f = 1+10*(f-1);
        if (f < 1)
        {
            glColor3f(0,1-f,f);
        }
        else
        {
            glColor3f(f-1,0,2-f);
        }
        helper::gl::glVertexT(x[i]);
    }


    float red[16]   =	{0.1f, 0.1f, 0.1f, 0.7f, 0.1f, 0.7f, 0.7f, 0.7f,  0.4f, 0.4f, 0.4f, 1.0f, 0.4f, 1.0f, 1.0f, 1.0f};
    float green[16] =	{0.1f, 0.1f, 0.7f, 0.1f, 0.7f, 0.7f, 0.1f, 0.7f,  0.4f, 0.4f, 1.0f, 0.4f, 1.0f, 1.0f, 0.4f, 1.0f};
    float blue[16]  =	{0.1f, 0.7f, 0.1f, 0.1f, 0.7f, 0.1f, 0.7f, 0.7f,  0.4f, 1.0f, 0.4f, 0.4f, 1.0f, 0.4f, 1.0f, 1.0f};
    for(unsigned int i=0; i<iterParticles.size(); i++)
    {
        glColor3f(red[i%16],green[i%16],blue[i%16]);
        VecCoord v = iterParticles[i];
        //std::cout << "iteration "<<i;
        for (unsigned int j=0; j<particles.size(); j++)
        {
            helper::gl::glVertexT(v[j]);
            //std::cout << "{" << v[j] << "} ";
        }
        //std::cout << std::endl;


        //if(particles.size()==2)
        //{
        //	std::cout << (v[0]-v[1]).norm() << std::endl;
        //}

    }

    //Real distancemin=100;
    //Real distanceparcourumax=0;
    //for (unsigned int i=0;i<iterParticles.size();i++)
    //{
    //	for (unsigned int j=0;j<iterParticles.size();j++)
    //	{
    //		Real distance1 = ((iterParticles[1])[i] -  (iterParticles[1])[j]).norm();
    //		if(distance1<distancemin && i!=j)distancemin=distance1;
    //	}
    //	Real distance2 = ((iterParticles[0])[i] -  (iterParticles[1])[i]).norm();
    //	if(distance2>distanceparcourumax)distanceparcourumax=distance2;
    //}

    //std::cout << "distancemin " << distancemin<< std::endl;
    //std::cout << "distanceparcourumax " << distanceparcourumax<< std::endl;

    glEnd();
    glPointSize(1);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_INL
