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
#ifndef SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_INL

#include <SofaSphFluid/SPHFluidForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaSphFluid/SpatialGridContainer.inl>
#include <sofa/helper/system/config.h>
#include <cmath>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
SPHFluidForceField<DataTypes>::SPHFluidForceField()
    : particleRadius (initData(&particleRadius, Real(1), "radius", "Radius of a Particle"))
    , particleMass (initData(&particleMass, Real(1), "mass", "Mass of a Particle"))
    , pressureStiffness (initData(&pressureStiffness, Real(100), "pressure", "Pressure"))
    , density0 (initData(&density0, Real(1), "density", "Density"))
    , viscosity (initData(&viscosity, Real(0.001f), "viscosity", "Viscosity"))
    , surfaceTension (initData(&surfaceTension, Real(0), "surfaceTension", "Surface Tension"))
    , kernelType(initData(&kernelType, 0, "kernelType", "0 = default kernels, 1 = cubic spline"))
    , pressureType(initData(&pressureType, 1, "pressureType", "0 = none, 1 = default pressure"))
    , viscosityType(initData(&viscosityType, 1, "viscosityType", "0 = none, 1 = default viscosity using kernel Laplacian, 2 = artificial viscosity"))
    , surfaceTensionType(initData(&surfaceTensionType, 1, "surfaceTensionType", "0 = none, 1 = default surface tension using kernel Laplacian, 2 = cohesion forces surface tension from Becker et al. 2007"))
    , d_debugGrid(initData(&d_debugGrid, false, "debugGrid", "If true will store additionnal information on the grid to check neighbors and draw them"))
    , grid(NULL)
{

}

template<class DataTypes>
void SPHFluidForceField<DataTypes>::init()
{
    this->Inherit::init();

    SPHKernel<SPH_KERNEL_CUBIC,Deriv> Kcubic(4);
    if (!Kcubic.CheckAll(2, sout.ostringstream(), serr.ostringstream())) serr << sendl;
    SPHKernel<SPH_KERNEL_DEFAULT_DENSITY,Deriv> Kd(4);
    if (!Kd.CheckAll(2, sout.ostringstream(), serr.ostringstream())) serr << sendl;
    SPHKernel<SPH_KERNEL_DEFAULT_PRESSURE,Deriv> Kp(4);
    if (!Kp.CheckAll(1, sout.ostringstream(), serr.ostringstream())) serr << sendl;
    SPHKernel<SPH_KERNEL_DEFAULT_VISCOSITY,Deriv> Kv(4);
    if (!Kv.CheckAll(2, sout.ostringstream(), serr.ostringstream())) serr << sendl;
    sout << sendl;

    this->getContext()->get(grid); //new Grid(particleRadius.getValue());
    if (grid==NULL)
        msg_error() << "SpatialGridContainer not found by SPHFluidForceField, slow O(n2) method will be used !!!";

    const unsigned n = this->mstate->getSize();
    particles.resize(n);
    for (unsigned i=0u; i<n; i++)
    {
        particles[i].neighbors.clear();
        particles[i].density = density0.getValue();
        particles[i].pressure = 0;
        particles[i].normal.clear();
        particles[i].curvature = 0;
    }

    if (d_debugGrid.getValue())
    {
        for (unsigned i = 0u; i < n; i++)
        {
            particles[i].neighbors2.clear();
        }
    }

    lastTime = (Real)this->getContext()->getTime();
}


template<class DataTypes>
void SPHFluidForceField<DataTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    computeNeighbors(mparams, d_x, d_v);

    switch(kernelType.getValue())
    {
    default:
        msg_error() << "Unsupported kernelType " << kernelType.getValue();
        // fallthrough
    case 0: // default
    {
        computeForce <SPHKernel<SPH_KERNEL_DEFAULT_DENSITY,Deriv>,
                     SPHKernel<SPH_KERNEL_DEFAULT_PRESSURE,Deriv>,
                     SPHKernel<SPH_KERNEL_DEFAULT_VISCOSITY,Deriv>,
                     SPHKernel<SPH_KERNEL_DEFAULT_DENSITY,Deriv> > (mparams, d_f, d_x, d_v);
        break;
    }
    case 1: // cubic
    {
        computeForce <SPHKernel<SPH_KERNEL_CUBIC,Deriv>,
                     SPHKernel<SPH_KERNEL_CUBIC,Deriv>,
                     SPHKernel<SPH_KERNEL_CUBIC,Deriv>,
                     SPHKernel<SPH_KERNEL_CUBIC,Deriv> > (mparams, d_f, d_x, d_v);
        break;
    }
    }

    msg_info() << "density[" << 0 << "] = " << particles[0].density  << "(" << particles[0].neighbors.size() << " neighbors)"
               << "density[" << particles.size()/2 << "] = " << particles[particles.size()/2].density ;
}


template<class DataTypes>
void SPHFluidForceField<DataTypes>::computeNeighbors(const core::MechanicalParams* /*mparams*/, const DataVecCoord& d_x, const DataVecDeriv& /*d_v*/)
{
    helper::ReadAccessor<DataVecCoord> x = d_x;

    const Real h = particleRadius.getValue();
    const Real h2 = h*h;

    const int n = x.size();
    particles.resize(n);
    for (int i=0; i<n; i++) {
        particles[i].neighbors.clear();
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
        grid->updateGrid(x.ref());
        grid->findNeighbors(this, h);


        if (!d_debugGrid.getValue())
            return;

        for (int i = 0; i < n; i++) {
            particles[i].neighbors2.clear();
        }

        // Check grid info
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
                msg_error() << "particle "<<i<<" "<< x[i] <<" : "<<particles[i].neighbors.size()<<" neighbors on grid, "<< particles[i].neighbors2.size() << " neighbors on bruteforce.";
                msg_error() << "grid-only neighbors:";
                for (unsigned int j=0; j<particles[i].neighbors.size(); j++)
                {
                    int index = particles[i].neighbors[j].first;
                    unsigned int j2 = 0;
                    while (j2 < particles[i].neighbors2.size() && particles[i].neighbors2[j2].first != index)
                        ++j2;
                    if (j2 == particles[i].neighbors2.size())
                        msg_error() << " "<< x[index] << "<"<< particles[i].neighbors[j].first<<","<<particles[i].neighbors[j].second<<">";
                }
                msg_error() << "";
                msg_error() << "bruteforce-only neighbors:";
                for (unsigned int j=0; j<particles[i].neighbors2.size(); j++)
                {
                    int index = particles[i].neighbors2[j].first;
                    unsigned int j2 = 0;
                    while (j2 < particles[i].neighbors.size() && particles[i].neighbors[j2].first != index)
                        ++j2;
                    if (j2 == particles[i].neighbors.size())
                        msg_error() << " "<< x[index] << "<"<< particles[i].neighbors2[j].first<<","<<particles[i].neighbors2[j].second<<">";
                }
                msg_error() << "";
            }
        }

    }
}


template<class DataTypes> template<class TKd, class TKp, class TKv, class TKc>
void SPHFluidForceField<DataTypes>::computeForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    helper::WriteAccessor<DataVecDeriv> f = d_f;
    helper::ReadAccessor<DataVecCoord> x = d_x;
    helper::ReadAccessor<DataVecDeriv> v = d_v;

    const Real h = particleRadius.getValue();
    const Real h2 = h*h;
    const Real m = particleMass.getValue();
    const Real m2 = m*m;
    const Real d0 = density0.getValue();
    const Real k = pressureStiffness.getValue();
    const Real time = (Real)this->getContext()->getTime();
    const Real viscosity = this->viscosity.getValue();
    const int viscosityT = (viscosity == 0) ? 0 : viscosityType.getValue();
    const Real surfaceTension = this->surfaceTension.getValue();
    const int surfaceTensionT = (surfaceTension <= 0) ? 0 : surfaceTensionType.getValue();
    //const Real dt = (Real)this->getContext()->getDt();
    lastTime = time;

    const int n = x.size();

    // Initialization
    f.resize(n);
    dforces.clear();
    //int n0 = particles.size();
    particles.resize(n);
    for (int i=0; i<n; i++)
    {
        particles[i].density = 0;
        particles[i].pressure = 0;
        particles[i].normal.clear();
        particles[i].curvature = 0;
    }

    TKd Kd(h);
    TKp Kp(h);
    TKv Kv(h);
    TKc Kc(h);

    // Compute density and pressure
    for (int i=0; i<n; i++)
    {
        Particle& Pi = particles[i];
        Real density = Pi.density;

        density += m*Kd.W(0); // density from current particle

        for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
        {
            const int j = it->first;
            const Real r_h = it->second;
            Particle& Pj = particles[j];
            Real d = m*Kd.W(r_h);
            density += d;
            Pj.density += d;

        }
        Pi.density = density;
        Pi.pressure = k*(density - d0);
    }

    // Compute surface normal and curvature
    if (surfaceTensionType == 1)
    {
        for (int i=0; i<n; i++)
        {
            Particle& Pi = particles[i];
            for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
            {
                const int j = it->first;
                const Real r_h = it->second;
                Particle& Pj = particles[j];
                Deriv n = Kc.gradW(x[i]-x[j],r_h) * (m / Pj.density - m / Pi.density);
                Pi.normal += n;
                Pj.normal -= n;
                Real c = Kc.laplacianW(r_h) * (m / Pj.density - m / Pi.density);
                Pi.curvature += c;
                Pj.curvature -= c;
            }
        }
    }

    
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

            Real pressureFV = ( - m2 * (Pi.pressure / (Pi.density*Pi.density) + Pj.pressure / (Pj.density*Pj.density)) );

            // Viscosity
            switch(viscosityT)
            {
            case 0: break;
            case 1:
            {
                Deriv fviscosity = ( v[j] - v[i] ) * ( m2 * viscosity / (Pi.density * Pj.density) * Kv.laplacianW(r_h) );
                f[i] += fviscosity;
                f[j] -= fviscosity;
                break;
            }
            case 2:
            {
                Real vx = dot(v[i]-v[j],x[i]-x[j]);
                if (vx < 0)
                {
                    pressureFV += (vx * viscosity * h * m / ((r_h*r_h + 0.01f*h2)*(Pi.density+Pj.density)*0.5f));
                }
                break;
            }
            default:
                break;
            }

            Deriv fpressure = Kp.gradW(x[i]-x[j],r_h) * pressureFV;
            f[i] += fpressure;
            f[j] -= fpressure;

        }

        switch(surfaceTensionT)
        {
        case 0: break;
        case 1:
        {
            Real n = Pi.normal.norm();
            if (n > 0.000001)
            {
                Deriv fsurface = Pi.normal * ( - m * surfaceTension * Pi.curvature / n );
                f[i] += fsurface;
            }
            break;
        }
        case 2:
        {
            break;
        }
        default:
            break;
        }
    }
}


template<class DataTypes>
void SPHFluidForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& f1 = *d_df.beginEdit();
    const VecDeriv& dx1 = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    const VecCoord& p1 = this->mstate->read(core::ConstVecCoordId::position())->getValue();
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

template <class DataTypes>
SReal SPHFluidForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* d_x */) const
{
    msg_error() << "getPotentialEnergy-not-implemented !!!";
    return 0;
}


template<class DataTypes>
void SPHFluidForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields())
        return;

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->disableLighting();
    vparams->drawTool()->enableBlending();
    vparams->drawTool()->disableDepthTest();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    std::vector<sofa::defaulttype::Vec4f> colorVector;
    std::vector<sofa::defaulttype::Vector3> vertices;
    if (d_debugGrid.getValue())
    {
        for (unsigned int i = 0; i < particles.size(); i++)
        {
            Particle& Pi = particles[i];
            if (Pi.neighbors.size() != Pi.neighbors2.size())
            {
                colorVector.push_back(sofa::defaulttype::Vec4f(1, 0, 0, 1));
                for (unsigned int j = 0; j < Pi.neighbors.size(); j++)
                {
                    int index = Pi.neighbors[j].first;
                    unsigned int j2 = 0;
                    while (j2 < Pi.neighbors2.size() && Pi.neighbors2[j2].first != index)
                        ++j2;
                    if (j2 == Pi.neighbors2.size())
                    {
                        vertices.push_back(sofa::defaulttype::Vector3(x[i]));
                        vertices.push_back(sofa::defaulttype::Vector3(x[index]));
                    }
                }
                vparams->drawTool()->drawLines(vertices, 1, colorVector[0]);
                vertices.clear();
                colorVector.clear();

                colorVector.push_back(sofa::defaulttype::Vec4f(1, 0, 1, 1));
                for (unsigned int j = 0; j < Pi.neighbors2.size(); j++)
                {
                    int index = Pi.neighbors2[j].first;
                    unsigned int j2 = 0;
                    while (j2 < Pi.neighbors.size() && Pi.neighbors[j2].first != index)
                        ++j2;
                    if (j2 == Pi.neighbors.size())
                    {
                        vertices.push_back(sofa::defaulttype::Vector3(x[i]));
                        vertices.push_back(sofa::defaulttype::Vector3(x[index]));
                    }
                }
                vparams->drawTool()->drawLines(vertices, 1, colorVector[0]);
                vertices.clear();
                colorVector.clear();
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < particles.size(); i++)
        {
            Particle& Pi = particles[i];
            for (typename std::vector< std::pair<int, Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
            {
                const int j = it->first;
                const float r_h = (float)it->second;
                float f = r_h * 2;
                if (f < 1)
                {
                    colorVector.push_back(sofa::defaulttype::Vec4f(0, 1 - f, f, 1 - r_h));
                }
                else
                {
                    colorVector.push_back(sofa::defaulttype::Vec4f(f - 1, 0, 2 - f, 1 - r_h));
                }
                vertices.push_back(sofa::defaulttype::Vector3(x[i]));
                vertices.push_back(sofa::defaulttype::Vector3(x[j]));
            }
            vparams->drawTool()->drawLines(vertices, 1, colorVector);
            vertices.clear();
            colorVector.clear();
        }
    }

    vparams->drawTool()->disableBlending();
    vparams->drawTool()->enableDepthTest();

    for (unsigned int i=0; i<particles.size(); i++)
    {
        Particle& Pi = particles[i];
        float f = (float)(Pi.density / density0.getValue());
        f = 1+10*(f-1);
        if (f < 1)
        {
            colorVector.push_back(sofa::defaulttype::Vec4f(0,1-f,f,1));
        }
        else
        {
            colorVector.push_back(sofa::defaulttype::Vec4f(f-1,0,2-f,1));
        }
        vertices.push_back(sofa::defaulttype::Vector3(x[i]));
    }

    vparams->drawTool()->drawPoints(vertices,5,colorVector);
    vertices.clear();
    colorVector.clear();

    vparams->drawTool()->restoreLastState();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_INL
