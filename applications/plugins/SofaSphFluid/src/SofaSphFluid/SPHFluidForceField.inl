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
#ifndef SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_INL

#include <SofaSphFluid/SPHFluidForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaSphFluid/SpatialGridContainer.inl>
#include <cmath>
#include <iostream>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
SPHFluidForceField<DataTypes>::SPHFluidForceField()
    : d_particleRadius (initData(&d_particleRadius, Real(1), "radius", "Radius of a Particle"))
    , d_particleMass (initData(&d_particleMass, Real(1), "mass", "Mass of a Particle"))
    , d_pressureStiffness (initData(&d_pressureStiffness, Real(100), "pressure", "Pressure"))
    , d_density0 (initData(&d_density0, Real(1), "density", "Density"))
    , d_viscosity (initData(&d_viscosity, Real(0.001f), "viscosity", "Viscosity"))
    , d_surfaceTension (initData(&d_surfaceTension, Real(0), "surfaceTension", "Surface Tension"))
    , d_kernelType(initData(&d_kernelType, 0, "kernelType", "0 = default kernels, 1 = cubic spline"))
    , d_pressureType(initData(&d_pressureType, 1, "pressureType", "0 = none, 1 = default pressure"))
    , d_viscosityType(initData(&d_viscosityType, 1, "viscosityType", "0 = none, 1 = default d_viscosity using kernel Laplacian, 2 = artificial d_viscosity"))
    , d_surfaceTensionType(initData(&d_surfaceTensionType, 1, "surfaceTensionType", "0 = none, 1 = default surface tension using kernel Laplacian, 2 = cohesion forces surface tension from Becker et al. 2007"))
    , d_debugGrid(initData(&d_debugGrid, false, "debugGrid", "If true will store additionnal information on the grid to check neighbors and draw them"))
    , m_grid(nullptr)
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

    this->getContext()->get(m_grid); //new Grid(d_particleRadius.getValue());
    if (m_grid==nullptr)
        msg_error() << "SpatialGridContainer not found by SPHFluidForceField, slow O(n2) method will be used !!!";

    size_t n = this->mstate->getSize();
    m_particles.resize(n);
    for (unsigned i=0u; i<n; i++)
    {
        m_particles[i].neighbors.clear();
        m_particles[i].density = d_density0.getValue();
        m_particles[i].pressure = 0;
        m_particles[i].normal.clear();
        m_particles[i].curvature = 0;
    }

    if (d_debugGrid.getValue())
    {
        for (unsigned i = 0u; i < n; i++)
        {
            m_particles[i].neighbors2.clear();
        }
    }

    m_lastTime = (Real)this->getContext()->getTime();
}


template<class DataTypes>
void SPHFluidForceField<DataTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    computeNeighbors(mparams, d_x, d_v);

    switch(d_kernelType.getValue())
    {
    default:
        msg_error() << "Unsupported d_kernelType " << d_kernelType.getValue();
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

    msg_info() << "density[" << 0 << "] = " << m_particles[0].density  << "(" << m_particles[0].neighbors.size() << " neighbors)"
               << "density[" << m_particles.size()/2 << "] = " << m_particles[m_particles.size()/2].density ;
}


template<class DataTypes>
void SPHFluidForceField<DataTypes>::computeNeighbors(const core::MechanicalParams* /*mparams*/, const DataVecCoord& d_x, const DataVecDeriv& /*d_v*/)
{
    helper::ReadAccessor<DataVecCoord> x = d_x;

    const Real h = d_particleRadius.getValue();
    const Real h2 = h*h;

    size_t n = x.size();
    m_particles.resize(n);
    for (size_t i=0; i<n; i++) {
        m_particles[i].neighbors.clear();
    }

    // First compute the neighbors
    // This is an O(n2) step, except if a hash-grid is used to optimize it
    if (m_grid == nullptr)
    {
        for (size_t i = 0; i<n; i++)
        {
            const Coord& ri = x[i];
            for (size_t j = i + 1; j<n; j++)
            {
                const Coord& rj = x[j];
                Real r2 = (rj - ri).norm2();
                if (r2 < h2)
                {
                    Real r_h = (Real)sqrt(r2 / h2);
                    m_particles[i].neighbors.push_back(std::make_pair(j, r_h));
                    //m_particles[j].neighbors.push_back(std::make_pair(i,r_h));
                }
            }
        }
    }
    else
    {
        m_grid->updateGrid(x.ref());
        m_grid->findNeighbors(this, h);

        if (!d_debugGrid.getValue())
            return;

        for (size_t i = 0; i < n; i++) {
            m_particles[i].neighbors2.clear();
        }

        // Check grid info
        for (size_t i=0; i<n; i++)
        {
            const Coord& ri = x[i];
            for (size_t j=i+1; j<n; j++)
            {
                const Coord& rj = x[j];
                Real r2 = (rj-ri).norm2();
                if (r2 < h2)
                {
                    Real r_h = (Real)sqrt(r2/h2);
                    m_particles[i].neighbors2.push_back(std::make_pair(j,r_h));
                }
            }
        }
        for (size_t i=0; i<n; i++)
        {
            if (m_particles[i].neighbors.size() != m_particles[i].neighbors2.size())
            {
                msg_error() << "particle "<<i<<" "<< x[i] <<" : "<<m_particles[i].neighbors.size()<<" neighbors on grid, "<< m_particles[i].neighbors2.size() << " neighbors on bruteforce.";
                msg_error() << "grid-only neighbors:";
                for (unsigned int j=0; j<m_particles[i].neighbors.size(); j++)
                {
                    int index = m_particles[i].neighbors[j].first;
                    unsigned int j2 = 0;
                    while (j2 < m_particles[i].neighbors2.size() && m_particles[i].neighbors2[j2].first != index)
                        ++j2;
                    if (j2 == m_particles[i].neighbors2.size())
                        msg_error() << " "<< x[index] << "<"<< m_particles[i].neighbors[j].first<<","<<m_particles[i].neighbors[j].second<<">";
                }
                msg_error() << "";
                msg_error() << "bruteforce-only neighbors:";
                for (unsigned int j=0; j<m_particles[i].neighbors2.size(); j++)
                {
                    int index = m_particles[i].neighbors2[j].first;
                    unsigned int j2 = 0;
                    while (j2 < m_particles[i].neighbors.size() && m_particles[i].neighbors[j2].first != index)
                        ++j2;
                    if (j2 == m_particles[i].neighbors.size())
                        msg_error() << " "<< x[index] << "<"<< m_particles[i].neighbors2[j].first<<","<<m_particles[i].neighbors2[j].second<<">";
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

    const Real h = d_particleRadius.getValue();
    const Real h2 = h*h;
    const Real m = d_particleMass.getValue();
    const Real m2 = m*m;
    const Real d0 = d_density0.getValue();
    const Real k = d_pressureStiffness.getValue();
    const Real time = (Real)this->getContext()->getTime();
    const Real viscosity = d_viscosity.getValue();
    const int viscosityT = (viscosity == 0) ? 0 : d_viscosityType.getValue();
    const Real surfaceTension = d_surfaceTension.getValue();
    const int surfaceTensionT = (surfaceTension <= 0) ? 0 : d_surfaceTensionType.getValue();
    //const Real dt = (Real)this->getContext()->getDt();
    m_lastTime = time;

    size_t n = x.size();

    // Initialization
    f.resize(n);
    dforces.clear();
    //int n0 = m_particles.size();
    m_particles.resize(n);
    for (size_t i=0; i<n; i++)
    {
        m_particles[i].density = 0;
        m_particles[i].pressure = 0;
        m_particles[i].normal.clear();
        m_particles[i].curvature = 0;
    }

    TKd Kd(h);
    TKp Kp(h);
    TKv Kv(h);
    TKc Kc(h);

    // Compute density and pressure
    for (size_t i=0; i<n; i++)
    {
        Particle& Pi = m_particles[i];
        Real density = Pi.density;

        density += m*Kd.W(0); // density from current particle

        for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
        {
            const int j = it->first;
            const Real r_h = it->second;
            Particle& Pj = m_particles[j];
            Real d = m*Kd.W(r_h);
            density += d;
            Pj.density += d;

        }
        Pi.density = density;
        Pi.pressure = k*(density - d0);
    }

    // Compute surface normal and curvature
    if (surfaceTensionT == 1)
    {
        for (size_t i=0; i<n; i++)
        {
            Particle& Pi = m_particles[i];
            for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
            {
                const int j = it->first;
                const Real r_h = it->second;
                Particle& Pj = m_particles[j];
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
    for (size_t i = 0; i < n; i++)
    {
        const Particle& Pi = m_particles[i];
        // Gravity
        //f[i] += g*(m*Pi.density);
        
        for (auto it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
        {
            const int j = it->first;
            const Real r_h = it->second;
            const Particle& Pj = m_particles[j];
            // Pressure

            Real pressureFV = ( - m2 * (Pi.pressure / (Pi.density*Pi.density) + Pj.pressure / (Pj.density*Pj.density)) );

            // Viscosity
            switch(viscosityT)
            {
            case 0: break;
            case 1:
            {
                Deriv fd_viscosity = ( v[j] - v[i] ) * ( m2 * viscosity / (Pi.density * Pj.density) * Kv.laplacianW(r_h) );
                f[i] += fd_viscosity;
                f[j] -= fd_viscosity;
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

    std::vector<sofa::helper::types::RGBAColor> colorVector;
    std::vector<sofa::defaulttype::Vector3> vertices;
    if (d_debugGrid.getValue())
    {
        for (unsigned int i = 0; i < m_particles.size(); i++)
        {
            Particle& Pi = m_particles[i];
            if (Pi.neighbors.size() != Pi.neighbors2.size())
            {
                colorVector.push_back(sofa::helper::types::RGBAColor::red());
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

                colorVector.push_back(sofa::helper::types::RGBAColor::magenta());
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
        for (unsigned int i = 0; i < m_particles.size(); i++)
        {
            Particle& Pi = m_particles[i];
            for (typename std::vector< std::pair<int, Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
            {
                const int j = it->first;
                const float r_h = (float)it->second;
                float f = r_h * 2;
                if (f < 1)
                {
                    colorVector.push_back({0.0f, 1.0f - f, f, 1.0f - r_h});
                }
                else
                {
                    colorVector.push_back({f - 1.0f, 0.0f, 2.0f - f, 1.0f - r_h});
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

    for (unsigned int i=0; i<m_particles.size(); i++)
    {
        Particle& Pi = m_particles[i];
        float f = (float)(Pi.density / d_density0.getValue());
        f = 1+10*(f-1);
        if (f < 1)
        {
            colorVector.push_back({0.0f, 1.0f - f, f, 1.0f});
        }
        else
        {
            colorVector.push_back( { f - 1.0f, 0.0f, 2.0f - f, 1.0f});
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
