#ifndef SOFA_COMPONENTS_SPHFLUIDFORCEFIELD_INL
#define SOFA_COMPONENTS_SPHFLUIDFORCEFIELD_INL

#include "SPHFluidForceField.h"
#include "SpatialGridContainer.inl"
#include "Common/config.h"
#include "GL/template.h"
#include <math.h>
#include <GL/gl.h>

namespace Sofa
{

namespace Components
{

template<class DataTypes>
SPHFluidForceField<DataTypes>::SPHFluidForceField(Sofa::Core::MechanicalModel<DataTypes>* /*object*/)
    : particleRadius(1), particleMass(1), pressureStiffness(100), density0(1), viscosity(0.001), surfaceTension(0), grid(NULL)
{
}

template<class DataTypes>
void SPHFluidForceField<DataTypes>::init()
{
    grid = new Grid(particleRadius);
    this->Inherit::init();
}

template<class DataTypes>
void SPHFluidForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    const Real h = particleRadius;
    const Real h2 = h*h;
    const Real m = particleMass;
    const Real m2 = m*m;
    const Real k = pressureStiffness;
    const Real d0 = density0;
    const double* localg = this->getContext()->getLocalGravity();
    Deriv g;
    DataTypes::set ( g, localg[0], localg[1], localg[2]);
    //const Deriv mg = g * mass;
    const int n = x.size();

    // Precompute constants for smoothing kernels
    const Real     CWd =     constWd(h);
    //const Real CgradWd = constGradWd(h);
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
    particles.resize(n);
    for (int i=0; i<n; i++)
    {
        particles[i].neighbors.clear();
#ifdef SOFA_DEBUG_SPATIALGRIDCONTAINER
        particles[i].neighbors2.clear();
#endif
        particles[i].density = 0;
        particles[i].pressure = 0;
        particles[i].normal.clear();
        particles[i].curvature = 0;
    }

    // First compute the neighbors
    // This is the only O(n2) step, and should be optimized later
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
        grid->begin();
        for (int i=0; i<n; i++)
        {
            grid->add(i, x[i]);
        }
        grid->end();
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
                std::cerr << "particle "<<i<<" "<< x[i] <<" : "<<particles[i].neighbors.size()<<" neighbors on grid, "<< particles[i].neighbors2.size() << " neighbors on bruteforce.\n";
                std::cerr << "grid-only neighbors:";
                for (unsigned int j=0; j<particles[i].neighbors.size(); j++)
                {
                    int index = particles[i].neighbors[j].first;
                    unsigned int j2 = 0;
                    while (j2 < particles[i].neighbors2.size() && particles[i].neighbors2[j2].first != index)
                        ++j2;
                    if (j2 == particles[i].neighbors2.size())
                        std::cerr << " "<< x[index] << "<"<< particles[i].neighbors[j].first<<","<<particles[i].neighbors[j].second<<">";
                }
                std::cerr << "\n";
                std::cerr << "bruteforce-only neighbors:";
                for (unsigned int j=0; j<particles[i].neighbors2.size(); j++)
                {
                    int index = particles[i].neighbors2[j].first;
                    unsigned int j2 = 0;
                    while (j2 < particles[i].neighbors.size() && particles[i].neighbors[j2].first != index)
                        ++j2;
                    if (j2 == particles[i].neighbors.size())
                        std::cerr << " "<< x[index] << "<"<< particles[i].neighbors2[j].first<<","<<particles[i].neighbors2[j].second<<">";
                }
                std::cerr << "\n";
            }
        }
#endif
    }

    // Compute density and pressure
    for (int i=0; i<n; i++)
    {
        Particle& Pi = particles[i];
        Real density = Pi.density;
        density += m*Wd(0,CWd); // density from current particle
        Deriv n;
        for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
        {
            const int j = it->first;
            const Real r_h = it->second;
            Particle& Pj = particles[j];
            Real d = m*Wd(r_h,CWd);
            density += d;
            Pj.density += d;
        }
        Pi.density = density;
        Pi.pressure = k*(density - d0);
    }

    // Compute surface normal and curvature
    if (surfaceTension > 0)
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
            Deriv fpressure = gradWp(x[i]-x[j],r_h,CgradWp) * ( - m2 * (Pi.pressure / (Pi.density*Pi.density) + Pj.pressure / (Pj.density*Pj.density)) );
            f[i] += fpressure;
            f[j] -= fpressure;

            // Viscosity
            Deriv fviscosity = ( v[j] - v[i] ) * ( m2 * viscosity / (Pi.density * Pj.density) * laplacianWv(r_h,ClaplacianWv) );
            f[i] += fviscosity;
            f[j] -= fviscosity;
        }

        if (surfaceTension > 0)
        {
            Real n = Pi.normal.norm();
            if (n > 0.000001)
            {
                Deriv fsurface = Pi.normal * ( - m * surfaceTension * Pi.curvature / n );
                f[i] += fsurface;
            }
        }
    }
}

template<class DataTypes>
void SPHFluidForceField<DataTypes>::addDForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& /*v*/, const VecDeriv& dx1)
{
    f1.resize(dx1.size());
    for (unsigned int i=0; i<this->dforces.size(); i++)
    {
        const DForce& df = this->dforces[i];
        const unsigned int ia = df.a;
        const unsigned int ib = df.b;
        const Deriv u = p1[ib]-p1[ia];
        const Deriv du = dx1[ib]-dx1[ia];
        const Deriv dforce = u * (df.df * (du*u));
        f1[ia] += dforce;
        f1[ib] -= dforce;
    }
}

template<class DataTypes>
void SPHFluidForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    //if (grid != NULL)
    //	grid->draw();
    VecCoord& x = *this->mmodel->getX();
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
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
                    GL::glVertexT(x[i]);
                    GL::glVertexT(x[index]);
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
                    GL::glVertexT(x[i]);
                    GL::glVertexT(x[index]);
                }
            }
        }
#else
        for (typename std::vector< std::pair<int,Real> >::const_iterator it = Pi.neighbors.begin(); it != Pi.neighbors.end(); ++it)
        {
            const int j = it->first;
            const Real r_h = it->second;
            float f = r_h*2;
            if (f < 1)
            {
                glColor4f(0,1-f,f,1-r_h);
            }
            else
            {
                glColor4f(f-1,0,2-f,1-r_h);
            }
            GL::glVertexT(x[i]);
            GL::glVertexT(x[j]);
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
        Real f = Pi.density / density0;
        f = 1+10*(f-1);
        if (f < 1)
        {
            glColor3f(0,1-f,f);
        }
        else
        {
            glColor3f(f-1,0,2-f);
        }
        GL::glVertexT(x[i]);
    }
    glEnd();
    glPointSize(1);
}

} // namespace Sofa

} // namespace Components

#endif
