#ifndef SOFA_COMPONENTS_LENNARDJONESFORCEFIELD_INL
#define SOFA_COMPONENTS_LENNARDJONESFORCEFIELD_INL

#include "LennardJonesForceField.h"
#include "Common/config.h"
#include "GL/template.h"
#include <math.h>
#include <GL/gl.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{

template<class DataTypes>
void LennardJonesForceField<DataTypes>::init()
{
    this->Inherit::init();

    assert( this->mmodel );

    a = (p0 * (Real)pow(d0,alpha)) / (1-alpha/beta);
    b = (p0 * (Real)pow(d0,beta)) / (beta/alpha-1);
    std::cout << "Lennard-Jones initialized: alpha="<<alpha<<" beta="<<beta<<" d0="<<d0<<" p0="<<p0<<" a="<<a<<" b="<<b<<std::endl;
    // Validity check: compute force and potential at d0
    Real f0 = a*alpha*(Real)pow(d0,-alpha-1)-b*beta*(Real)pow(d0,-beta-1);
    if (fabs(f0)>0.001)
        std::cerr << "Lennard-Jones initialization failed: f0="<<f0<<std::endl;
    Real cp0 = (a*(Real)pow(d0,-alpha)-b*(Real)pow(d0,-beta));
    if (fabs(cp0/p0-1)>0.001)
        std::cerr << "Lennard-Jones initialization failed: cp0="<<cp0<<std::endl;
    // Debug
    for (Real d = 0; d<dmax; d+= dmax/60)
    {
        Real f = a*alpha*(Real)pow(d,-alpha-1)-b*beta*(Real)pow(d,-beta-1);
        std::cout << "f("<<d<<")="<<f<<std::endl;
    }
}

template<class DataTypes>
void LennardJonesForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1)
{
    Real dmax2 = dmax*dmax;
    this->dforces.clear();
    f1.resize(p1.size());
    for (unsigned int ib=1; ib<p1.size(); ib++)
    {
        const Coord pb = p1[ib];
        for (unsigned int ia=0; ia<ib; ia++)
        {
            const Coord pa = p1[ia];
            const Deriv u = pb-pa;
            const Real d2 = u.norm2();
            if (d2 >= dmax2) continue;
            const Real d = (Real)sqrt(d2);
            const Real fa = a*alpha*(Real)pow(d,-alpha-1);
            const Real fb = b*beta*(Real)pow(d,-beta-1);
            Real forceIntensity = fa - fb;
            //std::cout << ia<<"-"<<ib<<" d="<<d<<" f="<<forceIntensity<<std::endl;
            DForce df;
            df.a = ia;
            df.b = ib;
            if (forceIntensity > fmax)
            {
                forceIntensity = fmax;
                df.df = 0;
            }
            else
            {
                df.df = ((-alpha-1)*fa - (-beta-1)*fb)/(d*d2);
            }
            this->dforces.push_back(df);
            Deriv force = u*(forceIntensity/d);

            // Add damping
            force += (v1[ib]-v1[ia])*damping;

            f1[ia]+=force;
            f1[ib]-=force;
        }
    }
}

template<class DataTypes>
void LennardJonesForceField<DataTypes>::addDForce(VecDeriv& f1, const VecDeriv& dx1)
{
    const VecCoord& p1 = *this->mmodel->getX();
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

template <class DataTypes>
double LennardJonesForceField<DataTypes>::getPotentialEnergy(const VecCoord& )
{
    cerr<<"LennardJonesForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}


template<class DataTypes>
void LennardJonesForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    VecCoord& p1 = *this->mmodel->getX();
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    const Real d02 = this->d0*this->d0;
    for (unsigned int i=0; i<this->dforces.size(); i++)
    {
        const DForce& df = this->dforces[i];
        if ((p1[df.b]-p1[df.a]).norm2() < d02)
            glColor4f(1,1,1,1);
        else
            glColor4f(0,0,1,1);
        GL::glVertexT(p1[df.a]);
        GL::glVertexT(p1[df.b]);
    }
    glEnd();
}

} // namespace Sofa

} // namespace Components

#endif
