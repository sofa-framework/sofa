#ifndef SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include "EllipsoidForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/helper/rmath.h>
#include <assert.h>
#include <iostream>
#if defined (__APPLE__)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

namespace sofa
{

namespace component
{

namespace forcefield
{

// v = sqrt(x0²/r0²+x1²/r1²+x2²/r2²)-1
// dv/dxj = xj/rj² * 1/sqrt(x0²/r0²+x1²/r1²+x2²/r2²)

// f  = -stiffness * v * normalize(dv/dp)
// f  = -stiffness * v * normalize(vec(x0/r0²,x1/r1²,x2/r2²))

// fi = -stiffness * (sqrt(x0²/r0²+x1²/r1²+x2²/r2²)-1) * (xi/ri²) / sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4)

// dfi/dxj = -stiffness * [ d(sqrt(x0²/r0²+x1²/r1²+x2²/r2²)-1)/dxj *   (xi/ri²) / sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4)
//	                      +  (sqrt(x0²/r0²+x1²/r1²+x2²/r2²)-1)     * d(xi/ri²)/dxj / sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4)
//	                      +  (sqrt(x0²/r0²+x1²/r1²+x2²/r2²)-1)     *  (xi/ri²) * d(1/sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4))/dxj ]
// dfi/dxj = -stiffness * [ xj/rj² * 1/sqrt(x0²/r0²+x1²/r1²+x2²/r2²) * (xi/ri²) / sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4)
//	                      +  (sqrt(x0²/r0²+x1²/r1²+x2²/r2²)-1)       * (i==j)/ri² / sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4)
//	                      +  (sqrt(x0²/r0²+x1²/r1²+x2²/r2²)-1)       * (xi/ri²) * (-1/2*2xj/rj^4*1/(x0²/r0^4+x1²/r1^4+x2²/r2^4) ]
// dfi/dxj = -stiffness * [ xj/rj² * 1/sqrt(x0²/r0²+x1²/r1²+x2²/r2²) * (xi/ri²) / sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4)
//	                      +  (sqrt(x0²/r0²+x1²/r1²+x2²/r2²)-1)       * (i==j)/ri² / sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4)
//	                      +  (sqrt(x0²/r0²+x1²/r1²+x2²/r2²)-1)       * (xi/ri²) * (-xj/rj^4*1/(x0²/r0^4+x1²/r1^4+x2²/r2^4) ]

// dfi/dxj = -stiffness * [ (xj/rj²) * (xi/ri²) * 1/(sqrt(x0²/r0²+x1²/r1²+x2²/r2²) * sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4))
//	                      +  v       * (i==j) / (ri²*sqrt(x0²/r0^4+x1²/r1^4+x2²/r2^4))
//	                      +  v       * (xi/ri²) * (xj/rj²) * 1/(rj²*(x0²/r0^4+x1²/r1^4+x2²/r2^4) ]


template<class DataTypes>
void EllipsoidForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1)
{
    const Coord center = this->center.getValue();
    const Coord r = this->vradius.getValue();
    const Real stiff = this->stiffness.getValue();
    Coord inv_r2;
    for (int j=0; j<N; j++) inv_r2[j] = 1/(r[j]*r[j]);
    sofa::helper::vector<Contact>* contacts = this->contacts.beginEdit();
    contacts->clear();
    f1.resize(p1.size());
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Coord dp = p1[i] - center;
        Real norm2 = 0;
        for (int j=0; j<N; j++) norm2 += (dp[j]*dp[j])*inv_r2[j];
        Real d = (norm2-1)*stiff;
        if (d<0)
        {
            d = helper::rsqrt(d*stiff);
            Coord grad;
            for (int j=0; j<N; j++) grad[j] = dp[j]*inv_r2[j];
            Real gnorm2 = grad.norm2();
            Real gnorm = helper::rsqrt(gnorm2);
            //grad /= gnorm; //.normalize();
            Real norm = helper::rsqrt(norm2);
            Real forceIntensity = -stiff*d/gnorm;
            Real dampingIntensity = -this->damping.getValue()*d;
            Deriv force = forceIntensity*grad - v1[i]*dampingIntensity;
            f1[i]+=force;
            Contact c;
            c.index = i;
            Real fact1 = -stiff / (helper::rsqrt(norm2) * gnorm);
            Real fact2 = d / gnorm;
            Real fact3 = d / gnorm2;
            for (int ci = 0; ci < N; ++ci)
            {
                for (int cj = 0; cj < N; ++cj)
                    c.m[ci][cj] = grad[ci]*grad[cj] * (fact1 + fact3*inv_r2[cj]);
                c.m[ci][ci] += fact2*inv_r2[ci];
            }
            contacts->push_back(c);
        }
    }
    this->contacts.endEdit();
}

template<class DataTypes>
void EllipsoidForceField<DataTypes>::addDForce(VecDeriv& df1, const VecDeriv& dx1)
{
    df1.resize(dx1.size());
    const sofa::helper::vector<Contact>& contacts = this->contacts.getValue();
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = contacts[i];
        assert((unsigned)c.index<dx1.size());
        Deriv du = dx1[c.index];
        Deriv dforce = c.m * du;
        df1[c.index] += dforce;
    }
}

template <class DataTypes>
double EllipsoidForceField<DataTypes>::getPotentialEnergy(const VecCoord&)
{
    std::cerr<<"EllipsoidForceField::getPotentialEnergy-not-implemented !!!"<<std::endl;
    return 0;
}

template<class DataTypes>
void EllipsoidForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    if (!bDraw.getValue()) return;

    double cx=0, cy=0, cz=0;
    DataTypes::get(cx, cy, cz, center.getValue());
    double rx=1, ry=1, rz=1;
    DataTypes::get(rx, ry, rz, vradius.getValue());

    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glColor3f(0.0f, 0.0f, 1.0f);
    glPushMatrix();
    glTranslated(cx, cy, cz);
    glScaled(rx, ry, rz);
    glutSolidSphere(1,32,16); // slightly reduce rendered radius
    glPopMatrix();
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
