#ifndef SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include "EllipsoidForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>
#include <assert.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{

// v = sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1
// dv/dxj = xj/rj^2 * 1/sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)

// f  = -stiffness * v * (dv/dp) / norm(dv/dp)

// fi = -stiffness * (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1) * (xi/ri^2) / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)

// dfi/dxj = -stiffness * [ d(sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)/dxj *   (xi/ri^2) / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)     * d(xi/ri^2)/dxj / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)     *  (xi/ri^2) * d(1/sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4))/dxj ]
// dfi/dxj = -stiffness * [ xj/rj^2 * 1/sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2) * (xi/ri^2) / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)       * (i==j)/ri^2 / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)       * (xi/ri^2) * (-1/2*2xj/rj^4*1/(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4) ]
// dfi/dxj = -stiffness * [ xj/rj^2 * 1/sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2) * (xi/ri^2) / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)       * (i==j)/ri^2 / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)       * (xi/ri^2) * (-xj/rj^4*1/(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4) ]

// dfi/dxj = -stiffness * [ (xj/rj^2) * (xi/ri^2) * 1/(sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2) * sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4))
//                          +  v       * (i==j) / (ri^2*sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4))
//                          +  v       * (xi/ri^2) * (xj/rj^2) * 1/(rj^2*(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4) ]


template<class DataTypes>
void EllipsoidForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1)
{
    const Coord center = this->center.getValue();
    const Coord r = this->vradius.getValue();
    const Real stiffness = this->stiffness.getValue();
    const Real stiffabs = helper::rabs(stiffness);
    //const Real s2 = (stiff < 0 ? - stiff*stiff : stiff*stiff );
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
        //Real d = (norm2-1)*s2;
        if ((norm2-1)*stiffness<0)
        {
            Real norm = helper::rsqrt(norm2);
            Real v = norm-1;
            Coord grad;
            for (int j=0; j<N; j++) grad[j] = dp[j]*inv_r2[j];
            Real gnorm2 = grad.norm2();
            Real gnorm = helper::rsqrt(gnorm2);
            //grad /= gnorm; //.normalize();
            Real forceIntensity = -stiffabs*v/gnorm;
            Real dampingIntensity = this->damping.getValue()*helper::rabs(v);
            Deriv force = forceIntensity*grad - v1[i]*dampingIntensity;
            f1[i]+=force;
            Contact c;
            c.index = i;
            Real fact1 = -stiffabs / (norm * gnorm);
            Real fact2 = -stiffabs*v / gnorm;
            Real fact3 = -stiffabs*v / gnorm2;
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
void EllipsoidForceField<DataTypes>::addDForce(VecDeriv& df1, const VecDeriv& dx1, double kFactor, double /*bFactor*/)
{
    df1.resize(dx1.size());
    const sofa::helper::vector<Contact>& contacts = this->contacts.getValue();
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = contacts[i];
        assert((unsigned)c.index<dx1.size());
        Deriv du = dx1[c.index];
        Deriv dforce = c.m * du;
        dforce *= kFactor;
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

    Real cx=0, cy=0, cz=0;
    DataTypes::get(cx, cy, cz, center.getValue());
    Real rx=1, ry=1, rz=1;
    DataTypes::get(rx, ry, rz, vradius.getValue());
    glEnable(GL_CULL_FACE);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glColor3f(color.getValue()[0],color.getValue()[1],color.getValue()[2]);
    glPushMatrix();
    glTranslated(cx, cy, cz);
    glScaled(rx, ry, (stiffness.getValue()>0?rz:-rz));
    glutSolidSphere(1,32,16);
    glPopMatrix();
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
