#ifndef SOFA_COMPONENTS_STIFFSPRINGFORCEFIELD_INL
#define SOFA_COMPONENTS_STIFFSPRINGFORCEFIELD_INL

#include "StiffSpringForceField.h"
#include "SpringForceField.inl"
#include <assert.h>

namespace Sofa
{

namespace Components
{

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addSpringForce(VecDeriv& f1, VecCoord& p1, VecDeriv& v1, VecDeriv& f2, VecCoord& p2, VecDeriv& v2, int i, const Spring& spring)
{
    int a = spring.m1;
    int b = spring.m2;
    Coord u = p2[b]-p1[a];
    Real d = u.norm();
    Real inverseLength = 1.0f/d;
    u *= inverseLength;
    Real elongation = d - spring.initpos;
    Deriv relativeVelocity = v2[b]-v1[a];
    Real elongationVelocity = dot(u,relativeVelocity);
    Real forceIntensity = spring.ks*elongation+spring.kd*elongationVelocity;
    Deriv force = u*forceIntensity;
    f1[a]+=force;
    f2[b]-=force;

    Mat3& m = this->dfdx[i];
    Real tgt = forceIntensity * inverseLength;
    for( int j=0; j<3; ++j )
    {
        for( int k=0; k<3; ++k )
        {
            m[j][k] = (spring.ks-tgt) * u[j] * u[k];
        }
        m[j][j] += tgt;
    }
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addSpringDForce(VecDeriv& f1, VecCoord& /*p1*/, VecDeriv& dx1, VecDeriv& f2, VecCoord& /*p2*/, VecDeriv& dx2, int i, const Spring& spring)
{
    const int a = spring.m1;
    const int b = spring.m2;
    const Coord d = dx2[b]-dx1[a];
    const Deriv dforce = this->dfdx[i]*d;
    f1[a]+=dforce;
    f2[b]-=dforce;
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addForce()
{
    assert(this->object1);
    assert(this->object2);
    this->dfdx.resize(this->springs.size());
    VecDeriv& f1 = *this->object1->getF();
    VecCoord& p1 = *this->object1->getX();
    VecDeriv& v1 = *this->object1->getV();
    VecDeriv& f2 = *this->object2->getF();
    VecCoord& p2 = *this->object2->getX();
    VecDeriv& v2 = *this->object2->getV();
    f1.resize(p1.size());
    f2.resize(p2.size());
    for (unsigned int i=0; i<this->springs.size(); i++)
    {
        this->addSpringForce(f1,p1,v1,f2,p2,v2, i, this->springs[i]);
    }
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addDForce()
{
    VecDeriv& f1  = *this->object1->getF();
    VecCoord& p1 = *this->object1->getX();
    VecDeriv& dx1 = *this->object1->getDx();
    VecDeriv& f2  = *this->object2->getF();
    VecCoord& p2 = *this->object2->getX();
    VecDeriv& dx2 = *this->object2->getDx();
    f1.resize(dx1.size());
    f2.resize(dx2.size());
    for (unsigned int i=0; i<this->springs.size(); i++)
    {
        this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, i, this->springs[i]);
    }
}

} // namespace Components

} // namespace Sofa

#endif
