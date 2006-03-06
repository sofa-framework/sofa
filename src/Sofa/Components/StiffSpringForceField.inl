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
    if (f1.size()<p1.size())
    {
        //std::cout << "Set F1 size "<<f1.size()<<"->"<<p1.size()<<std::endl;
        f1.resize(p1.size());
    }
    if (f2.size()<p2.size())
    {
        //std::cout << "Set F2 size "<<f2.size()<<"->"<<p2.size()<<std::endl;
        f2.resize(p2.size());
    }
    for (unsigned int i=0; i<this->springs.size(); i++)
    {
        int a = this->springs[i].m1;
        int b = this->springs[i].m2;
        Coord u = p2[b]-p1[a];
        Real d = u.norm();
        Real inverseLength = 1.0f/d;
        u *= inverseLength;
        Real elongation = d - this->springs[i].initpos;
        Deriv relativeVelocity = v2[b]-v1[a];
        Real elongationVelocity = dot(u,relativeVelocity);
        Real forceIntensity = this->springs[i].ks*elongation+this->springs[i].kd*elongationVelocity;
        Deriv force = u*forceIntensity;
        f1[a]+=force;
        f2[b]-=force;

        Mat3& m = this->dfdx[i];
        Real tgt = forceIntensity * inverseLength;
        for( int j=0; j<3; ++j )
        {
            for( int k=0; k<3; ++k )
            {
                m[j][k] = (this->springs[i].ks-tgt) * u[j] * u[k];
            }
            m[j][j] += tgt;
        }
    }
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addDForce()
{
    VecDeriv& dx1 = *this->object1->getDx();
    VecDeriv& f1  = *this->object1->getF();
    VecDeriv& dx2 = *this->object2->getDx();
    VecDeriv& f2  = *this->object2->getF();
    f1.resize(dx1.size());
    f2.resize(dx2.size());
    for (unsigned int i=0; i<this->springs.size(); i++)
    {
        int a = this->springs[i].m1;
        int b = this->springs[i].m2;
        Coord d = dx2[b]-dx1[a];
        Deriv dforce = this->dfdx[i]*d;
        f1[a]+=dforce;
        f2[b]-=dforce;
    }
}

} // namespace Components

} // namespace Sofa

#endif
