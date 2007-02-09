// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_INL

#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <sofa/component/forcefield/SpringForceField.inl>
#include <assert.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
void StiffSpringForceField<DataTypes>::init()
{
    this->SpringForceField<DataTypes>::init();
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addSpringForce( double& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, const Spring& spring)
{
    int a = spring.m1;
    int b = spring.m2;
    Coord u = p2[b]-p1[a];
    Real d = u.norm();
    Real inverseLength = 1.0f/d;
    u *= inverseLength;
    Real elongation = (Real)(d - spring.initpos);
    potentialEnergy += elongation * elongation * spring.ks / 2;
    //cerr<<"StiffSpringForceField<DataTypes>::addSpringForce, new potential energy = "<<potentialEnergy<<endl;
    Deriv relativeVelocity = v2[b]-v1[a];
    Real elongationVelocity = dot(u,relativeVelocity);
    Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
    Deriv force = u*forceIntensity;
    f1[a]+=force;
    f2[b]-=force;

    Mat3& m = this->dfdx[i];
    Real tgt = forceIntensity * inverseLength;
    for( int j=0; j<3; ++j )
    {
        for( int k=0; k<3; ++k )
        {
            m[j][k] = ((Real)spring.ks-tgt) * u[j] * u[k];
        }
        m[j][j] += tgt;
    }
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addSpringDForce(VecDeriv& f1, const VecDeriv& dx1, VecDeriv& f2, const VecDeriv& dx2, int i, const Spring& spring)
{
    const int a = spring.m1;
    const int b = spring.m2;
    const Coord d = dx2[b]-dx1[a];
    const Deriv dforce = this->dfdx[i]*d;
    f1[a]+=dforce;
    f2[b]-=dforce;
    //cerr<<"StiffSpringForceField<DataTypes>::addSpringDForce, a="<<a<<", b="<<b<<", dforce ="<<dforce<<endl;
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addForce()
{
    assert(this->object1);
    assert(this->object2);
    this->dfdx.resize(this->springs.size());
    VecDeriv& f1 = *this->object1->getF();
    const VecCoord& p1 = *this->object1->getX();
    const VecDeriv& v1 = *this->object1->getV();
    VecDeriv& f2 = *this->object2->getF();
    const VecCoord& p2 = *this->object2->getX();
    const VecDeriv& v2 = *this->object2->getV();
    f1.resize(p1.size());
    f2.resize(p2.size());
    m_potentialEnergy = 0;
    for (unsigned int i=0; i<this->springs.size(); i++)
    {
        this->addSpringForce(m_potentialEnergy,f1,p1,v1,f2,p2,v2, i, this->springs[i]);
    }
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addDForce()
{
    VecDeriv& f1  = *this->object1->getF();
// 	const VecCoord& p1 = *this->object1->getX();
    const VecDeriv& dx1 = *this->object1->getDx();
    VecDeriv& f2  = *this->object2->getF();
// 	const VecCoord& p2 = *this->object2->getX();
    const VecDeriv& dx2 = *this->object2->getDx();
    f1.resize(dx1.size());
    f2.resize(dx2.size());
    //cerr<<"StiffSpringForceField<DataTypes>::addDForce, dx1 = "<<dx1<<endl;
    //cerr<<"StiffSpringForceField<DataTypes>::addDForce, df1 before = "<<f1<<endl;
    for (unsigned int i=0; i<this->springs.size(); i++)
    {
        this->addSpringDForce(f1,dx1,f2,dx2, i, this->springs[i]);
    }
    //cerr<<"StiffSpringForceField<DataTypes>::addDForce, df1 = "<<f1<<endl;
    //cerr<<"StiffSpringForceField<DataTypes>::addDForce, df2 = "<<f2<<endl;
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
