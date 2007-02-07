#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_REPULSIVESPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_REPULSIVESPRINGFORCEFIELD_INL

#include <sofa/component/interactionforcefield/RepulsiveSpringForceField.h>
#include <sofa/component/forcefield/StiffSpringForceField.inl>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
void RepulsiveSpringForceField<DataTypes>::addForce()
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
    for (unsigned int i=0; i<this->springs.size(); i++)
    {
#if 1
        int a = this->springs[i].m1;
        int b = this->springs[i].m2;
        Coord u = p2[b]-p1[a];
        Real d = u.norm();
        if (d < this->springs[i].initpos)
        {
            Real inverseLength = 1.0f/d;
            u *= inverseLength;
            Real elongation = (Real)(d - this->springs[i].initpos);
            Deriv relativeVelocity = v2[b]-v1[a];
            Real elongationVelocity = dot(u,relativeVelocity);
            Real forceIntensity = (Real)(this->springs[i].ks*elongation+this->springs[i].kd*elongationVelocity);
            Deriv force = u*forceIntensity;
            f1[a]+=force;
            f2[b]-=force;

            Mat3& m = this->dfdx[i];
            Real tgt = forceIntensity * inverseLength;
            for( int j=0; j<3; ++j )
            {
                for( int k=0; k<3; ++k )
                {
                    m[j][k] = ((Real)this->springs[i].ks-tgt) * u[j] * u[k];
                }
                m[j][j] += tgt;
            }
        }
        else
#endif
        {
            Mat3& m = this->dfdx[i];
            for( int j=0; j<3; ++j )
                for( int k=0; k<3; ++k )
                    m[j][k] = 0.0;
        }
    }
}

template <class DataTypes>
double RepulsiveSpringForceField<DataTypes>::getPotentialEnergy()
{
    cerr<<"RepulsiveSpringForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
