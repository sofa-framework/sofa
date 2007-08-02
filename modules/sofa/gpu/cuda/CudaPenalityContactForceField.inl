#ifndef SOFA_GPU_CUDA_CUDAPENALITYCONTACTFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAPENALITYCONTACTFORCEFIELD_INL

#include "CudaPenalityContactForceField.h"
#include <sofa/component/forcefield/PenalityContactForceField.inl>
#if 0
namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void PenalityContactForceFieldCuda3f_addForce(unsigned int size, const void* contacts, void* penetration, void* f, const void* x, const void* v);
    void PenalityContactForceFieldCuda3f_addDForce(unsigned int size, const void* contacts, const void* penetration, void* f, const void* dx);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;

//template<>
void PenalityContactForceField<CudaVec3fTypes>::clear(int reserve)
{
    //prevContacts.swap(contacts); // save old contacts in prevContacts
    contacts.clear();
    pen.clear();
    if (reserve)
    {
        contacts.reserve(reserve);
        pen.reserve(reserve);
    }
}

//template<>
void PenalityContactForceField<CudaVec3fTypes>::addContact(int m1, int m2, const Deriv& norm, Real dist, Real ks, Real mu_s, Real mu_v, int oldIndex)
{
    int i = contacts.size();
    /*
    contacts.resize(i+1);
    Contact& c = contacts[i];
    c.m1 = m1;
    c.m2 = m2;
    c.norm = norm;
    c.dist = dist;
    c.ks = ks;
    c.mu_s = mu_s;
    c.mu_v = mu_v;
    c.pen = 0;
    if (oldIndex > 0 && oldIndex <= (int)prevContacts.size())
    {
       c.age = prevContacts[oldIndex-1].age+1;
    }
    else
    {
       c.age = 0;
    }
    */
    sofa::defaulttype::Vec4f c(norm, dist);
    Real fact = rsqrt(ks);
    c *= fact;
    contacts.push_back(c);
    pen.push_back(0);
}

//template<>
void PenalityContactForceField<CudaVec3fTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& /*v1*/, const VecDeriv& /*v2*/)
{
    f1.resize(x1.size());
    f2.resize(x2.size());
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        sofa::defaulttype::Vec4f c = contacts[i];
        //Coord u = x2[c.m2]-x1[c.m1];
        Coord u = x2[i]-x1[i];
        Coord norm(c[0],c[1],c[2]);
        //c.pen = c.dist - u*c.norm;
        Real p = c[3] - u*norm;
        pen[i] = p;
        if (p > 0)
        {
            //Real fN = c.ks * c.pen;
            Deriv force = -norm*p; //fN;
            f1[i]+=force;
            f2[i]-=force;
        }
    }
}

//template<>
void PenalityContactForceField<CudaVec3fTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2)
{
    df1.resize(dx1.size());
    df2.resize(dx2.size());
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        sofa::defaulttype::Vec4f c = contacts[i];
        if (pen[i] > 0) // + dpen > 0)
        {
            //Coord du = dx2[c.m2]-dx1[c.m1];
            Coord du = dx2[i]-dx1[i];
            Coord norm(c[0],c[1],c[2]);
            Real dpen = - du*norm;
            //if (c.pen < 0) dpen += c.pen; // start penality at distance 0
            //Real dfN = c.ks * dpen;
            Deriv dforce = -norm*dpen; //dfN;
            df1[i]+=dforce;
            df2[i]-=dforce;
        }
    }
}

//template<>
double PenalityContactForceField<CudaVec3fTypes>::getPotentialEnergy(const VecCoord&, const VecCoord&)
{
    cerr<<"PenalityContactForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}

//template<>
void PenalityContactForceField<CudaVec3fTypes>::draw()
{
    if (!((this->mstate1 == this->mstate2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->mstate1->getX();
    const VecCoord& p2 = *this->mstate2->getX();
    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        sofa::defaulttype::Vec4f c = contacts[i];
        Coord u = p2[i]-p1[i];
        Coord norm(c[0],c[1],c[2]);
        //c.pen = c.dist - u*c.norm;
        Real d = c[3] - u*norm;
        /*if (c.age > 10) //c.spen > c.mu_s * c.ks * 0.99)
            if (d > 0)
                glColor4f(1,0,1,1);
            else
                glColor4f(0,1,1,1);
        else*/
        if (d > 0)
            glColor4f(1,0,0,1);
        else
            glColor4f(0,1,0,1);
        helper::gl::glVertexT(p1[i]); //c.m1]);
        helper::gl::glVertexT(p2[i]); //c.m2]);
    }
    glEnd();
    /*
        if (getContext()->getShowNormals())
        {
            glColor4f(1,1,0,1);
            glBegin(GL_LINES);
            for (unsigned int i=0; i<contacts.size(); i++)
            {
                const Contact& c = contacts[i];
                Coord p = p1[c.m1] - c.norm;
                helper::gl::glVertexT(p1[c.m1]);
                helper::gl::glVertexT(p);
                p = p2[c.m2] + c.norm;
                helper::gl::glVertexT(p2[c.m2]);
                helper::gl::glVertexT(p);
            }
            glEnd();
        }*/
}


/*
template <>
void PenalityContactForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.sphere.center = sphereCenter.getValue();
    data.sphere.r = sphereRadius.getValue();
    data.sphere.stiffness = stiffness.getValue();
    data.sphere.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PenalityContactForceFieldCuda3f_addForce(x.size(), &data.sphere, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PenalityContactForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df, const VecCoord& dx)
{
    df.resize(dx.size());
    PenalityContactForceFieldCuda3f_addDForce(dx.size(), &data.sphere, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
}
*/

} // namespace forcefield

} // namespace component

} // namespace sofa
#endif
#endif
