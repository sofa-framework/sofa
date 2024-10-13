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
#pragma once

#include <SofaCUDA/component/collision/response/contact/CudaPenalityContactForceField.h>
#include <sofa/component/collision/response/contact/PenalityContactForceField.inl>
#include <sofa/gl/template.h>

namespace sofa::gpu::cuda
{

extern "C"
{
    void PenalityContactForceFieldCuda3f_setContacts(unsigned int size, unsigned int nbTests, unsigned int maxPoints, const void* tests, const void* outputs, void* contacts, float d0, float stiffness, type::Mat3x3f xform);
    void PenalityContactForceFieldCuda3f_addForce(unsigned int size, const void* contacts, void* penetration, void* f1, const void* x1, const void* v1, void* f2, const void* x2, const void* v2);
    void PenalityContactForceFieldCuda3f_addDForce(unsigned int size, const void* contacts, const void* penetration, void* f1, const void* dx1, void* f2, const void* dx2, double factor);
}

} // namespace sofa::gpu::cuda

namespace sofa::component::collision::response::contact
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
void PenalityContactForceField<CudaVec3fTypes>::addContact(int /*m1*/, int /*m2*/, const Deriv& norm, Real dist, Real ks, Real /*mu_s*/, Real /*mu_v*/, int /*oldIndex*/)
{
    /*
    int i = contacts.size();
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
    sofa::type::Vec4f c(norm, dist);
    const Real fact = helper::rsqrt(ks);
    c *= fact;
    contacts.push_back(c);
    pen.push_back(0);
}

void PenalityContactForceField<CudaVec3fTypes>::setContacts(Real d0, Real stiffness, sofacuda::GPUDetectionOutputVector* outputs, bool useDistance, type::Mat3x3f* normXForm)
{
#if 1
    int n = outputs->size();
    contacts.fastResize(n);
    pen.fastResize(n);
    if (!n) return;
    for (int i=0; i<n; i++)
    {
        const sofa::core::collision::GPUContact* o = outputs->get(i);
        const Real distance = (useDistance) ? d0 + o->distance : d0;
        Real ks = (distance > 1.0e-10) ? stiffness / distance : stiffness;
        Coord n = (normXForm)?(*normXForm)*o->normal : o->normal;
        type::Vec4f c(n, distance);
        c *= helper::rsqrt(ks);
        contacts[i] = c;
        pen[i] = 0;
    }
#else
    int n = outputs->size();
    int nt = outputs->nbTests();
    int maxp = 0;
    for (int i=0; i<nt; i++)
        if (outputs->rtest(i).curSize > maxp) maxp = outputs->rtest(i).curSize;
    contacts.fastResize(n);
    pen.fastResize(n);
    type::Mat3x3f xform;
    if (normXForm) xform = *normXForm; else xform.identity();
    PenalityContactForceFieldCuda3f_setContacts(n, nt, maxp, outputs->tests.deviceRead(), outputs->results.deviceRead(), contacts.deviceWrite(), d0, stiffness, xform);
#endif
}

//template<>
void PenalityContactForceField<CudaVec3fTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f1, DataVecDeriv& d_f2, const DataVecCoord& d_x1, const DataVecCoord& d_x2, const DataVecDeriv& d_v1, const DataVecDeriv& d_v2)
{
    VecDeriv& f1 = *d_f1.beginEdit();
    const VecCoord& x1 = d_x1.getValue();
    const VecDeriv& v1 = d_v1.getValue();
    VecDeriv& f2 = *d_f2.beginEdit();
    const VecCoord& x2 = d_x2.getValue();
    const VecDeriv& v2 = d_v2.getValue();

    pen.resize(contacts.size());
    f1.resize(x1.size());
    f2.resize(x2.size());
#if 0
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        sofa::type::Vec4f c = contacts[i];
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
#else
    PenalityContactForceFieldCuda3f_addForce(contacts.size(), contacts.deviceRead(), pen.deviceWrite(),
            f1.deviceWrite(), x1.deviceRead(), v1.deviceRead(),
            f2.deviceWrite(), x2.deviceRead(), v2.deviceRead());
#endif
    d_f1.endEdit();
    d_f2.endEdit();
}

//template<>
void PenalityContactForceField<CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df1, DataVecDeriv& d_df2, const DataVecDeriv& d_dx1, const DataVecDeriv& d_dx2)
{
    VecDeriv& df1 = *d_df1.beginEdit();
    const VecDeriv& dx1 = d_dx1.getValue();
    VecDeriv& df2 = *d_df2.beginEdit();
    const VecDeriv& dx2 = d_dx2.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    df1.resize(dx1.size());
    df2.resize(dx2.size());
#if 0
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        if (pen[i] > 0) // + dpen > 0)
        {
            sofa::type::Vec4f c = contacts[i];
            //Coord du = dx2[c.m2]-dx1[c.m1];
            Coord du = dx2[i]-dx1[i];
            Coord norm(c[0],c[1],c[2]);
            Real dpen = - du*norm;
            //if (c.pen < 0) dpen += c.pen; // start penality at distance 0
            //Real dfN = c.ks * dpen;
            Deriv dforce = -norm*(dpen*kFactor); //dfN;
            df1[i]+=dforce;
            df2[i]-=dforce;
        }
    }
#else
    PenalityContactForceFieldCuda3f_addDForce(contacts.size(), contacts.deviceRead(), pen.deviceWrite(),
            df1.deviceWrite(), dx1.deviceRead(),
            df2.deviceWrite(), dx2.deviceRead(), kFactor);
#endif
    d_df1.endEdit();
    d_df2.endEdit();
}

inline void PenalityContactForceField<CudaVectorTypes<Vec<3>, Vec<3>>>::buildDampingMatrix(core::behavior::DampingMatrix* damping_matrix)
{
    // No damping in this ForceField
}

//template<>
SReal PenalityContactForceField<CudaVec3fTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const
{
    msg_error()<<"PenalityContactForceField::getPotentialEnergy-not-implemented !!!";
    return 0;
}

//template<>
void PenalityContactForceField<CudaVec3fTypes>::draw(const core::visual::VisualParams* vparams)
{
#if SOFACUDA_HAVE_SOFA_GL == 1
    if (!((this->mstate1 == this->mstate2)?  vparams->displayFlags().getShowForceFields():vparams->displayFlags().getShowInteractionForceFields())) return;
    const VecCoord& p1 = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& p2 = this->mstate2->read(core::ConstVecCoordId::position())->getValue();
    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        sofa::type::Vec4f c = contacts[i];
        Coord u = p2[i]-p1[i];
        Coord norm(c[0],c[1],c[2]);
        //c.pen = c.dist - u*c.norm;
        const Real d = c[3] - u*norm;
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
        sofa::gl::glVertexT(p1[i]); //c.m1]);
        sofa::gl::glVertexT(p2[i]); //c.m2]);
    }
    glEnd();

    //if (getContext()->getShowNormals())
    {
        glColor4f(1,1,0,1);
        glBegin(GL_LINES);
        for (unsigned int i=0; i<contacts.size(); i++)
        {
            sofa::type::Vec4f c = contacts[i];
            Coord norm(c[0],c[1],c[2]); norm.normalize();
            Coord p = p1[i] - norm*0.1;
            sofa::gl::glVertexT(p1[i]);
            sofa::gl::glVertexT(p);
            p = p2[i] + norm*0.1;
            sofa::gl::glVertexT(p2[i]);
            sofa::gl::glVertexT(p);
        }
        glEnd();
    }
#endif // SOFACUDA_HAVE_SOFA_GL == 1
}

} // namespace sofa::component::collision::response::contact
