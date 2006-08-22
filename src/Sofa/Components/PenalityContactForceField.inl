#ifndef SOFA_COMPONENTS_PENALITYCONTACTFORCEFIELD_INL
#define SOFA_COMPONENTS_PENALITYCONTACTFORCEFIELD_INL

#include "Sofa/Core/ForceField.inl"
#include "PenalityContactForceField.h"
#include "Common/config.h"
#include <assert.h>
#include <GL/gl.h>
#include "GL/template.h"

namespace Sofa
{

namespace Components
{

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addContact(int m1, int m2, const Deriv& norm, Real dist, Real ks, Real mu_s, Real mu_v)
{
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
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addForce()
{
    assert(this->object1);
    assert(this->object2);
    VecDeriv& f1 = *this->object1->getF();
    VecCoord& p1 = *this->object1->getX();
    VecDeriv& v1 = *this->object1->getV();
    VecDeriv& f2 = *this->object2->getF();
    VecCoord& p2 = *this->object2->getX();
    VecDeriv& v2 = *this->object2->getV();
    f1.resize(p1.size());
    f2.resize(p2.size());
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        Coord u = p2[c.m2]-p1[c.m1];
        c.pen = c.dist - u*c.norm;
        if (c.pen > 0)
        {
            Real fN = c.ks * c.pen;
            Deriv v = v2[c.m2]-v1[c.m1];
            v -= v*(v*c.norm); // project velocity to normal plane
            Real fV = fN * c.mu_v;
            Deriv force = -c.norm*fN + v*fV;
            f1[c.m1]+=force;
            f2[c.m2]-=force;
        }
    }
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addDForce()
{
    VecDeriv& f1  = *this->object1->getF();
    //VecCoord& p1 = *this->object1->getX();
    VecDeriv& dx1 = *this->object1->getDx();
    VecDeriv& f2  = *this->object2->getF();
    //VecCoord& p2 = *this->object2->getX();
    VecDeriv& dx2 = *this->object2->getDx();
    f1.resize(dx1.size());
    f2.resize(dx2.size());
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = contacts[i];
        Coord du = dx2[c.m2]-dx1[c.m1];
        Real dpen = - du*c.norm;
        if (c.pen > 0) // + dpen > 0)
        {
            if (c.pen < 0) dpen += c.pen; // start penality at distance 0
            Real dfN = c.ks * dpen;
            Deriv dforce = -c.norm*dfN;
            f1[c.m1]+=dforce;
            f2[c.m2]-=dforce;
        }
    }
}


template<class DataTypes>
void PenalityContactForceField<DataTypes>::draw()
{
    if (!((this->object1 == this->object2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields())) return;
    VecCoord& p1 = *this->object1->getX();
    VecCoord& p2 = *this->object2->getX();
    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = contacts[i];
        Real d = c.dist - (p2[c.m2]-p1[c.m1])*c.norm;
        if (d > 0)
            glColor4f(1,0,0,1);
        else
            glColor4f(0,1,0,1);
        GL::glVertexT(p1[c.m1]);
        GL::glVertexT(p2[c.m2]);
    }
    glEnd();

    if (getContext()->getShowNormals())
    {
        glColor4f(1,1,0,1);
        glBegin(GL_LINES);
        for (unsigned int i=0; i<contacts.size(); i++)
        {
            const Contact& c = contacts[i];
            Coord p = p1[c.m1] - c.norm;
            GL::glVertexT(p1[c.m1]);
            GL::glVertexT(p);
            p = p2[c.m2] + c.norm;
            GL::glVertexT(p2[c.m2]);
            GL::glVertexT(p);
        }
        glEnd();
    }
}


} // namespace Components

} // namespace Sofa

#endif
