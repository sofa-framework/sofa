#ifndef SOFA_COMPONENTS_PLANEFORCEFIELD_INL
#define SOFA_COMPONENTS_PLANEFORCEFIELD_INL

#include "PlaneForceField.h"
#include "Scene.h"
#include <assert.h>
#include <GL/gl.h>

namespace Sofa
{

namespace Components
{

template<class DataTypes>
void PlaneForceField<DataTypes>::addForce()
{
    assert(this->object);
    VecDeriv& f1 = *this->object->getF();
    VecCoord& p1 = *this->object->getX();
    //VecDeriv& v1 = *this->object->getV();
    //this->dfdd.resize(p1.size());
    this->contacts.clear();
    f1.resize(p1.size());
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Real d = p1[i]*planeNormal-planeD;
        if (d<0)
        {
            Real forceIntensity = -this->stiffness*d;
            Deriv force = planeNormal*forceIntensity;
            f1[i]+=force;
            //this->dfdd[i] = -this->stiffness;
            this->contacts.push_back(i);
        }
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::addDForce()
{
    VecDeriv& f1  = *this->object->getF();
    //VecCoord& p1 = *this->object->getX();
    VecDeriv& dx1 = *this->object->getDx();
    f1.resize(dx1.size());
    for (unsigned int i=0; i<this->contacts.size(); i++)
    {
        unsigned int p = this->contacts[i];
        f1[p] += planeNormal * (-this->stiffness * (dx1[p]*planeNormal));
    }
}


template<class DataTypes>
void PlaneForceField<DataTypes>::draw()
{
    if (!Scene::getInstance()->getShowForceFields()) return;
    VecCoord& p1 = *this->object->getX();
    glDisable(GL_LIGHTING);
    glColor4f(1,0,0,1);
    glBegin(GL_LINES);
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Real d = p1[i]*planeNormal-planeD;
        Coord p2 = p1[i];
        p2 += planeNormal*(-d);
        if (d<0)
        {
            glVertex3d(p1[i][0],p1[i][1],p1[i][2]);
            glVertex3d(p2[0],p2[1],p2[2]);
        }
    }
    glEnd();
    glPointSize(1);
    glColor4f(0,1,0,1);
    glBegin(GL_POINTS);
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Real d = p1[i]*planeNormal-planeD;
        Coord p2 = p1[i];
        p2 += planeNormal*(-d);
        if (d>=0)
            glVertex3d(p2[0],p2[1],p2[2]);
    }
    glEnd();
}


} // namespace Components

} // namespace Sofa

#endif
