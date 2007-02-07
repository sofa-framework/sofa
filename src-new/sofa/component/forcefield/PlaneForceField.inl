#ifndef SOFA_COMPONENTS_PLANEFORCEFIELD_INL
#define SOFA_COMPONENTS_PLANEFORCEFIELD_INL

#include "Sofa-old/Core/ForceField.inl"
#include "PlaneForceField.h"
#include "Common/config.h"
#include <assert.h>
#include <GL/gl.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{

template<class DataTypes>
void PlaneForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1)
{
    //this->dfdd.resize(p1.size());
    this->contacts.clear();
    f1.resize(p1.size());
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Real d = p1[i]*planeNormal-planeD;
        if (d<0)
        {
            Real forceIntensity = -this->stiffness*d;
            Real dampingIntensity = -this->damping*d;
            Deriv force = planeNormal*forceIntensity - v1[i]*dampingIntensity;
            f1[i]+=force;
            //this->dfdd[i] = -this->stiffness;
            this->contacts.push_back(i);
        }
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::addDForce(VecDeriv& f1, const VecDeriv& dx1)
{
    f1.resize(dx1.size());
    for (unsigned int i=0; i<this->contacts.size(); i++)
    {
        unsigned int p = this->contacts[i];
        assert(p<dx1.size());
        f1[p] += planeNormal * (-this->stiffness * (dx1[p]*planeNormal));
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::updateStiffness( const VecCoord& x )
{
    this->contacts.clear();
    for (unsigned int i=0; i<x.size(); i++)
    {
        Real d = x[i]*planeNormal-planeD;
        if (d<0)
        {
            this->contacts.push_back(i);
        }
    }
}


template <class DataTypes>
double PlaneForceField<DataTypes>::getPotentialEnergy(const VecCoord&)
{
    cerr<<"PlaneForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}


template<class DataTypes>
void PlaneForceField<DataTypes>::rotate( Deriv axe, Real angle )
{
    Deriv v;
    v = planeNormal.cross(axe);
    v.normalize();

    planeNormal = planeNormal * cos( angle ) + v * sin( angle );
}



template<class DataTypes>
void PlaneForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    draw2();
}


template<class DataTypes>
void PlaneForceField<DataTypes>::draw2(float size)
{
    if (!getContext()->getShowForceFields()) return;

    const VecCoord& p1 = *this->mmodel->getX();




    // un vecteur quelconque du plan
    Deriv v1;
    if( 0.0 != planeNormal[0] ) v1 = Deriv((-planeNormal[2]-planeNormal[1])/planeNormal[0], 1.0, 1.0);
    else if ( 0.0 != planeNormal[1] ) v1 = Deriv(1.0, (-planeNormal[0]-planeNormal[2])/planeNormal[1],1.0);
    else if ( 0.0 != planeNormal[2] ) v1 = Deriv(1.0, 1.0, (-planeNormal[0]-planeNormal[1])/planeNormal[2]);
    v1.normalize();
    // un deuxiement vecteur quelconque du plan orthogonal au premier
    Deriv v2;
    v2 = v1.cross(planeNormal);
    v2.normalize();

    Coord center = planeNormal*planeD;
    Coord q0 = center-v1*size-v2*size;
    Coord q1 = center+v1*size-v2*size;
    Coord q2 = center+v1*size+v2*size;
    Coord q3 = center-v1*size+v2*size;

// 	glEnable(GL_LIGHTING);
    glEnable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glCullFace(GL_FRONT);

    glColor3d(_color.getValue()[0],_color.getValue()[1],_color.getValue()[2]);



    glBegin(GL_QUADS);
    glVertex3d(q0[0],q0[1],q0[2]);
    glVertex3d(q1[0],q1[1],q1[2]);
    glVertex3d(q2[0],q2[1],q2[2]);
    glVertex3d(q3[0],q3[1],q3[2]);
    glEnd();


    glDisable(GL_CULL_FACE);

    glColor4f(1,0,0,1);
    glDisable(GL_LIGHTING);
    // lignes pour les points passés dessous
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



    /*
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
    */
}




} // namespace Components

} // namespace Sofa

#endif
