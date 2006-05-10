#ifndef SOFA_COMPONENTS_SPRINGFORCEFIELD_INL
#define SOFA_COMPONENTS_SPRINGFORCEFIELD_INL

#include "SpringForceField.h"
#include "MassSpringLoader.h"
#include "Scene.h"
#include <assert.h>
#include <iostream>
// added by Sylvere F.
#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/gl.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{

template <class DataTypes>
class SpringForceField<DataTypes>::Loader : public MassSpringLoader
{
public:
    SpringForceField<DataTypes>* dest;
    Loader(SpringForceField<DataTypes>* dest) : dest(dest) {}
    virtual void addSpring(int m1, int m2, double ks, double kd, double initpos)
    {
        dest->springs.push_back(Spring(m1,m2,ks,kd,initpos));
    }
};

template <class DataTypes>
void SpringForceField<DataTypes>::init(const char *filename)
{
    //this->typeName = "SpringForceField";
    //this->_name = name;
    if (filename && filename[0])
    {
        Loader loader(this);
        loader.load(filename);
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addSpringForce(VecDeriv& f1, VecCoord& p1, VecDeriv& v1, VecDeriv& f2, VecCoord& p2, VecDeriv& v2, int /*i*/, const Spring& spring)
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
}

template<class DataTypes>
void SpringForceField<DataTypes>::addForce()
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
    for (unsigned int i=0; i<this->springs.size(); i++)
    {
        this->addSpringForce(f1,p1,v1,f2,p2,v2, i, this->springs[i]);
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addDForce()
{
    std::cerr << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead.\n";
}

template<class DataTypes>
void SpringForceField<DataTypes>::draw()
{
    if (!Scene::getInstance()->getShowForceFields()) return;
    VecCoord& p1 = *this->object1->getX();
    VecCoord& p2 = *this->object2->getX();
    /*        cerr<<"SpringForceField<DataTypes>::draw(), p1.size = "<<p1.size()<<endl;
            cerr<<"SpringForceField<DataTypes>::draw(), p1 = "<<p1<<endl;*/
    glDisable(GL_LIGHTING);
    bool external = (this->object1!=this->object2);
    if (!external)
        glColor4f(1,1,1,1);
    glBegin(GL_LINES);
    for (unsigned int i=0; i<this->springs.size(); i++)
    {
        if (external)
        {
            Real d = (p2[this->springs[i].m2]-p1[this->springs[i].m1]).norm();
            if (d<this->springs[i].initpos*0.9999)
                glColor4f(1,0,0,1);
            else
                glColor4f(0,1,0,1);
        }
        glVertex3d(p1[this->springs[i].m1][0],p1[this->springs[i].m1][1],p1[this->springs[i].m1][2]);
        glVertex3d(p2[this->springs[i].m2][0],p2[this->springs[i].m2][1],p2[this->springs[i].m2][2]);
    }
    glEnd();
}

} // namespace Components

} // namespace Sofa

#endif
