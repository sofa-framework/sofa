/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_INL

#include <sofa/component/forcefield/SpringForceField.h>
#include <sofa/core/componentmodel/behavior/PairInteractionForceField.inl>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/config.h>
#include <assert.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{

using std::cerr;
using std::endl;

template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(MechanicalState* mstate1, MechanicalState* mstate2, SReal _ks, SReal _kd)
    : Inherit(mstate1, mstate2)
    , ks(initData(&ks,_ks,"stiffness","uniform stiffness for the all springs"))
    , kd(initData(&kd,_kd,"damping","uniform damping for the all springs"))
    , springs(initData(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
{
}

template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(SReal _ks, SReal _kd)
    : ks(initData(&ks,_ks,"stiffness","uniform stiffness for the all springs"))
    , kd(initData(&kd,_kd,"damping","uniform damping for the all springs"))
    , springs(initData(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
{
}


template<class DataTypes>
void SpringForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    if (arg->getAttribute("filename"))
        this->load(arg->getAttribute("filename"));
    this->Inherit::parse(arg);
}

template <class DataTypes>
class SpringForceField<DataTypes>::Loader : public helper::io::MassSpringLoader
{
public:
    SpringForceField<DataTypes>* dest;
    Loader(SpringForceField<DataTypes>* dest) : dest(dest) {}
    virtual void addSpring(int m1, int m2, SReal ks, SReal kd, SReal initpos)
    {
        helper::vector<Spring>& springs = *dest->springs.beginEdit();
        springs.push_back(Spring(m1,m2,ks,kd,initpos));
        dest->springs.endEdit();
    }
};

template <class DataTypes>
bool SpringForceField<DataTypes>::load(const char *filename)
{
    bool ret = true;
    if (filename && filename[0])
    {
        Loader loader(this);
        ret &= loader.load(filename);
    }
    else ret = false;
    return ret;
}


template <class DataTypes>
void SpringForceField<DataTypes>::reinit()
{
    for (unsigned int i=0; i<springs.getValue().size(); ++i)
    {
        (*springs.beginEdit())[i].ks = ks.getValue()/springs.getValue()[i].initpos;
        (*springs.beginEdit())[i].kd = kd.getValue()/springs.getValue()[i].initpos;
    }
}

template <class DataTypes>
void SpringForceField<DataTypes>::init()
{
    this->Inherit::init();
}

template<class DataTypes>
void SpringForceField<DataTypes>::addSpringForce(SReal& ener, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int /*i*/, const Spring& spring)
{
    int a = spring.m1;
    int b = spring.m2;
    Coord u = p2[b]-p1[a];
    Real d = u.norm();
    Real inverseLength = 1.0f/d;
    if( d>1.0e-4 ) // null length => no force
        return;
    u *= inverseLength;
    Real elongation = (Real)(d - spring.initpos);
    ener += elongation * elongation * spring.ks /2;
    Deriv relativeVelocity = v2[b]-v1[a];
    Real elongationVelocity = dot(u,relativeVelocity);
    Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
    Deriv force = u*forceIntensity;
    f1[a]+=force;
    f2[b]-=force;
}

template<class DataTypes>
void SpringForceField<DataTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2)
{
    f1.resize(x1.size());
    f2.resize(x2.size());
    m_potentialEnergy = 0;
    for (unsigned int i=0; i<this->springs.getValue().size(); i++)
    {
        this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, this->springs.getValue()[i]);
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addDForce(VecDeriv&, VecDeriv&, const VecDeriv&, const VecDeriv&)
{
    std::cerr << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead.\n";
}

template<class DataTypes>
void SpringForceField<DataTypes>::draw()
{
    if (!((this->mstate1 == this->mstate2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->mstate1->getX();
    const VecCoord& p2 = *this->mstate2->getX();
    /*        cerr<<"SpringForceField<DataTypes>::draw() "<<getName()<<endl;
            cerr<<"SpringForceField<DataTypes>::draw(), p1.size = "<<p1.size()<<endl;
            cerr<<"SpringForceField<DataTypes>::draw(), p1 = "<<p1<<endl;
            cerr<<"SpringForceField<DataTypes>::draw(), p2 = "<<p2<<endl;*/
    glDisable(GL_LIGHTING);
    bool external = (this->mstate1!=this->mstate2);
    //if (!external)
    //	glColor4f(1,1,1,1);
    const helper::vector<Spring>& springs = this->springs.getValue();
    glBegin(GL_LINES);
    for (unsigned int i=0; i<springs.size(); i++)
    {
        Real d = (p2[springs[i].m2]-p1[springs[i].m1]).norm();
        if (external)
        {
            if (d<springs[i].initpos*0.9999)
                glColor4f(1,0,0,1);
            else
                glColor4f(0,1,0,1);
        }
        else
        {
            if (d<springs[i].initpos*0.9999)
                glColor4f(1,0.5f,0,1);
            else
                glColor4f(0,1,0.5f,1);
        }
        helper::gl::glVertexT(p1[springs[i].m1]);
        helper::gl::glVertexT(p2[springs[i].m2]);
    }
    glEnd();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
