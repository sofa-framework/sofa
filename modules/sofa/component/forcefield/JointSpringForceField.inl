/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_JOINTSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_JOINTSPRINGFORCEFIELD_INL

#include <sofa/component/forcefield/JointSpringForceField.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Cylinder.h>
#include <sofa/helper/system/config.h>
#include <assert.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
JointSpringForceField<DataTypes>::JointSpringForceField(MechanicalState* object1, MechanicalState* object2, Vec _kst, Vec _ksr, double _kd)
    : Inherit(object1, object2)
    , kst(dataField(&kst,_kst,"stiffnessTranslation","uniform stiffness for the all springs"))
    , ksr(dataField(&ksr,_ksr,"stiffnessRotation","uniform stiffness for the all springs"))
    , kd(dataField(&kd,_kd,"damping","uniform damping for the all springs"))
    , springs(dataField(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
{
}

template<class DataTypes>
JointSpringForceField<DataTypes>::JointSpringForceField(Vec _kst, Vec _ksr, double _kd)
    : kst(dataField(&kst,_kst,"stiffnessTranslation","uniform stiffness for the all springs"))
    , ksr(dataField(&ksr,_ksr,"stiffnessRotation","uniform stiffness for the all springs"))
    , kd(dataField(&kd,_kd,"damping","uniform damping for the all springs"))
    , springs(dataField(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
{
}



template <class DataTypes>
class JointSpringForceField<DataTypes>::Loader : public helper::io::MassSpringLoader
{
public:
    JointSpringForceField<DataTypes>* dest;
    Loader(JointSpringForceField<DataTypes>* dest) : dest(dest) {}
    virtual void addSpring(int m1, int m2, Vec kst, Vec ksr, Real kd)
    {
        helper::vector<Spring>& springs = *dest->springs.beginEdit();
        springs.push_back(Spring(m1,m2,kst, ksr,kd));
        dest->springs.endEdit();
    }
};

template <class DataTypes>
bool JointSpringForceField<DataTypes>::load(const char *filename)
{
    if (filename && filename[0])
    {
        Loader loader(this);
        return loader.load(filename);
    }
    else return false;
}

template <class DataTypes>
void JointSpringForceField<DataTypes>::init()
{
    this->Inherit::init();
}

template<class DataTypes>
void JointSpringForceField<DataTypes>::addSpringForce( double& /*potentialEnergy*/, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int , const Spring& spring)
{
    int a = spring.m1;
    int b = spring.m2;

    Mat Mr01, Mr10, Mr02, Mr20, Mr01t, Mr10t, Mr02t, Mr20t;
    p1[a].writeRotationMatrix(Mr01);
    Mr01t.transpose(Mr01);
    invertMatrix(Mr10, Mr01);
    //Mr10t.transpose(Mr10);
    p2[b].writeRotationMatrix(Mr02);
    Mr02t.transpose(Mr02);
    //invertMatrix(Mr20, Mr02);
    //Mr20t.transpose(Mr20);

    Vec damping(spring.kd, spring.kd, spring.kd);

    springRef[a] = p1[a];

    Coord Mp1p2 = p2[b] - p1[a];
    Deriv Vp1p2 = v2[b] - v1[a];

    //compute elongation
    Mp1p2.getCenter() -= spring.initTrans;
    //compute torsion
    Mp1p2.getOrientation() = spring.initRot.inverse() * Mp1p2.getOrientation();

    //compute directional force (relative translation is expressed in world coordinates)
    Vec f0 = Mr01 * (spring.kst.linearProduct(Mr10 * Mp1p2.getCenter())) + damping.linearProduct(Vp1p2.getVCenter());
    //compute rotational force (relative orientation is expressed in p1)
    Vec R0 = Mr01 * (spring.ksr.linearProduct(Mp1p2.getOrientation().toEulerVector())) + damping.linearProduct(Vp1p2.getVOrientation());
    /*
    	Vec ksr0 = Mr10 * spring.ksr;
    	std::cout<<" ksr : "<<spring.ksr<<endl;
    	std::cout<<" ksr0 : "<<ksr0<<endl;

    	Vec R0;
    	R0  = cross(Mr01t[0],Mr02t[0]) * spring.ksr[0];
    	R0 += cross(Mr01t[1],Mr02t[1]) * spring.ksr[1];
    	R0 += cross(Mr01t[2],Mr02t[2]) * spring.ksr[2];
    */
    const Deriv force(f0, R0 );

    //affect forces
    f1[a] += force;
    f2[b] -= force;

    //potentialEnergy = ????;

}

template<class DataTypes>
void JointSpringForceField<DataTypes>::addSpringDForce(VecDeriv& f1, const VecDeriv& dx1, VecDeriv& f2, const VecDeriv& dx2, int , const Spring& spring)
{
    const int a = spring.m1;
    const int b = spring.m2;
    const Deriv Mdx1dx2 = dx2[b]-dx1[a];

    Mat Mr01, Mr10, Mr02, Mr20, Mr01t, Mr10t, Mr02t, Mr20t;
    springRef[a].writeRotationMatrix(Mr01);
    Mr01t.transpose(Mr01);
    invertMatrix(Mr10, Mr01);
    //Mr10t.transpose(Mr10);
    springRef[b].writeRotationMatrix(Mr02);
    Mr02t.transpose(Mr02);
    //invertMatrix(Mr20, Mr02);
    //Mr20t.transpose(Mr20);

    //compute directional force
    Vec df0 = Mr01 * (spring.kst.linearProduct(Mr10*Mdx1dx2.getVCenter() ));
    //compute rotational force
    Vec dR0 = Mr01 * (spring.ksr.linearProduct(Mr10* Mdx1dx2.getVOrientation()));
    /*
    	Vec dR0;
    	dR0  = cross( cross(Mdx1dx2.getVOrientation(),Mr01t[0]) , Mr02t[0]) * -spring.ksr[0];
    	dR0 += cross( cross(Mdx1dx2.getVOrientation(),Mr01t[1]) , Mr02t[1]) * -spring.ksr[1];
    	dR0 += cross( cross(Mdx1dx2.getVOrientation(),Mr01t[2]) , Mr02t[2]) * -spring.ksr[2];
    */

    const Deriv dforce(df0,dR0);

    f1[a]+=dforce;
    f2[b]-=dforce;

}

template<class DataTypes>
void JointSpringForceField<DataTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2)
{
    helper::vector<Spring>& springs = *this->springs.beginEdit();

    springRef.resize(x1.size());
    f1.resize(x1.size());
    f2.resize(x2.size());
    m_potentialEnergy = 0;
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, springs[i]);
    }
    this->springs.endEdit();
}

template<class DataTypes>
void JointSpringForceField<DataTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2)
{
    df1.resize(dx1.size());
    df2.resize(dx2.size());

    const helper::vector<Spring>& springs = this->springs.getValue();
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringDForce(df1,dx1,df2,dx2, i, springs[i]);
    }
}

template<class DataTypes>
void JointSpringForceField<DataTypes>::draw()
{
    if (!((this->mstate1 == this->mstate2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->mstate1->getX();
    const VecCoord& p2 = *this->mstate2->getX();

    glDisable(GL_LIGHTING);
    bool external = (this->mstate1!=this->mstate2);
    const helper::vector<Spring>& springs = this->springs.getValue();

    for (unsigned int i=0; i<springs.size(); i++)
    {
        Real d = (p2[springs[i].m2]-p1[springs[i].m1]).getCenter().norm();
        if (external)
        {
            if (d<springs[i].initTrans.norm()*0.9999)
                glColor4f(1,0,0,1);
            else
                glColor4f(0,1,0,1);
        }
        else
        {
            if (d<springs[i].initTrans.norm()*0.9999)
                glColor4f(1,0.5f,0,1);
            else
                glColor4f(0,1,0.5f,1);
        }
        glBegin(GL_LINES);
        helper::gl::glVertexT(p1[springs[i].m1].getCenter());
        helper::gl::glVertexT(p2[springs[i].m2].getCenter());
        glEnd();

        if(springs[i].ksr[0] < 1000)
        {
            helper::gl::Cylinder::draw(p1[springs[i].m1].getCenter(), p1[springs[i].m1].getOrientation(), Vec(1,0,0));
        }
        if(springs[i].ksr[1] < 1000)
        {
            helper::gl::Cylinder::draw(p1[springs[i].m1].getCenter(), p1[springs[i].m1].getOrientation(), Vec(0,1,0));
        }
        if(springs[i].ksr[2] < 1000)
        {
            helper::gl::Cylinder::draw(p1[springs[i].m1].getCenter(), p1[springs[i].m1].getOrientation(), Vec(0,0,1));
        }
    }

}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif

