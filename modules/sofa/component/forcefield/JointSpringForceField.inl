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
    : object1(object1), object2(object2)
    , kst(dataField(&kst,_kst,"stiffnessTranslation","uniform stiffness for the all springs"))
    , ksr(dataField(&ksr,_ksr,"stiffnessRotation","uniform stiffness for the all springs"))
    , kd(dataField(&kd,_kd,"damping","uniform damping for the all springs"))
    , springs(dataField(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
{
}

template<class DataTypes>
JointSpringForceField<DataTypes>::JointSpringForceField(Vec _kst, Vec _ksr, double _kd)
    : object1(NULL), object2(NULL)
    , kst(dataField(&kst,_kst,"stiffnessTranslation","uniform stiffness for the all springs"))
    , ksr(dataField(&ksr,_ksr,"stiffnessRotation","uniform stiffness for the all springs"))
    , kd(dataField(&kd,_kd,"damping","uniform damping for the all springs"))
    , springs(dataField(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
{
}



template<class DataTypes>
void JointSpringForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    if (arg->getAttribute("filename"))
        this->load(arg->getAttribute("filename"));
    this->InteractionForceField::parse(arg);
}

template <class DataTypes>
class JointSpringForceField<DataTypes>::Loader : public helper::io::MassSpringLoader
{
public:
    JointSpringForceField<DataTypes>* dest;
    Loader(JointSpringForceField<DataTypes>* dest) : dest(dest) {}
    virtual void addSpring(int m1, int m2, Vec kst, Vec ksr, Real kd)
    {
        vector<Spring>& springs = *dest->springs.beginEdit();
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
    this->InteractionForceField::init();
    if( object1==NULL )
    {
        sofa::core::objectmodel::BaseObject* mstate = getContext()->getMechanicalState();
        assert(mstate!=NULL);
        MechanicalState* state = dynamic_cast<MechanicalState*>(mstate );
        assert( state!= NULL );
        object1 = object2 = state;
    }
}

template<class DataTypes>
void JointSpringForceField<DataTypes>::addSpringForce( double& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, const Spring& spring)
{
    int a = spring.m1;
    int b = spring.m2;

    Mat Mr01, Mr10;
    p1[a].writeRotationMatrix(Mr01);
    invertMatrix(Mr10, Mr01);
    Vec damping(spring.kd, spring.kd, spring.kd);

    springRef[a] = p1[a];

    Coord Mp1p2 = p2[b] - p1[a];
    Deriv Vp1p2 = v2[b] - v1[a];

    //compute elongation
    Mp1p2.getCenter() -= spring.initTrans;
    //compute torsion
    Mp1p2.getOrientation() = spring.initRot.inverse() * Mp1p2.getOrientation();

    //compute directional force
    Vec f0 = Mr01 * (spring.kst.linearProduct(Mr10 * Mp1p2.getCenter())) + damping.linearProduct(Vp1p2.getVCenter());
    //compute rotational force
    Vec R0 = Mr01 * (spring.ksr.linearProduct(/*Mr10 * */Mp1p2.getOrientation().toEulerVector())) + damping.linearProduct(Vp1p2.getVOrientation());

    const Deriv force(f0, R0 );

    //affect forces
    f1[a] += force;
    f2[b] -= force;

    //potentialEnergy = ????;

}

template<class DataTypes>
void JointSpringForceField<DataTypes>::addSpringDForce(VecDeriv& f1, const VecDeriv& dx1, VecDeriv& f2, const VecDeriv& dx2, int i, const Spring& spring)
{
    const int a = spring.m1;
    const int b = spring.m2;
    const Deriv Mdx1dx2 = dx2[b]-dx1[a];

    Mat Mr01, Mr10;
    springRef[a].writeRotationMatrix(Mr01);
    invertMatrix(Mr10, Mr01);

    //compute directional force
    Vec f0 = Mr01 * (spring.kst.linearProduct(Mr10*Mdx1dx2.getVCenter() ));
    //compute rotational force
    Vec R0 = Mr01 * (spring.ksr.linearProduct(Mr10* Mdx1dx2.getVOrientation()));

    const Deriv dforce(f0,R0);

    f1[a]+=dforce;
    f2[b]-=dforce;

}

template<class DataTypes>
void JointSpringForceField<DataTypes>::addForce()
{
    assert(this->object1);
    assert(this->object2);
    const vector<Spring>& springs= this->springs.getValue();
    VecDeriv& f1 = *this->object1->getF();
    const VecCoord& p1 = *this->object1->getX();
    const VecDeriv& v1 = *this->object1->getV();
    VecDeriv& f2 = *this->object2->getF();
    const VecCoord& p2 = *this->object2->getX();
    const VecDeriv& v2 = *this->object2->getV();
    springRef.resize(p1.size());
    f1.resize(p1.size());
    f2.resize(p2.size());
    m_potentialEnergy = 0;
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringForce(m_potentialEnergy,f1,p1,v1,f2,p2,v2, i, springs[i]);
    }
}

template<class DataTypes>
void JointSpringForceField<DataTypes>::addDForce()
{
    VecDeriv& f1  = *this->object1->getF();
    const VecDeriv& dx1 = *this->object1->getDx();

    VecDeriv& f2  = *this->object2->getF();
    const VecDeriv& dx2 = *this->object2->getDx();

    f1.resize(dx1.size());
    f2.resize(dx2.size());

    const vector<Spring>& springs = this->springs.getValue();
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringDForce(f1,dx1,f2,dx2, i, springs[i]);
    }
}

template<class DataTypes>
void JointSpringForceField<DataTypes>::draw()
{
    if (!((this->object1 == this->object2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->object1->getX();
    const VecCoord& p2 = *this->object2->getX();

    glDisable(GL_LIGHTING);
    bool external = (this->object1!=this->object2);
    const vector<Spring>& springs = this->springs.getValue();
    glBegin(GL_LINES);
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
        helper::gl::glVertexT(p1[springs[i].m1].getCenter());
        helper::gl::glVertexT(p2[springs[i].m2].getCenter());
    }
    glEnd();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif

