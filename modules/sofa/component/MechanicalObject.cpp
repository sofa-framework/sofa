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
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/component/MechanicalObject.inl>


namespace sofa
{

namespace component
{

using namespace core::componentmodel::behavior;
using namespace defaulttype;

SOFA_DECL_CLASS(MechanicalObject)

int MechanicalObjectVec3fClass = core::RegisterObject("mechanical state vectors")
        .add< MechanicalObject<Vec3dTypes> >(true) // default template
        .add< MechanicalObject<Vec3fTypes> >()
        .add< MechanicalObject<RigidTypes> >()
        .add< MechanicalObject<LaparoscopicRigidTypes> >()
        ;

// template specialization must be in the same namespace as original namespace for GCC 4.1

template <>
void MechanicalObject<defaulttype::Vec3dTypes>::getIndicesInSpace(std::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const
{
    const VecCoord& x = *getX();
    for( unsigned i=0; i<x.size(); ++i )
    {
        if( x[i][0] >= xmin && x[i][0] <= xmax && x[i][1] >= ymin && x[i][1] <= ymax && x[i][2] >= zmin && x[i][2] <= zmax )
        {
            indices.push_back(i);
        }
    }
}

template <>
void MechanicalObject<defaulttype::Vec3fTypes>::getIndicesInSpace(std::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const
{
    const VecCoord& x = *getX();
    for( unsigned i=0; i<x.size(); ++i )
    {
        if( x[i][0] >= xmin && x[i][0] <= xmax && x[i][1] >= ymin && x[i][1] <= ymax && x[i][2] >= zmin && x[i][2] <= zmax )
        {
            indices.push_back(i);
        }
    }
}

// overload for rigid bodies: use the center
template<>
void MechanicalObject<defaulttype::RigidTypes>::getIndicesInSpace(std::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const
{
    const VecCoord& x = *getX();
    for( unsigned i=0; i<x.size(); ++i )
    {
        if( x[i].getCenter()[0] >= xmin && x[i].getCenter()[0] <= xmax && x[i].getCenter()[1] >= ymin && x[i].getCenter()[1] <= ymax && x[i].getCenter()[2] >= zmin && x[i].getCenter()[2] <= zmax )
        {
            indices.push_back(i);
        }
    }
}


template<>
void MechanicalObject<defaulttype::RigidTypes>::getCompliance (double dt, double**W, double *dfree, int &numContact)
{
    const VecDeriv& v = *getV();
    const VecConst& contacts = *getC();
    Deriv weighedNormal;

    numContact = contacts.size();

    for(int c1=0; c1<numContact; c1++)
    {
        int sizeC1 = contacts[c1].size();
        for(int i=0; i<sizeC1; i++)
        {
            weighedNormal.getVCenter() = contacts[c1][i].data.getVCenter(); // weighed normal
            weighedNormal.getVOrientation() = contacts[c1][i].data.getVOrientation();
            dfree[c1] += (dot(v[0].getVCenter(), weighedNormal.getVCenter()) +
                    dot(v[0].getVOrientation(), weighedNormal.getVOrientation()))*dt; //0.01;
            for(int c2=c1; c2<numContact; c2++)
            {
                int sizeC2 = contacts[c2].size();
                for(int j=0; j<sizeC2; j++)
                {
                    W[c1][c2] += (dot(weighedNormal.getVCenter(), contacts[c2][j].data.getVCenter()) +
                            dot(weighedNormal.getVOrientation(), contacts[c2][j].data.getVOrientation()))*(dt*dt);//(0.01*0.01);
                }
            }
            for(int c2=c1+1; c2<numContact; c2++)
            {
                W[c2][c1] = W[c1][c2];
            }
        }
    }
}

template<>
void MechanicalObject<defaulttype::RigidTypes>::applyContactForce(double *f)
{
    VecDeriv& force = *this->externalForces;
    const VecConst& contacts = *getC();
    Deriv weighedNormal;

    int numContact = contacts.size();

    force.resize(0);
    force.resize(1);
    force[0] = Deriv();

    for(int c1=0; c1<numContact; c1++)
    {
        int sizeC1 = contacts[c1].size();
        for(int i=0; i<sizeC1; i++)
        {
            if (f[c1+numContact]!=0)
            {
                weighedNormal = contacts[c1][i].data; // weighed normal
                force[0].getVCenter() += weighedNormal.getVCenter() * f[c1+numContact];
                force[0].getVOrientation() += weighedNormal.getVOrientation() * f[c1+numContact];
            }
        }
    }
}


// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.

template class MechanicalObject<defaulttype::Vec3fTypes>;
template class MechanicalObject<defaulttype::Vec3dTypes>;
template class MechanicalObject<defaulttype::RigidTypes>;
template class MechanicalObject<defaulttype::LaparoscopicRigidTypes>;

} // namespace component

} // namespace sofa
