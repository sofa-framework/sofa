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

#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/mass/UniformMass.h>


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
    const VecDeriv& v = *getVfree();
    const VecConst& contacts = *getC();
    Deriv weighedNormal;
    Deriv InvM_wN;

    const sofa::defaulttype::RigidMass* massValue;

    simulation::tree::GNode *node = dynamic_cast<simulation::tree::GNode *>(getContext());
    if (node != NULL)
    {
        core::componentmodel::behavior::BaseMass*_m = node->mass;
        component::mass::UniformMass<defaulttype::RigidTypes, defaulttype::RigidMass> *m = dynamic_cast<component::mass::UniformMass<defaulttype::RigidTypes, defaulttype::RigidMass>*> (_m);
        massValue = 	&( m->getMass().getValue() );
    }
    else
    {
        massValue = new sofa::defaulttype::RigidMass();
        printf("\n WARNING : node is not found => massValue could be false in getCompliance function");
    }

    numContact = contacts.size();

    for(int c1=0; c1<numContact; c1++)
    {
        int sizeC1 = contacts[c1].size();
        for(int i=0; i<sizeC1; i++)
        {
            weighedNormal.getVCenter() = contacts[c1][i].data.getVCenter(); // weighed normal
            weighedNormal.getVOrientation() = contacts[c1][i].data.getVOrientation();

            InvM_wN = weighedNormal / (*massValue); // WARNING massValue is not a double but a massType !!
            // operator / is defined in "RigidTypes.h"
            /*
            			printf("\n contact[%d] = weighedNormal x : %f, y : %f, z : %f, u : %f, v : %f, w : %f",
            				c1, weighedNormal.getVCenter().x(), weighedNormal.getVCenter().y(), weighedNormal.getVCenter().z(),
            				weighedNormal.getVOrientation().x(), weighedNormal.getVOrientation().y(), weighedNormal.getVOrientation().z());

            			printf("\n contact[%d] = InvM_wN x : %f, y : %f, z : %f, u : %f, v : %f, w : %f",
            				c1, InvM_wN.getVCenter().x(), InvM_wN.getVCenter().y(), InvM_wN.getVCenter().z(),
            				InvM_wN.getVOrientation().x(), InvM_wN.getVOrientation().y(), InvM_wN.getVOrientation().z());
            */
//			dfree[c1] += (dot(v[0].getVCenter(), weighedNormal.getVCenter()) +
//				dot(v[0].getVOrientation(), weighedNormal.getVOrientation()))*dt; //0.01;
            for(int c2=c1; c2<numContact; c2++)
            {
                int sizeC2 = contacts[c2].size();
                for(int j=0; j<sizeC2; j++)
                {
//					W[c1][c2] += (dot(weighedNormal.getVCenter(), contacts[c2][j].data.getVCenter()) +
//						dot(weighedNormal.getVOrientation(), contacts[c2][j].data.getVOrientation()))*(dt*dt);//(0.01*0.01);

                    W[c1][c2] +=  contacts[c2][j].data*InvM_wN; // WARNING : this is a dot product defined in "RigidTypes.h"
                    W[c1][c2] *= dt*dt;
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
            if (f[c1+numContact]!=0.0)
            {
                weighedNormal = contacts[c1][i].data; // weighed normal
                force[0].getVCenter() += weighedNormal.getVCenter() * f[c1+numContact];
                force[0].getVOrientation() += weighedNormal.getVOrientation() * f[c1+numContact];
            }
        }
    }
//	printf("f = %f, %f, %f \n", force[0].getVCenter().x(), force[0].getVCenter().y(), force[0].getVCenter().z());
}


template<>
void MechanicalObject<defaulttype::RigidTypes>::resetContactForce()
{
    VecDeriv& force = *this->externalForces;
    for( unsigned i=0; i<force.size(); ++i )
        force[i] = Deriv();
}



// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.

template class MechanicalObject<defaulttype::Vec3fTypes>;
template class MechanicalObject<defaulttype::Vec3dTypes>;
template class MechanicalObject<defaulttype::RigidTypes>;
template class MechanicalObject<defaulttype::LaparoscopicRigidTypes>;

} // namespace component

} // namespace sofa
