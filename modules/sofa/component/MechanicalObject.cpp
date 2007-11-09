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
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/component/MechanicalObject.inl>

#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/mass/UniformMass.h>

#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace component
{

using namespace core::componentmodel::behavior;
using namespace defaulttype;

SOFA_DECL_CLASS(MechanicalObject)

int MechanicalObjectClass = core::RegisterObject("mechanical state vectors")
        .add< MechanicalObject<Vec3dTypes> >(true) // default template
        .add< MechanicalObject<Vec3fTypes> >()
        .add< MechanicalObject<Rigid3dTypes> >()
        .add< MechanicalObject<Rigid3fTypes> >()
        .add< MechanicalObject<LaparoscopicRigid3Types> >()
        .add< MechanicalObject<Vec2dTypes> >()
        .add< MechanicalObject<Vec2fTypes> >()
        .add< MechanicalObject<Rigid2dTypes> >()
        .add< MechanicalObject<Rigid2fTypes> >()
        .add< MechanicalObject<Vec1dTypes> >()
        .add< MechanicalObject<Vec1fTypes> >()
        .add< MechanicalObject<Vec6dTypes> >()
        .add< MechanicalObject<Vec6fTypes> >()
        ;

// template specialization must be in the same namespace as original namespace for GCC 4.1
template<>
void MechanicalObject<defaulttype::Rigid3Types>::getCompliance(double**W)
{
    const VecConst& constraints = *getC();
    Deriv weighedNormal;
    Deriv InvM_wN;

    const sofa::defaulttype::Rigid3Mass* massValue;

    simulation::tree::GNode *node = dynamic_cast<simulation::tree::GNode *>(getContext());

    if (node != NULL)
    {
        core::componentmodel::behavior::BaseMass*_m = node->mass;
        component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass> *m = dynamic_cast<component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass>*> (_m);
        massValue = &( m->getMass());//.getValue() );
    }
    else
    {
        massValue = new sofa::defaulttype::Rigid3Mass();
        printf("\n WARNING : node is not found => massValue could be false in getCompliance function");
    }

    unsigned int numConstraints = constraints.size();
    //std::cout << "constraints.size() = " << constraints.size() << std::endl;
    //std::cout << "constraintId.size() = " << constraintId.size() << std::endl;

    //std::cout << "Liste des index des contraintes: ";
    //for(unsigned int i=0; i<constraintId.size(); i++)
    //{
    //	std::cout << constraintId[i] << ", ";
    //}
    //std::cout << std::endl;

    double dt = this->getContext()->getDt();

    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        int sizeCurRowConst = constraints[curRowConst].size();
        int indexCurRowConst = constraintId[curRowConst];

        for(int i = 0; i < sizeCurRowConst; i++)
        {
            weighedNormal.getVCenter() = constraints[curRowConst][i].data.getVCenter(); // weighed normal
            weighedNormal.getVOrientation() = constraints[curRowConst][i].data.getVOrientation();

            InvM_wN = weighedNormal / (*massValue);
            InvM_wN *= dt*dt;

            int indexCurColConst;

            for(unsigned int curColConst = curRowConst; curColConst < numConstraints; curColConst++)
            {
                int sizeCurColConst = constraints[curColConst].size();
                indexCurColConst = constraintId[curColConst];

                for(int j = 0; j < sizeCurColConst; j++)
                {
                    W[indexCurRowConst][indexCurColConst] +=  constraints[curColConst][j].data * InvM_wN;
//					W[indexCurRowConst][indexCurColConst] *= dt*dt;
                }
            }

            for(unsigned int curColConst = curRowConst+1; curColConst < numConstraints; curColConst++)
            {
                indexCurColConst = constraintId[curColConst];
                W[indexCurColConst][indexCurRowConst] = W[indexCurRowConst][indexCurColConst];
            }

        }
    }
}

template<>
void MechanicalObject<defaulttype::Rigid3Types>::applyContactForce(double *f)
{
    VecDeriv& force = *this->externalForces;
    const VecConst& constraints = *getC();
    Deriv weighedNormal;

    force.resize(0);
    force.resize(1);
    force[0] = Deriv();

    int numConstraints = constraints.size();

    for(int c1 = 0; c1 < numConstraints; c1++)
    {
        int indexC1 = constraintId[c1];

        if (f[indexC1] != 0.0)
        {
            int sizeC1 = constraints[c1].size();
            for(int i = 0; i < sizeC1; i++)
            {
                weighedNormal = constraints[c1][i].data; // weighted normal
                force[0].getVCenter() += weighedNormal.getVCenter() * f[indexC1];
                force[0].getVOrientation() += weighedNormal.getVOrientation() * f[indexC1];
            }
        }
    }
}


template<>
void MechanicalObject<defaulttype::Rigid3Types>::resetContactForce()
{
    VecDeriv& force = *this->externalForces;
    for( unsigned i=0; i<force.size(); ++i )
        force[i] = Deriv();
}

template<>
void MechanicalObject<defaulttype::Vec1dTypes>::getCompliance(double**W)
{
    const VecConst& constraints = *getC();
    unsigned int numConstraints = constraints.size();

    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        int sizeCurRowConst = constraints[curRowConst].size();
        int indexCurRowConst = constraintId[curRowConst];

        for(int i = 0; i < sizeCurRowConst; i++)
        {
            int indexCurColConst;

            for(unsigned int curColConst = curRowConst; curColConst < numConstraints; curColConst++)
            {
                int sizeCurColConst = constraints[curColConst].size();
                indexCurColConst = constraintId[curColConst];

                for(int j = 0; j < sizeCurColConst; j++)
                {
                    if (constraints[curRowConst][i].index == constraints[curColConst][j].index)
                    {
                        W[indexCurRowConst][indexCurColConst] += (1.0/10000.0) * constraints[curRowConst][i].data.x() * constraints[curColConst][j].data.x();
                    }
                }
            }

            for(unsigned int curColConst = curRowConst+1; curColConst < numConstraints; curColConst++)
            {
                indexCurColConst = constraintId[curColConst];
                W[indexCurColConst][indexCurRowConst] = W[indexCurRowConst][indexCurColConst];
            }

        }
    }
}

template<>
void MechanicalObject<defaulttype::Vec1dTypes>::applyContactForce(double *f)
{

    VecDeriv& force = *this->externalForces;
    const VecConst& constraints = *getC();
    unsigned int numConstraints = constraints.size();

    force.resize((*this->x).size());

    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        int indexC1 = constraintId[c1];

        if (f[indexC1] != 0.0)
        {
            int sizeC1 = constraints[c1].size();
            for(int i = 0; i < sizeC1; i++)
            {
                force[constraints[c1][i].index] += constraints[c1][i].data * f[indexC1];
            }
        }
    }
    VecDeriv& dx = *this->dx;

    for (unsigned int i=0; i<dx.size(); i++)
    {
        dx[i] = force[i]/10000.0;
    }
}

template<>
void MechanicalObject<defaulttype::Vec1dTypes>::resetContactForce()
{
    VecDeriv& force = *this->externalForces;
    for( unsigned i=0; i<force.size(); ++i )
        force[i] = Deriv();
}

// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.

template class MechanicalObject<defaulttype::Vec3fTypes>;
template class MechanicalObject<defaulttype::Vec3dTypes>;
template class MechanicalObject<defaulttype::Vec2fTypes>;
template class MechanicalObject<defaulttype::Vec2dTypes>;
template class MechanicalObject<defaulttype::Vec1fTypes>;
template class MechanicalObject<defaulttype::Vec1dTypes>;

template class MechanicalObject<defaulttype::Rigid3dTypes>;
template class MechanicalObject<defaulttype::Rigid3fTypes>;
template class MechanicalObject<defaulttype::Rigid2dTypes>;
template class MechanicalObject<defaulttype::Rigid2fTypes>;

template class MechanicalObject<defaulttype::LaparoscopicRigid3Types>;

} // namespace component

} // namespace sofa
