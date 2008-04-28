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
#ifndef SOFA_COMPONENT_CONSTRAINT_BOXSTIFFSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_BOXSTIFFSPRINGFORCEFIELD_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/forcefield/BoxStiffSpringForceField.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <map>


namespace sofa
{

namespace component
{

namespace forcefield
{

template <class DataTypes>
BoxStiffSpringForceField<DataTypes>::BoxStiffSpringForceField(MechanicalState* object1, MechanicalState* object2, double ks, double kd)
    : StiffSpringForceField<DataTypes>(object1, object2, ks, kd),
      box_object1( initData( &box_object1, Vec6(0,0,0,1,1,1), "box_object1", "Box for the object1 where springs will be attached") ),
      box_object2( initData( &box_object2, Vec6(0,0,0,1,1,1), "box_object2", "Box for the object2 where springs will be attached") )

{
}

template <class DataTypes>
BoxStiffSpringForceField<DataTypes>::BoxStiffSpringForceField(double ks, double kd)
    : StiffSpringForceField<DataTypes>(ks, kd),
      box_object1( initData( &box_object1, Vec6(0,0,0,1,1,1), "box_object1", "Box for the object1 where springs will be attached") ),
      box_object2( initData( &box_object2, Vec6(0,0,0,1,1,1), "box_object2", "Box for the object2 where springs will be attached") )
{
}

template <class DataTypes>
void BoxStiffSpringForceField<DataTypes>::bwdInit()
{
    Inherit::bwdInit();
    sofa::helper::vector <unsigned int> indices1;
    const Vec6& b1=box_object1.getValue();
    this->mstate1->getIndicesInSpace( indices1, b1[0],b1[3],b1[1],b1[4],b1[2],b1[5] );

    sofa::helper::vector <unsigned int> indices2;
    const Vec6& b2=box_object2.getValue();
    this->mstate2->getIndicesInSpace( indices2, b2[0],b2[3],b2[1],b2[4],b2[2],b2[5] );


    const VecCoord& x1 = *this->mstate1->getX();
    const VecCoord& x2 = *this->mstate2->getX();

    //Attach springs using with priority the shortest distance between points
    float min_dist=0.0f;
    if (indices1.size() < indices2.size())
    {
        sofa::helper::vector< std::map<Real, unsigned int> > distance_spring(indices1.size());
        for(unsigned int i = 0; i < indices1.size(); ++i)
        {
            for(unsigned int j = 0; j < indices2.size(); ++j)
            {
                distance_spring[i][(Real)sqrt((x1[indices1[i]] - x2[indices2[j]]).norm2())] = j;
            }
            if (i==0 || min_dist> distance_spring[i].begin()->first) min_dist = distance_spring[i].begin()->first;

        }
        sofa::helper::vector< bool > indice_unused(indices2.size(),true);

        for(unsigned int i = 0; i<indices1.size(); ++i)
        {
            std::map<Real, unsigned int>::const_iterator it = distance_spring[i].begin();
            for (; it!=distance_spring[i].end(); it++)
            {
                if (indice_unused[it->second])
                {
                    indice_unused[it->second] = false;
                    this->addSpring(indices1[i], indices2[it->second], this->getStiffness()*it->first/min_dist, this->getDamping(), it->first );
                    break;
                }
            }
        }
    }
    else
    {
        sofa::helper::vector< std::map<Real, unsigned int> > distance_spring(indices2.size());
        for(unsigned int i = 0; i < indices2.size(); ++i)
        {
            for(unsigned int j = 0; j < indices1.size(); ++j)
            {
                distance_spring[i][(Real)sqrt((x1[indices1[j]] - x2[indices2[i]]).norm2())] = j;
            }

            if (i==0 || min_dist> distance_spring[i].begin()->first) min_dist = distance_spring[i].begin()->first;
        }
        sofa::helper::vector< bool > indice_unused(indices1.size(),true);

        for(unsigned int i = 0; i<indices2.size(); ++i)
        {
            std::map<Real, unsigned int>::const_iterator it = distance_spring[i].begin();
            for (; it!=distance_spring[i].end(); it++)
            {
                if (indice_unused[it->second])
                {
                    indice_unused[it->second] = false;
                    this->addSpring( indices1[it->second], indices2[i], this->getStiffness()*it->first/min_dist, this->getDamping(), it->first );
                    break;
                }
            }
        }
    }
}



template <class DataTypes>
void BoxStiffSpringForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowInteractionForceFields())
        return;

    Inherit::draw();
    //     const VecCoord& x = *this->mstate->getX();
    //     glDisable (GL_LIGHTING);
    //     glPointSize(10);
    //     glColor4f (1,0.5,0.5,1);
    //     glBegin (GL_POINTS);
    //     const SetIndex& indices = this->f_indices.getValue();
    //     for (typename SetIndex::const_iterator it = indices.begin();
    //         it != indices.end();
    //         ++it)
    //     {
    //         gl::glVertexT(x[*it]);
    //     }
    //     glEnd();

    ///draw the constraint box
    const Vec6& b1=box_object1.getValue();
    const Real& Xmin1=b1[0];
    const Real& Xmax1=b1[3];
    const Real& Ymin1=b1[1];
    const Real& Ymax1=b1[4];
    const Real& Zmin1=b1[2];
    const Real& Zmax1=b1[5];

    const Vec6& b2=box_object2.getValue();
    const Real& Xmin2=b2[0];
    const Real& Xmax2=b2[3];
    const Real& Ymin2=b2[1];
    const Real& Ymax2=b2[4];
    const Real& Zmin2=b2[2];
    const Real& Zmax2=b2[5];


    glBegin(GL_LINES);


    glColor4f (0,0.5,0.5,1);
    glVertex3d(Xmin1,Ymin1,Zmin1);
    glVertex3d(Xmin1,Ymin1,Zmax1);
    glVertex3d(Xmin1,Ymin1,Zmin1);
    glVertex3d(Xmax1,Ymin1,Zmin1);
    glVertex3d(Xmin1,Ymin1,Zmin1);
    glVertex3d(Xmin1,Ymax1,Zmin1);
    glVertex3d(Xmin1,Ymax1,Zmin1);
    glVertex3d(Xmax1,Ymax1,Zmin1);
    glVertex3d(Xmin1,Ymax1,Zmin1);
    glVertex3d(Xmin1,Ymax1,Zmax1);
    glVertex3d(Xmin1,Ymax1,Zmax1);
    glVertex3d(Xmin1,Ymin1,Zmax1);
    glVertex3d(Xmin1,Ymin1,Zmax1);
    glVertex3d(Xmax1,Ymin1,Zmax1);
    glVertex3d(Xmax1,Ymin1,Zmax1);
    glVertex3d(Xmax1,Ymax1,Zmax1);
    glVertex3d(Xmax1,Ymin1,Zmax1);
    glVertex3d(Xmax1,Ymin1,Zmin1);
    glVertex3d(Xmin1,Ymax1,Zmax1);
    glVertex3d(Xmax1,Ymax1,Zmax1);
    glVertex3d(Xmax1,Ymax1,Zmin1);
    glVertex3d(Xmax1,Ymin1,Zmin1);
    glVertex3d(Xmax1,Ymax1,Zmin1);
    glVertex3d(Xmax1,Ymax1,Zmax1);

    glColor4f (0.5,0.5,0,1);

    glVertex3d(Xmin2,Ymin2,Zmin2);
    glVertex3d(Xmin2,Ymin2,Zmax2);
    glVertex3d(Xmin2,Ymin2,Zmin2);
    glVertex3d(Xmax2,Ymin2,Zmin2);
    glVertex3d(Xmin2,Ymin2,Zmin2);
    glVertex3d(Xmin2,Ymax2,Zmin2);
    glVertex3d(Xmin2,Ymax2,Zmin2);
    glVertex3d(Xmax2,Ymax2,Zmin2);
    glVertex3d(Xmin2,Ymax2,Zmin2);
    glVertex3d(Xmin2,Ymax2,Zmax2);
    glVertex3d(Xmin2,Ymax2,Zmax2);
    glVertex3d(Xmin2,Ymin2,Zmax2);
    glVertex3d(Xmin2,Ymin2,Zmax2);
    glVertex3d(Xmax2,Ymin2,Zmax2);
    glVertex3d(Xmax2,Ymin2,Zmax2);
    glVertex3d(Xmax2,Ymax2,Zmax2);
    glVertex3d(Xmax2,Ymin2,Zmax2);
    glVertex3d(Xmax2,Ymin2,Zmin2);
    glVertex3d(Xmin2,Ymax2,Zmax2);
    glVertex3d(Xmax2,Ymax2,Zmax2);
    glVertex3d(Xmax2,Ymax2,Zmin2);
    glVertex3d(Xmax2,Ymin2,Zmin2);
    glVertex3d(Xmax2,Ymax2,Zmin2);
    glVertex3d(Xmax2,Ymax2,Zmax2);

    glEnd();
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
