/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPARSEGRIDSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPARSEGRIDSPRINGFORCEFIELD_INL

#include <sofa/component/interactionforcefield/SparseGridSpringForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaDeformable/StiffSpringForceField.inl>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace forcefield
{


using std::cout;
using std::cerr;
using std::endl;

template <class DataTypes>
void SparseGridSpringForceField<DataTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2)
{
    // Calc any custom springs
    this->StiffSpringForceField<DataTypes>::addForce(f1, f2, x1, x2, v1, v2);
    // Compute topological springs
    f1.resize(x1.size());
    f2.resize(p2.size());
    this->m_potentialEnergy = 0;


    if (this->object1==this->object2)
    {
        topology::MultiResSparseGridTopology* topology = dynamic_cast<topology::MultiResSparseGridTopology*>(this->object1->getContext()->getTopology());
        if (topology != NULL)
        {
            int index = this->springs.size();
            int size = index;
            size += topology->getNbVoxels()*24;
            this->dfdx.resize(size);

            int i,j,k;

            Voxels *voxels;

            voxels = &(topology->vectorSparseGrid[topology->resolution]);

            std::map<Voxels::Index3D,Voxels::Voxel>::iterator iter;

            for(iter = voxels->getVoxelsMapBegin(); iter != voxels->getVoxelsMapEnd(); iter++)
            {

                ///get the voxel's indices
                i = (*iter).first.i;
                j = (*iter).first.j;
                k = (*iter).first.k;

                if (this->linesStiffness.getValue() != 0.0 || this->linesDamping.getValue() != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring;

                    /// add x axis springs
                    spring.initpos = topology->getDx().norm();
                    spring.ks = this->linesStiffness.getValue() / spring.initpos;
                    spring.kd = this->linesDamping.getValue() / spring.initpos;

                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i+1,j,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i+1,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j+1,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);

                    /// add y axis springs
                    spring.initpos = topology->getDy().norm();
                    spring.ks = this->linesStiffness.getValue() / spring.initpos;
                    spring.kd = this->linesDamping.getValue() / spring.initpos;

                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i+1,j,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);

                    /// add z axis springs
                    spring.initpos = topology->getDz().norm();
                    spring.ks = this->linesStiffness.getValue() / spring.initpos;
                    spring.kd = this->linesDamping.getValue() / spring.initpos;

                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i+1,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring);
                }

                if (this->quadsStiffness.getValue() != 0.0 || this->quadsDamping.getValue() != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring1;
                    typename SparseGridSpringForceField<DataTypes>::Spring spring2;

                    /// add xy plane springs
                    // lines (x,y,z) -> (x+1,y+1,z)
                    spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                    spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                    spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y+1,z)
                    spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                    spring2.ks = this->linesStiffness.getValue() / spring2.initpos;
                    spring2.kd = this->linesDamping.getValue() / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k);
                    spring2.m2 = topology->point(i+1,j,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring2);
                    spring1.m1 = topology->point(i,j,k+1);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring2);

                    /// add xz plane springs
                    // lines (x,y,z) -> (x+1,y,z+1)
                    spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                    spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness.getValue() / spring2.initpos;
                    spring2.kd = this->linesDamping.getValue() / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i+1,j,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring2);
                    spring1.m1 = topology->point(i,j+1,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring2);

                    /// add yz plane springs
                    // lines (x,y,z) -> (x,y+1,z+1)
                    spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                    spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                    // lines (x,y+1,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDy()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness.getValue() / spring2.initpos;
                    spring2.kd = this->linesDamping.getValue() / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring2);
                    spring1.m1 = topology->point(i+1,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i+1,j,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,p2,v2, index++, spring2);
                }
            }

        }
    }
}

template<class DataTypes>
void SparseGridSpringForceField<DataTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double bFactor)
{
    // Calc any custom springs
    this->StiffSpringForceField<DataTypes>::addDForce(df1,df2,dx1,dx2, kFactor, bFactor);
    // Compute topological springs
    df1.resize(dx1.size());
    df2.resize(dx2.size());
    if (this->object1==this->object2)
    {
        topology::MultiResSparseGridTopology* topology = dynamic_cast<topology::MultiResSparseGridTopology*>(this->object1->getContext()->getTopology());
        if (topology != NULL)
        {
            int index = this->springs.size();
            int size = index;
            size += topology->getNbVoxels()*24;
            this->dfdx.resize(size);

            //variable permettant de calculer
            int i,j,k;

            Voxels *voxels;

            voxels = &(topology->vectorSparseGrid[topology->resolution]);

            std::map<Voxels::Index3D,Voxels::Voxel>::iterator iter;

            for(iter = voxels->getVoxelsMapBegin(); iter != voxels->getVoxelsMapEnd(); iter++)
            {

                i = (*iter).first.i;
                j = (*iter).first.j;
                k = (*iter).first.k;


                if (this->linesStiffness.getValue() != 0.0 || this->linesDamping.getValue() != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring;

                    /// add axis spring x
                    spring.initpos = topology->getDx().norm();
                    spring.ks = this->linesStiffness.getValue() / spring.initpos;
                    spring.kd = this->linesDamping.getValue() / spring.initpos;
                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i+1,j,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i+1,j,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                    spring.m1 = topology->point(i,j+1,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);

                    /// add axis spring y
                    spring.initpos = topology->getDy().norm();
                    spring.ks = this->linesStiffness.getValue() / spring.initpos;
                    spring.kd = this->linesDamping.getValue() / spring.initpos;
                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j+1,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i,j+1,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                    spring.m1 = topology->point(i+1,j,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);

                    ///add axis spring z
                    spring.initpos = topology->getDz().norm();
                    spring.ks = this->linesStiffness.getValue() / spring.initpos;
                    spring.kd = this->linesDamping.getValue() / spring.initpos;

                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i,j+1,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                    spring.m1 = topology->point(i+1,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                }

                if (this->quadsStiffness.getValue() != 0.0 || this->quadsDamping.getValue() != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring1;
                    typename SparseGridSpringForceField<DataTypes>::Spring spring2;

                    /// add plane springs  xy
                    // lines (x,y,z) -> (x+1,y+1,z)
                    spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                    spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                    spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y+1,z)
                    spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                    spring2.ks = this->linesStiffness.getValue() / spring2.initpos;
                    spring2.kd = this->linesDamping.getValue() / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k);
                    spring2.m2 = topology->point(i+1,j,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);
                    spring1.m1 = topology->point(i,j,k+1);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);

                    /// add plane springs  xz
                    // lines (x,y,z) -> (x+1,y,z+1)
                    spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                    spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness.getValue() / spring2.initpos;
                    spring2.kd = this->linesDamping.getValue() / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i+1,j,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);
                    spring1.m1 = topology->point(i,j+1,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);

                    /// add plane springs  yz
                    // lines (x,y,z) -> (x,y+1,z+1)
                    spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                    spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                    // lines (x,y+1,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDy()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness.getValue() / spring2.initpos;
                    spring2.kd = this->linesDamping.getValue() / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i,j+1,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i,j+1,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);
                    spring1.m1 = topology->point(i+1,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i+1,j,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);
                }
            }
        }
    }
}



template<class DataTypes>
void SparseGridSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields())
        return;
    assert(this->object1);
    assert(this->object2);
    // Draw any custom springs
    this->StiffSpringForceField<DataTypes>::draw(vparams);
    // Compute topological springs
    const VecCoord& p1 = *this->object1->getX();
    const VecCoord& p2 = *this->object2->getX();

    glDisable(GL_LIGHTING);
    glColor4f(0.0,0.5,0.0,1);
    glBegin(GL_LINES);
    if (this->object1==this->object2)
    {
        topology::MultiResSparseGridTopology* topology = dynamic_cast<topology::MultiResSparseGridTopology*>(this->object1->getContext()->getTopology());
        if (topology != NULL)
        {

            //const int nz = topology->getNz();

            //variable permettant de calculer
            int i,j,k;

            Voxels *voxels;

            voxels = &(topology->vectorSparseGrid[topology->resolution]);

            std::map<Voxels::Index3D,Voxels::Voxel>::iterator iter;

            for(iter = voxels->getVoxelsMapBegin(); iter != voxels->getVoxelsMapEnd(); iter++)
            {
                i = (*iter).first.i;
                j = (*iter).first.j;
                k = (*iter).first.k;

                if (this->linesStiffness.getValue() != 0.0 || this->linesDamping.getValue() != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring;

                    /// draw axis spring x
                    spring.initpos = topology->getDx().norm();
                    spring.ks = this->linesStiffness.getValue() / spring.initpos;
                    spring.kd = this->linesDamping.getValue() / spring.initpos;
                    /// draw the 4th springs

                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i+1,j,k);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i+1,j,k+1);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j+1,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);

                    ///draw axis spring y
                    spring.initpos = topology->getDy().norm();
                    spring.ks = this->linesStiffness.getValue() / spring.initpos;
                    spring.kd = this->linesDamping.getValue() / spring.initpos;
                    /// draw the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j+1,k);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i,j+1,k+1);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i+1,j,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);

                    ///draw axis spring z
                    spring.initpos = topology->getDz().norm();
                    spring.ks = this->linesStiffness.getValue() / spring.initpos;
                    spring.kd = this->linesDamping.getValue() / spring.initpos;

                    /// draw the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j,k+1);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j,k+1);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i,j+1,k+1);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i+1,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    helper::gl::glVertexT(p1[spring.m1]);
                    helper::gl::glVertexT(p2[spring.m2]);



                    typename SparseGridSpringForceField<DataTypes>::Spring spring1;
                    typename SparseGridSpringForceField<DataTypes>::Spring spring2;
                    /// draw plane springs  xy
                    // lines (x,y,z) -> (x+1,y+1,z)
                    spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                    spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                    spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y+1,z)
                    spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                    spring2.ks = this->linesStiffness.getValue() / spring2.initpos;
                    spring2.kd = this->linesDamping.getValue() / spring2.initpos;

                    /// draw the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k);
                    helper::gl::glVertexT(p1[spring1.m1]);
                    helper::gl::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j+1,k);
                    spring2.m2 = topology->point(i+1,j,k);
                    helper::gl::glVertexT(p1[spring2.m1]);
                    helper::gl::glVertexT(p2[spring2.m2]);
                    spring1.m1 = topology->point(i,j,k+1);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    helper::gl::glVertexT(p1[spring1.m1]);
                    helper::gl::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j,k+1);
                    helper::gl::glVertexT(p1[spring2.m1]);
                    helper::gl::glVertexT(p2[spring2.m2]);

                    /// draw plane springs  xz
                    // lines (x,y,z) -> (x+1,y,z+1)
                    spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                    spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness.getValue() / spring2.initpos;
                    spring2.kd = this->linesDamping.getValue() / spring2.initpos;

                    /// draw the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j,k+1);
                    helper::gl::glVertexT(p1[spring1.m1]);
                    helper::gl::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i+1,j,k);
                    helper::gl::glVertexT(p1[spring2.m1]);
                    helper::gl::glVertexT(p2[spring2.m2]);
                    spring1.m1 = topology->point(i,j+1,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    helper::gl::glVertexT(p1[spring1.m1]);
                    helper::gl::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    helper::gl::glVertexT(p1[spring2.m1]);
                    helper::gl::glVertexT(p2[spring2.m2]);

                    /// draw plane springs  yz
                    // lines (x,y,z) -> (x,y+1,z+1)
                    spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                    spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                    // lines (x,y+1,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDy()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness.getValue() / spring2.initpos;
                    spring2.kd = this->linesDamping.getValue() / spring2.initpos;

                    /// draw the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i,j+1,k+1);
                    helper::gl::glVertexT(p1[spring1.m1]);
                    helper::gl::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i,j+1,k);
                    helper::gl::glVertexT(p1[spring2.m1]);
                    helper::gl::glVertexT(p2[spring2.m2]);
                    spring1.m1 = topology->point(i+1,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    helper::gl::glVertexT(p1[spring1.m1]);
                    helper::gl::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i+1,j,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    helper::gl::glVertexT(p1[spring2.m1]);
                    helper::gl::glVertexT(p2[spring2.m2]);
                }
            }

        }
    }
    glEnd();
}

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPARSEGRIDSPRINGFORCEFIELD_INL */
