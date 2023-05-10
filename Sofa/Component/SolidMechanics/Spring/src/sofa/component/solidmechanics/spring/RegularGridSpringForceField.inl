/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/component/solidmechanics/spring/RegularGridSpringForceField.h>
#include <sofa/component/solidmechanics/spring/StiffSpringForceField.inl>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
RegularGridSpringForceField<DataTypes>::RegularGridSpringForceField()
    : RegularGridSpringForceField(nullptr, nullptr)
{
}

template<class DataTypes>
RegularGridSpringForceField<DataTypes>::RegularGridSpringForceField(core::behavior::MechanicalState<DataTypes>* object1, core::behavior::MechanicalState<DataTypes>* object2)
    : StiffSpringForceField<DataTypes>(object1, object2),
      linesStiffness  (initData(&linesStiffness,Real(100),"linesStiffness","Lines Stiffness"))
      , linesDamping  (initData(&linesDamping  ,Real(5),"linesDamping"  ,"Lines Damping"))
      , quadsStiffness(initData(&quadsStiffness,Real(100),"quadsStiffness","Quads Stiffness"))
      , quadsDamping  (initData(&quadsDamping  ,Real(5),"quadsDamping"  ,"Quads Damping"))
      , cubesStiffness(initData(&cubesStiffness,Real(100),"cubesStiffness","Cubes Stiffness"))
      , cubesDamping  (initData(&cubesDamping  ,Real(5),"cubesDamping"  ,"Cubes Damping"))
      , topology(nullptr)
{
    this->addAlias(&linesStiffness,    "stiffness"); this->addAlias(&linesDamping,    "damping");
    this->addAlias(&quadsStiffness,    "stiffness"); this->addAlias(&quadsDamping,    "damping");
    this->addAlias(&cubesStiffness,    "stiffness"); this->addAlias(&cubesDamping,    "damping");
}

template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::init()
{
    if (this->mstate1 == nullptr)
    {
        this->mstate1 = dynamic_cast<core::behavior::MechanicalState<DataTypes>* >(this->getContext()->getMechanicalState());
        this->mstate2 = this->mstate1;
    }
    if (this->mstate1==this->mstate2)
    {
        topology = dynamic_cast<topology::container::grid::RegularGridTopology*>(this->mstate1->getContext()->getMeshTopology());
    }
    this->StiffSpringForceField<DataTypes>::init();
}

template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 )
//addForce(VecDeriv& vf1, VecDeriv& vf2, const VecCoord& vx1, const VecCoord& vx2, const VecDeriv& vv1, const VecDeriv& vv2)
{
    // Calc any custom springs
    this->StiffSpringForceField<DataTypes>::addForce(mparams, data_f1, data_f2, data_x1, data_x2, data_v1, data_v2);
    // Compute topological springs

    VecDeriv& f1       = *data_f1.beginEdit();
    const VecCoord& x1 =  data_x1.getValue();
    const VecDeriv& v1 =  data_v1.getValue();
    VecDeriv& f2       = *data_f2.beginEdit();
    const VecCoord& x2 =  data_x2.getValue();
    const VecDeriv& v2 =  data_v2.getValue();

    f1.resize(x1.size());
    f2.resize(x2.size());
    this->m_potentialEnergy = 0;
    const type::vector<Spring>& springs = this->springs.getValue();
    if (this->mstate1==this->mstate2)
    {
        if (topology != nullptr)
        {
            const int nx = topology->getNx();
            const int ny = topology->getNy();
            const int nz = topology->getNz();
            int index = springs.size();
            int size = index;
            if (this->linesStiffness.getValue() != 0.0 || this->linesDamping.getValue() != 0.0)
                size += ((nx-1)*ny*nz+nx*(ny-1)*nz+nx*ny*(nz-1));
            if (this->quadsStiffness.getValue() != 0.0 || this->quadsDamping.getValue() != 0.0)
                size += ((nx-1)*(ny-1)*nz+(nx-1)*ny*(nz-1)+nx*(ny-1)*(nz-1))*2;
            if (this->cubesStiffness.getValue() != 0.0 || this->cubesDamping.getValue() != 0.0)
                size += ((nx-1)*(ny-1)*(nz-1))*4;
            this->dfdx.resize(size);
            if (this->linesStiffness.getValue() != 0.0 || this->linesDamping.getValue() != 0.0)
            {
                Spring spring;
                // lines along X
                spring.initpos = (Real)topology->getDx().norm();
                spring.ks = this->linesStiffness.getValue() / spring.initpos;
                spring.kd = this->linesDamping.getValue() / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x+1,y,z);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring);
                        }
                // lines along Y
                spring.initpos = (Real)topology->getDy().norm();
                spring.ks = this->linesStiffness.getValue() / spring.initpos;
                spring.kd = this->linesDamping.getValue() / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y+1,z);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring);
                        }
                // lines along Z
                spring.initpos = (Real)topology->getDz().norm();
                spring.ks = this->linesStiffness.getValue() / spring.initpos;
                spring.kd = this->linesDamping.getValue() / spring.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y,z+1);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring);
                        }

            }
            if (this->quadsStiffness.getValue() != 0.0 || this->quadsDamping.getValue() != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                // quads along XY plane
                // lines (x,y,z) -> (x+1,y+1,z)
                spring1.initpos = (Real)(topology->getDx()+topology->getDy()).norm();
                spring1.ks = this->quadsStiffness.getValue() / spring1.initpos;
                spring1.kd = this->quadsDamping.getValue() / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z)
                spring2.initpos = (Real)(topology->getDx()-topology->getDy()).norm();
                spring2.ks = this->quadsStiffness.getValue() / spring2.initpos;
                spring2.kd = this->quadsDamping.getValue() / spring2.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring2);
                        }
                // quads along XZ plane
                // lines (x,y,z) -> (x+1,y,z+1)
                spring1.initpos = (Real)(topology->getDx()+topology->getDz()).norm();
                spring1.ks = this->quadsStiffness.getValue() / spring1.initpos;
                spring1.kd = this->quadsDamping.getValue() / spring1.initpos;
                // lines (x+1,y,z) -> (x,y,z+1)
                spring2.initpos = (Real)(topology->getDx()-topology->getDz()).norm();
                spring2.ks = this->quadsStiffness.getValue() / spring2.initpos;
                spring2.kd = this->quadsDamping.getValue() / spring2.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y,z+1);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring2);
                        }
                // quads along YZ plane
                // lines (x,y,z) -> (x,y+1,z+1)
                spring1.initpos = (Real)(topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->quadsStiffness.getValue() / spring1.initpos;
                spring1.kd = this->quadsDamping.getValue() / spring1.initpos;
                // lines (x,y+1,z) -> (x,y,z+1)
                spring2.initpos = (Real)(topology->getDy()-topology->getDz()).norm();
                spring2.ks = this->quadsStiffness.getValue() / spring1.initpos;
                spring2.kd = this->quadsDamping.getValue() / spring1.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x,y+1,z+1);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring1);
                            spring2.m1 = topology->point(x,y+1,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring2);
                        }
            }
            if (this->cubesStiffness.getValue() != 0.0 || this->cubesDamping.getValue() != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                typename RegularGridSpringForceField<DataTypes>::Spring spring3;
                typename RegularGridSpringForceField<DataTypes>::Spring spring4;
                // lines (x,y,z) -> (x+1,y+1,z+1)
                spring1.initpos = (Real)(topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->cubesStiffness.getValue() / spring1.initpos;
                spring1.kd = this->cubesDamping.getValue() / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z+1)
                spring2.initpos = (Real)(-topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring2.ks = this->cubesStiffness.getValue() / spring2.initpos;
                spring2.kd = this->cubesDamping.getValue() / spring2.initpos;
                // lines (x,y+1,z) -> (x+1,y,z+1)
                spring3.initpos = (Real)(topology->getDx()-topology->getDy()+topology->getDz()).norm();
                spring3.ks = this->cubesStiffness.getValue() / spring3.initpos;
                spring3.kd = this->cubesDamping.getValue() / spring3.initpos;
                // lines (x,y,z+1) -> (x+1,y+1,z)
                spring4.initpos = (Real)(topology->getDx()+topology->getDy()-topology->getDz()).norm();
                spring4.ks = this->cubesStiffness.getValue() / spring4.initpos;
                spring4.kd = this->cubesDamping.getValue() / spring4.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z+1);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z+1);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring2);
                            spring3.m1 = topology->point(x,y+1,z);
                            spring3.m2 = topology->point(x+1,y,z+1);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring3);
                            spring4.m1 = topology->point(x,y,z+1);
                            spring4.m2 = topology->point(x+1,y+1,z);
                            this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring4);
                        }
            }
        }
    }
    data_f1.endEdit();
    data_f2.endEdit();
}

template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2)
//addDForce(VecDeriv& vdf1, VecDeriv& vdf2, const VecDeriv& vdx1, const VecDeriv& vdx2, double kFactor, double bFactor)
{
    // Calc any custom springs
    this->StiffSpringForceField<DataTypes>::addDForce(mparams, data_df1, data_df2, data_dx1, data_dx2);
    // Compute topological springs

    VecDeriv&        df1 = *data_df1.beginEdit();
    VecDeriv&        df2 = *data_df2.beginEdit();
    const VecDeriv&  dx1 =  data_dx1.getValue();
    const VecDeriv&  dx2 =  data_dx2.getValue();
    Real kFactor       =  (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams,this->rayleighStiffness.getValue());
    Real bFactor       =  (Real)sofa::core::mechanicalparams::bFactor(mparams);

    const type::vector<Spring>& springs = this->springs.getValue();
    if (this->mstate1==this->mstate2)
    {
        if (topology != nullptr)
        {
            const int nx = topology->getNx();
            const int ny = topology->getNy();
            const int nz = topology->getNz();
            int index = springs.size();
            if (this->linesStiffness.getValue() != 0.0 || this->linesDamping.getValue() != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring;
                // lines along X
                spring.initpos = (Real)topology->getDx().norm();
                spring.ks = this->linesStiffness.getValue() / spring.initpos;
                spring.kd = this->linesDamping.getValue() / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x+1,y,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                        }
                // lines along Y
                spring.initpos = (Real)topology->getDy().norm();
                spring.ks = this->linesStiffness.getValue() / spring.initpos;
                spring.kd = this->linesDamping.getValue() / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y+1,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                        }
                // lines along Z
                spring.initpos = (Real)topology->getDz().norm();
                spring.ks = this->linesStiffness.getValue() / spring.initpos;
                spring.kd = this->linesDamping.getValue() / spring.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring, kFactor, bFactor);
                        }

            }
            if (this->quadsStiffness.getValue() != 0.0 || this->quadsDamping.getValue() != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                // quads along XY plane
                // lines (x,y,z) -> (x+1,y+1,z)
                spring1.initpos = (Real)(topology->getDx()+topology->getDy()).norm();
                spring1.ks = this->quadsStiffness.getValue() / spring1.initpos;
                spring1.kd = this->quadsDamping.getValue() / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z)
                spring2.initpos = (Real)(topology->getDx()-topology->getDy()).norm();
                spring2.ks = this->quadsStiffness.getValue() / spring2.initpos;
                spring2.kd = this->quadsDamping.getValue() / spring2.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1, kFactor, bFactor);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2, kFactor, bFactor);
                        }
                // quads along XZ plane
                // lines (x,y,z) -> (x+1,y,z+1)
                spring1.initpos = (Real)(topology->getDx()+topology->getDz()).norm();
                spring1.ks = this->quadsStiffness.getValue() / spring1.initpos;
                spring1.kd = this->quadsDamping.getValue() / spring1.initpos;
                // lines (x+1,y,z) -> (x,y,z+1)
                spring2.initpos = (Real)(topology->getDx()-topology->getDz()).norm();
                spring2.ks = this->quadsStiffness.getValue() / spring2.initpos;
                spring2.kd = this->quadsDamping.getValue() / spring2.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1, kFactor, bFactor);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2, kFactor, bFactor);
                        }
                // quads along YZ plane
                // lines (x,y,z) -> (x,y+1,z+1)
                spring1.initpos = (Real)(topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->quadsStiffness.getValue() / spring1.initpos;
                spring1.kd = this->quadsDamping.getValue() / spring1.initpos;
                // lines (x,y+1,z) -> (x,y,z+1)
                spring1.initpos = (Real)(topology->getDy()-topology->getDz()).norm();
                spring1.ks = this->linesStiffness.getValue() / spring1.initpos;
                spring1.kd = this->linesDamping.getValue() / spring1.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x,y+1,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1, kFactor, bFactor);
                            spring2.m1 = topology->point(x,y+1,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2, kFactor, bFactor);
                        }
            }
            if (this->cubesStiffness.getValue() != 0.0 || this->cubesDamping.getValue() != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                typename RegularGridSpringForceField<DataTypes>::Spring spring3;
                typename RegularGridSpringForceField<DataTypes>::Spring spring4;
                // lines (x,y,z) -> (x+1,y+1,z+1)
                spring1.initpos = (Real)(topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->cubesStiffness.getValue() / spring1.initpos;
                spring1.kd = this->cubesDamping.getValue() / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z+1)
                spring2.initpos = (Real)(-topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring2.ks = this->cubesStiffness.getValue() / spring2.initpos;
                spring2.kd = this->cubesDamping.getValue() / spring2.initpos;
                // lines (x,y+1,z) -> (x+1,y,z+1)
                spring3.initpos = (Real)(topology->getDx()-topology->getDy()+topology->getDz()).norm();
                spring3.ks = this->cubesStiffness.getValue() / spring3.initpos;
                spring3.kd = this->cubesDamping.getValue() / spring3.initpos;
                // lines (x,y,z+1) -> (x+1,y+1,z)
                spring4.initpos = (Real)(topology->getDx()+topology->getDy()-topology->getDz()).norm();
                spring4.ks = this->cubesStiffness.getValue() / spring4.initpos;
                spring4.kd = this->cubesDamping.getValue() / spring4.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1, kFactor, bFactor);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2, kFactor, bFactor);
                            spring3.m1 = topology->point(x,y+1,z);
                            spring3.m2 = topology->point(x+1,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring3, kFactor, bFactor);
                            spring4.m1 = topology->point(x,y,z+1);
                            spring4.m2 = topology->point(x+1,y+1,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring4, kFactor, bFactor);
                        }
            }
        }
    }
    data_df1.endEdit();
    data_df2.endEdit();
}



template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    using namespace sofa::type;
    using namespace sofa::defaulttype;

    if (!((this->mstate1 == this->mstate2)?vparams->displayFlags().getShowForceFields():vparams->displayFlags().getShowInteractionForceFields())) return;
    assert(this->mstate1);
    assert(this->mstate2);

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    // Draw any custom springs
    this->StiffSpringForceField<DataTypes>::draw(vparams);
    // Compute topological springs
    const VecCoord& p1 =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& p2 =this->mstate2->read(core::ConstVecCoordId::position())->getValue();

    std::vector< Vec3 > points;
    Vec3 point1,point2;
    if (this->mstate1==this->mstate2)
    {
        if (topology != nullptr)
        {
            const int nx = topology->getNx();
            const int ny = topology->getNy();
            const int nz = topology->getNz();

            if (this->linesStiffness.getValue() != 0.0 || this->linesDamping.getValue() != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring;
                // lines along X
                spring.initpos = (Real)topology->getDx().norm();
                spring.ks = this->linesStiffness.getValue() / spring.initpos;
                spring.kd = this->linesDamping.getValue() / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x+1,y,z);
                            point1 = DataTypes::getCPos(p1[spring.m1]);
                            point2 = DataTypes::getCPos(p2[spring.m2]);
                            points.push_back(point1);
                            points.push_back(point2);
                        }
                // lines along Y
                spring.initpos = (Real)topology->getDy().norm();
                spring.ks = this->linesStiffness.getValue() / spring.initpos;
                spring.kd = this->linesDamping.getValue() / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y+1,z);
                            point1 = DataTypes::getCPos(p1[spring.m1]);
                            point2 = DataTypes::getCPos(p2[spring.m2]);
                            points.push_back(point1);
                            points.push_back(point2);
                        }
                // lines along Z
                spring.initpos = (Real)topology->getDz().norm();
                spring.ks = this->linesStiffness.getValue() / spring.initpos;
                spring.kd = this->linesDamping.getValue() / spring.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y,z+1);
                            point1 = DataTypes::getCPos(p1[spring.m1]);
                            point2 = DataTypes::getCPos(p2[spring.m2]);
                            points.push_back(point1);
                            points.push_back(point2);
                        }

            }
        }
    }

    vparams->drawTool()->drawLines(points, 1, sofa::type::RGBAColor(0.5,0.5,0.5,1));

}

} // namespace sofa::component::solidmechanics::spring
