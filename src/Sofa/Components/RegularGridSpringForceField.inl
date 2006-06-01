#ifndef SOFA_COMPONENTS_REGULARGRIDSPRINGFORCEFIELD_INL
#define SOFA_COMPONENTS_REGULARGRIDSPRINGFORCEFIELD_INL

#include "RegularGridSpringForceField.h"
#include "StiffSpringForceField.inl"
#include "RegularGridTopology.h"
#include "GL/template.h"

namespace Sofa
{

namespace Components
{

template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::addForce()
{
    assert(this->object1);
    assert(this->object2);
    // Calc any custom springs
    this->StiffSpringForceField<DataTypes>::addForce();
    // Compute topological springs
    VecDeriv& f1 = *this->object1->getF();
    VecCoord& p1 = *this->object1->getX();
    VecDeriv& v1 = *this->object1->getV();
    VecDeriv& f2 = *this->object2->getF();
    VecCoord& p2 = *this->object2->getX();
    VecDeriv& v2 = *this->object2->getV();
    f1.resize(p1.size());
    f2.resize(p2.size());
    if (this->object1==this->object2)
    {
        RegularGridTopology* topology = dynamic_cast<RegularGridTopology*>(this->object1->getContext()->getTopology());
        if (topology != NULL)
        {
            const int nx = topology->getNx();
            const int ny = topology->getNy();
            const int nz = topology->getNz();
            int index = this->springs.size();
            int size = index;
            if (this->linesStiffness != 0.0 || this->linesDamping != 0.0)
                size += ((nx-1)*ny*nz+nx*(ny-1)*nz+nx*ny*(nz-1));
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
                size += ((nx-1)*(ny-1)*nz+(nx-1)*ny*(nz-1)+nx*(ny-1)*(nz-1))*2;
            if (this->cubesStiffness != 0.0 || this->cubesDamping != 0.0)
                size += ((nx-1)*(ny-1)*(nz-1))*4;
            this->dfdx.resize(size);
            if (this->linesStiffness != 0.0 || this->linesDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring;
                // lines along X
                spring.initpos = topology->getDx().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x+1,y,z);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring);
                        }
                // lines along Y
                spring.initpos = topology->getDy().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y+1,z);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring);
                        }
                // lines along Z
                spring.initpos = topology->getDz().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y,z+1);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring);
                        }

            }
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                // quads along XY plane
                // lines (x,y,z) -> (x+1,y+1,z)
                spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z)
                spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                spring2.ks = this->linesStiffness / spring2.initpos;
                spring2.kd = this->linesDamping / spring2.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring2);
                        }
                // quads along XZ plane
                // lines (x,y,z) -> (x+1,y,z+1)
                spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y,z+1)
                spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                spring2.ks = this->linesStiffness / spring2.initpos;
                spring2.kd = this->linesDamping / spring2.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y,z+1);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring2);
                        }
                // quads along YZ plane
                // lines (x,y,z) -> (x,y+1,z+1)
                spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x,y+1,z) -> (x,y,z+1)
                spring1.initpos = (topology->getDy()-topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x,y+1,z+1);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring1);
                            spring2.m1 = topology->point(x,y+1,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring2);
                        }
            }
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                typename RegularGridSpringForceField<DataTypes>::Spring spring3;
                typename RegularGridSpringForceField<DataTypes>::Spring spring4;
                // lines (x,y,z) -> (x+1,y+1,z+1)
                spring1.initpos = (topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z+1)
                spring2.initpos = (-topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring2.ks = this->linesStiffness / spring2.initpos;
                spring2.kd = this->linesDamping / spring2.initpos;
                // lines (x,y+1,z) -> (x+1,y,z+1)
                spring3.initpos = (topology->getDx()-topology->getDy()+topology->getDz()).norm();
                spring3.ks = this->linesStiffness / spring3.initpos;
                spring3.kd = this->linesDamping / spring3.initpos;
                // lines (x,y,z+1) -> (x+1,y+1,z)
                spring4.initpos = (topology->getDx()+topology->getDy()-topology->getDz()).norm();
                spring4.ks = this->linesStiffness / spring4.initpos;
                spring4.kd = this->linesDamping / spring4.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z+1);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z+1);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring2);
                            spring3.m1 = topology->point(x,y+1,z);
                            spring3.m2 = topology->point(x+1,y,z+1);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring3);
                            spring4.m1 = topology->point(x,y,z+1);
                            spring4.m2 = topology->point(x+1,y+1,z);
                            this->addSpringForce(f1,p1,v1,f2,p2,v2, index++, spring4);
                        }
            }
        }
    }
}

template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::addDForce()
{
    // Calc any custom springs
    this->StiffSpringForceField<DataTypes>::addDForce();
    // Compute topological springs
    VecDeriv& f1  = *this->object1->getF();
    VecCoord& p1 = *this->object1->getX();
    VecDeriv& dx1 = *this->object1->getDx();
    VecDeriv& f2  = *this->object2->getF();
    VecCoord& p2 = *this->object2->getX();
    VecDeriv& dx2 = *this->object2->getDx();
    f1.resize(dx1.size());
    f2.resize(dx2.size());
    if (this->object1==this->object2)
    {
        RegularGridTopology* topology = dynamic_cast<RegularGridTopology*>(this->object1->getContext()->getTopology());
        if (topology != NULL)
        {
            const int nx = topology->getNx();
            const int ny = topology->getNy();
            const int nz = topology->getNz();
            int index = this->springs.size();
            if (this->linesStiffness != 0.0 || this->linesDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring;
                // lines along X
                spring.initpos = topology->getDx().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x+1,y,z);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                        }
                // lines along Y
                spring.initpos = topology->getDy().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y+1,z);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                        }
                // lines along Z
                spring.initpos = topology->getDz().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y,z+1);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                        }

            }
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                // quads along XY plane
                // lines (x,y,z) -> (x+1,y+1,z)
                spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z)
                spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                spring2.ks = this->linesStiffness / spring2.initpos;
                spring2.kd = this->linesDamping / spring2.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);
                        }
                // quads along XZ plane
                // lines (x,y,z) -> (x+1,y,z+1)
                spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y,z+1)
                spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                spring2.ks = this->linesStiffness / spring2.initpos;
                spring2.kd = this->linesDamping / spring2.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y,z+1);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);
                        }
                // quads along YZ plane
                // lines (x,y,z) -> (x,y+1,z+1)
                spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x,y+1,z) -> (x,y,z+1)
                spring1.initpos = (topology->getDy()-topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x,y+1,z+1);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                            spring2.m1 = topology->point(x,y+1,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);
                        }
            }
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                typename RegularGridSpringForceField<DataTypes>::Spring spring3;
                typename RegularGridSpringForceField<DataTypes>::Spring spring4;
                // lines (x,y,z) -> (x+1,y+1,z+1)
                spring1.initpos = (topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z+1)
                spring2.initpos = (-topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring2.ks = this->linesStiffness / spring2.initpos;
                spring2.kd = this->linesDamping / spring2.initpos;
                // lines (x,y+1,z) -> (x+1,y,z+1)
                spring3.initpos = (topology->getDx()-topology->getDy()+topology->getDz()).norm();
                spring3.ks = this->linesStiffness / spring3.initpos;
                spring3.kd = this->linesDamping / spring3.initpos;
                // lines (x,y,z+1) -> (x+1,y+1,z)
                spring4.initpos = (topology->getDx()+topology->getDy()-topology->getDz()).norm();
                spring4.ks = this->linesStiffness / spring4.initpos;
                spring4.kd = this->linesDamping / spring4.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z+1);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z+1);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);
                            spring3.m1 = topology->point(x,y+1,z);
                            spring3.m2 = topology->point(x+1,y,z+1);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring3);
                            spring4.m1 = topology->point(x,y,z+1);
                            spring4.m2 = topology->point(x+1,y+1,z);
                            this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring4);
                        }
            }
        }
    }
}



template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields()) return;
    assert(this->object1);
    assert(this->object2);
    // Draw any custom springs
    this->StiffSpringForceField<DataTypes>::draw();
    // Compute topological springs
    VecCoord& p1 = *this->object1->getX();
    VecCoord& p2 = *this->object2->getX();
    glDisable(GL_LIGHTING);
    glColor4f(0.5,0.5,0.5,1);
    glBegin(GL_LINES);
    if (this->object1==this->object2)
    {
        RegularGridTopology* topology = dynamic_cast<RegularGridTopology*>(this->object1->getContext()->getTopology());
        if (topology != NULL)
        {
            const int nx = topology->getNx();
            const int ny = topology->getNy();
            const int nz = topology->getNz();

            if (this->linesStiffness != 0.0 || this->linesDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring;
                // lines along X
                spring.initpos = topology->getDx().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x+1,y,z);
                            GL::glVertexT(p1[spring.m1]);
                            GL::glVertexT(p2[spring.m2]);
                        }
                // lines along Y
                spring.initpos = topology->getDy().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y+1,z);
                            GL::glVertexT(p1[spring.m1]);
                            GL::glVertexT(p2[spring.m2]);
                        }
                // lines along Z
                spring.initpos = topology->getDz().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y,z+1);
                            GL::glVertexT(p1[spring.m1]);
                            GL::glVertexT(p2[spring.m2]);
                        }

            }
#if 0
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                // quads along XY plane
                // lines (x,y,z) -> (x+1,y+1,z)
                spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z)
                spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                spring2.ks = this->linesStiffness / spring2.initpos;
                spring2.kd = this->linesDamping / spring2.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z);
                            GL::glVertexT(p1[spring1.m1]);
                            GL::glVertexT(p2[spring1.m2]);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z);
                            GL::glVertexT(p1[spring2.m1]);
                            GL::glVertexT(p2[spring2.m2]);
                        }
                // quads along XZ plane
                // lines (x,y,z) -> (x+1,y,z+1)
                spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y,z+1)
                spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                spring2.ks = this->linesStiffness / spring2.initpos;
                spring2.kd = this->linesDamping / spring2.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y,z+1);
                            GL::glVertexT(p1[spring1.m1]);
                            GL::glVertexT(p2[spring1.m2]);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            GL::glVertexT(p1[spring2.m1]);
                            GL::glVertexT(p2[spring2.m2]);
                        }
                // quads along YZ plane
                // lines (x,y,z) -> (x,y+1,z+1)
                spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x,y+1,z) -> (x,y,z+1)
                spring1.initpos = (topology->getDy()-topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x,y+1,z+1);
                            GL::glVertexT(p1[spring1.m1]);
                            GL::glVertexT(p2[spring1.m2]);
                            spring2.m1 = topology->point(x,y+1,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            GL::glVertexT(p1[spring2.m1]);
                            GL::glVertexT(p2[spring2.m2]);
                        }
            }
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                typename RegularGridSpringForceField<DataTypes>::Spring spring3;
                typename RegularGridSpringForceField<DataTypes>::Spring spring4;
                // lines (x,y,z) -> (x+1,y+1,z+1)
                spring1.initpos = (topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z+1)
                spring2.initpos = (-topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring2.ks = this->linesStiffness / spring2.initpos;
                spring2.kd = this->linesDamping / spring2.initpos;
                // lines (x,y+1,z) -> (x+1,y,z+1)
                spring3.initpos = (topology->getDx()-topology->getDy()+topology->getDz()).norm();
                spring3.ks = this->linesStiffness / spring3.initpos;
                spring3.kd = this->linesDamping / spring3.initpos;
                // lines (x,y,z+1) -> (x+1,y+1,z)
                spring4.initpos = (topology->getDx()+topology->getDy()-topology->getDz()).norm();
                spring4.ks = this->linesStiffness / spring4.initpos;
                spring4.kd = this->linesDamping / spring4.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z+1);
                            GL::glVertexT(p1[spring1.m1]);
                            GL::glVertexT(p2[spring1.m2]);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z+1);
                            GL::glVertexT(p1[spring2.m1]);
                            GL::glVertexT(p2[spring2.m2]);
                            spring3.m1 = topology->point(x,y+1,z);
                            spring3.m2 = topology->point(x+1,y,z+1);
                            GL::glVertexT(p1[spring3.m1]);
                            GL::glVertexT(p2[spring3.m2]);
                            spring4.m1 = topology->point(x,y,z+1);
                            spring4.m2 = topology->point(x+1,y+1,z);
                            GL::glVertexT(p1[spring4.m1]);
                            GL::glVertexT(p2[spring4.m2]);
                        }
            }
#endif
        }
    }
    glEnd();
}

} // namespace Components

} // namespace Sofa

#endif
