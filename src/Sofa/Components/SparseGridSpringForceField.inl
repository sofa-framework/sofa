#ifndef SOFA_COMPONENTS_REGULARGRIDSPRINGFORCEFIELD_INL
#define SOFA_COMPONENTS_REGULARGRIDSPRINGFORCEFIELD_INL

#include "SparseGridSpringForceField.h"
#include "StiffSpringForceField.inl"
#include "GL/template.h"

namespace Sofa
{

namespace Components
{


using std::cout;
using std::cerr;
using std::endl;

template <class DataTypes>
void SparseGridSpringForceField<DataTypes>::addForce()
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
    this->m_potentialEnergy = 0;


    if (this->object1==this->object2)
    {
        MultiResSparseGridTopology* topology = dynamic_cast<MultiResSparseGridTopology*>(this->object1->getContext()->getTopology());
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

                if (this->linesStiffness != 0.0 || this->linesDamping != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring;

                    /// add x axis springs
                    spring.initpos = topology->getDx().norm();
                    spring.ks = this->linesStiffness / spring.initpos;
                    spring.kd = this->linesDamping / spring.initpos;

                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i+1,j,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i+1,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j+1,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);

                    /// add y axis springs
                    spring.initpos = topology->getDy().norm();
                    spring.ks = this->linesStiffness / spring.initpos;
                    spring.kd = this->linesDamping / spring.initpos;

                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i+1,j,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);

                    /// add z axis springs
                    spring.initpos = topology->getDz().norm();
                    spring.ks = this->linesStiffness / spring.initpos;
                    spring.kd = this->linesDamping / spring.initpos;

                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                    spring.m1 = topology->point(i+1,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring);
                }

                if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring1;
                    typename SparseGridSpringForceField<DataTypes>::Spring spring2;

                    /// add xy plane springs
                    // lines (x,y,z) -> (x+1,y+1,z)
                    spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                    spring1.ks = this->linesStiffness / spring1.initpos;
                    spring1.kd = this->linesDamping / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y+1,z)
                    spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                    spring2.ks = this->linesStiffness / spring2.initpos;
                    spring2.kd = this->linesDamping / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k);
                    spring2.m2 = topology->point(i+1,j,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring2);
                    spring1.m1 = topology->point(i,j,k+1);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring2);

                    /// add xz plane springs
                    // lines (x,y,z) -> (x+1,y,z+1)
                    spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness / spring1.initpos;
                    spring1.kd = this->linesDamping / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness / spring2.initpos;
                    spring2.kd = this->linesDamping / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i+1,j,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring2);
                    spring1.m1 = topology->point(i,j+1,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring2);

                    /// add yz plane springs
                    // lines (x,y,z) -> (x,y+1,z+1)
                    spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness / spring1.initpos;
                    spring1.kd = this->linesDamping / spring1.initpos;
                    // lines (x,y+1,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDy()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness / spring2.initpos;
                    spring2.kd = this->linesDamping / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring2);
                    spring1.m1 = topology->point(i+1,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring1);
                    spring2.m1 = topology->point(i+1,j,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    this->addSpringForce(this->m_potentialEnergy,f1,p1,v1,f2,p2,v2, index++, spring2);
                }
            }

        }
    }
}

template<class DataTypes>
void SparseGridSpringForceField<DataTypes>::addDForce()
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
        MultiResSparseGridTopology* topology = dynamic_cast<MultiResSparseGridTopology*>(this->object1->getContext()->getTopology());
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


                if (this->linesStiffness != 0.0 || this->linesDamping != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring;

                    /// add axis spring x
                    spring.initpos = topology->getDx().norm();
                    spring.ks = this->linesStiffness / spring.initpos;
                    spring.kd = this->linesDamping / spring.initpos;
                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i+1,j,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i+1,j,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                    spring.m1 = topology->point(i,j+1,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);

                    /// add axis spring y
                    spring.initpos = topology->getDy().norm();
                    spring.ks = this->linesStiffness / spring.initpos;
                    spring.kd = this->linesDamping / spring.initpos;
                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j+1,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i,j+1,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                    spring.m1 = topology->point(i+1,j,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);

                    ///add axis spring z
                    spring.initpos = topology->getDz().norm();
                    spring.ks = this->linesStiffness / spring.initpos;
                    spring.kd = this->linesDamping / spring.initpos;

                    /// add the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i,j+1,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                    spring.m1 = topology->point(i+1,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring);
                }

                if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring1;
                    typename SparseGridSpringForceField<DataTypes>::Spring spring2;

                    /// add plane springs  xy
                    // lines (x,y,z) -> (x+1,y+1,z)
                    spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                    spring1.ks = this->linesStiffness / spring1.initpos;
                    spring1.kd = this->linesDamping / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y+1,z)
                    spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                    spring2.ks = this->linesStiffness / spring2.initpos;
                    spring2.kd = this->linesDamping / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k);
                    spring2.m2 = topology->point(i+1,j,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);
                    spring1.m1 = topology->point(i,j,k+1);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);

                    /// add plane springs  xz
                    // lines (x,y,z) -> (x+1,y,z+1)
                    spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness / spring1.initpos;
                    spring1.kd = this->linesDamping / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness / spring2.initpos;
                    spring2.kd = this->linesDamping / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i+1,j,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);
                    spring1.m1 = topology->point(i,j+1,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);

                    /// add plane springs  yz
                    // lines (x,y,z) -> (x,y+1,z+1)
                    spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness / spring1.initpos;
                    spring1.kd = this->linesDamping / spring1.initpos;
                    // lines (x,y+1,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDy()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness / spring2.initpos;
                    spring2.kd = this->linesDamping / spring2.initpos;

                    /// add the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i,j+1,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i,j+1,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);
                    spring1.m1 = topology->point(i+1,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring1);
                    spring2.m1 = topology->point(i+1,j,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, index++, spring2);
                }
            }
        }
    }
}



template<class DataTypes>
void SparseGridSpringForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields())
        return;
    assert(this->object1);
    assert(this->object2);
    // Draw any custom springs
    this->StiffSpringForceField<DataTypes>::draw();
    // Compute topological springs
    VecCoord& p1 = *this->object1->getX();
    VecCoord& p2 = *this->object2->getX();

    glDisable(GL_LIGHTING);
    glColor4f(0.0,0.5,0.0,1);
    glBegin(GL_LINES);
    if (this->object1==this->object2)
    {
        MultiResSparseGridTopology* topology = dynamic_cast<MultiResSparseGridTopology*>(this->object1->getContext()->getTopology());
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

                if (this->linesStiffness != 0.0 || this->linesDamping != 0.0)
                {
                    typename SparseGridSpringForceField<DataTypes>::Spring spring;

                    /// draw axis spring x
                    spring.initpos = topology->getDx().norm();
                    spring.ks = this->linesStiffness / spring.initpos;
                    spring.kd = this->linesDamping / spring.initpos;
                    /// draw the 4th springs

                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i+1,j,k);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i+1,j,k+1);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j+1,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);

                    ///draw axis spring y
                    spring.initpos = topology->getDy().norm();
                    spring.ks = this->linesStiffness / spring.initpos;
                    spring.kd = this->linesDamping / spring.initpos;
                    /// draw the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j+1,k);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j+1,k);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j,k+1);
                    spring.m2 = topology->point(i,j+1,k+1);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i+1,j,k+1);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);

                    ///draw axis spring z
                    spring.initpos = topology->getDz().norm();
                    spring.ks = this->linesStiffness / spring.initpos;
                    spring.kd = this->linesDamping / spring.initpos;

                    /// draw the 4th springs
                    spring.m1 = topology->point(i,j,k);
                    spring.m2 = topology->point(i,j,k+1);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i+1,j,k);
                    spring.m2 = topology->point(i+1,j,k+1);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i,j+1,k);
                    spring.m2 = topology->point(i,j+1,k+1);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);
                    spring.m1 = topology->point(i+1,j+1,k);
                    spring.m2 = topology->point(i+1,j+1,k+1);
                    GL::glVertexT(p1[spring.m1]);
                    GL::glVertexT(p2[spring.m2]);



                    typename SparseGridSpringForceField<DataTypes>::Spring spring1;
                    typename SparseGridSpringForceField<DataTypes>::Spring spring2;
                    /// draw plane springs  xy
                    // lines (x,y,z) -> (x+1,y+1,z)
                    spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                    spring1.ks = this->linesStiffness / spring1.initpos;
                    spring1.kd = this->linesDamping / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y+1,z)
                    spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                    spring2.ks = this->linesStiffness / spring2.initpos;
                    spring2.kd = this->linesDamping / spring2.initpos;

                    /// draw the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k);
                    GL::glVertexT(p1[spring1.m1]);
                    GL::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j+1,k);
                    spring2.m2 = topology->point(i+1,j,k);
                    GL::glVertexT(p1[spring2.m1]);
                    GL::glVertexT(p2[spring2.m2]);
                    spring1.m1 = topology->point(i,j,k+1);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    GL::glVertexT(p1[spring1.m1]);
                    GL::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j,k+1);
                    GL::glVertexT(p1[spring2.m1]);
                    GL::glVertexT(p2[spring2.m2]);

                    /// draw plane springs  xz
                    // lines (x,y,z) -> (x+1,y,z+1)
                    spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness / spring1.initpos;
                    spring1.kd = this->linesDamping / spring1.initpos;
                    // lines (x+1,y,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness / spring2.initpos;
                    spring2.kd = this->linesDamping / spring2.initpos;

                    /// draw the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i+1,j,k+1);
                    GL::glVertexT(p1[spring1.m1]);
                    GL::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i+1,j,k);
                    GL::glVertexT(p1[spring2.m1]);
                    GL::glVertexT(p2[spring2.m2]);
                    spring1.m1 = topology->point(i,j+1,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    GL::glVertexT(p1[spring1.m1]);
                    GL::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j+1,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    GL::glVertexT(p1[spring2.m1]);
                    GL::glVertexT(p2[spring2.m2]);

                    /// draw plane springs  yz
                    // lines (x,y,z) -> (x,y+1,z+1)
                    spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                    spring1.ks = this->linesStiffness / spring1.initpos;
                    spring1.kd = this->linesDamping / spring1.initpos;
                    // lines (x,y+1,z) -> (x,y,z+1)
                    spring2.initpos = (topology->getDy()-topology->getDz()).norm();
                    spring2.ks = this->linesStiffness / spring2.initpos;
                    spring2.kd = this->linesDamping / spring2.initpos;

                    /// draw the 4th springs
                    spring1.m1 = topology->point(i,j,k);
                    spring1.m2 = topology->point(i,j+1,k+1);
                    GL::glVertexT(p1[spring1.m1]);
                    GL::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i,j,k+1);
                    spring2.m2 = topology->point(i,j+1,k);
                    GL::glVertexT(p1[spring2.m1]);
                    GL::glVertexT(p2[spring2.m2]);
                    spring1.m1 = topology->point(i+1,j,k);
                    spring1.m2 = topology->point(i+1,j+1,k+1);
                    GL::glVertexT(p1[spring1.m1]);
                    GL::glVertexT(p2[spring1.m2]);
                    spring2.m1 = topology->point(i+1,j,k+1);
                    spring2.m2 = topology->point(i+1,j+1,k);
                    GL::glVertexT(p1[spring2.m1]);
                    GL::glVertexT(p2[spring2.m2]);
                }
            }

        }
    }
    glEnd();
}

} // namespace Components

} // namespace Sofa

#endif
