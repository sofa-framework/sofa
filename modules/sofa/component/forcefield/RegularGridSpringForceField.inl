#ifndef SOFA_COMPONENT_FORCEFIELD_REGULARGRIDSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_REGULARGRIDSPRINGFORCEFIELD_INL

#include <sofa/component/forcefield/RegularGridSpringForceField.h>
#include <sofa/component/forcefield/StiffSpringForceField.inl>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->Inherit::parse(arg);
    if (arg->getAttribute("stiffness")) this->setStiffness((Real)atof(arg->getAttribute("stiffness")));
    if (arg->getAttribute("linesStiffness")) this->setLinesStiffness((Real)atof(arg->getAttribute("linesStiffness")));
    if (arg->getAttribute("quadsStiffness")) this->setQuadsStiffness((Real)atof(arg->getAttribute("quadsStiffness")));
    if (arg->getAttribute("cubesStiffness")) this->setCubesStiffness((Real)atof(arg->getAttribute("cubesStiffness")));
    if (arg->getAttribute("damping")) this->setDamping((Real)atof(arg->getAttribute("damping")));
    if (arg->getAttribute("linesDamping")) this->setLinesDamping((Real)atof(arg->getAttribute("linesDamping")));
    if (arg->getAttribute("quadsDamping")) this->setQuadsDamping((Real)atof(arg->getAttribute("quadsDamping")));
    if (arg->getAttribute("cubesDamping")) this->setCubesDamping((Real)atof(arg->getAttribute("cubesDamping")));
}

template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::init()
{
    //this->StiffSpringForceField<DataTypes>::init();
    if (this->mstate1 == NULL)
    {
        this->mstate1 = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>* >(this->getContext()->getMechanicalState());
        this->mstate2 = this->mstate1;
    }
    if (this->mstate1==this->mstate2)
    {
        topology = dynamic_cast<topology::RegularGridTopology*>(this->mstate1->getContext()->getTopology());
        if (topology != NULL)
        {
            trimmedTopology = dynamic_cast<topology::FittedRegularGridTopology*>(topology);
        }
    }
    this->StiffSpringForceField<DataTypes>::init();
}

template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2)
{
    // Calc any custom springs
    this->StiffSpringForceField<DataTypes>::addForce(f1, f2, x1, x2, v1, v2);
    // Compute topological springs
    f1.resize(x1.size());
    f2.resize(x2.size());
    m_potentialEnergy = 0;
    const vector<Spring>& springs = this->springs.getValue();
    if (this->mstate1==this->mstate2)
    {
        if (topology != NULL)
        {
            const int nx = topology->getNx();
            const int ny = topology->getNy();
            const int nz = topology->getNz();
            int index = springs.size();
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
                Spring spring;
                // lines along X
                spring.initpos = topology->getDx().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x,y  ,z  ) &&
                                !trimmedTopology->isCubeActive(x,y-1,z  ) &&
                                !trimmedTopology->isCubeActive(x,y  ,z-1) &&
                                !trimmedTopology->isCubeActive(x,y-1,z-1))
                                continue;
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x+1,y,z);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring);
                        }
                // lines along Y
                spring.initpos = topology->getDy().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x  ,y,z  ) &&
                                !trimmedTopology->isCubeActive(x-1,y,z  ) &&
                                !trimmedTopology->isCubeActive(x  ,y,z-1) &&
                                !trimmedTopology->isCubeActive(x-1,y,z-1))
                                continue;
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y+1,z);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring);
                        }
                // lines along Z
                spring.initpos = topology->getDz().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x  ,y  ,z) &&
                                !trimmedTopology->isCubeActive(x-1,y  ,z) &&
                                !trimmedTopology->isCubeActive(x  ,y-1,z) &&
                                !trimmedTopology->isCubeActive(x-1,y-1,z))
                                continue;
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y,z+1);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring);
                        }

            }
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                // quads along XY plane
                // lines (x,y,z) -> (x+1,y+1,z)
                spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                spring1.ks = this->quadsStiffness / spring1.initpos;
                spring1.kd = this->quadsDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z)
                spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                spring2.ks = this->quadsStiffness / spring2.initpos;
                spring2.kd = this->quadsDamping / spring2.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x,y,z  ) &&
                                !trimmedTopology->isCubeActive(x,y,z-1))
                                continue;
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring2);
                        }
                // quads along XZ plane
                // lines (x,y,z) -> (x+1,y,z+1)
                spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                spring1.ks = this->quadsStiffness / spring1.initpos;
                spring1.kd = this->quadsDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y,z+1)
                spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                spring2.ks = this->quadsStiffness / spring2.initpos;
                spring2.kd = this->quadsDamping / spring2.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x,y  ,z) &&
                                !trimmedTopology->isCubeActive(x,y-1,z))
                                continue;
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y,z+1);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring2);
                        }
                // quads along YZ plane
                // lines (x,y,z) -> (x,y+1,z+1)
                spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->quadsStiffness / spring1.initpos;
                spring1.kd = this->quadsDamping / spring1.initpos;
                // lines (x,y+1,z) -> (x,y,z+1)
                spring2.initpos = (topology->getDy()-topology->getDz()).norm();
                spring2.ks = this->quadsStiffness / spring1.initpos;
                spring2.kd = this->quadsDamping / spring1.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x  ,y,z) &&
                                !trimmedTopology->isCubeActive(x-1,y,z))
                                continue;
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x,y+1,z+1);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring1);
                            spring2.m1 = topology->point(x,y+1,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring2);
                        }
            }
            if (this->cubesStiffness != 0.0 || this->cubesDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                typename RegularGridSpringForceField<DataTypes>::Spring spring3;
                typename RegularGridSpringForceField<DataTypes>::Spring spring4;
                // lines (x,y,z) -> (x+1,y+1,z+1)
                spring1.initpos = (topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->cubesStiffness / spring1.initpos;
                spring1.kd = this->cubesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z+1)
                spring2.initpos = (-topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring2.ks = this->cubesStiffness / spring2.initpos;
                spring2.kd = this->cubesDamping / spring2.initpos;
                // lines (x,y+1,z) -> (x+1,y,z+1)
                spring3.initpos = (topology->getDx()-topology->getDy()+topology->getDz()).norm();
                spring3.ks = this->cubesStiffness / spring3.initpos;
                spring3.kd = this->cubesDamping / spring3.initpos;
                // lines (x,y,z+1) -> (x+1,y+1,z)
                spring4.initpos = (topology->getDx()+topology->getDy()-topology->getDz()).norm();
                spring4.ks = this->cubesStiffness / spring4.initpos;
                spring4.kd = this->cubesDamping / spring4.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x,y,z))
                                continue;
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z+1);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z+1);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring2);
                            spring3.m1 = topology->point(x,y+1,z);
                            spring3.m2 = topology->point(x+1,y,z+1);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring3);
                            spring4.m1 = topology->point(x,y,z+1);
                            spring4.m2 = topology->point(x+1,y+1,z);
                            this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, index++, spring4);
                        }
            }
        }
    }
}

template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2)
{
    // Calc any custom springs
    this->StiffSpringForceField<DataTypes>::addDForce(df1, df2, dx1, dx2);
    // Compute topological springs
    const vector<Spring>& springs = this->springs.getValue();
    if (this->mstate1==this->mstate2)
    {
        if (topology != NULL)
        {
            const int nx = topology->getNx();
            const int ny = topology->getNy();
            const int nz = topology->getNz();
            int index = springs.size();
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
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x,y  ,z  ) &&
                                !trimmedTopology->isCubeActive(x,y-1,z  ) &&
                                !trimmedTopology->isCubeActive(x,y  ,z-1) &&
                                !trimmedTopology->isCubeActive(x,y-1,z-1))
                                continue;
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x+1,y,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring);
                        }
                // lines along Y
                spring.initpos = topology->getDy().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x  ,y,z  ) &&
                                !trimmedTopology->isCubeActive(x-1,y,z  ) &&
                                !trimmedTopology->isCubeActive(x  ,y,z-1) &&
                                !trimmedTopology->isCubeActive(x-1,y,z-1))
                                continue;
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y+1,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring);
                        }
                // lines along Z
                spring.initpos = topology->getDz().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x  ,y  ,z) &&
                                !trimmedTopology->isCubeActive(x-1,y  ,z) &&
                                !trimmedTopology->isCubeActive(x  ,y-1,z) &&
                                !trimmedTopology->isCubeActive(x-1,y-1,z))
                                continue;
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring);
                        }

            }
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                // quads along XY plane
                // lines (x,y,z) -> (x+1,y+1,z)
                spring1.initpos = (topology->getDx()+topology->getDy()).norm();
                spring1.ks = this->quadsStiffness / spring1.initpos;
                spring1.kd = this->quadsDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z)
                spring2.initpos = (topology->getDx()-topology->getDy()).norm();
                spring2.ks = this->quadsStiffness / spring2.initpos;
                spring2.kd = this->quadsDamping / spring2.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x,y,z  ) &&
                                !trimmedTopology->isCubeActive(x,y,z-1))
                                continue;
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);
                        }
                // quads along XZ plane
                // lines (x,y,z) -> (x+1,y,z+1)
                spring1.initpos = (topology->getDx()+topology->getDz()).norm();
                spring1.ks = this->quadsStiffness / spring1.initpos;
                spring1.kd = this->quadsDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y,z+1)
                spring2.initpos = (topology->getDx()-topology->getDz()).norm();
                spring2.ks = this->quadsStiffness / spring2.initpos;
                spring2.kd = this->quadsDamping / spring2.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x,y  ,z) &&
                                !trimmedTopology->isCubeActive(x,y-1,z))
                                continue;
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);
                        }
                // quads along YZ plane
                // lines (x,y,z) -> (x,y+1,z+1)
                spring1.initpos = (topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->quadsStiffness / spring1.initpos;
                spring1.kd = this->quadsDamping / spring1.initpos;
                // lines (x,y+1,z) -> (x,y,z+1)
                spring1.initpos = (topology->getDy()-topology->getDz()).norm();
                spring1.ks = this->linesStiffness / spring1.initpos;
                spring1.kd = this->linesDamping / spring1.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x  ,y,z) &&
                                !trimmedTopology->isCubeActive(x-1,y,z))
                                continue;
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x,y+1,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                            spring2.m1 = topology->point(x,y+1,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);
                        }
            }
            if (this->cubesStiffness != 0.0 || this->cubesDamping != 0.0)
            {
                typename RegularGridSpringForceField<DataTypes>::Spring spring1;
                typename RegularGridSpringForceField<DataTypes>::Spring spring2;
                typename RegularGridSpringForceField<DataTypes>::Spring spring3;
                typename RegularGridSpringForceField<DataTypes>::Spring spring4;
                // lines (x,y,z) -> (x+1,y+1,z+1)
                spring1.initpos = (topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring1.ks = this->cubesStiffness / spring1.initpos;
                spring1.kd = this->cubesDamping / spring1.initpos;
                // lines (x+1,y,z) -> (x,y+1,z+1)
                spring2.initpos = (-topology->getDx()+topology->getDy()+topology->getDz()).norm();
                spring2.ks = this->cubesStiffness / spring2.initpos;
                spring2.kd = this->cubesDamping / spring2.initpos;
                // lines (x,y+1,z) -> (x+1,y,z+1)
                spring3.initpos = (topology->getDx()-topology->getDy()+topology->getDz()).norm();
                spring3.ks = this->cubesStiffness / spring3.initpos;
                spring3.kd = this->cubesDamping / spring3.initpos;
                // lines (x,y,z+1) -> (x+1,y+1,z)
                spring4.initpos = (topology->getDx()+topology->getDy()-topology->getDz()).norm();
                spring4.ks = this->cubesStiffness / spring4.initpos;
                spring4.kd = this->cubesDamping / spring4.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx-1; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x,y,z))
                                continue;
                            spring1.m1 = topology->point(x,y,z);
                            spring1.m2 = topology->point(x+1,y+1,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring1);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring2);
                            spring3.m1 = topology->point(x,y+1,z);
                            spring3.m2 = topology->point(x+1,y,z+1);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring3);
                            spring4.m1 = topology->point(x,y,z+1);
                            spring4.m2 = topology->point(x+1,y+1,z);
                            this->addSpringDForce(df1,dx1,df2,dx2, index++, spring4);
                        }
            }
        }
    }
}



template<class DataTypes>
void RegularGridSpringForceField<DataTypes>::draw()
{
    if (!((this->mstate1 == this->mstate2)?this->getContext()->getShowForceFields():this->getContext()->getShowInteractionForceFields())) return;
    assert(this->mstate1);
    assert(this->mstate2);
    // Draw any custom springs
    this->StiffSpringForceField<DataTypes>::draw();
    // Compute topological springs
    const VecCoord& p1 = *this->mstate1->getX();
    const VecCoord& p2 = *this->mstate2->getX();
    glDisable(GL_LIGHTING);
    glColor4f(0.5,0.5,0.5,1);
    glBegin(GL_LINES);
    if (this->mstate1==this->mstate2)
    {
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
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x,y  ,z  ) &&
                                !trimmedTopology->isCubeActive(x,y-1,z  ) &&
                                !trimmedTopology->isCubeActive(x,y  ,z-1) &&
                                !trimmedTopology->isCubeActive(x,y-1,z-1))
                                continue;
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x+1,y,z);
                            helper::gl::glVertexT(p1[spring.m1]);
                            helper::gl::glVertexT(p2[spring.m2]);
                        }
                // lines along Y
                spring.initpos = topology->getDy().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz; z++)
                    for (int y=0; y<ny-1; y++)
                        for (int x=0; x<nx; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x  ,y,z  ) &&
                                !trimmedTopology->isCubeActive(x-1,y,z  ) &&
                                !trimmedTopology->isCubeActive(x  ,y,z-1) &&
                                !trimmedTopology->isCubeActive(x-1,y,z-1))
                                continue;
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y+1,z);
                            helper::gl::glVertexT(p1[spring.m1]);
                            helper::gl::glVertexT(p2[spring.m2]);
                        }
                // lines along Z
                spring.initpos = topology->getDz().norm();
                spring.ks = this->linesStiffness / spring.initpos;
                spring.kd = this->linesDamping / spring.initpos;
                for (int z=0; z<nz-1; z++)
                    for (int y=0; y<ny; y++)
                        for (int x=0; x<nx; x++)
                        {
                            if (trimmedTopology != NULL &&
                                !trimmedTopology->isCubeActive(x  ,y  ,z) &&
                                !trimmedTopology->isCubeActive(x-1,y  ,z) &&
                                !trimmedTopology->isCubeActive(x  ,y-1,z) &&
                                !trimmedTopology->isCubeActive(x-1,y-1,z))
                                continue;
                            spring.m1 = topology->point(x,y,z);
                            spring.m2 = topology->point(x,y,z+1);
                            helper::gl::glVertexT(p1[spring.m1]);
                            helper::gl::glVertexT(p2[spring.m2]);
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
                            helper::gl::glVertexT(p1[spring1.m1]);
                            helper::gl::glVertexT(p2[spring1.m2]);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z);
                            helper::gl::glVertexT(p1[spring2.m1]);
                            helper::gl::glVertexT(p2[spring2.m2]);
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
                            helper::gl::glVertexT(p1[spring1.m1]);
                            helper::gl::glVertexT(p2[spring1.m2]);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            helper::gl::glVertexT(p1[spring2.m1]);
                            helper::gl::glVertexT(p2[spring2.m2]);
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
                            helper::gl::glVertexT(p1[spring1.m1]);
                            helper::gl::glVertexT(p2[spring1.m2]);
                            spring2.m1 = topology->point(x,y+1,z);
                            spring2.m2 = topology->point(x,y,z+1);
                            helper::gl::glVertexT(p1[spring2.m1]);
                            helper::gl::glVertexT(p2[spring2.m2]);
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
                            helper::gl::glVertexT(p1[spring1.m1]);
                            helper::gl::glVertexT(p2[spring1.m2]);
                            spring2.m1 = topology->point(x+1,y,z);
                            spring2.m2 = topology->point(x,y+1,z+1);
                            helper::gl::glVertexT(p1[spring2.m1]);
                            helper::gl::glVertexT(p2[spring2.m2]);
                            spring3.m1 = topology->point(x,y+1,z);
                            spring3.m2 = topology->point(x+1,y,z+1);
                            helper::gl::glVertexT(p1[spring3.m1]);
                            helper::gl::glVertexT(p2[spring3.m2]);
                            spring4.m1 = topology->point(x,y,z+1);
                            spring4.m2 = topology->point(x+1,y+1,z);
                            helper::gl::glVertexT(p1[spring4.m1]);
                            helper::gl::glVertexT(p2[spring4.m2]);
                        }
            }
#endif
        }
    }
    glEnd();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
