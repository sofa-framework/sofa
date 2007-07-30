#ifndef SOFA_COMPONENT_FORCEFIELD_MESHSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_MESHSPRINGFORCEFIELD_INL

#include <sofa/component/forcefield/MeshSpringForceField.h>
#include <sofa/component/forcefield/StiffSpringForceField.inl>
#include <sofa/component/topology/MeshTopology.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace forcefield
{

template <class DataTypes>
double MeshSpringForceField<DataTypes>::getPotentialEnergy()
{
    cerr<<"MeshSpringForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}

template<class DataTypes>
void MeshSpringForceField<DataTypes>::addSpring(std::set<std::pair<int,int> >& sset, int m1, int m2, Real stiffness, Real damping)
{
    if (m1<m2)
    {
        if (sset.count(std::make_pair(m1,m2))>0) return;
        sset.insert(std::make_pair(m1,m2));
    }
    else
    {
        if (sset.count(std::make_pair(m2,m1))>0) return;
        sset.insert(std::make_pair(m2,m1));
    }
    Real l = ((*this->mstate2->getX())[m2] - (*this->mstate1->getX())[m1]).norm();
    this->springs.beginEdit()->push_back(typename SpringForceField<DataTypes>::Spring(m1,m2,stiffness/l, damping/l, l));
    this->springs.endEdit();
}

template<class DataTypes>
void MeshSpringForceField<DataTypes>::init()
{
    this->StiffSpringForceField<DataTypes>::clear();
    if(!(this->mstate1) || !(this->mstate2))
        this->mstate2 = this->mstate1 = dynamic_cast<sofa::core::componentmodel::behavior::MechanicalState<DataTypes> *>(this->getContext()->getMechanicalState());

    if (this->mstate1==this->mstate2)
    {
        topology::MeshTopology* topology = dynamic_cast<topology::MeshTopology*>(this->mstate1->getContext()->getTopology());
        if (topology != NULL)
        {
            std::set< std::pair<int,int> > sset;
            int n;
            Real s, d;
            if (this->linesStiffness != 0.0 || this->linesDamping != 0.0)
            {
                s = this->linesStiffness;
                d = this->linesDamping;
                n = topology->getNbLines();
                for (int i=0; i<n; ++i)
                {
                    topology::MeshTopology::Line e = topology->getLine(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                }
            }
            if (this->trianglesStiffness != 0.0 || this->trianglesDamping != 0.0)
            {
                s = this->trianglesStiffness;
                d = this->trianglesDamping;
                n = topology->getNbTriangles();
                for (int i=0; i<n; ++i)
                {
                    topology::MeshTopology::Triangle e = topology->getTriangle(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                }
            }
            if (this->quadsStiffness != 0.0 || this->quadsDamping != 0.0)
            {
                s = this->quadsStiffness;
                d = this->quadsDamping;
                n = topology->getNbQuads();
                for (int i=0; i<n; ++i)
                {
                    topology::MeshTopology::Quad e = topology->getQuad(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[0], e[3], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                    this->addSpring(sset, e[1], e[3], s, d);
                    this->addSpring(sset, e[2], e[3], s, d);
                }
            }
            if (this->tetrasStiffness != 0.0 || this->tetrasDamping != 0.0)
            {
                s = this->tetrasStiffness;
                d = this->tetrasDamping;
                n = topology->getNbTetras();
                for (int i=0; i<n; ++i)
                {
                    topology::MeshTopology::Tetra e = topology->getTetra(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[0], e[3], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                    this->addSpring(sset, e[1], e[3], s, d);
                    this->addSpring(sset, e[2], e[3], s, d);
                }
            }
            if (this->cubesStiffness != 0.0 || this->cubesDamping != 0.0)
            {
                s = this->cubesStiffness;
                d = this->cubesDamping;
                n = topology->getNbCubes();
                for (int i=0; i<n; ++i)
                {
                    if (!topology->isCubeActive(i)) continue;
                    topology::MeshTopology::Cube e = topology->getCube(i);
                    for (int i=0; i<8; i++)
                        for (int j=i+1; j<8; j++)
                            this->addSpring(sset, e[i], e[j], s, d);
                }
            }
        }
    }
    this->StiffSpringForceField<DataTypes>::init();
}

template<class DataTypes>
void MeshSpringForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->StiffSpringForceField<DataTypes>::parse(arg);
    if (arg->getAttribute("stiffness"))          this->setStiffness         ((Real)atof(arg->getAttribute("stiffness")));
    if (arg->getAttribute("linesStiffness"))     this->setLinesStiffness    ((Real)atof(arg->getAttribute("linesStiffness")));
    if (arg->getAttribute("trianglesStiffness")) this->setTrianglesStiffness((Real)atof(arg->getAttribute("trianglesStiffness")));
    if (arg->getAttribute("quadsStiffness"))     this->setQuadsStiffness    ((Real)atof(arg->getAttribute("quadsStiffness")));
    if (arg->getAttribute("tetrasStiffness"))    this->setTetrasStiffness   ((Real)atof(arg->getAttribute("tetrasStiffness")));
    if (arg->getAttribute("cubesStiffness"))     this->setCubesStiffness    ((Real)atof(arg->getAttribute("cubesStiffness")));
    if (arg->getAttribute("damping"))            this->setDamping           ((Real)atof(arg->getAttribute("damping")));
    if (arg->getAttribute("linesDamping"))       this->setLinesDamping      ((Real)atof(arg->getAttribute("linesDamping")));
    if (arg->getAttribute("trianglesDamping"))   this->setTrianglesDamping  ((Real)atof(arg->getAttribute("trianglesDamping")));
    if (arg->getAttribute("quadsDamping"))       this->setQuadsDamping      ((Real)atof(arg->getAttribute("quadsDamping")));
    if (arg->getAttribute("tetrasDamping"))      this->setTetrasDamping     ((Real)atof(arg->getAttribute("tetrasDamping")));
    if (arg->getAttribute("cubesDamping"))       this->setCubesDamping      ((Real)atof(arg->getAttribute("cubesDamping")));
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
