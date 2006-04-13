#ifndef SOFA_COMPONENTS_DIAGONALMASS_INL
#define SOFA_COMPONENTS_DIAGONALMASS_INL

#include "DiagonalMass.h"
#include "Scene.h"
#include "MassSpringLoader.h"
#include "GL/template.h"
#include "Common/RigidTypes.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::DiagonalMass()
    : mmodel(NULL)
{
    DataTypes::set(gravity,0,-9.8,0);
}


template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::DiagonalMass(Core::MechanicalModel<DataTypes>* mmodel, const std::string& /*name*/)
    : mmodel(mmodel)
{
    DataTypes::set(gravity,0,-9.8,0);
}

template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::~DiagonalMass()
{
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::setMechanicalModel(Core::MechanicalModel<DataTypes>* mm)
{
    this->mmodel = mm;
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::clear()
{
    this->masses.clear();
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addMass(const MassType& m)
{
    this->masses.push_back(m);
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::resize(int vsize)
{
    this->masses.resize(vsize);
}

// -- Mass interface
template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addMDx()
{
    VecDeriv& res = *mmodel->getF();
    VecDeriv& dx = *mmodel->getDx();
    for (unsigned int i=0; i<dx.size(); i++)
    {
        res[i] += dx[i] * masses[i];
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::accFromF()
{
    VecDeriv& a = *mmodel->getDx();
    VecDeriv& f = *mmodel->getF();
    for (unsigned int i=0; i<f.size(); i++)
    {
        a[i] = f[i] / masses[i];
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::computeForce()
{
    VecDeriv& f = *mmodel->getF();
    for (unsigned int i=0; i<f.size(); i++)
    {
        f[i] += gravity * masses[i];
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::setGravity( const Deriv& g )
{
    this->gravity = g;
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::draw()
{
    if (!Scene::getInstance()->getShowBehaviorModels()) return;
    VecCoord& x = *mmodel->getX();
    glDisable (GL_LIGHTING);
    glPointSize(5);
    glColor4f (1,1,1,1);
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<x.size(); i++)
    {
        GL::glVertexT(x[i]);
    }
    glEnd();
}

template <class DataTypes, class MassType>
class DiagonalMass<DataTypes, MassType>::Loader : public MassSpringLoader
{
public:
    DiagonalMass<DataTypes, MassType>* dest;
    Loader(DiagonalMass<DataTypes, MassType>* dest) : dest(dest) {}
    virtual void addMass(double /*px*/, double /*py*/, double /*pz*/, double /*vx*/, double /*vy*/, double /*vz*/, double mass, double /*elastic*/, bool /*fixed*/, bool /*surface*/)
    {
        dest->addMass(MassType(mass));
    }
    virtual void setGravity(double gx, double gy, double gz)
    {
        typename DataTypes::Deriv g;
        DataTypes::set(g,gx,gy,gz);
        dest->setGravity(g);
    }
};

template <class DataTypes, class MassType>
bool DiagonalMass<DataTypes, MassType>::load(const char *filename)
{
    clear();
    if (filename!=NULL && filename[0]!='\0')
    {
        Loader loader(this);
        return loader.load(filename);
    }
    else return false;
}

// Specialization for rigids
template <>
void DiagonalMass<RigidTypes, RigidMass>::draw();


} // namespace Components

} // namespace Sofa

#endif
