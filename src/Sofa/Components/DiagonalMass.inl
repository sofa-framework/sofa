#ifndef SOFA_COMPONENTS_DIAGONALMASS_INL
#define SOFA_COMPONENTS_DIAGONALMASS_INL

#include "DiagonalMass.h"
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
{
}


template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::DiagonalMass(Core::MechanicalModel<DataTypes>* mmodel, const std::string& /*name*/)
    : Core::Mass<DataTypes>(mmodel)
{
}

template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::~DiagonalMass()
{
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
void DiagonalMass<DataTypes, MassType>::addMDx(VecDeriv& res, const VecDeriv& dx)
{
    for (unsigned int i=0; i<dx.size(); i++)
    {
        res[i] += dx[i] * masses[i];
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::accFromF(VecDeriv& a, const VecDeriv& f)
{
    for (unsigned int i=0; i<f.size(); i++)
    {
        a[i] = f[i] / masses[i];
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::computeForce(VecDeriv& f, const VecCoord& /*x*/, const VecDeriv& /*v*/)
{
    /*Deriv gravity ( getContext()->getGravity() );
    for (unsigned int i=0;i<f.size();i++)
    {
    	f[i] += gravity * masses[i];
    }*/

    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);

    // velocity-based stuff
#if 0
    Core::Context::SpatialVelocity vframe = getContext()->getSpatialVelocity();
    Core::Context::Vec aframe = getContext()->getLinearAcceleration() ;
    // project back to local frame
    vframe = getContext()->getLocalToWorld() / vframe;
    aframe = getContext()->getLocalToWorld().backProjectVector( aframe );

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); i++)
    {
        f[i] += theGravity*masses[i] + inertiaForce(vframe,aframe,masses[i],x[i],v[i]);
    }
#else
    for (unsigned int i=0; i<f.size(); i++)
    {
        f[i] += theGravity*masses[i];
    }
#endif
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::draw()
{
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *this->mmodel->getX();
    glDisable (GL_LIGHTING);
    glPointSize(2);
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
