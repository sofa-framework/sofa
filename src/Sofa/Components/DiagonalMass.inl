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
    : f_mass( dataField(&f_mass, "mass", "values of the particles' masses") )
{
    //f_mass = addField( &masses, "mass", "values of the particles' masses");
}


template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::DiagonalMass(Core::MechanicalModel<DataTypes>* mmodel, const std::string& /*name*/)
    : Core::Mass<DataTypes>(mmodel)
    , f_mass( dataField(&f_mass, "mass", "values of the particles' masses") )
{
}

template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::~DiagonalMass()
{
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::clear()
{
    VecMass& masses = *f_mass.beginEdit();
    masses.clear();
    f_mass.endEdit();
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addMass(const MassType& m)
{
    VecMass& masses = *f_mass.beginEdit();
    masses.push_back(m);
    f_mass.endEdit();
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::resize(int vsize)
{
    VecMass& masses = *f_mass.beginEdit();
    masses.resize(vsize);
    f_mass.endEdit();
}

// -- Mass interface
template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addMDx(VecDeriv& res, const VecDeriv& dx)
{
    const VecMass& masses = f_mass.getValue();
    for (unsigned int i=0; i<dx.size(); i++)
    {
        res[i] += dx[i] * masses[i];
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::accFromF(VecDeriv& a, const VecDeriv& f)
{
    const VecMass& masses = f_mass.getValue();
    for (unsigned int i=0; i<f.size(); i++)
    {
        a[i] = f[i] / masses[i];
    }
}

template <class DataTypes, class MassType>
double DiagonalMass<DataTypes, MassType>::getKineticEnergy( const VecDeriv& v )
{
    const VecMass& masses = f_mass.getValue();
    double e = 0;
    for (unsigned int i=0; i<masses.size(); i++)
    {
        e += v[i]*masses[i]*v[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }
    return e/2;
}

template <class DataTypes, class MassType>
double DiagonalMass<DataTypes, MassType>::getPotentialEnergy( const VecCoord& x )
{
    const VecMass& masses = f_mass.getValue();
    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<masses.size(); i++)
    {

        e -= theGravity*masses[i]*x[i];
    }
    return e;
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    const VecMass& masses = f_mass.getValue();

    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);

    // velocity-based stuff
    Core::Context::SpatialVector vframe = getContext()->getVelocityInWorld();
    Core::Context::Vec3 aframe = getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = getContext()->getPositionInWorld() / vframe;
    aframe = getContext()->getPositionInWorld().backProjectVector( aframe );

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); i++)
    {
        f[i] += theGravity*masses[i] + Core::inertiaForce(vframe,aframe,masses[i],x[i],v[i]);
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::draw()
{
    if (!getContext()->getShowBehaviorModels()) return;
    const VecCoord& x = *this->mmodel->getX();
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
bool DiagonalMass<DataTypes, MassType>::addBBox(double* minBBox, double* maxBBox)
{
    const VecCoord& x = *this->mmodel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        //const Coord& p = x[i];
        double p[3] = {0.0, 0.0, 0.0};
        DataTypes::get(p[0],p[1],p[2],x[i]);
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    return true;
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
