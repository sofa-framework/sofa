#ifndef SOFA_COMPONENTS_UNIFORMMASS_H
#define SOFA_COMPONENTS_UNIFORMMASS_H

#include "Common/Vec3Types.h"
#include "Sofa/Core/Mass.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Components/CoordinateSystem.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class DataTypes, class MassType>
class UniformMass : public Core::Mass<DataTypes>, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
protected:
    MassType mass;
    DataField<double> totalMass; ///< if >0 : total mass of this body

public:
    UniformMass();

    UniformMass(Core::MechanicalModel<DataTypes>* mmodel);

    ~UniformMass();

    virtual const char* getTypeName() const
    {
        return "UniformMass";
    }

    void setMass(const MassType& mass);

    void setTotalMass(double m);

    // -- Mass interface

    void init();

    void addMDx(VecDeriv& f, const VecDeriv& dx);

    void accFromF(VecDeriv& a, const VecDeriv& f);

    void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    double getKineticEnergy(const VecDeriv& v);  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const VecCoord& x);   ///< Mgx potential in a uniform gravity field, null at origin

    // -- VisualModel interface

    void draw();

    bool addBBox(double* minBBox, double* maxBBox);

    void initTextures()
    { }

    void update()
    { }
};

} // namespace Components

} // namespace Sofa

#endif

