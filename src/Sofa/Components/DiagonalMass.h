#pragma once

#include "Common/Vec3Types.h"
#include "Sofa/Core/Mass.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Components/Common/vector.h"

namespace Sofa
{

namespace Components
{

using namespace Common;
// using Abstract::Field;

template <class DataTypes, class MassType>
class DiagonalMass : public Core::Mass<DataTypes>, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef vector<MassType> VecMass;
protected:
    //VecMass masses;

    class Loader;
public:
    DiagonalMass();

    DiagonalMass(Core::MechanicalModel<DataTypes>* mmodel, const std::string& name="");

    ~DiagonalMass();

    virtual const char* getTypeName() const { return "DiagonalMass"; }

    bool load(const char *filename);

    void clear();

    DataField< VecMass > f_mass;

    void addMass(const MassType& mass);

    void resize(int vsize);

    // -- Mass interface
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
}
;

} // namespace Components

} // namespace Sofa

