#ifndef SOFA_COMPONENTS_DIAGONALMASS_H
#define SOFA_COMPONENTS_DIAGONALMASS_H

#include "Common/Vec3Types.h"
#include "Sofa/Core/Mass.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class DataTypes, class MassType>
class DiagonalMass : public Core::Mass, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
protected:
    Core::MechanicalModel<DataTypes>* mmodel;

    std::vector<MassType> masses;

    Deriv gravity;

    class Loader;
public:
    DiagonalMass();

    DiagonalMass(Core::MechanicalModel<DataTypes>* mmodel, const std::string& name="");

    ~DiagonalMass();

    void setMechanicalModel(Core::MechanicalModel<DataTypes>* mm);

    bool load(const char *filename);

    void clear();

    void addMass(const MassType& mass);

    void resize(int vsize);

    // -- Mass interface
    void addMDx();

    void accFromF();

    void computeForce();

    void setGravity( const Deriv& g );

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
