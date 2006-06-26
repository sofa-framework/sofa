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
class DiagonalMass : public Core::Mass<DataTypes>, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
protected:
    std::vector<MassType> masses;

    class Loader;
public:
    DiagonalMass();

    DiagonalMass(Core::MechanicalModel<DataTypes>* mmodel, const std::string& name="");

    ~DiagonalMass();

    bool load(const char *filename);

    void clear();

    void addMass(const MassType& mass);

    void resize(int vsize);

    // -- Mass interface
    void addMDx(VecDeriv& f, const VecDeriv& dx);

    void accFromF(VecDeriv& a, const VecDeriv& f);

    void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
