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

public:
    UniformMass();

    UniformMass(Core::MechanicalModel<DataTypes>* mmodel);

    ~UniformMass();

    void setMass(const MassType& mass);

    // -- Mass interface
    void addMDx(VecDeriv& f, const VecDeriv& dx);

    void accFromF(VecDeriv& a, const VecDeriv& f);

    void computeForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    // -- VisualModel interface

    void draw();

    void initTextures()
    { }

    void update()
    { }
};

} // namespace Components

} // namespace Sofa

#endif

