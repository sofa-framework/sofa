#ifndef SOFA_COMPONENTS_UNIFORMMASS_H
#define SOFA_COMPONENTS_UNIFORMMASS_H

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
class UniformMass : public Core::Mass, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
protected:
    Core::MechanicalModel<DataTypes>* mmodel;

    MassType mass;

    Deriv gravity;
public:
    UniformMass();

    UniformMass(Core::MechanicalModel<DataTypes>* mmodel, const std::string& name="");

    ~UniformMass();

    void setMechanicalModel(Core::MechanicalModel<DataTypes>* mm);

    void setMass(const MassType& mass);

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
