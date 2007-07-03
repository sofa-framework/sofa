#ifndef SOFA_COMPONENT_CONSTANTFORCEFIELD_H
#define SOFA_COMPONENT_CONSTANTFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/DataField.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/** Apply constant forces to given degrees of freedom.  */
template<class DataTypes>
class ConstantForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;


public:

    DataField<vector<unsigned> > points;
    DataField<VecDeriv > forces;

    ConstantForceField();

    /// Add the forces
    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    /// Constant force has null variation
    virtual void addDForce (VecDeriv& , const VecDeriv& ) {}


    virtual double getPotentialEnergy(const VecCoord& x);


    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
    bool addBBox(double* minBBox, double* maxBBox);

};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
