#ifndef SOFA_COMPONENT_MASS_UNIFORMMASS_H
#define SOFA_COMPONENT_MASS_UNIFORMMASS_H

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/contextobject/CoordinateSystem.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

template <class DataTypes, class MassType>
class UniformMass : public core::componentmodel::behavior::Mass<DataTypes>, public core::VisualModel
{
public:
    typedef core::componentmodel::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
protected:
    DataField<MassType> mass;    ///< the mass of each particle
    DataField<double> totalMass; ///< if >0 : total mass of this body

public:
    UniformMass();

    UniformMass(core::componentmodel::behavior::MechanicalState<DataTypes>* mstate);

    ~UniformMass();

    //virtual const char* getTypeName() const { return "UniformMass"; }

    void setMass(const MassType& mass);

    double getTotalMass() { return totalMass.getValue();}
    void setTotalMass(double m);

    // -- Mass interface

    virtual void parse(core::objectmodel::BaseObjectDescription* arg);
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

} // namespace mass

} // namespace component

} // namespace sofa

#endif

