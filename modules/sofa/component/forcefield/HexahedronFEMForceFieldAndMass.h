#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONANDMASSFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONANDMASSFEMFORCEFIELD_H


#include "HexahedronFEMForceField.h"
#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/VisualModel.h>
#include <sofa/helper/gl/template.h>
namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using sofa::helper::vector;
using sofa::core::componentmodel::behavior::Mass;

/** Compute Finite Element forces based on hexahedral elements including continuum mass matrices
 */
template<class DataTypes>
class HexahedronFEMForceFieldAndMass : virtual public Mass<DataTypes>, virtual public HexahedronFEMForceField<DataTypes>
{
public:
    typedef HexahedronFEMForceField<DataTypes> HexahedronFEMForceField;
    typedef Mass<DataTypes> Mass;


    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename HexahedronFEMForceField::Mat33 Mat33;
    typedef typename HexahedronFEMForceField::VecElement VecElement;
    typedef typename HexahedronFEMForceField::VecElementStiffness VecElementMass;
    typedef typename HexahedronFEMForceField::ElementStiffness ElementMass;








    HexahedronFEMForceFieldAndMass();


    virtual void init( );
    virtual void reinit( );

    void computeElementMasses( ); ///< compute the mass matrices
    void computeElementMass( ElementMass &Mass, const Vec<8,Coord> &nodes); ///< compute the mass matrix of an element
    Real integrateMass( const Real xmin, const Real xmax, const Real ymin, const Real ymax, const Real zmin, const Real zmax, int signx0, int signy0, int signz0, int signx1, int signy1, int signz1  );

    virtual std::string getTemplateName() const;

    // -- Mass interface
    void addMDx(VecDeriv& f, const VecDeriv& dx, double factor = 1.0);

    void accFromF(VecDeriv& a, const VecDeriv& f);

    void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    double getKineticEnergy(const VecDeriv& /*v*/)  ///< vMv/2 using dof->getV()
    {std::cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::getKineticEnergy not yet implemented\n"; return 0;}

    double getPotentialEnergy(const VecCoord& /*x*/)   ///< Mgx potential in a uniform gravity field, null at origin
    {std::cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::getPotentialEnergy not yet implemented\n"; return 0;}

    virtual void addDForce(VecDeriv& df, const VecDeriv& dx);


    // visual model

    virtual void draw();

    bool addBBox(double* minBBox, double* maxBBox);

    void initTextures() { }

    void update() { }



    void setDensity(Real d) {_density.setValue( d );}
    Real getDensity() {return _density.getValue();}



protected :

    VecElementMass _elementMasses; ///< mass matrices per element

    DataField<Real> _density;


};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
