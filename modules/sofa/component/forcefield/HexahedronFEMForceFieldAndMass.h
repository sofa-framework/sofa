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
    typedef HexahedronFEMForceField<DataTypes> HexahedronFEMForceFieldT;
    typedef Mass<DataTypes> MassT;


    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename HexahedronFEMForceFieldT::Mat33 Mat33;
    typedef typename HexahedronFEMForceFieldT::VecElement VecElement;
    typedef typename HexahedronFEMForceFieldT::VecElementStiffness VecElementMass;
    typedef typename HexahedronFEMForceFieldT::ElementStiffness ElementMass;
    typedef helper::vector<Real> MassVector;



    HexahedronFEMForceFieldAndMass();


    virtual void init( );
    virtual void reinit( );

    virtual void computeElementMasses( ); ///< compute the mass matrices
    virtual void computeElementMass( ElementMass &Mass, const helper::fixed_array<Coord,8> &nodes, const int elementIndice); ///< compute the mass matrix of an element
    Real integrateMass( int signx, int signy, int signz, Real l0, Real l1, Real l2 );

    virtual std::string getTemplateName() const;

    // -- Mass interface
    virtual  void addMDx(VecDeriv& f, const VecDeriv& dx, double factor = 1.0);

    virtual  void accFromF(VecDeriv& a, const VecDeriv& f);

    virtual  void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual double getKineticEnergy(const VecDeriv& /*v*/)  ///< vMv/2 using dof->getV()
    {std::cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::getKineticEnergy not yet implemented\n"; return 0;}

    virtual double getPotentialEnergy(const VecCoord& /*x*/)   ///< Mgx potential in a uniform gravity field, null at origin
    {std::cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::getPotentialEnergy not yet implemented\n"; return 0;}

    virtual void addDForce(VecDeriv& df, const VecDeriv& dx);

    virtual void addGravityToV(double dt);

    double getElementMass(unsigned int index);

    // visual model

    virtual void draw();

    virtual bool addBBox(double* minBBox, double* maxBBox);

    virtual void initTextures() { }

    virtual void update() { }



    void setDensity(Real d) {_density.setValue( d );}
    Real getDensity() {return _density.getValue();}



protected :

    Data<VecElementMass> _elementMasses; ///< mass matrices per element

    Data<Real> _density;

    MassVector _particleMasses; ///< masses per particle in order to compute gravity


};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
