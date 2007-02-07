#ifndef SOFA_COMPONENT_MASS_DIAGONALMASS_H
#define SOFA_COMPONENT_MASS_DIAGONALMASS_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/helper/vector.h>
#include <sofa/component/topology/PointData.h>
#include <sofa/component/topology/PointData.inl>

namespace sofa
{

namespace component
{

namespace mass
{

using sofa::helper::vector;

template <class DataTypes, class MassType>
class DiagonalMass : public core::componentmodel::behavior::Mass<DataTypes>, public core::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef topology::PointData<MassType> VecMass;
    typedef vector<MassType> MassVector;

    typedef enum
    {
        TOPOLOGY_UNKNOWN=0,
        TOPOLOGY_EDGESET=1
    } TopologyType;

    DataField< VecMass > f_mass;
    /// the mass density used to compute the mass from a mesh topology and geometry
    DataField< Real > m_massDensity;
protected:
    //VecMass masses;

    class Loader;
    /// The type of topology to build the mass from the topology
    TopologyType topologyType;

public:
    DiagonalMass();

    DiagonalMass(core::componentmodel::behavior::MechanicalState<DataTypes>* mstate, const std::string& name="");

    ~DiagonalMass();

    virtual const char* getTypeName() const { return "DiagonalMass"; }

    bool load(const char *filename);

    void clear();

    virtual void init();
    // handle topological changes

    virtual void handleEvent( Event* );

    TopologyType getMassTopologyType() const
    {
        return topologyType;
    }
    Real getMassDensity() const
    {
        return m_massDensity.getValue();
    }

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
};

} // namespace mass

} // namespace component

} // namespace sofa

#endif
