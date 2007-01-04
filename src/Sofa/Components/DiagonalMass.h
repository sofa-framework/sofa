#ifndef SOFA_COMPONENTS_DIAGONALMASS_H
#define SOFA_COMPONENTS_DIAGONALMASS_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "Common/Vec3Types.h"
#include "Sofa/Core/Mass.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Abstract/Event.h"
#include "Sofa/Components/Common/vector.h"
#include "Sofa/Components/Topology/PointData.h"
#include "Sofa/Components/Topology/PointData.inl"
namespace Sofa
{
using namespace Abstract;
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
    typedef typename DataTypes::Real Real;

    typedef PointData<MassType> VecMass;
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

    DiagonalMass(Core::MechanicalModel<DataTypes>* mmodel, const std::string& name="");

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
}
;

} // namespace Components

} // namespace Sofa

#endif
