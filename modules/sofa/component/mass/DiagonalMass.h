/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MASS_DIAGONALMASS_H
#define SOFA_COMPONENT_MASS_DIAGONALMASS_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/component/topology/PointData.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace mass
{

// template<class Vec> void readVec1(Vec& vec, const char* str);

template <class DataTypes, class MassType>
class DiagonalMass : public core::componentmodel::behavior::Mass<DataTypes>, public core::VisualModel
{
public:
    typedef core::componentmodel::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef sofa::component::topology::PointData<MassType> VecMass;
    typedef helper::vector<MassType> MassVector;


    typedef enum
    {
        TOPOLOGY_UNKNOWN=0,
        TOPOLOGY_EDGESET=1,
        TOPOLOGY_TRIANGLESET=2,
        TOPOLOGY_TETRAHEDRONSET=3
    } TopologyType;

    Data< VecMass > f_mass;
    /// the mass density used to compute the mass from a mesh topology and geometry
    Data< Real > m_massDensity;
protected:
    //VecMass masses;

    class Loader;
    /// The type of topology to build the mass from the topology
    TopologyType topologyType;

public:
    DiagonalMass();

    ~DiagonalMass();

    //virtual const char* getTypeName() const { return "DiagonalMass"; }

    bool load(const char *filename);

    void clear();

    virtual void init();
    virtual void parse(core::objectmodel::BaseObjectDescription* arg);

    TopologyType getMassTopologyType() const
    {
        return topologyType;
    }
    Real getMassDensity() const
    {
        return m_massDensity.getValue();
    }


    // handle topological changes
    virtual void handleTopologyChange();

    void setMassDensity(Real m)
    {
        m_massDensity.setValue(m);
    }


    void addMass(const MassType& mass);

    void resize(int vsize);

    // -- Mass interface
    void addMDx(VecDeriv& f, const VecDeriv& dx, double factor = 1.0);

    void accFromF(VecDeriv& a, const VecDeriv& f);

    void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    double getKineticEnergy(const VecDeriv& v);  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const VecCoord& x);   ///< Mgx potential in a uniform gravity field, null at origin

    void addGravityToV(double dt/*, defaulttype::BaseVector& v*/);

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
