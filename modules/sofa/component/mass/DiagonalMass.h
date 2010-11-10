/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MASS_DIAGONALMASS_H
#define SOFA_COMPONENT_MASS_DIAGONALMASS_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/component/topology/PointData.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/QuadSetGeometryAlgorithms.h>
#include <sofa/component/topology/HexahedronSetGeometryAlgorithms.h>

#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::component::topology;


// template<class Vec> void readVec1(Vec& vec, const char* str);

template<class DataTypes, class TMassType>
class DiagonalMassInternalData
{
public :

    typedef sofa::component::topology::PointData<TMassType> VecMass;
    typedef helper::vector<TMassType> MassVector;

};

template <class DataTypes, class TMassType>
class DiagonalMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DiagonalMass,DataTypes,TMassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef TMassType MassType;

    // In case of non 3D template
    typedef Vec<3,MassType>                            Vec3;
    typedef DataTypes     GeometricalTypes ; /// assumes the geometry object type is 3D

    typedef enum
    {
        TOPOLOGY_UNKNOWN=0,
        TOPOLOGY_EDGESET=1,
        TOPOLOGY_TRIANGLESET=2,
        TOPOLOGY_TETRAHEDRONSET=3,
        TOPOLOGY_QUADSET=4,
        TOPOLOGY_HEXAHEDRONSET=5
    } TopologyType;

    typedef typename DiagonalMassInternalData<DataTypes,TMassType>::VecMass VecMass;
    typedef typename DiagonalMassInternalData<DataTypes,TMassType>::MassVector MassVector;

    VecMass f_mass;

    /// the mass density used to compute the mass from a mesh topology and geometry
    Data< Real > m_massDensity;

    /// to display the center of gravity of the system
    Data< bool > showCenterOfGravity;
    Data< float > showAxisSize;
    core::objectmodel::DataFileName fileMass;

protected:
    //VecMass masses;

    class Loader;
    /// The type of topology to build the mass from the topology
    TopologyType topologyType;

public:

    sofa::core::topology::BaseMeshTopology* _topology;

    sofa::component::topology::EdgeSetGeometryAlgorithms<GeometricalTypes>* edgeGeo;
    sofa::component::topology::TriangleSetGeometryAlgorithms<GeometricalTypes>* triangleGeo;
    sofa::component::topology::QuadSetGeometryAlgorithms<GeometricalTypes>* quadGeo;
    sofa::component::topology::TetrahedronSetGeometryAlgorithms<GeometricalTypes>* tetraGeo;
    sofa::component::topology::HexahedronSetGeometryAlgorithms<GeometricalTypes>* hexaGeo;

    DiagonalMass();

    ~DiagonalMass();

    //virtual const char* getTypeName() const { return "DiagonalMass"; }

    bool load(const char *filename);

    void clear();

    virtual void reinit();
    virtual void init();


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
    void addMDx(DataVecDeriv& f, const DataVecDeriv& dx, double factor, const core::MechanicalParams* mparams);

    void accFromF(DataVecDeriv& a, const DataVecDeriv& f, const core::MechanicalParams* mparams);

    void addForce(DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v, const core::MechanicalParams* mparams);

    double getKineticEnergy(const DataVecDeriv& v, const core::MechanicalParams* mparams) const;  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const DataVecCoord& x, const core::MechanicalParams* mparams) const;   ///< Mgx potential in a uniform gravity field, null at origin

    void addGravityToV(DataVecDeriv& d_v, const core::MechanicalParams* mparams);

    /// Add Mass contribution to global Matrix assembling
    // void addMToMatrix(defaulttype::BaseMatrix * mat, double mFact, unsigned int &offset);
    void addMToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, const core::MechanicalParams *mparams);

    double getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

    bool isDiagonal() {return true;};

    void draw();

    bool addBBox(double* minBBox, double* maxBBox);

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const DiagonalMass<DataTypes, TMassType>* = NULL)
    {
        return DataTypes::Name();
    }
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_MASS_DIAGONALMASS_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec3dTypes,double>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec2dTypes,double>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec1dTypes,double>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Rigid3dTypes,defaulttype::Rigid3dMass>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Rigid2dTypes,defaulttype::Rigid2dMass>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec3fTypes,float>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec2fTypes,float>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec1fTypes,float>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Rigid3fTypes,defaulttype::Rigid3fMass>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Rigid2fTypes,defaulttype::Rigid2fMass>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
