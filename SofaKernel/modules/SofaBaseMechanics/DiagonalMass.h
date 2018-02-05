/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MASS_DIAGONALMASS_H
#define SOFA_COMPONENT_MASS_DIAGONALMASS_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>

#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>

#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace mass
{

template<class DataTypes, class TMassType>
class DiagonalMassInternalData
{
public :
    typedef typename DataTypes::Real Real;
    typedef helper::vector<TMassType> MassVector;
    typedef sofa::component::topology::PointData<MassVector> VecMass;

    // In case of non 3D template
    typedef sofa::defaulttype::Vec<3,Real> Vec3;
    typedef sofa::defaulttype::StdVectorTypes< Vec3, Vec3, Real > GeometricalTypes ; /// assumes the geometry object type is 3D
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
    typedef typename DiagonalMassInternalData<DataTypes,TMassType>::GeometricalTypes GeometricalTypes;

    VecMass d_mass;

    typedef core::topology::BaseMeshTopology::Point Point;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Quad Quad;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
    typedef core::topology::BaseMeshTopology::Hexahedron Hexahedron;

    class DMassPointHandler : public topology::TopologyDataHandler<Point,MassVector>
    {
    public:
        typedef typename DiagonalMass<DataTypes,TMassType>::MassVector MassVector;
        DMassPointHandler(DiagonalMass<DataTypes,TMassType>* _dm, sofa::component::topology::PointData<MassVector>* _data) : topology::TopologyDataHandler<Point,MassVector>(_data), dm(_dm) {}

        void applyCreateFunction(unsigned int pointIndex, TMassType& m, const Point&, const sofa::helper::vector< unsigned int > &,
                                 const sofa::helper::vector< double > &);

        using topology::TopologyDataHandler<Point,MassVector>::ApplyTopologyChange;

        ///////////////////////// Functions on Edges //////////////////////////////////////
        /// Apply adding edges elements.
        void applyEdgeCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                               const sofa::helper::vector< Edge >& /*elems*/,
                               const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                               const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);
        /// Apply removing edges elements.
        void applyEdgeDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add edges elements.
        virtual void ApplyTopologyChange(const core::topology::EdgesAdded* /*event*/);
        /// Callback to remove edges elements.
        virtual void ApplyTopologyChange(const core::topology::EdgesRemoved* /*event*/);

        ///////////////////////// Functions on Triangles //////////////////////////////////////
        /// Apply adding triangles elements.
        void applyTriangleCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                                   const sofa::helper::vector< Triangle >& /*elems*/,
                                   const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                                   const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);
        /// Apply removing triangles elements.
        void applyTriangleDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add triangles elements.
        virtual void ApplyTopologyChange(const core::topology::TrianglesAdded* /*event*/);
        /// Callback to remove triangles elements.
        virtual void ApplyTopologyChange(const core::topology::TrianglesRemoved* /*event*/);

        ///////////////////////// Functions on Tetrahedron //////////////////////////////////////
        /// Apply adding tetrahedron elements.
        void applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                                      const sofa::helper::vector< Tetrahedron >& /*elems*/,
                                      const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                                      const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);
        /// Apply removing tetrahedron elements.
        void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add tetrahedron elements.
        virtual void ApplyTopologyChange(const core::topology::TetrahedraAdded* /*event*/);
        /// Callback to remove tetrahedron elements.
        virtual void ApplyTopologyChange(const core::topology::TetrahedraRemoved* /*event*/);

        ///////////////////////// Functions on Hexahedron //////////////////////////////////////
        /// Apply adding hexahedron elements.
        void applyHexahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                                     const sofa::helper::vector< Hexahedron >& /*elems*/,
                                     const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                                     const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);
        /// Apply removing hexahedron elements.
        void applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);
        /// Callback to add hexahedron elements.
        virtual void ApplyTopologyChange(const core::topology::HexahedraAdded* /*event*/);
        /// Callback to remove hexahedron elements.
        virtual void ApplyTopologyChange(const core::topology::HexahedraRemoved* /*event*/);

    protected:
        DiagonalMass<DataTypes,TMassType>* dm;
    };
    DMassPointHandler* m_pointHandler;
    /// the mass density used to compute the mass from a mesh topology and geometry
    Data< Real > d_massDensity;

    /// if true, the mass of every element is computed based on the rest position rather than the position
    Data< bool > d_computeMassOnRest;

    /// total mass of the object
    Data< Real > d_totalMass;

    /// to display the center of gravity of the system
    Data< bool > d_showCenterOfGravity;
    Data< float > d_showAxisSize;
    core::objectmodel::DataFileName d_fileMass;

protected:
    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using core::behavior::ForceField<DataTypes>::mstate ;
    using core::objectmodel::BaseObject::getContext;
    using core::objectmodel::BaseObject::m_componentstate ;
    ////////////////////////////////////////////////////////////////////////////


    class Loader;
    /// The type of topology to build the mass from the topology
    TopologyType m_topologyType;


public:
    sofa::core::topology::BaseMeshTopology* _topology;

    sofa::component::topology::EdgeSetGeometryAlgorithms<GeometricalTypes>* edgeGeo;
    sofa::component::topology::TriangleSetGeometryAlgorithms<GeometricalTypes>* triangleGeo;
    sofa::component::topology::QuadSetGeometryAlgorithms<GeometricalTypes>* quadGeo;
    sofa::component::topology::TetrahedronSetGeometryAlgorithms<GeometricalTypes>* tetraGeo;
    sofa::component::topology::HexahedronSetGeometryAlgorithms<GeometricalTypes>* hexaGeo;
protected:
    DiagonalMass();

    ~DiagonalMass() override;
public:

    bool load(const char *filename);

    void clear();

    virtual void reinit() override;
    virtual void init() override;


    TopologyType getMassTopologyType() const
    {
        return m_topologyType;
    }

    Real getMassDensity() const
    {
        return d_massDensity.getValue();
    }

protected:
    void initTopologyHandlers();

public:

    void setMassDensity(Real m)
    {
        d_massDensity.setValue(m);
    }

    SReal getTotalMass() const { return d_totalMass.getValue(); }
    int getMassCount() { return d_mass.getValue().size(); }


    void addMass(const MassType& mass);

    void resize(int vsize);

    // -- Mass interface
    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;

    void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f) override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    SReal getKineticEnergy(const core::MechanicalParams* mparams, const DataVecDeriv& v) const override;  ///< vMv/2 using dof->getV() override

    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;   ///< Mgx potential in a uniform gravity field, null at origin

    defaulttype::Vector6 getMomentum(const core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const override;  ///< (Mv,cross(x,Mv)+Iw) override

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;


    SReal getElementMass(unsigned int index) const override;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const override;

    bool isDiagonal() override {return true;}

    void draw(const core::visual::VisualParams* vparams) override;


    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const DiagonalMass<DataTypes, TMassType>* = NULL)
    {
        return DataTypes::Name();
    }

private:
    template <class T>
    SReal getPotentialEnergyRigidImpl( const core::MechanicalParams* mparams,
                                       const DataVecCoord& x) const ;

    template <class T>
    void drawRigid3dImpl(const core::visual::VisualParams* vparams) ;

    template <class T>
    void drawRigid2dImpl(const core::visual::VisualParams* vparams) ;

    template <class T>
    void initRigidImpl() ;

    template <class T>
    defaulttype::Vector6 getMomentumRigid3Impl ( const core::MechanicalParams*,
                                                 const DataVecCoord& vx,
                                                 const DataVecDeriv& vv ) const ;

    template <class T>
    defaulttype::Vector6 getMomentumVec3Impl ( const core::MechanicalParams*,
                                               const DataVecCoord& vx,
                                               const DataVecDeriv& vv ) const ;
};


// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
SReal DiagonalMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x) const;
template <>
SReal DiagonalMass<defaulttype::Rigid2dTypes, defaulttype::Rigid2dMass>::getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x) const;
template <>
void DiagonalMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::draw(const core::visual::VisualParams* vparams);
template <>
void DiagonalMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::reinit();
template <>
void DiagonalMass<defaulttype::Rigid2dTypes, defaulttype::Rigid2dMass>::reinit();
template <>
void DiagonalMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::init();
template <>
void DiagonalMass<defaulttype::Rigid2dTypes, defaulttype::Rigid2dMass>::init();
template <>
void DiagonalMass<defaulttype::Rigid2dTypes, defaulttype::Rigid2dMass>::draw(const core::visual::VisualParams* vparams);
template <>
defaulttype::Vector6 DiagonalMass<defaulttype::Vec3dTypes, double>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
template <>
defaulttype::Vector6 DiagonalMass<defaulttype::Rigid3dTypes,defaulttype::Rigid3dMass>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
#endif

#ifndef SOFA_DOUBLE
template <>
SReal DiagonalMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x) const;
template <>
SReal DiagonalMass<defaulttype::Rigid2fTypes, defaulttype::Rigid2fMass>::getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x) const;
template <>
void DiagonalMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::draw(const core::visual::VisualParams* vparams);
template <>
void DiagonalMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::reinit();
template <>
void DiagonalMass<defaulttype::Rigid2fTypes, defaulttype::Rigid2fMass>::reinit();
template <>
void DiagonalMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::init();
template <>
void DiagonalMass<defaulttype::Rigid2fTypes, defaulttype::Rigid2fMass>::init();
template <>
void DiagonalMass<defaulttype::Rigid2fTypes, defaulttype::Rigid2fMass>::draw(const core::visual::VisualParams* vparams);
template <>
defaulttype::Vector6 DiagonalMass<defaulttype::Vec3fTypes, float>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
template <>
defaulttype::Vector6 DiagonalMass<defaulttype::Rigid3fTypes,defaulttype::Rigid3fMass>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
#endif


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MASS_DIAGONALMASS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Vec3dTypes,double>;
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Vec2dTypes,double>;
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Vec1dTypes,double>;
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Rigid3dTypes,defaulttype::Rigid3dMass>;
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Rigid2dTypes,defaulttype::Rigid2dMass>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Vec3fTypes,float>;
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Vec2fTypes,float>;
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Vec1fTypes,float>;
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Rigid3fTypes,defaulttype::Rigid3fMass>;
extern template class SOFA_BASE_MECHANICS_API DiagonalMass<defaulttype::Rigid2fTypes,defaulttype::Rigid2fMass>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
