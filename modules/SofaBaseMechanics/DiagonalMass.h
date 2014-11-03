/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

    VecMass f_mass;

    typedef sofa::component::topology::Point Point;
    typedef sofa::component::topology::Edge Edge;
    typedef sofa::component::topology::Triangle Triangle;
    typedef sofa::component::topology::Tetrahedron Tetrahedron;
    typedef sofa::component::topology::Hexahedron Hexahedron;

    class DMassPointHandler : public topology::TopologyDataHandler<Point,MassVector>
    {
    public:
        typedef typename DiagonalMass<DataTypes,TMassType>::MassVector MassVector;
        DMassPointHandler(DiagonalMass<DataTypes,TMassType>* _dm, sofa::component::topology::PointData<MassVector>* _data) : topology::TopologyDataHandler<Point,MassVector>(_data), dm(_dm) {}

        void applyCreateFunction(unsigned int pointIndex, TMassType& m, const Point&, const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);

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
    DMassPointHandler* pointHandler;
    /// the mass density used to compute the mass from a mesh topology and geometry
    Data< Real > m_massDensity;

    /// if true, the mass of every element is computed based on the rest position rather than the position
    Data< bool > m_computeMassOnRest;

    /// total mass of the object
    Data< Real > m_totalMass;

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
protected:
    DiagonalMass();

    ~DiagonalMass();
public:
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

protected:
    void initTopologyHandlers();

public:

    void setMassDensity(Real m)
    {
        m_massDensity.setValue(m);
    }


    void addMass(const MassType& mass);

    void resize(int vsize);

    // -- Mass interface
    void addMDx(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, double factor);

    void accFromF(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f);

    void addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    double getKineticEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecDeriv& v) const;  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;   ///< Mgx potential in a uniform gravity field, null at origin

    defaulttype::Vec6d getMomentum(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x, const DataVecDeriv& v) const;  ///< (Mv,cross(x,Mv)+Iw)

    void addGravityToV(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_v);

    /// Add Mass contribution to global Matrix assembling
    // void addMToMatrix(defaulttype::BaseMatrix * mat, double mFact, unsigned int &offset);
    void addMToMatrix(const core::MechanicalParams *mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    double getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

    bool isDiagonal() {return true;}

    void draw(const core::visual::VisualParams* vparams);


    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const DiagonalMass<DataTypes, TMassType>* = NULL)
    {
        return DataTypes::Name();
    }
};


// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
double DiagonalMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::getPotentialEnergy( const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;
template <>
double DiagonalMass<defaulttype::Rigid2dTypes, defaulttype::Rigid2dMass>::getPotentialEnergy( const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;
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
defaulttype::Vec6d DiagonalMass<defaulttype::Vec3dTypes, double>::getMomentum ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
template <>
defaulttype::Vec6d DiagonalMass<defaulttype::Rigid3dTypes,defaulttype::Rigid3dMass>::getMomentum ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
#endif

#ifndef SOFA_DOUBLE
template <>
double DiagonalMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::getPotentialEnergy( const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;
template <>
double DiagonalMass<defaulttype::Rigid2fTypes, defaulttype::Rigid2fMass>::getPotentialEnergy( const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;
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
defaulttype::Vec6d DiagonalMass<defaulttype::Vec3fTypes, float>::getMomentum ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
template <>
defaulttype::Vec6d DiagonalMass<defaulttype::Rigid3fTypes,defaulttype::Rigid3fMass>::getMomentum ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
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
