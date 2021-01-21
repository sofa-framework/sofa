/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <SofaBaseMechanics/config.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/objectmodel/Event.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Vec.h>

#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>

#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa::component::mass
{

template<class DataTypes, class TMassType>
class DiagonalMassInternalData
{
public :
    typedef typename DataTypes::Real Real;
    typedef helper::vector<TMassType> MassVector;
    typedef sofa::component::topology::PointData<MassVector> VecMass;

    // In case of non 3D template
    typedef sofa::type::Vec<3,Real> Vec3;
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

    typedef typename DiagonalMassInternalData<DataTypes,TMassType>::VecMass VecMass;
    typedef typename DiagonalMassInternalData<DataTypes,TMassType>::MassVector MassVector;
    typedef typename DiagonalMassInternalData<DataTypes,TMassType>::GeometricalTypes GeometricalTypes;

    VecMass d_vertexMass; ///< values of the particles masses

    typedef core::topology::BaseMeshTopology::Point Point;
    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef core::topology::BaseMeshTopology::Quad Quad;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::TriangleID TriangleID;
    typedef core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
    typedef core::topology::BaseMeshTopology::TetrahedronID TetrahedronID;
    typedef core::topology::BaseMeshTopology::Hexahedron Hexahedron;
    typedef core::topology::BaseMeshTopology::HexahedronID HexahedronID;

    class DMassPointHandler : public topology::TopologyDataHandler<Point,MassVector>
    {
    public:
        typedef typename DiagonalMass<DataTypes,TMassType>::MassVector MassVector;
        DMassPointHandler(DiagonalMass<DataTypes,TMassType>* _dm, sofa::component::topology::PointData<MassVector>* _data)
            : topology::TopologyDataHandler<Point,MassVector>(_data), dm(_dm)
        {}

        void applyCreateFunction(PointID pointIndex, TMassType& m, const Point&, const sofa::helper::vector< PointID > &,
                                 const sofa::helper::vector< double > &);

        using topology::TopologyDataHandler<Point,MassVector>::ApplyTopologyChange;

        ///////////////////////// Functions on Points //////////////////////////////////////
        /// Apply removing points.
        void applyPointDestruction(const sofa::helper::vector<PointID> & /*indices*/);
        /// Callback to remove points.
        virtual void ApplyTopologyChange(const core::topology::PointsRemoved* /*event*/);

        ///////////////////////// Functions on Edges //////////////////////////////////////
        /// Apply adding edges elements.
        void applyEdgeCreation(const sofa::helper::vector< EdgeID >& /*indices*/,
                               const sofa::helper::vector< Edge >& /*elems*/,
                               const sofa::helper::vector< sofa::helper::vector< EdgeID > >& /*ancestors*/,
                               const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);
        /// Apply removing edges elements.
        void applyEdgeDestruction(const sofa::helper::vector<EdgeID> & /*indices*/);

        /// Callback to add edges elements.
        virtual void ApplyTopologyChange(const core::topology::EdgesAdded* /*event*/);
        /// Callback to remove edges elements.
        virtual void ApplyTopologyChange(const core::topology::EdgesRemoved* /*event*/);

        ///////////////////////// Functions on Triangles //////////////////////////////////////
        /// Apply adding triangles elements.
        void applyTriangleCreation(const sofa::helper::vector< TriangleID >& /*indices*/,
                                   const sofa::helper::vector< Triangle >& /*elems*/,
                                   const sofa::helper::vector< sofa::helper::vector< TriangleID > >& /*ancestors*/,
                                   const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);
        /// Apply removing triangles elements.
        void applyTriangleDestruction(const sofa::helper::vector<TriangleID> & /*indices*/);

        /// Callback to add triangles elements.
        virtual void ApplyTopologyChange(const core::topology::TrianglesAdded* /*event*/);
        /// Callback to remove triangles elements.
        virtual void ApplyTopologyChange(const core::topology::TrianglesRemoved* /*event*/);

        ///////////////////////// Functions on Tetrahedron //////////////////////////////////////
        /// Apply adding tetrahedron elements.
        void applyTetrahedronCreation(const sofa::helper::vector< TetrahedronID >& /*indices*/,
                                      const sofa::helper::vector< Tetrahedron >& /*elems*/,
                                      const sofa::helper::vector< sofa::helper::vector< TetrahedronID > >& /*ancestors*/,
                                      const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);
        /// Apply removing tetrahedron elements.
        void applyTetrahedronDestruction(const sofa::helper::vector<TetrahedronID> & /*indices*/);

        /// Callback to add tetrahedron elements.
        virtual void ApplyTopologyChange(const core::topology::TetrahedraAdded* /*event*/);
        /// Callback to remove tetrahedron elements.
        virtual void ApplyTopologyChange(const core::topology::TetrahedraRemoved* /*event*/);

        ///////////////////////// Functions on Hexahedron //////////////////////////////////////
        /// Apply adding hexahedron elements.
        void applyHexahedronCreation(const sofa::helper::vector< HexahedronID >& /*indices*/,
                                     const sofa::helper::vector< Hexahedron >& /*elems*/,
                                     const sofa::helper::vector< sofa::helper::vector< HexahedronID > >& /*ancestors*/,
                                     const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);
        /// Apply removing hexahedron elements.
        void applyHexahedronDestruction(const sofa::helper::vector<HexahedronID> & /*indices*/);
        /// Callback to add hexahedron elements.
        virtual void ApplyTopologyChange(const core::topology::HexahedraAdded* /*event*/);
        /// Callback to remove hexahedron elements.
        virtual void ApplyTopologyChange(const core::topology::HexahedraRemoved* /*event*/);

    protected:
        DiagonalMass<DataTypes,TMassType>* dm;
    };
    /// the mass density used to compute the mass from a mesh topology and geometry
    Data< Real > d_massDensity;

    /// total mass of the object
    Data< Real > d_totalMass;

    /// if true, the mass of every element is computed based on the rest position rather than the position
    Data< bool > d_computeMassOnRest;

    /// to display the center of gravity of the system
    Data< bool > d_showCenterOfGravity;

    Data< float > d_showAxisSize; ///< factor length of the axis displayed (only used for rigids)
    core::objectmodel::DataFileName d_fileMass; ///< an Xsp3.0 file to specify the mass parameters

    DMassPointHandler* m_pointHandler;

    /// value defining the initialization process of the mass (0 : totalMass, 1 : massDensity, 2 : vertexMass)
    int m_initializationProcess;

    /// Link to be set to the topology container in the component graph. 
    SingleLink<DiagonalMass<DataTypes, TMassType>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using core::behavior::ForceField<DataTypes>::mstate ;
    using core::objectmodel::BaseObject::getContext;
    ////////////////////////////////////////////////////////////////////////////


    class Loader;
    /// The type of topology to build the mass from the topology
    sofa::core::topology::TopologyElementType m_massTopologyType;

    /// Pointer to the topology container. Will be set by link @sa l_topology
    sofa::core::topology::BaseMeshTopology* m_topology;

public:
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

    void reinit() override;
    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* ) override;

    void doUpdateInternal() override;

    sofa::core::topology::TopologyElementType getMassTopologyType() const
    {
        return m_massTopologyType;
    }

    Real getMassDensity() const
    {
        return d_massDensity.getValue();
    }

protected:
    bool checkTopology();
    void initTopologyHandlers();
    void massInitialization();

public:

    SReal getTotalMass() const { return d_totalMass.getValue(); }
    std::size_t getMassCount() { return d_vertexMass.getValue().size(); }

    /// Print key mass informations (totalMass, vertexMass and massDensity)
    void printMass();

    /// Compute the mass from input values
    void computeMass();


    /// @name Read and write access functions in mass information
    /// @{
    virtual const Real &getMassDensity();
    virtual const Real &getTotalMass();

    virtual void setVertexMass(sofa::helper::vector< Real > vertexMass);
    virtual void setMassDensity(Real massDensityValue);
    virtual void setTotalMass(Real totalMass);
    /// @}


    /// @name Check and standard initialization functions from mass information
    /// @{
    virtual bool checkVertexMass();
    virtual void initFromVertexMass();

    virtual bool checkMassDensity();
    virtual void initFromMassDensity();

    virtual bool checkTotalMass();
    virtual void checkTotalMassInit();
    virtual void initFromTotalMass();
    /// @}


    void addMass(const MassType& mass);

    void resize(int vsize);

    // -- Mass interface
    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;

    void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f) override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    SReal getKineticEnergy(const core::MechanicalParams* mparams, const DataVecDeriv& v) const override;  ///< vMv/2 using dof->getV() override

    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;   ///< Mgx potential in a uniform gravity field, null at origin

    type::Vector6 getMomentum(const core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const override;  ///< (Mv,cross(x,Mv)+Iw) override

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;


    SReal getElementMass(sofa::Index index) const override;
    void getElementMass(sofa::Index, defaulttype::BaseMatrix *m) const override;

    bool isDiagonal() override {return true;}

    void draw(const core::visual::VisualParams* vparams) override;

    //Temporary function to warn the user when old attribute names are used
    void parse( sofa::core::objectmodel::BaseObjectDescription* arg ) override
    {
        if (arg->getAttribute("mass"))
        {
            msg_warning() << "input data 'mass' changed for 'vertexMass', please update your scene (see PR#637)";
        }
        Inherited::parse(arg);
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
    type::Vector6 getMomentumRigid3Impl ( const core::MechanicalParams*,
                                                 const DataVecCoord& vx,
                                                 const DataVecDeriv& vv ) const ;

    template <class T>
    type::Vector6 getMomentumVec3Impl ( const core::MechanicalParams*,
                                               const DataVecCoord& vx,
                                               const DataVecDeriv& vv ) const ;
};


// Specialization for rigids
template <>
SReal DiagonalMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass>::getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x) const;
template <>
SReal DiagonalMass<defaulttype::Rigid2Types, defaulttype::Rigid2Mass>::getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x) const;
template <>
void DiagonalMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass>::draw(const core::visual::VisualParams* vparams);
template <>
void DiagonalMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass>::reinit();
template <>
void DiagonalMass<defaulttype::Rigid2Types, defaulttype::Rigid2Mass>::reinit();
template <>
void DiagonalMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass>::init();
template <>
void DiagonalMass<defaulttype::Rigid2Types, defaulttype::Rigid2Mass>::init();
template <>
void DiagonalMass<defaulttype::Rigid2Types, defaulttype::Rigid2Mass>::draw(const core::visual::VisualParams* vparams);
template <>
type::Vector6 DiagonalMass<defaulttype::Vec3Types, double>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
template <>
type::Vector6 DiagonalMass<defaulttype::Rigid3Types,defaulttype::Rigid3Mass>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const;



#if  !defined(SOFA_COMPONENT_MASS_DIAGONALMASS_CPP)
extern template class SOFA_SOFABASEMECHANICS_API DiagonalMass<defaulttype::Vec3Types,double>;
extern template class SOFA_SOFABASEMECHANICS_API DiagonalMass<defaulttype::Vec2Types,double>;
extern template class SOFA_SOFABASEMECHANICS_API DiagonalMass<defaulttype::Vec1Types,double>;
extern template class SOFA_SOFABASEMECHANICS_API DiagonalMass<defaulttype::Rigid3Types,defaulttype::Rigid3Mass>;
extern template class SOFA_SOFABASEMECHANICS_API DiagonalMass<defaulttype::Rigid2Types,defaulttype::Rigid2Mass>;

#endif

} // namespace sofa::component::mass
