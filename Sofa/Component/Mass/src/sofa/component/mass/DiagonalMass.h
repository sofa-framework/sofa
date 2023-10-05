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

#include <sofa/component/mass/config.h>

#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/component/mass/VecMassType.h>
#include <sofa/component/mass/RigidMassType.h>

#include <type_traits>

namespace sofa::component::mass
{

template<class DataTypes, class TMassType, class GeometricalTypes>
class DiagonalMassInternalData
{
public :
    typedef typename DataTypes::Real Real;
    typedef type::vector<TMassType> MassVector;
    typedef sofa::core::topology::PointData<MassVector> VecMass;

    // In case of non 3D template
    typedef sofa::type::Vec<3,Real> Vec3;
};

/**
* @class    DiagonalMass
* @brief    This component computes the integral of this mass density over the volume of the object geometry but it supposes that the Mass matrix is diagonal.
* @remark   Similar to MeshMatrixMass but it does not simplify the Mass Matrix as diagonal.
* @remark   https://www.sofa-framework.org/community/doc/components/masses/diagonalmass/
* @tparam   DataTypes type of the state associated with this mass
* @tparam   GeometricalTypes type of the geometry, i.e type of the state associated with the topology (if the topology and the mass relates to the same state, this will be the same as DataTypes)
*/
template <class DataTypes, class GeometricalTypes = DataTypes>
class DiagonalMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DiagonalMass,DataTypes, GeometricalTypes), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    using TMassType = typename sofa::component::mass::MassType<DataTypes>::type;

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef TMassType MassType;

    typedef typename DiagonalMassInternalData<DataTypes,TMassType,GeometricalTypes>::VecMass VecMass;
    typedef typename DiagonalMassInternalData<DataTypes,TMassType,GeometricalTypes>::MassVector MassVector;

    VecMass d_vertexMass; ///< values of the particles masses

    typedef core::topology::BaseMeshTopology::Point Point;
    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef core::topology::BaseMeshTopology::Quad Quad;
    typedef core::topology::BaseMeshTopology::QuadID QuadID;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::TriangleID TriangleID;
    typedef core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
    typedef core::topology::BaseMeshTopology::TetrahedronID TetrahedronID;
    typedef core::topology::BaseMeshTopology::Hexahedron Hexahedron;
    typedef core::topology::BaseMeshTopology::HexahedronID HexahedronID;

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

    /// value defining the initialization process of the mass (0 : totalMass, 1 : massDensity, 2 : vertexMass)
    int m_initializationProcess;

    /// Link to be set to the topology container in the component graph. 
    SingleLink<DiagonalMass<DataTypes, GeometricalTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
    /// Link to be set to the MechanicalObject associated with the geometry
    SingleLink<DiagonalMass<DataTypes, GeometricalTypes>, sofa::core::behavior::MechanicalState<GeometricalTypes>, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_geometryState;

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
    sofa::geometry::ElementType m_massTopologyType;

protected:
    DiagonalMass();

    ~DiagonalMass() override = default;
public:

    bool load(const char *filename);

    void clear();

    void reinit() override;
    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* ) override;

    void doUpdateInternal() override;

    sofa::geometry::ElementType getMassTopologyType() const
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

    /// Compute the vertexMass using input density and return the corresponding full mass.
    Real computeVertexMass(Real density);

    /** Method to initialize @sa MassVector when a new Point is created.
    * Will be set as creation callback in the PointData @sa d_vertexMass
    */
    void applyPointCreation(PointID pointIndex, MassType& m, const Point&,
        const sofa::type::vector< PointID >&,
        const sofa::type::vector< SReal >&);

    /** Method to update @sa d_vertexMass when a Point is removed.
    * Will be set as destruction callback in the PointData @sa d_vertexMass
    */
    void applyPointDestruction(Index id, MassType& VertexMass);


    /** Method to update @sa d_vertexMass when a new Edge is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when EDGESADDED event is fired.    
    */
    void applyEdgeCreation(const sofa::type::vector< EdgeID >& /*indices*/,
        const sofa::type::vector< Edge >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< EdgeID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/);

    /** Method to update @sa d_vertexMass when a Edge is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when EDGESREMOVED event is fired.
    */
    void applyEdgeDestruction(const sofa::type::vector<EdgeID>& /*indices*/);


    /** Method to update @sa d_vertexMass when a new Triangle is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when TRIANGLESADDED event is fired.
    */
    template <typename T = GeometricalTypes, typename std::enable_if_t<T::spatial_dimensions >= 2, int > = 0 >
    void applyTriangleCreation(const sofa::type::vector< TriangleID >& /*indices*/,
        const sofa::type::vector< Triangle >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< TriangleID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/);

    /** Method to update @sa d_vertexMass when a Triangle is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when TRIANGLESREMOVED event is fired.
    */
    template <typename T = GeometricalTypes, typename std::enable_if_t<T::spatial_dimensions >= 2, int > = 0 >
    void applyTriangleDestruction(const sofa::type::vector<TriangleID>& /*indices*/);


    /** Method to update @sa d_vertexMass when a new Quad is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when QUADSADDED event is fired.
    */
    template <typename T = GeometricalTypes, typename std::enable_if_t<T::spatial_dimensions >= 2, int > = 0 >
    void applyQuadCreation(const sofa::type::vector< QuadID >& /*indices*/,
        const sofa::type::vector< Quad >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< QuadID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/);

    /** Method to update @sa d_vertexMass when a Quad is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when QUADSREMOVED event is fired.
    */
    template <typename T = GeometricalTypes, typename std::enable_if_t<T::spatial_dimensions >= 2, int > = 0 >
    void applyQuadDestruction(const sofa::type::vector<QuadID>& /*indices*/);
    

    /** Method to update @sa d_vertexMass when a new Tetrahedron is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when TETRAHEDRAADDED event is fired.
    */
    template <typename T = GeometricalTypes, typename std::enable_if_t<T::spatial_dimensions >= 3, int > = 0 >
    void applyTetrahedronCreation(const sofa::type::vector< TetrahedronID >& /*indices*/,
        const sofa::type::vector< Tetrahedron >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< TetrahedronID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/);

    /** Method to update @sa d_vertexMass when a Tetrahedron is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when TETRAHEDRAREMOVED event is fired.
    */
    template <typename T = GeometricalTypes, typename std::enable_if_t<T::spatial_dimensions >= 3, int > = 0 >
    void applyTetrahedronDestruction(const sofa::type::vector<TetrahedronID>& /*indices*/);


    /** Method to update @sa d_vertexMass when a new Hexahedron is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when HEXAHEDRAADDED event is fired.
    */
    template <typename T = GeometricalTypes, typename std::enable_if_t<T::spatial_dimensions >= 3, int > = 0 >
    void applyHexahedronCreation(const sofa::type::vector< HexahedronID >& /*indices*/,
        const sofa::type::vector< Hexahedron >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< HexahedronID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/);
    
    /** Method to update @sa d_vertexMass when a Hexahedron is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when HEXAHEDRAREMOVED event is fired.
    */
    template <typename T = GeometricalTypes, typename std::enable_if_t<T::spatial_dimensions >= 3, int > = 0 >
    void applyHexahedronDestruction(const sofa::type::vector<HexahedronID>& /*indices*/);

public:

    SReal getTotalMass() const { return d_totalMass.getValue(); }
    std::size_t getMassCount() { return d_vertexMass.getValue().size(); }

    /// Print key mass informations (totalMass, vertexMass and massDensity)
    void printMass();

    /// @name Read and write access functions in mass information
    /// @{
    virtual const Real &getMassDensity();
    virtual const Real &getTotalMass();

    virtual void setVertexMass(sofa::type::vector< Real > vertexMass);
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

    type::Vec6 getMomentum(const core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const override;  ///< (Mv,cross(x,Mv)+Iw) override

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* /* matrix */) override {}
    void buildDampingMatrix(core::behavior::DampingMatrix* /* matrices */) override {}

    SReal getElementMass(sofa::Index index) const override;
    void getElementMass(sofa::Index, linearalgebra::BaseMatrix *m) const override;

    bool isDiagonal() const override {return true;}

    void draw(const core::visual::VisualParams* vparams) override;

    //Temporary function to warn the user when old attribute names are used and if it tried to specify the masstype (deprecated)
    void parse(sofa::core::objectmodel::BaseObjectDescription* arg) override
    {
        Inherited::parse(arg);
        parseMassTemplate<MassType>(arg, this);
        if (arg->getAttribute("mass"))
        {
            msg_warning() << "input data 'mass' changed for 'vertexMass', please update your scene (see PR#637)";
        }
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
    type::Vec6 getMomentumRigid3Impl ( const core::MechanicalParams*,
                                                 const DataVecCoord& vx,
                                                 const DataVecDeriv& vv ) const ;

    template <class T>
    type::Vec6 getMomentumVec3Impl ( const core::MechanicalParams*,
                                               const DataVecCoord& vx,
                                               const DataVecDeriv& vv ) const ;
};


// Specialization for rigids
template <>
SReal DiagonalMass<defaulttype::Rigid3Types>::getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x) const;
template <>
SReal DiagonalMass<defaulttype::Rigid2Types>::getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x) const;
template <>
void DiagonalMass<defaulttype::Rigid3Types>::draw(const core::visual::VisualParams* vparams);
template <>
void DiagonalMass<defaulttype::Rigid3Types>::reinit();
template <>
void DiagonalMass<defaulttype::Rigid2Types>::reinit();
template <>
void DiagonalMass<defaulttype::Rigid3Types>::init();
template <>
void DiagonalMass<defaulttype::Rigid2Types>::init();
template <>
void DiagonalMass<defaulttype::Rigid2Types>::draw(const core::visual::VisualParams* vparams);
template <>
type::Vec6 DiagonalMass<defaulttype::Vec3Types>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const;
template <>
type::Vec6 DiagonalMass<defaulttype::Rigid3Types>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const;



#if !defined(SOFA_COMPONENT_MASS_DIAGONALMASS_CPP)
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec2Types, defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec1Types, defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec1Types, defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Rigid2Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Rigid2Types, defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::mass
