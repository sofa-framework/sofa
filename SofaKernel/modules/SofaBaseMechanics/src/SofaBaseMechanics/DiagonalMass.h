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
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/type/vector.h>
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
    typedef type::vector<TMassType> MassVector;
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

    /// Compute the vertexMass using input density and return the corresponding full mass.
    Real computeVertexMass(Real density);

    /** Method to initialize @sa MassVector when a new Point is created.
    * Will be set as creation callback in the PointData @sa d_vertexMass
    */
    void applyPointCreation(PointID pointIndex, MassType& m, const Point&,
        const sofa::type::vector< PointID >&,
        const sofa::type::vector< double >&);

    /** Method to update @sa d_vertexMass when a Point is removed.
    * Will be set as destruction callback in the PointData @sa d_vertexMass
    */
    void applyPointDestruction(const sofa::type::vector<PointID>& /*indices*/);


    /** Method to update @sa d_vertexMass when a new Edge is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when EDGESADDED event is fired.    
    */
    void applyEdgeCreation(const sofa::type::vector< EdgeID >& /*indices*/,
        const sofa::type::vector< Edge >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< EdgeID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< double > >& /*coefs*/);

    /** Method to update @sa d_vertexMass when a new Edge is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when EDGESREMOVED event is fired.
    */
    void applyEdgeDestruction(const sofa::type::vector<EdgeID>& /*indices*/);


    /** Method to update @sa d_vertexMass when a new Triangle is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when TRIANGLESADDED event is fired.
    */
    void applyTriangleCreation(const sofa::type::vector< TriangleID >& /*indices*/,
        const sofa::type::vector< Triangle >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< TriangleID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< double > >& /*coefs*/);

    /** Method to update @sa d_vertexMass when a new Triangle is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when TRIANGLESREMOVED event is fired.
    */
    void applyTriangleDestruction(const sofa::type::vector<TriangleID>& /*indices*/);


    /** Method to update @sa d_vertexMass when a new Quad is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when QUADSADDED event is fired.
    */
    void applyQuadCreation(const sofa::type::vector< QuadID >& /*indices*/,
        const sofa::type::vector< Quad >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< QuadID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< double > >& /*coefs*/);

    /** Method to update @sa d_vertexMass when a new Quad is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when QUADSREMOVED event is fired.
    */
    void applyQuadDestruction(const sofa::type::vector<QuadID>& /*indices*/);
    

    /** Method to update @sa d_vertexMass when a new Tetrahedron is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when TETRAHEDRAADDED event is fired.
    */
    void applyTetrahedronCreation(const sofa::type::vector< TetrahedronID >& /*indices*/,
        const sofa::type::vector< Tetrahedron >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< TetrahedronID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< double > >& /*coefs*/);

    /** Method to update @sa d_vertexMass when a new Tetrahedron is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when TETRAHEDRAREMOVED event is fired.
    */
    void applyTetrahedronDestruction(const sofa::type::vector<TetrahedronID>& /*indices*/);


    /** Method to update @sa d_vertexMass when a new Hexahedron is created.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when HEXAHEDRAADDED event is fired.
    */
    void applyHexahedronCreation(const sofa::type::vector< HexahedronID >& /*indices*/,
        const sofa::type::vector< Hexahedron >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< HexahedronID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< double > >& /*coefs*/);
    
    /** Method to update @sa d_vertexMass when a new Edge is removed.
    * Will be set as callback in the PointData @sa d_vertexMass to update the mass vector when HEXAHEDRAREMOVED event is fired.
    */
    void applyHexahedronDestruction(const sofa::type::vector<HexahedronID>& /*indices*/);

public:

    SReal getTotalMass() const { return d_totalMass.getValue(); }
    std::size_t getMassCount() { return d_vertexMass.getValue().size(); }

    /// Print key mass informations (totalMass, vertexMass and massDensity)
    void printMass();

    /// Compute the mass from input values
    SOFA_ATTRIBUTE_DEPRECATED("v21.06", "v21.12", "ComputeMass should not be called from outside. Changing one of the Data: density, totalMass or vertexMass will recompute the mass.")
    void computeMass();


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

    type::Vector6 getMomentum(const core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const override;  ///< (Mv,cross(x,Mv)+Iw) override

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;


    SReal getElementMass(sofa::Index index) const override;
    void getElementMass(sofa::Index, defaulttype::BaseMatrix *m) const override;

    bool isDiagonal() const override {return true;}

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
