/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MASS_MESHMATRIXMASS_H
#define SOFA_COMPONENT_MASS_MESHMATRIXMASS_H
#include "config.h"



#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/RigidTypes.h>
//VERY IMPORTANT FOR GRAPHS
#include <sofa/helper/map.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/DataTracker.h>

namespace sofa
{
namespace component
{
namespace topology
{
	/// forward declaration to avoid adding includes in .h
	template< class DataTypes> class EdgeSetGeometryAlgorithms;
	template< class DataTypes> class TriangleSetGeometryAlgorithms;
	template< class DataTypes> class TetrahedronSetGeometryAlgorithms;
	template< class DataTypes> class QuadSetGeometryAlgorithms;
	template< class DataTypes> class HexahedronSetGeometryAlgorithms;
}

namespace mass
{

template<class DataTypes, class TMassType>
class MeshMatrixMassInternalData
{
public:
    typedef typename DataTypes::Real Real;

    /// In case of non 3D template
    typedef defaulttype::Vec<3,Real> Vec3;
    /// assumes the geometry object type is 3D
    typedef defaulttype::StdVectorTypes< Vec3, Vec3, Real > GeometricalTypes;
};


template <class DataTypes, class TMassType>
class MeshMatrixMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(MeshMatrixMass,DataTypes,TMassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord                    VecCoord;
    typedef typename DataTypes::VecDeriv                    VecDeriv;
    typedef typename DataTypes::Coord                       Coord;
    typedef typename DataTypes::Deriv                       Deriv;
    typedef typename DataTypes::Real                        Real;
    typedef core::objectmodel::Data<VecCoord>               DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>               DataVecDeriv;
    typedef TMassType                                       MassType;
    typedef helper::vector<MassType>                        MassVector;
    typedef helper::vector<MassVector>                      MassVectorVector;

    /// Topological enum to classify encounter meshes
    typedef enum
    {
        TOPOLOGY_UNKNOWN=0,
        TOPOLOGY_EDGESET=1,
        TOPOLOGY_TRIANGLESET=2,
        TOPOLOGY_TETRAHEDRONSET=3,
        TOPOLOGY_QUADSET=4,
        TOPOLOGY_HEXAHEDRONSET=5
    } TopologyType;

    typedef typename MeshMatrixMassInternalData<DataTypes,TMassType>::GeometricalTypes GeometricalTypes;

    /// @name Data of mass information
    /// @{
    /// Mass stored on vertices
    Data< sofa::helper::vector< Real > > d_vertexMass;
    /// Mass density of the object
    Data< sofa::helper::vector< Real > > d_massDensity;
    /// Total mass of the object
    Data< Real > d_totalMass;
    /// @}


    /// Values of the particles masses stored on vertices
    topology::PointData<helper::vector<MassType> >  d_vertexMassInfo;
    /// Values of the particles masses stored on edges
    topology::EdgeData<helper::vector<MassType> >   d_edgeMassInfo;

    /// to display the center of gravity of the system
    Data< sofa::helper::vector< Real > > d_edgeMass;

    /// if true, the mass of every element is computed based on the rest position rather than the position
    Data< bool > d_computeMassOnRest;
    /// to display the center of gravity of the system
    Data< bool >         d_showCenterOfGravity;
    /// scale to change the axis size
    Data< Real >         d_showAxisSize;  ///< factor length of the axis displayed (only used for rigids)
    /// if mass lumping should be performed (only compute mass on vertices)
    Data< bool >         d_lumping;
    /// if specific mass information should be outputed
    Data< bool >         d_printMass; ///< Boolean to print the mass
    Data< std::map < std::string, sofa::helper::vector<double> > > f_graph; ///< Graph of the controlled potential


protected:

    /// The type of topology to build the mass from the topology
    TopologyType m_topologyType;
    Real m_massLumpingCoeff;

    MeshMatrixMass();
    ~MeshMatrixMass();

    bool checkTopology();
    void initTopologyHandlers();
    void massInitialization();

    /// Internal data required for Cuda computation (copy of vertex mass for deviceRead)
    MeshMatrixMassInternalData<DataTypes, MassType> data;
    friend class MeshMatrixMassInternalData<DataTypes, MassType>;

    /// Data tracker
    sofa::core::DataTracker m_dataTrackerVertex;
    sofa::core::DataTracker m_dataTrackerEdge;
    sofa::core::DataTracker m_dataTrackerDensity;
    sofa::core::DataTracker m_dataTrackerTotal;


public:

    sofa::core::topology::BaseMeshTopology* _topology;

    sofa::component::topology::EdgeSetGeometryAlgorithms<GeometricalTypes>* edgeGeo;
    sofa::component::topology::TriangleSetGeometryAlgorithms<GeometricalTypes>* triangleGeo;
    sofa::component::topology::QuadSetGeometryAlgorithms<GeometricalTypes>* quadGeo;
    sofa::component::topology::TetrahedronSetGeometryAlgorithms<GeometricalTypes>* tetraGeo;
    sofa::component::topology::HexahedronSetGeometryAlgorithms<GeometricalTypes>* hexaGeo;


    virtual void clear();

    virtual void reinit() override;
    virtual void init() override;
    virtual void handleEvent(sofa::core::objectmodel::Event */*event*/) override;
    bool update();

    TopologyType getMassTopologyType() const
    {
        return m_topologyType;
    }

    void setMassTopologyType(TopologyType t)
    {
        m_topologyType = t;
    }

    int getMassCount() {
        return d_vertexMassInfo.getValue().size();
    }

    /// Print key mass informations (totalMass, vertexMass and massDensity)
    void printMass();

    /// Compute the mass from input values
    void computeMass();


    /// @name Read and write access functions in mass information
    /// @{
    virtual const sofa::helper::vector< Real > &getVertexMass();
    virtual const sofa::helper::vector< Real > &getMassDensity();
    virtual const Real &getTotalMass();

    virtual void setVertexMass(sofa::helper::vector< Real > vertexMass);
    virtual void setMassDensity(sofa::helper::vector< Real > massDensity);
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

    bool checkEdgeMass();
    void initFromVertexAndEdgeMass();
    /// @}


    /// Copy the vertex mass scalar (in case of CudaTypes)
    void copyVertexMass();


    // -- Mass interface
    virtual void addMDx(const core::MechanicalParams*, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;

    virtual void accFromF(const core::MechanicalParams*, DataVecDeriv& a, const DataVecDeriv& f) override; // This function can't be used as it use M^-1

    virtual void addForce(const core::MechanicalParams*, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    virtual SReal getKineticEnergy(const core::MechanicalParams*, const DataVecDeriv& v) const override;  ///< vMv/2 using dof->getV() override

    virtual SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& x) const override;   ///< Mgx potential in a uniform gravity field, null at origin

    virtual defaulttype::Vector6 getMomentum(const core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const override;  ///< (Mv,cross(x,Mv)) override

    virtual void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;

    virtual bool isDiagonal() override { return false; }



    /// Add Mass contribution to global Matrix assembling
    virtual void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    virtual SReal getElementMass(unsigned int index) const override;
    virtual void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const override;

    virtual void draw(const core::visual::VisualParams* vparams) override;

    /// Answer wether mass matrix is lumped or not
    bool isLumped() { return d_lumping.getValue(); }


protected:

    class VertexMassHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Point,MassVector>
    {
    public:
        VertexMassHandler(MeshMatrixMass<DataTypes,TMassType>* _m, topology::PointData<helper::vector<TMassType> >* _data) : topology::TopologyDataHandler<core::topology::BaseMeshTopology::Point,helper::vector<TMassType> >(_data), m(_m) {}

        /// Mass initialization Creation Functions:
        /// Vertex mass coefficient matrix creation function
        void applyCreateFunction(unsigned int pointIndex, TMassType & VertexMass,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double >&);


        ///////////////////////// Functions on Triangles //////////////////////////////////////

        /// Mass coefficient Creation/Destruction functions for Triangular Mesh:
        /// Vertex coefficient of mass matrix creation function to handle creation of new triangles
        void applyTriangleCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< core::topology::BaseMeshTopology::Triangle >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Vertex coefficient of mass matrix destruction function to handle creation of new triangles
        void applyTriangleDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        using topology::TopologyDataHandler<core::topology::BaseMeshTopology::Point,MassVector>::ApplyTopologyChange;
        /// Callback to add triangles elements.
        void ApplyTopologyChange(const core::topology::TrianglesAdded* /*event*/);
        /// Callback to remove triangles elements.
        void ApplyTopologyChange(const core::topology::TrianglesRemoved* /*event*/);


        ///////////////////////// Functions on Quads //////////////////////////////////////

        /// Mass coefficient Creation/Destruction functions for Quad Mesh:
        /// Vertex coefficient of mass matrix creation function to handle creation of new quads
        void applyQuadCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< core::topology::BaseMeshTopology::Quad >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Vertex coefficient of mass matrix destruction function to handle creation of new quads
        void applyQuadDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add quads elements.
        void ApplyTopologyChange(const core::topology::QuadsAdded* /*event*/);
        /// Callback to remove quads elements.
        void ApplyTopologyChange(const core::topology::QuadsRemoved* /*event*/);


        ///////////////////////// Functions on Tetrahedron //////////////////////////////////////

        /// Mass coefficient Creation/Destruction functions for Tetrahedral Mesh:
        /// Vertex coefficient of mass matrix creation function to handle creation of new tetrahedra
        void applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< core::topology::BaseMeshTopology::Tetrahedron >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Vertex coefficient of mass matrix destruction function to handle creation of new tetrahedra
        void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add tetrahedron elements.
        void ApplyTopologyChange(const core::topology::TetrahedraAdded* /*event*/);
        /// Callback to remove tetrahedron elements.
        void ApplyTopologyChange(const core::topology::TetrahedraRemoved* /*event*/);


        ///////////////////////// Functions on Hexahedron //////////////////////////////////////

        /// Mass coefficient Creation/Destruction functions for Hexahedral Mesh:
        /// Vertex coefficient of mass matrix creation function to handle creation of new hexahedra
        void applyHexahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< core::topology::BaseMeshTopology::Hexahedron >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Vertex coefficient of mass matrix destruction function to handle creation of new hexahedra
        void applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add hexahedron elements.
        virtual void ApplyTopologyChange(const core::topology::HexahedraAdded* /*event*/);
         /// Callback to remove hexahedron elements.
        virtual void ApplyTopologyChange(const core::topology::HexahedraRemoved* /*event*/);

    protected:
        MeshMatrixMass<DataTypes,TMassType>* m;
    };
    VertexMassHandler* m_vertexMassHandler;

    class EdgeMassHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,MassVector>
    {
    public:
        EdgeMassHandler(MeshMatrixMass<DataTypes,TMassType>* _m, topology::EdgeData<helper::vector<TMassType> >* _data) : topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,helper::vector<TMassType> >(_data), m(_m) {}

        /// Edge mass coefficient matrix creation function
        void applyCreateFunction(unsigned int edgeIndex, MassType & EdgeMass,
                const core::topology::BaseMeshTopology::Edge&,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double >&);

        using topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,MassVector>::ApplyTopologyChange;

        ///////////////////////// Functions on Triangles //////////////////////////////////////

        /// Edge coefficient of mass matrix creation function to handle creation of new triangles
        void applyTriangleCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< core::topology::BaseMeshTopology::Triangle >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Edge coefficient of mass matrix destruction function to handle creation of new triangles
        void applyTriangleDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add triangles elements.
        void ApplyTopologyChange(const core::topology::TrianglesAdded* /*event*/);
        /// Callback to remove triangles elements.
        void ApplyTopologyChange(const core::topology::TrianglesRemoved* /*event*/);


        ///////////////////////// Functions on Quads //////////////////////////////////////

        /// Edge coefficient of mass matrix creation function to handle creation of new quads
        void applyQuadCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< core::topology::BaseMeshTopology::Quad >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Edge coefficient of mass matrix destruction function to handle creation of new quads
        void applyQuadDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add quads elements.
        void ApplyTopologyChange(const core::topology::QuadsAdded* /*event*/);
        /// Callback to remove quads elements.
        void ApplyTopologyChange(const core::topology::QuadsRemoved* /*event*/);


        ///////////////////////// Functions on Tetrahedron //////////////////////////////////////

        /// Edge coefficient of mass matrix creation function to handle creation of new tetrahedra
        void applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< core::topology::BaseMeshTopology::Tetrahedron >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Edge coefficient of mass matrix destruction function to handle creation of new tetrahedra
        void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add tetrahedron elements.
        void ApplyTopologyChange(const core::topology::TetrahedraAdded* /*event*/);
        /// Callback to remove tetrahedron elements.
        void ApplyTopologyChange(const core::topology::TetrahedraRemoved* /*event*/);


        ///////////////////////// Functions on Hexahedron //////////////////////////////////////

        /// Edge coefficient of mass matrix creation function to handle creation of new hexahedra
        void applyHexahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< core::topology::BaseMeshTopology::Hexahedron >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Edge coefficient of mass matrix destruction function to handle creation of new hexahedra
        void applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Callback to add hexahedron elements.
        void ApplyTopologyChange(const core::topology::HexahedraAdded* /*event*/);
         /// Callback to remove hexahedron elements.
        void ApplyTopologyChange(const core::topology::HexahedraRemoved* /*event*/);

    protected:
        MeshMatrixMass<DataTypes,TMassType>* m;
    };

    EdgeMassHandler* m_edgeMassHandler;
};

#if  !defined(SOFA_COMPONENT_MASS_MESHMATRIXMASS_CPP)
extern template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<defaulttype::Vec3Types,defaulttype::Vec3Types::Real>;
extern template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<defaulttype::Vec2Types,defaulttype::Vec2Types::Real>;
extern template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<defaulttype::Vec1Types,defaulttype::Vec1Types::Real>;

#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
