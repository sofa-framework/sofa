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
#ifndef SOFA_COMPONENT_MASS_MESHMATRIXMASS_H
#define SOFA_COMPONENT_MASS_MESHMATRIXMASS_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/RigidTypes.h>
//VERY IMPORTANT FOR GRAPHS
#include <sofa/helper/map.h>

#include <sofa/core/topology/BaseMeshTopology.h>

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
};



// template<class Vec> void readVec1(Vec& vec, const char* str);
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

    // In case of non 3D template
    typedef defaulttype::Vec<3,Real> Vec3;
    /// assumes the geometry object type is 3D
    typedef defaulttype::StdVectorTypes< Vec3, Vec3, Real > GeometricalTypes ;

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
	/// the way the mass should be computed on non-linear elements
	typedef enum 
	{
		EXACT_INTEGRATION=1,
		NUMERICAL_INTEGRATION=2,
		AFFINE_ELEMENT_INTEGRATION=3
	} IntegrationMethod;


    /// Mass info are stocked on vertices and edges (if lumped matrix)
    topology::PointData<helper::vector<MassType> >  vertexMassInfo;
    topology::EdgeData<helper::vector<MassType> >   edgeMassInfo;

    /* ---------- Specific data for Bezier Elements ------*/
    /// use this data structure to store mass for Bezier tetrahedra. 
    //// The size of the vector is nbControlPoints*(nbControlPoints+1)/2 where nbControlPoints=(degree+1)*(degree+2)*(degree+3)/2
    topology::TetrahedronData<helper::vector<MassVector> > tetrahedronMassInfo;
    // array of Tetrahedral Bezier indices
    //sofa::helper::vector<TetrahedronBezierIndex> tbiArray;
    /* ---------- end ------*/

    /// the mass density used to compute the mass from a mesh topology and geometry
    Data< Real >         m_massDensity;

    /// to display the center of gravity of the system
    Data< bool >         showCenterOfGravity;
    Data< Real >         showAxisSize;
    /// if mass lumping should be performed (only compute mass on vertices)
    Data< bool >         lumping;
    /// if specific mass information should be outputed
    Data< bool >         printMass;
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graph;
    /// the order of integration for numerical integration
    Data<size_t>	     numericalIntegrationOrder;
    /// the type of numerical integration method chosen
    Data<size_t>	     numericalIntegrationMethod;
    /// the type of integration method chosen for non linear element.
    Data<std::string>	 d_integrationMethod; 
    IntegrationMethod    integrationMethod;



protected:

    /// The type of topology to build the mass from the topology
    TopologyType topologyType;
    Real massLumpingCoeff;
    Real savedMass;

    MeshMatrixMass();
    ~MeshMatrixMass();

    /// Internal data required for Cuda computation (copy of vertex mass for deviceRead)
    MeshMatrixMassInternalData<DataTypes, MassType> data;
    friend class MeshMatrixMassInternalData<DataTypes, MassType>;

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

    TopologyType getMassTopologyType() const
    {
        return topologyType;
    }

    void setMassTopologyType(TopologyType t)
    {
        topologyType = t;
    }


    Real getMassDensity() const
    {
        return m_massDensity.getValue();
    }

    void setMassDensity(Real m)
    {
        m_massDensity.setValue(m);
    }

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

    virtual bool isDiagonal() override {return false;}



    /// Add Mass contribution to global Matrix assembling
    virtual void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    virtual SReal getElementMass(unsigned int index) const override;
    virtual void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const override;

    virtual void draw(const core::visual::VisualParams* vparams) override;

    /// Answer wether mass matrix is lumped or not
    bool isLumped() { return lumping.getValue(); }


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
    VertexMassHandler* vertexMassHandler;

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

    EdgeMassHandler* edgeMassHandler;

    class TetrahedronMassHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Tetrahedron,MassVectorVector>
    {
    public:
        typedef typename DataTypes::Real Real;
        TetrahedronMassHandler(MeshMatrixMass<DataTypes,TMassType>* _m, topology::TetrahedronData<helper::vector<MassVector> >* _data) : topology::TopologyDataHandler<core::topology::BaseMeshTopology::Tetrahedron,helper::vector<MassVector> >(_data), m(_m) {}

        /// Edge mass coefficient matrix creation function
        void applyCreateFunction(unsigned int tetrahedronIndex, MassVector & tetrahedronMass,
                const core::topology::BaseMeshTopology::Tetrahedron&,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double >&);

               /// Edge coefficient of mass matrix destruction function to handle creation of new tetrahedra
//        void applyDestructionFunction(const sofa::helper::vector<unsigned int> & /*indices*/);

    protected:
        MeshMatrixMass<DataTypes,TMassType>* m;
    };

    TetrahedronMassHandler* tetrahedronMassHandler;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MASS_MESHMATRIXMASS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<defaulttype::Vec3dTypes,double>;
extern template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<defaulttype::Vec2dTypes,double>;
extern template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<defaulttype::Vec1dTypes,double>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<defaulttype::Vec3fTypes,float>;
extern template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<defaulttype::Vec2fTypes,float>;
extern template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<defaulttype::Vec1fTypes,float>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
