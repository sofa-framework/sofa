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
#ifndef SOFA_COMPONENT_MASS_MESHMATRIXMASS_H
#define SOFA_COMPONENT_MASS_MESHMATRIXMASS_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/component/topology/TopologyData.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/QuadSetGeometryAlgorithms.h>
#include <sofa/component/topology/HexahedronSetGeometryAlgorithms.h>
#include <sofa/core/topology/BaseMeshTopology.h>


//VERY IMPORTANT FOR GRAPHS
#include <sofa/helper/map.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::component::topology;

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
    typedef helper::vector<MassType> MassVector;

    // In case of non 3D template
    typedef Vec<3,Real> Vec3;
    typedef StdVectorTypes< Vec3, Vec3, Real >     GeometricalTypes ; /// assumes the geometry object type is 3D

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


    /// Mass info are stocked on vertices and edges (if lumped matrix)
    PointData<helper::vector<MassType> >  vertexMassInfo;
    EdgeData<helper::vector<MassType> >   edgeMassInfo;


    /// the mass density used to compute the mass from a mesh topology and geometry
    Data< Real >         m_massDensity;

    /// to display the center of gravity of the system
    Data< bool >         showCenterOfGravity;
    Data< Real >         showAxisSize;
    Data< bool >         lumping;
    Data< bool >         printMass;
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graph;



protected:

    /// The type of topology to build the mass from the topology
    TopologyType topologyType;
    Real massLumpingCoeff;
    Real savedMass;

    MeshMatrixMass();
    ~MeshMatrixMass();

public:

    sofa::core::topology::BaseMeshTopology* _topology;

    sofa::component::topology::EdgeSetGeometryAlgorithms<GeometricalTypes>* edgeGeo;
    sofa::component::topology::TriangleSetGeometryAlgorithms<GeometricalTypes>* triangleGeo;
    sofa::component::topology::QuadSetGeometryAlgorithms<GeometricalTypes>* quadGeo;
    sofa::component::topology::TetrahedronSetGeometryAlgorithms<GeometricalTypes>* tetraGeo;
    sofa::component::topology::HexahedronSetGeometryAlgorithms<GeometricalTypes>* hexaGeo;

    void clear();

    virtual void reinit();
    virtual void init();

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


    // -- Mass interface
    void addMDx(const core::MechanicalParams* /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, double factor);

    void accFromF(const core::MechanicalParams* /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f); // This function can't be used as it use M^-1

    void addForce(const core::MechanicalParams* /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    double getKineticEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecDeriv& v) const;  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& x) const;   ///< Mgx potential in a uniform gravity field, null at origin

    void addGravityToV(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_v);

    bool isDiagonal() {return false;}



    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(const core::MechanicalParams *mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    double getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

    void draw(const core::visual::VisualParams* vparams);

    /// Answer wether mass matrix is lumped or not
    bool isLumped() { return lumping.getValue(); }

protected:

    class VertexMassHandler : public topology::TopologyDataHandler<Point,MassVector>
    {
    public:
        VertexMassHandler(MeshMatrixMass<DataTypes,TMassType>* _m, PointData<helper::vector<TMassType> >* _data) : topology::TopologyDataHandler<Point,helper::vector<TMassType> >(_data), m(_m) {}

        /// Mass initialization Creation Functions:
        /// Vertex mass coefficient matrix creation function
        void applyCreateFunction(unsigned int pointIndex, TMassType & VertexMass,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double >&);

        /// Mass coefficient Creation/Destruction functions for Triangular Mesh:
        /// Vertex coefficient of mass matrix creation function to handle creation of new triangles
        void applyTriangleCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< Triangle >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Vertex coefficient of mass matrix destruction function to handle creation of new triangles
        void applyTriangleDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Mass coefficient Creation/Destruction functions for Quad Mesh:
        /// Vertex coefficient of mass matrix creation function to handle creation of new quads
        void applyQuadCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< Quad >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Vertex coefficient of mass matrix destruction function to handle creation of new quads
        void applyQuadDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Mass coefficient Creation/Destruction functions for Tetrahedral Mesh:
        /// Vertex coefficient of mass matrix creation function to handle creation of new tetrahedra
        void applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< Tetrahedron >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Vertex coefficient of mass matrix destruction function to handle creation of new tetrahedra
        void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Mass coefficient Creation/Destruction functions for Hexahedral Mesh:
        /// Vertex coefficient of mass matrix creation function to handle creation of new hexahedra
        void applyHexahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< Hexahedron >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Vertex coefficient of mass matrix destruction function to handle creation of new hexahedra
        void applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

    protected:
        MeshMatrixMass<DataTypes,TMassType>* m;
    };
    VertexMassHandler* vertexMassHandler;

    class EdgeMassHandler : public topology::TopologyDataHandler<Edge,MassVector>
    {
    public:
        EdgeMassHandler(MeshMatrixMass<DataTypes,TMassType>* _m, EdgeData<helper::vector<TMassType> >* _data) : topology::TopologyDataHandler<Edge,helper::vector<TMassType> >(_data), m(_m) {}

        /// Edge mass coefficient matrix creation function
        void applyCreateFunction(unsigned int edgeIndex, MassType & EdgeMass,
                const Edge&,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double >&);

        /// Edge coefficient of mass matrix creation function to handle creation of new triangles
        void applyTriangleCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< Triangle >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Edge coefficient of mass matrix destruction function to handle creation of new triangles
        void applyTriangleDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Edge coefficient of mass matrix creation function to handle creation of new quads
        void applyQuadCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< Quad >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Edge coefficient of mass matrix destruction function to handle creation of new quads
        void applyQuadDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Edge coefficient of mass matrix creation function to handle creation of new tetrahedra
        void applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< Tetrahedron >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Edge coefficient of mass matrix destruction function to handle creation of new tetrahedra
        void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

        /// Edge coefficient of mass matrix creation function to handle creation of new hexahedra
        void applyHexahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
                const sofa::helper::vector< Hexahedron >& /*elems*/,
                const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
                const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/);

        /// Edge coefficient of mass matrix destruction function to handle creation of new hexahedra
        void applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/);

    protected:
        MeshMatrixMass<DataTypes,TMassType>* m;
    };

    EdgeMassHandler* edgeMassHandler;

};

#if defined(WIN32) && !defined(SOFA_COMPONENT_MASS_MESHMATRIXMASS_CPP)
#pragma warning(disable : 4231)
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
