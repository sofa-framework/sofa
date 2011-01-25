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
#include <sofa/component/topology/PointData.h>
#include <sofa/component/topology/EdgeData.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/QuadSetGeometryAlgorithms.h>
#include <sofa/component/topology/HexahedronSetGeometryAlgorithms.h>
#include <sofa/core/topology/BaseMeshTopology.h>

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
    typedef Vec<3,MassType>                            Vec3;
    typedef StdVectorTypes< Vec3, Vec3, MassType >     GeometricalTypes ; /// assumes the geometry object type is 3D

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
    PointData<MassType>  vertexMassInfo;
    EdgeData<MassType>   edgeMassInfo;


    /// the mass density used to compute the mass from a mesh topology and geometry
    Data< Real >         m_massDensity;

    /// to display the center of gravity of the system
    Data< bool >         showCenterOfGravity;
    Data< float >        showAxisSize;
    Data< bool >         lumping;


protected:

    /// The type of topology to build the mass from the topology
    TopologyType topologyType;
    float massLumpingCoeff;
    Real savedMass;


public:

    sofa::core::topology::BaseMeshTopology* _topology;

    sofa::component::topology::EdgeSetGeometryAlgorithms<GeometricalTypes>* edgeGeo;
    sofa::component::topology::TriangleSetGeometryAlgorithms<GeometricalTypes>* triangleGeo;
    sofa::component::topology::QuadSetGeometryAlgorithms<GeometricalTypes>* quadGeo;
    sofa::component::topology::TetrahedronSetGeometryAlgorithms<GeometricalTypes>* tetraGeo;
    sofa::component::topology::HexahedronSetGeometryAlgorithms<GeometricalTypes>* hexaGeo;

    MeshMatrixMass();

    ~MeshMatrixMass() {};

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


    // handle topological changes
    virtual void handleTopologyChange();


    // -- Mass interface
    void addMDx(DataVecDeriv& f, const DataVecDeriv& dx, double factor, const core::MechanicalParams*);

    void accFromF(DataVecDeriv& a, const DataVecDeriv& f, const core::MechanicalParams*); // This function can't be used as it use M^-1

    void addForce(DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v, const core::MechanicalParams*);

    double getKineticEnergy(const DataVecDeriv& v, const core::MechanicalParams*) const;  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const DataVecCoord& x, const core::MechanicalParams*) const;   ///< Mgx potential in a uniform gravity field, null at origin

    void addGravityToV(DataVecDeriv& d_v, const core::MechanicalParams* mparams);

    bool isDiagonal() {return false;}



    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, const core::MechanicalParams *mparams);

    double getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

    void draw();

    bool addBBox(double* minBBox, double* maxBBox);

    /// Answer wether mass matrix is lumped or not
    bool isLumped() { return lumping.getValue(); }


protected:
    /// Mass initialization Creation Functions:
    /// Vertex mass coefficient matrix creation function
    static void VertexMassCreationFunction(int, void* , MassType & VertexMass,
            const sofa::helper::vector< unsigned int > &,
            const sofa::helper::vector< double >&);

    /// Edge mass coefficient matrix creation function
    static void EdgeMassCreationFunction(int, void* , MassType & EdgeMass,
            const Edge&,
            const sofa::helper::vector< unsigned int > &,
            const sofa::helper::vector< double >&);


    /// Mass coefficient Creation/Destruction functions for Triangular Mesh:
    /// Vertex coefficient of mass matrix creation function to handle creation of new triangles
    static void VertexMassTriangleCreationFunction(const sofa::helper::vector<unsigned int> &triangleAdded,
            void* param, vector<MassType> &VertexMasses);

    /// Edge coefficient of mass matrix creation function to handle creation of new triangles
    static void EdgeMassTriangleCreationFunction(const sofa::helper::vector<unsigned int> &triangleAdded,
            void* param, vector<MassType> &EdgeMasses);

    /// Vertex coefficient of mass matrix destruction function to handle creation of new triangles
    static void VertexMassTriangleDestroyFunction(const sofa::helper::vector<unsigned int> &triangleRemoved,
            void* param, vector<MassType> &VertexMasses);

    /// Edge coefficient of mass matrix destruction function to handle creation of new triangles
    static void EdgeMassTriangleDestroyFunction(const sofa::helper::vector<unsigned int> &triangleRemoved,
            void* param, vector<MassType> &EdgeMasses);


    /// Mass coefficient Creation/Destruction functions for Quad Mesh:
    /// Vertex coefficient of mass matrix creation function to handle creation of new quads
    static void VertexMassQuadCreationFunction(const sofa::helper::vector<unsigned int> &quadAdded,
            void* param, vector<MassType> &VertexMasses);

    /// Edge coefficient of mass matrix creation function to handle creation of new quads
    static void EdgeMassQuadCreationFunction(const sofa::helper::vector<unsigned int> &quadAdded,
            void* param, vector<MassType> &EdgeMasses);

    /// Vertex coefficient of mass matrix destruction function to handle creation of new quads
    static void VertexMassQuadDestroyFunction(const sofa::helper::vector<unsigned int> &quadRemoved,
            void* param, vector<MassType> &VertexMasses);

    /// Edge coefficient of mass matrix destruction function to handle creation of new quads
    static void EdgeMassQuadDestroyFunction(const sofa::helper::vector<unsigned int> &quadRemoved,
            void* param, vector<MassType> &EdgeMasses);


    /// Mass coefficient Creation/Destruction functions for Tetrahedral Mesh:
    /// Vertex coefficient of mass matrix creation function to handle creation of new tetrahedra
    static void VertexMassTetrahedronCreationFunction(const sofa::helper::vector<unsigned int> &tetrahedronAdded,
            void* param, vector<MassType> &VertexMasses);

    /// Edge coefficient of mass matrix creation function to handle creation of new tetrahedra
    static void EdgeMassTetrahedronCreationFunction(const sofa::helper::vector<unsigned int>  &tetrahedronAdded,
            void* param, vector<MassType> &EdgeMasses);

    /// Vertex coefficient of mass matrix destruction function to handle creation of new tetrahedra
    static void VertexMassTetrahedronDestroyFunction(const sofa::helper::vector<unsigned int> &tetrahedronRemoved,
            void* param, vector<MassType> &VertexMasses);

    /// Edge coefficient of mass matrix destruction function to handle creation of new tetrahedra
    static void EdgeMassTetrahedronDestroyFunction(const sofa::helper::vector<unsigned int> &tetrahedronRemoved,
            void* param, vector<MassType> &EdgeMasses);


    /// Mass coefficient Creation/Destruction functions for Hexahedral Mesh:
    /// Vertex coefficient of mass matrix creation function to handle creation of new hexahedra
    static void VertexMassHexahedronCreationFunction(const sofa::helper::vector<unsigned int> &hexahedronAdded,
            void* param, vector<MassType> &VertexMasses);

    /// Edge coefficient of mass matrix creation function to handle creation of new hexahedra
    static void EdgeMassHexahedronCreationFunction(const sofa::helper::vector<unsigned int> &hexahedronAdded,
            void* param, vector<MassType> &EdgeMasses);

    /// Vertex coefficient of mass matrix destruction function to handle creation of new hexahedra
    static void VertexMassHexahedronDestroyFunction(const sofa::helper::vector<unsigned int> &hexahedronRemoved,
            void* param, vector<MassType> &VertexMasses);

    /// Edge coefficient of mass matrix destruction function to handle creation of new hexahedra
    static void EdgeMassHexahedronDestroyFunction(const sofa::helper::vector<unsigned int> &hexahedronRemoved,
            void* param, vector<MassType> &EdgeMasses);

};

#if defined(WIN32) && !defined(SOFA_COMPONENT_MASS_MESHMATRIXMASS_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec3dTypes,double>;
extern template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec2dTypes,double>;
extern template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec1dTypes,double>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec3fTypes,float>;
extern template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec2fTypes,float>;
extern template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec1fTypes,float>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
