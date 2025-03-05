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
#include <sofa/component/topology/container/dynamic/config.h>

#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/NumericalIntegrationDescriptor.h>

namespace sofa::component::topology::container::dynamic
{

class TetrahedronSetTopologyContainer;
class TetrahedronSetTopologyModifier;

/**
* A class that provides geometry information on an TetrahedronSet.
*/
template < class DataTypes >
class TetrahedronSetGeometryAlgorithms : public TriangleSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TetrahedronSetGeometryAlgorithms,DataTypes),SOFA_TEMPLATE(TriangleSetGeometryAlgorithms,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;



    typedef core::topology::BaseMeshTopology::TetraID TetraID;    
    typedef core::topology::BaseMeshTopology::TetrahedronID TetrahedronID;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundVertex TetrahedraAroundVertex;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundEdge TetrahedraAroundEdge;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundTriangle TetrahedraAroundTriangle;
    typedef core::topology::BaseMeshTopology::EdgesInTetrahedron EdgesInTetrahedron;
    typedef core::topology::BaseMeshTopology::TrianglesInTetrahedron TrianglesInTetrahedron;
    typedef Tetra Tetrahedron;

protected:
    bool initializedCubatureTables;
    void defineTetrahedronCubaturePoints();

    TetrahedronSetGeometryAlgorithms()
        : TriangleSetGeometryAlgorithms<DataTypes>()
        ,initializedCubatureTables(false)
        , d_showTetrahedraIndices (initData(&d_showTetrahedraIndices, (bool) false, "showTetrahedraIndices", "Debug : view Tetrahedrons indices"))
        , d_drawTetrahedra(initData(&d_drawTetrahedra, false, "drawTetrahedra","if true, draw the tetrahedra in the topology"))
        , d_drawScaleTetrahedra(initData(&d_drawScaleTetrahedra, (float) 1.0, "drawScaleTetrahedra", "Scale of the terahedra (between 0 and 1; if <1.0, it produces gaps between the tetrahedra)"))
        , d_drawColorTetrahedra(initData(&d_drawColorTetrahedra, sofa::type::RGBAColor(1.0f,1.0f,0.0f,1.0f), "drawColorTetrahedra", "RGBA code color used to draw tetrahedra."))
    {
        core::objectmodel::Base::addAlias(&d_showTetrahedraIndices, "showTetrasIndices");
        core::objectmodel::Base::addAlias(&d_drawTetrahedra, "drawTetra");
        core::objectmodel::Base::addAlias(&d_drawTetrahedra, "drawTetrahedron");
    }

    ~TetrahedronSetGeometryAlgorithms() override {}
public:
    void init() override;

    void draw(const core::visual::VisualParams* vparams) override;

    void computeTetrahedronAABB(const TetraID i, Coord& minCoord, Coord& maxCoord) const;

    Coord computeTetrahedronCenter(const TetraID i) const;

    Coord computeTetrahedronCircumcenter(const TetraID i) const;

    bool isPointInTetrahedron(const TetraID i, const sofa::type::Vec<3,Real>& p) const;

    /// return (if the point is in the tetrahedron) the barycentric coordinates of the point in the tetrahedron
    bool isPointInTetrahedron(const TetraID ind_t, const sofa::type::Vec<3,Real>& pTest, sofa::type::Vec<4,Real>& barycentricCoordinates) const;

    void getTetrahedronVertexCoordinates(const TetraID i, Coord[4]) const;

    void getRestTetrahedronVertexCoordinates(const TetraID i, Coord[4]) const;

    /// computes the volume of tetrahedron no i and returns it
    Real computeTetrahedronVolume(const TetraID i) const;

    /// computes the tetrahedron volume of all tetrahedra are store in the array interface
    void computeTetrahedronVolume( BasicArrayInterface<Real> &ai) const;

    /// computes the tetrahedron volume  of tetrahedron no i and returns it
    Real computeRestTetrahedronVolume(const TetraID i) const;
    Real computeRestTetrahedronVolume(const Tetrahedron& t) const;


    Real computeDihedralAngle(const TetraID tetraId, const EdgeID edgeId) const;

    /// finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
    void getTetraInBall(const TetraID ind_ta, const TetraID ind_tb,
            sofa::type::vector<TetrahedronID> &indices) const;

    /// finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
    void getTetraInBall(const TetraID ind_ta, Real r,
            sofa::type::vector<TetrahedronID> &indices) const;
    void getTetraInBall(const Coord& c, Real r,
            sofa::type::vector<TetrahedronID> &indices) const;

    /// finds the intersection point with plane which is defined by c and normal
    void getIntersectionPointWithPlane(const TetraID ind_ta, const sofa::type::Vec<3,Real>& planP0, const sofa::type::Vec<3,Real>& normal, sofa::type::vector< sofa::type::Vec<3,Real> >& intersectedPoint, SeqEdges& intersectedEdge);

    /// finds the intersection point between edge and plane
    bool computeIntersectionEdgeWithPlane(const sofa::type::Vec<3,Real>& edgeP1,
                                          const sofa::type::Vec<3,Real>& edgeP2,
                                          const sofa::type::Vec<3,Real>& planP0,
                                          const sofa::type::Vec<3,Real>& normal,
                                          sofa::type::Vec<3,Real>& intersection);
    
    /// Method to check if points stored inside the Tetrahedron, given by the tetrahedron id, are in the right order (by checking the cross products between edges).
    bool checkNodeSequence(const TetraID tetraId) const;
    
    /// Method to check if points stored inside the Tetrahedron, given as parameter, are in the right order (by checking the cross products between edges).
    bool checkNodeSequence(const Tetrahedron& tetra) const;

    /// Method to check if the dihedral angles of the tetrahedron have correct values (between 20 and 160 degrees).
    bool checkTetrahedronDihedralAngles(const TetraID tetraId, SReal minAngle = 20, SReal maxAngle = 160) const;

    /// Method to check if Tetrahedron is elongated, meaning the longest edge > 10x min edge
    bool isTetrahedronElongated(const TetraID tetraId, SReal factorLength = 10) const;

    /// Return false if one of the test method: @sa isTetrahedronElongated, @sa checkTetrahedronDihedralAngles and @sa checkNodeSequence return false for the given Tetrahedron Id.
    bool checkTetrahedronValidity(const TetraID tetraId, SReal minAngle = 20, SReal maxAnglemaxAngle = 160, SReal factorLength = 10) const;

    /// Will call @sa checkTetrahedronValidity for each Tetrahedron of the mesh and store the bad tetrahedron ID in @sa m_badTetraIds
    const sofa::type::vector<TetraID>& computeBadTetrahedron(SReal minAngle = 20, SReal maxAngle = 160, SReal factorLength = 10);

    /// Return bad tetrahedron ID: @sa m_badTetraIds
    const sofa::type::vector<TetraID>& getBadTetrahedronIds();
    
    /// return a pointer to the container of cubature points
    NumericalIntegrationDescriptor<Real,4> &getTetrahedronNumericalIntegrationDescriptor();

    void subDivideTetrahedronsWithPlane(sofa::type::vector< sofa::type::vector<SReal> >& coefs, sofa::type::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal);
    void subDivideTetrahedronsWithPlane(sofa::type::vector<Coord>& intersectedPoints, sofa::type::vector<EdgeID>& intersectedEdgeID, Coord planePos, Coord planeNormal);
    int subDivideTetrahedronWithPlane(TetraID tetraIdx, sofa::type::vector<EdgeID>& intersectedEdgeID, sofa::type::vector<PointID>& intersectedPointID, Coord planeNormal, sofa::type::vector<Tetra>& toBeAddedTetra);

    void subDivideRestTetrahedronsWithPlane(sofa::type::vector< sofa::type::vector<SReal> >& coefs, sofa::type::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal);
    void subDivideRestTetrahedronsWithPlane(sofa::type::vector<Coord>& intersectedPoints, sofa::type::vector<EdgeID>& intersectedEdgeID, Coord planePos, Coord planeNormal);
    int subDivideRestTetrahedronWithPlane(TetraID tetraIdx, sofa::type::vector<EdgeID>& intersectedEdgeID, sofa::type::vector<PointID>& intersectedPointID, Coord planeNormal, sofa::type::vector<Tetra>& toBeAddedTetra);

    SOFA_ATTRIBUTE_DISABLED("v23.12", "v23.12", "Method writeMSHfile has been disabled. To export the topology as .gmsh file, use the sofa::component::io::mesh::MeshExporter.")
    void writeMSHfile(const char *filename) const {msg_deprecated() << "Method writeMSHfile has been disabled. To export the topology as " << filename << " file, use the sofa::component::io::mesh::MeshExporter."; }

protected:
    Data<bool> d_showTetrahedraIndices; ///< Debug : view Tetrahedrons indices
    Data<bool> d_drawTetrahedra; ///< if true, draw the tetrahedra in the topology
    Data<float> d_drawScaleTetrahedra; ///< Scale of the terahedra (between 0 and 1; if <1.0, it produces gaps between the tetrahedra)
    Data<sofa::type::RGBAColor> d_drawColorTetrahedra; ///< RGBA code color used to draw tetrahedra.
    /// include cubature points
    NumericalIntegrationDescriptor<Real,4> tetrahedronNumericalIntegration;

    /// vector of Tetrahedron ID which do not respect @sa checkTetrahedronValidity . buffer updated only by method @sa computeBadTetrahedron
    sofa::type::vector<TetraID> m_badTetraIds;

    TetrahedronSetTopologyContainer*					m_container;
    TetrahedronSetTopologyModifier*						m_modifier;
    unsigned int	m_intialNbPoints;

    bool mustComputeBBox() const override;
};

#if !defined(SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_CPP)
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TetrahedronSetGeometryAlgorithms<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TetrahedronSetGeometryAlgorithms<defaulttype::Vec2Types>;
#endif

} //namespace sofa::component::topology::container::dynamic
