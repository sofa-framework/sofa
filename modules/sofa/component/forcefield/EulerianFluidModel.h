/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_EULERIANFLUIDMODEL_H
#define SOFA_COMPONENT_FORCEFIELD_EULERIANFLUIDMODEL_H

#include <NewMAT/newmat.h>
#include <NewMAT/newmatap.h>

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/BehaviorModel.h>

#include <sofa/helper/fixed_array.h>
#include <sofa/helper/system/thread/ctime.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>

#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/EdgeSetGeometryAlgorithms.inl>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.inl>
#include <sofa/component/topology/QuadSetGeometryAlgorithms.h>
#include <sofa/component/topology/QuadSetGeometryAlgorithms.inl>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::helper::system::thread;

template<class DataTypes>
class EulerianFluidModel : public sofa::core::BehaviorModel
{
public:
    typedef sofa::core::BehaviorModel Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef typename topology::MeshTopology::PointID PointID;
    typedef typename topology::MeshTopology::EdgeID EdgeID;
    typedef typename topology::MeshTopology::index_type FaceID;

    typedef typename topology::MeshTopology::Edge Edge;
    typedef typename topology::MeshTopology::Triangle Triangle;
    typedef typename topology::MeshTopology::Quad Quad;

    typedef typename topology::MeshTopology::VertexEdges VertexEdges;
    typedef typename sofa::helper::vector<FaceID> VertexFaces;
    typedef typename sofa::helper::vector<FaceID> EdgeFaces;
    typedef typename topology::MeshTopology::TriangleEdges TriangleEdges;
    typedef typename topology::MeshTopology::QuadEdges QuadEdges;
    typedef bool CenterType;

    //at present only for 2D
    enum MeshType {TriangleMesh = 3, QuadMesh, RegularQuadMesh};
    enum { Barycenter = 0, Circumcenter = 1};

protected:

    // profiling information
    ctime_t m_dTime1, m_dTime2;

    //arguments
    Data< bool > m_bAddForces;
    Data< Real > m_viscosity;
    Data< bool > m_bDisplayBoundary;
    Data< bool > m_bDisplayVorticity;
    Data< bool > m_bDisplayVelocity;
    Data< Real > m_harmonicVx;
    Data< Real > m_harmonicVy;
    Data< Real > m_harmonicVz;
    Data< Real > m_bdX;
    Data< Real > m_bdY;
    Data< Real > m_bdZ;
    Data< CenterType > m_centerType;

    //topology and geometry related data
    MechanicalState *m_mstate;
    topology::MeshTopology* m_topology;
    MeshType m_meshType;
    sofa::component::topology::TriangleSetGeometryAlgorithms<DataTypes>* m_triGeo;
    sofa::component::topology::QuadSetGeometryAlgorithms<DataTypes>* m_quadGeo;
    std::map<int, double> m_bdConstraints;	//boundary constraints

    unsigned int m_nbPoints;
    unsigned int m_nbEdges;
    unsigned int m_nbFaces;
    unsigned int m_nbVolumes;

public:

    EulerianFluidModel();
    ~EulerianFluidModel();

    Real getBdX() const { return m_bdX.getValue(); }
    void setBdX(Real val) { m_bdX.setValue(val); }
    Real getBdY() const { return m_bdY.getValue(); }
    void setBdY(Real val) { m_bdY.setValue(val); }
    Real getBdZ() const { return m_bdZ.getValue(); }
    void setBdZ(Real val) { m_bdZ.setValue(val); }
    bool getDisplayBoundary() const { return m_bDisplayBoundary.getValue(); }
    void setDisplayBoundary(bool val) { m_bDisplayBoundary.setValue(val); }
    bool getDisplayVorticity() const { return m_bDisplayVorticity.getValue(); }
    void setDisplayVorticity(bool val) { m_bDisplayVorticity.setValue(val); }
    bool getDisplayVelocity() const { return m_bDisplayVelocity.getValue(); }
    void setDisplayVelocity(bool val) { m_bDisplayVelocity.setValue(val); }
    Real getHarmonicVx() const { return m_harmonicVx.getValue(); }
    void setHarmonicVx(Real val) { m_harmonicVx.setValue(val); }
    Real getHarmonicVy() const { return m_harmonicVy.getValue(); }
    void setHarmonicVy(Real val) { m_harmonicVy.setValue(val); }
    Real getHarmonicVz() const { return m_harmonicVz.getValue(); }
    void setHarmonicVz(Real val) { m_harmonicVz.setValue(val); }
    Real getViscosity() const { return m_viscosity.getValue(); }
    void setViscosity(Real val) { m_viscosity.setValue(val); }

    CenterType getCenterType() const { return m_centerType.getValue(); }
    void setCenterType(CenterType ct) { m_centerType.setValue(ct); }

    virtual void parse(core::objectmodel::BaseObjectDescription* arg);

    virtual void init();

    virtual void updatePosition(double dt);

    virtual void draw();

protected:
    //Information on the mesh
    class PointInformation
    {
    public:
        typedef sofa::defaulttype::Vec<3, double> VertexOfDualFace;
        typedef sofa::helper::vector<VertexOfDualFace> DualFace;
        typedef Coord Normal;
        typedef sofa::defaulttype::Vec<2, Normal> VertexNormal;
        typedef sofa::helper::vector<VertexNormal> DualFaceVertexNormal;
        typedef sofa::helper::vector<double> Volumes;

        sofa::helper::vector<bool> m_isBoundary;
        sofa::helper::vector<DualFace> m_dualFaces;	//the poistions of vertices of dual face according to the points

        //values for interpolation
        sofa::helper::vector<DualFaceVertexNormal> m_dualFaceVertexNormals;
        sofa::helper::vector<Volumes> m_dualFaceVertexVolumes;

        //values for display
        sofa::helper::vector<float> m_values;
    };
    class EdgeInformation
    {
    public:
        sofa::helper::vector<bool> m_isBoundary;
        sofa::helper::vector< sofa::defaulttype::Vec<3, double> > m_centers;
    };
    class FaceInformation
    {
    public:
        sofa::helper::vector<bool> m_isBoundary;
        sofa::helper::vector< sofa::defaulttype::Vec<3, double> > m_centers;

        typedef NewMAT::SymmetricMatrix ProjectMatrix;
        sofa::helper::vector< ProjectMatrix > m_AtAInv;
        sofa::helper::vector< NewMAT::Matrix > m_At;

        //values for display
        sofa::helper::vector< sofa::defaulttype::Vec<3, double> > m_vectors;
    };

    PointInformation m_pInfo;
    EdgeInformation m_eInfo;
    FaceInformation m_fInfo;

    typedef typename PointInformation::Normal Normal;
    typedef typename PointInformation::VertexNormal VertexNormal;
    typedef typename PointInformation::VertexOfDualFace VertexOfDualFace;
    typedef typename PointInformation::DualFace DualFace;

    //operators
    sofa::component::linearsolver::SparseMatrix<int> d0;
    sofa::component::linearsolver::SparseMatrix<int> d1;
    sofa::component::linearsolver::FullVector<double> star0;
    sofa::component::linearsolver::FullVector<double> star1;
    sofa::component::linearsolver::FullVector<double> star2;
    sofa::component::linearsolver::SparseMatrix<double> curl;
    sofa::component::linearsolver::SparseMatrix<double> laplace;
    NewMAT::Matrix m_laplace;
    NewMAT::Matrix m_laplace_inv;
    NewMAT::Matrix m_d0;

    //state variables
    //sofa::component::linearsolver::FullVector<double> m_flux;	//flux on the edge
    NewMAT::ColumnVector m_flux;
    NewMAT::ColumnVector m_vorticity;
    NewMAT::ColumnVector m_phi;
    VecDeriv m_vels;											//velocity on the dual vertices
    VecCoord m_bkCenters;										//backtrack centers
    VecDeriv m_bkVels;											//velocity on the backtrack centers

    //compute isBoundary
    void computeBoundary2D();
    //compute the centers of face, barycenter or circumcenter
    void computeFaceCenters();
    //compute the centers of edge
    void computeEdgeCenters();
    //compute vertices of dual face
    void computeDualFaces();
    //calculate operators d, star, curl, laplace
    void computeOperators();
    void computeDerivativesForTriMesh();
    void computeDerivativesForQuadMesh();
    void computeHodgeStarsForTriMesh();
    void computeHodgeStarsForQuadMesh();
    //calculate the project matrix for U => v
    void computeProjectMats();

    //set boundary contraints
    void setBdConstraints(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax);
    //set initial values
    void setInitialVorticity();

    // U => v
    //calculate velocity at dual vertex
    void calcVelocityAtDualVertex();

    // v => Omega
    //search the face(tri/quad) index in which pt is in
    FaceID searchFaceForTriMesh(const sofa::defaulttype::Vec<3, double>& pt, FaceID startFace) const;
    //search the dual face in which pt is, with the start face iFace
    PointID searchDualFaceForTriMesh(const Coord& pt, const FaceID startFace) const;
    PointID searchDualFaceForQuadMesh(const Coord & pt, PointID startDualFace) const;
    //interpolate velocity
    Deriv interpolateVelocity(const Coord& pt, unsigned int start);
    //backtrack face centers
    void backtrack(double dt);
    //calculate vorticity
    void calcVorticity();
    //add forces
    void addForces();

    // Omega => Phi
    void calcPhi(bool reset);

    // Phi => U
    void calcFlux();

    // boundary condition
    void setBoundaryFlux();

    //normalize the values for displaying
    void normalizeDisplayValues();

    //save data
    void saveMeshData() const;
    void saveOperators() const;
    void saveVorticity() const;
    void savePhi() const;
    void saveFlux() const;
    void saveVelocity() const;
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
