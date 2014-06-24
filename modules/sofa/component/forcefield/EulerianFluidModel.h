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
#ifndef SOFA_COMPONENT_FORCEFIELD_EULERIANFLUIDMODEL_H
#define SOFA_COMPONENT_FORCEFIELD_EULERIANFLUIDMODEL_H

#include <newmat/newmat.h>
#include <newmat/newmatap.h>

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/BehaviorModel.h>

#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/system/thread/ctime.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>

#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.inl>
//#include <sofa/component/topology/PointSubset.h>
//#include <sofa/component/topology/EdgeSubsetData.h>


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
    SOFA_CLASS(SOFA_TEMPLATE(EulerianFluidModel, DataTypes), SOFA_TEMPLATE(sofa::core::BehaviorModel));

    typedef sofa::core::BehaviorModel Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef typename topology::MeshTopology::PointID PointID;
    typedef typename topology::MeshTopology::EdgeID EdgeID;
    typedef typename topology::MeshTopology::index_type FaceID;

    typedef typename topology::MeshTopology::Edge Edge;
    typedef typename topology::MeshTopology::Triangle Triangle;
    typedef typename topology::MeshTopology::Quad Quad;

    typedef typename topology::MeshTopology::EdgesAroundVertex EdgesAroundVertex;
    typedef typename sofa::helper::vector<FaceID> VertexFaces;
    typedef typename sofa::helper::vector<FaceID> EdgeFaces;
    typedef typename topology::MeshTopology::EdgesInTriangle EdgesInTriangle;
    typedef typename topology::MeshTopology::EdgesInQuad EdgesInQuad;

    typedef typename topology::PointSetGeometryAlgorithms<DataTypes>::Angle Angle;

    typedef bool CenterType;

    typedef sofa::helper::vector< unsigned int > SetIndex;

    //at present only for 2D
    enum MeshType {TriangleMesh = 3, QuadMesh, RegularQuadMesh};
    enum {Barycenter = 0, Circumcenter = 1};

    EulerianFluidModel();
    ~EulerianFluidModel();

    Real getForce() const { return m_force.getValue(); }
    void setForce(Real val) { m_force.setValue(val); }

    bool getDisplayBoundary() const { return m_bDisplayBoundary.getValue(); }
    void setDisplayBoundary(bool val) { m_bDisplayBoundary.setValue(val); }
    bool getDisplayDualMesh() const { return m_bDisplayDualMesh.getValue(); }
    void setDisplayDualMesh(bool val) { m_bDisplayDualMesh.setValue(val); }
    bool getDisplayBkMesh() const { return m_bDisplayBkMesh.getValue(); }
    void setDisplayBkMesh(bool val) { m_bDisplayBkMesh.setValue(val); }
    bool getDisplayVorticity() const { return m_bDisplayVorticity.getValue(); }
    void setDisplayVorticity(bool val) { m_bDisplayVorticity.setValue(val); }
    bool getDisplayVelocity() const { return m_bDisplayVelocity.getValue(); }
    void setDisplayVelocity(bool val) { m_bDisplayVelocity.setValue(val); }
    Real getVisCoef1() const { return m_visCoef1.getValue(); }
    void setVisCoef1(Real val) { m_visCoef1.setValue(val); }
    Real getVisCoef2() const { return m_visCoef2.getValue(); }
    void setVisCoef2(Real val) { m_visCoef2.setValue(val); }
    Real getVisCoef3() const { return m_visCoef3.getValue(); }
    void setVisCoef3(Real val) { m_visCoef3.setValue(val); }

    Real getBdXmin1() const { return m_bdXmin1.getValue(); }
    void setBdXmin1(Real val) { m_bdXmin1.setValue(val); }
    Real getBdXmax1() const { return m_bdXmax1.getValue(); }
    void setBdXmax1(Real val) { m_bdXmax1.setValue(val); }
    Real getBdYmin1() const { return m_bdYmin1.getValue(); }
    void setBdYmin1(Real val) { m_bdYmin1.setValue(val); }
    Real getBdYmax1() const { return m_bdYmax1.getValue(); }
    void setBdYmax1(Real val) { m_bdYmax1.setValue(val); }
    Real getBdZmin1() const { return m_bdZmin1.getValue(); }
    void setBdZmin1(Real val) { m_bdZmin1.setValue(val); }
    Real getBdZmax1() const { return m_bdZmax1.getValue(); }
    void setBdZmax1(Real val) { m_bdZmax1.setValue(val); }
    Real getBdValue1() const { return m_bdValue1.getValue(); }
    void setBdValue1(Real val) { m_bdValue1.setValue(val); }

    Real getBdXmin2() const { return m_bdXmin2.getValue(); }
    void setBdXmin2(Real val) { m_bdXmin2.setValue(val); }
    Real getBdXmax2() const { return m_bdXmax2.getValue(); }
    void setBdXmax2(Real val) { m_bdXmax2.setValue(val); }
    Real getBdYmin2() const { return m_bdYmin2.getValue(); }
    void setBdYmin2(Real val) { m_bdYmin2.setValue(val); }
    Real getBdYmax2() const { return m_bdYmax2.getValue(); }
    void setBdYmax2(Real val) { m_bdYmax2.setValue(val); }
    Real getBdZmin2() const { return m_bdZmin2.getValue(); }
    void setBdZmin2(Real val) { m_bdZmin2.setValue(val); }
    Real getBdZmax2() const { return m_bdZmax2.getValue(); }
    void setBdZmax2(Real val) { m_bdZmax2.setValue(val); }
    Real getBdValue2() const { return m_bdValue2.getValue();}
    void setBdValue2(Real val) { m_bdValue2.setValue(val); }


    Real getHarmonicVx() const { return m_harmonicVx.getValue();}
    void setHarmonicVx(Real val) { m_harmonicVx.setValue(val);}
    Real getHarmonicVy() const { return m_harmonicVy.getValue();}
    void setHarmonicVy(Real val) { m_harmonicVy.setValue(val);}
    Real getHarmonicVz() const { return m_harmonicVz.getValue();}
    void setHarmonicVz(Real val) { m_harmonicVz.setValue(val);}

    Real getViscosity() const { return m_viscosity.getValue();}
    void setViscosity(Real val) { m_viscosity.setValue(val);}

    CenterType getCenterType() const { return m_centerType.getValue(); }
    void setCenterType(CenterType ct) { m_centerType.setValue(ct); }

    virtual void parse(core::objectmodel::BaseObjectDescription* arg);
    virtual void init();
    virtual void reinit();
    virtual void updatePosition(double dt);
    virtual void draw(const core::visual::VisualParams* vparams);

protected:

    // profiling information
    ctime_t m_dTime1, m_dTime2;

    //arguments
    Data< bool > m_bViscous;
    Data< Real > m_viscosity;

    Data< bool > m_bAddForces;
    Data< Real > m_force;
    Data<SetIndex> m_addForcePointSet;

    Data< bool > m_bDisplayBoundary;
    Data< bool > m_bDisplayDualMesh;
    Data< bool > m_bDisplayBkMesh;
    Data< bool > m_bDisplayVelocity;
    Data< bool > m_bDisplayBkVelocity;
    Data< bool > m_bDisplayVorticity;
    Data< Real > m_visCoef1;	//visualization coef
    Data< Real > m_visCoef2;	//visualization coef
    Data< Real > m_visCoef3;	//visualization coef

    Data< Real > m_harmonicVx;
    Data< Real > m_harmonicVy;
    Data< Real > m_harmonicVz;

    Data<SetIndex> m_cstrEdgeSet_1;
    Data<SetIndex> m_cstrEdgeSet_2;
    Data<SetIndex> m_cstrEdgeSet_3;
    Data< Real > m_cstrValue_1;
    Data< Real > m_cstrValue_2;
    Data< Real > m_cstrValue_3;

    Data< Real > m_bdXmin1;
    Data< Real > m_bdXmax1;
    Data< Real > m_bdYmin1;
    Data< Real > m_bdYmax1;
    Data< Real > m_bdZmin1;
    Data< Real > m_bdZmax1;
    Data< Real> m_bdValue1;

    Data< Real > m_bdXmin2;
    Data< Real > m_bdXmax2;
    Data< Real > m_bdYmin2;
    Data< Real > m_bdYmax2;
    Data< Real > m_bdZmin2;
    Data< Real > m_bdZmax2;
    Data< Real> m_bdValue2;

    Data< CenterType > m_centerType;

    //topology and geometry related data
    MechanicalState *m_mstate;
    topology::MeshTopology* m_topology;
    MeshType m_meshType;
    sofa::component::topology::TriangleSetGeometryAlgorithms<DataTypes>* m_triGeo;
    sofa::component::topology::QuadSetGeometryAlgorithms<DataTypes>* m_quadGeo;
    unsigned int m_nbPoints;
    unsigned int m_nbEdges;
    unsigned int m_nbFaces;

    //mesh element information
    class PointInformation
    {
    public:
        typedef VecCoord DualFace;
        sofa::helper::vector<bool> m_isBoundary;
        sofa::helper::vector<VecCoord> m_dualFaces;

        //values for display
        sofa::helper::vector<float> m_values;
    };
    class EdgeInformation
    {
    public:
        sofa::helper::vector<bool> m_isBoundary;
        sofa::helper::vector<double> m_lengths;
        VecCoord m_unitTangentVectors;
        VecCoord m_centers;
    };
    class FaceInformation
    {
    public:
        sofa::helper::vector<bool> m_isBoundary;
        VecCoord m_centers;

        typedef NEWMAT::SymmetricMatrix ProjectMatrix;
        sofa::helper::vector< ProjectMatrix > m_AtAInv;
        sofa::helper::vector< NEWMAT::Matrix > m_At;

        //vectors for display
        VecCoord m_vectors;

        // list of incorrect (obtuse) triangles
        sofa::helper::vector< FaceID > m_obtuseTris;
    };
    class BoundaryPointInformation
    {
    public:
        Coord m_bdVel;	//boundary velocity
        Coord m_bkPoint;//backtrack point position
        Coord m_bkVel;	//velocity at backtrack point
        Coord m_vector;	//vector for display
        BoundaryPointInformation(const Coord& vel)
            : m_bdVel(vel), m_bkPoint(Coord(0, 0, 0)), m_bkVel(Coord(0, 0, 0)), m_vector(Coord(0, 0, 0)) {};
        BoundaryPointInformation()
            : m_bdVel(Coord(0, 0, 0)), m_bkPoint(Coord(0, 0, 0)), m_bkVel(Coord(0, 0, 0)), m_vector(Coord(0, 0, 0)) {};
    };
    class BoundaryEdgeInformation
    {
    public:
        double m_bdConstraint;	//boundary constraints
        Coord m_bdVel;			//boundary velocity
        Coord m_unitFluxVector;	//unit vector of flux
        Coord m_bkECenter;		//backtrack edge center
        Coord m_bkVel;			//velocity at backtrack edge center
        Coord m_vector;			//vector for display
        BoundaryEdgeInformation(const double c, const Coord& vel, const Coord& vec)
            : m_bdConstraint(c), m_bdVel(vel), m_unitFluxVector(vec),
              m_bkECenter(Coord(0, 0, 0)), m_bkVel(Coord(0, 0, 0)),m_vector(Coord(0, 0, 0)) {};
        BoundaryEdgeInformation()
            : m_bdConstraint(0.0), m_bdVel(Coord(0, 0, 0)), m_unitFluxVector(Coord(0, 0, 0)),
              m_bkECenter(Coord(0, 0, 0)), m_bkVel(Coord(0, 0, 0)), m_vector(Coord(0, 0, 0)) {};
    };

    typedef typename PointInformation::DualFace DualFace;
    typedef typename std::map<EdgeID, BoundaryEdgeInformation>::iterator BoundaryEdgeIterator;
    typedef typename std::map<PointID, BoundaryPointInformation>::iterator BoundaryPointIterator;

    PointInformation m_pInfo;
    EdgeInformation m_eInfo;
    FaceInformation m_fInfo;
    std::map<PointID, BoundaryPointInformation> m_bdPointInfo;
    std::map<EdgeID, BoundaryEdgeInformation> m_bdEdgeInfo;

    //operators
    sofa::component::linearsolver::SparseMatrix<int> d0;
    sofa::component::linearsolver::SparseMatrix<int> d1;
    sofa::component::linearsolver::FullVector<double> star0;
    sofa::component::linearsolver::FullVector<double> star1;
    sofa::component::linearsolver::FullVector<double> star2;
    sofa::component::linearsolver::SparseMatrix<double> curl;
    sofa::component::linearsolver::SparseMatrix<double> laplace;
    NEWMAT::Matrix m_d0;
    NEWMAT::Matrix m_laplace;
    NEWMAT::Matrix m_laplace_inv;
    NEWMAT::Matrix m_diffusion;
    NEWMAT::Matrix m_diffusion_inv;
    bool m_bFirstComputation;

    //state variables
    NEWMAT::ColumnVector m_flux;
    NEWMAT::ColumnVector m_vorticity;
    NEWMAT::ColumnVector m_phi;
    VecDeriv m_vels;											//velocity at the dual vertices
    VecDeriv m_bkVels;											//velocity at the backtrack centers
    VecCoord m_bkCenters;										//backtrack centers

    //compute element information: point, edge, face
    void computeElementInformation();
    //calculate operators d, star, curl, laplace
    void computeOperators();
    void computeDerivativesForTriMesh();
    void computeDerivativesForQuadMesh();
    void computeHodgeStarsForTriMesh();
    void computeHodgeStarsForQuadMesh();
    //calculate the project matrix for U => v
    void computeProjectMats();

    //set boundary contraints
    void setBdConstraints(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax, double value);
    void setBdConstraints();
    //set initial values
    void setInitialVorticity();

    // U => v
    //calculate velocity at dual vertices (face centers), boudary edge centers and boundary points
    void calcVelocity();

    // v => Omega
    //search the face(tri/quad) index in which pt is in
    unsigned int searchFaceForTriMesh(const Coord& pt, FaceID startFace) const;
    //search the dual face in which pt is, with the start face iFace
    unsigned int searchDualFaceForTriMesh(const Coord& pt, PointID startDualFace) const;
    unsigned int searchDualFaceForQuadMesh(const Coord & pt, PointID startDualFace) const;
    //interpolate velocity
    Deriv interpolateVelocity(const Coord& pt, const PointID start);
    //backtrack face centers
    void backtrack(double dt);
    //calculate vorticity
    void calcVorticity();
    //add forces
    void addForces();
    //add diffusion for viscous fluid
    void calcDiffusion(double dt);
    void addDiffusion();

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
    void saveOperators();
    void saveDiffusion();
    void saveVorticity() const;
    void savePhi() const;
    void saveFlux() const;
    void saveVelocity() const;

    //test
    void test();
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
