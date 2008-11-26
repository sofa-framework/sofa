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
#ifndef SOFA_COMPONENT_FORCEFIELD_EULERIANFLUIDMODEL_INL
#define SOFA_COMPONENT_FORCEFIELD_EULERIANFLUIDMODEL_INL

#include <sofa/component/forcefield/EulerianFluidModel.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/template.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
EulerianFluidModel<DataTypes>::EulerianFluidModel()
    :
    m_bAddForces(initData(&m_bAddForces, bool(0), "addFroces", "Add Forces")),
    m_bDisplayBoundary  (initData(&m_bDisplayBoundary, bool(0), "displayBoundary", "Display Boundary")),
    m_bDisplayVorticity  (initData(&m_bDisplayVorticity, bool(0), "displayVorticity", "Display Vorticity")),
    m_bDisplayVelocity  (initData(&m_bDisplayVelocity, bool(0), "displayVelocity", "Display Velocity")),
    m_visCoef1(initData(&m_visCoef1, Real(0.2), "visCoef1", "Visualization Coefficent 1")),
    m_visCoef2(initData(&m_visCoef2, Real(0.2), "visCoef2", "Visualization Coefficent 2")),
    m_visCoef3(initData(&m_visCoef3, Real(10), "visCoef3", "Visualization Coefficent 3")),
    m_harmonicVx(initData(&m_harmonicVx, Real(0), "harmonicVx", "Harmonic Velocity x")),
    m_harmonicVy(initData(&m_harmonicVy, Real(0), "harmonicVy", "Harmonic Velocity y")),
    m_harmonicVz(initData(&m_harmonicVz, Real(0), "harmonicVz", "Harmonic Velocity z")),
    m_bdXmin1 (initData(&m_bdXmin1, Real(0), "bdXmin1", "BoundaryX")),
    m_bdYmin1 (initData(&m_bdYmin1, Real(0), "bdYmin1", "BoundaryY")),
    m_bdZmin1 (initData(&m_bdZmin1, Real(0), "bdZmin1", "BoundaryZ")),
    m_bdXmax1 (initData(&m_bdXmax1, Real(0), "bdXmax1", "BoundaryX")),
    m_bdYmax1 (initData(&m_bdYmax1, Real(0), "bdYmax1", "BoundaryY")),
    m_bdZmax1 (initData(&m_bdZmax1, Real(0), "bdZmax1", "BoundaryZ")),
    m_bdXmin2 (initData(&m_bdXmin2, Real(0), "bdXmin2", "BoundaryX")),
    m_bdYmin2 (initData(&m_bdYmin2, Real(0), "bdYmin2", "BoundaryY")),
    m_bdZmin2 (initData(&m_bdZmin2, Real(0), "bdZmin2", "BoundaryZ")),
    m_bdXmax2 (initData(&m_bdXmax2, Real(0), "bdXmax2", "BoundaryX")),
    m_bdYmax2 (initData(&m_bdYmax2, Real(0), "bdYmax2", "BoundaryY")),
    m_bdZmax2 (initData(&m_bdZmax2, Real(0), "bdZmax2", "BoundaryZ")),
    m_bdValue1 (initData(&m_bdValue1, Real(0), "bdValue1", "Value")),
    m_bdValue2 (initData(&m_bdValue2, Real(0), "bdValue2", "Value")),
    m_viscosity  (initData(&m_viscosity, Real(0), "viscosity", "Fluid Viscosity")),
    m_centerType  (initData(&m_centerType, CenterType(0), "centerType", "Center Type")),
    m_mstate(NULL), m_topology(NULL), m_triGeo(NULL), m_quadGeo(NULL), m_nbPoints(0), m_nbEdges(0), m_nbFaces(0)/*, m_nbVolumes(0)*/
{
}

template<class DataTypes>
EulerianFluidModel<DataTypes>::~EulerianFluidModel()
{
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->Inherit::parse(arg);
    if (arg->getAttribute("viscosity")) this->setViscosity((Real)atof(arg->getAttribute("viscosity")));
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::init()
{
    if (this->m_mstate == NULL)
    {
        this->m_mstate = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>* >(this->getContext()->getMechanicalState());
    }
    assert(this->m_mstate != NULL);

    m_topology = dynamic_cast<topology::MeshTopology*>(this->m_mstate->getContext()->getMeshTopology());

    //at present only for 2D
    assert(!m_topology->isVolume() && m_topology->isSurface());

    //judge the type of mesh
    sofa::component::topology::RegularGridTopology* _t = dynamic_cast<sofa::component::topology::RegularGridTopology*>(m_topology);
    if(_t == m_topology)
        m_meshType = RegularQuadMesh;
    else
    {
        if(m_topology->getNbQuads() == 0 && m_topology->getNbTriangles() != 0 )
            m_meshType = TriangleMesh;
        else if(m_topology->getNbTriangles() == 0 && m_topology->getNbQuads() != 0)
            m_meshType = QuadMesh;
        else
            std::cerr << "WARNING: Unsolvable mesh type" << endl;
    }

    std::cout << "MeshType : ";
    switch(m_meshType)
    {
    case TriangleMesh:
        std::cout << "TriangleMesh" << endl;
        break;

    case QuadMesh:
        std::cout << "QuadMesh" << endl;
        break;
    case RegularQuadMesh:
        std::cout << "RegularQuadMesh" << endl;
        break;
    }


    //get geometry of the mesh
    switch(m_meshType)
    {
    case TriangleMesh:
        this->getContext()->get(m_triGeo);
        if (m_triGeo == NULL)
            std::cerr << "WARNING. EulerianFluidModel has no binding TriangleSetGeomet" <<endl;
        break;

    case QuadMesh:
    case RegularQuadMesh:
        this->getContext()->get(m_quadGeo);
        if (m_quadGeo == NULL)
            std::cerr << "WARNING. EulerianFluidModel has no binding QuadSetGeomet" <<endl;
        break;
    }


    //initialize numbers of elements
    m_nbPoints = m_topology->getNbPoints();
    m_nbEdges = m_topology->getNbEdges();
    if (m_meshType == TriangleMesh)
        m_nbFaces = m_topology->getNbTriangles();
    else
        m_nbFaces = m_topology->getNbQuads();
    //m_nbVolumes = 0;

    cout << "Number of nodes: " << m_nbPoints << endl;
    cout << "Number of edges: " << m_nbEdges << endl;
    cout << "Number of faces: " << m_nbFaces << endl;
    cout << "Euler-Poincarre formula: " << m_nbPoints - m_nbEdges + m_nbFaces << endl;

    //initialize m_pInfo, m_eInfo, m_fInfo, m_bdEdgeInfo, m_bdPointInfo
    computeElementInformation();

    //initialize dual faces
    computeDualFaces();

    //calculate Operators
    computeOperators();

    //initialize boundary constraints
    setBdConstraints(m_bdXmin1.getValue(), m_bdXmax1.getValue(), m_bdYmin1.getValue(), m_bdYmax1.getValue(), m_bdZmin1.getValue(), m_bdZmax1.getValue(), m_bdValue1.getValue());
    setBdConstraints(m_bdXmin2.getValue(), m_bdXmax2.getValue(), m_bdYmin2.getValue(), m_bdYmax2.getValue(), m_bdZmin2.getValue(), m_bdZmax2.getValue(), m_bdValue2.getValue());

    //initialize project matrices
    computeProjectMats();

    //initialize the size of state variables
    m_vorticity.ReSize(m_nbPoints + m_bdEdgeInfo.size());
    m_flux.ReSize(m_nbEdges);

    m_vels.resize(m_nbFaces);
    m_bkCenters.resize(m_nbFaces);
    m_bkVels.resize(m_nbFaces);

    //initialize the size of visualization variables
    m_pInfo.m_values.resize(m_nbPoints);
    m_fInfo.m_vectors.resize(m_nbFaces);

    //save data for testing
    //std::cout<<"save()"<<endl;
    //saveMeshData();
    //saveOperators();

    //set boudary contditions and intial value
    setInitialVorticity();
    addForces();
    //saveVorticity();
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::updatePosition(double dt)
{
    static double time_LE = 0.0;
    static double time_BT = 0.0;

    ctime_t startTime = CTime::getTime();
    // Omega => Phi
    //std::cout << "Omega => Phi" << endl;
    calcPhi(false);
    //savePhi();

    // Phi => U
    //std::cout << "Phi => U" << endl;
    calcFlux();
    //saveFlux();
    ctime_t endTime = CTime::getTime();
    time_LE += endTime - startTime;
//	std::cout << "time_linear_equations = " << (endTime - startTime)/1e6 << endl;

    // U => v
    //std::cout << "U => v" << endl;
    calcVelocity();
    //saveVelocity();

    // v = > Omega
    startTime = CTime::getTime();
    //std::cout << "v = > Omega" << endl;
    backtrack(dt);
    calcVorticity();
    //saveVorticity();
    endTime = CTime::getTime();
    time_BT += endTime - startTime;
//	std::cout << "time_backtrack = " << (endTime - startTime)/1e6 << endl;

    //add Forces
    if(m_bAddForces.getValue())
    {
        addForces();
        m_bAddForces.setValue(false);
    }
    //saveVorticity();
}


template<class DataTypes>
void EulerianFluidModel<DataTypes>::draw()
{
    assert(this->m_mstate);

    if (m_topology != NULL)
    {
        normalizeDisplayValues();

        switch(m_meshType)
        {
        case TriangleMesh:
            // draw vorticity
            if(getContext()->getShowBehaviorModels() && m_bDisplayVorticity.getValue())
            {
                glDisable(GL_LIGHTING);
                glShadeModel(GL_SMOOTH);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

                for(FaceID i = 0; i < m_nbFaces; ++i)
                {
                    glBegin(GL_TRIANGLES);
                    Triangle face = m_topology->getTriangle(i);
                    for(PointID j = 0; j < face.size(); ++j)
                    {
                        if(fabs(m_pInfo.m_values[face[j]]) < 0.005)
                            glColor3f(0.0, 0.0, 1.0);
                        else
                        {
                            if(m_pInfo.m_values[face[j]] > 0)
                                glColor3f(m_pInfo.m_values[face[j]], 0.0, 1.0 - m_pInfo.m_values[face[j]]);
                            else
                                glColor3f(0.0, -m_pInfo.m_values[face[j]], 1.0 + m_pInfo.m_values[face[j]]);
                        }
                        glVertex3f(m_topology->getPX(face[j]), m_topology->getPY(face[j]), m_topology->getPZ(face[j]));
                    }
                    glEnd();
                }
            }
            break;
        case QuadMesh:
        case RegularQuadMesh:

            // draw vorticity
            if(getContext()->getShowBehaviorModels() && m_bDisplayVorticity.getValue())
            {
                glDisable(GL_LIGHTING);
                glShadeModel(GL_SMOOTH);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

                for(FaceID i = 0; i < m_nbFaces; ++i)
                {
                    glBegin(GL_POLYGON);
                    Quad face = m_topology->getQuad(i);
                    for(PointID j = 0; j < face.size(); ++j)
                    {
                        if(fabs(m_pInfo.m_values[face[j]]) < 0.005)
                            glColor3f(0.0, 0.0, 1.0);
                        else
                        {
                            if(m_pInfo.m_values[face[j]] > 0)
                                glColor3f(m_pInfo.m_values[face[j]], 0.0, 1.0 - m_pInfo.m_values[face[j]]);
                            else
                                glColor3f(0.0, -m_pInfo.m_values[face[j]], 1.0 + m_pInfo.m_values[face[j]]);
                        }
                        glVertex3f(m_topology->getPX(face[j]), m_topology->getPY(face[j]), m_topology->getPZ(face[j]));
                    }
                    glEnd();
                }
            }
            break;
        default:
            break;
        }

        // draw velocity
        if(getContext()->getShowBehaviorModels() && m_bDisplayVelocity.getValue())
        {
            glDisable(GL_LIGHTING);
            glLineWidth(2.0f);
            //velocity at face centers
            for(FaceID i = 0; i < m_nbFaces; ++i)
            {
                Coord pt = m_fInfo.m_centers[i] + m_fInfo.m_vectors[i];
                glBegin(GL_LINES);
                glColor3f(0.0, 0.0, 0.0);
                glVertex3f(m_fInfo.m_centers[i][0], m_fInfo.m_centers[i][1], m_fInfo.m_centers[i][2]);
                glColor3f(1.0, 1.0, 1.0);
                glVertex3f(pt[0], pt[1], pt[2]);
                glEnd();
            }
            //velocity at boundary points
            for(BoundaryPointIterator it = m_bdPointInfo.begin(); it != m_bdPointInfo.end(); ++it)
            {
                Coord pt1(m_topology->getPX(it->first), m_topology->getPY(it->first), m_topology->getPZ(it->first));
                Coord pt2 = pt1 + it->second.m_vector;
                glBegin(GL_LINES);
                glColor3f(0.0, 0.0, 0.0);
                glVertex3f(pt1[0], pt1[1], pt1[2]);
                glColor3f(1.0, 1.0, 1.0);
                glVertex3f(pt2[0], pt2[1], pt2[2]);
                glEnd();
            }
            //velocity at boundary edge centers
            for(BoundaryEdgeIterator it = m_bdEdgeInfo.begin(); it != m_bdEdgeInfo.end(); ++it)
            {
                Coord pt = m_eInfo.m_centers[it->first] + it->second.m_vector;
                glBegin(GL_LINES);
                glColor3f(0.0, 0.0, 0.0);
                glVertex3f(m_eInfo.m_centers[it->first][0], m_eInfo.m_centers[it->first][1], m_eInfo.m_centers[it->first][2]);
                glColor3f(1.0, 1.0, 1.0);
                glVertex3f(pt[0], pt[1], pt[2]);
                glEnd();
            }
        }
        // draw constraint boudary
        if(getContext()->getShowBehaviorModels() && m_bDisplayBoundary.getValue())
        {
            glDisable(GL_LIGHTING);
            glLineWidth(3.0f);

            // display obtuse triangles
            if(m_meshType == TriangleMesh)
                for(unsigned int i = 0; i < m_fInfo.m_obtuseTris.size(); ++i)
                {
                    glBegin(GL_TRIANGLES);
                    Triangle face = m_topology->getTriangle(m_fInfo.m_obtuseTris[i]);
                    for(PointID j = 0; j < face.size(); ++j)
                    {
                        glColor3f(1.0, 1.0, 1.0);
                        glVertex3f(m_topology->getPX(face[j]), m_topology->getPY(face[j]), m_topology->getPZ(face[j]));
                    }
                    glEnd();
                }

            for(BoundaryEdgeIterator it = m_bdEdgeInfo.begin(); it != m_bdEdgeInfo.end(); ++it)
            {
                Edge e = m_topology->getEdge(it->first);
                glBegin(GL_LINES);
                if(it->second.m_bdConstraint > 0.0)
                    glColor3f(0.0, 0.0, 1.0);
                else
                {
                    if(it->second.m_bdConstraint < 0.0)
                        glColor3f(0.0, 1.0, 0.0);
                    else
                        glColor3f(1.0, 0.0, 0.0);
                }
                glVertex3f(m_topology->getPX(e[0]), m_topology->getPY(e[0]), m_topology->getPZ(e[0]));
                glVertex3f(m_topology->getPX(e[1]), m_topology->getPY(e[1]), m_topology->getPZ(e[1]));
                glEnd();
            }
        }
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeElementInformation()
{
    m_pInfo.m_isBoundary.resize(m_nbPoints);
    m_pInfo.m_isBoundary.fill(false);

    m_eInfo.m_isBoundary.resize(m_nbEdges);
    m_eInfo.m_isBoundary.fill(false);
    m_eInfo.m_lengths.clear();
    m_eInfo.m_lengths.resize(m_nbEdges);
    m_eInfo.m_unitTangentVectors.clear();
    m_eInfo.m_unitTangentVectors.resize(m_nbEdges);
    m_eInfo.m_centers.clear();
    m_eInfo.m_centers.resize(m_nbEdges);

    m_fInfo.m_isBoundary.resize(m_nbFaces);
    m_fInfo.m_isBoundary.fill(false);
    m_fInfo.m_centers.clear();
    m_fInfo.m_centers.resize(m_nbFaces);

    m_bdEdgeInfo.clear();
    m_bdPointInfo.clear();

    Coord vel(0.0, 0.0, 0.0);
    Coord vec(0.0, 0.0, 0.0);

    switch(m_meshType)
    {
    case TriangleMesh:
        //compute m_fInfo.m_centers
        if(m_centerType == Barycenter)
            for(FaceID i = 0; i < m_nbFaces; ++i)
                m_fInfo.m_centers[i] = m_triGeo->computeTriangleCenter(i);
        else
            for(FaceID i = 0; i < m_nbFaces; ++i)
                m_fInfo.m_centers[i] = m_triGeo->computeTriangleCircumcenter(i);

        for(EdgeID i = 0; i < m_nbEdges; ++i)
        {
            const Edge e = m_topology->getEdge(i);
            const EdgeFaces& eFaces = m_topology->getTriangleEdgeShell(i);
            //compute m_eInfo.m_lengths
            m_eInfo.m_lengths[i] = m_triGeo->computeEdgeLength(i);
            //compute m_eInfo.m_unitTangentVectors
            m_eInfo.m_unitTangentVectors[i] = m_triGeo->computeEdgeDirection(i) / m_eInfo.m_lengths[i];
            //compute m_eInfo.m_centers
            m_eInfo.m_centers[i] = m_triGeo->computeEdgeCenter(i);

            //compute m_pInfo.m_isBoundary, m_eInfo.m_isBoundary, m_fInfo.m_isBoundary
            //initialize m_bdEdgeInfo, m_bdPointInfo
            if(eFaces.size() == 1)
            {
                m_pInfo.m_isBoundary[e[0]] = true;
                m_pInfo.m_isBoundary[e[1]] = true;
                m_eInfo.m_isBoundary[i] = true;
                m_fInfo.m_isBoundary[eFaces[0]] = true;

                vec = (m_eInfo.m_centers[i] - m_fInfo.m_centers[eFaces[0]]) * m_topology->computeRelativeOrientationInTri(e[0], e[1], eFaces[0]);
                vec = vec / vec.norm();
                BoundaryEdgeInformation eInfo(0.0, vel, vec);
                m_bdEdgeInfo.insert(make_pair(i, eInfo));

                BoundaryPointInformation pInfo(vel);
                m_bdPointInfo.insert(make_pair(e[0], pInfo));
                m_bdPointInfo.insert(make_pair(e[1], pInfo));
            }
        }
        break;

    case QuadMesh:
        //compute m_fInfo.m_centers
        if(m_centerType == Barycenter)
            for(FaceID i = 0; i < m_nbFaces; ++i)
                m_fInfo.m_centers[i] = m_quadGeo->computeQuadCenter(i);
        else
            std::cerr << "At present, not implemented: computationof circumcenters of quads" << endl;

        for(EdgeID i = 0; i < m_nbEdges; ++i)
        {
            const Edge e = m_topology->getEdge(i);
            const EdgeFaces& eFaces = m_topology->getQuadEdgeShell(i);

            //compute m_eInfo.m_lengths
            m_eInfo.m_lengths[i] = m_quadGeo->computeEdgeLength(i);
            //compute m_eInfo.m_unitTangentVectors
            m_eInfo.m_unitTangentVectors[i] = m_quadGeo->computeEdgeDirection(i) / m_eInfo.m_lengths[i];
            //compute m_eInfo.m_centers
            m_eInfo.m_centers[i] = m_quadGeo->computeEdgeCenter(i);

            //compute m_pInfo.m_isBoundary, m_eInfo.m_isBoundary, m_fInfo.m_isBoundary
            //initialize m_bdEdgeInfo, m_bdPointInfo
            if(eFaces.size() == 1)
            {
                m_pInfo.m_isBoundary[e[0]] = true;
                m_pInfo.m_isBoundary[e[1]] = true;
                m_eInfo.m_isBoundary[i] = true;
                m_fInfo.m_isBoundary[eFaces[0]] = true;

                vec = (m_eInfo.m_centers[i] - m_fInfo.m_centers[eFaces[0]]) * m_topology->computeRelativeOrientationInQuad(e[0], e[1], eFaces[0]);
                vec = vec / vec.norm();
                BoundaryEdgeInformation eInfo(0.0, vel, vec);
                m_bdEdgeInfo.insert(make_pair(i, eInfo));

                BoundaryPointInformation pInfo(vel);
                m_bdPointInfo.insert(make_pair(e[0], pInfo));
                m_bdPointInfo.insert(make_pair(e[1], pInfo));
            }
        }
        break;

    case RegularQuadMesh:
        //compute m_fInfo.m_centers
        for(FaceID i= 0; i < m_nbFaces; ++i)
            m_fInfo.m_centers[i] = m_quadGeo->computeQuadCenter(i);

        for(EdgeID i = 0; i < m_nbEdges; ++i)
        {
            const Edge e = m_topology->getEdge(i);
            const EdgeFaces& eFaces = m_topology->getQuadEdgeShell(i);

            //compute m_eInfo.m_lengths
            m_eInfo.m_lengths[i] = m_quadGeo->computeEdgeLength(i);
            //compute m_eInfo.m_unitTangentVectors
            m_eInfo.m_unitTangentVectors[i] = m_quadGeo->computeEdgeDirection(i) / m_eInfo.m_lengths[i];
            //compute m_eInfo.m_centers
            m_eInfo.m_centers[i] = m_quadGeo->computeEdgeCenter(i);

            //compute m_pInfo.m_isBoundary, m_eInfo.m_isBoundary, m_fInfo.m_isBoundary
            //initialize m_bdEdgeInfo, m_bdPointInfo
            if(eFaces.size() == 1)
            {
                m_pInfo.m_isBoundary[e[0]] = true;
                m_pInfo.m_isBoundary[e[1]] = true;
                m_eInfo.m_isBoundary[i] = true;
                m_fInfo.m_isBoundary[eFaces[0]] = true;

                vec = (m_eInfo.m_centers[i] - m_fInfo.m_centers[eFaces[0]]) * m_topology->computeRelativeOrientationInQuad(e[0], e[1], eFaces[0]);
                vec = vec / vec.norm();
                BoundaryEdgeInformation eInfo(0.0, vel, vec);
                m_bdEdgeInfo.insert(make_pair(i, eInfo));

                BoundaryPointInformation pInfo(vel);
                m_bdPointInfo.insert(make_pair(e[0], pInfo));
                m_bdPointInfo.insert(make_pair(e[1], pInfo));
            }
        }
        break;

    default:
        break;
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeDualFaces()
{
    assert(!m_pInfo.m_isBoundary.empty() && !m_eInfo.m_centers.empty());
    m_pInfo.m_dualFaces.clear();
    m_pInfo.m_dualFaceVertexNormals.clear();
    m_pInfo.m_dualFaceVertexVolumes.clear();

    m_pInfo.m_dualFaces.resize(m_nbPoints);
    m_pInfo.m_dualFaceVertexNormals.resize(m_nbPoints);
    m_pInfo.m_dualFaceVertexVolumes.resize(m_nbPoints);

    if(m_centerType == Barycenter)
        std::cerr << "At present, not implemented: ComputeDualFaces() for barycenter" <<endl;
    else
    {
        switch(m_meshType)
        {
        case TriangleMesh:

            for(PointID i = 0; i < m_nbPoints; ++i)
            {
                Coord p = m_triGeo->getPointPosition(i);
                VertexEdges pEdges = m_topology->getOrientedEdgeVertexShell(i);
                VertexFaces pFaces = m_topology->getOrientedTriangleVertexShell(i);

                if(m_pInfo.m_isBoundary[i])
                    //for a boundary point, dual face is not only composed of face centers, but also the point itself and two edge centers
                {
                    Normal norm1, norm2, norm3, norm4;
                    Edge e;
                    EdgeFaces eFaces;

                    // one of the edge center
                    m_pInfo.m_dualFaces[i].push_back(m_eInfo.m_centers[pEdges.back()]);
                    e = m_topology->getEdge(pEdges.back());
                    norm1 = m_triGeo->computeEdgeDirection(pEdges.back());
                    if(e[1] == i)
                        norm1 = -norm1;
                    eFaces = m_topology->getTriangleEdgeShell(pEdges.back());
                    norm2 = m_fInfo.m_centers[eFaces[0]] - m_eInfo.m_centers[pEdges.back()];

                    // point
                    m_pInfo.m_dualFaces[i].push_back(p);

                    // the other edge center
                    m_pInfo.m_dualFaces[i].push_back(m_eInfo.m_centers[pEdges.front()]);
                    e = m_topology->getEdge(pEdges.front());
                    eFaces = m_topology->getTriangleEdgeShell(pEdges.front());
                    norm3 = m_fInfo.m_centers[eFaces[0]] - m_eInfo.m_centers[pEdges.front()];
                    norm4 = m_triGeo->computeEdgeDirection(pEdges.back());
                    if(e[1] == i)
                        norm4 = -norm4;

                    VertexNormal vNorms;
                    //VertexNormal for eCenter1
                    vNorms[0] = norm1, vNorms[1] = norm2;
                    m_pInfo.m_dualFaceVertexNormals[i].push_back(vNorms);
                    m_pInfo.m_dualFaceVertexVolumes[i].push_back((vNorms[0].cross(vNorms[1]).norm()));
                    //VertexNormal for p
                    vNorms[0] = norm2, vNorms[1] = norm3;
                    m_pInfo.m_dualFaceVertexNormals[i].push_back(vNorms);
                    m_pInfo.m_dualFaceVertexVolumes[i].push_back((vNorms[0].cross(vNorms[1]).norm()));
                    //VertexNormal for eCenter2
                    vNorms[0] = norm3, vNorms[1] = norm4;
                    m_pInfo.m_dualFaceVertexNormals[i].push_back(vNorms);
                    m_pInfo.m_dualFaceVertexVolumes[i].push_back((vNorms[0].cross(vNorms[1]).norm()));
                }

                //dual face is composed of face centers
                for(FaceID j = 0; j < pFaces.size(); ++j)
                {
                    m_pInfo.m_dualFaces[i].push_back(m_fInfo.m_centers[pFaces[j]]);

                    VertexNormal vNorms;
                    vNorms[0] = m_triGeo->computeEdgeDirection(pEdges[j]);
                    Edge e = m_topology->getEdge(pEdges[j]);
                    if(e[1] == i)
                        vNorms[0] = -vNorms[0];
                    vNorms[1] = m_triGeo->computeEdgeDirection(pEdges[(j+1)%pEdges.size()]);
                    e = m_topology->getEdge(pEdges[(j+1)%pEdges.size()]);
                    if(e[1] == i)
                        vNorms[1] = -vNorms[1];

                    m_pInfo.m_dualFaceVertexNormals[i].push_back(vNorms);
                    m_pInfo.m_dualFaceVertexVolumes[i].push_back((vNorms[0].cross(vNorms[1]).norm()));
                }
            }
            break;

        case QuadMesh:
        case RegularQuadMesh:
            for(PointID i = 0; i < m_nbPoints; ++i)
            {
                Coord p = m_quadGeo->getPointPosition(i);
                VertexEdges pEdges = m_topology->getOrientedEdgeVertexShell(i);
                VertexFaces pFaces = m_topology->getOrientedQuadVertexShell(i);

                if(m_pInfo.m_isBoundary[i])
                    //for a boundary point, dual face is not only composed of face centers, but also the point itself and edgecenters
                {
                    Normal norm1, norm2, norm3, norm4;
                    Edge e;
                    EdgeFaces eFaces;

                    // one of the edge center
                    m_pInfo.m_dualFaces[i].push_back(m_eInfo.m_centers[pEdges.back()]);
                    e = m_topology->getEdge(pEdges.back());
                    norm1 = m_quadGeo->computeEdgeDirection(pEdges.back());
                    if(e[1] == i)
                        norm1 = -norm1;
                    eFaces = m_topology->getQuadEdgeShell(pEdges.back());
                    norm2 = m_fInfo.m_centers[eFaces[0]] - m_eInfo.m_centers[pEdges.back()];

                    // point
                    m_pInfo.m_dualFaces[i].push_back(p);

                    // the other edge center
                    m_pInfo.m_dualFaces[i].push_back(m_eInfo.m_centers[pEdges.front()]);
                    e = m_topology->getEdge(pEdges.front());
                    eFaces = m_topology->getQuadEdgeShell(pEdges.front());
                    norm3 = m_fInfo.m_centers[eFaces[0]] - m_eInfo.m_centers[pEdges.front()];
                    norm4 = m_quadGeo->computeEdgeDirection(pEdges.back());
                    if(e[1] == i)
                        norm4 = -norm4;

                    VertexNormal vNorms;
                    //VertexNormal for eCenter1
                    vNorms[0] = norm1, vNorms[1] = norm2;
                    m_pInfo.m_dualFaceVertexNormals[i].push_back(vNorms);
                    m_pInfo.m_dualFaceVertexVolumes[i].push_back((vNorms[0].cross(vNorms[1]).norm()));
                    //VertexNormal for p
                    vNorms[0] = norm2, vNorms[1] = norm3;
                    m_pInfo.m_dualFaceVertexNormals[i].push_back(vNorms);
                    m_pInfo.m_dualFaceVertexVolumes[i].push_back((vNorms[0].cross(vNorms[1]).norm()));
                    //VertexNormal for eCenter2
                    vNorms[0] = norm3, vNorms[1] = norm4;
                    m_pInfo.m_dualFaceVertexNormals[i].push_back(vNorms);
                    m_pInfo.m_dualFaceVertexVolumes[i].push_back((vNorms[0].cross(vNorms[1]).norm()));
                }

                //dual face is composed of face centers
                for(FaceID j = 0; j < pFaces.size(); ++j)
                {
                    m_pInfo.m_dualFaces[i].push_back(m_fInfo.m_centers[pFaces[j]]);

                    VertexNormal vNorms;
                    vNorms[0] = m_quadGeo->computeEdgeDirection(pEdges[j]);
                    Edge e = m_topology->getEdge(pEdges[j]);
                    if(e[1] == i)
                        vNorms[0] = -vNorms[0];
                    vNorms[1] = m_quadGeo->computeEdgeDirection(pEdges[(j+1)%pEdges.size()]);
                    e = m_topology->getEdge(pEdges[(j+1)%pEdges.size()]);
                    if(e[1] == i)
                        vNorms[1] = -vNorms[1];

                    m_pInfo.m_dualFaceVertexNormals[i].push_back(vNorms);
                    m_pInfo.m_dualFaceVertexVolumes[i].push_back((vNorms[0].cross(vNorms[1]).norm()));
                }
            }
            break;
        }
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeOperators()
{
    //FullVector, resize and then set all values to be zero
    star0.resize(m_nbPoints);
    star1.resize(m_nbEdges);
    star2.resize(m_nbFaces);
    star0.clear();
    star1.clear();
    star2.clear();

    //SparseMatrix, delete all the elements and then resize
    d0.clear();
    d1.clear();
    curl.clear();
    laplace.clear();
    d0.resize(m_nbEdges, m_nbPoints);
    d1.resize(m_nbFaces, m_nbEdges);
    curl.resize(m_nbPoints, m_nbEdges);
    laplace.resize(m_nbPoints, m_nbPoints);

    switch(m_meshType)
    {
    case TriangleMesh:
        computeDerivativesForTriMesh();
        computeHodgeStarsForTriMesh();
        break;

    case QuadMesh:
    case RegularQuadMesh:
        computeDerivativesForQuadMesh();
        computeHodgeStarsForQuadMesh();
        break;
    }

    //calculate curl
    sofa::component::linearsolver::SparseMatrix<int>::LineIterator row1;
    sofa::component::linearsolver::SparseMatrix<int>::LElementIterator ele1;
    for(row1 = d0.begin(); row1 != d0.end(); ++row1)
    {
        int j = row1->first;
        for(ele1 = row1->second.begin(); ele1 != row1->second.end(); ++ele1)
        {
            int i = ele1->first;
            curl.set(i, j, ele1->second * star1.element(j));
        }
    }

    //calculate laplace
    sofa::component::linearsolver::SparseMatrix<double>::LineIterator row2;
    sofa::component::linearsolver::SparseMatrix<double>::LElementIterator ele2;
    for(row2 = curl.begin(); row2 != curl.end(); ++row2)
    {
        int i = row2->first;
        for(int j = 0; j <= i; ++j)
            //laplace is symtric
        {
            double value = 0;
            for(ele2 = row2->second.begin(); ele2 != row2->second.end(); ++ele2)
            {
                int k = ele2->first;
                value += ele2->second * d0.element(k, j);
            }
            laplace.set(i, j, value);
            laplace.set(j, i, value);
        }
    }

    m_d0.ReSize(m_nbEdges, m_nbPoints);
    m_d0 = 0.0;
    for(EdgeID i = 0; i < m_nbEdges; ++i)
        for(PointID j = 0 ; j < m_nbPoints; ++j)
            m_d0.element(i, j) = d0.element(i, j);

    unsigned int nb_constraints = m_bdEdgeInfo.size();
    m_laplace.ReSize(m_nbPoints + nb_constraints, m_nbPoints + nb_constraints);
    m_laplace = 0.0;
    for(unsigned int i = 0; i < m_nbPoints; ++i)
    {
        if(m_pInfo.m_isBoundary[i])
            m_laplace.element(i, i) = 1.0;
        else
            for(unsigned int j = 0; j < m_nbPoints; ++j)
            {
                m_laplace.element(i, j) = laplace.element(i, j);
            }
    }

    BoundaryEdgeIterator it = m_bdEdgeInfo.begin();
    sofa::component::linearsolver::SparseMatrix<int>::LElementIterator ele3;
    for(unsigned int i = 0; i < nb_constraints; ++i, ++it)
    {
        for (ele3 = d0[it->first].begin(); ele3 != d0[it->first].end(); ++ele3)
        {
            //  ele3->first = index of the column containing the first non-zero value of d0
            //  ele3->second = value of the non-zero element at this index
            m_laplace.element(m_nbPoints+i, ele3->first) = ele3->second;
            m_laplace.element(ele3->first, m_nbPoints+i) = ele3->second;
        }
    }

    m_laplace_inv = m_laplace.i();
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeDerivativesForTriMesh()
{
    //calculate d0
    for(EdgeID i = 0; i < m_nbEdges; ++i)
    {
        const Edge e = m_topology->getEdge(i);
        d0.set(i, e[0], -1);
        d0.set(i, e[1], 1);
    }

    //calculate d1
    for(FaceID i = 0; i < m_nbFaces; ++i)
    {
        const Triangle f = m_topology->getTriangle(i);
        const TriangleEdges& fEdges = m_topology->getEdgeTriangleShell(i);
        for(EdgeID j = 0; j < fEdges.size(); ++j)
        {
            const Edge e = m_topology->getEdge(fEdges[j]);
            int k = 0;
            while(e[0] != f[k])
                ++k;
            if(e[1] == f[(k+1)%fEdges.size()])
                d1.set(i, fEdges[j], 1);
            else
                d1.set(i, fEdges[j], -1);
        }
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeDerivativesForQuadMesh()
{
    //calculate d0
    for(EdgeID i = 0; i < m_nbEdges; ++i)
    {
        const Edge e = m_topology->getEdge(i);
        d0.set(i, e[0], -1);
        d0.set(i, e[1], 1);
    }

    //calculate d1
    for(FaceID i = 0; i < m_nbFaces; ++i)
    {
        const Quad f = m_topology->getQuad(i);
        const QuadEdges& fEdges = m_topology->getEdgeQuadShell(i);
        for(EdgeID j = 0; j < fEdges.size(); ++j)
        {
            const Edge e = m_topology->getEdge(fEdges[j]);
            int k = 0;
            while(e[0] != f[k])
                ++k;
            if(e[1] == f[(k+1)%fEdges.size()])
                d1.set(i, fEdges[j], 1);
            else
                d1.set(i, fEdges[j], -1);
        }
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeHodgeStarsForTriMesh()
{
    assert(!m_eInfo.m_centers.empty() && !m_fInfo.m_centers.empty());
    m_fInfo.m_obtuseTris.clear();

    Coord p, m, c, c0, c1;
    double dualEdgeLength, dualFaceArea;

    //calculate star2
    for(FaceID i = 0; i < m_nbFaces; ++i)
        star2.set(i, 1.0 / m_triGeo->computeTriangleArea(i));

    //calculate star1
    for(EdgeID i = 0; i < m_nbEdges; ++i)
    {
        const Edge e = m_topology->getEdge(i);
        m = m_eInfo.m_centers[i];

        dualEdgeLength = 0.0;
        const EdgeFaces& eFaces = m_topology->getTriangleEdgeShell(i);
        for(FaceID j = 0; j < eFaces.size(); ++j)
        {
            c = m_fInfo.m_centers[eFaces[j]];
            Triangle f = m_topology->getTriangle(eFaces[j]);
            PointID k = 0;
            while(f[k] == e[0] || f[k] == e[1])
                ++k;
            assert(k < f.size() && f[k] != e[0] && f[k] != e[1]);
            if(m_triGeo->computeAngle(f[k], e[0], e[1]) == sofa::component::topology::PointSetGeometryAlgorithms<DataTypes> :: ACUTE)
                dualEdgeLength += (m - c).norm();
            else
            {
                dualEdgeLength -= (m - c).norm();
                m_fInfo.m_obtuseTris.push_back(eFaces[j]);
            }
        }
        //star1.set(i, dualEdgeLength / m_triGeo->computeEdgeLength(i));
        star1.set(i, dualEdgeLength / m_eInfo.m_lengths[i]);
    }

    //calculate star0
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        dualFaceArea = 0;
        p = m_triGeo->getPointPosition(i);
        const VertexEdges vEdges = m_topology->getOrientedEdgeVertexShell(i);
        for(EdgeID j = 0; j < vEdges.size(); ++j)
        {
            const EdgeFaces& eFaces = m_topology->getTriangleEdgeShell(vEdges[j]);
            c0 = m_fInfo.m_centers[eFaces[0]];
            if(!m_eInfo.m_isBoundary[vEdges[j]])
                c1 = m_fInfo.m_centers[eFaces[1]];
            else
                c1 = m_eInfo.m_centers[vEdges[j]];
            dualFaceArea += 0.5 * (((p - c0).cross(p - c1)).norm());
        }
        star0.set(i, dualFaceArea);
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeHodgeStarsForQuadMesh()
{
    assert(!m_eInfo.m_centers.empty() && !m_fInfo.m_centers.empty());

    Coord p, c0, c1;
    double dualEdgeLength, dualFaceArea;

    //calculate star2
    for(FaceID i = 0; i < m_nbFaces; ++i)
        star2.set(i, 1.0 / m_quadGeo->computeQuadArea(i));

    //calculate star1
    for(EdgeID i = 0; i < m_nbEdges; ++i)
    {
        const Edge e = m_topology->getEdge(i);
        const EdgeFaces& eFaces = m_topology->getQuadEdgeShell(i);
        c0 = m_fInfo.m_centers[eFaces[0]];
        if(!m_eInfo.m_isBoundary[i])
            c1 = m_fInfo.m_centers[eFaces[1]];
        else
            c1 = m_eInfo.m_centers[i];
        dualEdgeLength = (c0 - c1).norm();
        //star1.set(i, dualEdgeLength / m_quadGeo->computeEdgeLength(i));
        star1.set(i, dualEdgeLength / m_eInfo.m_lengths[i]);

    }

    //calculate star0
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        dualFaceArea = 0;
        p = m_quadGeo->getPointPosition(i);
        const VertexEdges vEdges = m_topology->getOrientedEdgeVertexShell(i);
        for(EdgeID j = 0; j < vEdges.size(); ++j)
        {
            const EdgeFaces& eFaces = m_topology->getQuadEdgeShell(vEdges[j]);
            c0 = m_fInfo.m_centers[eFaces[0]];
            if(!m_eInfo.m_isBoundary[vEdges[j]])
                c1 = m_fInfo.m_centers[eFaces[1]];
            else
                c1 = m_eInfo.m_centers[vEdges[j]];
            dualFaceArea += 0.5 * (((p - c0).cross(p - c1)).norm());
        }
        star0.set(i, dualFaceArea);
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeProjectMats()
{
    assert(!m_eInfo.m_centers.empty() && !m_fInfo.m_centers.empty());

    //initialize
    m_fInfo.m_At.clear();
    m_fInfo.m_At.resize(m_nbFaces);
    m_fInfo.m_AtAInv.clear();
    m_fInfo.m_AtAInv.resize(m_nbFaces);
    for(FaceID i = 0; i < m_nbFaces; ++i)
    {
        m_fInfo.m_At[i] = 0.0;
        m_fInfo.m_AtAInv[i] = 0.0;
    }

    Coord c0, c1;
    Coord v;
    sofa::defaulttype::Vec<3,double> norm;
    int nRows, nCols;

    switch(m_meshType)
    {
    case TriangleMesh:
        nRows = 4;
        nCols = 3;
        //for each face: i
        for(FaceID i = 0; i < m_nbFaces; ++i)
        {
            NewMAT::Matrix A(nRows, nCols);
            A = 0.0;
            c0 = m_fInfo.m_centers[i];
            const TriangleEdges fEdges = m_topology->getEdgeTriangleShell(i);
            //for each edge adjacent to face i: fEdges[j]
            //for each row of mat: j
            for(EdgeID j = 0; j < fEdges.size(); ++j)
            {
                c1 = m_eInfo.m_centers[fEdges[j]];
                v = (c1 - c0) * (double)d1.element(i, fEdges[j]) * m_eInfo.m_lengths[fEdges[j]] / (c1 - c0).norm();

                Triangle f = m_topology->getTriangle(i);
                const Edge e = m_topology->getEdge(fEdges[j]);
                PointID k = 0;
                while(f[k] == e[0] || f[k] == e[1])
                    ++k;
                assert(k < f.size() && f[k] != e[0] && f[k] != e[1]);
                if(m_triGeo->computeAngle(f[k], e[0], e[1]) == sofa::component::topology::PointSetGeometryAlgorithms<DataTypes> :: OBTUSE)
                    v = -v;

                //for each col of mat: k
                for(int k = 0; k < nCols; ++k)
                    A.element(j, k) = v[k];
            }

            //calculate the equation of the plane, set the last row to be the plane eqation
            norm = m_triGeo->computeTriangleNormal(i);
            for(int k = 0; k < nCols; ++k)
                A.element(nRows-1, k) = norm[k];

            //calculate AtA.i()
            NewMAT::SymmetricMatrix AtA;
            AtA << A.t() * A;
            m_fInfo.m_At[i] = A.t();
            m_fInfo.m_AtAInv[i] = AtA.i();

        }
        break;

    case QuadMesh:
    case RegularQuadMesh:
        nRows = 5;
        nCols = 3;
        //for each face: i
        for(FaceID i = 0; i < m_nbFaces; ++i)
        {
            NewMAT::Matrix A(nRows, nCols);
            A = 0.0;
            c0 = m_fInfo.m_centers[i];
            const QuadEdges fEdges = m_topology->getEdgeQuadShell(i);
            //for each edge adjacent to i: fEdges[j]
            //for each row of mat: j
            for(EdgeID j = 0; j < fEdges.size(); ++j)
            {
                c1 = m_eInfo.m_centers[fEdges[j]];
                //v = (c1 - c0) * (double)d1.element(i, fEdges[j]) * m_quadGeo->computeEdgeLength(fEdges[j]) / (c1 - c0).norm();
                v = (c1 - c0) * (double)d1.element(i, fEdges[j]) * m_eInfo.m_lengths[fEdges[j]] / (c1 - c0).norm();

                //for each col of mat: k
                for(int k = 0; k < nCols; ++k)
                    A.element(j, k) = v[k];
            }

            //calculate the equation of the plane, set the last row to be the plane eqation
            norm = m_quadGeo->computeQuadNormal(i);
            for(int k = 0; k < nCols; ++k)
                A.element(nRows-1, k) = norm[k];

            //calculate AtA.i()
            NewMAT::SymmetricMatrix AtA;
            AtA << A.t() * A;
            m_fInfo.m_At[i] = A.t();
            m_fInfo.m_AtAInv[i] = AtA.i();
        }
        break;
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::setBdConstraints(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax, double value)
{
    for(BoundaryEdgeIterator it = m_bdEdgeInfo.begin(); it !=  m_bdEdgeInfo.end(); ++it)
    {
        if(m_eInfo.m_centers[it->first].x() > xMin && m_eInfo.m_centers[it->first].x() < xMax &&
           m_eInfo.m_centers[it->first].y() > yMin && m_eInfo.m_centers[it->first].y() < yMax &&
           m_eInfo.m_centers[it->first].z() > zMin && m_eInfo.m_centers[it->first].z() < zMax)
        {
            const EdgeFaces eFaces = (m_meshType == TriangleMesh) ?
                    m_topology->getTriangleEdgeShell(it->first) : m_topology->getQuadEdgeShell(it->first);
            it->second.m_bdConstraint = value * d1.element(eFaces[0], it->first);
        }
    }
}


template<class DataTypes>
void EulerianFluidModel<DataTypes>::setInitialVorticity()
{
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        m_vorticity.element(i) = 0.0;
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::calcVelocity()
{
    switch(m_meshType)
    {
    case TriangleMesh:
        //calculate velocity at face centers
        for(FaceID i = 0; i < m_nbFaces; ++i)
        {
            //calculate right-hand side U
            const TriangleEdges fEdges = m_topology->getEdgeTriangleShell(i);
            NewMAT::ColumnVector u(fEdges.size()+1);
            u = 0.0;
            for(EdgeID j = 0; j < fEdges.size(); ++j)
            {
                u.element(j) = m_flux.element(fEdges[j]);
            }
            //solver the project equations
            NewMAT::ColumnVector v = m_fInfo.m_AtAInv[i] * (m_fInfo.m_At[i] * u);
            m_vels[i][0] = m_harmonicVx.getValue() + v.element(0);
            m_vels[i][1] = m_harmonicVy.getValue() + v.element(1);
            m_vels[i][2] = m_harmonicVz.getValue() + v.element(2);
        }

        //calculate velocity at boundary edge centers
        for(BoundaryEdgeIterator it = m_bdEdgeInfo.begin(); it !=  m_bdEdgeInfo.end(); ++it)
        {
            //it->first: index of a boundary edge
            //it->second: info of this edge
            const EdgeFaces eFaces = m_topology->getTriangleEdgeShell(it->first);
            if(it->second.m_bdConstraint == 0.0)	//no flux can penetrate this edge
                //the velocity is along the direction of tangent vector
                it->second.m_bdVel = m_eInfo.m_unitTangentVectors[it->first] * (m_vels[eFaces[0]] * m_eInfo.m_unitTangentVectors[it->first]);
            else
                //the velocity is in the direction of flux
                it->second.m_bdVel = it->second.m_unitFluxVector * it->second.m_bdConstraint / m_eInfo.m_lengths[it->first];
        }

        //calculate velocity at boundary points
        for(BoundaryPointIterator it = m_bdPointInfo.begin(); it !=  m_bdPointInfo.end(); ++it)
        {
            //it->first: index of a boundary point
            //it->second: info of this point
            const VertexEdges vEdges = m_topology->getOrientedEdgeVertexShell(it->first);
            it->second.m_bdVel = (m_bdEdgeInfo[vEdges.front()].m_bdVel + m_bdEdgeInfo[vEdges.back()].m_bdVel) * 0.5;
        }
        break;
    case QuadMesh:
    case RegularQuadMesh:
        //calculate velocity at face centers
        for(FaceID i = 0; i < m_nbFaces; ++i)
        {
            //calculate right-hand side U
            const QuadEdges fEdges = m_topology->getEdgeQuadShell(i);
            NewMAT::ColumnVector u(fEdges.size()+1);
            u = 0.0;
            for(EdgeID j = 0; j < fEdges.size(); ++j)
            {
                u.element(j) = m_flux.element(fEdges[j]);
            }
            //solver the project equations
            NewMAT::ColumnVector v = m_fInfo.m_AtAInv[i] * (m_fInfo.m_At[i] * u);
            m_vels[i][0] = m_harmonicVx.getValue() + v.element(0);
            m_vels[i][1] = m_harmonicVy.getValue() + v.element(1);
            m_vels[i][2] = m_harmonicVz.getValue() + v.element(2);
        }

        //calculate velocity at boundary edge centers
        for(BoundaryEdgeIterator it = m_bdEdgeInfo.begin(); it !=  m_bdEdgeInfo.end(); ++it)
        {
            //it->first: index of a boundary edge
            //it->second: info of this edge
            const EdgeFaces eFaces = m_topology->getQuadEdgeShell(it->first);
            if(it->second.m_bdConstraint == 0.0)	//no flux can penetrate this edge
                //the velocity is along the direction of tangent vector
                it->second.m_bdVel = m_eInfo.m_unitTangentVectors[it->first] * (m_vels[eFaces[0]] * m_eInfo.m_unitTangentVectors[it->first]);
            else
                //the velocity is in the direction of flux
                it->second.m_bdVel = it->second.m_unitFluxVector * it->second.m_bdConstraint / m_eInfo.m_lengths[it->first];
        }

        //calculate velocity at boundary points
        for(BoundaryPointIterator it = m_bdPointInfo.begin(); it !=  m_bdPointInfo.end(); ++it)
        {
            //it->first: index of a boundary point
            //it->second: info of this point
            const VertexEdges vEdges = m_topology->getOrientedEdgeVertexShell(it->first);
            it->second.m_bdVel = (m_bdEdgeInfo[vEdges.front()].m_bdVel + m_bdEdgeInfo[vEdges.back()].m_bdVel) * 0.5;
        }
        break;
    default:
        break;
    }
}

template<class DataTypes>
unsigned int EulerianFluidModel<DataTypes>::searchFaceForTriMesh(const Coord& pt, FaceID startFace) const
{
    sofa::defaulttype::Vec<3, double> p(pt.x(), pt.y(), pt.z());

    if(startFace == sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID || startFace >= m_nbFaces)
        startFace = 0;

    sofa::helper::vector<bool> flag((int)m_nbFaces, false);
    sofa::helper::vector<FaceID> faces;

    flag[startFace] = true;
    faces.push_back(startFace);

    unsigned int temp;
    for(unsigned int i = 0; i < 20/*faces.size()*/; ++i)
    {
        if(m_triGeo->isPointInTriangle(faces[i], false, p, temp))
            return faces[i];
        else	//push back all the neighbor unsolved faces
        {
            Triangle currentFace = m_topology->getTriangle(faces[i]);

            for(PointID j = 0; j < currentFace.size(); ++j)
                //for each point of the current face
            {
                VertexFaces vFaces = m_topology->getOrientedTriangleVertexShell(currentFace[j]);
                for(FaceID k = 0; k < vFaces.size(); ++k)
                    //for each face adjacent to the point
                {
                    if(!flag.at(vFaces[k]))
                        //if it is unsolved, push back
                    {
                        flag[vFaces[k]] = true;
                        faces.push_back(vFaces[k]);
                    }
                }
            }
        }
    }

    return sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID;
}

template<class DataTypes>
unsigned int EulerianFluidModel<DataTypes>::searchDualFaceForTriMesh(const Coord & pt, const FaceID startFace) const
{
    sofa::defaulttype::Vec<3,double> p(pt.x(), pt.y(), pt.z());

    //search the face(tri/quad) in which pt is
    FaceID indFace = searchFaceForTriMesh(pt, startFace);
    if(indFace == sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID) //out of boundary
        return sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID;

    //compute the barycentric coefficients
    sofa::helper::vector<double> baryCoefs = m_triGeo->computeTriangleBarycoefs(indFace, p);
    Triangle face = m_topology->getTriangle(indFace);

    //order the baycentric coefficients from the largest to the smallest
    int order[3] = {0};
    for(int i = 1; i < 3; ++i)
    {
        if(baryCoefs[order[0]] < baryCoefs[i])
            order[0] = i;
    }
    order[1] = (order[0]+1) % 3;
    order[2] = (order[0]+2) % 3;
    if(baryCoefs[order[1]] < baryCoefs[order[2]])
        std::swap(order[1], order[2]);

    //search the dual face according to the order
    switch(m_centerType.getValue())
    {
    case Circumcenter:
    {
        VecCoord v(3);
        Deriv vp, vv1, vv2;
        v[0] = m_triGeo->getPointPosition(face[order[0]]);
        v[1] = m_triGeo->getPointPosition(face[order[1]]);
        v[2] = m_triGeo->getPointPosition(face[order[2]]);
        for(int i = 0; i < 2; ++i)
        {
            vp = pt - v[i];
            vv1 = v[(i+1)%3] - v[i];
            vv2 = v[(i+2)%3] - v[i];
            if(vp * vv1 < 0.5 * vv1.norm2() && vp * vv2 < 0.5 * vv2.norm2())
                return face[order[i]];
        }
        return face[order[2]];
    }
    break;
    case Barycenter:
        std::cerr << "At present, not implemented: searchDualFaceForTriMesh() for Barycenter" << endl;
        return sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID;
        break;

    default:
        return sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID;
    }
}

template<class DataTypes>
unsigned int EulerianFluidModel<DataTypes>::searchDualFaceForQuadMesh(const Coord & pt, PointID startDualFace) const
{
    if(startDualFace == sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID || startDualFace >= m_nbPoints)
        startDualFace = 0;

    sofa::helper::vector<bool> flag((int)m_nbPoints, false);
    sofa::helper::vector<PointID> dualFaceIDs;

    flag[startDualFace] = true;
    dualFaceIDs.push_back(startDualFace);

    sofa::defaulttype::Vec<3, double> p(pt.x(), pt.y(), pt.z());

    for(unsigned int i = 0; i < 20/*dualFaceIDs.size()*/; ++i)
    {
        DualFace dualFace = m_pInfo.m_dualFaces[dualFaceIDs[i]];
        if(!m_pInfo.m_isBoundary[dualFaceIDs[i]])
        {
            if(sofa::component::topology::is_point_in_quad(pt, dualFace[0], dualFace[1], dualFace[2], dualFace[3]))
                return dualFaceIDs[i];
        }
        else
        {
            //judge whether p is in a boundary dual face (dualFaceIDs[i])
            const VertexFaces vFaces = m_topology->getOrientedQuadVertexShell(dualFaceIDs[i]);
            for(FaceID j = 0; j < vFaces.size(); ++j)
            {
                if(m_quadGeo->isPointInQuad(vFaces[j], p))
                {
                    Quad f = m_topology->getQuad(vFaces[j]);
                    for(PointID k = 0; k < f.size(); ++k)
                    {
                        Coord v(m_topology->getPX(f[k]), m_topology->getPY(f[k]), m_topology->getPZ(f[k]));
                        Coord v1(m_topology->getPX(f[(k+1)%4]), m_topology->getPY(f[(k+1)%4]), m_topology->getPZ(f[(k+1)%4]));
                        Coord v2(m_topology->getPX(f[(k+3)%4]), m_topology->getPY(f[(k+3)%4]), m_topology->getPZ(f[(k+3)%4]));
                        Deriv vv1 = v1 - v;
                        Deriv vv2 = v2 - v;
                        Deriv vc = m_fInfo.m_centers[vFaces[j]] - v;
                        Deriv vp = pt - v;
                        if( vp*vv1 <= vc*vv1 && vp*vv2 <= vc*vv2 )
                            return f[k];
                    }
                }
            }
        }

        //push back all the neighbor unsolved dual faces
        const VertexEdges vEdges = m_topology->getOrientedEdgeVertexShell(dualFaceIDs[i]);
        for(EdgeID j = 0; j < vEdges.size(); ++j)
        {
            Edge e = m_topology->getEdge(vEdges[j]);
            if(e[0] == dualFaceIDs[i])
            {
                if(!flag[e[1]])
                {
                    flag[e[1]] = true;
                    dualFaceIDs.push_back(e[1]);
                }
            }
            else
            {
                if(!flag[e[0]])
                {
                    flag[e[0]] = true;
                    dualFaceIDs.push_back(e[0]);
                }
            }
        }

    }

    return sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID;
}

template<class DataTypes>
typename DataTypes::Deriv EulerianFluidModel<DataTypes>:: interpolateVelocity(const Coord& pt, unsigned int start)
//for TriangleMesh start = startFace
//for QuadMesh start = startDualFace (i.e.startPoint)
{
    ctime_t startTime1, endTime1;
    ctime_t startTime2, endTime2;

    startTime1 = CTime::getTime();

    Deriv vel;
    vel.fill(0);

    //search the dual face in which pt is
    PointID ind_p;
    switch(m_meshType)
    {
    case TriangleMesh:
        ind_p = searchDualFaceForTriMesh(pt, start);
        break;
    case QuadMesh:
    case RegularQuadMesh:
        ind_p = searchDualFaceForQuadMesh(pt, start);
        break;
    default:
        ind_p = sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID;
    }

    endTime1 = CTime::getTime();
    m_dTime1 += endTime1-startTime1;

    if(ind_p == sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID)
        //pt is out of boundary
    {
        return vel;
    }

    startTime2 = CTime::getTime();

    //calculate velocities on the vertices of dual face
    VecDeriv vels;
    if(m_pInfo.m_isBoundary[ind_p])
    {
        VertexEdges pEdges = m_topology->getOrientedEdgeVertexShell(ind_p);
        //the velocity at the boudary edge: pEdges.back()
        vels.push_back(m_bdEdgeInfo[pEdges.back()].m_bdVel);
        //the velocity at the boudary point: ind_p
        vels.push_back(m_bdPointInfo[ind_p].m_bdVel);
        //the velocity on the boudary edge: pEdges.front()
        vels.push_back(m_bdEdgeInfo[pEdges.front()].m_bdVel);
    }

    VertexFaces pFaces = (m_meshType == TriangleMesh) ?
            m_topology->getOrientedTriangleVertexShell(ind_p) : m_topology->getOrientedQuadVertexShell(ind_p);
    for(FaceID j = 0; j < pFaces.size(); ++j)
    {
        vels.push_back(m_vels[pFaces[j]]);
    }

    //interpolate
    int nbVertices = m_pInfo.m_dualFaces[ind_p].size();

    const double ZERO = 1e-6;
    const double ZERO2 = 1e-12;
    Coord p = (m_meshType == TriangleMesh) ?
            m_triGeo->getPointPosition(ind_p) : m_quadGeo->getPointPosition(ind_p);
    double w, wNormalize = 0;
    for(int i = 0; i < nbVertices; ++i)
    {
        Coord c = m_pInfo.m_dualFaces[ind_p].at(i);
        Deriv pc = c - p;

        if( pc.norm2() < ZERO2 ) // p == c
            return vels.at(i);

        double dis0 = pc * m_pInfo.m_dualFaceVertexNormals[ind_p].at(i)[0];
        double dis1 = pc * m_pInfo.m_dualFaceVertexNormals[ind_p].at(i)[1];
        if(dis0 < ZERO) // p is on the dual edge c[i-1]c[i]
        {
            Coord c1 = m_pInfo.m_dualFaces[ind_p].at((i+nbVertices-1)%nbVertices);
            sofa::helper::vector< double > baryCoefs = sofa::component::topology::compute_2points_barycoefs(p, c, c1);
            return vels.at(i) * baryCoefs[0] + vels.at((i+nbVertices-1)%nbVertices) * baryCoefs[1];
        }
        if(dis1 < ZERO) // p is on the dual edge c[i]c[i+1]
        {
            Coord c1 = m_pInfo.m_dualFaces[ind_p].at((i+1)%nbVertices);
            sofa::helper::vector< double > baryCoefs = sofa::component::topology::compute_2points_barycoefs(p, c, c1);
            return vels.at(i) * baryCoefs[0] + vels.at((i+1)%nbVertices) * baryCoefs[1];
        }

        w = m_pInfo.m_dualFaceVertexVolumes[ind_p].at(i) / dis0 * dis1;
        wNormalize += w;
        vel += vels.at(i) * w;
    }

    endTime2 = CTime::getTime();
    m_dTime2 += endTime2-startTime2;

    return vel / wNormalize;

}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::backtrack(double dt)
{
    m_dTime1 = 0.0;
    m_dTime2 = 0.0;


    switch(m_meshType)
    {
    case TriangleMesh:
        //backtrack face centers
        for(FaceID i = 0; i < m_nbFaces; ++i)
        {
            m_bkCenters[i] = m_fInfo.m_centers[i] + m_vels[i] * (-dt);
            m_bkVels[i] = interpolateVelocity(m_bkCenters[i], i);
        }
        ////backtrack boundary edge centers
        //for(std::map<EdgeID, BoundaryEdgeInformation>::iterator it = m_bdEdgeInfo.begin(); it !=  m_bdEdgeInfo.end(); ++it)
        //{
        //	const EdgeFaces eFaces = m_topology->getTriangleEdgeShell(it->first);
        //	it->second.m_bkECenter = m_eInfo.m_centers[it->first] + it->second.m_bdVel * (-dt);
        //	it->second.m_bkVel = interpolateVelocity(it->second.m_bkECenter, eFaces[0]);
        //}
        ////backtrack boundary points
        //for(std::map<PointID, BoundaryPointInformation>::iterator it = m_bdPointInfo.begin(); it !=  m_bdPointInfo.end(); ++it)
        //{
        //	const VertexFaces vFaces = m_topology->getOrientedTriangleVertexShell(it->first);
        //	Coord p(m_topology->getPX(it->first), m_topology->getPY(it->first), m_topology->getPZ(it->first));
        //	it->second.m_bkPoint = p + it->second.m_bdVel * (-dt);
        //	it->second.m_bkVel = interpolateVelocity(it->second.m_bkPoint, vFaces[0]);
        //}
        break;
    case QuadMesh:
    case RegularQuadMesh:
        //backtrack face centers
        for(FaceID i = 0; i < m_nbFaces; ++i)
        {
            m_bkCenters[i] = m_fInfo.m_centers[i] + m_vels[i] * (-dt);
            Quad f = m_topology->getQuad(i);
            m_bkVels[i] = interpolateVelocity(m_bkCenters[i], f[0]);
        }
        ////backtrack boundary edge centers
        //for(std::map<EdgeID, BoundaryEdgeInformation>::iterator it = m_bdEdgeInfo.begin(); it !=  m_bdEdgeInfo.end(); ++it)
        //{
        //	const Edge e = m_topology->getEdge(it->first);
        //	it->second.m_bkECenter = m_eInfo.m_centers[it->first] + it->second.m_bdVel * (-dt);
        //	it->second.m_bkVel = interpolateVelocity(it->second.m_bkECenter, e[0]);
        //}
        ////backtrack boundary points
        //for(std::map<PointID, BoundaryPointInformation>::iterator it = m_bdPointInfo.begin(); it !=  m_bdPointInfo.end(); ++it)
        //{
        //	Coord p(m_topology->getPX(it->first), m_topology->getPY(it->first), m_topology->getPZ(it->first));
        //	it->second.m_bkPoint = p + it->second.m_bdVel * (-dt);
        //	it->second.m_bkVel = interpolateVelocity(it->second.m_bkPoint, it->first);
        //}
        break;
    default:
        break;
    }

//  cout << "interpolateVelocity() => m_dTime1 = " << m_dTime1/1e6 << endl;
//	cout << "interpolateVelocity() => m_dTime2 = " << m_dTime2/1e6 << endl;
//	cout << "backtrack() => dTime = " << dTime/1e6 << endl;
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::calcVorticity()
{
    m_vorticity = 0.0;
    switch(m_meshType)
    {
    case TriangleMesh:
        //calculate inner vorticity
        for(PointID i = 0; i < m_nbPoints; ++i)
        {
            if(!m_pInfo.m_isBoundary[i])
            {
                const VertexFaces vFaces = m_topology->getOrientedTriangleVertexShell(i);
                for(FaceID j = 0; j < vFaces.size(); ++j)
                    m_vorticity.element(i) += (m_bkVels[vFaces[(j+1)%vFaces.size()]] + m_bkVels[vFaces[j]]) * (m_bkCenters[vFaces[(j+1)%vFaces.size()]] - m_bkCenters[vFaces[j]]);
                m_vorticity.element(i) *= 0.5;
            }
        }
        ////calculate boundary vorticity
        //for(std::map<PointID, BoundaryPointInformation>::iterator it = m_bdPointInfo.begin(); it !=  m_bdPointInfo.end(); ++it)
        //{
        //	const VertexEdges vEdges = m_topology->getOrientedEdgeVertexShell(it->first);
        //	const VertexFaces vFaces = m_topology->getOrientedTriangleVertexShell(it->first);
        //	m_vorticity.element(it->first) =
        //		(m_bdEdgeInfo[vEdges.back()].m_bkVel + m_bkVels[vFaces.back()]) * (m_bdEdgeInfo[vEdges.back()].m_bkECenter - m_bkCenters[vFaces.back()]) +		//fCenter_last->eCenter_last
        //		(it->second.m_bkVel + m_bdEdgeInfo[vEdges.back()].m_bkVel) * (it->second.m_bkPoint - m_bdEdgeInfo[vEdges.back()].m_bkECenter) +					//eCenter_last->p
        //		(m_bdEdgeInfo[vEdges.front()].m_bkVel + it->second.m_bkVel) * (m_bdEdgeInfo[vEdges.front()].m_bkECenter - it->second.m_bkPoint) +				//p->eCenter_first
        //		(m_bkVels[vFaces.front()] + m_bdEdgeInfo[vEdges.front()].m_bkVel) * (m_bkCenters[vFaces.front()] - m_bdEdgeInfo[vEdges.front()].m_bkECenter);	//eCenter_fisrt->fCenter_first
        //	for(FaceID i = 0; i < vFaces.size()-1; ++i)
        //	{
        //		m_vorticity.element(it->first) += (m_bkVels[vFaces[i+1]] + m_bkVels[vFaces[i]]) * (m_bkCenters[vFaces[i+1]] - m_bkCenters[vFaces[i]]);
        //	}
        //	m_vorticity.element(it->first) *= 0.5;
        //}

        break;
    case QuadMesh:
    case RegularQuadMesh:
        //calculate inner vorticity
        for(PointID i = 0; i < m_nbPoints; ++i)
        {
            if(!m_pInfo.m_isBoundary[i])
            {
                const VertexFaces vFaces = m_topology->getOrientedQuadVertexShell(i);
                for(FaceID j = 0; j < vFaces.size(); ++j)
                    m_vorticity.element(i) += (m_bkVels[vFaces[(j+1)%vFaces.size()]] + m_bkVels[vFaces[j]]) * (m_bkCenters[vFaces[(j+1)%vFaces.size()]] - m_bkCenters[vFaces[j]]);
                m_vorticity.element(i) *= 0.5 ;
            }
        }
        ////calculate boundary vorticity
        //for(std::map<PointID, BoundaryPointInformation>::iterator it = m_bdPointInfo.begin(); it !=  m_bdPointInfo.end(); ++it)
        //{
        //	const VertexEdges vEdges = m_topology->getOrientedEdgeVertexShell(it->first);
        //	const VertexFaces vFaces = m_topology->getOrientedQuadVertexShell(it->first);
        //	m_vorticity.element(it->first) =
        //		(m_bdEdgeInfo[vEdges.back()].m_bkVel + m_bkVels[vFaces.back()]) * (m_bdEdgeInfo[vEdges.back()].m_bkECenter - m_bkCenters[vFaces.back()]) +		//fCenter_last->eCenter_last
        //		(it->second.m_bkVel + m_bdEdgeInfo[vEdges.back()].m_bkVel) * (it->second.m_bkPoint - m_bdEdgeInfo[vEdges.back()].m_bkECenter) +					//eCenter_last->p
        //		(m_bdEdgeInfo[vEdges.front()].m_bkVel + it->second.m_bkVel) * (m_bdEdgeInfo[vEdges.front()].m_bkECenter - it->second.m_bkPoint) +				//p->eCenter_first
        //		(m_bkVels[vFaces.front()] + m_bdEdgeInfo[vEdges.front()].m_bkVel) * (m_bkCenters[vFaces.front()] - m_bdEdgeInfo[vEdges.front()].m_bkECenter);	//eCenter_fisrt->fCenter_first
        //	for(FaceID i = 0; i < vFaces.size()-1; ++i)
        //	{
        //		m_vorticity.element(it->first) += (m_bkVels[vFaces[i+1]] + m_bkVels[vFaces[i]]) * (m_bkCenters[vFaces[i+1]] - m_bkCenters[vFaces[i]]);
        //	}
        //	m_vorticity.element(it->first) *= 0.5;
        //}
        break;
    default:
        break;
    }

}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::addForces()
{
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        Coord p(m_topology->getPX(i), m_topology->getPY(i), m_topology->getPZ(i));
        Coord o(5.0, 5.0, 0.0);
        if((o-p).norm2() < 2.0)
            m_vorticity.element(i) += 3.0;
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::calcPhi(bool reset)
{
    if(reset)
        m_laplace_inv =  m_laplace.i();

    //set constraints, using Lagragian Multiplier method
    BoundaryEdgeIterator it = m_bdEdgeInfo.begin();
    for(unsigned int i = 0; i < m_bdEdgeInfo.size(); ++i, ++it)
    {
        m_vorticity.element(m_nbPoints+i) = it->second.m_bdConstraint;
    }
    //solve the equations
    NewMAT::Matrix phi = m_laplace_inv * m_vorticity;
    //discard extra solutions
    m_phi = phi.Rows(1, m_nbPoints);
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::calcFlux()
{
    m_flux = m_d0 * m_phi;
    //setBoundaryFlux();
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::setBoundaryFlux()
{
    assert(!m_bdEdgeInfo.empty());
    for(BoundaryEdgeIterator it = m_bdEdgeInfo.begin(); it != m_bdEdgeInfo.end(); ++it)
    {
        m_flux.element(it->first) = it->second.m_bdConstraint;
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::normalizeDisplayValues()
{
    //normailize the velocity
    //velocity at face centers
    for(FaceID i = 0; i < m_nbFaces; ++i)
        m_fInfo.m_vectors[i] =  m_vels[i] * m_visCoef1.getValue();
    //velocity at boundary points
    for(BoundaryPointIterator it = m_bdPointInfo.begin(); it !=  m_bdPointInfo.end(); ++it)
        it->second.m_vector = it->second.m_bdVel * m_visCoef1.getValue();
    //velocity at boudary edge centers
    for(BoundaryEdgeIterator it = m_bdEdgeInfo.begin(); it !=  m_bdEdgeInfo.end(); ++it)
        it->second.m_vector = it->second.m_bdVel * m_visCoef1.getValue();

    //normailize the vorticity
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        if(m_vorticity.element(i) > 0)
            m_pInfo.m_values[i] = m_visCoef2.getValue() * log(1 + m_visCoef3.getValue() * m_vorticity.element(i));
        else
            m_pInfo.m_values[i] = -m_visCoef2.getValue() * log(1 - m_visCoef3.getValue() * m_vorticity.element(i));
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::saveMeshData() const
{
    std::string str = "Points.txt";
    std::ofstream outfile(str.c_str());
    outfile << m_nbPoints << " Points" << std::endl;
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        outfile << "[" << i << "]" << "  " << m_topology->getPX(i) << "," << m_topology->getPY(i) << "," << m_topology->getPZ(i) << endl;
        outfile << "isBoundary " << m_pInfo.m_isBoundary[i] << endl;
    }
    outfile.close();
    outfile.clear();

    str = "Edges.txt";
    outfile.open(str.c_str(), std::ios::out);
    outfile << m_nbEdges << " Edges" << std::endl;
    for(EdgeID i = 0; i < m_nbEdges; ++i)
    {
        const Edge e = m_topology->getEdge(i);
        outfile << "[" << i << "]" << "  " << e[0] << "," << e[1] << endl;
        outfile << "isBoundary " << m_eInfo.m_isBoundary[i] << endl;
        outfile << "EdgeCenter " << m_eInfo.m_centers[i] << endl;
    }
    outfile.close();
    outfile.clear();

    str = "Faces.txt";
    outfile.open(str.c_str(), std::ios::out);
    outfile << m_nbFaces << " Faces" << std::endl;
    for(FaceID i = 0; i < m_nbFaces; ++i)
    {
        if(m_meshType == TriangleMesh)
        {
            const Triangle f = m_topology->getTriangle(i);
            const TriangleEdges& edges = m_topology->getEdgeTriangleShell(i);
            outfile << "[" << i << "]" << "  " << f[0] << "," << f[1] << "," << f[2] << endl;
            outfile << "[" << i << "]" << "  " << edges[0] << "," << edges[1] << "," << edges[2] << endl;
        }
        if(m_meshType == QuadMesh)
        {
            const Quad f = m_topology->getQuad(i);
            const QuadEdges& edges = m_topology->getEdgeQuadShell(i);
            outfile << "[" << i << "]" << "  " << f[0] << "," << f[1] << "," << f[2] << "," << f[3] << endl;
            outfile << "[" << i << "]" << "  " << edges[0] << "," << edges[1] << "," << edges[2] << "," << edges[3] << endl;

        }
        outfile << "isBoundary " << m_fInfo.m_isBoundary[i] << endl;
        outfile << "FaceCenter" << m_fInfo.m_centers[i] << endl;
        outfile << "At =  " << endl;
        for(int j = 0; j < m_fInfo.m_At[i].Nrows(); ++j)
        {
            for(int k = 0; k <m_fInfo.m_At[i].Ncols(); ++k)
                outfile << m_fInfo.m_At[i].element(j, k) << "  ";
            outfile << endl;
        }
    }
    outfile.close();
    outfile.clear();
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::saveOperators() const
{
    std::string str = "d0.txt";
    std::ofstream outfile(str.c_str());

    outfile << d0.rowSize() << "*" << d0.colSize() << endl;
    outfile << d0;
    outfile.close();
    outfile.clear();

    str = "d1.txt";
    outfile.open(str.c_str(), std::ios::out);
    outfile << d1.rowSize() << "*" << d1.colSize() << endl;
    outfile << d1;
    outfile.close();
    outfile.clear();

    str = "star0.txt";
    outfile.open(str.c_str(), std::ios::out);
    outfile << star0.size() << endl;
    outfile << star0;
    outfile.close();
    outfile.clear();

    str = "star1.txt";
    outfile.open(str.c_str(), std::ios::out);
    outfile << star1.size() << endl;
    outfile << star1;
    outfile.close();
    outfile.clear();

    str = "star2.txt";
    outfile.open(str.c_str(), std::ios::out);
    outfile << star2.size() << endl;
    outfile << star2;
    outfile.close();
    outfile.clear();

    str = "curl.txt";
    outfile.open(str.c_str(), std::ios::out);
    outfile << curl.rowSize() << "*" << curl.colSize() << endl;
    outfile << curl;
    outfile.close();
    outfile.clear();

    str = "laplace.txt";
    outfile.open(str.c_str(), std::ios::out);
    outfile << laplace.rowSize() << "*" << laplace.colSize() << endl;
    outfile << laplace;
    outfile.close();
    outfile.clear();

    str = "m_laplace.txt";
    outfile.open(str.c_str(), std::ios::out);
    outfile << m_laplace.Nrows() << "*" << m_laplace.Ncols() << endl;
    outfile << "[" << endl;
    for(int i = 0; i < m_laplace.Nrows(); ++i)
    {

        for(int j = 0; j < m_laplace.Ncols(); ++j)
        {
            outfile << m_laplace.element(i, j) << " ";
        }
        outfile <<endl;
    }
    outfile << "]" << endl;
    outfile.close();
    outfile.clear();

}



template<class DataTypes>
void EulerianFluidModel<DataTypes>::saveVorticity() const
{
    std::string str = "vorticity_.txt";
    std::ofstream outfile(str.c_str());
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        outfile << "[" << i << "] "<< m_vorticity.element(i) << endl;
    }
    outfile.close();
    outfile.clear();
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::savePhi() const
{
    std::string str = "phi_.txt";
    std::ofstream outfile(str.c_str());
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        outfile << "[" << i << "] "<< m_phi.element(i) << endl;
    }
    outfile.close();
    outfile.clear();
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::saveFlux() const
{
    std::string str = "flux_.txt";
    std::ofstream outfile(str.c_str());
    for(EdgeID i = 0; i < m_nbEdges; ++i)
    {
        outfile << "[" << i << "] "<< m_flux.element(i) << endl;
    }
    outfile.close();
    outfile.clear();
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::saveVelocity() const
{
    std::string str = "velocity_.txt";
    std::ofstream outfile(str.c_str());
    for(FaceID i = 0; i < m_nbFaces; ++i)
    {
        outfile << "[" << i << "] "<< m_vels[i] << endl;
    }
    outfile.close();
    outfile.clear();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
