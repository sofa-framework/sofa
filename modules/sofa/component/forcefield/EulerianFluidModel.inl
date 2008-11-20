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
    m_nbVolumes = 0;

    //initialize the size of state variables
    m_vorticity.ReSize(m_nbPoints);
    m_flux.ReSize(m_nbEdges);

    m_vels.resize(m_nbFaces);
    m_bkCenters.resize(m_nbFaces);
    m_bkVels.resize(m_nbFaces);

    //initialize the size of visualization variables
    m_pInfo.m_values.resize(m_nbPoints);
    m_fInfo.m_vectors.resize(m_nbFaces);

    //initialize Boundary
    computeBoundary2D();

    //initialize centers
    computeEdgeCenters();
    computeFaceCenters();

    //initialize dual faces
    computeDualFaces();

    //calculate Operators
    computeOperators();

    //initialize boundary constraints
    setBdConstraints(-m_bdX.getValue(), m_bdX.getValue(), -m_bdY.getValue(), m_bdY.getValue(), -m_bdZ.getValue(), m_bdZ.getValue());

    //initialize project matrices
    computeProjectMats();

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

    using namespace sofa::helper::system::thread;

    ctime_t startTime = CTime::getTime();
    // Omega => Phi
    //std::cout << "Omega => Phi" << endl;
    calcPhi();
    //savePhi();

    // Phi => U
    //std::cout << "Phi => U" << endl;
    calcFlux();
    //saveFlux();
    ctime_t endTime = CTime::getTime();
    time_LE += endTime - startTime;
    std::cout << "time_linear_equations = " << (endTime - startTime)/1e6 << endl;


    // U => v
    //std::cout << "U => v" << endl;
    calcVelocityAtDualVertex();
    //saveVelocity();

    // v = > Omega
    startTime = CTime::getTime();
    //std::cout << "v = > Omega" << endl;
    backtrack(dt);
    calcVorticity();
    //saveVorticity();
    endTime = CTime::getTime();
    time_BT += endTime - startTime;
    std::cout << "time_backtrack = " << (endTime - startTime)/1e6 << endl;

    //add Forces
    if(m_bAddForces.getValue())
        addForces();
    //saveVorticity();


}


template<class DataTypes>
void EulerianFluidModel<DataTypes>::draw()
{
    assert(this->m_mstate);

    // Compute topological springs
    const VecCoord& p1 = *this->m_mstate->getX();

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
                        if(abs(m_pInfo.m_values[face[j]]) < 0.005)
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
                        if(abs(m_pInfo.m_values[face[j]]) < 0.005)
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
        }

        // draw velocity
        if(getContext()->getShowBehaviorModels() && m_bDisplayVelocity.getValue())
        {
            glDisable(GL_LIGHTING);
            glLineWidth(2.0f);
            for(FaceID i = 0; i < m_nbFaces; ++i)
            {
                sofa::defaulttype::Vec<3, double> pt(m_fInfo.m_centers[i][0] + m_fInfo.m_vectors[i][0],
                        m_fInfo.m_centers[i][1] + m_fInfo.m_vectors[i][1],
                        m_fInfo.m_centers[i][2] + m_fInfo.m_vectors[i][2]);
                glBegin(GL_LINES);
                glColor3f(0.0, 0.0, 0.0);
                glVertex3f(m_fInfo.m_centers[i][0], m_fInfo.m_centers[i][1], m_fInfo.m_centers[i][2]);
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

            for(std::map<int, double>::iterator it = m_bdConstraints.begin(); it != m_bdConstraints.end(); ++it)
            {
                Edge e = m_topology->getEdge(it->first);
                glBegin(GL_LINES);
                if(it->second > 0.0)
                    glColor3f(0.0, 1.0, 0.0);
                else
                {
                    if(it->second < 0.0)
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
void EulerianFluidModel<DataTypes>::computeBoundary2D()
{
    m_pInfo.m_isBoundary.resize(m_nbPoints);
    m_eInfo.m_isBoundary.resize(m_nbEdges);
    m_fInfo.m_isBoundary.resize(m_nbFaces);

    m_pInfo.m_isBoundary.fill(false);
    m_eInfo.m_isBoundary.fill(false);
    m_fInfo.m_isBoundary.fill(false);

    for(EdgeID i = 0; i < m_nbEdges; ++i)
    {
        const Edge e = m_topology->getEdge(i);
        const EdgeFaces& faces = (m_meshType == TriangleMesh) ? m_topology->getTriangleEdgeShell(i) : m_topology->getQuadEdgeShell(i);
        if(faces.size() == 1)
        {
            m_pInfo.m_isBoundary[e[0]] = true;
            m_pInfo.m_isBoundary[e[1]] = true;
            m_eInfo.m_isBoundary[i] = true;
            m_fInfo.m_isBoundary[faces[0]] = true;
        }
    }
}


template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeEdgeCenters()
{
    m_eInfo.m_centers.clear();
    m_eInfo.m_centers.resize(m_nbEdges);

    switch(m_meshType)
    {
    case TriangleMesh:
        for(EdgeID i = 0; i < m_nbEdges; ++i)
        {
            m_eInfo.m_centers[i] = m_triGeo->computeEdgeCenter(i);
        }
        break;
    case QuadMesh:
    case RegularQuadMesh:
        for(EdgeID i = 0; i < m_nbEdges; ++i)
        {
            m_eInfo.m_centers[i] = m_quadGeo->computeEdgeCenter(i);
        }
        break;
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::computeFaceCenters()
{
    m_fInfo.m_centers.clear();
    m_fInfo.m_centers.resize(m_nbFaces);

    switch(m_meshType)
    {
    case TriangleMesh:
        for(FaceID i = 0; i < m_nbFaces; ++i)
        {
            if(m_centerType == Barycenter)
                m_fInfo.m_centers[i] = m_triGeo->computeTriangleCenter(i);
            else
                m_fInfo.m_centers[i] = m_triGeo->computeTriangleCircumcenter(i);
        }
        break;

    case QuadMesh:
        for(FaceID i= 0; i < m_nbFaces; ++i)
        {
            if(m_centerType == Barycenter)
                m_fInfo.m_centers[i] = m_quadGeo->computeQuadCenter(i);
            else
                std::cerr << "At present, not implemented: computeFaceCenters() for circumcenters of quads" << endl;
        }
        break;

    case RegularQuadMesh:
        for(FaceID i= 0; i < m_nbFaces; ++i)
            m_fInfo.m_centers[i] = m_quadGeo->computeQuadCenter(i);
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
                    PointInformation::Normal norm1, norm2, norm3, norm4;
                    Edge e;
                    EdgeFaces eFaces;

                    // one of the edge center
                    m_pInfo.m_dualFaces[i].push_back(m_eInfo.m_centers[pEdges.back()]);
                    e = m_topology->getEdge(pEdges.back());
                    norm1 = m_triGeo->computeEdgeDirection(pEdges.back());
                    if(e[1] == i)
                        norm1 = -norm1;
                    eFaces = m_topology->getTriangleEdgeShell(pEdges.back());
                    assert(eFaces.size() == 1);
                    norm2 = m_fInfo.m_centers[eFaces[0]] - m_eInfo.m_centers[pEdges.back()];

                    // point
                    m_pInfo.m_dualFaces[i].push_back(PointInformation::VertexOfDualFace(p.x(), p.y(), p.z()));

                    // the other edge center
                    m_pInfo.m_dualFaces[i].push_back(m_eInfo.m_centers[pEdges.front()]);
                    e = m_topology->getEdge(pEdges.front());
                    eFaces = m_topology->getTriangleEdgeShell(pEdges.front());
                    assert(eFaces.size() == 1);
                    norm3 = m_fInfo.m_centers[eFaces[0]] - m_eInfo.m_centers[pEdges.front()];
                    norm4 = m_triGeo->computeEdgeDirection(pEdges.back());
                    if(e[1] == i)
                        norm4 = -norm4;

                    PointInformation::VertexNormal vNorms;
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

                    PointInformation::VertexNormal vNorms;
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
                    PointInformation::Normal norm1, norm2, norm3, norm4;
                    Edge e;
                    EdgeFaces eFaces;

                    // one of the edge center
                    m_pInfo.m_dualFaces[i].push_back(m_eInfo.m_centers[pEdges.back()]);
                    e = m_topology->getEdge(pEdges.back());
                    norm1 = m_quadGeo->computeEdgeDirection(pEdges.back());
                    if(e[1] == i)
                        norm1 = -norm1;
                    eFaces = m_topology->getQuadEdgeShell(pEdges.back());
                    assert(eFaces.size() == 1);
                    norm2 = m_fInfo.m_centers[eFaces[0]] - m_eInfo.m_centers[pEdges.back()];

                    // point
                    m_pInfo.m_dualFaces[i].push_back(PointInformation::VertexOfDualFace(p.x(), p.y(), p.z()));

                    // the other edge center
                    m_pInfo.m_dualFaces[i].push_back(m_eInfo.m_centers[pEdges.front()]);
                    e = m_topology->getEdge(pEdges.front());
                    eFaces = m_topology->getQuadEdgeShell(pEdges.front());
                    assert(eFaces.size() == 1);
                    norm3 = m_fInfo.m_centers[eFaces[0]] - m_eInfo.m_centers[pEdges.front()];
                    norm4 = m_quadGeo->computeEdgeDirection(pEdges.back());
                    if(e[1] == i)
                        norm4 = -norm4;

                    PointInformation::VertexNormal vNorms;
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

                    PointInformation::VertexNormal vNorms;
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

    //边界上的phi指定为0
    m_laplace.ReSize(m_nbPoints, m_nbPoints);
    m_laplace = 0.0;
    for(int i = 0; i < m_nbPoints; ++i)
    {
        if(m_pInfo.m_isBoundary[i])
            m_laplace.element(i, i) = 1.0;
        else
        {
            for(int j = 0; j < m_nbPoints; ++j)
                m_laplace.element(i, j) = laplace.element(i, j);
        }
    }

    /*
    //unsymmetric L
    m_laplace.ReSize(m_nbPoints, m_nbPoints);
    m_laplace = 0.0;
    std::map<int, double>::iterator it = m_bdConstraints.begin();
    for(int i = 0; i < m_nbPoints; ++i)
    {
    	if(m_pInfo.m_isBoundary[i])
    	{
    		assert(it != m_bdConstraints.end());
    		for(int j = 0; j < m_nbPoints; ++j)
    			m_laplace.element(i, j) = d0.element(it->first, j);
    		++it;
    	}
    	else
    	{
    		for(int j = 0; j < m_nbPoints; ++j)
    			m_laplace.element(i, j) = laplace.element(i, j);
    	}
    }*/
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
                dualEdgeLength -= (m - c).norm();
        }
        star1.set(i, dualEdgeLength / m_triGeo->computeEdgeLength(i));
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
        star1.set(i, dualEdgeLength / m_quadGeo->computeEdgeLength(i));
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
                v = (c1 - c0) * (double)d1.element(i, fEdges[j]) * m_triGeo->computeEdgeLength(fEdges[j]) / (c1 - c0).norm();
                if(star1.element(fEdges[j]) < 0.0)
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
                v = (c1 - c0) * (double)d1.element(i, fEdges[j]) * m_quadGeo->computeEdgeLength(fEdges[j]) / (c1 - c0).norm();

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
void EulerianFluidModel<DataTypes>::setBdConstraints(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax)
{
    m_bdConstraints.clear();

    if(xMin == xMax)
        //for yz plane
    {
        for(EdgeID i = 0; i < m_nbEdges; ++i)
        {
            if(m_eInfo.m_isBoundary[i])
            {
                if(m_eInfo.m_centers[i].y() < yMin || m_eInfo.m_centers[i].y() > yMax)
                {
                    m_bdConstraints.insert(make_pair(i, 0));
                    continue;
                }
                const EdgeFaces& eFaces = (m_meshType == MeshType::TriangleMesh) ?
                        m_topology->getTriangleEdgeShell(i) : m_topology->getQuadEdgeShell(i);
                //double value = 1.0 * d1.element(eFaces[0], i);
                double value = 0;
                if(m_eInfo.m_centers[i].z() < zMin)
                {
                    m_bdConstraints.insert(make_pair(i, value));
                    continue;
                }
                if(m_eInfo.m_centers[i].z() > zMax)
                {
                    m_bdConstraints.insert(make_pair(i, -value));
                    continue;
                }

            }
        }
    }

    if(zMin == zMax)
        //for xy plane
    {
        for(EdgeID i = 0; i < m_nbEdges; ++i)
        {
            if(m_eInfo.m_isBoundary[i])
            {
                if(m_eInfo.m_centers[i].y() < yMin || m_eInfo.m_centers[i].y() > yMax)
                {
                    m_bdConstraints.insert(make_pair(i, 0));
                    continue;
                }
                const EdgeFaces& eFaces = (m_meshType == MeshType::TriangleMesh) ?
                        m_topology->getTriangleEdgeShell(i) : m_topology->getQuadEdgeShell(i);

                //double value = 1.0 * d1.element(eFaces[0], i);
                double value = 0;
                if(m_eInfo.m_centers[i].x() < xMin)
                {
                    m_bdConstraints.insert(make_pair(i, value));
                    continue;
                }
                if(m_eInfo.m_centers[i].x() > xMax)
                {
                    m_bdConstraints.insert(make_pair(i, -value));
                    continue;
                }

            }
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
void EulerianFluidModel<DataTypes>::calcVelocityAtDualVertex()
{
    switch(m_meshType)
    {
    case TriangleMesh:
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
            assert(v.Nrows() == 3);
            m_vels[i][0] = m_harmonicVx.getValue() + v.element(0);
            m_vels[i][1] = m_harmonicVy.getValue() + v.element(1);
            m_vels[i][2] = m_harmonicVz.getValue() + v.element(2);
        }
        break;
    case QuadMesh:
    case RegularQuadMesh:
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
            assert(v.Nrows() == 3);
            m_vels[i][0] = m_harmonicVx.getValue() + v.element(0);
            m_vels[i][1] = m_harmonicVy.getValue() + v.element(1);
            m_vels[i][2] = m_harmonicVz.getValue() + v.element(2);
        }
        break;
    }
}

template<class DataTypes>
unsigned int EulerianFluidModel<DataTypes>::searchFaceForTriMesh(const sofa::defaulttype::Vec<3, double>& pt, FaceID startFace) const
{
    //at present, only for triangle mesh
    assert(m_meshType == TriangleMesh);

    if(startFace < 0 || startFace >= m_nbFaces)
        startFace = 0;

    sofa::helper::vector<bool> flag(m_nbFaces, false);
    sofa::helper::vector<FaceID> faces;

    flag[startFace] = true;
    faces.push_back(startFace);

    unsigned int temp;
    for(int i = 0; i < faces.size(); ++i)
    {
        if(m_triGeo->isPointInTriangle(faces[i], false, pt, temp))
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
    //at present, only for triangle mesh
    assert(m_meshType == TriangleMesh);

    sofa::defaulttype::Vec<3,double> p(pt.x(), pt.y(), pt.z());

    //search the face(tri/quad) in which pt is
    FaceID indFace = searchFaceForTriMesh(p, startFace);
    if(indFace == sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID)
        //out of boundary
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
        break;
    }
}

template<class DataTypes>
unsigned int EulerianFluidModel<DataTypes>::searchDualFaceForQuadMesh(const Coord & pt, PointID startDualFace) const
{
    if(startDualFace < 0 || startDualFace >= m_nbPoints)
        startDualFace = 0;

    sofa::helper::vector<bool> flag(m_nbPoints, false);
    sofa::helper::vector<PointID> dualFaceIDs;

    flag[startDualFace] = true;
    dualFaceIDs.push_back(startDualFace);

    sofa::defaulttype::Vec<3, double> p(pt.x(), pt.y(), pt.z());

    for(int i = 0; i < dualFaceIDs.size(); ++i)
    {
        PointInformation::DualFace dualFace = m_pInfo.m_dualFaces[dualFaceIDs[i]];
        if(!m_pInfo.m_isBoundary[dualFaceIDs[i]])
        {
            if(sofa::component::topology::is_point_in_quad(p, dualFace[0], dualFace[1], dualFace[2], dualFace[3]))
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
                    Coord c(m_fInfo.m_centers[vFaces[j]][0], m_fInfo.m_centers[vFaces[j]][1], m_fInfo.m_centers[vFaces[j]][2]);

                    for(PointID k = 0; k < f.size(); ++k)
                    {
                        Coord v(m_topology->getPX(f[k]), m_topology->getPY(f[k]), m_topology->getPZ(f[k]));
                        Coord v1(m_topology->getPX(f[(k+1)%4]), m_topology->getPY(f[(k+1)%4]), m_topology->getPZ(f[(k+1)%4]));
                        Coord v2(m_topology->getPX(f[(k+3)%4]), m_topology->getPY(f[(k+3)%4]), m_topology->getPZ(f[(k+3)%4]));
                        Deriv vv1 = v1 - v;
                        Deriv vv2 = v2 - v;
                        Deriv vc = c - v;
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
typename DataTypes::Deriv EulerianFluidModel<DataTypes>:: interpolateVelocity(const Coord& pt, unsigned int start) const
//for TriangleMesh start = startFace
//for QuadMesh start = startDualFace (i.e.startPoint)
{
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
    }

    if(ind_p == sofa::core::componentmodel::topology::BaseMeshTopology::InvalidID)
        //pt is out of boundary
    {
        return vel;
    }

    //calculate velocities on the vertices of dual face
    VecDeriv vels;
    if(m_pInfo.m_isBoundary[ind_p])
    {
        // at present, let the boundary velocity equal to the harmonic velocity
        VertexEdges pEdges = m_topology->getOrientedEdgeVertexShell(ind_p);
        //the velocity on the boudary edge: pEdges.back()
        vels.push_back(Deriv(m_harmonicVx.getValue(), m_harmonicVy.getValue(), m_harmonicVz.getValue()));
        //the velocity on the boudary point: ind_p
        vels.push_back(Deriv(m_harmonicVx.getValue(), m_harmonicVy.getValue(), m_harmonicVz.getValue()));
        //the velocity on the boudary edge: pEdges.front()
        vels.push_back(Deriv(m_harmonicVx.getValue(), m_harmonicVy.getValue(), m_harmonicVz.getValue()));
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
        Coord c(m_pInfo.m_dualFaces[ind_p].at(i)[0], m_pInfo.m_dualFaces[ind_p].at(i)[1], m_pInfo.m_dualFaces[ind_p].at(i)[2]);
        Deriv pc = c - p;

        if( pc.norm2() < ZERO2 ) // p == c
            return vels.at(i);

        double dis0 = pc * m_pInfo.m_dualFaceVertexNormals[ind_p].at(i)[0];
        double dis1 = pc * m_pInfo.m_dualFaceVertexNormals[ind_p].at(i)[1];
        if(dis0 < ZERO) // p is on the dual edge c[i-1]c[i]
        {
            Coord c1(m_pInfo.m_dualFaces[ind_p].at((i+nbVertices-1)%nbVertices)[0],
                    m_pInfo.m_dualFaces[ind_p].at((i+nbVertices-1)%nbVertices)[1],
                    m_pInfo.m_dualFaces[ind_p].at((i+nbVertices-1)%nbVertices)[2]);
            sofa::helper::vector< double > baryCoefs = sofa::component::topology::compute_2points_barycoefs(p, c, c1);
            return vels.at(i) * baryCoefs[0] + vels.at((i+nbVertices-1)%nbVertices) * baryCoefs[1];
        }
        if(dis1 < ZERO) // p is on the dual edge c[i]c[i+1]
        {
            Coord c1(m_pInfo.m_dualFaces[ind_p].at((i+1)%nbVertices)[0],
                    m_pInfo.m_dualFaces[ind_p].at((i+1)%nbVertices)[1],
                    m_pInfo.m_dualFaces[ind_p].at((i+1)%nbVertices)[2]);
            sofa::helper::vector< double > baryCoefs = sofa::component::topology::compute_2points_barycoefs(p, c, c1);
            return vels.at(i) * baryCoefs[0] + vels.at((i+1)%nbVertices) * baryCoefs[1];
        }

        w = m_pInfo.m_dualFaceVertexVolumes[ind_p].at(i) / dis0 * dis1;
        wNormalize += w;
        vel += vels.at(i) * w;
    }

    return vel / wNormalize;

}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::backtrack(double dt)
{
    for(FaceID i = 0; i < m_nbFaces; ++i)
    {
        Coord c(m_fInfo.m_centers[i][0], m_fInfo.m_centers[i][1], m_fInfo.m_centers[i][2]);
        m_bkCenters[i] = c + m_vels[i] * (-dt);
        switch(m_meshType)
        {
        case TriangleMesh:
            m_bkVels[i] = interpolateVelocity(m_bkCenters[i], i);
            break;
        case QuadMesh:
        case RegularQuadMesh:
            Quad f = m_topology->getQuad(i);
            m_bkVels[i] = interpolateVelocity(m_bkCenters[i], f[0]);
            break;
        }
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::calcVorticity()
{
    m_vorticity = 0.0;

    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        if(!m_pInfo.m_isBoundary[i])
            //set the boundary voritcity to be zero
        {
            const VertexFaces vFaces = (m_meshType == TriangleMesh) ?
                    m_topology->getOrientedTriangleVertexShell(i) : m_topology->getOrientedQuadVertexShell(i);
            for(FaceID j = 0; j < vFaces.size(); ++j)
            {
                m_vorticity.element(i) += 0.5 * (
                        (m_bkVels[vFaces[j]] + m_bkVels[vFaces[(j+1)%vFaces.size()]]) *
                        (m_bkCenters[vFaces[(j+1)%vFaces.size()]] - m_bkCenters[vFaces[j]]) );
            }
        }
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::addForces()
{
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        Coord p = (m_meshType == TriangleMesh) ?
                m_triGeo->getPointPosition(i) : m_quadGeo->getPointPosition(i);
        Coord o(0.0, 0.0, 0.0);
        if((o-p).norm2() < 5)
            m_vorticity.element(i) += 3.0;
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::calcPhi()
{
    m_phi = m_laplace.i() * m_vorticity;
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::calcFlux()
{
    m_flux = m_d0 * m_phi;
    setBoundaryFlux();
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::setBoundaryFlux()
{
    assert(!m_bdConstraints.empty());
    for(std::map<int, double>::iterator it = m_bdConstraints.begin(); it != m_bdConstraints.end(); ++it)
    {
        m_flux.element(it->first) = it->second;
    }
}

template<class DataTypes>
void EulerianFluidModel<DataTypes>::normalizeDisplayValues()
{
    //normailize the velocity
    for(FaceID i = 0; i < m_nbFaces; ++i)
    {
        m_fInfo.m_vectors[i][0] = 5 * m_vels[i][0];
        m_fInfo.m_vectors[i][1] = 5 * m_vels[i][1];
        m_fInfo.m_vectors[i][2] = 5 * m_vels[i][2];
    }

    //normailize the vorticity
    float value;
    for(PointID i = 0; i < m_nbPoints; ++i)
    {
        value = m_vorticity.element(i);
        if(value > 0)
            m_pInfo.m_values[i] = 0.2*log(1+50*value);
        else
            m_pInfo.m_values[i] = -0.2*log(1-50*value);
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
