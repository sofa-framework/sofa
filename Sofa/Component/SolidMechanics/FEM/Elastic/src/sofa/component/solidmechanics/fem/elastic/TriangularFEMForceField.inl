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

#include <sofa/component/solidmechanics/fem/elastic/TriangularFEMForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/ColorMap.h>
#include <sofa/type/RGBAColor.h>

#include <sofa/core/topology/TopologyData.inl>

#include <Eigen/SVD>
#include <limits>

namespace sofa::component::solidmechanics::fem::elastic
{

using namespace sofa::core::topology;

// --------------------------------------------------------------------------------------
// --- constructor
// --------------------------------------------------------------------------------------
template <class DataTypes>
TriangularFEMForceField<DataTypes>::TriangularFEMForceField()
    : triangleInfo(initData(&triangleInfo, "triangleInfo", "Internal triangle data"))
    , vertexInfo(initData(&vertexInfo, "vertexInfo", "Internal point data"))
    , m_topology(nullptr)
    , method(LARGE)
    , f_method(initData(&f_method, std::string("large"), "method", "large: large displacements, small: small displacements"))
    , f_poisson(initData(&f_poisson, type::vector<Real>(1, static_cast<Real>(0.3)), "poissonRatio", "Poisson ratio in Hooke's law (vector)"))
    , f_young(initData(&f_young, type::vector<Real>(1, static_cast<Real>(1000.0)), "youngModulus", "Young modulus in Hooke's law (vector)"))
    , m_rotatedInitialElements(initData(&m_rotatedInitialElements, "rotatedInitialElements", "Flag activating rendering of stress directions within each triangle"))
    , m_initialTransformation(initData(&m_initialTransformation, "initialTransformation", "Flag activating rendering of stress directions within each triangle"))
    , hosfordExponant(initData(&hosfordExponant, (Real)1.0, "hosfordExponant", "Exponant in the Hosford yield criteria"))
    , criteriaValue(initData(&criteriaValue, (Real)1e15, "criteriaValue", "Fracturable threshold used to draw fracturable triangles"))
    , showStressValue(initData(&showStressValue, true, "showStressValue", "Flag activating rendering of stress values as a color in each triangle"))
    , showStressVector(initData(&showStressVector, false, "showStressVector", "Flag activating rendering of stress directions within each triangle"))
    , showFracturableTriangles(initData(&showFracturableTriangles, false, "showFracturableTriangles", "Flag activating rendering of triangles to fracture"))
    , f_computePrincipalStress(initData(&f_computePrincipalStress, false, "computePrincipalStress", "Compute principal stress for each triangle"))
    , l_topology(initLink("topology", "link to the topology container"))
#ifdef PLOT_CURVE
    , elementID(initData(&elementID, (Real)0, "id", "element id to follow in the graphs"))
    , f_graphStress(initData(&f_graphStress, "graphMaxStress", "Graph of max stress corresponding to the element id"))
    , f_graphCriteria(initData(&f_graphCriteria, "graphCriteria", "Graph of the fracture criteria corresponding to the element id"))
    , f_graphOrientation(initData(&f_graphOrientation, "graphOrientation", "Graph of the orientation of the principal stress direction corresponding to the element id"))
#endif
    , p_computeDrawInfo(false)
{
    _anisotropicMaterial = false;
#ifdef PLOT_CURVE
    f_graphStress.setWidget("graph");
    f_graphCriteria.setWidget("graph");
    f_graphOrientation.setWidget("graph");
#endif

    f_poisson.setRequired(true);
    f_young.setRequired(true);
    p_drawColorMap = new helper::ColorMap(256, "Blue to Red");
}


template <class DataTypes>
TriangularFEMForceField<DataTypes>::~TriangularFEMForceField()
{
    if (p_drawColorMap) delete p_drawColorMap;
}


// --------------------------------------------------------------------------------------
// --- Initialization stage
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::init()
{
    this->Inherited::init();

#ifdef PLOT_CURVE
    allGraphStress.clear();
    allGraphCriteria.clear();
    allGraphOrientation.clear();

    f_graphStress.beginEdit()->clear();
    f_graphStress.endEdit();
    f_graphCriteria.beginEdit()->clear();
    f_graphCriteria.endEdit();
    f_graphOrientation.beginEdit()->clear();
    f_graphOrientation.endEdit();
#endif

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (m_topology->getNbTriangles() == 0)
    {
        msg_warning() << "No triangles found in linked Topology.";
    }

    // Create specific Engine for TriangleData
    triangleInfo.createTopologyHandler(m_topology);
    vertexInfo.createTopologyHandler(m_topology);

    triangleInfo.setCreationCallback([this](Index triangleIndex, TriangleInformation& triInfo,
        const core::topology::BaseMeshTopology::Triangle& triangle,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        createTriangleInformation(triangleIndex, triInfo, triangle, ancestors, coefs);
    });


    if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;

    reinit();
}


// --------------------------------------------------------------------------------------
// --- Store the initial position of the nodes
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::initSmall(int i, Index& a, Index& b, Index& c)
{
    auto triangleInf = sofa::helper::getWriteOnlyAccessor(triangleInfo);
    TriangleInformation* tinfo = &triangleInf[i];

    tinfo->initialTransformation.identity();

    if (m_rotatedInitialElements.isSet())
        tinfo->rotatedInitialElements = m_rotatedInitialElements.getValue()[i];
    else
    {
        const  VecCoord& initialPoints = (this->mstate->read(core::ConstVecCoordId::restPosition())->getValue());
        const Coord& pA = initialPoints[a];
        const Coord& pB = initialPoints[b];
        const Coord& pC = initialPoints[c];

        tinfo->rotatedInitialElements[0] = Coord(0, 0, 0); // always (0,0,0): pA - pA
        tinfo->rotatedInitialElements[1] = pB - pA;
        tinfo->rotatedInitialElements[2] = pC - pA;
    }

    try
    {
        m_triangleUtils.computeStrainDisplacementGlobal(tinfo->strainDisplacementMatrix, tinfo->rotatedInitialElements[0], tinfo->rotatedInitialElements[1], tinfo->rotatedInitialElements[2]);
    }
    catch (const std::exception& e)
    {
        msg_error() << e.what();
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // store area of each triangle
    Real determinant = cross(tinfo->rotatedInitialElements[1], tinfo->rotatedInitialElements[2]).norm();
    tinfo->area = determinant * 0.5;
}

// --------------------------------------------------------------------------------------
// --- Store the initial position of the nodes
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::initLarge(int i, Index& a, Index& b, Index& c)
{
    auto triangleInf = sofa::helper::getWriteOnlyAccessor(triangleInfo);
    if (sofa::Size(i) >= triangleInf.size())
    {
        msg_error() << "Try to access an element which indices bigger than the size of the vector: i=" << i << " and size=" << triangleInf.size();
        return;
    }

    TriangleInformation* tinfo = &triangleInf[i];

    if (m_initialTransformation.isSet() && m_rotatedInitialElements.isSet())
    {
        Transformation R_0_1;
        R_0_1 = m_initialTransformation.getValue()[i];
        tinfo->initialTransformation.transpose(R_0_1);
        tinfo->rotatedInitialElements = m_rotatedInitialElements.getValue()[i];
    }
    else
    {
        const VecCoord& initialPoints = (this->mstate->read(core::ConstVecCoordId::restPosition())->getValue());
        tinfo->rotation = tinfo->initialTransformation;
        if (a >= (initialPoints).size() || b >= (initialPoints).size() || c >= (initialPoints).size())
        {
            msg_error() << "Try to access an element which indices bigger than the size of the vector: a=" << a <<
                " b=" << b << " and c=" << c << " and size=" << (initialPoints).size() << msgendl;
            return;
        }

        const Coord& pA = initialPoints[a];
        const Coord& pB = initialPoints[b];
        const Coord& pC = initialPoints[c];
        Coord pAB = pB - pA;
        Coord pAC = pC - pA;

        // Rotation matrix (initial triangle/world)
        // first vector on first edge
        // second vector in the plane of the two first edges
        // third vector orthogonal to first and second
        Transformation R_0_1;
        m_triangleUtils.computeRotationLarge(R_0_1, pA, pB, pC);
        tinfo->initialTransformation.transpose(R_0_1);
        tinfo->rotation = tinfo->initialTransformation;

        // coordinates of the triangle vertices in their local frames with origin at vertex a
        tinfo->rotatedInitialElements[0] = Coord(0, 0, 0);
        tinfo->rotatedInitialElements[1] = R_0_1 * pAB;
        tinfo->rotatedInitialElements[2] = R_0_1 * pAC;
    }

    try
    {
        m_triangleUtils.computeStrainDisplacementLocal(tinfo->strainDisplacementMatrix, tinfo->rotatedInitialElements[1], tinfo->rotatedInitialElements[2]);
    }
    catch (const std::exception& e)
    {
        msg_error() << e.what();
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // store area of each triangle
    tinfo->area = tinfo->rotatedInitialElements[1][0] * tinfo->rotatedInitialElements[2][1] * 0.5;
}

// --------------------------------------------------------------------------------------
// --- Re-initialization (called when we change a parameter through the GUI)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::reinit()
{
    if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;

    unsigned int nbPoints = m_topology->getNbPoints();
    type::vector<VertexInformation>& vi = *(vertexInfo.beginWriteOnly());
    vi.resize(nbPoints);
    vertexInfo.endEdit();


    /// prepare to store info in the triangle array
    type::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginWriteOnly());
    triangleInf.resize(m_topology->getNbTriangles());
    for (Topology::TriangleID i = 0; i < m_topology->getNbTriangles(); ++i)
    {
        createTriangleInformation(i, triangleInf[i], m_topology->getTriangle(i), (const sofa::type::vector< Index >)0, (const sofa::type::vector< Real >)0);
    }
    triangleInfo.endEdit();


    // checking inputs using setter
    setMethod(f_method.getValue());
    if (f_poisson.getValue().size() == 1) // array option is not checked
        setPoisson(f_poisson.getValue()[0]);
    else
        setPoissonArray(f_poisson.getValue());

    if (f_young.getValue().size() == 1)
        setYoung(f_young.getValue()[0]);
    else
        setYoungArray(f_young.getValue());

#ifdef PLOT_CURVE
    std::map<std::string, sofa::type::vector<double> >& stress = *(f_graphStress.beginEdit());
    stress.clear();
    if (allGraphStress.size() > elementID.getValue())
        stress = allGraphStress[elementID.getValue()];
    f_graphStress.endEdit();

    std::map<std::string, sofa::type::vector<double> >& criteria = *(f_graphCriteria.beginEdit());
    criteria.clear();
    if (allGraphCriteria.size() > elementID.getValue())
        criteria = allGraphCriteria[elementID.getValue()];
    f_graphCriteria.endEdit();

    std::map<std::string, sofa::type::vector<double> >& orientation = *(f_graphOrientation.beginEdit());
    orientation.clear();
    if (allGraphOrientation.size() > elementID.getValue())
        orientation = allGraphOrientation[elementID.getValue()];
    f_graphOrientation.endEdit();
#endif
}

// --------------------------------------------------------------------------------------
// ---  Topology Creation/Destruction functions
// --------------------------------------------------------------------------------------

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::createTriangleInformation(Index triangleIndex, TriangleInformation&,
    const core::topology::BaseMeshTopology::Triangle& t,
    const sofa::type::vector< Index >&,
    const sofa::type::vector< SReal >&)
{
    Index a = t[0];
    Index b = t[1];
    Index c = t[2];

    switch (method)
    {
    case SMALL:
        initSmall(triangleIndex, a, b, c);
        computeMaterialStiffness(triangleIndex, a, b, c);
        break;

    case LARGE:
        initLarge(triangleIndex, a, b, c);
        computeMaterialStiffness(triangleIndex, a, b, c);
        break;
    }
}


// --------------------------------------------------------------------------------------
// --- Get/Set methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
SReal TriangularFEMForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* x */) const
{
    msg_error() << "TriangularFEMForceField::getPotentialEnergy-not-implemented !!!";
    return 0;
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeElementStiffnessMatrix(type::Mat<9, 9, typename TriangularFEMForceField<DataTypes>::Real>& S, type::Mat<9, 9, typename TriangularFEMForceField<DataTypes>::Real>& SR, const MaterialStiffness& K, const StrainDisplacement& J, const Transformation& Rot)
{
    type::MatNoInit<3, 6, Real> Jt;
    Jt.transpose(J);

    type::MatNoInit<6, 6, Real> JKJt;
    JKJt = J * K * Jt;  // in-plane stiffness matrix, 6x6

    // stiffness JKJt expanded to 3 dimensions
    type::Mat<9, 9, Real> Ke; // initialized to 0
    // for each 2x2 block i,j
    for (unsigned i = 0; i < 3; i++)
    {
        for (unsigned j = 0; j < 3; j++)
        {
            // copy the block in the expanded matrix
            for (unsigned k = 0; k < 2; k++)
                for (unsigned l = 0; l < 2; l++)
                    Ke[3 * i + k][3 * j + l] = JKJt[2 * i + k][2 * j + l];
        }
    }

    // rotation matrices. TODO: use block-diagonal matrices, more efficient.
    type::Mat<9, 9, Real> RR, RRt; // initialized to 0
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            RR[i][j] = RR[i + 3][j + 3] = RR[i + 6][j + 6] = Rot[i][j];
            RRt[i][j] = RRt[i + 3][j + 3] = RRt[i + 6][j + 6] = Rot[j][i];
        }

    S = RR * Ke;
    SR = S * RRt;
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix* mat, SReal k, unsigned int& offset)
{
    const auto& triangleInf = triangleInfo.getValue();
    const auto& triangles = m_topology->getTriangles();
    const auto nbTriangles = m_topology->getNbTriangles();

    for (sofa::Index i = 0; i < nbTriangles; i++)
    {
        const TriangleInformation& tInfo = triangleInf[i];
        const Triangle& tri = triangles[i];

        type::Mat<9, 9, Real> JKJt(type::NOINIT), RJKJtRt(type::NOINIT);
        computeElementStiffnessMatrix(JKJt, RJKJtRt, tInfo.materialMatrix, tInfo.strainDisplacementMatrix, tInfo.rotation);
        this->addToMatrix(mat, offset, tri, RJKJtRt, -k);
    }
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    type::Mat<9, 9, Real> JKJt, RJKJtRt;
    sofa::type::Mat<3, 3, Real> localMatrix(type::NOINIT);

    constexpr auto S = DataTypes::deriv_total_size; // size of node blocks

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    const auto& triangleInf = triangleInfo.getValue();
    const auto& triangles = m_topology->getTriangles();
    const auto nbTriangles = m_topology->getNbTriangles();

    for (sofa::Index i = 0; i < nbTriangles; i++)
    {
        const TriangleInformation& tInfo = triangleInf[i];
        const Triangle& tri = triangles[i];

        computeElementStiffnessMatrix(JKJt, RJKJtRt, tInfo.materialMatrix, tInfo.strainDisplacementMatrix, tInfo.rotation);

        for (sofa::Index n1 = 0; n1 < Element::size(); ++n1)
        {
            for (sofa::Index n2 = 0; n2 < Element::size(); ++n2)
            {
                RJKJtRt.getsub(S * n1, S * n2, localMatrix); //extract the submatrix corresponding to the coupling of nodes n1 and n2
                dfdx(tri[n1] * S, tri[n2] * S) += -localMatrix;
            }
        }
    }
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::setPoisson(Real val)
{
    if (val < 0)
    {
        msg_warning() << "Input Poisson Coefficient is not possible: " << val << ", setting default value: 0.3";
        type::vector<Real> newP(1, 0.3);
        f_poisson.setValue(newP);
    }
    else {
        type::vector<Real> newP(1, val);
        f_poisson.setValue(newP);
    }
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::setPoissonArray(const type::vector<Real>& values)
{
    auto nbrTri = triangleInfo.getValue().size();
    if (values.size() != nbrTri)
    {
        msg_warning() << "Input Poisson Coefficient array size is not possible: " << values.size() << ", compare to number of triangles: " << nbrTri << ". Values will not be set.";
        return;
    }

    sofa::helper::WriteAccessor< core::objectmodel::Data< type::vector<Real> > > _poisson = f_poisson;
    for (auto id = 0u; id < values.size(); ++id)
    {
        Real val = values[id];
        if (val < 0) {
            msg_warning() << "Input Poisson Coefficient at position: " << id << " is not possible: " << val << ", setting default value: 0.3";
            val = 1000;
        }
        _poisson[id] = val;
    }
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::setYoung(Real val)
{
    if (val < 0)
    {
        msg_warning() << "Input Young Modulus is not possible: " << val << ", setting default value: 1000";
        type::vector<Real> newY(1, 1000);
        f_young.setValue(newY);
    }
    else {
        type::vector<Real> newY(1, val);
        f_young.setValue(newY);
    }
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::setYoungArray(const type::vector<Real>& values)
{
    auto nbrTri = triangleInfo.getValue().size();
    if (values.size() != nbrTri)
    {
        msg_warning() << "Input Young Modulus array size is not possible: " << values.size() << ", compare to number of triangles: " << nbrTri << ". Values will not be set.";
        return;
    }

    sofa::helper::WriteAccessor< core::objectmodel::Data< type::vector<Real> > > _young = f_young;
    for (auto id = 0u; id<values.size(); ++id)
    {
        Real val = values[id];
        if (val < 0) {
            msg_warning() << "Input Young Modulus at position: " << id << " is not possible: " << val << ", setting default value: 1000";
            val = 1000;
        }
        _young[id] = val;
    }
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::setMethod(int val)
{
    if (val != 0 && val != 1)
    {
        msg_warning() << "Input Method is not possible: " << val << ", should be 0 (Large) or 1 (Small). Setting default value: Large";
        method = LARGE;
    }
    else
        method = val;
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::setMethod(const std::string& methodName)
{
    if (methodName == "small")
        method = SMALL;
    else if (methodName == "large")
        method = LARGE;
    else
    {
        msg_warning() << "Input Method is not possible: " << methodName << ", should be 0 (Large) or 1 (Small). Setting default value: Large";
        method = LARGE;
    }
}


// --------------------------------------------------------------------------------------
// --- Get the rotation of node
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::getRotation(Transformation& R, Index nodeIdx)
{
    type::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());
    size_t numNeiTri = m_topology->getTrianglesAroundVertex(nodeIdx).size();
    Transformation r;
    for (size_t i = 0; i < numNeiTri; i++)
    {
        int triIdx = m_topology->getTrianglesAroundVertex(nodeIdx)[i];
        TriangleInformation* tinfo = &triangleInf[triIdx];
        Transformation r01, r21;
        r01 = tinfo->initialTransformation;
        r21 = tinfo->rotation * r01;
        r += r21;
    }
    R = r / static_cast<Real>(numNeiTri);

    //orthogonalization
    Coord ex, ey, ez;
    for (int i = 0; i < 3; i++)
    {
        ex[i] = R[0][i];
        ey[i] = R[1][i];
    }
    ex.normalize();
    ey.normalize();

    ez = cross(ex, ey);
    ez.normalize();

    ey = cross(ez, ex);
    ey.normalize();

    for (int i = 0; i < 3; i++)
    {
        R[0][i] = ex[i];
        R[1][i] = ey[i];
        R[2][i] = ez[i];
    }
    triangleInfo.endEdit();
}

// --------------------------------------------------------------------------------------
// --- Get the rotation of node
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::getRotations()
{
    type::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());
    type::vector<VertexInformation>& vertexInf = *(vertexInfo.beginEdit());
    const int numPoint = m_topology->getNbPoints();
    const int numTri = m_topology->getNbTriangles();

    //reset the rotation matrix
    for (int i = 0; i < numPoint; i++)
    {
        VertexInformation* vinfo = &vertexInf[i];
        vinfo->rotation.clear();
    }

    //add the rotation matrix
    for (int i = 0; i < numTri; i++)
    {
        TriangleInformation* tinfo = &triangleInf[i];
        Transformation r01, r21;
        r01 = tinfo->initialTransformation;
        r21 = tinfo->rotation * r01;

        const Triangle& tri = m_topology->getTriangle(i);
        for (auto idx : tri)
        {
            VertexInformation* vinfo = &vertexInf[idx];
            vinfo->rotation += r21;
        }
    }

    //averaging the rotation matrix
    for (int i = 0; i < numPoint; i++)
    {
        VertexInformation* vinfo = &vertexInf[i];
        size_t numNeiTri = m_topology->getTrianglesAroundVertex(i).size();
        vinfo->rotation /= static_cast<Real>(numNeiTri);

        //orthogonalization
        Coord ex, ey, ez;
        for (int i = 0; i < 3; i++)
        {
            ex[i] = vinfo->rotation[0][i];
            ey[i] = vinfo->rotation[1][i];
        }
        ex.normalize();
        ey.normalize();

        ez = cross(ex, ey);
        ez.normalize();

        ey = cross(ez, ex);
        ey.normalize();

        for (int i = 0; i < 3; i++)
        {
            vinfo->rotation[0][i] = ex[i];
            vinfo->rotation[1][i] = ey[i];
            vinfo->rotation[2][i] = ez[i];
        }
    }
    triangleInfo.endEdit();
    vertexInfo.endEdit();
}


// --------------------------------------------------------------------------------------
// --- Get Fracture Criteria
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::getFractureCriteria(int elementIndex, Deriv& direction, Real& value)
{
    //TODO(dmarchal 2017-05-03) Who wrote this todo ? When will you fix this ? In one year I remove this one.
    /// @todo evaluate the criteria on the current position instead of relying on the computations during the force evaluation (based on the previous position)
    auto triangleInf = sofa::helper::getWriteOnlyAccessor(triangleInfo);

    if ((unsigned)elementIndex < triangleInf.size())
    {
        computePrincipalStress(elementIndex, triangleInf[elementIndex]);
        direction = triangleInf[elementIndex].principalStressDirection;
        value = fabs(triangleInf[elementIndex].maxStress);
        if (value < 0)
        {
            direction.clear();
            value = 0;
        }
    }
    else
    {
        direction.clear();
        value = 0;
    }
}


// --------------------------------------------------------------------------------------------------------
// --- Stiffness = K = J*D*Jt
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStiffness(Stiffness& K, const StrainDisplacement& J, const MaterialStiffness& D)
{
    type::Mat<3, 6, Real> Jt;
    Jt.transpose(J);
    K = J * D * Jt;
}

// --------------------------------------------------------------------------------------
// ---	Compute direction of maximum strain (strain = JtD = BD)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computePrincipalStrain(Index elementIndex, TriangleInformation& triangleInfo)
{
    SOFA_UNUSED(elementIndex);

    Eigen::Matrix<Real, -1, -1> e;
    e.resize(2, 2);

    e(0,0) = triangleInfo.strain[0];
    e(0,1) = triangleInfo.strain[2];
    e(1,0) = triangleInfo.strain[2];
    e(1,1) = triangleInfo.strain[1];
    
    //compute eigenvalues and eigenvectors
    Eigen::JacobiSVD svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const auto& S = svd.singularValues();
    const auto& V = svd.matrixV();

    Coord v(V(0, 0), V(1, 0), 0.0);
    v.normalize();

    triangleInfo.maxStrain = S(0);

    triangleInfo.principalStrainDirection = triangleInfo.rotation * Coord(v[0], v[1], v[2]);
    triangleInfo.principalStrainDirection *= triangleInfo.maxStrain / 100.0;
}

// --------------------------------------------------------------------------------------
// ---	Compute direction of maximum stress (stress = KJtD = KBD)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computePrincipalStress(Index elementIndex, TriangleInformation& triangleInfo)
{
    SOFA_UNUSED(elementIndex);

    Eigen::Matrix<Real, -1, -1> e;
    e.resize(2, 2);

    //voigt notation to symmetric matrix
    e(0,0) = triangleInfo.stress[0];
    e(0,1) = triangleInfo.stress[2];
    e(1,0) = triangleInfo.stress[2];
    e(1,1) = triangleInfo.stress[1];

    //compute eigenvalues and eigenvectors
    Eigen::JacobiSVD svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const auto& S = svd.singularValues();
    const auto& V = svd.matrixV();
    //get the index of the biggest eigenvalue in absolute value
    unsigned int biggestIndex = 0;
    if (fabs(S(0)) > fabs(S(1)))
        biggestIndex = 0;
    else
        biggestIndex = 1;

    //get the eigenvectors corresponding to the biggest eigenvalue
    Coord direction(V(0,biggestIndex), V(1,biggestIndex), 0.0);    
    direction.normalize();

    //Hosford yield criterion
    //for plane stress : 1/2 * ( |S_1|^n + |S_2|^n) + 1/2 * |S_1 - S_2|^n = S_y^n
    //with S_i the principal stresses, n is a material-dependent exponent and S_y is the yield stress in uniaxial tension/compression
    const double n = this->hosfordExponant.getValue();
    triangleInfo.differenceToCriteria = (Real)
            pow(0.5 * (pow((double)fabs(S(0)), n) +  pow((double)fabs(S(1)), n) + pow((double)fabs(S(0) - S(1)),n)), 1.0/ n) - this->criteriaValue.getValue();

    //max stress is the highest eigenvalue
    triangleInfo.maxStress = fabs(S(biggestIndex));

    //the principal stress direction is the eigenvector corresponding to the highest eigenvalue
    Coord principalStressDir = triangleInfo.rotation * direction;//need to rotate to be in global frame instead of local
    principalStressDir *= triangleInfo.maxStress / 100.0;


    //make an average of the n1 and n2 last stress direction to smooth it and avoid discontinuities
    unsigned int n2 = 30;
    unsigned int n1 = 10;
    triangleInfo.lastNStressDirection.push_back(principalStressDir);

    //remove useless data
    if (triangleInfo.lastNStressDirection.size() > n2)
    {
        for (unsigned int i = 0; i < triangleInfo.lastNStressDirection.size() - n2; i++)
            triangleInfo.lastNStressDirection.erase(triangleInfo.lastNStressDirection.begin() + i);
    }

    //make the average
    Coord averageVector2(0.0, 0.0, 0.0);
    Coord averageVector1(0.0, 0.0, 0.0);
    for (unsigned int i = 0; i < triangleInfo.lastNStressDirection.size(); i++)
    {
        averageVector2 = triangleInfo.lastNStressDirection[i] + averageVector2;
        if (i == n1)
            averageVector1 = averageVector2 / n1;
    }
    if (triangleInfo.lastNStressDirection.size())
        averageVector2 /= triangleInfo.lastNStressDirection.size();

    triangleInfo.principalStressDirection = averageVector2;
}

// --------------------------------------------------------------------------------------
// ---	Compute material stiffness
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeMaterialStiffness(int i, Index&/*a*/, Index&/*b*/, Index&/*c*/)
{
    type::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    const type::vector<Real>& youngArray = f_young.getValue();
    const type::vector<Real>& poissonArray = f_poisson.getValue();

    TriangleInformation* tinfo = &triangleInf[i];

    Real y = ((int)youngArray.size() > i) ? youngArray[i] : youngArray[0];
    Real p = ((int)poissonArray.size() > i) ? poissonArray[i] : poissonArray[0];

    tinfo->materialMatrix[0][0] = 1;
    tinfo->materialMatrix[0][1] = p;//poissonArray[i];//f_poisson.getValue();
    tinfo->materialMatrix[0][2] = 0;
    tinfo->materialMatrix[1][0] = p;//poissonArray[i];//f_poisson.getValue();
    tinfo->materialMatrix[1][1] = 1;
    tinfo->materialMatrix[1][2] = 0;
    tinfo->materialMatrix[2][0] = 0;
    tinfo->materialMatrix[2][1] = 0;
    tinfo->materialMatrix[2][2] = (1.0f - p) * 0.5f;//poissonArray[i]);

    tinfo->materialMatrix *= (y / (1.0f - p * p)) * tinfo->area;

    triangleInfo.endEdit();
}


/// Compute current stress
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStress(type::Vec<3, Real>& stress, Index elementIndex)
{
    Displacement D;
    StrainDisplacement J;
    type::Vec<3, Real> strain;
    Transformation R_0_2, R_2_0;
    const VecCoord& p = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const Triangle& tri = m_topology->getTriangle(elementIndex);
    Index a = tri[0];
    Index b = tri[1];
    Index c = tri[2];

    auto triangleInf = sofa::helper::getWriteOnlyAccessor(triangleInfo);
    if (method == SMALL)
    {
        // classic linear elastic method
        Coord deforme_b = p[b] - p[a];
        Coord deforme_c = p[c] - p[a];
        m_triangleUtils.computeDisplacementSmall(D, triangleInf[elementIndex].rotatedInitialElements, deforme_b, deforme_c);

        try
        {
            m_triangleUtils.computeStrainDisplacementLocal(J, deforme_b, deforme_c);
        }
        catch (const std::exception& e)
        {
            msg_error() << e.what();
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
        m_triangleUtils.computeStrain(strain, J, D, true);
        m_triangleUtils.computeStress(stress, triangleInf[elementIndex].materialMatrix, strain, true);
    }
    else
    {
        // co-rotational method
        // first, compute rotation matrix into co-rotational frame
        m_triangleUtils.computeRotationLarge(R_0_2, p[a], p[b], p[c]);

        // then compute displacement in this frame
        m_triangleUtils.computeDisplacementLarge(D, R_0_2, triangleInf[elementIndex].rotatedInitialElements, p[a], p[b], p[c]);

        // and compute postions of a, b, c in the co-rotational frame
        Coord A = Coord(0, 0, 0); SOFA_UNUSED(A);
        Coord B = R_0_2 * (p[b] - p[a]);
        Coord C = R_0_2 * (p[c] - p[a]);

        try
        {
            m_triangleUtils.computeStrainDisplacementLocal(J, B, C);
        }
        catch (const std::exception& e)
        {
            msg_error() << e.what();
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
        m_triangleUtils.computeStrain(strain, J, D, _anisotropicMaterial);
        m_triangleUtils.computeStress(stress, triangleInf[elementIndex].materialMatrix, strain, _anisotropicMaterial);
    }
    // store newly computed values for next time
    R_2_0.transpose(R_0_2);
    triangleInf[elementIndex].strainDisplacementMatrix = J;
    triangleInf[elementIndex].rotation = R_2_0;
    triangleInf[elementIndex].strain = strain;
    triangleInf[elementIndex].stress = stress;
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStressPerVertex()
{
    auto vertexInf = sofa::helper::getWriteOnlyAccessor(vertexInfo);
    const auto& triangleInf = sofa::helper::getReadAccessor(triangleInfo);

    m_minStress = std::numeric_limits<Real>::max();
    m_maxStress = std::numeric_limits<Real>::lowest();
    for (unsigned int i = 0; i < vertexInf.size(); i++)
    {
        const core::topology::BaseMeshTopology::TrianglesAroundVertex& triangles = m_topology->getTrianglesAroundVertex(i);
        Real averageStress = 0.0;
        double sumArea = 0.0;
        for (auto triID : triangles)
        {
            if (triangleInf[triID].area)
            {
                averageStress += (fabs(triangleInf[triID].maxStress) * triangleInf[triID].area);
                sumArea += triangleInf[triID].area;
            }
        }
        if (sumArea)
            averageStress /= sumArea;

        vertexInf[i].stress = averageStress;
        if (averageStress < m_minStress)
            m_minStress = averageStress;
        if (averageStress > m_maxStress)
            m_maxStress = averageStress;
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------
// ---	Compute value of stress along a given direction (typically the fiber direction and transverse direction in anisotropic materials)
// ----------------------------------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStressAlongDirection(Real& stress_along_dir, Index elementIndex, const Coord& dir, const type::Vec<3, Real>& stress)
{
    type::Mat<3, 3, Real> R, Rt;

    type::vector<TriangleInformation>& triangleInf = *(this->triangleInfo.beginEdit());

    // transform 'dir' into local coordinates
    R = triangleInf[elementIndex].rotation;
    Rt.transpose(R);
    Coord dir_local = Rt * dir;
    dir_local[2] = 0; // project direction
    dir_local.normalize();

    // compute stress along specified direction 'dir'
    Real cos_theta = dir_local[0];
    Real sin_theta = dir_local[1];
    stress_along_dir = stress[0] * cos_theta * cos_theta + stress[1] * sin_theta * sin_theta + stress[2] * 2 * cos_theta * sin_theta;
    triangleInfo.endEdit();
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStressAcrossDirection(Real& stress_across_dir, Index elementIndex, const Coord& dir, const type::Vec<3, Real>& stress)
{
    const Triangle& tri = m_topology->getTriangle(elementIndex);
    Index a = tri[0];
    Index b = tri[1];
    Index c = tri[2];
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Coord n = cross(x[b] - x[a], x[c] - x[a]);
    Coord dir_t = cross(dir, n);
    this->computeStressAlongDirection(stress_across_dir, elementIndex, dir_t, stress);
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStressAcrossDirection(Real& stress_across_dir, Index elementIndex, const Coord& dir)
{
    const Triangle& tri = m_topology->getTriangle(elementIndex);
    Index a = tri[0];
    Index b = tri[1];
    Index c = tri[2];
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Coord n = cross(x[b] - x[a], x[c] - x[a]);
    Coord dir_t = cross(dir, n);
    this->computeStressAlongDirection(stress_across_dir, elementIndex, dir_t);
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStressAlongDirection(Real& stress_along_dir, Index elementIndex, const Coord& dir)
{
    type::Vec<3, Real> stress;
    this->computeStress(stress, elementIndex);
    this->computeStressAlongDirection(stress_along_dir, elementIndex, dir, stress);
}




// --------------------------------------------------------------------------------------
// --- Apply functions
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::applyStiffnessSmall(VecCoord& v, Real h, const VecCoord& x, const Real& kFactor)
{
    Displacement dX;
    type::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());
    const unsigned int nbTriangles = m_topology->getNbTriangles();
    for (unsigned int i = 0; i < nbTriangles; i++)
    {
        TriangleInformation& tInfo = triangleInf[i];
        const Triangle& tri = m_topology->getTriangle(i);
        Index a = tri[0];
        Index b = tri[1];
        Index c = tri[2];

        dX[0] = x[a][0];
        dX[1] = x[a][1];

        dX[2] = x[b][0];
        dX[3] = x[b][1];

        dX[4] = x[c][0];
        dX[5] = x[c][1];


        // compute strain
        type::Vec<3, Real> strain;
        m_triangleUtils.computeStrain(strain, tInfo.strainDisplacementMatrix, dX, true);

        // compute stress
        type::Vec<3, Real> stress;
        m_triangleUtils.computeStress(stress, tInfo.materialMatrix, strain, true);

        // compute force on element
        Displacement F;
        F = tInfo.strainDisplacementMatrix * stress;

        v[a] += (Coord(-h * F[0], -h * F[1], 0)) * kFactor;
        v[b] += (Coord(-h * F[2], -h * F[3], 0)) * kFactor;
        v[c] += (Coord(-h * F[4], -h * F[5], 0)) * kFactor;
    }
    triangleInfo.endEdit();
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::applyStiffness(VecCoord& v, Real h, const VecCoord& x, const Real& kFactor)
{
    if (method == SMALL)
    {
        applyStiffnessSmall(v, h, x, kFactor);
    }
    else
    {
        applyStiffnessLarge(v, h, x, kFactor);
    }
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::applyStiffnessLarge(VecCoord& v, Real h, const VecCoord& x, const Real& kFactor)
{
    Displacement dX;
    Coord x_2;
    type::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginWriteOnly());

    const unsigned int nbTriangles = m_topology->getNbTriangles();
    const auto& triangles = m_topology->getTriangles();
    for (unsigned int i = 0; i < nbTriangles; i++)
    {
        TriangleInformation& tInfo = triangleInf[i];
        const Element& tri = triangles[i];
        const Index a = tri[0];
        const Index b = tri[1];
        const Index c = tri[2];

        Transformation R_0_2;
        R_0_2.transpose(tInfo.rotation);

        x_2 = R_0_2 * x[a];
        dX[0] = x_2[0];
        dX[1] = x_2[1];

        x_2 = R_0_2 * x[b];
        dX[2] = x_2[0];
        dX[3] = x_2[1];

        x_2 = R_0_2 * x[c];
        dX[4] = x_2[0];
        dX[5] = x_2[1];

        // compute strain
        type::Vec<3, Real> strain;
        m_triangleUtils.computeStrain(strain, tInfo.strainDisplacementMatrix, dX, _anisotropicMaterial);

        // compute stress
        type::Vec<3, Real> stress;
        m_triangleUtils.computeStress(stress, tInfo.materialMatrix, strain, _anisotropicMaterial);

        // compute force on element, in local frame
        Displacement F;
        m_triangleUtils.computeForceLarge(F, tInfo.strainDisplacementMatrix, stress);

        v[a] += (tInfo.rotation * Coord(-h * F[0], -h * F[1], 0)) * kFactor;
        v[b] += (tInfo.rotation * Coord(-h * F[2], -h * F[3], 0)) * kFactor;
        v[c] += (tInfo.rotation * Coord(-h * F[4], -h * F[5], 0)) * kFactor;
    }
    triangleInfo.endEdit();
}


// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateForceSmall(VecCoord& f, const VecCoord& p)
{
    type::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginWriteOnly());
    const unsigned int nbTriangles = m_topology->getNbTriangles();
    for (unsigned int i = 0; i < nbTriangles; i++)
    {
        TriangleInformation& tInfo = triangleInf[i];
        const Element& tri = m_topology->getTriangle(i);
        Index a = tri[0];
        Index b = tri[1];
        Index c = tri[2];

        Coord deforme_a, deforme_b, deforme_c;
        deforme_b = p[b] - p[a];
        deforme_c = p[c] - p[a];
        deforme_a = Coord(0, 0, 0);

        // displacements
        Displacement Depl(type::NOINIT);
        m_triangleUtils.computeDisplacementSmall(Depl, tInfo.rotatedInitialElements, deforme_b, deforme_c);

        StrainDisplacement J(type::NOINIT);

        try
        {
            m_triangleUtils.computeStrainDisplacementGlobal(J, deforme_a, deforme_b, deforme_c);
        }
        catch (const std::exception& e)
        {
            msg_error() << e.what();
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            break;
        }        

        // compute strain
        type::Vec<3, Real> strain(type::NOINIT);
        m_triangleUtils.computeStrain(strain, J, Depl, true);

        // compute stress
        type::Vec<3, Real> stress(type::NOINIT);
        m_triangleUtils.computeStress(stress, tInfo.materialMatrix, strain, true);

        // compute force on element
        Displacement F = J * stress;

        f[a] += Coord(F[0], F[1], 0);
        f[b] += Coord(F[2], F[3], 0);
        f[c] += Coord(F[4], F[5], 0);

        // store newly computed values for next time
        tInfo.strainDisplacementMatrix = J;
        tInfo.strain = strain;
        tInfo.stress = stress;
    }

    triangleInfo.endEdit();
}


// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateForceLarge(VecCoord& f, const VecCoord& p)
{
    type::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginWriteOnly());
    const sofa::Size nbTriangles = m_topology->getNbTriangles();
    const auto& triangles = m_topology->getTriangles();
    for (sofa::Index i = 0; i < nbTriangles; i++)
    {
        TriangleInformation& tInfo = triangleInf[i];
        const Triangle& tri = triangles[i];
        const Coord& pA = p[ tri[0] ];
        const Coord& pB = p[ tri[1] ];
        const Coord& pC = p[ tri[2] ];

        // co-rotational method
        // first, compute rotation matrix into co-rotational frame
        Transformation R_2_0, R_0_2;
        m_triangleUtils.computeRotationLarge(R_0_2, pA, pB, pC);

        // then compute displacement in this frame
        Displacement Depl(type::NOINIT);
        m_triangleUtils.computeDisplacementLarge(Depl, R_0_2, tInfo.rotatedInitialElements, pA, pB, pC);

        // positions of the deformed points in the local frame
        const Coord deforme_b = R_0_2 * (pB - pA);
        const Coord deforme_c = R_0_2 * (pC - pA);

        // Strain-displacement matrix
        StrainDisplacement J(type::NOINIT);
        try
        {
            m_triangleUtils.computeStrainDisplacementLocal(J, deforme_b, deforme_c);
        }
        catch (const std::exception& e)
        {
            msg_error() << e.what();
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            break;
        }        

        // compute strain
        type::Vec<3, Real> strain(type::NOINIT);
        m_triangleUtils.computeStrain(strain, J, Depl, _anisotropicMaterial);

        // compute stress
        type::Vec<3, Real> stress(type::NOINIT);
        m_triangleUtils.computeStress(stress, tInfo.materialMatrix, strain, _anisotropicMaterial);

        // compute force on element, in local frame
        Displacement F(type::NOINIT);
        m_triangleUtils.computeForceLarge(F, J, stress);

        // transform force back into global ref. frame
        R_2_0.transpose(R_0_2);
        f[tri[0]] += R_2_0 * Coord(F[0], F[1], 0);
        f[tri[1]] += R_2_0 * Coord(F[2], F[3], 0);
        f[tri[2]] += R_2_0 * Coord(F[4], F[5], 0);

        //Stiffness K;
        //computeStiffness(K, J, tInfo.materialMatrix);

        // store newly computed values for next time
        tInfo.strainDisplacementMatrix = J;
        tInfo.strain = strain;
        tInfo.stress = stress;
        tInfo.rotation = R_2_0;
        //tInfo.stiffness = K;
    }
    triangleInfo.endEdit();
}


// --------------------------------------------------------------------------------------
// --- AddForce and AddDForce methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
    VecDeriv& f1 = *f.beginEdit();
    const VecCoord& x1 = x.getValue();

    f1.resize(x1.size());
    if (method == SMALL)
    {
        accumulateForceSmall(f1, x1);
    }
    else
    {
        accumulateForceLarge(f1, x1);
    }
    f.endEdit();


    if (f_computePrincipalStress.getValue() || p_computeDrawInfo)
    {
        const unsigned int nbTriangles = m_topology->getNbTriangles();
        auto triangleInf = sofa::helper::getWriteOnlyAccessor(triangleInfo);
        for (unsigned int i = 0; i < nbTriangles; ++i)
            computePrincipalStress(i, triangleInf[i]);

        if (showStressValue.getValue()) // if true will compute averageStress per point
        {
            computeStressPerVertex();
        }
    }
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    VecDeriv& df1 = *df.beginEdit();
    const VecDeriv& dx1 = dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    Real h = 1;
    df1.resize(dx1.size());

    if (method == SMALL)
        applyStiffnessSmall(df1, h, dx1, kFactor);
    else
        applyStiffnessLarge(df1, h, dx1, kFactor);

    df.endEdit();
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}


// --------------------------------------------------------------------------------------
// --- Display methods
// --------------------------------------------------------------------------------------
template<class DataTypes>
void TriangularFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) {
        p_computeDrawInfo = false;
        return;
    }

    p_computeDrawInfo = showStressVector.getValue() || showStressValue.getValue() || showFracturableTriangles.getValue();

    if (!p_computeDrawInfo) {
        return;
    }

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    vparams->drawTool()->disableLighting();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const type::vector<TriangleInformation>& triangleInf = triangleInfo.getValue();
    const auto& triangles = m_topology->getTriangles();
    const Size nbTriangles = triangles.size();

    if (showStressVector.getValue())
    {
        std::vector<sofa::type::Vec3> vertices;
        for (Size i = 0; i < nbTriangles; ++i)
        {
            const Triangle& tri = triangles[i];
            Index a = tri[0];
            Index b = tri[1];
            Index c = tri[2];
            Coord center = (x[a] + x[b] + x[c]) / 3;
            Coord d = triangleInf[i].principalStressDirection * 2.5; //was 0.25
            vertices.push_back(sofa::type::Vec3(center));
            vertices.push_back(sofa::type::Vec3(center + d));
        }
        vparams->drawTool()->drawLines(vertices, 1, sofa::type::RGBAColor(1, 0, 1, 1));
    }

    if (showStressValue.getValue())
    {
        const type::vector<VertexInformation>& vertexInf = vertexInfo.getValue();
        std::vector<sofa::type::Vec3> vertices;

        std::vector<sofa::type::RGBAColor> colorVector;
 
        auto evalColor = p_drawColorMap->getEvaluator(m_minStress, m_maxStress);
        for (Size i = 0; i < nbTriangles; ++i)
        {
            const Triangle& tri = triangles[i];
            Index a = tri[0];
            Index b = tri[1];
            Index c = tri[2];

            colorVector.push_back(evalColor(vertexInf[a].stress));
            vertices.push_back(sofa::type::Vec3(x[a]));
            colorVector.push_back(evalColor(vertexInf[b].stress));
            vertices.push_back(sofa::type::Vec3(x[b]));
            colorVector.push_back(evalColor(vertexInf[c].stress));
            vertices.push_back(sofa::type::Vec3(x[c]));
        }
        vparams->drawTool()->drawTriangles(vertices, colorVector);
        vertices.clear();
        colorVector.clear();
        vertexInfo.endEdit();
    }

    if (showFracturableTriangles.getValue())
    {
        std::vector<sofa::type::RGBAColor> colorVector;
        std::vector<sofa::type::Vec3> vertices;
        sofa::type::RGBAColor color;

        Real maxDifference = std::numeric_limits<Real>::min();
        Real minDifference = std::numeric_limits<Real>::max();
        for (Size i = 0; i < nbTriangles; i++)
        {
            if (triangleInf[i].differenceToCriteria > 0)
            {
                if (triangleInf[i].differenceToCriteria > maxDifference)
                    maxDifference = triangleInf[i].differenceToCriteria;

                if (triangleInf[i].differenceToCriteria < minDifference)
                    minDifference = triangleInf[i].differenceToCriteria;
            }
        }

        for (Size i = 0; i < nbTriangles; i++)
        {
            if (triangleInf[i].differenceToCriteria > 0)
            {
                color = sofa::type::RGBAColor(float(0.4 + 0.4 * (triangleInf[i].differenceToCriteria - minDifference) / (maxDifference - minDifference)), 0.0f, 0.0f, 0.5f);
                const Triangle& tri = triangles[i];
                Index a = tri[0];
                Index b = tri[1];
                Index c = tri[2];

                colorVector.push_back(color);
                vertices.push_back(sofa::type::Vec3(x[a]));
                colorVector.push_back(color);
                vertices.push_back(sofa::type::Vec3(x[b]));
                colorVector.push_back(color);
                vertices.push_back(sofa::type::Vec3(x[c]));
            }
        }
        vparams->drawTool()->drawTriangles(vertices, colorVector);
        vertices.clear();
        colorVector.clear();
    }


}

} // namespace sofa::component::solidmechanics::fem::elastic
