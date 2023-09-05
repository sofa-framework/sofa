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

#include <sofa/component/solidmechanics/fem/elastic/config.h>

#include <sofa/component/solidmechanics/fem/elastic/TriangleFEMForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes>
TriangleFEMForceField<DataTypes>::
TriangleFEMForceField()
    : _indexedElements(nullptr)
    , _initialPoints(initData(&_initialPoints, "initialPoints", "Initial Position"))
    , m_topology(nullptr)
    , method(LARGE)
    , f_method(initData(&f_method,std::string("large"),"method","large: large displacements, small: small displacements"))
    , f_poisson(initData(&f_poisson,Real(0.3),"poissonRatio","Poisson ratio in Hooke's law"))
    , f_young(initData(&f_young,Real(1000.),"youngModulus","Young modulus in Hooke's law"))
    , f_thickness(initData(&f_thickness,Real(1.),"thickness","Thickness of the elements"))
    , f_planeStrain(initData(&f_planeStrain,false,"planeStrain","Plane strain or plane stress assumption"))
    , l_topology(initLink("topology", "link to the topology container"))    
{
    f_poisson.setRequired(true);
    f_young.setRequired(true);
}

template <class DataTypes>
TriangleFEMForceField<DataTypes>::~TriangleFEMForceField()
{
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::init()
{
    this->Inherited::init();

    // checking inputs using setter
    setMethod(f_method.getValue());
    setPoisson(f_poisson.getValue());
    setYoung(f_young.getValue());

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

    if (m_topology->getTriangles().empty() && m_topology->getQuads().empty())
    {
        msg_warning() << "No triangles found in linked Topology.";
        _indexedElements = &(m_topology->getTriangles());
    }
    else if (!m_topology->getTriangles().empty())
    {
        msg_info() << "Init using triangles mesh: " << m_topology->getTriangles().size() << " triangles.";
        _indexedElements = &(m_topology->getTriangles());
    }
    else if (m_topology->getNbQuads() > 0)
    {
        msg_info() << "Init using quads mesh: " << m_topology->getNbQuads() * 2 << " triangles.";
        sofa::core::topology::BaseMeshTopology::SeqTriangles* trias = new sofa::core::topology::BaseMeshTopology::SeqTriangles;
        const int nbcubes = m_topology->getNbQuads();
        trias->reserve(nbcubes * 2);
        for (int i = 0; i < nbcubes; i++)
        {
            sofa::core::topology::BaseMeshTopology::Quad q = m_topology->getQuad(i);
            trias->push_back(Element(q[0], q[1], q[2]));
            trias->push_back(Element(q[0], q[2], q[3]));
        }
        _indexedElements = trias;
    }

    if (_initialPoints.getValue().size() == 0)
    {
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints.setValue(p);
    }

    _strainDisplacements.resize(_indexedElements->size());
    _rotations.resize(_indexedElements->size());

    if (method == SMALL) 
    {
        initSmall();
    }
    else 
    {
        initLarge();
    }

    computeMaterialStiffnesses();
}



template <class DataTypes>
void TriangleFEMForceField<DataTypes>::reinit()
{
    if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;

    if (method == SMALL) 
    {
        //    initSmall();  // useful ? The rotations are recomputed later
    }
    else 
    {
        initLarge(); // compute the per-element strain-displacement matrices
    }

    computeMaterialStiffnesses();
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
    VecDeriv& f1 = *f.beginEdit();
    const VecCoord& x1 = x.getValue();

    f1.resize(x1.size());

    if (method == SMALL)
    {
        accumulateForceSmall(f1, x1, true);
    }
    else
    {
        accumulateForceLarge(f1, x1, true);
    }

    f.endEdit();
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    VecDeriv& df1 = *df.beginEdit();
    const VecDeriv& dx1 = dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    Real h = 1;
    df1.resize(dx1.size());

    if (method == SMALL)
    {
        applyStiffnessSmall(df1, h, dx1, kFactor);
    }
    else
    {
        applyStiffnessLarge(df1, h, dx1, kFactor);
    }

    df.endEdit();
}

template <class DataTypes>
void TriangleFEMForceField<DataTypes>::applyStiffness(VecCoord& v, Real h, const VecCoord& x, const Real& kFactor)
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

template <class DataTypes>
void TriangleFEMForceField<DataTypes>::computeMaterialStiffnesses()
{
    _materialsStiffnesses.resize(_indexedElements->size());
    const VecCoord& p = _initialPoints.getValue();

    const Real _p = f_poisson.getValue();
    const Real _1_p = 1 - _p;
    const Real Estrain = f_young.getValue() / ((1 + _p) * (1 - 2 * _p));
    const Real Estress = f_young.getValue() / (1 - _p * _p);

    for (unsigned i = 0; i < _indexedElements->size(); ++i)
    {
        Index a = (*_indexedElements)[i][0];
        Index b = (*_indexedElements)[i][1];
        Index c = (*_indexedElements)[i][2];
        const Real triangleVolume = (Real)0.5 * f_thickness.getValue() * cross(p[b] - p[a], p[c] - p[a]).norm();

        if (f_planeStrain.getValue() == true)
        {
            _materialsStiffnesses[i][0][0] = _1_p;
            _materialsStiffnesses[i][0][1] = _p;
            _materialsStiffnesses[i][0][2] = 0;
            _materialsStiffnesses[i][1][0] = _p;
            _materialsStiffnesses[i][1][1] = _1_p;
            _materialsStiffnesses[i][1][2] = 0;
            _materialsStiffnesses[i][2][0] = 0;
            _materialsStiffnesses[i][2][1] = 0;
            _materialsStiffnesses[i][2][2] = 0.5f - _p;

            _materialsStiffnesses[i] *= Estrain * triangleVolume;
        }
        else // plane stress
        {
            _materialsStiffnesses[i][0][0] = 1;
            _materialsStiffnesses[i][0][1] = _p;
            _materialsStiffnesses[i][0][2] = 0;
            _materialsStiffnesses[i][1][0] = _p;
            _materialsStiffnesses[i][1][1] = 1;
            _materialsStiffnesses[i][1][2] = 0;
            _materialsStiffnesses[i][2][0] = 0;
            _materialsStiffnesses[i][2][1] = 0;
            _materialsStiffnesses[i][2][2] = 0.5f * (_1_p);

            _materialsStiffnesses[i] *= Estress * triangleVolume;
        }
    }
}


/*
** SMALL DEFORMATION METHODS
*/
template <class DataTypes>
void TriangleFEMForceField<DataTypes>::initSmall()
{
    Transformation identity;
    identity[0][0] = identity[1][1] = identity[2][2] = 1;
    identity[0][1] = identity[0][2] = 0;
    identity[1][0] = identity[1][2] = 0;
    identity[2][0] = identity[2][1] = 0;
    for (unsigned i = 0; i < _indexedElements->size(); ++i)
    {
        _rotations[i] = identity;
    }
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::accumulateForceSmall(VecCoord& f, const VecCoord& p, bool implicit)
{
    typename VecElement::const_iterator it;
    unsigned int elementIndex(0);
    for (it = _indexedElements->begin(); it != _indexedElements->end(); ++it, ++elementIndex)
    {
        Index a = (*_indexedElements)[elementIndex][0];
        Index b = (*_indexedElements)[elementIndex][1];
        Index c = (*_indexedElements)[elementIndex][2];

        const auto deforme_b = p[b] - p[a];
        const auto deforme_c = p[c] - p[a];

        // displacements
        Displacement Depl(type::NOINIT);
        m_triangleUtils.computeDisplacementSmall(Depl, _rotatedInitialElements[elementIndex], deforme_b, deforme_c);

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
        
        if (implicit)
            _strainDisplacements[elementIndex] = J;

        // compute strain
        type::Vec<3, Real> strain(type::NOINIT);
        m_triangleUtils.computeStrain(strain, J, Depl, true);

        // compute stress
        type::Vec<3, Real> stress(type::NOINIT);
        m_triangleUtils.computeStress(stress, _materialsStiffnesses[elementIndex], strain, true);

        // compute force on element
        const Displacement F = J * stress;

        f[a] += Coord(F[0], F[1], 0);
        f[b] += Coord(F[2], F[3], 0);
        f[c] += Coord(F[4], F[5], 0);
    }
}

template <class DataTypes>
void TriangleFEMForceField<DataTypes>::applyStiffnessSmall(VecCoord& v, Real h, const VecCoord& x, const Real& kFactor)
{
    typename VecElement::const_iterator it;
    unsigned int i(0);
    for (it = _indexedElements->begin(); it != _indexedElements->end(); ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];

        Displacement dX;

        dX[0] = x[a][0];
        dX[1] = x[a][1];

        dX[2] = x[b][0];
        dX[3] = x[b][1];

        dX[4] = x[c][0];
        dX[5] = x[c][1];


        // compute strain
        type::Vec<3, Real> strain(type::NOINIT);
        m_triangleUtils.computeStrain(strain, _strainDisplacements[i], dX, true);

        // compute stress
        type::Vec<3, Real> stress(type::NOINIT);
        m_triangleUtils.computeStress(stress, _materialsStiffnesses[i], strain, true);

        // compute force on element
        const Displacement F = _strainDisplacements[i] * stress;

        v[a] += Coord(-h * F[0], -h * F[1], 0) * kFactor;
        v[b] += Coord(-h * F[2], -h * F[3], 0) * kFactor;
        v[c] += Coord(-h * F[4], -h * F[5], 0) * kFactor;
    }
}


/*
** LARGE DEFORMATION METHODS
*/

template <class DataTypes>
void TriangleFEMForceField<DataTypes>::initLarge()
{
    _rotatedInitialElements.resize(_indexedElements->size());

    typename VecElement::const_iterator it;
    unsigned int i(0);
    const VecCoord& pos = _initialPoints.getValue();

    for (it = _indexedElements->begin(); it != _indexedElements->end(); ++it, ++i)
    {
        const Coord& pA = pos[(*it)[0]];
        const Coord& pB = pos[(*it)[1]];
        const Coord& pC = pos[(*it)[2]];

        // Rotation matrix (transpose of initial triangle/world)
        // first vector on first edge
        // second vector in the plane of the two first edges
        // third vector orthogonal to first and second
        Transformation R_0_1(type::NOINIT);
        m_triangleUtils.computeRotationLarge(R_0_1, pA, pB, pC);
        _rotations[i].transpose(R_0_1);

        // coordinates of the triangle vertices in their local frames
        _rotatedInitialElements[i][0] = R_0_1 * pA;
        _rotatedInitialElements[i][1] = R_0_1 * pB;
        _rotatedInitialElements[i][2] = R_0_1 * pC;
        // set the origin of the local frame at vertex a
        _rotatedInitialElements[i][1] -= _rotatedInitialElements[i][0];
        _rotatedInitialElements[i][2] -= _rotatedInitialElements[i][0];
        _rotatedInitialElements[i][0] = Coord(0, 0, 0);

        try
        {
            m_triangleUtils.computeStrainDisplacementLocal(_strainDisplacements[i], _rotatedInitialElements[i][1], _rotatedInitialElements[i][2]);
        }
        catch (const std::exception& e)
        {
            msg_error() << e.what();
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }
}

template <class DataTypes>
void TriangleFEMForceField<DataTypes>::accumulateForceLarge(VecCoord& f, const VecCoord& p, bool implicit)
{
    typename VecElement::const_iterator it;
    unsigned int elementIndex(0);
    for (it = _indexedElements->begin(); it != _indexedElements->end(); ++it, ++elementIndex)
    {
        // triangle vertex indices
        const Index a = (*_indexedElements)[elementIndex][0];
        const Index b = (*_indexedElements)[elementIndex][1];
        const Index c = (*_indexedElements)[elementIndex][2];

        const Coord& pA = p[a];
        const Coord& pB = p[b];
        const Coord& pC = p[c];

        // Rotation matrix (deformed and displaced Triangle/world)
        Transformation R_2_0(type::NOINIT), R_0_2(type::NOINIT);
        m_triangleUtils.computeRotationLarge(R_0_2, pA, pB, pC);

        // positions of the deformed points in the local frame
        const Coord deforme_b = R_0_2 * (pB - pA);
        const Coord deforme_c = R_0_2 * (pC - pA);

        // displacements in the local frame
        Displacement Depl(type::NOINIT);
        m_triangleUtils.computeDisplacementLarge(Depl, R_0_2, _rotatedInitialElements[elementIndex], pA, pB, pC);

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
        m_triangleUtils.computeStrain(strain, J, Depl, false);

        // compute stress
        type::Vec<3, Real> stress(type::NOINIT);
        m_triangleUtils.computeStress(stress, _materialsStiffnesses[elementIndex], strain, false);

        // compute force on element, in local frame
        Displacement F(type::NOINIT);
        m_triangleUtils.computeForceLarge(F, J, stress);

        // project forces to world frame
        R_2_0.transpose(R_0_2);
        f[a] += R_2_0 * Coord(F[0], F[1], 0);
        f[b] += R_2_0 * Coord(F[2], F[3], 0);
        f[c] += R_2_0 * Coord(F[4], F[5], 0);

        // store for re-use in matrix-vector products
        if (implicit)
        {
            _strainDisplacements[elementIndex] = J;
            _rotations[elementIndex] = R_2_0;
        }
    }
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::applyStiffnessLarge(VecCoord& v, Real h, const VecCoord& x, const Real& kFactor)
{
    typename VecElement::const_iterator it;
    unsigned int i(0);

    for (it = _indexedElements->begin(); it != _indexedElements->end(); ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];

        Transformation R_0_2(type::NOINIT);
        R_0_2.transpose(_rotations[i]);

        Displacement dX(type::NOINIT);
        
        Coord x_2 = R_0_2 * x[a];
        dX[0] = x_2[0];
        dX[1] = x_2[1];

        x_2 = R_0_2 * x[b];
        dX[2] = x_2[0];
        dX[3] = x_2[1];

        x_2 = R_0_2 * x[c];
        dX[4] = x_2[0];
        dX[5] = x_2[1];

        // compute strain
        type::Vec<3, Real> strain(type::NOINIT);
        m_triangleUtils.computeStrain(strain, _strainDisplacements[i], dX, false);

        // compute stress
        type::Vec<3, Real> stress(type::NOINIT);
        m_triangleUtils.computeStress(stress, _materialsStiffnesses[i], strain, false);

        // compute force on element, in local frame
        Displacement F(type::NOINIT);
        m_triangleUtils.computeForceLarge(F, _strainDisplacements[i], stress);

        v[a] += (_rotations[i] * Coord(-h * F[0], -h * F[1], 0)) * kFactor;
        v[b] += (_rotations[i] * Coord(-h * F[2], -h * F[3], 0)) * kFactor;
        v[c] += (_rotations[i] * Coord(-h * F[4], -h * F[5], 0)) * kFactor;
    }
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if (!onlyVisible) return;
    if (!this->mstate) return;

    const auto bbox = this->mstate->computeBBox(); //this may compute twice the mstate bbox, but there is no way to determine if the bbox has already been computed
    this->f_bbox.setValue(std::move(bbox));
}

template<class DataTypes>
void TriangleFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    std::vector<sofa::type::RGBAColor> colorVector;
    std::vector<sofa::type::Vec3> vertices;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    typename VecElement::const_iterator it;
    for (it = _indexedElements->begin(); it != _indexedElements->end(); ++it)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];

        colorVector.push_back(sofa::type::RGBAColor::green());
        vertices.push_back(sofa::type::Vec3(x[a]));
        colorVector.push_back(sofa::type::RGBAColor(0, 0.5, 0.5, 1));
        vertices.push_back(sofa::type::Vec3(x[b]));
        colorVector.push_back(sofa::type::RGBAColor(0, 0, 1, 1));
        vertices.push_back(sofa::type::Vec3(x[c]));
    }
    vparams->drawTool()->drawTriangles(vertices, colorVector);


}


template<class DataTypes>
void TriangleFEMForceField<DataTypes>::computeElementStiffnessMatrix(StiffnessMatrix& S, StiffnessMatrix& SR, const MaterialStiffness& K, const StrainDisplacement& J, const Transformation& Rot)
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


template<class DataTypes>
void TriangleFEMForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix *mat, SReal k, unsigned int &offset)
{
    for (unsigned i = 0; i < _indexedElements->size(); i++)
    {
        StiffnessMatrix JKJt(type::NOINIT), RJKJtRt(type::NOINIT);
        computeElementStiffnessMatrix(JKJt, RJKJtRt, _materialsStiffnesses[i], _strainDisplacements[i], _rotations[i]);
        this->addToMatrix(mat, offset, (*_indexedElements)[i], RJKJtRt, -k);
    }
}

template <class DataTypes>
void TriangleFEMForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    StiffnessMatrix JKJt, RJKJtRt;
    sofa::type::Mat<3, 3, Real> localMatrix(type::NOINIT);

    constexpr auto S = DataTypes::deriv_total_size; // size of node blocks
    unsigned int i = 0;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (const auto nodeIndex : *_indexedElements)
    {
        computeElementStiffnessMatrix(JKJt, RJKJtRt, _materialsStiffnesses[i], _strainDisplacements[i], _rotations[i]);

        for (sofa::Index n1 = 0; n1 < Element::size(); ++n1)
        {
            for (sofa::Index n2 = 0; n2 < Element::size(); ++n2)
            {
                RJKJtRt.getsub(S * n1, S * n2, localMatrix); //extract the submatrix corresponding to the coupling of nodes n1 and n2
                dfdx(nodeIndex[n1] * S, nodeIndex[n2] * S) += -localMatrix;
            }
        }
        ++i;
    }
}

template <class DataTypes>
void TriangleFEMForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void TriangleFEMForceField<DataTypes>::setPoisson(Real val)
{
    if (val < 0)
    {
        msg_warning() << "Input Poisson Coefficient is not possible: " << val << ", setting default value: 0.3";
        f_poisson.setValue(0.3);
    }
    else if (val != f_poisson.getValue())
    {
        f_poisson.setValue(val);
    }
}

template<class DataTypes>
void TriangleFEMForceField<DataTypes>::setYoung(Real val)
{
    if (val < 0)
    {
        msg_warning() << "Input Young Modulus is not possible: " << val << ", setting default value: 1000";
        f_young.setValue(Real(1000));
    }
    else if (val != f_young.getValue())
    {
        f_young.setValue(val);
    }
}

template<class DataTypes>
void TriangleFEMForceField<DataTypes>::setMethod(int val)
{
    if (val != 0 && val != 1)
    {
        msg_warning() << "Input Method is not possible: " << val << ", should be 0 (Large) or 1 (Small). Setting default value: Large";
        method = LARGE;
    }
    else if (method != val)
    {
        method = val;
    }
}

template<class DataTypes>
void TriangleFEMForceField<DataTypes>::setMethod(std::string val)
{
    if (val == "small")
        method = SMALL;
    else if (val == "large")
        method = LARGE;
    else {
        msg_warning() << "Input Method is not possible: " << val << ", should be 0 (Large) or 1 (Small). Setting default value: Large";
        method = LARGE;
    }
}


template<class DataTypes>
const type::fixed_array <typename TriangleFEMForceField<DataTypes>::Coord, 3>& TriangleFEMForceField<DataTypes>::getRotatedInitialElement(Index elemId)
{
    if (elemId != sofa::InvalidID && elemId < _rotatedInitialElements.size())
        return _rotatedInitialElements[elemId];

    msg_warning() << "Method getRotatedInitialElement called with element index: " << elemId
        << " which is out of bounds: [0, " << _rotatedInitialElements.size() << "]. Returning default empty array of coordinates.";
    return InvalidCoords;
}

template<class DataTypes>
const typename TriangleFEMForceField<DataTypes>::Transformation& TriangleFEMForceField<DataTypes>::getRotationMatrix(Index elemId)
{
    if (elemId != sofa::InvalidID && elemId < _rotations.size())
        return _rotations[elemId];

    msg_warning() << "Method getRotationMatrix called with element index: "
        << elemId << " which is out of bounds: [0, " << _rotations.size() << "]. Returning default empty rotation.";
    return InvalidTransform;
}

template<class DataTypes>
const typename TriangleFEMForceField<DataTypes>::MaterialStiffness& TriangleFEMForceField<DataTypes>::getMaterialStiffness(Index elemId)
{
    if (elemId != sofa::InvalidID && elemId < _materialsStiffnesses.size())
        return _materialsStiffnesses[elemId];

    msg_warning() << "Method getMaterialStiffness called with element index: "
        << elemId << " which is out of bounds: [0, " << _materialsStiffnesses.size() << "]. Returning default empty matrix.";
    return InvalidTransform;
}

template<class DataTypes>
const typename TriangleFEMForceField<DataTypes>::StrainDisplacement& TriangleFEMForceField<DataTypes>::getStrainDisplacements(Index elemId)
{
    if (elemId != sofa::InvalidID && elemId < _strainDisplacements.size())
        return _strainDisplacements[elemId];

    msg_warning() << "Method getStrainDisplacements called with element index: "
        << elemId << " which is out of bounds: [0, " << _strainDisplacements.size() << "]. Returning default empty displacements.";
    return InvalidStrainDisplacement;
}

} // namespace sofa::component::solidmechanics::fem::elastic
