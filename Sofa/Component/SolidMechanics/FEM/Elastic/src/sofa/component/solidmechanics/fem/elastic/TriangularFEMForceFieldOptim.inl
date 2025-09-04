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
#include <sofa/component/solidmechanics/fem/elastic/TriangularFEMForceFieldOptim.h>
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.inl>
#include <sofa/core/behavior/BlocMatrixWriter.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyData.inl>
#include <limits>


namespace sofa::component::solidmechanics::fem::elastic
{

// --------------------------------------------------------------------------------------
// --- constructor
// --------------------------------------------------------------------------------------
template <class DataTypes>
TriangularFEMForceFieldOptim<DataTypes>::TriangularFEMForceFieldOptim()
    : d_triangleInfo(initData(&d_triangleInfo, "triangleInfo", "Internal triangle data (persistent)"))
    , d_triangleState(initData(&d_triangleState, "triangleState", "Internal triangle data (time-dependent)"))
    , d_damping(initData(&d_damping,(Real)0.,"damping","Ratio damping/stiffness"))
    , d_restScale(initData(&d_restScale,(Real)1.,"restScale","Scale factor applied to rest positions (to simulate pre-stretched materials)"))
    , d_computePrincipalStress(initData(&d_computePrincipalStress, false, "computePrincipalStress", "Compute principal stress for each triangle"))
    , d_stressMaxValue(initData(&d_stressMaxValue, (Real)0., "stressMaxValue", "Max stress value computed over the triangulation"))
    , d_showStressVector(initData(&d_showStressVector,false,"showStressVector","Flag activating rendering of stress directions within each triangle"))
    , d_showStressThreshold(initData(&d_showStressThreshold,(Real)0.0,"showStressThreshold","Threshold value to render only stress vectors higher to this threshold"))
{
    d_stressMaxValue.setReadOnly(true);
}


template <class DataTypes>
TriangularFEMForceFieldOptim<DataTypes>::~TriangularFEMForceFieldOptim()
{

}


// --------------------------------------------------------------------------------------
// --- Initialization stage
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::init()
{
    this->Inherited::init();

    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        return;
    }

    // Create specific handler for TriangleData
    d_triangleInfo.createTopologyHandler(this->l_topology);
    d_triangleInfo.setCreationCallback([this](Index triangleIndex, TriangleInfo& ti,
        const core::topology::BaseMeshTopology::Triangle& t,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        createTriangleInfo(triangleIndex, ti, t, ancestors, coefs);
    });

    d_triangleState.createTopologyHandler(this->l_topology);
    d_triangleState.setCreationCallback([this](Index triangleIndex, TriangleState& ti,
        const core::topology::BaseMeshTopology::Triangle& t,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        createTriangleState(triangleIndex, ti, t, ancestors, coefs);
    });

    if (this->l_topology->getNbTriangles() == 0)
    {
        msg_warning() << "No triangles found in linked Topology.";
        if (this->l_topology->getNbQuads() != 0)
        {
            msg_warning() << "The topology only contains quads while this forcefield only supports triangles." << msgendl;
        }
    }

    reinit();
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::parse( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    const char* method = arg->getAttribute("method");
    if (method && *method && std::string(method) != std::string("large"))
    {
        msg_warning() << "Attribute method was specified as \""<<method<<"\" while this version only implements the \"large\" method. Ignoring...";
    }
    Inherited::parse(arg);
}

// --------------------------------------------------------------------------------------
// --- Compute the initial info of the triangles
// --------------------------------------------------------------------------------------
template<class DataTypes>
inline void TriangularFEMForceFieldOptim<DataTypes>::computeTriangleRotation(Transformation& result, Coord eab, Coord eac)
{
    result[0] = eab;
    Coord n = eab.cross(eac);
    result[1] = n.cross(eab);
    result[0].normalize();
    result[1].normalize();
}
template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::initTriangleInfo(Index i, TriangleInfo& ti, const Triangle t, const VecCoord& x0)
{
    if (t[0] >= x0.size() || t[1] >= x0.size() || t[2] >= x0.size())
    {
        msg_error() << "INVALID point index >= " << x0.size() << " in triangle " << i << " : " << t
            << " | nb points: " << this->l_topology->getNbPoints()
            << " | nb triangles: " << this->l_topology->getNbTriangles() << msgendl;

        return;
    }
    Coord a  = x0[t[0]];
    Coord ab = x0[t[1]]-a;
    Coord ac = x0[t[2]]-a;
    if (this->d_restScale.isSet())
    {
        const Real restScale = this->d_restScale.getValue();
        ab *= restScale;
        ac *= restScale;
    }
    // equivalent to computeRotationLarge but in 2D == do not store the orthogonal vector are framex ^ framey
    computeTriangleRotation(ti.init_frame, ab, ac);

    // compute initial position in local space A[0, 0] B[x, 0] C[x, y]
    ti.bx = ti.init_frame[0] * ab;
    ti.cx = ti.init_frame[0] * ac;
    ti.cy = ti.init_frame[1] * ac;
    ti.ss_factor = ((Real)0.5)/(ti.bx*ti.cy);
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::initTriangleState(Index i, TriangleState& ts, const Triangle t, const VecCoord& x)
{
    if (t[0] >= x.size() || t[1] >= x.size() || t[2] >= x.size())
    {
        msg_error() << "INVALID point index >= " << x.size() << " in triangle " << i << " : " << t
            << " | nb points: " << this->l_topology->getNbPoints()
            << " | nb triangles: " << this->l_topology->getNbTriangles() << msgendl;

        return;
    }
    Coord a  = x[t[0]];
    Coord ab = x[t[1]]-a;
    Coord ac = x[t[2]]-a;
    computeTriangleRotation(ts.frame, ab, ac);

    ts.stress.clear();
}


// --------------------------------------------------------------------------------------
// ---  Topology Creation/Destruction functions
// --------------------------------------------------------------------------------------

template< class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::createTriangleInfo(Index triangleIndex, TriangleInfo& ti, const Triangle& t, const sofa::type::vector<Index>&, const sofa::type::vector<SReal>&)
{
    initTriangleInfo(triangleIndex, ti, t, this->mstate->read(core::vec_id::read_access::restPosition)->getValue());
}

template< class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::createTriangleState(Index triangleIndex, TriangleState& ti, const Triangle& t, const sofa::type::vector<Index>&, const sofa::type::vector<SReal>&)
{
    initTriangleState(triangleIndex, ti, t, this->mstate->read(core::vec_id::read_access::position)->getValue());
}

// --------------------------------------------------------------------------------------
// --- Re-initialization (called when we change a parameter through the GUI)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::reinit()
{
    /// prepare to store info in the triangle array
    const unsigned int nbTriangles = this->l_topology->getNbTriangles();
    const VecElement& triangles = this->l_topology->getTriangles();
    const  VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
    const  VecCoord& x0 = this->mstate->read(core::vec_id::read_access::restPosition)->getValue();
    VecTriangleInfo& triangleInf = *(d_triangleInfo.beginEdit());
    VecTriangleState& triangleSta = *(d_triangleState.beginEdit());
    triangleInf.resize(nbTriangles);
    triangleSta.resize(nbTriangles);

    for (unsigned int i=0; i<nbTriangles; ++i)
    {
        initTriangleInfo(i, triangleInf[i], triangles[i], x0);
        initTriangleState(i, triangleSta[i], triangles[i], x);
    }
    d_triangleInfo.endEdit();
    d_triangleState.endEdit();

    data.reinit(this);
}



template <class DataTypes>
SReal TriangularFEMForceFieldOptim<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* x */) const
{
    msg_warning()<<"TriangularFEMForceFieldOptim::getPotentialEnergy-not-implemented !!!"<<msgendl;
    return 0;
}


// --------------------------------------------------------------------------------------
// --- AddForce and AddDForce methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > f = d_f;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > x = d_x;
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecTriangleState > > triState = d_triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;

    const unsigned int nbTriangles = this->l_topology->getNbTriangles();
    const VecElement& triangles = this->l_topology->getTriangles();

    f.resize(x.size());

    for ( Index i=0; i<nbTriangles; i+=1)
    {
        Triangle t = triangles[i];
        const TriangleInfo& ti = triInfo[i];
        TriangleState& ts = triState[i];
        Coord a = x[t[0]];
        Coord ab = x[t[1]] -a;
        Coord ac = x[t[2]] -a;

        const auto [mu, gamma] = computeMuGamma(this->getYoungModulusInElement(i), this->getPoissonRatioInElement(i));

        computeTriangleRotation(ts.frame, ab, ac);

        // Displacement in local space (rest pos - current pos), dby == 0
        Real dbx = ti.bx - ts.frame[0] * ab;
        Real dcx = ti.cx - ts.frame[0] * ac;
        Real dcy = ti.cy - ts.frame[1] * ac;

        /// Full StrainDisplacement matrix.
        // | beta1  0       beta2  0        beta3  0      |
        // | 0      gamma1  0      gamma2   0      gamma3 | / (2 * A)
        // | gamma1 beta1   gamma2 beta2    gamma3 beta3 |

        // As no displacement for Pa nor in Pb[y], Beta1, gamma1 and beta3 are not considered. Therefore we obtain:
        // | beta2  0        beta3  0      |
        // | 0      gamma2   0      gamma3 | / (2 * A)
        // | gamma2 beta2    gamma3 beta3 |

        // |   cy     0     0      0   |
        // |   0     -cx    0      bx  |
        // |  -cx     cy    bx     0   |

        // Directly apply division by determinant(Area = det * 0.5 in local space; det = bx * cy)
        // |   1/bx        0        0        0   |
        // |   0       -cx/(bx*cy)  0       1/cy |
        // | -cx/(bx*cy)  1/bx     1/cy      0   |

        // StrainDisplacement:
        // beta2 = ti.cy;
        // gamma2 = -ti.cx;
        // gamma3 = ti.bx;

        // Strain = StrainDisplacement * Displacement
        type::Vec<3,Real> strain (
            ti.cy * dbx,                   // ( cy,   0,  0,  0) * (dbx, dby(0), dcx, dcy)
            ti.bx * dcy,                   // (  0, -cx,  0, bx) * (dbx, dby(0), dcx, dcy)
            ti.bx * dcx - ti.cx * dbx);    // ( -cx, cy, bx,  0) * (dbx, dby(0), dcx, dcy)

        // Stress = K * Strain
        Real gammaXY = gamma * (strain[0] + strain[1]);
        type::Vec<3,Real> stress (
            mu*strain[0] + gammaXY,      // (gamma+mu, gamma   ,    0) * strain
            mu*strain[1] + gammaXY,      // (gamma   , gamma+mu,    0) * strain
            (Real)(0.5)*mu*strain[2]);   // (       0,        0, mu/2) * strain

        stress *= ti.ss_factor;

        Deriv fb = ts.frame[0] * (ti.cy * stress[0] - ti.cx * stress[2])  // (cy,   0, -cx) * stress
                + ts.frame[1] * (ti.cy * stress[2] - ti.cx * stress[1]);  // ( 0, -cx,  cy) * stress
        Deriv fc = ts.frame[0] * (ti.bx * stress[2])                      // ( 0,   0,  bx) * stress
                + ts.frame[1] * (ti.bx * stress[1]);                      // ( 0,  bx,   0) * stress
        Deriv fa = -fb-fc;

        f[t[0]] += fa;
        f[t[1]] += fb;
        f[t[2]] += fc;

        // store data for reuse
        ts.stress = stress;
    }

    // compute principal stress if requested or for rendering
    if (this->d_computePrincipalStress.getValue() || this->d_showStressVector.getValue())
    {
        computePrincipalStress();
    }
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > df = d_df;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > dx = d_dx;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = d_triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;

    const unsigned int nbTriangles = this->l_topology->getNbTriangles();
    const VecElement& triangles = this->l_topology->getTriangles();
    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    df.resize(dx.size());

    for ( Index i=0; i<nbTriangles; i+=1)
    {
        const auto [mu, gamma] = computeMuGamma(this->getYoungModulusInElement(i), this->getPoissonRatioInElement(i));

        Triangle t = triangles[i];
        const TriangleInfo& ti = triInfo[i];
        const TriangleState& ts = triState[i];
        Deriv da  = dx[t[0]];
        Deriv dab = dx[t[1]]-da;
        Deriv dac = dx[t[2]]-da;

        Real dbx = ts.frame[0]*dab;
        Real dby = ts.frame[1]*dab;
        Real dcx = ts.frame[0]*dac;
        Real dcy = ts.frame[1]*dac;

        // Strain = StrainDisplacement * Displacement
        type::Vec<3, Real> dstrain(
            ti.cy * dbx,                                // ( cy,   0,  0,  0) * (dbx, dby, dcx, dcy)
            ti.bx * dcy - ti.cx * dby,                  // (  0, -cx,  0, bx) * (dbx, dby, dcx, dcy)
            ti.bx * dcx - ti.cx * dbx + ti.cy * dby);   // ( -cx, cy, bx,  0) * (dbx, dby, dcx, dcy)


        // Stress = K * Strain
        Real gammaXY = gamma*(dstrain[0]+dstrain[1]);
        type::Vec<3,Real> dstress (
            mu*dstrain[0] + gammaXY,        // (gamma+mu, gamma   ,    0) * dstrain
            mu*dstrain[1] + gammaXY,        // (gamma   , gamma+mu,    0) * dstrain
            (Real)(0.5)*mu*dstrain[2]);     // (       0,        0, mu/2) * dstrain

        dstress *= ti.ss_factor * kFactor;
        Deriv dfb = ts.frame[0] * (ti.cy * dstress[0] - ti.cx * dstress[2])  // (cy,   0, -cx) * stress
            + ts.frame[1] * (ti.cy * dstress[2] - ti.cx * dstress[1]);       // ( 0, -cx,  cy) * stress
        Deriv dfc = ts.frame[0] * (ti.bx * dstress[2])                       // ( 0,   0,  bx) * stress
            + ts.frame[1] * (ti.bx * dstress[1]);                            // ( 0,  bx,   0) * stress
        Deriv dfa = -dfb - dfc;

        df[t[0]] -= dfa;
        df[t[1]] -= dfb;
        df[t[2]] -= dfc;
    }
}


// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal kFact, unsigned int &offset)
{
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = d_triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;
    const unsigned int nbTriangles = this->l_topology->getNbTriangles();
    const VecElement& triangles = this->l_topology->getTriangles();

    constexpr auto S = DataTypes::deriv_total_size;

    for ( Index i=0; i<nbTriangles; i+=1)
    {
        const auto [mu, gamma] = computeMuGamma(this->getYoungModulusInElement(i), this->getPoissonRatioInElement(i));

        Triangle t = triangles[i];
        const TriangleInfo& ti = triInfo[i];
        const TriangleState& ts = triState[i];
        sofa::type::MatNoInit<3,4,Real> KJt;
        const Real factor = -kFact * ti.ss_factor;
        const Real fG = factor * gamma;
        const Real fGM = factor * (gamma+mu);
        const Real fM_2 = factor * (0.5f*mu);
        KJt(0,0) = fGM  *  ti.cy ;    KJt(0,1) = fG   *(-ti.cx);    KJt(0,2) = 0;    KJt(0,3) = fG   *ti.bx;
        KJt(1,0) = fG   *  ti.cy ;    KJt(1,1) = fGM  *(-ti.cx);    KJt(1,2) = 0;    KJt(1,3) = fGM  *ti.bx;
        KJt(2,0) = fM_2 *(-ti.cx);    KJt(2,1) = fM_2 *( ti.cy);    KJt(2,2) = fM_2 *ti.bx;    KJt(2,3) = 0;

        sofa::type::MatNoInit<2,2,Real> JKJt11, JKJt12, JKJt22;
        JKJt11(0,0) = ti.cy*KJt(0,0) - ti.cx*KJt(2,0);
        JKJt11(0,1) = ti.cy*KJt(0,1) - ti.cx*KJt(2,1);
        JKJt11(1,0) = JKJt11(0,1); //ti.cy*KJt(2,0) - ti.cx*KJt(1,0);
        JKJt11(1,1) = ti.cy*KJt(2,1) - ti.cx*KJt(1,1);

        JKJt12(0,0) = -ti.cx*KJt(2,2);
        JKJt12(0,1) =  ti.cy*KJt(0,3);
        JKJt12(1,0) =  ti.cy*KJt(2,2);
        JKJt12(1,1) = -ti.cx*KJt(1,3);

        JKJt22(0,0) = ti.bx*KJt(2,2);
        JKJt22(0,1) = 0; //ti.bx*KJt(2,3);
        JKJt22(1,0) = 0; //ti.bx*KJt(1,2);
        JKJt22(1,1) = ti.bx*KJt(1,3);

        sofa::type::MatNoInit<2,2,Real> JKJt00, JKJt01, JKJt02;
        // fA = -fB-fC, dxB/dxA = -1, dxC/dxA = -1
        // dfA/dxA = -dfB/dxA - dfC/dxA
        //         = -dfB/dxB * dxB/dxA -dfB/dxC * dxC/dxA   -dfC/dxB * dxB/dxA -dfC/dxC * dxC/dxA
        //         = dfB/dxB + dfB/dxC + dfC/dxB + dfC/dxC
        JKJt00 = JKJt11+JKJt12+JKJt22+JKJt12.transposed();
        // dfA/dxB = -dfB/dxB -dfC/dxB
        JKJt01 = -JKJt11-JKJt12.transposed();
        // dfA/dxC = -dfB/dxC -dfC/dxC
        JKJt02 = -JKJt12-JKJt22;

        Transformation frame = ts.frame;

        matrix->add(offset + S * t[0], offset + S * t[0], frame.multTranspose(JKJt00*frame));

        const MatBloc M01 = frame.multTranspose(JKJt01*frame);
        matrix->add(offset + S * t[0], offset + S * t[1], M01);
        matrix->add(offset + S * t[1], offset + S * t[0], M01.transposed());

        const MatBloc M02 = frame.multTranspose(JKJt02*frame);
        matrix->add(offset + S * t[0], offset + S * t[2], M02);
        matrix->add(offset + S * t[2], offset + S * t[0], M02.transposed());
        matrix->add(offset + S * t[1], offset + S * t[1], frame.multTranspose(JKJt11*frame));

        const MatBloc M12 = frame.multTranspose(JKJt12*frame);
        matrix->add(offset + S * t[1], offset + S * t[2], M12);
        matrix->add(offset + S * t[2], offset + S * t[1], M12.transposed());
        matrix->add(offset + S * t[2], offset + S * t[2], frame.multTranspose(JKJt22*frame));
    }
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = d_triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;
    const unsigned int nbTriangles = this->l_topology->getNbTriangles();
    const VecElement& triangles = this->l_topology->getTriangles();

    static constexpr auto S = DataTypes::deriv_total_size;

    for (Index i = 0; i < nbTriangles; ++i)
    {
        const auto [mu, gamma] = computeMuGamma(this->getYoungModulusInElement(i), this->getPoissonRatioInElement(i));

        Triangle t = triangles[i];
        const TriangleInfo& ti = triInfo[i];
        const TriangleState& ts = triState[i];
        sofa::type::MatNoInit<3, 4, Real> KJt;
        const Real factor = -ti.ss_factor;
        const Real fG = factor * gamma;
        const Real fGM = factor * (gamma + mu);
        const Real fM_2 = factor * mu / 2;
        KJt(0,0) = fGM  *  ti.cy ;    KJt(0,1) = fG   *(-ti.cx);    KJt(0,2) = 0;    KJt(0,3) = fG   *ti.bx;
        KJt(1,0) = fG   *  ti.cy ;    KJt(1,1) = fGM  *(-ti.cx);    KJt(1,2) = 0;    KJt(1,3) = fGM  *ti.bx;
        KJt(2,0) = fM_2 *(-ti.cx);    KJt(2,1) = fM_2 *( ti.cy);    KJt(2,2) = fM_2 *ti.bx;    KJt(2,3) = 0;

        sofa::type::MatNoInit<2, 2, Real> JKJt11, JKJt12, JKJt22;
        JKJt11(0,0) = ti.cy * KJt(0,0) - ti.cx * KJt(2,0);
        JKJt11(0,1) = ti.cy * KJt(0,1) - ti.cx * KJt(2,1);
        JKJt11(1,0) = JKJt11(0,1); //ti.cy*KJt(2,0) - ti.cx*KJt(1,0);
        JKJt11(1,1) = ti.cy * KJt(2,1) - ti.cx * KJt(1,1);

        JKJt12(0,0) = -ti.cx * KJt(2,2);
        JKJt12(0,1) = ti.cy * KJt(0,3);
        JKJt12(1,0) = ti.cy * KJt(2,2);
        JKJt12(1,1) = -ti.cx * KJt(1,3);

        JKJt22(0,0) = ti.bx * KJt(2,2);
        JKJt22(0,1) = 0; //ti.bx*KJt(2,3);
        JKJt22(1,0) = 0; //ti.bx*KJt(1,2);
        JKJt22(1,1) = ti.bx * KJt(1,3);

        sofa::type::MatNoInit<2,2,Real> JKJt00, JKJt01, JKJt02;
        // fA = -fB-fC, dxB/dxA = -1, dxC/dxA = -1
        // dfA/dxA = -dfB/dxA - dfC/dxA
        //         = -dfB/dxB * dxB/dxA -dfB/dxC * dxC/dxA   -dfC/dxB * dxB/dxA -dfC/dxC * dxC/dxA
        //         = dfB/dxB + dfB/dxC + dfC/dxB + dfC/dxC
        JKJt00 = JKJt11 + JKJt12 + JKJt22 + JKJt12.transposed();
        // dfA/dxB = -dfB/dxB -dfC/dxB
        JKJt01 = -JKJt11 - JKJt12.transposed();
        // dfA/dxC = -dfB/dxC -dfC/dxC
        JKJt02 = -JKJt12 - JKJt22;

        Transformation frame = ts.frame;

        dfdx(S * t[0], S * t[0]) += frame.multTranspose(JKJt00*frame);

        const MatBloc M01 = frame.multTranspose(JKJt01*frame);
        dfdx(S * t[0], S * t[1]) += M01;
        dfdx(S * t[1], S * t[0]) += M01.transposed();

        const MatBloc M02 = frame.multTranspose(JKJt02*frame);
        dfdx(S * t[0], S * t[2]) += M02;
        dfdx(S * t[2], S * t[0]) += M02.transposed();

        dfdx(S * t[1], S * t[1]) += frame.multTranspose(JKJt11*frame);

        const MatBloc M12 = frame.multTranspose(JKJt12*frame);
        dfdx(S * t[1], S * t[2]) += M12;
        dfdx(S * t[2], S * t[1]) += M12.transposed();

        dfdx(S * t[2], S * t[2]) += frame.multTranspose(JKJt22*frame);
    }
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::getTriangleVonMisesStress(Index i, Real& stressValue)
{
    const Deriv& s = d_triangleState.getValue()[i].stress;
    Real vonMisesStress = sofa::helper::rsqrt(s[0]*s[0] - s[0]*s[1] + s[1]*s[1] + 3*s[2]);
    stressValue = vonMisesStress;
}

template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::getTrianglePrincipalStress(Index i, Real& stressValue, Deriv& stressDirection)
{
    Real stressValue2;
    Deriv stressDirection2;
    getTrianglePrincipalStress(i, stressValue, stressDirection, stressValue2, stressDirection2);
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if (!onlyVisible) return;
    if (!this->mstate) return;

    const auto bbox = this->mstate->computeBBox(); //this may compute twice the mstate bbox, but there is no way to determine if the bbox has already been computed
    this->f_bbox.setValue(std::move(bbox));
}

template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::getTrianglePrincipalStress(Index i, Real& stressValue, Deriv& stressDirection, Real& stressValue2, Deriv& stressDirection2)
{
    const TriangleState& ts = d_triangleState.getValue()[i];
    Deriv s = ts.stress;

    // If A = [ a b ] is a real symmetric 2x2 matrix
    //        [ b d ]
    // the eigen values are :
    //   L1,L2 = (T +- sqrt(T^2 - 4*D))/2
    // with T = trace(A) = a+d
    // and D = det(A) = ad-b^2
    // and the eigen vectors are [ b   L-a ]
    //         ( or equivalently [ L-d   b ] )

    Real tr = (s[0]+s[1]);
    Real det = s[0]*s[1]-s[2]*s[2];
    Real deltaV2 = tr*tr-4*det;
    Real deltaV = helper::rsqrt(std::max((Real)0.0,deltaV2));
    Real eval1, eval2;
    type::Vec<2,Real> evec1, evec2;
    eval1 = (tr + deltaV)/2;
    eval2 = (tr - deltaV)/2;
    if (s[2] == 0)
    {
        if (s[0] > s[1])
        {
            evec1[0] = 1; evec1[1] = 0;
            evec2[0] = 0; evec2[1] = 1;
        }
        else
        {
            evec1[0] = 0; evec1[1] = 1;
            evec2[0] = 1; evec2[1] = 0;
        }
    }
    else
    {
        evec1[0] = s[2]; evec1[1] = eval1-s[0];
        evec2[0] = s[2]; evec2[1] = eval2-s[0];
    }
    Deriv edir1 = ts.frame.multTranspose(evec1);
    Deriv edir2 = ts.frame.multTranspose(evec2);
    edir1.normalize();
    edir2.normalize();

    if (helper::rabs(eval1) > helper::rabs(eval2))
    {
        stressValue  = eval1;  stressDirection  = edir1;
        stressValue2 = eval2;  stressDirection2 = edir2;
    }
    else
    {
        stressValue  = eval2;  stressDirection  = edir2;
        stressValue2 = eval1;  stressDirection2 = edir1;
    }
}


template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::computePrincipalStress()
{
    const VecElement& triangles = this->l_topology->getTriangles();
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfos = d_triangleInfo;

    Real minStress = 0;
    Real maxStress = 0;

    for (std::size_t i = 0; i < triangles.size(); i++)
    {
        TriangleInfo& triInfo = triInfos[i];

        getTrianglePrincipalStress(i, triInfo.stress, triInfo.stressVector, triInfo.stress2, triInfo.stressVector2);

        minStress = std::min({minStress, triInfo.stress, triInfo.stress2});
        maxStress = std::max({maxStress , triInfo.stress, triInfo.stress2});
    }

    d_stressMaxValue.setValue(maxStress);

    if (!d_showStressThreshold.isSet() && d_showStressVector.getValue())
        d_showStressThreshold.setValue(minStress);
}


template<class DataTypes>
type::fixed_array <typename TriangularFEMForceFieldOptim<DataTypes>::Coord, 3> TriangularFEMForceFieldOptim<DataTypes>::getRotatedInitialElement(Index elemId)
{
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;
    type::fixed_array <Coord, 3> positions;
    if (elemId >= triInfo.size())
    {
        msg_warning() << "Method getRotatedInitialElement called with element index: " << elemId
            << " which is out of bounds: [0, " << triInfo.size() << "]. Returning default empty array of coordinates.";
        return positions;
    }

    const TriangleInfo& ti = triInfo[elemId];
    positions[0] = Coord(0, 0, 0);
    positions[1] = Coord(ti.bx, 0, 0);
    positions[2] = Coord(ti.cx, ti.cy, 0);

    return positions;
}


template<class DataTypes>
typename TriangularFEMForceFieldOptim<DataTypes>::Transformation TriangularFEMForceFieldOptim<DataTypes>::getRotationMatrix(Index elemId)
{
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = d_triangleState;
    if (elemId < triState.size())
        return triState[elemId].frame;

    msg_warning() << "Method getRotationMatrix called with element index: "
        << elemId << " which is out of bounds: [0, " << triState.size() << "]. Returning default empty rotation.";
    return Transformation();
}


template<class DataTypes>
typename TriangularFEMForceFieldOptim<DataTypes>::MaterialStiffness TriangularFEMForceFieldOptim<DataTypes>::getMaterialStiffness(Index elemId)
{
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;
    if (elemId >= triInfo.size())
    {
        msg_warning() << "Method getMaterialStiffness called with element index: "
            << elemId << " which is out of bounds: [0, " << triInfo.size() << "]. Returning default empty matrix.";
        return MaterialStiffness();
    }

    // (gamma+mu, gamma   ,    0)
    // (gamma   , gamma+mu,    0)
    // (       0,        0, mu/2)
    const auto [mu, gamma] = computeMuGamma(this->getYoungModulusInElement(elemId), this->getPoissonRatioInElement(elemId));

    MaterialStiffness mat;
    mat(0,0) = mat(1,1) = gamma + mu;
    mat(0,1) = mat(1,0) = gamma;
    mat(2,2) = (Real)(0.5) * mu;

    return mat;
}


template<class DataTypes>
sofa::type::Vec3 TriangularFEMForceFieldOptim<DataTypes>::getStrainDisplacementFactors(Index elemId)
{
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;
    if (elemId >= triInfo.size())
    {
        msg_warning() << "Method getStrainDisplacementFactors called with element index: "
            << elemId << " which is out of bounds: [0, " << triInfo.size() << "]. Returning default empty displacements.";
        return type::Vec< 3, Real>();
    }

    const TriangleInfo& ti = triInfo[elemId];
    return type::Vec< 3, Real>(ti.cy, -ti.cx, ti.bx);
}

template<class DataTypes>
typename TriangularFEMForceFieldOptim<DataTypes>::Real TriangularFEMForceFieldOptim<DataTypes>::getTriangleFactor(Index elemId)
{
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;
    if (elemId >= triInfo.size())
    {
        msg_warning() << "Method getTriangleFactor called with element index: "
            << elemId << " which is out of bounds: [0, " << triInfo.size() << "]. Returning 0.";
        return Real(0);
    }

    return triInfo[elemId].ss_factor;
}

template <class DataTypes>
auto TriangularFEMForceFieldOptim<
    DataTypes>::computeMuGamma(Real youngModulus,
                               Real poissonRatio) -> std::pair<Real, Real>
{
    const Real mu = (youngModulus) / (1 + poissonRatio);
    const Real gamma = (youngModulus * poissonRatio) / (1 - poissonRatio * poissonRatio);
    return {mu, gamma};
}


// --------------------------------------------------------------------------------------
// --- Display methods
// --------------------------------------------------------------------------------------

template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!this->l_topology || !this->mstate) return;

    if (!vparams->displayFlags().getShowForceFields())
        return;

    using type::Vec3;
    using type::Vec3i;
    using type::Vec4f;

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
    unsigned int nbTriangles=this->l_topology->getNbTriangles();
    const VecElement& triangles = this->l_topology->getTriangles();
    const Real& stressThresold = d_showStressThreshold.getValue();

    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfos = d_triangleInfo;
    if (this->d_showStressVector.getValue() && stressThresold > 0)
    {
        std::vector< Vec3 > points[2];
        const Real& maxStress = d_stressMaxValue.getValue();
        for (unsigned int i=0; i<nbTriangles; i++)
        {
            const TriangleInfo& triInfo = triInfos[i];
            Real s1 = triInfo.stress;
            Real s2 = triInfo.stress2;

            if (helper::rabs(s1) < stressThresold)
                continue;
            
            const Triangle& t = triangles[i];
            Vec3 a = x[t[0]];
            Vec3 b = x[t[1]];
            Vec3 c = x[t[2]];
            Vec3 center = (a + b + c) / 3;

            Vec3 d1 = triInfo.stressVector;            
            Vec3 d2 = triInfo.stressVector2;
                        
            d1.normalize();
            Vec3 colorD1 = (d1 + Vec3(1, 1, 1)) * 0.5; 
            d1 *= helper::rabs(s1) / maxStress;

            d2.normalize();
            Vec3 colorD2 = (d2 + Vec3(1, 1, 1)) * 0.5;
            d2 *= helper::rabs(s2) / maxStress;
            
            vparams->drawTool()->drawArrow(center, sofa::type::Vec3(center + d1), 0.01, sofa::type::RGBAColor(colorD1[0], colorD1[1], colorD1[2], 1));
            vparams->drawTool()->drawArrow(center, sofa::type::Vec3(center + d2), 0.01, sofa::type::RGBAColor(colorD2[0], colorD2[1], colorD2[2], 1));
        }
    }
    else
    {
        sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = d_triangleState;
        std::vector< Vec3 > points[4];

        constexpr sofa::type::RGBAColor c0 = sofa::type::RGBAColor::red();
        constexpr sofa::type::RGBAColor c1 = sofa::type::RGBAColor::green();
        constexpr sofa::type::RGBAColor c2(1,0.5,0,1);
        constexpr sofa::type::RGBAColor c3 = sofa::type::RGBAColor::blue();

        points[0].reserve(nbTriangles*2);
        points[1].reserve(nbTriangles*2);
        points[2].reserve(nbTriangles*6);
        points[3].reserve(nbTriangles*6);
        for (unsigned int i=0; i<nbTriangles; ++i)
        {
            Triangle t = triangles[i];
            const TriangleInfo& ti = triInfos[i];
            const TriangleState& ts = triState[i];
            Coord a = x[t[0]];
            Coord b = x[t[1]];
            Coord c = x[t[2]];
            Coord fx = ts.frame[0];
            Coord fy = ts.frame[1];
            Vec3 center = (a+b+c)*(1.0_sreal/3.0_sreal);
            Real scale = (Real)(sqrt((b-a).cross(c-a).norm()*0.25f));
            points[0].push_back(center);
            points[0].push_back(center + ts.frame[0] * scale);
            points[1].push_back(center);
            points[1].push_back(center + ts.frame[1] * scale);
            Coord a0 = center - fx * (ti.bx/3 + ti.cx/3) - fy * (ti.cy/3);
            Coord b0 = a0 + fx * ti.bx;
            Coord c0 = a0 + fx * ti.cx + fy * ti.cy;
            points[2].push_back(a0);
            points[2].push_back(b0);
            points[2].push_back(b0);
            points[2].push_back(c0);
            points[2].push_back(c0);
            points[2].push_back(a0);
            points[3].push_back(a0);
            points[3].push_back(a );
            points[3].push_back(b0);
            points[3].push_back(b );
            points[3].push_back(c0);
            points[3].push_back(c );
        }

        vparams->drawTool()->drawLines(points[0], 1, c0);
        vparams->drawTool()->drawLines(points[1], 1, c1);
        vparams->drawTool()->drawLines(points[2], 1, c2);
        vparams->drawTool()->drawLines(points[3], 1, c3);
    }
}

} // namespace sofa::component::solidmechanics::fem::elastic
