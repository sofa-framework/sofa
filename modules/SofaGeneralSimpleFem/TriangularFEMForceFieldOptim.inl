/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "TriangularFEMForceFieldOptim.h"

#include <SofaBaseLinearSolver/BlocMatrixWriter.h>

#include <sofa/core/visual/VisualParams.h>

#include <SofaBaseTopology/TopologyData.inl>

#include <limits>


namespace sofa
{

namespace component
{

namespace forcefield
{

// --------------------------------------------------------------------------------------
// ---  Topology Creation/Destruction functions
// --------------------------------------------------------------------------------------

template< class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::TFEMFFOTriangleInfoHandler::applyCreateFunction(unsigned int triangleIndex, TriangleInfo &ti, const Triangle &t, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {
        ff->initTriangleInfo(triangleIndex,ti,t, ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue());
    }
}

template< class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::TFEMFFOTriangleStateHandler::applyCreateFunction(unsigned int triangleIndex, TriangleState &ti, const Triangle &t, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {
        ff->initTriangleState(triangleIndex,ti,t, ff->mstate->read(core::ConstVecCoordId::position())->getValue());
    }
}


// --------------------------------------------------------------------------------------
// --- constructor
// --------------------------------------------------------------------------------------
template <class DataTypes>
TriangularFEMForceFieldOptim<DataTypes>::TriangularFEMForceFieldOptim()
    : triangleInfo(initData(&triangleInfo, "triangleInfo", "Internal triangle data (persistent)"))
    , triangleState(initData(&triangleState, "triangleState", "Internal triangle data (time-dependent)"))
    , vertexInfo(initData(&vertexInfo, "vertexInfo", "Internal point data"))
    , edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
    , _topology(NULL)
#ifdef SIMPLEFEM_COLORMAP
#ifndef SOFA_NO_OPENGL
	, showStressColorMapReal(sofa::core::objectmodel::New< visualmodel::ColorMap >())
#endif
#endif
    , f_poisson(initData(&f_poisson,(Real)(0.45),"poissonRatio","Poisson ratio in Hooke's law"))
    , f_young(initData(&f_young,(Real)(1000.0),"youngModulus","Young modulus in Hooke's law"))
    , f_damping(initData(&f_damping,(Real)0.,"damping","Ratio damping/stiffness"))
    , f_restScale(initData(&f_restScale,(Real)1.,"restScale","Scale factor applied to rest positions (to simulate pre-stretched materials)"))
#ifdef SIMPLEFEM_COLORMAP
    , showStressValue(initData(&showStressValue,true,"showStressValue","Flag activating rendering of stress values as a color in each triangle"))
#endif
    , showStressVector(initData(&showStressVector,false,"showStressVector","Flag activating rendering of stress directions within each triangle"))
#ifdef SIMPLEFEM_COLORMAP
    , showStressColorMap(initData(&showStressColorMap,"showStressColorMap", "Color map used to show stress values"))
#endif
    , showStressMaxValue(initData(&showStressMaxValue,(Real)0.0,"showStressMaxValue","Max value for rendering of stress values"))
#ifdef SIMPLEFEM_COLORMAP
    , showStressValueAlpha(initData(&showStressValueAlpha,(float)1.0,"showStressValueAlpha","Alpha (1-transparency) value for rendering of stress values"))
#endif
    , drawPrevMaxStress((Real)-1.0)
{
    triangleInfoHandler = new TFEMFFOTriangleInfoHandler(this, &triangleInfo);
    triangleStateHandler = new TFEMFFOTriangleStateHandler(this, &triangleState);

	f_poisson.setRequired(true);
	f_young.setRequired(true);
}


template <class DataTypes>
TriangularFEMForceFieldOptim<DataTypes>::~TriangularFEMForceFieldOptim()
{
    if(triangleInfoHandler) delete triangleInfoHandler;
    if(triangleStateHandler) delete triangleStateHandler;
}


// --------------------------------------------------------------------------------------
// --- Initialization stage
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::init()
{
    this->Inherited::init();

    _topology = this->getContext()->getMeshTopology();

    // Create specific handler for TriangleData
    triangleInfo.createTopologicalEngine(_topology, triangleInfoHandler);
    triangleInfo.registerTopologicalData();

    triangleState.createTopologicalEngine(_topology, triangleStateHandler);
    triangleState.registerTopologicalData();

    edgeInfo.createTopologicalEngine(_topology);
    edgeInfo.registerTopologicalData();

    vertexInfo.createTopologicalEngine(_topology);
    vertexInfo.registerTopologicalData();

    if (_topology->getNbTriangles()==0 && _topology->getNbQuads()!=0 )
    {
        serr << "The topology only contains quads while this forcefield only supports triangles."<<sendl;
    }

    reinit();
}


// --------------------------------------------------------------------------------------
// --- Compute the initial info of the triangles
// --------------------------------------------------------------------------------------

template <class DataTypes>
inline void TriangularFEMForceFieldOptim<DataTypes>::computeTriangleRotation(Transformation& result, Coord eab, Coord eac)
{
    result[0] = eab;
    Coord n = eab.cross(eac);
    result[1] = n.cross(eab);
    result[0].normalize();
    result[1].normalize();
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::initTriangleInfo(unsigned int i, TriangleInfo& ti, const Triangle t, const VecCoord& x0)
{
    if (t[0] >= x0.size() || t[1] >= x0.size() || t[2] >= x0.size())
    {
        serr << "INVALID point index >= " << x0.size() << " in triangle " << i << " : " << t << sendl;
        serr << this->getContext()->getMeshTopology()->getNbPoints() << "/"
             << this->mstate->getContext()->getMeshTopology()->getNbPoints() << " points,"
             << this->getContext()->getMeshTopology()->getNbTriangles() << "/"
             << this->mstate->getContext()->getMeshTopology()->getNbTriangles() << " triangles." << sendl;
        return;
    }
    Coord a  = x0[t[0]];
    Coord ab = x0[t[1]]-a;
    Coord ac = x0[t[2]]-a;
    if (this->f_restScale.isSet())
    {
        const Real restScale = this->f_restScale.getValue();
        ab *= restScale;
        ac *= restScale;
    }
    computeTriangleRotation(ti.init_frame, ab, ac);
    ti.bx = ti.init_frame[0] * ab;
    ti.cx = ti.init_frame[0] * ac;
    ti.cy = ti.init_frame[1] * ac;
    //ti.ss_factor = ((Real)1.0)/(ti.bx*ti.bx*ti.cy*ti.cy);
    ti.ss_factor = ((Real)0.5)/(ti.bx*ti.cy);
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::initTriangleState(unsigned int i, TriangleState& ti, const Triangle t, const VecCoord& x)
{
    if (t[0] >= x.size() || t[1] >= x.size() || t[2] >= x.size())
    {
        serr << "INVALID point index >= " << x.size() << " in triangle " << i << " : " << t << sendl;
        serr << this->getContext()->getMeshTopology()->getNbPoints() << "/"
             << this->mstate->getContext()->getMeshTopology()->getNbPoints() << " points,"
             << this->getContext()->getMeshTopology()->getNbTriangles() << "/"
             << this->mstate->getContext()->getMeshTopology()->getNbTriangles() << " triangles." << sendl;
        return;
    }
    Coord a  = x[t[0]];
    Coord ab = x[t[1]]-a;
    Coord ac = x[t[2]]-a;
    computeTriangleRotation(ti.frame, ab, ac);
    ti.stress.clear();
}

// --------------------------------------------------------------------------------------
// --- Re-initialization (called when we change a parameter through the GUI)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::reinit()
{
    // Compute material-dependent constants

    // mu = (1-p)*y/(1-p^2) = (1-p)*y/((1-p)(1+p)) = y/(1+p)

    const Real youngModulus = f_young.getValue();
    const Real poissonRatio = f_poisson.getValue();
    mu = (youngModulus) / (1+poissonRatio);
    gamma = (youngModulus * poissonRatio) / (1-poissonRatio*poissonRatio);

    /// prepare to store info in the triangle array
    const unsigned int nbTriangles = _topology->getNbTriangles();
    const VecElement& triangles = _topology->getTriangles();
    const  VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const  VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    VecTriangleInfo& triangleInf = *(triangleInfo.beginEdit());
    VecTriangleState& triangleSta = *(triangleState.beginEdit());
    triangleInf.resize(nbTriangles);
    triangleSta.resize(nbTriangles);

    for (unsigned int i=0; i<nbTriangles; ++i)
    {
        initTriangleInfo(i, triangleInf[i], triangles[i], x0);
        initTriangleState(i, triangleSta[i], triangles[i], x);
    }
    triangleInfo.endEdit();
    triangleState.endEdit();

    /// prepare to store info in the edge array
    VecEdgeInfo& edgeInf = *(edgeInfo.beginEdit());
    edgeInf.resize(_topology->getNbEdges());
    edgeInfo.endEdit();

    /// prepare to store info in the vertex array
    unsigned int nbPoints = _topology->getNbPoints();
    VecVertexInfo& vi = *(vertexInfo.beginEdit());
    vi.resize(nbPoints);
    vertexInfo.endEdit();

#ifdef SIMPLEFEM_COLORMAP
#ifndef SOFA_NO_OPENGL
    // TODO: This is deprecated. Use ColorMap as a component.
     visualmodel::ColorMap* colorMap = NULL;
    this->getContext()->get(colorMap,sofa::core::objectmodel::BaseContext::Local);
    if (colorMap)
        showStressColorMapReal = colorMap;
    else
        showStressColorMapReal->initOld(showStressColorMap.getValue());
#endif
#endif

    data.reinit(this);
}



template <class DataTypes>
SReal TriangularFEMForceFieldOptim<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* x */) const
{
    serr<<"TriangularFEMForceFieldOptim::getPotentialEnergy-not-implemented !!!"<<sendl;
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
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecTriangleState > > triState = triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = triangleInfo;

    const unsigned int nbTriangles = _topology->getNbTriangles();
    const VecElement& triangles = _topology->getTriangles();
    const Real gamma = this->gamma;
    const Real mu = this->mu;

    f.resize(x.size());

    for ( Index i=0; i<nbTriangles; i+=1)
    {
        Triangle t = triangles[i];
        const TriangleInfo& ti = triInfo[i];
        TriangleState& ts = triState[i];
        Coord a  = x[t[0]];
        Coord ab = x[t[1]]-a;
        Coord ac = x[t[2]]-a;
        computeTriangleRotation(ts.frame, ab, ac);
        Real dbx = ti.bx - ts.frame[0]*ab;
        // Real dby = 0
        Real dcx = ti.cx - ts.frame[0]*ac;
        Real dcy = ti.cy - ts.frame[1]*ac;
        //sout << "Elem" << i << ": D= 0 0  " << dbx << " 0  " << dcx << " " << dcy << sendl;

        defaulttype::Vec<3,Real> strain (
            ti.cy * dbx,                // ( cy,   0,  0,  0) * (dbx, dby, dcx, dcy)
            ti.bx * dcy,                // (  0, -cx,  0, bx) * (dbx, dby, dcx, dcy)
            ti.bx * dcx - ti.cx * dbx); // (-cx,  cy, bx,  0) * (dbx, dby, dcx, dcy)

        Real gammaXY = gamma*(strain[0]+strain[1]);

        defaulttype::Vec<3,Real> stress (
            mu*strain[0] + gammaXY,    // (gamma+mu, gamma   ,    0) * strain
            mu*strain[1] + gammaXY,    // (gamma   , gamma+mu,    0) * strain
            (Real)(0.5)*mu*strain[2]); // (       0,        0, mu/2) * strain

        ts.stress = stress;

        stress *= ti.ss_factor;
        //sout << "Elem" << i << ": F= " << -(ti.cy * stress[0] - ti.cx * stress[2] + ti.bx * stress[2]) << " " << -(ti.cy * stress[2] - ti.cx * stress[1] + ti.bx * stress[1]) << "  " << (ti.cy * stress[0] - ti.cx * stress[2]) << " " << (ti.cy * stress[2] - ti.cx * stress[1]) << "  " << (ti.bx * stress[2]) << " " << (ti.bx * stress[1]) << sendl;
        Deriv fb = ts.frame[0] * (ti.cy * stress[0] - ti.cx * stress[2])  // (cy,   0, -cx) * stress
                + ts.frame[1] * (ti.cy * stress[2] - ti.cx * stress[1]); // ( 0, -cx,  cy) * stress
        Deriv fc = ts.frame[0] * (ti.bx * stress[2])                      // ( 0,   0,  bx) * stress
                + ts.frame[1] * (ti.bx * stress[1]);                     // ( 0,  bx,   0) * stress
        Deriv fa = -fb-fc;
        f[t[0]] += fa;
        f[t[1]] += fb;
        f[t[2]] += fc;
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
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = triangleInfo;

    const unsigned int nbTriangles = _topology->getNbTriangles();
    const VecElement& triangles = _topology->getTriangles();
    const Real gamma = this->gamma;
    const Real mu = this->mu;
    const Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    df.resize(dx.size());

    for ( Index i=0; i<nbTriangles; i+=1)
    {
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

        defaulttype::Vec<3,Real> dstrain (
            ti.cy  * dbx,                             // ( cy,   0,  0,  0) * (dbx, dby, dcx, dcy)
            ti.bx * dcy - ti.cx * dby,                // (  0, -cx,  0, bx) * (dbx, dby, dcx, dcy)
            ti.bx * dcx - ti.cx * dbx + ti.cy * dby); // (-cx,  cy, bx,  0) * (dbx, dby, dcx, dcy)

        Real gammaXY = gamma*(dstrain[0]+dstrain[1]);

        defaulttype::Vec<3,Real> dstress (
            mu*dstrain[0] + gammaXY,    // (gamma+mu, gamma   ,    0) * dstrain
            mu*dstrain[1] + gammaXY,    // (gamma   , gamma+mu,    0) * dstrain
            (Real)(0.5)*mu*dstrain[2]); // (       0,        0, mu/2) * dstrain

        dstress *= ti.ss_factor * kFactor;
        Deriv dfb = ts.frame[0] * (ti.cy * dstress[0] - ti.cx * dstress[2])  // (cy,   0, -cx) * dstress
                + ts.frame[1] * (ti.cy * dstress[2] - ti.cx * dstress[1]); // ( 0, -cx,  cy) * dstress
        Deriv dfc = ts.frame[0] * (ti.bx * dstress[2])                       // ( 0,   0,  bx) * dstress
                + ts.frame[1] * (ti.bx * dstress[1]);                      // ( 0,  bx,   0) * dstress
        Deriv dfa = -dfb-dfc;
        df[t[0]] -= dfa;
        df[t[1]] -= dfb;
        df[t[2]] -= dfc;
    }
}


// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    linearsolver::BlocMatrixWriter<MatBloc> writer;
    writer.addKToMatrix(this, mparams, matrix->getMatrix(this->mstate));
}

template<class DataTypes>
template<class MatrixWriter>
void TriangularFEMForceFieldOptim<DataTypes>::addKToMatrixT(const core::MechanicalParams* mparams, MatrixWriter mwriter)
{
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = triangleInfo;
    const Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    const unsigned int nbTriangles = _topology->getNbTriangles();
    const VecElement& triangles = _topology->getTriangles();
    const Real gamma = this->gamma;
    const Real mu = this->mu;

    for ( Index i=0; i<nbTriangles; i+=1)
    {
        Triangle t = triangles[i];
        const TriangleInfo& ti = triInfo[i];
        const TriangleState& ts = triState[i];
        sofa::defaulttype::Mat<3,4,Real> KJt;
        Real factor = -kFactor * ti.ss_factor;
        Real fG = factor * gamma;
        Real fGM = factor * (gamma+mu);
        Real fM_2 = factor * (0.5f*mu);
        KJt[0][0] = fGM  *  ti.cy ;    KJt[0][1] = fG   *(-ti.cx);    KJt[0][2] = 0;    KJt[0][3] = fG   *ti.bx;
        KJt[1][0] = fG   *  ti.cy ;    KJt[1][1] = fGM  *(-ti.cx);    KJt[1][2] = 0;    KJt[1][3] = fGM  *ti.bx;

        KJt[2][0] = fM_2 *(-ti.cx);    KJt[2][1] = fM_2 *( ti.cy);    KJt[2][2] = fM_2 *ti.bx;    KJt[2][3] = 0;
        /*
        sofa::defaulttype::Mat<4,4,Real> JKJt;
        for (int j=0;j<4;++j)
        {
            JKJt[0][j] = cy*KJt[0][j] - cx*KJt[2][j];
            JKJt[1][j] = cy*KJt[2][j] - cx*KJt[1][j];
            JKJt[2][j] = bx*KJt[2][j];
            JKJt[3][j] = bx*KJt[1][j];
        }
        */
        sofa::defaulttype::Mat<2,2,Real> JKJt11, JKJt12, JKJt22;
        JKJt11[0][0] = ti.cy*KJt[0][0] - ti.cx*KJt[2][0];
        JKJt11[0][1] = ti.cy*KJt[0][1] - ti.cx*KJt[2][1];
        JKJt11[1][0] = JKJt11[0][1]; //ti.cy*KJt[2][0] - ti.cx*KJt[1][0];
        JKJt11[1][1] = ti.cy*KJt[2][1] - ti.cx*KJt[1][1];

        JKJt12[0][0] = -ti.cx*KJt[2][2];
        JKJt12[0][1] =  ti.cy*KJt[0][3];
        JKJt12[1][0] =  ti.cy*KJt[2][2];
        JKJt12[1][1] = -ti.cx*KJt[1][3];

        JKJt22[0][0] = ti.bx*KJt[2][2];
        JKJt22[0][1] = 0; //ti.bx*KJt[2][3];
        JKJt22[1][0] = 0; //ti.bx*KJt[1][2];
        JKJt22[1][1] = ti.bx*KJt[1][3];

        sofa::defaulttype::Mat<2,2,Real> JKJt00, JKJt01, JKJt02;
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

        mwriter.add(t[0],t[0],frame.multTranspose(JKJt00*frame));
        MatBloc M01 = frame.multTranspose(JKJt01*frame);
        mwriter.add(t[0],t[1],M01);    mwriter.add(t[1],t[0],M01.transposed());
        MatBloc M02 = frame.multTranspose(JKJt02*frame);
        mwriter.add(t[0],t[2],M02);    mwriter.add(t[2],t[0],M02.transposed());

        mwriter.add(t[1],t[1],frame.multTranspose(JKJt11*frame));
        MatBloc M12 = frame.multTranspose(JKJt12*frame);
        mwriter.add(t[1],t[2],M12);    mwriter.add(t[2],t[1],M12.transposed());

        mwriter.add(t[2],t[2],frame.multTranspose(JKJt22*frame));
    }
}

template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::getTriangleVonMisesStress(unsigned int i, Real& stressValue)
{
    Deriv s = triangleState[i].stress;
    Real vonMisesStress = sofa::helper::rsqrt(s[0]*s[0] - s[0]*s[1] + s[1]*s[1] + 3*s[2]);
    stressValue = vonMisesStress;
}

template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::getTrianglePrincipalStress(unsigned int i, Real& stressValue, Deriv& stressDirection)
{
    Real stressValue2;
    Deriv stressDirection2;
    getTrianglePrincipalStress(i, stressValue, stressDirection, stressValue2, stressDirection2);
}

template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::getTrianglePrincipalStress(unsigned int i, Real& stressValue, Deriv& stressDirection, Real& stressValue2, Deriv& stressDirection2)
{
    const TriangleState& ts = triangleState[i];
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
    Real deltaV = helper::rsqrt(tr*tr-4*det);
    Real eval1, eval2;
    defaulttype::Vec<2,Real> evec1, evec2;
    eval1 = (tr + deltaV)/2;
    eval2 = (tr - deltaV)/2;
    if (s[2] == 0)
    {
        evec1[0] = 1; evec1[1] = 0;
        evec2[0] = 0; evec2[1] = 1;
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

// --------------------------------------------------------------------------------------
// --- Display methods
// --------------------------------------------------------------------------------------

template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!_topology || !this->mstate) return;

    if (!vparams->displayFlags().getShowForceFields())
        return;

    using defaulttype::Vector3;
    using defaulttype::Vec3i;
    using defaulttype::Vec4f;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    unsigned int nbTriangles=_topology->getNbTriangles();
    const VecElement& triangles = _topology->getTriangles();

    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = triangleInfo;
    const bool showStressValue = this->showStressValue.getValue();
    const bool showStressVector = this->showStressVector.getValue();
    if (showStressValue || showStressVector)
    {
        Real minStress = 0;
        Real maxStress = 0;
        std::vector<Real> stresses;
        std::vector<std::pair<int,Real> > pstresses;
        std::vector<Deriv> stressVectors;
        std::vector<Real> stresses2;
        std::vector<Deriv> stressVectors2;
        stresses.resize(nbTriangles);
        //if (showStressVector)
        {
            stressVectors.resize(nbTriangles);
            stresses2.resize(nbTriangles);
            stressVectors2.resize(nbTriangles);
        }
        if (showStressValue)
        {
            pstresses.resize(x.size());
        }
        for (unsigned int i=0; i<nbTriangles; i++)
        {
            //if (showStressVector)
            {
                getTrianglePrincipalStress(i,stresses[i],stressVectors[i],stresses2[i],stressVectors2[i]);
            }
            //else
            //{
            //    getTriangleVonMisesStress(i,stresses[i]);
            //}
            if ( stresses[i] < minStress ) minStress = stresses[i];
            if ( stresses[i] > maxStress ) maxStress = stresses[i];
            if ( stresses2[i] < minStress ) minStress = stresses2[i];
            if ( stresses2[i] > maxStress ) maxStress = stresses2[i];
            if (showStressValue)
            {
                Real maxs = std::min(stresses[i],stresses2[i]);
                Triangle t = triangles[i];
                for (unsigned int j=0;j<t.size();++j)
                {
                    unsigned int p = t[j];
                    pstresses[p].first += 1;
                    pstresses[p].second += helper::rabs(maxs);
                }
            }
        }
        maxStress = std::max(-minStress, maxStress);
        minStress = 0;
        if (drawPrevMaxStress > maxStress)
        {
            maxStress = drawPrevMaxStress; //(Real)(maxStress * 0.01 + drawPrevMaxStress * 0.99);
        }
        else
        {
            drawPrevMaxStress = maxStress;
            sout << "max stress = " << maxStress << sendl;
        }
        if (showStressMaxValue.isSet())
        {
            maxStress = showStressMaxValue.getValue();
        }
#ifdef SIMPLEFEM_COLORMAP
        visualmodel::ColorMap::evaluator<Real> evalColor = showStressColorMapReal->getEvaluator(minStress, maxStress);
        if (showStressValue)
        {
            for (unsigned int i=0;i<pstresses.size();++i)
            {
                if (pstresses[i].first != 0)
                    pstresses[i].second /= pstresses[i].first;
            }
            std::vector< Vector3 > pnormals;
            pnormals.resize(x.size());
            for (unsigned int i=0; i<nbTriangles; i++)
            {
                Triangle t = triangles[i];
                Vector3 a = x[t[0]];
                Vector3 b = x[t[1]];
                Vector3 c = x[t[2]];
                Vector3 n = cross(b-a,c-a);
                n.normalize();
                pnormals[t[0]] += n;
                pnormals[t[1]] += n;
                pnormals[t[2]] += n;
            }
            for (unsigned int i=0; i<x.size(); i++)
                pnormals[i].normalize();

            std::vector< Vector3 > points;
            std::vector< Vector3 > normals;
            std::vector< Vec4f > colors;
            const float stressValueAlpha = this->showStressValueAlpha.getValue();
            if (stressValueAlpha < 1.0f)
                vparams->drawTool()->setMaterial(Vec4f(1.0f,1.0f,1.0f,stressValueAlpha));
            for (unsigned int i=0; i<nbTriangles; i++)
            {
                Triangle t = triangles[i];
                Vector3 a = x[t[0]];
                Vector3 b = x[t[1]];
                Vector3 c = x[t[2]];
                Vector3 an = pnormals[t[0]];
                Vector3 bn = pnormals[t[1]];
                Vector3 cn = pnormals[t[2]];
                //Vec4f color = evalColor(helper::rabs(stresses[i]));
                //color[3] = stressValueAlpha;
                Vector3 ab = (a+b)*0.5+(((a-b)*an)*an+((b-a)*bn)*bn)*0.25;
                Vector3 bc = (b+c)*0.5+(((b-c)*bn)*bn+((c-b)*cn)*cn)*0.25;
                Vector3 ca = (c+a)*0.5+(((c-a)*cn)*cn+((a-c)*an)*an)*0.25;
                Vec4f colora = evalColor(helper::rabs(pstresses[t[0]].second));
                Vec4f colorb = evalColor(helper::rabs(pstresses[t[1]].second));
                Vec4f colorc = evalColor(helper::rabs(pstresses[t[2]].second));
                Vec4f colorab = evalColor(helper::rabs((pstresses[t[0]].second+pstresses[t[1]].second)/2));
                Vec4f colorbc = evalColor(helper::rabs((pstresses[t[1]].second+pstresses[t[2]].second)/2));
                Vec4f colorca = evalColor(helper::rabs((pstresses[t[2]].second+pstresses[t[0]].second)/2));
                colora[3] = stressValueAlpha;
                colorb[3] = stressValueAlpha;
                colorc[3] = stressValueAlpha;
                colorab[3] = stressValueAlpha;
                colorbc[3] = stressValueAlpha;
                colorca[3] = stressValueAlpha;
                {
                Vector3 n = cross(ab-a,ca-a);
                n.normalize();
                normals.push_back(n);
                points.push_back(a);
                points.push_back(ab);
                points.push_back(ca);
                colors.push_back(colora);
                colors.push_back(colorab);
                colors.push_back(colorca);
                }
                {
                Vector3 n = cross(b-ab,bc-ab);
                n.normalize();
                normals.push_back(n);
                points.push_back(ab);
                points.push_back(b);
                points.push_back(bc);
                colors.push_back(colorab);
                colors.push_back(colorb);
                colors.push_back(colorbc);
                }
                {
                Vector3 n = cross(bc-ca,c-ca);
                n.normalize();
                normals.push_back(n);
                points.push_back(ca);
                points.push_back(bc);
                points.push_back(c);
                colors.push_back(colorca);
                colors.push_back(colorbc);
                colors.push_back(colorc);
                }
                {
                Vector3 n = cross(bc-ab,ca-ab);
                n.normalize();
                normals.push_back(n);
                points.push_back(ab);
                points.push_back(bc);
                points.push_back(ca);
                colors.push_back(colorab);
                colors.push_back(colorbc);
                colors.push_back(colorca);
                }
                /*
                {
                Vector3 n = cross(b-a,c-a);
                n.normalize();
                normals.push_back(n);
                points.push_back(a);
                points.push_back(b);
                points.push_back(c);
                colors.push_back(colora);
                colors.push_back(colorb);
                colors.push_back(colorc);
                }
                */
            }
            if (vparams->displayFlags().getShowWireFrame())
                vparams->drawTool()->setPolygonMode(0,true);
            else
            {
                vparams->drawTool()->setPolygonMode(2,true);
                vparams->drawTool()->setPolygonMode(1,false);
            }

            vparams->drawTool()->setLightingEnabled(true);
            vparams->drawTool()->drawTriangles(points, normals, colors);
            vparams->drawTool()->setLightingEnabled(false);
            if (stressValueAlpha < 1.0f)
                vparams->drawTool()->resetMaterial(Vec4f(1.0f,1.0f,1.0f,stressValueAlpha));
 
            vparams->drawTool()->setPolygonMode(0,false);
       }
#endif
        if (showStressVector && maxStress > 0)
        {
            std::vector< Vector3 > points[2];
            for (unsigned int i=0; i<nbTriangles; i++)
            {
                Triangle t = triangles[i];
                Vector3 a = x[t[0]];
                Vector3 b = x[t[1]];
                Vector3 c = x[t[2]];
                Vector3 d1 = stressVectors[i];
                Real s1 = stresses[i];
                Vector3 d2 = stressVectors2[i];
                Real s2 = stresses2[i];
                Vector3 center = (a+b+c)/3;
                Vector3 n = cross(b-a,c-a);
                Real fact = (Real)helper::rsqrt(n.norm())*(Real)0.5;
                int g1 = (s1 < 0) ? 1 : 0;
                int g2 = (s2 < 0) ? 1 : 0;
                d1 *= fact*helper::rsqrt(helper::rabs(s1)/maxStress);
                d2 *= fact*helper::rsqrt(helper::rabs(s2)/maxStress);
                points[g1].push_back(center - d1);
                points[g1].push_back(center + d1);
                points[g2].push_back(center - d2);
                points[g2].push_back(center + d2);
            }
            vparams->drawTool()->drawLines(points[0], 2, Vec4f(1,1,0,1));
            vparams->drawTool()->drawLines(points[1], 2, Vec4f(1,0,1,1));
        }
    }
    else
    {
        std::vector< Vector3 > points[4];

        const Vec4f c0(1,0,0,1);
        const Vec4f c1(0,1,0,1);
        const Vec4f c2(1,0.5,0,1);
        const Vec4f c3(0,0,1,1);

        points[0].reserve(nbTriangles*2);
        points[1].reserve(nbTriangles*2);
        points[2].reserve(nbTriangles*6);
        points[3].reserve(nbTriangles*6);
        for (unsigned int i=0; i<nbTriangles; ++i)
        {
            Triangle t = triangles[i];
            const TriangleInfo& ti = triInfo[i];
            const TriangleState& ts = triState[i];
            Coord a = x[t[0]];
            Coord b = x[t[1]];
            Coord c = x[t[2]];
            Coord fx = ts.frame[0];
            Coord fy = ts.frame[1];
            Vector3 center = (a+b+c)*(1.0f/3.0f);
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

} // namespace forcefield

} // namespace component

} // namespace sofa


#endif //SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_INL
