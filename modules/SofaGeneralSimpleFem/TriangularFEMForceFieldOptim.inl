/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
    : d_triangleInfo(initData(&d_triangleInfo, "triangleInfo", "Internal triangle data (persistent)"))
    , d_triangleState(initData(&d_triangleState, "triangleState", "Internal triangle data (time-dependent)"))
    , d_vertexInfo(initData(&d_vertexInfo, "vertexInfo", "Internal point data"))
    , d_edgeInfo(initData(&d_edgeInfo, "edgeInfo", "Internal edge data"))
    , _topology(NULL)
    , d_poisson(initData(&d_poisson,(Real)(0.45),"poissonRatio","Poisson ratio in Hooke's law"))
    , d_young(initData(&d_young,(Real)(1000.0),"youngModulus","Young modulus in Hooke's law"))
    , d_damping(initData(&d_damping,(Real)0.,"damping","Ratio damping/stiffness"))
    , d_restScale(initData(&d_restScale,(Real)1.,"restScale","Scale factor applied to rest positions (to simulate pre-stretched materials)"))
    , d_showStressVector(initData(&d_showStressVector,false,"showStressVector","Flag activating rendering of stress directions within each triangle"))
    , d_showStressMaxValue(initData(&d_showStressMaxValue,(Real)0.0,"showStressMaxValue","Max value for rendering of stress values"))
    , drawPrevMaxStress((Real)-1.0)
{
    triangleInfoHandler = new TFEMFFOTriangleInfoHandler(this, &d_triangleInfo);
    triangleStateHandler = new TFEMFFOTriangleStateHandler(this, &d_triangleState);

    d_poisson.setRequired(true);
    d_young.setRequired(true);
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
    d_triangleInfo.createTopologicalEngine(_topology, triangleInfoHandler);
    d_triangleInfo.registerTopologicalData();

    d_triangleState.createTopologicalEngine(_topology, triangleStateHandler);
    d_triangleState.registerTopologicalData();

    d_edgeInfo.createTopologicalEngine(_topology);
    d_edgeInfo.registerTopologicalData();

    d_vertexInfo.createTopologicalEngine(_topology);
    d_vertexInfo.registerTopologicalData();

    if (_topology->getNbTriangles()==0 && _topology->getNbQuads()!=0 )
    {
        msg_warning() << "The topology only contains quads while this forcefield only supports triangles."<<msgendl;
    }

    reinit();
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::parse( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    const char* method = arg->getAttribute("method");
    if (method && *method && std::string(method) != std::string("large"))
    {
        msg_warning() << "Attribute method was specified as \""<<method<<"\" while this version only implements the \"large\" method. Ignoring..." << sendl;
    }
    Inherited::parse(arg);
}

template<class DataTypes>
class TriangularFEMForceFieldOptim<DataTypes>::EdgeInfo
{
public:
    bool fracturable;

    EdgeInfo()
        : fracturable(false) { }

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const EdgeInfo& /*ei*/ )
    {
        return os;
    }

    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, EdgeInfo& /*ei*/ )
    {
        return in;
    }
};

template<class DataTypes>
class TriangularFEMForceFieldOptim<DataTypes>::VertexInfo
{
public:
    VertexInfo()
    /*:sumEigenValues(0.0)*/ {}

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const VertexInfo& /*vi*/)
    {
        return os;
    }
    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, VertexInfo& /*vi*/)
    {
        return in;
    }
};

template<class DataTypes>
class TriangularFEMForceFieldOptim<DataTypes>::TriangleState
{
public:
    Transformation frame; // Mat<2,3,Real>
    Deriv stress;

    TriangleState() { }

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const TriangleState& ti )
    {
        return os << "frame= " << ti.frame << " stress= " << ti.stress << " END";
    }

    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, TriangleState& ti )
    {
        std::string str;
        while (in >> str)
        {
            if (str == "END") break;
            else if (str == "frame=") in >> ti.frame;
            else if (str == "stress=") in >> ti.stress;
            else if (!str.empty() && str[str.length()-1]=='=') in >> str; // unknown value
        }
        return in;
    }
};

template<class DataTypes>
class TriangularFEMForceFieldOptim<DataTypes>::TriangleInfo
{
public:
    //Index ia, ib, ic;
    Real bx, cx, cy, ss_factor;
    Transformation init_frame; // Mat<2,3,Real>

    TriangleInfo() { }

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const TriangleInfo& ti )
    {
        return os << "bx= " << ti.bx << " cx= " << ti.cx << " cy= " << ti.cy << " ss_factor= " << ti.ss_factor << " init_frame= " << ti.init_frame << " END";
    }

    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, TriangleInfo& ti )
    {
        std::string str;
        while (in >> str)
        {
            if (str == "END") break;
            else if (str == "bx=") in >> ti.bx;
            else if (str == "cx=") in >> ti.cx;
            else if (str == "cy=") in >> ti.cy;
            else if (str == "ss_factor=") in >> ti.ss_factor;
            else if (str == "init_frame=") in >> ti.init_frame;
            else if (!str.empty() && str[str.length()-1]=='=') in >> str; // unknown value
        }
        return in;
    }
};
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
        msg_error() << "INVALID point index >= " << x0.size() << " in triangle " << i << " : " << t
             << this->getContext()->getMeshTopology()->getNbPoints() << "/"
             << this->mstate->getContext()->getMeshTopology()->getNbPoints() << " points,"
             << this->getContext()->getMeshTopology()->getNbTriangles() << "/"
             << this->mstate->getContext()->getMeshTopology()->getNbTriangles() << " triangles." << msgendl;
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
    computeTriangleRotation(ti.init_frame, ab, ac);
    ti.bx = ti.init_frame[0] * ab;
    ti.cx = ti.init_frame[0] * ac;
    ti.cy = ti.init_frame[1] * ac;
    ti.ss_factor = ((Real)0.5)/(ti.bx*ti.cy);
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::initTriangleState(unsigned int i, TriangleState& ti, const Triangle t, const VecCoord& x)
{
    if (t[0] >= x.size() || t[1] >= x.size() || t[2] >= x.size())
    {
        msg_error() << "INVALID point index >= " << x.size() << " in triangle " << i << " : " << t
             << this->getContext()->getMeshTopology()->getNbPoints() << "/"
             << this->mstate->getContext()->getMeshTopology()->getNbPoints() << " points,"
             << this->getContext()->getMeshTopology()->getNbTriangles() << "/"
             << this->mstate->getContext()->getMeshTopology()->getNbTriangles() << " triangles." << msgendl;
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
    const Real youngModulus = d_young.getValue();
    const Real poissonRatio = d_poisson.getValue();
    mu = (youngModulus) / (1+poissonRatio);
    gamma = (youngModulus * poissonRatio) / (1-poissonRatio*poissonRatio);

    /// prepare to store info in the triangle array
    const unsigned int nbTriangles = _topology->getNbTriangles();
    const VecElement& triangles = _topology->getTriangles();
    const  VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const  VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
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

    /// prepare to store info in the edge array
    VecEdgeInfo& edgeInf = *(d_edgeInfo.beginEdit());
    edgeInf.resize(_topology->getNbEdges());
    d_edgeInfo.endEdit();

    /// prepare to store info in the vertex array
    unsigned int nbPoints = _topology->getNbPoints();
    VecVertexInfo& vi = *(d_vertexInfo.beginEdit());
    vi.resize(nbPoints);
    d_vertexInfo.endEdit();
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
        Real dcx = ti.cx - ts.frame[0]*ac;
        Real dcy = ti.cy - ts.frame[1]*ac;

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
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = d_triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;

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
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = d_triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;
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
    Deriv s = d_triangleState[i].stress;
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
    const TriangleState& ts = d_triangleState[i];
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
    defaulttype::Vec<2,Real> evec1, evec2;
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

    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleState > > triState = d_triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecTriangleInfo > > triInfo = d_triangleInfo;
    const bool showStressValue = this->d_showStressValue.getValue();
    const bool showStressVector = this->d_showStressVector.getValue();
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
        stressVectors.resize(nbTriangles);
        stresses2.resize(nbTriangles);
        stressVectors2.resize(nbTriangles);

        if (showStressValue)
        {
            pstresses.resize(x.size());
        }
        for (unsigned int i=0; i<nbTriangles; i++)
        {
            getTrianglePrincipalStress(i,stresses[i],stressVectors[i],stresses2[i],stressVectors2[i]);

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
            maxStress = drawPrevMaxStress;
        }
        else
        {
            drawPrevMaxStress = maxStress;
            msg_info() << "max stress = " << maxStress << sendl;
        }
        if (d_showStressMaxValue.isSet())
        {
            maxStress = d_showStressMaxValue.getValue();
        }
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
