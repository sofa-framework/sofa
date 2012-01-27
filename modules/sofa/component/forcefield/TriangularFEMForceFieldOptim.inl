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

#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/forcefield/TriangularFEMForceFieldOptim.h>
#include <sofa/core/behavior/ForceField.inl>

#include <sofa/core/visual/VisualParams.h>

#include <sofa/component/topology/TopologyData.inl>

#include <limits>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace	sofa::component::topology;
using namespace core::topology;

// --------------------------------------------------------------------------------------
// ---  Topology Creation/Destruction functions
// --------------------------------------------------------------------------------------

template< class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::TFEMFFOTriangleInfoHandler::applyCreateFunction(unsigned int triangleIndex, TriangleInfo &ti, const Triangle &t, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {
        ff->initTriangleInfo(triangleIndex,ti,t, *ff->mstate->getX0());
    }
}

template< class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::TFEMFFOTriangleStateHandler::applyCreateFunction(unsigned int triangleIndex, TriangleState &ti, const Triangle &t, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {
        ff->initTriangleState(triangleIndex,ti,t, *ff->mstate->getX());
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
    , f_poisson(initData(&f_poisson,(Real)(0.45),"poissonRatio","Poisson ratio in Hooke's law"))
    , f_young(initData(&f_young,(Real)(1000.0),"youngModulus","Young modulus in Hooke's law"))
    , f_damping(initData(&f_damping,(Real)0.,"damping","Ratio damping/stiffness"))
    , showStressValue(initData(&showStressValue,false,"showStressValue","Flag activating rendering of stress values as a color in each triangle"))
    , showStressVector(initData(&showStressVector,false,"showStressVector","Flag activating rendering of stress directions within each triangle"))
{
    triangleInfoHandler = new TFEMFFOTriangleInfoHandler(this, &triangleInfo);
    triangleStateHandler = new TFEMFFOTriangleStateHandler(this, &triangleState);
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

    if (_topology->getNbTriangles()==0)
    {
        serr << "Topology is empty of not triangular."<<sendl;
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
void TriangularFEMForceFieldOptim<DataTypes>::initTriangleInfo(unsigned int /*i*/, TriangleInfo& ti, const Triangle t, const VecCoord& x0)
{
    Coord a  = x0[t[0]];
    Coord ab = x0[t[1]]-a;
    Coord ac = x0[t[2]]-a;
    computeTriangleRotation(ti.init_frame, ab, ac);
    ti.bx = ti.init_frame[0] * ab;
    ti.cx = ti.init_frame[0] * ac;
    ti.cy = ti.init_frame[1] * ac;
    //ti.ss_factor = ((Real)1.0)/(ti.bx*ti.bx*ti.cy*ti.cy);
    ti.ss_factor = ((Real)0.5)/(ti.bx*ti.cy);
}

template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::initTriangleState(unsigned int /*i*/, TriangleState& ti, const Triangle t, const VecCoord& x)
{
    Coord a  = x[t[0]];
    Coord ab = x[t[1]]-a;
    Coord ac = x[t[2]]-a;
    computeTriangleRotation(ti.frame, ab, ac);
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
    const  VecCoord& x = *this->mstate->getX();
    const  VecCoord& x0 = *this->mstate->getX0();
    helper::vector<TriangleInfo>& triangleInf = *(triangleInfo.beginEdit());
    helper::vector<TriangleState>& triangleSta = *(triangleState.beginEdit());
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
    helper::vector<EdgeInfo>& edgeInf = *(edgeInfo.beginEdit());
    edgeInf.resize(_topology->getNbEdges());
    edgeInfo.endEdit();

    /// prepare to store info in the vertex array
    unsigned int nbPoints = _topology->getNbPoints();
    helper::vector<VertexInfo>& vi = *(vertexInfo.beginEdit());
    vi.resize(nbPoints);
    vertexInfo.endEdit();
}



template <class DataTypes>
double TriangularFEMForceFieldOptim<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, const DataVecCoord& /* x */) const
{
    serr<<"TriangularFEMForceFieldOptim::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}


// --------------------------------------------------------------------------------------
// --- AddForce and AddDForce methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > f = d_f;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > x = d_x;
    sofa::helper::WriteAccessor< core::objectmodel::Data< helper::vector<TriangleState> > > triState = triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< helper::vector<TriangleInfo> > > triInfo = triangleInfo;

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

        Vec<3,Real> strain (
            ti.cy * dbx,                // ( cy,   0,  0,  0) * (dbx, dby, dcx, dcy)
            ti.bx * dcy,                // (  0, -cx,  0, bx) * (dbx, dby, dcx, dcy)
            ti.bx * dcx - ti.cx * dbx); // (-cx,  cy, bx,  0) * (dbx, dby, dcx, dcy)

        Real gammaXY = gamma*(strain[0]+strain[1]);

        Vec<3,Real> stress (
            mu*strain[0] + gammaXY,    // (gamma+mu, gamma   ,    0) * strain
            mu*strain[1] + gammaXY,    // (gamma   , gamma+mu,    0) * strain
            (Real)(0.5)*mu*strain[2]); // (       0,        0, mu/2) * strain

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
void TriangularFEMForceFieldOptim<DataTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > df = d_df;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > dx = d_dx;
    sofa::helper::ReadAccessor< core::objectmodel::Data< helper::vector<TriangleState> > > triState = triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< helper::vector<TriangleInfo> > > triInfo = triangleInfo;

    const unsigned int nbTriangles = _topology->getNbTriangles();
    const VecElement& triangles = _topology->getTriangles();
    const Real gamma = this->gamma;
    const Real mu = this->mu;
    const Real kFactor = (Real)mparams->kFactor();

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

        Vec<3,Real> dstrain (
            ti.cy  * dbx, // (cy,  0, 0, 0) * (dbx, dby, dcx, dcy)
            ti.bx * dcy - ti.cx * dby, // (0, -cx, 0, bx) * (dbx, dby, dcx, dcy)
            ti.bx * dcx - ti.cx * dbx + ti.cy * dby); // (-cx, cy, bx, 0) * (dbx, dby, dcx, dcy)

        Real gammaXY = gamma*(dstrain[0]+dstrain[1]);

        Vec<3,Real> dstress (
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
// --- Display methods
// --------------------------------------------------------------------------------------

template<class DataTypes>
void TriangularFEMForceFieldOptim<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!_topology || !this->mstate) return;

    if (!vparams->displayFlags().getShowForceFields())
        return;

    std::vector< Vector3 > points[4];

    const Vec<4,float> c0(1,0,0,1);
    const Vec<4,float> c1(0,1,0,1);
    const Vec<4,float> c2(1,0.5,0,1);
    const Vec<4,float> c3(0,0,1,1);

    const VecCoord& x = *this->mstate->getX();
    unsigned int nbTriangles=_topology->getNbTriangles();
    const VecElement& triangles = _topology->getTriangles();

    sofa::helper::ReadAccessor< core::objectmodel::Data< helper::vector<TriangleState> > > triState = triangleState;
    sofa::helper::ReadAccessor< core::objectmodel::Data< helper::vector<TriangleInfo> > > triInfo = triangleInfo;
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

} // namespace forcefield

} // namespace component

} // namespace sofa


#endif //SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_INL
