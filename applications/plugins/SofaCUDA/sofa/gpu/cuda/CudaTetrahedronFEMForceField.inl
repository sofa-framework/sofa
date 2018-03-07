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
#ifndef SOFA_GPU_CUDA_CUDATETRAHEDRONFEMFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDATETRAHEDRONFEMFORCEFIELD_INL

#include "CudaTetrahedronFEMForceField.h"
#include <SofaSimpleFem/TetrahedronFEMForceField.inl>
#if 0 //defined(SOFA_DEV)
#include <sofa/gpu/cuda/CudaDiagonalMatrix.h>
#include <sofa/gpu/cuda/CudaRotationMatrix.h>
#include <sofa/core/behavior/RotationMatrix.h>
#endif // SOFA_DEV
namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void TetrahedronFEMForceFieldCuda3f_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

    void TetrahedronFEMForceFieldCuda3f1_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f1_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

    void TetrahedronFEMForceFieldCuda3f_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
    void TetrahedronFEMForceFieldCuda3f_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void TetrahedronFEMForceFieldCuda3d_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3d_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

    void TetrahedronFEMForceFieldCuda3d1_addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3d1_addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

    void TetrahedronFEMForceFieldCuda3d_getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations);
    void TetrahedronFEMForceFieldCuda3d_getElementRotations(unsigned int nbElem, const void* rotationsAos, void* rotations);

#endif // SOFA_GPU_CUDA_DOUBLE

} // extern "C"

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3fTypes>
{
public:
    static void addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   TetrahedronFEMForceFieldCuda3f_addForce(bsize,pt,nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, f, x, v); }
    static void addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
    {   TetrahedronFEMForceFieldCuda3f_addDForce(bsize,pt,nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, df, dx, factor); }

    static void getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
    {   TetrahedronFEMForceFieldCuda3f_getRotations(nbElem, nbVertex, initState, state, rotationIdx, rotations); }

    static void getRotationsElement(unsigned int nbElem, const void* rotationsAos, void* rotations)
    {   TetrahedronFEMForceFieldCuda3f_getElementRotations(nbElem, rotationsAos, rotations); }
};

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3f1Types>
{
public:
    static void addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   TetrahedronFEMForceFieldCuda3f1_addForce(bsize,pt,nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, f, x, v); }
    static void addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
    {   TetrahedronFEMForceFieldCuda3f1_addDForce(bsize,pt,nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, df, dx, factor); }

    static void getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
    {   TetrahedronFEMForceFieldCuda3f_getRotations(nbElem, nbVertex, initState, state, rotationIdx, rotations); }

    static void getRotationsElement(unsigned int nbElem, const void* rotationsAos, void* rotations)
    {   TetrahedronFEMForceFieldCuda3f_getElementRotations(nbElem, rotationsAos, rotations); }
};

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3dTypes>
{
public:
    static void addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   TetrahedronFEMForceFieldCuda3d_addForce(bsize,pt,nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, f, x, v); }
    static void addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
    {   TetrahedronFEMForceFieldCuda3d_addDForce(bsize,pt,nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, df, dx, factor); }

    static void getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
    {   TetrahedronFEMForceFieldCuda3d_getRotations(nbElem, nbVertex, initState, state, rotationIdx, rotations); }

    static void getRotationsElement(unsigned int nbElem, const void* rotationsAos, void* rotations)
    {   TetrahedronFEMForceFieldCuda3d_getElementRotations(nbElem, rotationsAos, rotations); }

};

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3d1Types>
{
public:
    static void addForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   TetrahedronFEMForceFieldCuda3d1_addForce(bsize,pt,nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, f, x, v); }
    static void addDForce(int bsize,int pt,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
    {   TetrahedronFEMForceFieldCuda3d1_addDForce(bsize,pt,nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, df, dx, factor); }

    static void getRotations(unsigned int nbElem, unsigned int nbVertex, const void* initState, const void* state, const void* rotationIdx, void* rotations)
    {   TetrahedronFEMForceFieldCuda3d_getRotations(nbElem, nbVertex, initState, state, rotationIdx, rotations); }

    static void getRotationsElement(unsigned int nbElem, const void* rotationsAos, void* rotations)
    {   TetrahedronFEMForceFieldCuda3d_getElementRotations(nbElem, rotationsAos, rotations); }

};

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::reinit(Main* m)
{
    if (!m->_mesh->getTetrahedra().empty())
    {
        m->_indexedElements = & (m->_mesh->getTetrahedra());
    }

    Data& data = m->data;
    m->strainDisplacements.resize( m->_indexedElements->size() );
    m->materialsStiffnesses.resize(m->_indexedElements->size() );

    const VecElement& elems = *m->_indexedElements;

    const VecCoord& p = m->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    m->_initialPoints.setValue(p);

    m->rotations.resize( m->_indexedElements->size() );
    m->_initialRotations.resize( m->_indexedElements->size() );
    m->_rotationIdx.resize(m->_indexedElements->size() *4);
    m->_rotatedInitialElements.resize(m->_indexedElements->size());

    std::vector<int> activeElems;
    for (unsigned int i=0; i<elems.size(); i++)
    {
        {
            activeElems.push_back(i);
        }
    }

    for (unsigned int i=0; i<activeElems.size(); i++)
    {
        int ei = activeElems[i];
        Index a = elems[ei][0];
        Index b = elems[ei][1];
        Index c = elems[ei][2];
        Index d = elems[ei][3];
        m->computeMaterialStiffness(ei,a,b,c,d);
        m->initLarge(ei,a,b,c,d);
    }

    std::map<int,int> nelems;
    for (unsigned int i=0; i<activeElems.size(); i++)
    {
        int ei = activeElems[i];
        const Element& e = elems[ei];
        for (unsigned int j=0; j<e.size(); j++)
            ++nelems[e[j]];
    }
    int nmax = 0;
    for (std::map<int,int>::const_iterator it = nelems.begin(); it != nelems.end(); ++it)
        if (it->second > nmax)
            nmax = it->second;
    int v0 = 0;
    int nbv = 0;
    if (!nelems.empty())
    {
        v0 = nelems.begin()->first;
        nbv = nelems.rbegin()->first - v0 + 1;
    }

    data.init(activeElems.size(), v0, nbv, nmax);

//     nelems.clear();
//     for (unsigned int i=0;i<activeElems.size();i++)
//     {
//         int ei = activeElems[i];
//         const Element& e = elems[ei];
//         const Coord& a = m->_rotatedInitialElements[ei][0];
//         const Coord& b = m->_rotatedInitialElements[ei][1];
//         const Coord& c = m->_rotatedInitialElements[ei][2];
//         const Coord& d = m->_rotatedInitialElements[ei][3];
//         data.setE(i, e, a, b, c, d, m->materialsStiffnesses[ei], m->strainDisplacements[ei]);
//         for (unsigned int j=0;j<e.size();j++)
//             data.setV(e[j], nelems[e[j]]++, i*e.size()+j);
//     }


    data.nbElementPerVertex = nmax;
    std::istringstream ptchar(m->_gatherPt.getValue().getSelectedItem());
    std::istringstream bschar(m->_gatherBsize.getValue().getSelectedItem());
    ptchar >> data.GATHER_PT;
    bschar >> data.GATHER_BSIZE;

    int nbElemPerThread = (data.nbElementPerVertex+data.GATHER_PT-1)/data.GATHER_PT;
    int nbBpt = (data.nbVertex*data.GATHER_PT + data.GATHER_BSIZE-1)/data.GATHER_BSIZE;
    data.velems.resize(nbBpt*nbElemPerThread*data.GATHER_BSIZE);

    nelems.clear();
    for (unsigned eindex = 0; eindex < activeElems.size(); ++eindex)
    {
        int ei = activeElems[eindex];
        const Element& e = elems[ei];

        const Coord& a = m->_rotatedInitialElements[ei][0];
        const Coord& b = m->_rotatedInitialElements[ei][1];
        const Coord& c = m->_rotatedInitialElements[ei][2];
        const Coord& d = m->_rotatedInitialElements[ei][3];
        data.setE(eindex, e, a, b, c, d, m->materialsStiffnesses[ei], m->strainDisplacements[ei]);

        for (unsigned j = 0; j < e.size(); ++j)
        {
            int p = e[j] - data.vertex0;
            int num = nelems[p]++;

            if (data.GATHER_PT > 1)
            {
                const int block  = (p*data.GATHER_PT) / data.GATHER_BSIZE;
                const int thread = (p*data.GATHER_PT+(num%data.GATHER_PT)) % data.GATHER_BSIZE;
                num = num/data.GATHER_PT;
                data.velems[ block * (nbElemPerThread * data.GATHER_BSIZE) + num * data.GATHER_BSIZE + thread ] = 1 + eindex * e.size() + j;
            }
            else
            {
                const int block  = p / data.GATHER_BSIZE;
                const int thread = p % data.GATHER_BSIZE;
                data.velems[ block * (data.nbElementPerVertex * data.GATHER_BSIZE) + num * data.GATHER_BSIZE + thread ] = 1 + eindex * e.size() + j;
            }
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addForce(Main* m, VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    if (m->needUpdateTopology)
    {
        reinit(m);
        m->needUpdateTopology = false;
    }
    Data& data = m->data;

    f.resize(x.size());

    Kernels::addForce(
        data.GATHER_BSIZE,
        data.GATHER_PT,
        data.size(),
        data.nbVertex,
        data.nbElementPerVertex,
        data.elems.deviceRead(),
        data.state.deviceWrite(),
        data.eforce.deviceWrite(),
        data.velems.deviceRead(),
        (      Deriv*)f.deviceWrite() + data.vertex0,
        (const Coord*)x.deviceRead()  + data.vertex0,
        (const Deriv*)v.deviceRead()  + data.vertex0);

#if 0
    // compare with CPU version

    const VecElement& elems = *m->_indexedElements;
    for (unsigned int i=0; i<elems.size(); i++)
    {
        Index a = elems[i][0];
        Index b = elems[i][1];
        Index c = elems[i][2];
        Index d = elems[i][3];
        typename Main::Transformation Rt;
        m->computeRotationLarge(Rt, x, a, b, c);
        const GPUElementState& s = data.state[i];
        const GPUElement& e = data.elems[i];
        Mat3x3f Rdiff = Rt-s.Rt;
        if ((Rdiff[0].norm2()+Rdiff[1].norm2()+Rdiff[2].norm2()) > 0.000001f)
        {
            sout << "CPU Rt "<<i<<" = "<<Rt<<sendl;
            sout << "GPU Rt "<<i<<" = "<<s.Rt<<sendl;
            sout << "DIFF   "<<i<<" = "<<Rdiff<<sendl;
        }
        Coord xb = Rt*(x[b]-x[a]);
        Coord xc = Rt*(x[c]-x[a]);
        Coord xd = Rt*(x[d]-x[a]);

        typename Main::Displacement D;
        D[0] = 0;
        D[1] = 0;
        D[2] = 0;
        D[3] = m->_rotatedInitialElements[i][1][0] - xb[0];
        D[4] = m->_rotatedInitialElements[i][1][1] - xb[1];
        D[5] = m->_rotatedInitialElements[i][1][2] - xb[2];
        D[6] = m->_rotatedInitialElements[i][2][0] - xc[0];
        D[7] = m->_rotatedInitialElements[i][2][1] - xc[1];
        D[8] = m->_rotatedInitialElements[i][2][2] - xc[2];
        D[9] = m->_rotatedInitialElements[i][3][0] - xd[0];
        D[10]= m->_rotatedInitialElements[i][3][1] - xd[1];
        D[11]= m->_rotatedInitialElements[i][3][2] - xd[2];
        Vec<6,Real> S = -((m->materialsStiffnesses[i]) * ((m->strainDisplacements[i]).multTranspose(D)))*(e.bx);

        Vec<6,Real> Sdiff = S-s.S;

        if (Sdiff.norm2() > 0.0001f)
        {
            sout << "    D "<<i<<" = "<<D<<sendl;
            sout << "CPU S "<<i<<" = "<<S<<sendl;
            sout << "GPU S "<<i<<" = "<<s.S<<sendl;
            sout << "DIFF   "<<i<<" = "<<Sdiff<<sendl;
        }

    }
#endif

}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addDForce (Main* m, VecDeriv& df, const VecDeriv& dx, SReal kFactor, SReal /*bFactor*/)
{
    Data& data = m->data;
    df.resize(dx.size());
    Kernels::addDForce(
        data.GATHER_BSIZE,
        data.GATHER_PT,
        data.size(),
        data.nbVertex,
        data.nbElementPerVertex,
        data.elems.deviceRead(),
        data.state.deviceRead(),
        data.eforce.deviceWrite(),
        data.velems.deviceRead(),
        (      Deriv*)df.deviceWrite() + data.vertex0,
        (const Deriv*)dx.deviceRead()  + data.vertex0,
        kFactor);
}


template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addKToMatrix(Main* m, sofa::defaulttype::BaseMatrix *mat, SReal k, unsigned int &offset)
{
        Data& data = m->data;

        if (sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> > * crsmat = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> > * >(mat))
        {
            const VecElement& elems = *m->_indexedElements;

            helper::ReadAccessor< gpu::cuda::CudaVector<GPUElementState> > state = data.state;

            // Build Matrix Block for this ForceField
            int i,j,n1, n2;
            int offd3 = offset/3;

            typename Main::Transformation Rot;
            typename Main::StiffnessMatrix JKJt,tmp;

            Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
            Rot[0][1]=Rot[0][2]=0;
            Rot[1][0]=Rot[1][2]=0;
            Rot[2][0]=Rot[2][1]=0;

            for (int ei=0; ei<data.nbElement; ++ei)
            {
                const Element& e = elems[ei];

                int blockIdx = ei / BSIZE;
                int threadIdx = ei % BSIZE;

                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        Rot[j][i] = state[blockIdx].Rt[i][j][threadIdx];

                m->computeStiffnessMatrix(JKJt, tmp, m->materialsStiffnesses[ei], m->strainDisplacements[ei], Rot);
                defaulttype::Mat<3,3,double> tmpBlock[4][4];

                // find index of node 1
                for (n1=0; n1<4; n1++)
                {
                    for(i=0; i<3; i++)
                    {
                        for (n2=0; n2<4; n2++)
                        {
                            for (j=0; j<3; j++)
                            {
                                tmpBlock[n1][n2][i][j] = - tmp[n1*3+i][n2*3+j]*k;
                            }
                        }
                    }
                }

                *crsmat->wbloc(offd3 + e[0], offd3 + e[0],true) += tmpBlock[0][0];
                *crsmat->wbloc(offd3 + e[0], offd3 + e[1],true) += tmpBlock[0][1];
                *crsmat->wbloc(offd3 + e[0], offd3 + e[2],true) += tmpBlock[0][2];
                *crsmat->wbloc(offd3 + e[0], offd3 + e[3],true) += tmpBlock[0][3];

                *crsmat->wbloc(offd3 + e[1], offd3 + e[0],true) += tmpBlock[1][0];
                *crsmat->wbloc(offd3 + e[1], offd3 + e[1],true) += tmpBlock[1][1];
                *crsmat->wbloc(offd3 + e[1], offd3 + e[2],true) += tmpBlock[1][2];
                *crsmat->wbloc(offd3 + e[1], offd3 + e[3],true) += tmpBlock[1][3];

                *crsmat->wbloc(offd3 + e[2], offd3 + e[0],true) += tmpBlock[2][0];
                *crsmat->wbloc(offd3 + e[2], offd3 + e[1],true) += tmpBlock[2][1];
                *crsmat->wbloc(offd3 + e[2], offd3 + e[2],true) += tmpBlock[2][2];
                *crsmat->wbloc(offd3 + e[2], offd3 + e[3],true) += tmpBlock[2][3];

                *crsmat->wbloc(offd3 + e[3], offd3 + e[0],true) += tmpBlock[3][0];
                *crsmat->wbloc(offd3 + e[3], offd3 + e[1],true) += tmpBlock[3][1];
                *crsmat->wbloc(offd3 + e[3], offd3 + e[2],true) += tmpBlock[3][2];
                *crsmat->wbloc(offd3 + e[3], offd3 + e[3],true) += tmpBlock[3][3];
            }
        }
        else if (sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> > * crsmat = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> > * >(mat))
        {
            const VecElement& elems = *m->_indexedElements;

            helper::ReadAccessor< gpu::cuda::CudaVector<GPUElementState> > state = data.state;

            // Build Matrix Block for this ForceField
            int i,j,n1, n2;
            int offd3 = offset/3;

            typename Main::Transformation Rot;
            typename Main::StiffnessMatrix JKJt,tmp;

            Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
            Rot[0][1]=Rot[0][2]=0;
            Rot[1][0]=Rot[1][2]=0;
            Rot[2][0]=Rot[2][1]=0;

            for (int ei=0; ei<data.nbElement; ++ei)
            {
                const Element& e = elems[ei];

                int blockIdx = ei / BSIZE;
                int threadIdx = ei % BSIZE;

                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        Rot[j][i] = state[blockIdx].Rt[i][j][threadIdx];

                m->computeStiffnessMatrix(JKJt, tmp, m->materialsStiffnesses[ei], m->strainDisplacements[ei], Rot);
                defaulttype::Mat<3,3,double> tmpBlock[4][4];

                // find index of node 1
                for (n1=0; n1<4; n1++)
                {
                    for(i=0; i<3; i++)
                    {
                        for (n2=0; n2<4; n2++)
                        {
                            for (j=0; j<3; j++)
                            {
                                tmpBlock[n1][n2][i][j] = - tmp[n1*3+i][n2*3+j]*k;
                            }
                        }
                    }
                }

                *crsmat->wbloc(offd3 + e[0], offd3 + e[0],true) += tmpBlock[0][0];
                *crsmat->wbloc(offd3 + e[0], offd3 + e[1],true) += tmpBlock[0][1];
                *crsmat->wbloc(offd3 + e[0], offd3 + e[2],true) += tmpBlock[0][2];
                *crsmat->wbloc(offd3 + e[0], offd3 + e[3],true) += tmpBlock[0][3];

                *crsmat->wbloc(offd3 + e[1], offd3 + e[0],true) += tmpBlock[1][0];
                *crsmat->wbloc(offd3 + e[1], offd3 + e[1],true) += tmpBlock[1][1];
                *crsmat->wbloc(offd3 + e[1], offd3 + e[2],true) += tmpBlock[1][2];
                *crsmat->wbloc(offd3 + e[1], offd3 + e[3],true) += tmpBlock[1][3];

                *crsmat->wbloc(offd3 + e[2], offd3 + e[0],true) += tmpBlock[2][0];
                *crsmat->wbloc(offd3 + e[2], offd3 + e[1],true) += tmpBlock[2][1];
                *crsmat->wbloc(offd3 + e[2], offd3 + e[2],true) += tmpBlock[2][2];
                *crsmat->wbloc(offd3 + e[2], offd3 + e[3],true) += tmpBlock[2][3];

                *crsmat->wbloc(offd3 + e[3], offd3 + e[0],true) += tmpBlock[3][0];
                *crsmat->wbloc(offd3 + e[3], offd3 + e[1],true) += tmpBlock[3][1];
                *crsmat->wbloc(offd3 + e[3], offd3 + e[2],true) += tmpBlock[3][2];
                *crsmat->wbloc(offd3 + e[3], offd3 + e[3],true) += tmpBlock[3][3];
            }
        }
        else
        {
            const VecElement& elems = *m->_indexedElements;

            helper::ReadAccessor< gpu::cuda::CudaVector<GPUElementState> > state = data.state;

            // Build Matrix Block for this ForceField
            int i,j,n1, n2, row, column, ROW, COLUMN;

            typename Main::Transformation Rot;
            typename Main::StiffnessMatrix JKJt,tmp;

            Index noeud1, noeud2;

            Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
            Rot[0][1]=Rot[0][2]=0;
            Rot[1][0]=Rot[1][2]=0;
            Rot[2][0]=Rot[2][1]=0;

            for (int ei=0; ei<data.nbElement; ++ei)
            {
                const Element& e = elems[ei];

                int blockIdx = ei / BSIZE;
                int threadIdx = ei % BSIZE;

                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        Rot[j][i] = state[blockIdx].Rt[i][j][threadIdx];

                m->computeStiffnessMatrix(JKJt, tmp, m->materialsStiffnesses[ei], m->strainDisplacements[ei], Rot);

                // find index of node 1
                for (n1=0; n1<4; n1++)
                {
                    noeud1 = e[n1];

                    for(i=0; i<3; i++)
                    {
                        ROW = offset+3*noeud1+i;
                        row = 3*n1+i;
                        // find index of node 2
                        for (n2=0; n2<4; n2++)
                        {
                            noeud2 = e[n2];

                            for (j=0; j<3; j++)
                            {
                                COLUMN = offset+3*noeud2+j;
                                column = 3*n2+j;
                                mat->add(ROW, COLUMN, - tmp[row][column]*k);
                            }
                        }
                    }
                }
            }
        }
}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addSubKToMatrix(Main* m, sofa::defaulttype::BaseMatrix *mat, const helper::vector<unsigned> & subMatrixIndex, SReal k, unsigned int &offset)
{
    Data& data = m->data;

    helper::vector<unsigned> itTetraBuild;
    const VecElement& elems = *m->_indexedElements;

    for(unsigned e = 0;e< subMatrixIndex.size();e++) {
        // search all the tetra connected to the point in subMatrixIndex
        for(unsigned IT = 0; IT < elems.size(); ++IT) {
            const Element& elm = elems[IT];

            if (elm[0] == subMatrixIndex[e] || elm[1] == subMatrixIndex[e] || elm[2] == subMatrixIndex[e] || elm[3] == subMatrixIndex[e]) {

                /// try to add the tetra in the set of point subMatrixIndex (add it only once)
                unsigned i=0;
                for (;i<itTetraBuild.size();i++) {
                    if (itTetraBuild[i] == IT) break;
                }
                if (i == itTetraBuild.size()) itTetraBuild.push_back(IT);
            }
        }
    }

    if (sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> > * crsmat = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> > * >(mat))
    {
        const VecElement& elems = *m->_indexedElements;

        helper::ReadAccessor< gpu::cuda::CudaVector<GPUElementState> > state = data.state;

        // Build Matrix Block for this ForceField
        int i,j,n1, n2;
        int offd3 = offset/3;

        typename Main::Transformation Rot;
        typename Main::StiffnessMatrix JKJt,tmp;

        Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
        Rot[0][1]=Rot[0][2]=0;
        Rot[1][0]=Rot[1][2]=0;
        Rot[2][0]=Rot[2][1]=0;

        for (unsigned eit=0; eit<itTetraBuild.size(); ++eit)
        {
            unsigned ei = itTetraBuild[eit];
            const Element& e = elems[ei];

            int blockIdx = ei / BSIZE;
            int threadIdx = ei % BSIZE;

            for(i=0; i<3; i++)
                for (j=0; j<3; j++)
                    Rot[j][i] = state[blockIdx].Rt[i][j][threadIdx];

            m->computeStiffnessMatrix(JKJt, tmp, m->materialsStiffnesses[ei], m->strainDisplacements[ei], Rot);
            defaulttype::Mat<3,3,double> tmpBlock[4][4];

            // find index of node 1
            for (n1=0; n1<4; n1++)
            {
                for(i=0; i<3; i++)
                {
                    for (n2=0; n2<4; n2++)
                    {
                        for (j=0; j<3; j++)
                        {
                            tmpBlock[n1][n2][i][j] = - tmp[n1*3+i][n2*3+j]*k;
                        }
                    }
                }
            }

            *crsmat->wbloc(offd3 + e[0], offd3 + e[0],true) += tmpBlock[0][0];
            *crsmat->wbloc(offd3 + e[0], offd3 + e[1],true) += tmpBlock[0][1];
            *crsmat->wbloc(offd3 + e[0], offd3 + e[2],true) += tmpBlock[0][2];
            *crsmat->wbloc(offd3 + e[0], offd3 + e[3],true) += tmpBlock[0][3];

            *crsmat->wbloc(offd3 + e[1], offd3 + e[0],true) += tmpBlock[1][0];
            *crsmat->wbloc(offd3 + e[1], offd3 + e[1],true) += tmpBlock[1][1];
            *crsmat->wbloc(offd3 + e[1], offd3 + e[2],true) += tmpBlock[1][2];
            *crsmat->wbloc(offd3 + e[1], offd3 + e[3],true) += tmpBlock[1][3];

            *crsmat->wbloc(offd3 + e[2], offd3 + e[0],true) += tmpBlock[2][0];
            *crsmat->wbloc(offd3 + e[2], offd3 + e[1],true) += tmpBlock[2][1];
            *crsmat->wbloc(offd3 + e[2], offd3 + e[2],true) += tmpBlock[2][2];
            *crsmat->wbloc(offd3 + e[2], offd3 + e[3],true) += tmpBlock[2][3];

            *crsmat->wbloc(offd3 + e[3], offd3 + e[0],true) += tmpBlock[3][0];
            *crsmat->wbloc(offd3 + e[3], offd3 + e[1],true) += tmpBlock[3][1];
            *crsmat->wbloc(offd3 + e[3], offd3 + e[2],true) += tmpBlock[3][2];
            *crsmat->wbloc(offd3 + e[3], offd3 + e[3],true) += tmpBlock[3][3];
        }
    }
    else if (sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> > * crsmat = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> > * >(mat))
    {
        const VecElement& elems = *m->_indexedElements;

        helper::ReadAccessor< gpu::cuda::CudaVector<GPUElementState> > state = data.state;

        // Build Matrix Block for this ForceField
        int i,j,n1, n2;
        int offd3 = offset/3;

        typename Main::Transformation Rot;
        typename Main::StiffnessMatrix JKJt,tmp;

        Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
        Rot[0][1]=Rot[0][2]=0;
        Rot[1][0]=Rot[1][2]=0;
        Rot[2][0]=Rot[2][1]=0;

        for (unsigned eit=0; eit<itTetraBuild.size(); ++eit)
        {
            unsigned ei = itTetraBuild[eit];
            const Element& e = elems[ei];

            int blockIdx = ei / BSIZE;
            int threadIdx = ei % BSIZE;

            for(i=0; i<3; i++)
                for (j=0; j<3; j++)
                    Rot[j][i] = state[blockIdx].Rt[i][j][threadIdx];

            m->computeStiffnessMatrix(JKJt, tmp, m->materialsStiffnesses[ei], m->strainDisplacements[ei], Rot);
            defaulttype::Mat<3,3,double> tmpBlock[4][4];

            // find index of node 1
            for (n1=0; n1<4; n1++)
            {
                for(i=0; i<3; i++)
                {
                    for (n2=0; n2<4; n2++)
                    {
                        for (j=0; j<3; j++)
                        {
                            tmpBlock[n1][n2][i][j] = - tmp[n1*3+i][n2*3+j]*k;
                        }
                    }
                }
            }

            *crsmat->wbloc(offd3 + e[0], offd3 + e[0],true) += tmpBlock[0][0];
            *crsmat->wbloc(offd3 + e[0], offd3 + e[1],true) += tmpBlock[0][1];
            *crsmat->wbloc(offd3 + e[0], offd3 + e[2],true) += tmpBlock[0][2];
            *crsmat->wbloc(offd3 + e[0], offd3 + e[3],true) += tmpBlock[0][3];

            *crsmat->wbloc(offd3 + e[1], offd3 + e[0],true) += tmpBlock[1][0];
            *crsmat->wbloc(offd3 + e[1], offd3 + e[1],true) += tmpBlock[1][1];
            *crsmat->wbloc(offd3 + e[1], offd3 + e[2],true) += tmpBlock[1][2];
            *crsmat->wbloc(offd3 + e[1], offd3 + e[3],true) += tmpBlock[1][3];

            *crsmat->wbloc(offd3 + e[2], offd3 + e[0],true) += tmpBlock[2][0];
            *crsmat->wbloc(offd3 + e[2], offd3 + e[1],true) += tmpBlock[2][1];
            *crsmat->wbloc(offd3 + e[2], offd3 + e[2],true) += tmpBlock[2][2];
            *crsmat->wbloc(offd3 + e[2], offd3 + e[3],true) += tmpBlock[2][3];

            *crsmat->wbloc(offd3 + e[3], offd3 + e[0],true) += tmpBlock[3][0];
            *crsmat->wbloc(offd3 + e[3], offd3 + e[1],true) += tmpBlock[3][1];
            *crsmat->wbloc(offd3 + e[3], offd3 + e[2],true) += tmpBlock[3][2];
            *crsmat->wbloc(offd3 + e[3], offd3 + e[3],true) += tmpBlock[3][3];
        }
    }
    else
    {
        const VecElement& elems = *m->_indexedElements;

        helper::ReadAccessor< gpu::cuda::CudaVector<GPUElementState> > state = data.state;

        // Build Matrix Block for this ForceField
        int i,j,n1, n2, row, column, ROW, COLUMN;

        typename Main::Transformation Rot;
        typename Main::StiffnessMatrix JKJt,tmp;

        Index noeud1, noeud2;

        Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
        Rot[0][1]=Rot[0][2]=0;
        Rot[1][0]=Rot[1][2]=0;
        Rot[2][0]=Rot[2][1]=0;

        for (unsigned eit=0; eit<itTetraBuild.size(); ++eit)
        {
            unsigned ei = itTetraBuild[eit];
            const Element& e = elems[ei];

            int blockIdx = ei / BSIZE;
            int threadIdx = ei % BSIZE;

            for(i=0; i<3; i++)
                for (j=0; j<3; j++)
                    Rot[j][i] = state[blockIdx].Rt[i][j][threadIdx];

            m->computeStiffnessMatrix(JKJt, tmp, m->materialsStiffnesses[ei], m->strainDisplacements[ei], Rot);

            // find index of node 1
            for (n1=0; n1<4; n1++)
            {
                noeud1 = e[n1];

                for(i=0; i<3; i++)
                {
                    ROW = offset+3*noeud1+i;
                    row = 3*n1+i;
                    // find index of node 2
                    for (n2=0; n2<4; n2++)
                    {
                        noeud2 = e[n2];

                        for (j=0; j<3; j++)
                        {
                            COLUMN = offset+3*noeud2+j;
                            column = 3*n2+j;
                            mat->add(ROW, COLUMN, - tmp[row][column]*k);
                        }
                    }
                }
            }
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::getRotations(Main* m, VecReal& rotations)
{
    Data& data = m->data;
    if (data.initState.empty())
    {
        data.initState.resize((data.nbElement+BSIZE-1)/BSIZE);
        data.rotationIdx.resize(data.nbVertex);
        for (int i=0; i<data.nbVertex; ++i)
        {
            data.rotationIdx[i] = m->_rotationIdx[i];
            //m->sout << "RotationIdx["<<i<<"] = " << data.rotationIdx[i]<<m->sendl;
        }
        for (int i=0; i<data.nbElement; ++i)
        {
            defaulttype::Mat<3,3,TReal> initR, curR;
            for (int l=0; l<3; ++l)
                for (int c=0; c<3; ++c)
                {
                    initR[l][c] = m->_initialRotations[i][c][l];
                    data.initState[i/BSIZE].Rt[l][c][i%BSIZE] = m->_initialRotations[i][c][l];
                    curR[l][c] = data.state[i/BSIZE].Rt[l][c][i%BSIZE];
                }
            //m->sout << "rotation element "<<i<<": init = " << initR << ", cur = " << curR <<m->sendl;
        }
    }
    if ((int)rotations.size() < data.nbVertex*9)
        rotations.resize(data.nbVertex*9);

    Kernels::getRotations(data.size(),
            data.nbVertex,
            data.initState.deviceRead(),
            data.state.deviceRead(),
            data.rotationIdx.deviceRead(),
            rotations.deviceWrite());
}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::getRotations(Main* m,defaulttype::BaseMatrix * rotations,int offset)
{
    Data& data = m->data;

#if 0 //defined(SOFA_DEV)
    if (CudaRotationMatrix<TReal> * diagd = dynamic_cast<CudaRotationMatrix<TReal> * >(rotations))
    {
        data.getRotations(m,diagd->getVector());
    }
    else
#endif // SOFA_DEV
    {
        data.vecTmpRotation.resize(data.nbVertex*9);
        data.getRotations(m,data.vecTmpRotation);


#if 0 //defined(SOFA_DEV)
        if (CudaRotationMatrix<float> * diagd = dynamic_cast<CudaRotationMatrix<float> * >(rotations))   //if the test with real didn pass that mean that rotation are different than real so we test both float and double
        {
            for (unsigned i=0; i<data.vecTmpRotation.size(); i++) diagd->getVector()[i] = data.vecTmpRotation[i];
#ifdef SOFA_GPU_CUDA_DOUBLE
        }
        else if (CudaRotationMatrix<double> * diagd = dynamic_cast<CudaRotationMatrix<double> * >(rotations))
        {
            for (unsigned i=0; i<data.vecTmpRotation.size(); i++) diagd->getVector()[i] = data.vecTmpRotation[i];
#endif
        }
        else if (component::linearsolver::RotationMatrix<float> * diagd = dynamic_cast<component::linearsolver::RotationMatrix<float> * >(rotations))
        {
            for (unsigned i=0; i<data.vecTmpRotation.size(); i++) diagd->getVector()[i] = data.vecTmpRotation[i];
#ifdef SOFA_GPU_CUDA_DOUBLE
        }
        else if (component::linearsolver::RotationMatrix<double> * diagd = dynamic_cast<component::linearsolver::RotationMatrix<double> * >(rotations))
        {
            for (unsigned i=0; i<data.vecTmpRotation.size(); i++) diagd->getVector()[i] = data.vecTmpRotation[i];
#endif
        }
        else
#endif // SOFA_DEV
        {
            for (int i=0; i<data.nbVertex; i++)
            {
                int i9 = i*9;
                int e = offset+i*3;
                rotations->set(e+0,e+0,data.vecTmpRotation[i9+0]);
                rotations->set(e+0,e+1,data.vecTmpRotation[i9+1]);
                rotations->set(e+0,e+2,data.vecTmpRotation[i9+2]);

                rotations->set(e+1,e+0,data.vecTmpRotation[i9+3]);
                rotations->set(e+1,e+1,data.vecTmpRotation[i9+4]);
                rotations->set(e+1,e+2,data.vecTmpRotation[i9+5]);

                rotations->set(e+2,e+0,data.vecTmpRotation[i9+6]);
                rotations->set(e+2,e+1,data.vecTmpRotation[i9+7]);
                rotations->set(e+2,e+2,data.vecTmpRotation[i9+8]);
            }
        }
    }
}

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaTetrahedronFEMForceField_ImplMethods(T) \
    template<> inline void TetrahedronFEMForceField< T >::reinit() \
    { data.reinit(this); } \
    template<> inline void TetrahedronFEMForceField< T >::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) \
    { \
		VecDeriv& f = *d_f.beginEdit(); \
		const VecCoord& x = d_x.getValue(); \
		const VecDeriv& v = d_v.getValue(); \
		data.addForce(this, f, x, v); \
		d_f.endEdit(); \
	} \
    template<> inline void TetrahedronFEMForceField< T >::getRotations(VecReal & rotations) \
    { data.getRotations(this, rotations); } \
    template<> inline void TetrahedronFEMForceField< T >::getRotations(defaulttype::BaseMatrix * rotations,int offset) \
    { data.getRotations(this, rotations,offset); } \
    template<> inline void TetrahedronFEMForceField< T >::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) \
    { \
		VecDeriv& df = *d_df.beginEdit(); \
		const VecDeriv& dx = d_dx.getValue(); \
		data.addDForce(this, df, dx, mparams->kFactor(), mparams->bFactor()); \
		d_df.endEdit(); \
    } \
    template<> inline void TetrahedronFEMForceField< T >::addKToMatrix(sofa::defaulttype::BaseMatrix* mat, SReal kFactor, unsigned int& offset) \
    { data.addKToMatrix(this, mat, kFactor, offset); } \
    template<> inline void TetrahedronFEMForceField< T >::addSubKToMatrix(sofa::defaulttype::BaseMatrix* mat, const helper::vector<unsigned> & subMatrixIndex, SReal kFactor, unsigned int& offset) \
    { data.addSubKToMatrix(this, mat, subMatrixIndex, kFactor, offset); }


CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3fTypes)
CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3f1Types)

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3dTypes);
CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaTetrahedronFEMForceField_ImplMethods

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
