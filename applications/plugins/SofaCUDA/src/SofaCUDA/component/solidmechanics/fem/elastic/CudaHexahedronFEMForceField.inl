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
#include <SofaCUDA/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceField.inl>
#include <sofa/gpu/cuda/mycuda.h>

namespace sofa::gpu::cuda
{

extern "C"
{

void HexahedronFEMForceFieldCuda3f_addForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* rotation, const void * kmatrix, void* eforce, const void* velems, void* f, const void* x, const void* v);
void HexahedronFEMForceFieldCuda3f_addDForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* rotation,const void* kmatrix,  void* eforce, const void* velems, void* df, const void* dx,double kfactor);
void HexaahedronFEMForceFieldCuda3f_getRotations(int gatherpt,int gatherbs,int nbVertex, unsigned int nbElemPerVertex, const void * velems, const void * erotation, const void * irotation,void * nrotation);

void HexahedronFEMForceFieldCuda3f1_addForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* rotation, const void* kmatrix,  void* eforce, const void* velems, void* f, const void* x, const void* v);
void HexahedronFEMForceFieldCuda3f1_addDForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* rotation, const void * kmatrix, void* eforce, const void* velems, void* df, const void* dx,double kfactor);
void HexaahedronFEMForceFieldCuda3f1_getRotations(int gatherpt,int gatherbs,int nbVertex, unsigned int nbElemPerVertex, const void * velems, const void * erotation, const void * irotation,void * nrotation);

} // extern "C"

template<>
class CudaKernelsHexahedronFEMForceField<CudaVec3fTypes>
{
public:
    static void addForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems,  void* rotation, const void* kmatrix, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   HexahedronFEMForceFieldCuda3f_addForce(gatherpt,gatherbs,nbElem, nbVertex, nbElemPerVertex, elems, rotation, kmatrix, eforce, velems, f, x, v); }
    static void addDForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* rotation, const void* kmatrix, void* eforce, const void* velems, void* df, const void* dx,double kfactor)
    {   HexahedronFEMForceFieldCuda3f_addDForce(gatherpt,gatherbs,nbElem, nbVertex, nbElemPerVertex, elems, rotation, kmatrix, eforce, velems, df, dx,kfactor); }
    static void getRotations(int gatherpt,int gatherbs,unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, const void* erotation, const void * irotation, void * nrotation)
    {   HexaahedronFEMForceFieldCuda3f_getRotations(gatherpt,gatherbs,nbVertex, nbElemPerVertex, velems, erotation, irotation, nrotation); }
};

template<>
class CudaKernelsHexahedronFEMForceField<CudaVec3f1Types>
{
public:
    static void addForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* rotation, const void* kmatrix, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   HexahedronFEMForceFieldCuda3f1_addForce(gatherpt,gatherbs,nbElem, nbVertex, nbElemPerVertex, elems, rotation,kmatrix, eforce, velems, f, x, v); }
    static void addDForce(int gatherpt,int gatherbs,unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* rotation, const void* kmatrix, void* eforce, const void* velems, void* df, const void* dx,double kfactor)
    {   HexahedronFEMForceFieldCuda3f1_addDForce(gatherpt,gatherbs,nbElem, nbVertex, nbElemPerVertex, elems, rotation,kmatrix, eforce, velems, df, dx,kfactor); }
    static void getRotations(int gatherpt,int gatherbs,unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, const void* erotation, const void * irotation, void * nrotation)
    {   HexaahedronFEMForceFieldCuda3f1_getRotations(gatherpt,gatherbs,nbVertex, nbElemPerVertex, velems, erotation, irotation, nrotation); }
};

} // namespace sofa::cuda::gpu


namespace sofa::component::solidmechanics::fem::elastic
{

using namespace gpu::cuda;
using namespace core::behavior;

template<class TCoord, class TDeriv, class TReal>
void HexahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::reinit(Main* m)
{
  Data& data = *m->data;
  m->setMethod(m->LARGE);

  const VecCoord& p = m->mstate->read(sofa::core::ConstVecCoordId::restPosition())->getValue();
  m->_initialPoints.setValue(p);

  m->_materialsStiffnesses.resize( m->getIndexedElements()->size() );
  m->_elementStiffnesses.beginEdit()->resize( m->getIndexedElements()->size() );

  m->_rotatedInitialElements.resize(m->getIndexedElements()->size());
  m->_rotations.resize( m->getIndexedElements()->size() );
  m->_initialrotations.resize( m->getIndexedElements()->size() );

	unsigned int i = 0;
	typename VecElement::const_iterator it;
        for(it = m->getIndexedElements()->begin() ; it != m->getIndexedElements()->end() ; ++it, ++i)
	{
		m->computeMaterialStiffness(i);
		m->initLarge(i, *it);
	}

  const VecElement& elems = *m->getIndexedElements();
  
  std::vector<int> activeElems;
  for (unsigned int i=0;i<elems.size();i++)
  {
		activeElems.push_back(i);
   }

  std::map<int,int> nelems;
  for (unsigned int i=0;i<activeElems.size();i++)
  {
      int ei = activeElems[i];
      const Element& e = elems[ei];
      for (unsigned int j=0;j<e.size();j++)
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


  data.nbElementPerVertex = nmax;
  std::istringstream ptchar(m->_gatherPt.getValue().getSelectedItem());
  std::istringstream bschar(m->_gatherBsize.getValue().getSelectedItem());
  ptchar >> data.GATHER_PT;
  bschar >> data.GATHER_BSIZE;

  int nbElemPerThread = (data.nbElementPerVertex+data.GATHER_PT-1)/data.GATHER_PT;
  int nbBpt = (data.nbVertex*data.GATHER_PT + data.GATHER_BSIZE-1)/data.GATHER_BSIZE;
  data.velems.resize(nbBpt*nbElemPerThread*data.GATHER_BSIZE);

  nelems.clear();
  for (unsigned int i=0;i<activeElems.size();i++)
  {
      int ei = activeElems[i];
      const Element& e = elems[ei];

      data.setE(ei, e, &(m->_rotatedInitialElements[ei]));

      data.setS(ei, m->_rotations[ei], m->_elementStiffnesses.getValue()[i]);

      for (unsigned j = 0; j < e.size(); ++j)
      {
          int p = e[j] - data.vertex0;
          int num = nelems[p]++;

          if (data.GATHER_PT > 1)
          {
              const int block  = (p*data.GATHER_PT) / data.GATHER_BSIZE;
              const int thread = (p*data.GATHER_PT+(num%data.GATHER_PT)) % data.GATHER_BSIZE;
              num = num/data.GATHER_PT;
              data.velems[ block * (nbElemPerThread * data.GATHER_BSIZE) + num * data.GATHER_BSIZE + thread ] = 1 + i * e.size() + j;
          }
          else
          {
              const int block  = p / data.GATHER_BSIZE;
              const int thread = p % data.GATHER_BSIZE;
              data.velems[ block * (data.nbElementPerVertex * data.GATHER_BSIZE) + num * data.GATHER_BSIZE + thread ] = 1 + i * e.size() + j;
          }
      }
  }

  m->_elementStiffnesses.endEdit();

}

template<class TCoord, class TDeriv, class TReal>
void HexahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addForce(Main* m, VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
  Data& data = *m->data;

    f.resize(x.size());
    Kernels::addForce(
        data.GATHER_PT,
        data.GATHER_BSIZE,
        data.size(),
        data.nbVertex,
        data.nbElementPerVertex,
        data.elems.deviceRead(),
        data.erotation.deviceWrite(),
        data.ekmatrix.deviceRead(),
        data.eforce.deviceWrite(),
        data.velems.deviceRead(),
        (      Deriv*)f.deviceWrite() + data.vertex0,
        (const Coord*)x.deviceRead()  + data.vertex0,
        (const Deriv*)v.deviceRead()  + data.vertex0);
} // addForce

template<class TCoord, class TDeriv, class TReal>
void HexahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addDForce (Main* m, VecDeriv& df, const VecDeriv& dx, double kFactor/*, double bFactor*/)
{
    Data& data = *m->data;
    df.resize(dx.size());

    Kernels::addDForce(        
        data.GATHER_PT,
        data.GATHER_BSIZE,
        data.size(),
        data.nbVertex,
        data.nbElementPerVertex,
        data.elems.deviceRead(),
        data.erotation.deviceRead(),
        data.ekmatrix.deviceRead(),
        data.eforce.deviceWrite(),
        data.velems.deviceRead(),
        (      Deriv*)df.deviceWrite() + data.vertex0,
        (const Deriv*)dx.deviceRead()  + data.vertex0,
         kFactor);

} // addDForce


template<>
void HexahedronFEMForceField< gpu::cuda::CudaVec3fTypes >::reinit()	{ data->reinit(this);}
template<>
void HexahedronFEMForceField< gpu::cuda::CudaVec3fTypes >::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{ 
	VecDeriv& f = *d_f.beginEdit();
	const VecCoord& x = d_x.getValue();
	const VecDeriv& v = d_v.getValue();

	data->addForce(this, f, x, v); 

	d_f.endEdit();
}
template<>
void HexahedronFEMForceField< gpu::cuda::CudaVec3fTypes >::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{ 
	VecDeriv& df = *d_df.beginEdit();
	const VecDeriv& dx = d_dx.getValue();
    const double kFactor = mparams->kFactor();

    data->addDForce(this, df, dx, kFactor/*, bFactor*/);

	d_df.endEdit();
}

template<>
const HexahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::Transformation& HexahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::getElementRotation(const unsigned elemidx) {
    return data->erotation[elemidx].Rt;
}

template<>
void HexahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::getRotations(linearalgebra::BaseMatrix * rotations,int offset)
{
    const auto nbdof = this->mstate->getSize();

    if (auto* diag = dynamic_cast<linearalgebra::RotationMatrix<float> *>(rotations))
    {
        Transformation R;
        for (unsigned int e=0; e<nbdof; ++e)
        {
            getNodeRotation(R,e);
            for(int j=0; j<3; j++)
            {
                for(int i=0; i<3; i++)
                {
                    diag->getVector()[e*9 + j*3 + i] = (float)R[j][i];
                }
            }
        }
    } 
    else if (auto* diag = dynamic_cast<linearalgebra::RotationMatrix<double> *>(rotations))
    {
        Transformation R;
        for (unsigned int e=0; e<nbdof; ++e)
        {
            getNodeRotation(R,e);
            for(int j=0; j<3; j++)
            {
                for(int i=0; i<3; i++)
                {
                    diag->getVector()[e*9 + j*3 + i] = R[j][i];
                }
            }
        }
    } 
    else 
    {
        for (unsigned int i=0; i<nbdof; ++i)
        {
            Transformation t;
            getNodeRotation(t,i);
            const int e = offset+i*3;
            rotations->set(e+0,e+0,t[0][0]); rotations->set(e+0,e+1,t[0][1]); rotations->set(e+0,e+2,t[0][2]);
            rotations->set(e+1,e+0,t[1][0]); rotations->set(e+1,e+1,t[1][1]); rotations->set(e+1,e+2,t[1][2]);
            rotations->set(e+2,e+0,t[2][0]); rotations->set(e+2,e+1,t[2][1]); rotations->set(e+2,e+2,t[2][2]);
        }
    }
}

} // namespace sofa::component::solidmechanics::fem::elastic
