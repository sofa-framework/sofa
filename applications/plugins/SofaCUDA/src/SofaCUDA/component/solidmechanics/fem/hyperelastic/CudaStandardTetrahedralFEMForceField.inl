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

#include <SofaCUDA/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#define EDGEDEBUG 100

namespace sofa::gpu::cuda
{
extern "C"
{
void StandardTetrahedralFEMForceFieldCuda3f_addForce(int nbTetra, int nbPoints, int nbMaxTetraPerNode, const void* neighbourhoodPoints, void* contribTetra, void* tetraInfo, void* f, const void* x, bool anisotropy, const void* anisoDirection, float paramArray0, float paramArray1);
void StandardTetrahedralFEMForceFieldCuda3f_addDForce(int nbTetra, int nbEdges, int nbMaxTetraPerEdge, void* tetraInfo, void* edgeInfo, void* contribDfDx, const void* neighbourhoodEdges, float param0, float param1);
#ifdef SOFA_GPU_CUDA_DOUBLE
void StandardTetrahedralFEMForceFieldCuda3d_addForce(int nbTetra, int nbPoints, int nbMaxTetraPerNode, const void* neighbourhoodPoints, void* contribTetra, void* tetraInfo, void* f, const void* x, bool anisotropy, const void* anisoDirection, double paramArray0, double paramArray1);
void StandardTetrahedralFEMForceFieldCuda3d_addDForce(int nbTetra, int nbEdges, int nbMaxTetraPerEdge, void* tetraInfo, void* edgeInfo, void* contribDfDx, const void* neighbourhoodEdges, double param0, double param1);
#endif
}

} // namespace sofa::gpu::cuda

namespace sofa::component::solidmechanics::fem::hyperelastic
{

using namespace gpu::cuda;


template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /*d_v*/)
{
    SCOPED_TIMER("addForceStandardTetraFEM");

    VecDeriv& f = *d_f.beginEdit();
	const VecCoord& x = d_x.getValue();

    const unsigned int nbTetrahedra=m_topology->getNbTetrahedra();
    const unsigned int nbPoints=m_topology->getNbPoints();

	tetrahedronRestInfoVector& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    Coord anisoDirection;
    bool anisotropy = false;
    if(globalParameters.anisotropyDirection.size()>0)
    {
        anisotropy = true;
        anisoDirection = globalParameters.anisotropyDirection[0];
    }
    VecCoord anisoVec;
    anisoVec.push_back(anisoDirection);

    const Real paramArray0 = globalParameters.parameterArray[0];
    const Real paramArray1 = globalParameters.parameterArray[1];

    StandardTetrahedralFEMForceField_contribTetra().resize(12*nbTetrahedra);
    StandardTetrahedralFEMForceFieldCuda3f_addForce(nbTetrahedra, nbPoints, StandardTetrahedralFEMForceField_nbMaxTetraPerNode(), StandardTetrahedralFEMForceField_neighbourhoodPoints().deviceRead(), StandardTetrahedralFEMForceField_contribTetra().deviceWrite(), tetrahedronInf.deviceWrite(), f.deviceWrite(), x.deviceRead(), anisotropy, anisoVec.deviceRead(), paramArray0, paramArray1);

	/// indicates that the next call to addDForce will need to update the stiffness matrix
	updateMatrix=true;
	tetrahedronInfo.endEdit();

	d_f.endEdit();
}

template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    SCOPED_TIMER("addDForceStandardTetraFEM");

    VecDeriv& df = *d_df.beginEdit();
	const VecDeriv& dx = d_dx.getValue();
    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const unsigned int nbEdges=m_topology->getNbEdges();
    const type::vector< core::topology::BaseMeshTopology::Edge> &edgeArray=m_topology->getEdges() ;

    const unsigned int nbTetrahedra=m_topology->getNbTetrahedra();

    edgeInformationVector& edgeInf = *(edgeInfo.beginEdit());
	tetrahedronRestInfoVector& tetrahedronInf = *(tetrahedronInfo.beginEdit());

	EdgeInformation *einfo;
    const Real paramArray0 = globalParameters.parameterArray[0];
    const Real paramArray1 = globalParameters.parameterArray[1];

	/// if the  matrix needs to be updated
	if (updateMatrix) {

        for(unsigned int l=0; l<nbEdges; l++ )edgeInf[l].DfDx.clear();

        StandardTetrahedralFEMForceField_contribDfDx().resize(54*nbTetrahedra);
        StandardTetrahedralFEMForceFieldCuda3f_addDForce(nbTetrahedra, nbEdges, StandardTetrahedralFEMForceField_nbMaxTetraPerEdge(), tetrahedronInf.deviceWrite(), edgeInf.deviceWrite(), StandardTetrahedralFEMForceField_contribDfDx().deviceWrite(), StandardTetrahedralFEMForceField_neighbourhoodEdges().deviceRead(), paramArray0, paramArray1);

		updateMatrix=false;
	}// end of if

	/// performs matrix vector computation
	unsigned int v0,v1;
	Deriv deltax;	Deriv dv0,dv1;

	for(unsigned int l=0; l<nbEdges; l++ )
	{
		einfo=&edgeInf[l];
		v0=edgeArray[l][0];
		v1=edgeArray[l][1];

		deltax= dx[v0] - dx[v1];
		dv0 = einfo->DfDx * deltax;
		// do the transpose multiply:
		dv1[0] = (Real)(deltax[0]*einfo->DfDx[0][0] + deltax[1]*einfo->DfDx[1][0] + deltax[2]*einfo->DfDx[2][0]);
		dv1[1] = (Real)(deltax[0]*einfo->DfDx[0][1] + deltax[1]*einfo->DfDx[1][1] + deltax[2]*einfo->DfDx[2][1]);
		dv1[2] = (Real)(deltax[0]*einfo->DfDx[0][2] + deltax[1]*einfo->DfDx[1][2] + deltax[2]*einfo->DfDx[2][2]);
		// add forces
		df[v0] += dv1 * kFactor;
		df[v1] -= dv0 * kFactor;
	}
    edgeInfo.endEdit();
	tetrahedronInfo.endEdit();
	d_df.beginEdit();
}

template<>
void StandardTetrahedralFEMForceField<CudaVec3fTypes>::initNeighbourhoodPoints()
{

    StandardTetrahedralFEMForceField_nbMaxTetraPerNode() = 0;

    for(Size i=0; i<m_topology->getNbPoints();++i)
    {
        if( (int)m_topology->getTetrahedraAroundVertex(i).size() > StandardTetrahedralFEMForceField_nbMaxTetraPerNode())
            StandardTetrahedralFEMForceField_nbMaxTetraPerNode() = m_topology->getTetrahedraAroundVertex(i).size();
    }

    StandardTetrahedralFEMForceField_neighbourhoodPoints().resize( (m_topology->getNbPoints())*StandardTetrahedralFEMForceField_nbMaxTetraPerNode());

    for(Size i=0; i<m_topology->getNbPoints();++i)
    {
        for(int j=0; j<StandardTetrahedralFEMForceField_nbMaxTetraPerNode(); ++j)
        {
            if( j >(int)m_topology->getTetrahedraAroundVertex(i).size()-1)
                StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = -1;
            else
            {
                unsigned int tetraID;
                tetraID = m_topology->getTetrahedraAroundVertex(i)[j];
                if( (unsigned)i == m_topology->getTetra(tetraID)[0])
                    StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = 4*tetraID;
                else if ( (unsigned)i == m_topology->getTetra(tetraID)[1])
                    StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = 4*tetraID+1;
                else if ( (unsigned)i == m_topology->getTetra(tetraID)[2])
                    StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = 4*tetraID+2;
                else
                    StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = 4*tetraID+3;
            }
        }
    }
}

template<>
void StandardTetrahedralFEMForceField<CudaVec3fTypes>::initNeighbourhoodEdges()
{
    StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() = 0;

    for(int i=0; i<int(m_topology->getNbEdges()); ++i)
    {
        if( (int)m_topology->getTetrahedraAroundEdge(i).size() > StandardTetrahedralFEMForceField_nbMaxTetraPerEdge())
            StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() = m_topology->getTetrahedraAroundEdge(i).size();
    }

    StandardTetrahedralFEMForceField_neighbourhoodEdges().resize((m_topology->getNbEdges())*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge());

    for(int i=0; i<int(m_topology->getNbEdges()); ++i)
    {
        for(int j=0; j<StandardTetrahedralFEMForceField_nbMaxTetraPerEdge(); ++j)
        {
            if( j > (int)m_topology->getTetrahedraAroundEdge(i).size()-1)
                StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = -1;
            else
            {
                unsigned int tetraID;
                tetraID = m_topology->getTetrahedraAroundEdge(i)[j];
                core::topology::BaseMeshTopology::EdgesInTetrahedron te=m_topology->getEdgesInTetrahedron(tetraID);
                if( (unsigned)i == te[0])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 0;
                else if( (unsigned)i == te[1])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 1;
                else if( (unsigned)i == te[2])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 2;
                else if( (unsigned)i == te[3])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 3;
                else if( (unsigned)i == te[4])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 4;
                else if( (unsigned)i == te[5])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 5;
            }
        }
    }    
}

#ifdef SOFA_GPU_CUDA_DOUBLE
template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3dTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /*d_v*/)
{
	VecDeriv& f = *d_f.beginEdit();
	const VecCoord& x = d_x.getValue();

	unsigned int nbTetrahedra=m_topology->getNbTetrahedra();
    unsigned int nbPoints=m_topology->getNbPoints();

	tetrahedronRestInfoVector& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    VecDeriv solution;
    solution.resize(nbTetrahedra);

    Coord anisoDirection;
    bool anisotropy;
    if(globalParameters.anisotropyDirection.size()>0)
    {
        anisotropy = true;
        anisoDirection = globalParameters.anisotropyDirection[0];
    }
    VecCoord anisoVec;
    anisoVec.push_back(anisoDirection);

    Real paramArray0 = globalParameters.parameterArray[0];
    Real paramArray1 = globalParameters.parameterArray[1];

    StandardTetrahedralFEMForceField_contribTetra().resize(12*nbTetrahedra);
    StandardTetrahedralFEMForceFieldCuda3d_addForce(nbTetrahedra, nbPoints, StandardTetrahedralFEMForceField_nbMaxTetraPerNode(), StandardTetrahedralFEMForceField_neighbourhoodPoints().deviceRead(), StandardTetrahedralFEMForceField_contribTetra().deviceWrite(), tetrahedronInf.deviceWrite(), f.deviceWrite(), x.deviceRead(), anisotropy, anisoVec.deviceRead(), paramArray0, paramArray1);

	/// indicates that the next call to addDForce will need to update the stiffness matrix
	updateMatrix=true;
	tetrahedronInfo.endEdit();

	d_f.endEdit();
}

template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
	VecDeriv& df = *d_df.beginEdit();
	const VecDeriv& dx = d_dx.getValue();
	Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

	unsigned int nbEdges=m_topology->getNbEdges();
    const type::vector< core::topology::BaseMeshTopology::Edge> &edgeArray=m_topology->getEdges() ;

    unsigned int nbTetrahedra=m_topology->getNbTetrahedra();

    edgeInformationVector& edgeInf = *(edgeInfo.beginEdit());
	tetrahedronRestInfoVector& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    EdgeInformation *einfo;
    Real paramArray0 = globalParameters.parameterArray[0];
    Real paramArray1 = globalParameters.parameterArray[1];

	/// if the  matrix needs to be updated
	if (updateMatrix) {

        VecDeriv solution;
        solution.resize(nbTetrahedra*12);

        for(unsigned int l=0; l<nbEdges; l++ )edgeInf[l].DfDx.clear();

        StandardTetrahedralFEMForceField_contribDfDx().resize(54*nbTetrahedra);
        StandardTetrahedralFEMForceFieldCuda3d_addDForce(nbTetrahedra, nbEdges, StandardTetrahedralFEMForceField_nbMaxTetraPerEdge(), tetrahedronInf.deviceWrite(), edgeInf.deviceWrite(), StandardTetrahedralFEMForceField_contribDfDx().deviceWrite(), StandardTetrahedralFEMForceField_neighbourhoodEdges().deviceRead(), paramArray0, paramArray1);

		updateMatrix=false;
	}// end of if

	/// performs matrix vector computation
	unsigned int v0,v1;
	Deriv deltax;	Deriv dv0,dv1;

	for(unsigned int l=0; l<nbEdges; l++ )
	{
		einfo=&edgeInf[l];
		v0=edgeArray[l][0];
		v1=edgeArray[l][1];

		deltax= dx[v0] - dx[v1];
		dv0 = einfo->DfDx * deltax;
		// do the transpose multiply:
		dv1[0] = (Real)(deltax[0]*einfo->DfDx[0][0] + deltax[1]*einfo->DfDx[1][0] + deltax[2]*einfo->DfDx[2][0]);
		dv1[1] = (Real)(deltax[0]*einfo->DfDx[0][1] + deltax[1]*einfo->DfDx[1][1] + deltax[2]*einfo->DfDx[2][1]);
		dv1[2] = (Real)(deltax[0]*einfo->DfDx[0][2] + deltax[1]*einfo->DfDx[1][2] + deltax[2]*einfo->DfDx[2][2]);
		// add forces
		df[v0] += dv1 * kFactor;
		df[v1] -= dv0 * kFactor;
	}
    edgeInfo.endEdit();
	tetrahedronInfo.endEdit();
	d_df.beginEdit();
}

template<>
void StandardTetrahedralFEMForceField<CudaVec3dTypes>::initNeighbourhoodPoints()
{
    msg_info() << "(StandardTetrahedralFEMForceField) GPU-GEMS activated";

    StandardTetrahedralFEMForceField_nbMaxTetraPerNode() = 0;

    for(Size i=0; i<m_topology->getNbPoints();++i)
    {
        if( (int)m_topology->getTetrahedraAroundVertex(i).size() > StandardTetrahedralFEMForceField_nbMaxTetraPerNode())
            StandardTetrahedralFEMForceField_nbMaxTetraPerNode() = m_topology->getTetrahedraAroundVertex(i).size();
    }

    StandardTetrahedralFEMForceField_neighbourhoodPoints().resize( (m_topology->getNbPoints())*StandardTetrahedralFEMForceField_nbMaxTetraPerNode());

    for(Size i=0; i<m_topology->getNbPoints();++i)
    {
        for(int j=0; j<StandardTetrahedralFEMForceField_nbMaxTetraPerNode(); ++j)
        {
            if( j >(int)m_topology->getTetrahedraAroundVertex(i).size()-1)
                StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = -1;
            else
            {
                unsigned int tetraID;
                tetraID = m_topology->getTetrahedraAroundVertex(i)[j];
                if( (unsigned) i == m_topology->getTetra(tetraID)[0])
                    StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = 4*tetraID;
                else if ( (unsigned) i == m_topology->getTetra(tetraID)[1])
                    StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = 4*tetraID+1;
                else if ( (unsigned) i == m_topology->getTetra(tetraID)[2])
                    StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = 4*tetraID+2;
                else
                    StandardTetrahedralFEMForceField_neighbourhoodPoints()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerNode() + j] = 4*tetraID+3;
            }
        }
    }
}

template<>
void StandardTetrahedralFEMForceField<CudaVec3dTypes>::initNeighbourhoodEdges()
{
    StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() = 0;

    for(Size i=0; i<m_topology->getNbEdges(); ++i)
    {
        if( (int)m_topology->getTetrahedraAroundEdge(i).size() > StandardTetrahedralFEMForceField_nbMaxTetraPerEdge())
            StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() = m_topology->getTetrahedraAroundEdge(i).size();
    }

    StandardTetrahedralFEMForceField_neighbourhoodEdges().resize((m_topology->getNbEdges())*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge());

    for(Size i=0; i<m_topology->getNbEdges(); ++i)
    {
        for(int j=0; j<StandardTetrahedralFEMForceField_nbMaxTetraPerEdge(); ++j)
        {
            if( j > (int)m_topology->getTetrahedraAroundEdge(i).size()-1)
                StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = -1;
            else
            {
                unsigned int tetraID;
                tetraID = m_topology->getTetrahedraAroundEdge(i)[j];
                core::topology::BaseMeshTopology::EdgesInTetrahedron te=m_topology->getEdgesInTetrahedron(tetraID);
                if( (unsigned) i == te[0])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 0;
                else if( (unsigned) i == te[1])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 1;
                else if( (unsigned) i == te[2])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 2;
                else if( (unsigned) i == te[3])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 3;
                else if( (unsigned) i == te[4])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 4;
                else if( (unsigned) i == te[5])
                    StandardTetrahedralFEMForceField_neighbourhoodEdges()[i*StandardTetrahedralFEMForceField_nbMaxTetraPerEdge() + j] = tetraID*6 + 5;
            }
        }
    }    
}
#endif
} // namespace sofa::component::solidmechanics::fem::hyperelastic
