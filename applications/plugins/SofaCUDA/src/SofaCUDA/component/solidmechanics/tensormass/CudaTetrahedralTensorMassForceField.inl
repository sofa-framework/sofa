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

#include <SofaCUDA/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.h>
#include <sofa/component/solidmechanics/tensormass/TetrahedralTensorMassForceField.inl>

namespace sofa::gpu::cuda
{
    extern "C"
    {
        void TetrahedralTensorMassForceFieldCuda3f_addForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* f, const void* x, const void* initialPoints, const void* edgeInfo );
        void TetrahedralTensorMassForceFieldCuda3f_addDForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* df, const void* dx, const void* edgeInfo, float kFactor );
#ifdef SOFA_GPU_CUDA_DOUBLE
        void TetrahedralTensorMassForceFieldCuda3d_addForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* f, const void* x, const void* initialPoints, const void* edgeInfo );
        void TetrahedralTensorMassForceFieldCuda3d_addDForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* df, const void* dx, const void* edgeInfo, double kFactor );
#endif
    }
} // namespace sofa::gpu::cuda

namespace sofa::component::solidmechanics::tensormass
{
/*
	// TODO: warning - we should have a TetrahedralTensorMassForceFieldData<DataType> because
		- TetrahedralTensorMassForceField_nbMaxEdgesPerNode
		- TetrahedralTensorMassForceField_neighbourhoodPoints
		- TetrahedralTensorMassForceField_contribEdge
	are shared between TetrahedralTensorMassForceField<CudaVecTypes> and i guess, should not

*/
using namespace gpu::cuda;

    template <>
    void TetrahedralTensorMassForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /*d_v*/)
    {
		SCOPED_TIMER("addForceTetraTensorMass");

        VecDeriv& f = *d_f.beginEdit();
        const VecCoord& x = d_x.getValue();

		const int nbEdges=m_topology->getNbEdges();
		const int nbPoints=m_topology->getNbPoints();

		const edgeRestInfoVector& edgeInf = *(edgeInfo.beginEdit());

        TetrahedralTensorMassForceField_contribEdge().resize(6*nbEdges);
        TetrahedralTensorMassForceFieldCuda3f_addForce(nbPoints, TetrahedralTensorMassForceField_nbMaxEdgesPerNode(), TetrahedralTensorMassForceField_neighbourhoodPoints().deviceRead(), TetrahedralTensorMassForceField_contribEdge().deviceWrite(), nbEdges,  f.deviceWrite(), x.deviceRead(), _initialPoints.deviceRead(), edgeInf.deviceRead());

        edgeInfo.endEdit();
        d_f.endEdit();
    }

    template <>
    void TetrahedralTensorMassForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
    {
		SCOPED_TIMER("addDForceTetraTensorMass");

        VecDeriv& df = *d_df.beginEdit();
        const VecDeriv& dx = d_dx.getValue();
		const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

		const int nbEdges=m_topology->getNbEdges();
		const int nbPoints=m_topology->getNbPoints();
		const edgeRestInfoVector& edgeInf = *(edgeInfo.beginEdit());

        TetrahedralTensorMassForceField_contribEdge().resize(6*nbEdges);
        TetrahedralTensorMassForceFieldCuda3f_addDForce(nbPoints, TetrahedralTensorMassForceField_nbMaxEdgesPerNode(), TetrahedralTensorMassForceField_neighbourhoodPoints().deviceRead(), TetrahedralTensorMassForceField_contribEdge().deviceWrite(), nbEdges,  df.deviceWrite(), dx.deviceRead(), edgeInf.deviceRead(), (float)kFactor);

        edgeInfo.endEdit();
        d_df.endEdit();
    }


    template<>
    void TetrahedralTensorMassForceField<CudaVec3fTypes>::initNeighbourhoodPoints()
    {
        msg_info() <<"GPU-GEMS activated";

        /// Initialize the number max of edges per node
        TetrahedralTensorMassForceField_nbMaxEdgesPerNode() = 0;

        /// Compute it
        for(Size i=0;i<m_topology->getNbPoints();++i)
        {
            if((int)m_topology->getEdgesAroundVertex(i).size()>TetrahedralTensorMassForceField_nbMaxEdgesPerNode())
                TetrahedralTensorMassForceField_nbMaxEdgesPerNode() = m_topology->getEdgesAroundVertex(i).size();
        }

        /// Initialize the vector neighbourhoodPoints
        TetrahedralTensorMassForceField_neighbourhoodPoints().resize((m_topology->getNbPoints())*TetrahedralTensorMassForceField_nbMaxEdgesPerNode());

        unsigned int edgeID;

        for (Size i=0;i<m_topology->getNbPoints();++i)
        {
            for(int j=0;j<TetrahedralTensorMassForceField_nbMaxEdgesPerNode();++j)
            {
                if(j>(int)m_topology->getEdgesAroundVertex(i).size()-1)
                    TetrahedralTensorMassForceField_neighbourhoodPoints()[i*TetrahedralTensorMassForceField_nbMaxEdgesPerNode()+j] = -1;
                else
                {
                    edgeID = m_topology->getEdgesAroundVertex(i)[j];
                    if(i == Size(m_topology->getEdge(edgeID)[0]))
                        TetrahedralTensorMassForceField_neighbourhoodPoints()[i*TetrahedralTensorMassForceField_nbMaxEdgesPerNode()+j] = 2*edgeID;   //v0
                    else
                        TetrahedralTensorMassForceField_neighbourhoodPoints()[i*TetrahedralTensorMassForceField_nbMaxEdgesPerNode()+j] = 2*edgeID+1; //v1
                }
            }
        }


    }


#ifdef SOFA_GPU_CUDA_DOUBLE
    template <>
    void TetrahedralTensorMassForceField<gpu::cuda::CudaVec3dTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /*d_v*/)
    {
        VecDeriv& f = *d_f.beginEdit();
        const VecCoord& x = d_x.getValue();

        int nbEdges=m_topology->getNbEdges();
        int nbPoints=m_topology->getNbPoints();

        edgeRestInfoVector& edgeInf = *(edgeInfo.beginEdit());

        TetrahedralTensorMassForceField_contribEdge().resize(6*nbEdges);
        TetrahedralTensorMassForceFieldCuda3d_addForce(nbPoints, TetrahedralTensorMassForceField_nbMaxEdgesPerNode(), TetrahedralTensorMassForceField_neighbourhoodPoints().deviceRead(), TetrahedralTensorMassForceField_contribEdge().deviceWrite(), nbEdges,  f.deviceWrite(), x.deviceRead(), _initialPoints.deviceRead(), edgeInf.deviceRead());

        edgeInfo.endEdit();
        d_f.endEdit();


    }

    template <>
    void TetrahedralTensorMassForceField<gpu::cuda::CudaVec3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
    {
        VecDeriv& df = *d_df.beginEdit();
        const VecDeriv& dx = d_dx.getValue();
        Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

        int nbEdges=m_topology->getNbEdges();
        int nbPoints=m_topology->getNbPoints();
        edgeRestInfoVector& edgeInf = *(edgeInfo.beginEdit());

        TetrahedralTensorMassForceField_contribEdge().resize(6*nbEdges);
        TetrahedralTensorMassForceFieldCuda3d_addDForce(nbPoints, TetrahedralTensorMassForceField_nbMaxEdgesPerNode(), TetrahedralTensorMassForceField_neighbourhoodPoints().deviceRead(), TetrahedralTensorMassForceField_contribEdge().deviceWrite(), nbEdges,  df.deviceWrite(), dx.deviceRead(), edgeInf.deviceRead(), kFactor);

        edgeInfo.endEdit();
        d_df.endEdit();
    }

	template<>
	void TetrahedralTensorMassForceField<CudaVec3dTypes>::initNeighbourhoodPoints()
	{
        msg_info() <<"GPU-GEMS activated";

		/// Initialize the number max of edges per node
		TetrahedralTensorMassForceField_nbMaxEdgesPerNode() = 0;

		/// Compute it
        for(Size i=0;i<m_topology->getNbPoints();++i)
		{
			if((int)m_topology->getEdgesAroundVertex(i).size()>TetrahedralTensorMassForceField_nbMaxEdgesPerNode())
				TetrahedralTensorMassForceField_nbMaxEdgesPerNode() = m_topology->getEdgesAroundVertex(i).size();
		}

		/// Initialize the vector neighbourhoodPoints
		TetrahedralTensorMassForceField_neighbourhoodPoints().resize((m_topology->getNbPoints())*TetrahedralTensorMassForceField_nbMaxEdgesPerNode());

		unsigned int edgeID;

        for (Size i=0;i<m_topology->getNbPoints();++i)
		{
			for(int j=0;j<TetrahedralTensorMassForceField_nbMaxEdgesPerNode();++j)
			{
				if(j>(int)m_topology->getEdgesAroundVertex(i).size()-1)
					TetrahedralTensorMassForceField_neighbourhoodPoints()[i*TetrahedralTensorMassForceField_nbMaxEdgesPerNode()+j] = -1;
				else
				{
					edgeID = m_topology->getEdgesAroundVertex(i)[j];
                    if((unsigned) i == m_topology->getEdge(edgeID)[0])
						TetrahedralTensorMassForceField_neighbourhoodPoints()[i*TetrahedralTensorMassForceField_nbMaxEdgesPerNode()+j] = 2*edgeID;   //v0
					else
						TetrahedralTensorMassForceField_neighbourhoodPoints()[i*TetrahedralTensorMassForceField_nbMaxEdgesPerNode()+j] = 2*edgeID+1; //v1
				}
			}
		}


	}

#endif


} // namespace sofa::component::solidmechanics::tensormass
