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

#include "CudaHexahedronTLEDForceField.h"
#include "mycuda.h"
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>

#include <fstream>
using namespace std;

namespace sofa
{

namespace gpu
{

namespace cuda
{

int CudaHexahedronTLEDForceFieldCudaClass = core::RegisterObject("GPU-side TLED hexahedron forcefield using CUDA")
        .add< CudaHexahedronTLEDForceField >()
        ;

extern "C"
{
    void CudaHexahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int isViscoelastic, unsigned int isAnisotropic, const void* x, const void* x0, void* f, int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2, float* detJarray, float* hourglassControlArray, float3* preferredDirection, float4* Di1, float4* Di2, float4* Dv1, float4* Dv2, int2* forceCoordinates, float4* F0, float4* F1, float4* F2, float4* F3, float4* F4, float4* F5, float4* F6, float4* F7);
    void InitGPU_TLED(int valence, int nbVertex, int nbElements);
    void InitGPU_Visco(float * Ai, float * Av, int Ni, int Nv);
    void InitGPU_Aniso();
    void ClearGPU_TLED(void);
    void ClearGPU_Visco(void);
    void ClearGPU_Aniso(void);
}

// --------------------------------------------------------------------------------------
// Constructor - Initialises member variables from scene file
// --------------------------------------------------------------------------------------
CudaHexahedronTLEDForceField::CudaHexahedronTLEDForceField()
    : nbVertex(0), nbElementPerVertex(0)
    , poissonRatio(initData(&poissonRatio,(Real)0.45,"poissonRatio","Poisson ratio in Hooke's law"))
    , youngModulus(initData(&youngModulus,(Real)3000.,"youngModulus","Young modulus in Hooke's law"))
    , timestep(initData(&timestep,(Real)0.001,"timestep","Simulation timestep"))
    , isViscoelastic(initData(&isViscoelastic,(unsigned int)0,"isViscoelastic","Viscoelasticity flag"))
    , isAnisotropic(initData(&isAnisotropic,(unsigned int)0,"isAnisotropic","Anisotropy flag"))
    , preferredDirection(initData(&preferredDirection, "preferredDirection","Transverse isotropy direction"))
{
}

// --------------------------------------------------------------------------------------
// Destructor - Cleans GPU memory
// --------------------------------------------------------------------------------------
CudaHexahedronTLEDForceField::~CudaHexahedronTLEDForceField()
{
    ClearGPU_TLED();

    if (m_device_nodesPerElement)
    {
        mycudaFree(m_device_nodesPerElement);
    }

    if (m_device_DhC0)
    {
        mycudaFree(m_device_DhC0);
    }
    if (m_device_DhC1)
    {
        mycudaFree(m_device_DhC1);
    }
    if (m_device_DhC2)
    {
        mycudaFree(m_device_DhC2);
    }

    if (m_device_detJ)
    {
        mycudaFree(m_device_detJ);
    }

    if (m_device_hourglassControl)
    {
        mycudaFree(m_device_hourglassControl);
    }

    if (m_device_preferredDirection)
    {
        mycudaFree(m_device_preferredDirection);
    }

    if (m_device_Di1)
    {
        mycudaFree(m_device_Di1);
    }
    if (m_device_Di2)
    {
        mycudaFree(m_device_Di2);
    }
    if (m_device_Dv1)
    {
        mycudaFree(m_device_Dv1);
    }
    if (m_device_Dv2)
    {
        mycudaFree(m_device_Dv2);
    }

    if (m_device_forceCoordinates)
    {
        mycudaFree(m_device_forceCoordinates);
    }

    if (m_device_F0)
    {
        mycudaFree(m_device_F0);
    }
    if (m_device_F1)
    {
        mycudaFree(m_device_F1);
    }
    if (m_device_F2)
    {
        mycudaFree(m_device_F2);
    }
    if (m_device_F3)
    {
        mycudaFree(m_device_F3);
    }
    if (m_device_F4)
    {
        mycudaFree(m_device_F4);
    }
    if (m_device_F5)
    {
        mycudaFree(m_device_F5);
    }
    if (m_device_F6)
    {
        mycudaFree(m_device_F6);
    }
    if (m_device_F7)
    {
        mycudaFree(m_device_F7);
    }

    if (isViscoelastic.getValue())
    {
        ClearGPU_Visco();
    }
    if (isAnisotropic.getValue())
    {
        ClearGPU_Aniso();
    }
}

void CudaHexahedronTLEDForceField::init()
{
    core::behavior::ForceField<CudaVec3fTypes>::init();
    reinit();
}

// --------------------------------------------------------------------------------------
// Initialisation and precomputations
// --------------------------------------------------------------------------------------
void CudaHexahedronTLEDForceField::reinit()
{
    // Gets the mesh
    sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();
    if (topology==NULL || topology->getNbHexahedra()==0)
    {
        msg_error() << "no elements found.\n";
        return;
    }
    VecElement inputElems = topology->getHexahedra();

    nbElems = inputElems.size();

    // Number of elements attached to each node
    std::map<int,int> nelems;
    for (int i=0; i<nbElems; i++)
    {
        const Element& e = inputElems[i];
        for (unsigned int j=0; j<e.size(); j++)
        {
            ++nelems[e[j]];
        }
    }

    // Gets the maximum of elements attached to a vertex
    nbElementPerVertex = 0;
    for (std::map<int,int>::const_iterator it = nelems.begin(); it != nelems.end(); ++it)
    {
        if (it->second > nbElementPerVertex)
        {
            nbElementPerVertex = it->second;
        }
    }

    // Number of nodes
    nbVertex = 0;
    if (!nelems.empty())
    {
        nbVertex = nelems.rbegin()->first + 1;
    }

    msg_info() << "CudaHexahedronTLEDForceField: " << nbElems << " elements, " << nbVertex << " nodes, max " << nbElementPerVertex << " elements per node";


    /**
     * Precomputations
     */
    msg_info() << "CudaHexahedronTLEDForceField: precomputations...";

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    nelems.clear();

    // Shape function natural derivatives DhDr
    float DhDr[8][3];
    const float a = 1./8;
    DhDr[0][0] = -a; DhDr[0][1] = -a; DhDr[0][2] = -a;
    DhDr[1][0] = a;  DhDr[1][1] = -a; DhDr[1][2] = -a;
    DhDr[2][0] = a;  DhDr[2][1] = a;  DhDr[2][2] = -a;
    DhDr[3][0] = -a; DhDr[3][1] = a;  DhDr[3][2] = -a;
    DhDr[4][0] = -a; DhDr[4][1] = -a; DhDr[4][2] = a;
    DhDr[5][0] = a;  DhDr[5][1] = -a; DhDr[5][2] = a;
    DhDr[6][0] = a;  DhDr[6][1] = a;  DhDr[6][2] = a;
    DhDr[7][0] = -a; DhDr[7][1] = a;  DhDr[7][2] = a;

    // Hourglass control
    sofa::type::vector<float> hourglassControl(64 * nbElems);

    // 3 data pointers for the shape function global derivatives (DhDx matrix columns for each element stored in separated arrays)
    sofa::type::vector<float4> DhC0(2 * nbElems);
    sofa::type::vector<float4> DhC1(2 * nbElems);
    sofa::type::vector<float4> DhC2(2 * nbElems);

    // Element volume (useful to compute shape function global derivatives and Hourglass control coefficients)
    float * Volume = new float[nbElems];
    // Allocates the texture data for Jacobian determinants
    sofa::type::vector<float> DetJ(nbElems);

    // Retrieves force coordinates (slice number and index) for each node
    sofa::type::vector<int2> FCrds(nbVertex * nbElementPerVertex, {-1, -1});
    int * index = new int[nbVertex];
    memset(index, 0, nbVertex*sizeof(int));


    // Stores list of nodes for each element
    sofa::type::vector<int4> nodesPerElement(2 * nbElems);

    for (int i=0; i<nbElems; i++)
    {
        Element& e = inputElems[i];

        // Compute Jacobian (J = DhDr^T * x)
        DetJ[i] = ComputeDetJ(e, x, DhDr);

        // Compute element volume
        Volume[i] = CompElVolHexa(e, x);

        // Compute shape function global derivatives DhDx
        float DhDx[8][3];
        ComputeDhDxHexa(e, x, Volume[i], DhDx);

        // Hourglass control
        float HG[8][8];
        ComputeHGParams(e, x, DhDx, Volume[i], HG);

        // Store HG values
        int m = 0;
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                hourglassControl[64*i+m] = HG[j][k];
                m++;
            }
        }

        nodesPerElement[2 * i].x = e[0];
        nodesPerElement[2 * i].y = e[1];
        nodesPerElement[2 * i].z = e[2];
        nodesPerElement[2 * i].w = e[3];

        nodesPerElement[2 * i + 1].x = e[4];
        nodesPerElement[2 * i + 1].y = e[5];
        nodesPerElement[2 * i + 1].z = e[6];
        nodesPerElement[2 * i + 1].w = e[7];

        DhC0[2 * i].x = DhDx[0][0];
        DhC0[2 * i].y = DhDx[1][0];
        DhC0[2 * i].z = DhDx[2][0];
        DhC0[2 * i].w = DhDx[3][0];
        DhC0[2 * i + 1].x = DhDx[4][0];
        DhC0[2 * i + 1].y = DhDx[5][0];
        DhC0[2 * i + 1].z = DhDx[6][0];
        DhC0[2 * i + 1].w = DhDx[7][0];

        DhC1[2 * i].x = DhDx[0][1];
        DhC1[2 * i].y = DhDx[1][1];
        DhC1[2 * i].z = DhDx[2][1];
        DhC1[2 * i].w = DhDx[3][1];
        DhC1[2 * i + 1].x = DhDx[4][1];
        DhC1[2 * i + 1].y = DhDx[5][1];
        DhC1[2 * i + 1].z = DhDx[6][1];
        DhC1[2 * i + 1].w = DhDx[7][1];

        DhC2[2 * i].x = DhDx[0][2];
        DhC2[2 * i].y = DhDx[1][2];
        DhC2[2 * i].z = DhDx[2][2];
        DhC2[2 * i].w = DhDx[3][2];
        DhC2[2 * i + 1].x = DhDx[4][2];
        DhC2[2 * i + 1].y = DhDx[5][2];
        DhC2[2 * i + 1].z = DhDx[6][2];
        DhC2[2 * i + 1].w = DhDx[7][2];

        for (int j=0; j<e.size(); j++)
        {
            // Force coordinates (slice number and index) for each node
            FCrds[ nbElementPerVertex * e[j] + index[e[j]] ] = int2{j, i};

            index[e[j]]++;
        }
    }

    mycudaMalloc((void**)&m_device_nodesPerElement, nodesPerElement.size() * sizeof(int4));
    mycudaMemcpyHostToDevice(m_device_nodesPerElement, nodesPerElement.data(), nodesPerElement.size() * sizeof(int4));

    mycudaMalloc((void**)&m_device_DhC0, DhC0.size() * sizeof(float4));
    mycudaMemcpyHostToDevice(m_device_DhC0, DhC0.data(), DhC0.size() * sizeof(float4));

    mycudaMalloc((void**)&m_device_DhC1, DhC1.size() * sizeof(float4));
    mycudaMemcpyHostToDevice(m_device_DhC1, DhC1.data(), DhC1.size() * sizeof(float4));

    mycudaMalloc((void**)&m_device_DhC2, DhC2.size() * sizeof(float4));
    mycudaMemcpyHostToDevice(m_device_DhC2, DhC2.data(), DhC2.size() * sizeof(float4));

    mycudaMalloc((void**)&m_device_detJ, DetJ.size() * sizeof(float));
    mycudaMemcpyHostToDevice(m_device_detJ, DetJ.data(), DetJ.size() * sizeof(float));

    mycudaMalloc((void**)&m_device_hourglassControl, hourglassControl.size() * sizeof(float));
    mycudaMemcpyHostToDevice(m_device_hourglassControl, hourglassControl.data(), DetJ.size() * sizeof(float));

    mycudaMalloc((void**)&m_device_forceCoordinates, FCrds.size() * sizeof(int2));
    mycudaMemcpyHostToDevice(m_device_forceCoordinates, FCrds.data(), FCrds.size() * sizeof(int2));

    mycudaMalloc((void**)&m_device_F0, nbElems * sizeof(float4));
    mycudaMalloc((void**)&m_device_F1, nbElems * sizeof(float4));
    mycudaMalloc((void**)&m_device_F2, nbElems * sizeof(float4));
    mycudaMalloc((void**)&m_device_F3, nbElems * sizeof(float4));
    mycudaMalloc((void**)&m_device_F4, nbElems * sizeof(float4));
    mycudaMalloc((void**)&m_device_F5, nbElems * sizeof(float4));
    mycudaMalloc((void**)&m_device_F6, nbElems * sizeof(float4));
    mycudaMalloc((void**)&m_device_F7, nbElems * sizeof(float4));

    /**
     * Initialises GPU textures with the precomputed arrays for the TLED algorithm
     */
    InitGPU_TLED(nbElementPerVertex, nbVertex, nbElems);
    delete [] index;


    /**
     * Initialises GPU textures with the precomputed arrays needed for viscoelastic formulation
     * We use viscoelastic isochoric terms only, with a single Prony series term for simplicity
     */
    if (isViscoelastic.getValue())
    {
        int Ni, Nv;
        float * Ai = 0;
        float * Av = 0;

        // Number of terms in the Prony series
        Ni = 1;
        Nv = 0;

        if (Ni != 0)
        {
            // Constants in the Prony series
            float * Visco_iso = new float[2*Ni];

            Visco_iso[0] = 0.5f;    // Denoted αi in Taylor et al. (see header file) / 0.5 for liver
            Visco_iso[1] = 0.58f;   // Dentoed τi in Taylor et al. (see header file) / 0.58 liver

            // Set up isochoric terms
            Ai = new float[2*Ni];
            for (int i = 0; i < Ni; i++)
            {
                Ai[2*i]   = timestep.getValue()*Visco_iso[2*i]/(timestep.getValue() + Visco_iso[2*i+1]);    // Denoted A in Taylor et al.
                Ai[2*i+1] = Visco_iso[2*i+1]/(timestep.getValue() + Visco_iso[2*i+1]);                      // Denoted B in Taylor et al.
            }

            delete[] Visco_iso;
        }

        if (Nv != 0)
        {
            // Constants in the Prony series
            float * Visco_vol = new float[2*Nv];

            Visco_vol[0] = 0.5f;
            Visco_vol[1] = 2.0f;

            // Set up volumetric terms
            Av = new float[2*Nv];
            for (int i = 0; i < Nv; i++)
            {
                Av[2*i]   = timestep.getValue()*Visco_vol[2*i]/(timestep.getValue() + Visco_vol[2*i+1]);
                Av[2*i+1] = Visco_vol[2*i+1]/(timestep.getValue() + Visco_vol[2*i+1]);
            }

            delete[] Visco_vol;
        }

        InitGPU_Visco(Ai, Av, Ni, Nv);
        delete [] Ai; delete [] Av;
    }

    /**
     * Initialisation of precomputed arrays needed for the anisotropic formulation
     */
    if (isAnisotropic.getValue())
    {
        // Stores the preferred direction for each element (used with transverse isotropic formulation)
        sofa::type::vector<float3> A (inputElems.size());

        Vec3f a = preferredDirection.getValue();
        for (unsigned int i = 0; i<inputElems.size(); i++)
        {
            A[3*i].x =   a[0];
            A[3*i+1].y = a[1];
            A[3*i+2].z = a[2];
        }

        // Stores the precomputed information on GPU
        InitGPU_Aniso();

        mycudaMalloc((void**)&m_device_preferredDirection, A.size() * sizeof(float));
        mycudaMemcpyHostToDevice(m_device_preferredDirection, A.data(), A.size() * sizeof(float));
    }


    msg_info() << "reinit() DONE.";
}

// --------------------------------------------------------------------------------------
// Compute internal forces
// --------------------------------------------------------------------------------------
void CudaHexahedronTLEDForceField::addForce (const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& dataF, const DataVecCoord& dataX, const DataVecDeriv& /*dataV*/)
{
    VecDeriv& f        = *(dataF.beginEdit());
    const VecCoord& x  =   dataX.getValue()  ;

    // Gets initial positions (allow to compute displacements by doing the difference between initial and current positions)
    const VecCoord& x0 = mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    f.resize(x.size());
    CudaHexahedronTLEDForceField3f_addForce(
        Lambda,
        Mu,
        nbElems,
        nbVertex,
        nbElementPerVertex,
        isViscoelastic.getValue(),
        isAnisotropic.getValue(),
        x.deviceRead(),
        x0.deviceRead(),
        f.deviceWrite(),
        m_device_nodesPerElement,
        m_device_DhC0, m_device_DhC1, m_device_DhC2,
        m_device_detJ, m_device_hourglassControl,
        m_device_preferredDirection,
        m_device_Di1, m_device_Di2, m_device_Dv1, m_device_Dv2,
        m_device_forceCoordinates,
        m_device_F0, m_device_F1, m_device_F2, m_device_F3, m_device_F4, m_device_F5, m_device_F6, m_device_F7);

    dataF.endEdit();
}

// --------------------------------------------------------------------------------------
// Only useful for implicit formulations
// --------------------------------------------------------------------------------------
void CudaHexahedronTLEDForceField::addDForce (const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& /*datadF*/, const DataVecDeriv& /*datadX*/)
{

}

// --------------------------------------------------------------------------------------
// Computes Jacobian determinant
// --------------------------------------------------------------------------------------
float CudaHexahedronTLEDForceField::ComputeDetJ(const Element& e, const VecCoord& x, float DhDr[8][3])
{
    float J[3][3];
    for (int j = 0; j < 3; j++)
    {
        for (int k = 0; k < 3; k++)
        {
            J[j][k] = 0;
            for (unsigned int m = 0; m < e.size(); m++)
            {
                J[j][k] += DhDr[m][j]*x[e[m]][k];
            }
        }
    }

    // Jacobian determinant
    const float detJ = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1]) +
            J[1][0]*(J[0][2]*J[2][1] - J[0][1]*J[2][2]) +
            J[2][0]*(J[0][1]*J[1][2] - J[0][2]*J[1][1]);

    return detJ;
}


// --------------------------------------------------------------------------------------
// Computes element volumes for hexahedral elements
// --------------------------------------------------------------------------------------
float CudaHexahedronTLEDForceField::CompElVolHexa(const Element& e, const VecCoord& x)
{
    // Calc CIJK first
    float C[8][8][8];
    ComputeCIJK(C);
    // Calc volume
    float Vol = 0;
    for (int I = 0; I < 8; I++)
    {
        for (int J = 0; J < 8; J++)
        {
            for (int K = 0; K < 8; K++)
            {
                Vol += x[e[I]][0]*x[e[J]][1]*x[e[K]][2]*C[I][J][K];
            }
        }
    }

    return Vol;
}

// --------------------------------------------------------------------------------------
// Computes coefficients CIJK used by Hourglass control
// --------------------------------------------------------------------------------------
void CudaHexahedronTLEDForceField::ComputeCIJK(float C[8][8][8])
{
    const float a = (float)(1./12);
    const float Ctemp[8*8*8] =
    {
        0,0,0,0,0,0,0,0,
        0,0,-a,-a,a,a,0,0,
        0,a,0,-a,0,0,0,0,
        0,a,a,0,-a,0,0,-a,
        0,-a,0,a,0,-a,0,a,
        0,-a,0,0,a,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,a,-a,0,0,0,

        0,0,a,a,-a,-a,0,0,
        0,0,0,0,0,0,0,0,
        -a,0,0,-a,0,a,a,0,
        -a,0,a,0,0,0,0,0,
        a,0,0,0,0,-a,0,0,
        a,0,-a,0,a,0,-a,0,
        0,0,-a,0,0,a,0,0,
        0,0,0,0,0,0,0,0,

        0,-a,0,a,0,0,0,0,
        a,0,0,a,0,-a,-a,0,
        0,0,0,0,0,0,0,0,
        -a,-a,0,0,0,0,a,a,
        0,0,0,0,0,0,0,0,
        0,a,0,0,0,0,-a,0,
        0,a,0,-a,0,a,0,-a,
        0,0,0,-a,0,0,a,0,

        0,-a,-a,0,a,0,0,a,
        a,0,-a,0,0,0,0,0,
        a,a,0,0,0,0,-a,-a,
        0,0,0,0,0,0,0,0,
        -a,0,0,0,0,0,0,a,
        0,0,0,0,0,0,0,0,
        0,0,a,0,0,0,0,-a,
        -a,0,a,0,-a,0,a,0,

        0,a,0,-a,0,a,0,-a,
        -a,0,0,0,0,a,0,0,
        0,0,0,0,0,0,0,0,
        a,0,0,0,0,0,0,-a,
        0,0,0,0,0,0,0,0,
        -a,-a,0,0,0,0,a,a,
        0,0,0,0,0,-a,0,a,
        a,0,0,a,0,-a,-a,0,

        0,a,0,0,-a,0,0,0,
        -a,0,a,0,-a,0,a,0,
        0,-a,0,0,0,0,a,0,
        0,0,0,0,0,0,0,0,
        a,a,0,0,0,0,-a,-a,
        0,0,0,0,0,0,0,0,
        0,-a,-a,0,a,0,0,a,
        0,0,0,0,a,0,-a,0,

        0,0,0,0,0,0,0,0,
        0,0,a,0,0,-a,0,0,
        0,-a,0,a,0,-a,0,a,
        0,0,-a,0,0,0,0,a,
        0,0,0,0,0,a,0,-a,
        0,a,a,0,-a,0,0,-a,
        0,0,0,0,0,0,0,0,
        0,0,-a,-a,a,a,0,0,

        0,0,0,-a,a,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,a,0,0,-a,0,
        a,0,-a,0,a,0,-a,0,
        -a,0,0,-a,0,a,a,0,
        0,0,0,0,-a,0,a,0,
        0,0,a,a,-a,-a,0,0,
        0,0,0,0,0,0,0,0
    };

    memcpy(C,Ctemp,sizeof(float)*8*8*8);
}


// --------------------------------------------------------------------------------------
// Computes shape function global derivatives DhDx for hexahedral elements
// --------------------------------------------------------------------------------------
void CudaHexahedronTLEDForceField::ComputeDhDxHexa(const Element& e, const VecCoord& x, float Vol, float DhDx[8][3])
{
    // Calc B matrix
    float B[8][3];
    ComputeBmat(e, x, B);

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            DhDx[i][j] = B[i][j]/Vol;
        }
    }
}

// --------------------------------------------------------------------------------------
// Computes matrix B used by Hourglass control
// --------------------------------------------------------------------------------------
void CudaHexahedronTLEDForceField::ComputeBmat(const Element& e, const VecCoord& x, float B[8][3])
{
    // Calc CIJK first
    float C[8][8][8];
    ComputeCIJK(C);
    // Calc B
    memset(B,0,sizeof(float)*8*3);
    for (int I = 0; I < 8; I++)
    {
        for (int J = 0; J < 8; J++)
        {
            for (int K = 0; K < 8; K++)
            {
                B[I][0] += x[e[J]][1]*x[e[K]][2]*C[I][J][K];
                B[I][1] += x[e[J]][2]*x[e[K]][0]*C[I][J][K];
                B[I][2] += x[e[J]][0]*x[e[K]][1]*C[I][J][K];
            }
        }
    }
}


// --------------------------------------------------------------------------------------
// Computes parameters for hourglass control
// --------------------------------------------------------------------------------------
void CudaHexahedronTLEDForceField::ComputeHGParams(const Element& e, const VecCoord& x, float DhDx[8][3], float volume, float HG[8][8])
{
    float a = 0;
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            a += DhDx[i][j]*DhDx[i][j];
        }
    }

    // Computes Lame coefficients
    updateLameCoefficients();

    // This value is hard coded for simplicity. We are not sure of its actual meaning, but this value (0.04) seems to work well
    const float HourGlassKappa = 0.5f;

    const float k = HourGlassKappa*volume*(Lambda+2*Mu)*a/8;

    const float Gamma[8][4] = {   {1,1,1,-1},
                                  {-1,1,-1,1},
                                  {1,-1,-1,-1},
                                  {-1,-1,1,1},
                                  {1,-1,-1,1},
                                  {-1,-1,1,-1},
                                  {1,1,1,1},
                                  {-1,1,-1,-1}
    };

    // A = DhDx * x^T
    float A[8][8];
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            A[i][j] = DhDx[i][0]*x[e[j]][0] + DhDx[i][1]*x[e[j]][1] + DhDx[i][2]*x[e[j]][2];
        }
    }

    // gamma = A * Gamma
    float gamma[8][4];
    memset(gamma, 0, 8*4*sizeof(float));
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                gamma[i][j] += A[i][k]*Gamma[k][j];
            }
        }
    }

    // gamma = Gamma - gamma
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            gamma[i][j] = Gamma[i][j] - gamma[i][j];
        }
    }

    // HG = gamma * gamma^T
    memset(HG, 0, 8*8*sizeof(float));
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                HG[i][j] += gamma[i][k]*gamma[j][k];
            }
        }
    }

    // HG = HG * k
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            HG[i][j] *= k;
        }
    }
}

// --------------------------------------------------------------------------------------
// Computes lambda and mu based on Young's modulus and Poisson ratio
// --------------------------------------------------------------------------------------
void CudaHexahedronTLEDForceField::updateLameCoefficients(void)
{
    Lambda = youngModulus.getValue()*poissonRatio.getValue()/((1 + poissonRatio.getValue())*(1 - 2*poissonRatio.getValue()));
    Mu = youngModulus.getValue()/(2*(1 + poissonRatio.getValue()));
}

} // namespace cuda

} // namespace gpu

} // namespace sofa
