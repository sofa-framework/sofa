#include "CudaHexahedronTLEDForceField.h"
#include "mycuda.h"
#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaHexahedronTLEDForceField)

int CudaHexahedronTLEDForceFieldCudaClass = core::RegisterObject("GPU-side test forcefield using CUDA")
        .add< CudaHexahedronTLEDForceField >()
        ;

extern "C"
{
    void CudaHexahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int viscoelasticity, unsigned int anisotropy, const void* x, const void* x0, void* f);
    void CudaHexahedronTLEDForceField3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx);
    void InitGPU_TLED(int* NodesPerElement, float* DhC0, float* DhC1, float* DhC2, float* DetJ, float* HG, int* FCrds, int valence, int nbVertex, int nbElements);
    void InitGPU_Visco(float * Ai, float * Av, int Ni, int Nv, int nbElements);
    void InitGPU_Aniso(void);
    void ClearGPU_TLED(void);
    void ClearGPU_Visco(void);
}

CudaHexahedronTLEDForceField::CudaHexahedronTLEDForceField()
    : nbVertex(0), nbElementPerVertex(0)
    , poissonRatio(initData(&poissonRatio,(Real)0.45,"poissonRatio","Poisson ratio in Hooke's law"))
    , youngModulus(initData(&youngModulus,(Real)3000.,"youngModulus","Young modulus in Hooke's law"))
    , timestep(initData(&timestep,(Real)0.001,"timestep","Simulation timestep"))
    , viscoelasticity(initData(&viscoelasticity,(unsigned int)0,"viscoelasticity","Viscoelasticity flag"))
    , anisotropy(initData(&anisotropy,(unsigned int)0,"anisotropy","Anisotropy flag"))
{
}

CudaHexahedronTLEDForceField::~CudaHexahedronTLEDForceField()
{
    ClearGPU_TLED();
    if (viscoelasticity.getValue())
    {
        ClearGPU_Visco();
    }
}

void CudaHexahedronTLEDForceField::init()
{
    core::componentmodel::behavior::ForceField<CudaVec3fTypes>::init();
    reinit();
}

void CudaHexahedronTLEDForceField::reinit()
{
    /// Gets the mesh
    component::topology::MeshTopology* topology = getContext()->get<component::topology::MeshTopology>();
    if (topology==NULL || topology->getNbHexas()==0)
    {
        std::cerr << "ERROR(CudaHexahedronTLEDForceField): no elements found.\n";
        return;
    }
    VecElement inputElems = topology->getHexas();

    /// Changes the winding order (faces given in counterclockwise instead of giving edges)
    for (unsigned int i=0; i<inputElems.size(); i++)
    {
        Element& e = inputElems[i];
        int temp;
        temp = e[2];
        e[2] = e[3];
        e[3] = temp;

        temp = e[6];
        e[6] = e[7];
        e[7] = temp;

        inputElems[i] = e;
    }


    /// Number of elements attached to each node
    std::map<int,int> nelems;
    for (unsigned int i=0; i<inputElems.size(); i++)
    {
        Element& e = inputElems[i];
        for (unsigned int j=0; j<e.size(); j++)
        {
            ++nelems[e[j]];
        }
    }

    /// Gets the maximum of elements attached to a vertex
    int nmax = 0;
    for (std::map<int,int>::const_iterator it = nelems.begin(); it != nelems.end(); ++it)
    {
        if (it->second > nmax)
        {
            nmax = it->second;
        }
    }

    /// Number of nodes
    int nbv = 0;
    if (!nelems.empty())
    {
        nbv = nelems.rbegin()->first + 1;
    }

    std::cout << "CudaHexahedronTLEDForceField: "<<inputElems.size()<<" elements, "<<nbv<<" nodes, max "<<nmax<<" elements per node"<<std::endl;


    /** Precomputations
    */
    init(inputElems.size(), nbv, nmax);
    std::cout << "CudaHexahedronTLEDForceField: precomputations..." << std::endl;

    const VecCoord& x = *this->mstate->getX();
    nelems.clear();

    /// Shape function natural derivatives DhDr
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

    /// Force coordinates (slice number and index) for each node
    int * FCrds = 0;

    /// Hourglass control
    float * HourglassControl = new float[64*inputElems.size()];

    /// 3 texture data for the shape function global derivatives (DhDx matrix columns for each element stored in separated arrays)
    float * DhC0 = new float[8*inputElems.size()];
    float * DhC1 = new float[8*inputElems.size()];
    float * DhC2 = new float[8*inputElems.size()];

    /// Element volume (useful to compute shape function global derivatives and Hourglass control coefficients)
    float * Volume = new float[inputElems.size()];
    /// Allocates the texture data for Jacobian determinants
    float * DetJ = new float[inputElems.size()];

    /// Retrieves force coordinates (slice number and index) for each node
    FCrds = new int[nbv*2*nmax];
    memset(FCrds, -1, nbv*2*nmax*sizeof(int));
    int * index = new int[nbv];
    memset(index, 0, nbv*sizeof(int));

    /// Stores list of nodes for each element
    int * NodesPerElement = new int[8*inputElems.size()];

    for (unsigned int i=0; i<inputElems.size(); i++)
    {
        Element& e = inputElems[i];

        /// Compute Jacobian (J = DhDr^T * x)
        DetJ[i] = ComputeDetJ(e, x, DhDr);
//         std::cout << "detJ el" << i << ":  = " << DetJ[i] << std::endl;

        /// Compute element volume
        Volume[i] = CompElVolHexa(e, x);
//         std::cout << "volume el" << i << ":  = " << Volume[i] << std::endl;

        /// Compute shape function global derivatives DhDx
        float DhDx[8][3];
        ComputeDhDxHexa(e, x, Volume[i], DhDx);

//         for (int y = 0; y < 8; y++)
//         {
//             std::cout << DhDx[y][0] << " " << DhDx[y][1] << " " << DhDx[y][2] << " " << std::endl;
//         }
//         std::cout << std::endl;

        /// Hourglass control
        float HG[8][8];
        ComputeHGParams(e, x, DhDx, Volume[i], HG);

        /// Store HG values
        int m = 0;
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                HourglassControl[64*i+m] = HG[j][k];
                m++;
            }
        }

//         for (int y = 0; y < 8; y++)
//         {
//             for (int z = 0; z < 8; z++)
//             {
//                 std::cout << HG[y][z] << " ";
//             }
//             std::cout << std::endl;
//         }
//         std::cout << std::endl;

        /// Write the list of vertices for each element
//         setE(i, e);


        for (unsigned int j=0; j<e.size(); j++)
        {
//             setV(e[j], nelems[e[j]]++, i*e.size()+j);
            /// List of nodes belonging to current element
            NodesPerElement[e.size()*i+j] = e[j];

            /// Store DhDx values in 3 texture data arrays (the 3 columns of the shape function derivatives matrix)
            DhC0[e.size()*i+j] = DhDx[j][0];
            DhC1[e.size()*i+j] = DhDx[j][1];
            DhC2[e.size()*i+j] = DhDx[j][2];

            /// Force coordinates (slice number and index) for each node
            FCrds[ 2*nmax * e[j] + 2*index[e[j]] ] = j;
            FCrds[ 2*nmax * e[j] + 2*index[e[j]]+1 ] = i;

            index[e[j]]++;
        }
    }

//     for (int i = 0; i < inputElems.size(); i++)
//     {
//         for (int j = 0; j<8; j++)
//         {
//             std::cout << DhC0[8*i+j] << " " ;
//         }
//         std::cout << std::endl;
//     }

//     for (int i = 0; i < nbv; i++)
//     {
//         for (int val = 0; val<nmax; val++)
//         {
//             std::cout << "(" << FCrds[ 2*nmax * i + 2*val ] << "," << FCrds[ 2*nmax * i + 2*val+1 ] << ") ";
//         }
//         std::cout << std::endl;
//     }

    /** Initialise GPU textures with the precomputed array for the TLED algorithm
     */
    InitGPU_TLED(NodesPerElement, DhC0, DhC1, DhC2, DetJ, HourglassControl, FCrds, nmax, nbv, inputElems.size());
    delete [] NodesPerElement; delete [] DhC0; delete [] DhC1; delete [] DhC2; delete [] index;
    delete [] DetJ; delete [] FCrds; delete [] HourglassControl;


    /** Initialise GPU textures with the precomputed array needed for viscoelastic formulation
     */
    if (viscoelasticity.getValue())
    {
        int Ni, Nv;
        float * Ai = 0;
        float * Av = 0;

        /// Number of terms in the Prony series
        Ni = 1;
        Nv = 0;

        if (Ni != 0)
        {
            /// Constants in the Prony series
            float * Visco_iso = new float[2*Ni];

            Visco_iso[0] = 0.5f;    // 0.5 liver
            Visco_iso[1] = 0.58f;   // 0.58 liver

            /// Set up isochoric terms
            Ai = new float[2*Ni];
            for (int i = 0; i < Ni; i++)
            {
                Ai[2*i]   = timestep.getValue()*Visco_iso[2*i]/(timestep.getValue() + Visco_iso[2*i+1]);
                Ai[2*i+1] = Visco_iso[2*i+1]/(timestep.getValue() + Visco_iso[2*i+1]);
            }
        }

        if (Nv != 0)
        {
            /// Constants in the Prony series
            float * Visco_vol = new float[2*Nv];

            Visco_vol[0] = 0.5f;
            Visco_vol[1] = 2.0f;

            /// Set up volumetric terms
            Av = new float[2*Nv];
            for (int i = 0; i < Nv; i++)
            {
                Av[2*i]   = timestep.getValue()*Visco_vol[2*i]/(timestep.getValue() + Visco_vol[2*i+1]);
                Av[2*i+1] = Visco_vol[2*i+1]/(timestep.getValue() + Visco_vol[2*i+1]);
            }
        }

        InitGPU_Visco(Ai, Av, Ni, Nv, inputElems.size());
        delete [] Ai; delete [] Av;
    }

    /** Initialise GPU textures with the precomputed array needed for anisotropic formulation
     */
    if (anisotropy.getValue())
    {
        InitGPU_Aniso();
    }


    std::cout << "CudaHexahedronTLEDForceField::reinit() DONE."<<std::endl;
}

void CudaHexahedronTLEDForceField::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    // Gets initial positions (allow to compute displacements by doing the difference between initial and current positions)
    const VecCoord& x0 = *mstate->getX0();

    f.resize(x.size());
    CudaHexahedronTLEDForceField3f_addForce(
        Lambda,
        Mu,
        elems.size(),
        nbVertex,
        nbElementPerVertex,
        viscoelasticity.getValue(),
        anisotropy.getValue(),
        x.deviceRead(),
        x0.deviceRead(),
        f.deviceWrite());
}

void CudaHexahedronTLEDForceField::addDForce (VecDeriv& df, const VecDeriv& dx)
{
    df.resize(dx.size());
    CudaHexahedronTLEDForceField3f_addDForce(
        elems.size(),
        nbVertex,
        nbElementPerVertex,
        elems.deviceRead(),
        state.deviceWrite(),
        velems.deviceRead(),
        df.deviceWrite(),
        dx.deviceRead());
}


/**Compute Jacobian determinant
*/
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

    /// Jacobian determinant
    float detJ = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1]) +
            J[1][0]*(J[0][2]*J[2][1] - J[0][1]*J[2][2]) +
            J[2][0]*(J[0][1]*J[1][2] - J[0][2]*J[1][1]);

    return detJ;
}


/** Compute element volumes for hexahedral elements
 */
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

void CudaHexahedronTLEDForceField::ComputeCIJK(float C[8][8][8])
{
    float a = (float)(1./12);
    float Ctemp[8*8*8] =
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


/** Compute shape function global derivatives DhDx for hexahedral helements
 */
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

/** Compute parameters for hourglass control
 */
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

    /// Set Lame coefficients
    updateLameCoefficients();

    // Zeike: Hard coded for simplicity.
    // Still haven't got my head around this parameter exactly, but this value (0.04) seems to work well
    float HourGlassKappa = 0.5f;

    float k = HourGlassKappa*volume*(Lambda+2*Mu)*a/8;

    float Gamma[8][4] = {   {1,1,1,-1},
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

//     for (int i = 0; i < 8; i++)
//     {
//         for (int j = 0; j < 8; j++)
//         {
//             cout << HG[i][j] << ", " ;
//         }
//         cout << endl;
//     }

}

/** Compute lambda and mu based on the Young modulus and Poisson ratio
 */
void CudaHexahedronTLEDForceField::updateLameCoefficients(void)
{
    Lambda = youngModulus.getValue()*poissonRatio.getValue()/((1 + poissonRatio.getValue())*(1 - 2*poissonRatio.getValue()));
    Mu = youngModulus.getValue()/(2*(1 + poissonRatio.getValue()));
}

} // namespace cuda

} // namespace gpu

} // namespace sofa
