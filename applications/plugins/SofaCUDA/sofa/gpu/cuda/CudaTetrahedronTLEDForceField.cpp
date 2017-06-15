/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "CudaTetrahedronTLEDForceField.h"
#include "mycuda.h"
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <SofaBaseTopology/RegularGridTopology.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaTetrahedronTLEDForceField)

int CudaTetrahedronTLEDForceFieldCudaClass = core::RegisterObject("GPU TLED tetrahedron forcefield using CUDA")
        .add< CudaTetrahedronTLEDForceField >()
        ;

extern "C"
{
    void CudaTetrahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int isViscoelastic, unsigned int isAnisotropic, const void* x, const void* x0, void* f);
    void InitGPU_TetrahedronTLED(int* NodesPerElement, float* DhC0, float* DhC1, float* DhC2, float* Volume, int* FCrds, int valence, int nbVertex, int nbElements);
    void InitGPU_TetrahedronVisco(float * Ai, float * Av, int Ni, int Nv);
    void InitGPU_TetrahedronAniso(float* A);
    void ClearGPU_TetrahedronTLED(void);
    void ClearGPU_TetrahedronVisco(void);
    void ClearGPU_TetrahedronAniso(void);
}

// --------------------------------------------------------------------------------------
// Constructor - Initialises member variables from scene file
// --------------------------------------------------------------------------------------
CudaTetrahedronTLEDForceField::CudaTetrahedronTLEDForceField()
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
CudaTetrahedronTLEDForceField::~CudaTetrahedronTLEDForceField()
{
    ClearGPU_TetrahedronTLED();

    if (isViscoelastic.getValue())
    {
        ClearGPU_TetrahedronVisco();
    }

    if (isAnisotropic.getValue())
    {
        ClearGPU_TetrahedronAniso();
    }
}

void CudaTetrahedronTLEDForceField::init()
{
    core::behavior::ForceField<CudaVec3fTypes>::init();
    reinit();
}

// --------------------------------------------------------------------------------------
// Initialisation and precomputations
// --------------------------------------------------------------------------------------
void CudaTetrahedronTLEDForceField::reinit()
{
    // Gets the mesh
    sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();

    if (topology==NULL)
    {
        serr << "ERROR(CudaTetrahedronTLEDForceField): no topology found." << sendl;
        return;
    }
    VecElement inputElems = topology->getTetrahedra();

    // If hexahedral topology, splits every hexahedron into 6 tetrahedra
    if (inputElems.empty())
    {
        if (topology->getNbHexahedra() == 0)
        {
            serr << "ERROR(CudaTetrahedronTLEDForceField): this forcefield requires a tetrahedral or hexahedral topology." << sendl;
            return;
        }
        int nbcubes = topology->getNbHexahedra();
        // These values are only correct if the mesh is a grid topology
        int nx = 2;
        int ny = 1;
//        int nz = 1;
        {
            component::topology::GridTopology* grid = dynamic_cast<component::topology::GridTopology*>(topology);
            if (grid != NULL)
            {
                nx = grid->getNx()-1;
                ny = grid->getNy()-1;
//                nz = grid->getNz()-1;
            }
        }

        // Tesselation of each cube into 6 tetrahedra
        inputElems.reserve(nbcubes*6);
        for (int i=0; i<nbcubes; i++)
        {
            // if (flags && !flags->isCubeActive(i)) continue;
            core::topology::BaseMeshTopology::Hexa c = topology->getHexahedron(i);
#define swap(a,b) { int t = a; a = b; b = t; }
            if (!((i%nx)&1))
            {
                // swap all points on the X edges
                swap(c[0],c[1]);
                swap(c[3],c[2]);
                swap(c[4],c[5]);
                swap(c[7],c[6]);
            }
            if (((i/nx)%ny)&1)
            {
                // swap all points on the Y edges
                swap(c[0],c[3]);
                swap(c[1],c[2]);
                swap(c[4],c[7]);
                swap(c[5],c[6]);
            }
            if ((i/(nx*ny))&1)
            {
                // swap all points on the Z edges
                swap(c[0],c[4]);
                swap(c[1],c[5]);
                swap(c[2],c[6]);
                swap(c[3],c[7]);
            }
#undef swap
            typedef core::topology::BaseMeshTopology::Tetra Tetra;
            inputElems.push_back(Tetra(c[0],c[5],c[1],c[6]));
            inputElems.push_back(Tetra(c[0],c[1],c[3],c[6]));
            inputElems.push_back(Tetra(c[1],c[3],c[6],c[2]));
            inputElems.push_back(Tetra(c[6],c[3],c[0],c[7]));
            inputElems.push_back(Tetra(c[6],c[7],c[0],c[5]));
            inputElems.push_back(Tetra(c[7],c[5],c[4],c[0]));
        }
        std::cout << "WARNING(CudaTetrahedronTLEDForceField): each hexahedron has been split into 6 tetrahedra. You might want to use CudaHexahedronTLEDForceField instead." << std::endl;
    }

    // Gets the number of elements
    nbElems = inputElems.size();

    // Number of elements attached to each node
    std::map<int,int> nelems;
    for (int i=0; i<nbElems; i++)
    {
        Element& e = inputElems[i];
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

    std::cout << "CudaTetrahedronTLEDForceField: " << nbElems << " elements, " << nbVertex << " nodes, max " << nbElementPerVertex << " elements per node" << std::endl;


    /**
     * Precomputations
     */
    std::cout << "CudaTetrahedronTLEDForceField: precomputations..." << std::endl;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    nelems.clear();

    // Shape function natural derivatives DhDr
    float DhDr[4][3];
    DhDr[0][0] = -1; DhDr[0][1] = -1; DhDr[0][2] = -1;
    DhDr[1][0] = 1;  DhDr[1][1] = 0;  DhDr[1][2] = 0;
    DhDr[2][0] = 0;  DhDr[2][1] = 1;  DhDr[2][2] = 0;
    DhDr[3][0] = 0;  DhDr[3][1] = 0;  DhDr[3][2] = 1;

    // Force coordinates (slice number and index) for each node
    int * FCrds = 0;

    // 3 texture data for the shape function global derivatives (DhDx matrix columns for each element stored in separated arrays)
    float * DhC0 = new float[4*nbElems];
    float * DhC1 = new float[4*nbElems];
    float * DhC2 = new float[4*nbElems];

    // Element volume (useful to compute shape function global derivatives)
    float * Volume = new float[nbElems];

    // Retrieves force coordinates (slice number and index) for each node
    FCrds = new int[nbVertex*2*nbElementPerVertex];
    memset(FCrds, -1, nbVertex*2*nbElementPerVertex*sizeof(int));
    int * index = new int[nbVertex];
    memset(index, 0, nbVertex*sizeof(int));

    // Stores list of nodes for each element
    int * NodesPerElement = new int[4*nbElems];

    // Stores shape function global derivatives
    float DhDx[4][3];

    for (int i=0; i<nbElems; i++)
    {
        Element& e = inputElems[i];

        // Compute element volume
        Volume[i] = CompElVolTetra(e, x);

        // Compute shape function global derivatives DhDx (DhDx = DhDr * invJ^T)
        ComputeDhDxTetra(e, x, DhDr, DhDx);

        for (unsigned int j=0; j<e.size(); j++)
        {
            // List of nodes belonging to current element
            NodesPerElement[e.size()*i+j] = e[j];

            // Store DhDx values in 3 texture data arrays (the 3 columns of the shape function derivatives matrix)
            DhC0[e.size()*i+j] = DhDx[j][0];
            DhC1[e.size()*i+j] = DhDx[j][1];
            DhC2[e.size()*i+j] = DhDx[j][2];

            // Force coordinates (slice number and index) for each node
            FCrds[ 2*nbElementPerVertex * e[j] + 2*index[e[j]] ] = j;
            FCrds[ 2*nbElementPerVertex * e[j] + 2*index[e[j]]+1 ] = i;

            index[e[j]]++;
        }
    }

    /** Initialises GPU textures with the precomputed arrays for the TLED algorithm
     */
    InitGPU_TetrahedronTLED(NodesPerElement, DhC0, DhC1, DhC2, Volume, FCrds, nbElementPerVertex, nbVertex, nbElems);
    delete [] NodesPerElement; delete [] DhC0; delete [] DhC1; delete [] DhC2; delete [] index;
    delete [] FCrds; delete [] Volume;


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
        }

        InitGPU_TetrahedronVisco(Ai, Av, Ni, Nv);
        delete [] Ai; delete [] Av;
    }

    /**
     * Initialisation of precomputed arrays needed for the anisotropic formulation
     */
    if (isAnisotropic.getValue())
    {
        // Stores the preferred direction for each element (used with transverse isotropic formulation)
        float* A = new float[3*inputElems.size()];

        // By default, every element is set up with the same direction (given by the vector preferredDirection provided by the scene file)
        Vec3f a = preferredDirection.getValue();
        for (unsigned int i = 0; i<inputElems.size(); i++)
        {
            A[3*i] =   a[0];
            A[3*i+1] = a[1];
            A[3*i+2] = a[2];
        }

        // Stores the precomputed information on GPU
        InitGPU_TetrahedronAniso(A);
    }

    // Computes Lame coefficients
    updateLameCoefficients();

    sout << "CudaTetrahedronTLEDForceField::reinit() DONE." << sendl;
}

// --------------------------------------------------------------------------------------
// Compute internal forces
// --------------------------------------------------------------------------------------
void CudaTetrahedronTLEDForceField::addForce (const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& dataF, const DataVecCoord& dataX, const DataVecDeriv& /*dataV*/)
{
    VecDeriv& f        = *(dataF.beginEdit());
    const VecCoord& x  =   dataX.getValue()  ;

    // Gets initial positions (allow to compute displacements by doing the difference between initial and current positions)
    const VecCoord& x0 = mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    f.resize(x.size());
    CudaTetrahedronTLEDForceField3f_addForce(
        Lambda,
        Mu,
        nbElems,
        nbVertex,
        nbElementPerVertex,
        isViscoelastic.getValue(),
        isAnisotropic.getValue(),
        x.deviceRead(),
        x0.deviceRead(),
        f.deviceWrite());

    dataF.endEdit();
}

// --------------------------------------------------------------------------------------
// Only useful for implicit formulations
// --------------------------------------------------------------------------------------
void CudaTetrahedronTLEDForceField::addDForce (const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& /*datadF*/, const DataVecDeriv& /*datadX*/)
{

}


// --------------------------------------------------------------------------------------
// Computes element volumes for tetrahedral elements
// --------------------------------------------------------------------------------------
float CudaTetrahedronTLEDForceField::CompElVolTetra( const Element& e, const VecCoord& x )
{
    float Vol;
    Vol = fabs((x[e[0]][0]*x[e[1]][2]*x[e[2]][1] - x[e[0]][0]*x[e[1]][1]*x[e[2]][2] - x[e[1]][2]*x[e[2]][1]*x[e[3]][0] +
            x[e[1]][1]*x[e[2]][2]*x[e[3]][0] - x[e[0]][0]*x[e[1]][2]*x[e[3]][1] + x[e[1]][2]*x[e[2]][0]*x[e[3]][1] +
            x[e[0]][0]*x[e[2]][2]*x[e[3]][1] - x[e[1]][0]*x[e[2]][2]*x[e[3]][1] + x[e[0]][2]*(x[e[1]][1]*(x[e[2]][0]-x[e[3]][0]) -
                    x[e[1]][0]*x[e[2]][1] + x[e[2]][1]*x[e[3]][0] + x[e[1]][0]*x[e[3]][1] - x[e[2]][0]*x[e[3]][1])
            + x[e[0]][0]*x[e[1]][1]*x[e[3]][2] - x[e[1]][1]*x[e[2]][0]*x[e[3]][2] - x[e[0]][0]*x[e[2]][1]*x[e[3]][2] +
            x[e[1]][0]*x[e[2]][1]*x[e[3]][2] + x[e[0]][1]*(x[e[1]][0]*x[e[2]][2] - x[e[1]][2]*x[e[2]][0] + x[e[1]][2]*x[e[3]][0] -
                    x[e[2]][2]*x[e[3]][0] - x[e[1]][0]*x[e[3]][2] + x[e[2]][0]*x[e[3]][2]))/6);

    return Vol;
}


// -----------------------------------------------------------------------------------------------
// Computes shape function global derivatives DhDx for tetrahedral elements (DhDx = DhDr * invJ^T)
// -----------------------------------------------------------------------------------------------
void CudaTetrahedronTLEDForceField::ComputeDhDxTetra(const Element& e, const VecCoord& x, float DhDr[4][3], float DhDx[4][3])
{
    // Compute Jacobian
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
    float detJ = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1]) +
            J[1][0]*(J[0][2]*J[2][1] - J[0][1]*J[2][2]) +
            J[2][0]*(J[0][1]*J[1][2] - J[0][2]*J[1][1]);

    // Jacobian inverse
    double invJ[3][3];
    invJ[0][0] = (J[1][1]*J[2][2] - J[1][2]*J[2][1])/detJ;
    invJ[0][1] = (J[0][2]*J[2][1] - J[0][1]*J[2][2])/detJ;
    invJ[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1])/detJ;
    invJ[1][0] = (J[1][2]*J[2][0] - J[1][0]*J[2][2])/detJ;
    invJ[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0])/detJ;
    invJ[1][2] = (J[0][2]*J[1][0] - J[0][0]*J[1][2])/detJ;
    invJ[2][0] = (J[1][0]*J[2][1] - J[1][1]*J[2][0])/detJ;
    invJ[2][1] = (J[0][1]*J[2][0] - J[0][0]*J[2][1])/detJ;
    invJ[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0])/detJ;


    // Compute shape function global derivatives
    for (int j = 0; j < 4; j++)
    {
        for (int k = 0; k < 3; k++)
        {
            DhDx[j][k] = 0;
            for (int m = 0; m < 3; m++)
            {
                DhDx[j][k] += (float)(DhDr[j][m]*invJ[k][m]);
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------
// Computes lambda and mu based on Young's modulus and Poisson ratio
// -----------------------------------------------------------------------------------------------
void CudaTetrahedronTLEDForceField::updateLameCoefficients(void)
{
    Lambda = youngModulus.getValue()*poissonRatio.getValue()/((1 + poissonRatio.getValue())*(1 - 2*poissonRatio.getValue()));
    Mu = youngModulus.getValue()/(2*(1 + poissonRatio.getValue()));
}

} // namespace cuda

} // namespace gpu

} // namespace sofa
