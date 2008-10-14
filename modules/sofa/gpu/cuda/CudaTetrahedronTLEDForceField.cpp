/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include "CudaTetrahedronTLEDForceField.h"
#include "mycuda.h"
#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaTetrahedronTLEDForceField)

int CudaTetrahedronTLEDForceFieldCudaClass = core::RegisterObject("GPU-side test forcefield using CUDA")
        .add< CudaTetrahedronTLEDForceField >()
        ;

extern "C"
{
    void CudaTetrahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int viscoelasticity, unsigned int anisotropy, const void* x, const void* x0, void* f);
    void CudaTetrahedronTLEDForceField3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx);
    void InitGPU_TetrahedronTLED(int* NodesPerElement, float* DhC0, float* DhC1, float* DhC2, float* Volume, int* FCrds, int valence, int nbVertex, int nbElements);
    void InitGPU_TetrahedronVisco(float * Ai, float * Av, int Ni, int Nv, int nbElements);
    void InitGPU_TetrahedronAniso(void);
    void ClearGPU_TetrahedronTLED(void);
    void ClearGPU_TetrahedronVisco(void);
}

CudaTetrahedronTLEDForceField::CudaTetrahedronTLEDForceField()
    : nbVertex(0), nbElementPerVertex(0)
    , poissonRatio(initData(&poissonRatio,(Real)0.45,"poissonRatio","Poisson ratio in Hooke's law"))
    , youngModulus(initData(&youngModulus,(Real)3000.,"youngModulus","Young modulus in Hooke's law"))
    , timestep(initData(&timestep,(Real)0.001,"timestep","Simulation timestep"))
    , viscoelasticity(initData(&viscoelasticity,(unsigned int)0,"viscoelasticity","Viscoelasticity flag"))
    , anisotropy(initData(&anisotropy,(unsigned int)0,"anisotropy","Anisotropy flag"))
{
}

CudaTetrahedronTLEDForceField::~CudaTetrahedronTLEDForceField()
{
    ClearGPU_TetrahedronTLED();
    if (viscoelasticity.getValue())
    {
        ClearGPU_TetrahedronVisco();
    }
}

void CudaTetrahedronTLEDForceField::init()
{
    core::componentmodel::behavior::ForceField<CudaVec3fTypes>::init();
    reinit();
}

void CudaTetrahedronTLEDForceField::reinit()
{
    /// Gets the mesh
    component::topology::MeshTopology* topology = getContext()->get<component::topology::MeshTopology>();
    if (topology==NULL || topology->getNbTetras()==0)
    {
        std::cerr << "ERROR(CudaTetrahedronTLEDForceField): no elements found.\n";
        return;
    }
    VecElement inputElems = topology->getTetras();

    /// Changes the winding order (faces given in counterclockwise instead of giving edges)
    /// Needed ?

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

    std::cout << "CudaTetrahedronTLEDForceField: "<<inputElems.size()<<" elements, "<<nbv<<" nodes, max "<<nmax<<" elements per node"<<std::endl;


    /** Precomputations
    */
    init(inputElems.size(), nbv, nmax);
    std::cout << "CudaTetrahedronTLEDForceField: precomputations..." << std::endl;

    const VecCoord& x = *this->mstate->getX();
    nelems.clear();

    /// Shape function natural derivatives DhDr
    float DhDr[4][3];
    DhDr[0][0] = -1; DhDr[0][1] = -1; DhDr[0][2] = -1;
    DhDr[1][0] = 1;  DhDr[1][1] = 0;  DhDr[1][2] = 0;
    DhDr[2][0] = 0;  DhDr[2][1] = 1;  DhDr[2][2] = 0;
    DhDr[3][0] = 0;  DhDr[3][1] = 0;  DhDr[3][2] = 1;

    /// Force coordinates (slice number and index) for each node
    int * FCrds = 0;

    /// 3 texture data for the shape function global derivatives (DhDx matrix columns for each element stored in separated arrays)
    float * DhC0 = new float[4*inputElems.size()];
    float * DhC1 = new float[4*inputElems.size()];
    float * DhC2 = new float[4*inputElems.size()];

    /// Element volume (useful to compute shape function global derivatives and Hourglass control coefficients)
    float * Volume = new float[inputElems.size()];

    /// Retrieves force coordinates (slice number and index) for each node
    FCrds = new int[nbv*2*nmax];
    memset(FCrds, -1, nbv*2*nmax*sizeof(int));
    int * index = new int[nbv];
    memset(index, 0, nbv*sizeof(int));

    /// Stores list of nodes for each element
    int * NodesPerElement = new int[4*inputElems.size()];

    /// Stores shape function global derivatives
    float DhDx[4][3];

    for (unsigned int i=0; i<inputElems.size(); i++)
    {
        Element& e = inputElems[i];

        /// Compute element volume
        Volume[i] = CompElVolTetra(e, x);

        /// Compute shape function global derivatives DhDx (DhDx = DhDr * invJ^T)
        ComputeDhDxTetra(e, x, DhDr, DhDx);

        for (unsigned int j=0; j<e.size(); j++)
        {
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
//         for (int j = 0; j<4; j++)
//         {
//             std::cout << DhC0[4*i+j] << " " ;
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
    InitGPU_TetrahedronTLED(NodesPerElement, DhC0, DhC1, DhC2, Volume, FCrds, nmax, nbv, inputElems.size());
    delete [] NodesPerElement; delete [] DhC0; delete [] DhC1; delete [] DhC2; delete [] index;
    delete [] FCrds; delete [] Volume;


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

        InitGPU_TetrahedronVisco(Ai, Av, Ni, Nv, inputElems.size());
        delete [] Ai; delete [] Av;
    }

    /** Initialise GPU textures with the precomputed array needed for anisotropic formulation
     */
    if (anisotropy.getValue())
    {
        InitGPU_TetrahedronAniso();
    }

    /// Set up Lame coefficients
    updateLameCoefficients();

    std::cout << "CudaTetrahedronTLEDForceField::reinit() DONE."<<std::endl;
}

void CudaTetrahedronTLEDForceField::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    // Gets initial positions (allow to compute displacements by doing the difference between initial and current positions)
    const VecCoord& x0 = *mstate->getX0();

    f.resize(x.size());
    CudaTetrahedronTLEDForceField3f_addForce(
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

void CudaTetrahedronTLEDForceField::addDForce (VecDeriv& df, const VecDeriv& dx)
{
    df.resize(dx.size());
    CudaTetrahedronTLEDForceField3f_addDForce(
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
// float CudaTetrahedronTLEDForceField::ComputeDetJ(const Element& e, const VecCoord& x, float DhDr[8][3])
// {
//     float J[3][3];
//     for (int j = 0; j < 3; j++)
//     {
//         for (int k = 0; k < 3; k++)
//         {
//             J[j][k] = 0;
//             for (unsigned int m = 0; m < e.size(); m++)
//             {
//                 J[j][k] += DhDr[m][j]*x[e[m]][k];
//             }
//         }
//     }
//
//     /// Jacobian determinant
//     float detJ = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1]) +
//                  J[1][0]*(J[0][2]*J[2][1] - J[0][1]*J[2][2]) +
//                  J[2][0]*(J[0][1]*J[1][2] - J[0][2]*J[1][1]);
//
//     return detJ;
// }


/** Compute element volumes for tetrahedral elements
 */
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

/**
 * Compute shape function global derivatives DhDx for tetrahedral elements (DhDx = DhDr * invJ^T)
 */
void CudaTetrahedronTLEDForceField::ComputeDhDxTetra(const Element& e, const VecCoord& x, float DhDr[4][3], float DhDx[4][3])
{
    /// Compute Jacobian
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

    /// Jacobian inverse
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


    /// Compute shape function global derivatives
    for (int j = 0; j < 4; j++)
    {
        for (int k = 0; k < 3; k++)
        {
            DhDx[j][k] = 0;
            for (int m = 0; m < 3; m++)
            {
                DhDx[j][k] += DhDr[j][m]*invJ[k][m];
            }
        }
    }
}

/** Compute lambda and mu based on the Young modulus and Poisson ratio
 */
void CudaTetrahedronTLEDForceField::updateLameCoefficients(void)
{
    Lambda = youngModulus.getValue()*poissonRatio.getValue()/((1 + poissonRatio.getValue())*(1 - 2*poissonRatio.getValue()));
    Mu = youngModulus.getValue()/(2*(1 + poissonRatio.getValue()));
}

} // namespace cuda

} // namespace gpu

} // namespace sofa
