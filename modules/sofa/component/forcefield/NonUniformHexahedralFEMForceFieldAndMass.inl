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
#ifndef SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_INL
#define SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_INL

#include <sofa/component/forcefield/NonUniformHexahedralFEMForceFieldAndMass.h>

#include <sofa/component/topology/MultilevelHexahedronSetTopologyContainer.h>

#include <sofa/component/topology/PointData.inl>
#include <sofa/component/topology/HexahedronData.inl>

#include <sofa/core/objectmodel/Base.h>

using std::set;

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template <class DataTypes>
NonUniformHexahedralFEMForceFieldAndMass<DataTypes>::NonUniformHexahedralFEMForceFieldAndMass()
    : HexahedralFEMForceFieldAndMassT()
{}

template <class DataTypes>
void NonUniformHexahedralFEMForceFieldAndMass<DataTypes>::init()
{
    this->core::componentmodel::behavior::ForceField<DataTypes>::init();

    this->getContext()->get(this->_topology);

    if(this->_topology == NULL)
    {
        serr << "ERROR(NonUniformHexahedralFEMForceFieldAndMass): object must have a HexahedronSetTopology."<<sendl;
        return;
    }

    _multilevelTopology = dynamic_cast<topology::MultilevelHexahedronSetTopologyContainer*>(this->_topology);

    if(_multilevelTopology == NULL)
    {
        serr << "ERROR(NonUniformHexahedralFEMForceFieldAndMass): object must have a MultilevelHexahedronSetTopologyContainer";
    }

    this->reinit();
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::reinit()
{
    if (this->f_method.getValue() == "large")
        this->setMethod(HexahedralFEMForceFieldT::LARGE);
    else if (this->f_method.getValue() == "polar")
        this->setMethod(HexahedralFEMForceFieldT::POLAR);

    helper::vector<typename HexahedralFEMForceField<T>::HexahedronInformation>& hexahedronInf = *(this->hexahedronInfo.beginEdit());
    hexahedronInf.resize(this->_topology->getNbHexas());
    this->hexahedronInfo.endEdit();

    helper::vector<ElementMass>& elementMasses = *this->_elementMasses.beginEdit();
    elementMasses.resize( this->_topology->getNbHexas() );
    this->_elementMasses.endEdit();

    helper::vector<Real>& elementTotalMass = *this->_elementTotalMass.beginEdit();
    elementTotalMass.resize( this->_topology->getNbHexas() );
    this->_elementTotalMass.endEdit();

    const int level = _multilevelTopology->getLevel();
    const int coarseNodeSize = (1 << level);

    const VecCoord *X0=this->mstate->getX0();
    Vec<8,Coord> nodesCoarse;
    for(int w=0; w<8; ++w)
        nodesCoarse[w] = (*X0)[this->_topology->getHexa(0)[w]];

    Vec<8,Coord> nodesFine;
    for(int w=0; w<8; ++w)
        nodesFine[w] = (nodesCoarse[w] - nodesCoarse[0]) / coarseNodeSize;

    HexahedralFEMForceField<T>::computeMaterialStiffness(_material.C, this->f_youngModulus.getValue(), this->f_poissonRatio.getValue());
    HexahedralFEMForceField<T>::computeElementStiffness(_material.K, _material.C, nodesFine);
    HexahedralFEMForceFieldAndMass<T>::computeElementMass(_material.M, _material.mass, nodesFine);


    const float cube[8][3]=
    {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
    };

    for(int i=0; i<8; ++i) // child
    {
        for(int w=0; w<8; ++w) // childNodeId
        {
            float x, y, z;

            x = 0.5f * (cube[i][0] + cube[w][0]);
            y = 0.5f * (cube[i][1] + cube[w][1]);
            z = 0.5f * (cube[i][2] + cube[w][2]);

            _H[i][w][0] = (1-x)*(1-y)*(1-z);
            _H[i][w][1] = (x)*(1-y)*(1-z);
            _H[i][w][2] = (x)*(y)*(1-z);
            _H[i][w][3] = (1-x)*(y)*(1-z);
            _H[i][w][4] = (1-x)*(1-y)*(z);
            _H[i][w][5] = (x)*(1-y)*(z);
            _H[i][w][6] = (x)*(y)*(z);
            _H[i][w][7] = (1-x)*(y)*(z);
        }
    }

    const float fineNodeSize = 1.0f / (float) coarseNodeSize;
    int idx=0;

    __H.resize(coarseNodeSize * coarseNodeSize * coarseNodeSize);

    for(int k=0; k<coarseNodeSize; ++k)
        for(int j=0; j<coarseNodeSize; ++j)
            for(int i=0; i<coarseNodeSize; ++i, ++idx)
            {
                for(int w=0; w<8; ++w) // childNodeId
                {
                    const float x = fineNodeSize * (i + cube[w][0]);
                    const float y = fineNodeSize * (j + cube[w][1]);
                    const float z = fineNodeSize * (k + cube[w][2]);

                    // entree dans la matrice pour le sommet w
                    __H[idx][w][0] = (1-x)*(1-y)*(1-z);
                    __H[idx][w][1] = (x)*(1-y)*(1-z);
                    __H[idx][w][2] = (x)*(y)*(1-z);
                    __H[idx][w][3] = (1-x)*(y)*(1-z);
                    __H[idx][w][4] = (1-x)*(1-y)*(z);
                    __H[idx][w][5] = (x)*(1-y)*(z);
                    __H[idx][w][6] = (x)*(y)*(z);
                    __H[idx][w][7] = (1-x)*(y)*(z);
                }
            }


    switch(this->method)
    {
    case HexahedralFEMForceFieldT::LARGE:
    {
        for (int i=0; i<this->_topology->getNbHexas(); ++i)
            initLarge(i);
    }
    break;
    case HexahedralFEMForceFieldT::POLAR:
    {
        for(int i=0; i<this->_topology->getNbHexas(); ++i)
            initPolar(i);
    }
    break;
    }

    HexahedralFEMForceFieldAndMassT::computeParticleMasses();
    HexahedralFEMForceFieldAndMassT::computeLumpedMasses();
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::handleTopologyChange(core::componentmodel::topology::Topology* t)
{
    if(t != this->_topology)
        return;

    std::list<const TopologyChange *>::const_iterator itBegin=this->_topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=this->_topology->lastChange();

    // handle point events
    this->_particleMasses.handleTopologyEvents(itBegin,itEnd);

    if( this->_useLumpedMass.getValue() )
        this->_lumpedMasses.handleTopologyEvents(itBegin,itEnd);

    for(std::list<const TopologyChange *>::const_iterator iter = itBegin;
        iter != itEnd; ++iter)
    {
        std::list<const TopologyChange *>::const_iterator next_iter = iter;
        ++next_iter;

        switch((*iter)->getChangeType())
        {
            // for added elements:
            // init element
            // add particle masses and lumped masses of adjacent particles
        case core::componentmodel::topology::HEXAHEDRAADDED:
        {
            // handle hexa events
            this->hexahedronInfo.handleTopologyEvents(iter, next_iter);
            this->_elementMasses.handleTopologyEvents(iter, next_iter);
            this->_elementTotalMass.handleTopologyEvents(iter, next_iter);

            const VecElement& hexas = this->_topology->getHexas();
            const sofa::helper::vector<unsigned int> &hexaModif = (static_cast< const HexahedraAdded *> (*iter))->hexahedronIndexArray;

            switch(this->method)
            {
            case HexahedralFEMForceFieldT::LARGE:
            {
                for(unsigned int i=0; i<hexaModif.size(); ++i)
                    initLarge(hexaModif[i]);
            }
            break;
            case HexahedralFEMForceFieldT::POLAR:
            {
                for(unsigned int i=0; i<hexaModif.size(); ++i)
                    initPolar(hexaModif[i]);
            }
            break;
            }

            helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();

            for(unsigned int i=0; i<hexaModif.size(); ++i)
            {
                const unsigned int hexaId = hexaModif[i];

                Real mass = this->_elementTotalMass.getValue()[hexaId] * (Real) 0.125;

                for(int w=0; w<8; ++w)
                    particleMasses[ hexas[hexaId][w] ] += mass;
            }

            this->_particleMasses.endEdit();

            if( this->_useLumpedMass.getValue() )
            {
                helper::vector<Coord>&	lumpedMasses = *this->_lumpedMasses.beginEdit();

                for(unsigned int i=0; i<hexaModif.size(); ++i)
                {
                    const unsigned int hexaId = hexaModif[i];
                    const ElementMass& mass = this->_elementMasses.getValue()[hexaId];

                    for(int w=0; w<8; ++w)
                    {
                        for(int j=0; j<8*3; ++j)
                        {
                            lumpedMasses[ hexas[hexaId][w] ][0] += mass[w*3  ][j];
                            lumpedMasses[ hexas[hexaId][w] ][1] += mass[w*3+1][j];
                            lumpedMasses[ hexas[hexaId][w] ][2] += mass[w*3+2][j];
                        }
                    }
                }

                this->_lumpedMasses.endEdit();
            }
        }
        break;

        // for removed elements:
        // subtract particle masses and lumped masses of adjacent particles
        case core::componentmodel::topology::HEXAHEDRAREMOVED:
        {
            const VecElement& hexas = this->_topology->getHexas();
            const sofa::helper::vector<unsigned int> &hexaModif = (static_cast< const HexahedraRemoved *> (*iter))->getArray();

            helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();

            for(unsigned int i=0; i<hexaModif.size(); ++i)
            {
                const unsigned int hexaId = hexaModif[i];

                Real mass = this->_elementTotalMass.getValue()[hexaId] * (Real) 0.125;

                for(int w=0; w<8; ++w)
                    particleMasses[ hexas[hexaId][w] ] -= mass;
            }

            this->_particleMasses.endEdit();

            if( this->_useLumpedMass.getValue() )
            {
                helper::vector<Coord>&	lumpedMasses = *this->_lumpedMasses.beginEdit();

                for(unsigned int i=0; i<hexaModif.size(); ++i)
                {
                    const unsigned int hexaId = hexaModif[i];
                    const ElementMass& mass = this->_elementMasses.getValue()[hexaId];

                    for(int w=0; w<8; ++w)
                    {
                        for(int j=0; j<8*3; ++j)
                        {
                            lumpedMasses[ hexas[hexaId][w] ][0] -= mass[w*3  ][j];
                            lumpedMasses[ hexas[hexaId][w] ][1] -= mass[w*3+1][j];
                            lumpedMasses[ hexas[hexaId][w] ][2] -= mass[w*3+2][j];
                        }
                    }
                }

                this->_lumpedMasses.endEdit();
            }

            // handle hexa events
            this->hexahedronInfo.handleTopologyEvents(iter, next_iter);
            this->_elementMasses.handleTopologyEvents(iter, next_iter);
            this->_elementTotalMass.handleTopologyEvents(iter, next_iter);
        }
        break;

        // the structure of the coarse hexas has changed
        // subtract the contributions of removed voxels
        case ((core::componentmodel::topology::TopologyChangeType) component::topology::MultilevelModification::MULTILEVEL_MODIFICATION) :
        {
            const VecElement& hexas = this->_topology->getHexas();
            const component::topology::MultilevelModification *modEvent = static_cast< const component::topology::MultilevelModification *> (*iter);
            const sofa::helper::vector<unsigned int> &hexaModif = modEvent->getArray();

            const int level = _multilevelTopology->getLevel();
            const int coarseNodeSize = (1 << level);

            // subtract the contributions of removed voxels from element matrices
            helper::vector<typename HexahedralFEMForceFieldT::HexahedronInformation>& hexahedronInf = *this->hexahedronInfo.beginEdit();
            helper::vector<ElementMass>& elementMasses = *this->_elementMasses.beginEdit();
            helper::vector<Real>& elementTotalMass = *this->_elementTotalMass.beginEdit();

            helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();
            helper::vector<Coord>&	lumpedMasses = *this->_lumpedMasses.beginEdit();

            for(unsigned int i=0; i<hexaModif.size(); ++i)
            {
                const unsigned int hexaId = hexaModif[i];

                ElementStiffness K;
                ElementMass M;
                Real totalMass = (Real) 0.0;

                const std::list<Vec3i>& removedVoxels = modEvent->getRemovedVoxels(hexaId);
                for(std::list<Vec3i>::const_iterator it = removedVoxels.begin(); it != removedVoxels.end(); ++it)
                {
                    const MultilevelHexahedronSetTopologyContainer::Vec3i& voxelId = *it;

                    const Mat88& H = __H[(voxelId[0]%coarseNodeSize) + coarseNodeSize * ((voxelId[1]%coarseNodeSize) + coarseNodeSize * (voxelId[2]%coarseNodeSize))];

                    // add the fine element into the coarse
                    this->addHtfineHtoCoarse(H, _material.K, K);
                    this->addHtfineHtoCoarse(H, _material.M, M);
                    totalMass += _material.mass;
                }

                hexahedronInf[hexaId].stiffness -= K;
                elementMasses[hexaId] -= M;
                elementTotalMass[hexaId] -= totalMass;

                const Real partMass = totalMass * (Real) 0.125;

                for(int w=0; w<8; ++w)
                    particleMasses[ hexas[hexaId][w] ] -= partMass;

                if( this->_useLumpedMass.getValue() )
                {
                    for(int w=0; w<8; ++w)
                    {
                        for(int j=0; j<8*3; ++j)
                        {
                            lumpedMasses[ hexas[hexaId][w] ][0] -= M[w*3  ][j];
                            lumpedMasses[ hexas[hexaId][w] ][1] -= M[w*3+1][j];
                            lumpedMasses[ hexas[hexaId][w] ][2] -= M[w*3+2][j];
                        }
                    }
                }
            }

            this->_elementTotalMass.endEdit();
            this->_elementMasses.endEdit();
            this->hexahedronInfo.endEdit();
            this->_particleMasses.endEdit();
            this->_lumpedMasses.endEdit();
        }
        break;
        default:
        {
        }
        break;
        }
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::initLarge( const int i)

{
    const VecCoord *X0=this->mstate->getX0();

    Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = (*X0)[this->_topology->getHexa(i)[w]];

    // compute initial configuration in order to compute corotationnal deformations
    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    typename HexahedralFEMForceFieldT::Transformation R_0_1;
    computeRotationLarge(R_0_1, horizontal, vertical);

    helper::vector<typename HexahedralFEMForceFieldT::HexahedronInformation>& hexahedronInf = *this->hexahedronInfo.beginEdit();
    helper::vector<ElementMass>& elementMasses = *this->_elementMasses.beginEdit();
    helper::vector<Real>& elementTotalMass = *this->_elementTotalMass.beginEdit();

    for(int w=0; w<8; ++w)
        hexahedronInf[i].rotatedInitialElements[w] = R_0_1*nodes[w];

    /// compute mechanichal matrices (mass and stiffness) by condensating from finest level
    computeMechanicalMatricesByCondensation( hexahedronInf[i].stiffness,
            elementMasses[i],
            elementTotalMass[i], i);

    this->_elementTotalMass.endEdit();
    this->_elementMasses.endEdit();
    this->hexahedronInfo.endEdit();
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::initPolar( const int i)

{
    const VecCoord *X0=this->mstate->getX0();

    Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = (*X0)[this->_topology->getHexa(i)[j]];

    typename HexahedralFEMForceFieldT::Transformation R_0_1; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationPolar( R_0_1, nodes );


    helper::vector<typename HexahedralFEMForceFieldT::HexahedronInformation>& hexahedronInf = *this->hexahedronInfo.beginEdit();
    helper::vector<ElementMass>& elementMasses = *this->_elementMasses.beginEdit();
    helper::vector<Real>& elementTotalMass = *this->_elementTotalMass.beginEdit();

    for(int w=0; w<8; ++w)
        hexahedronInf[i].rotatedInitialElements[w] = R_0_1*nodes[w];

    /// compute mechanichal matrices (mass and stiffness) by condensating from finest level
    computeMechanicalMatricesByCondensation( hexahedronInf[i].stiffness,
            elementMasses[i],
            elementTotalMass[i], i);

    this->_elementTotalMass.endEdit();
    this->_elementMasses.endEdit();
    this->hexahedronInfo.endEdit();
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation( ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const int elementIndex)
{
    K.clear();
    M.clear();
    totalMass = (Real) 0.0;

    helper::vector<unsigned int>	fineElements;
    _multilevelTopology->getHexaChildren(elementIndex, fineElements);

    const int level = _multilevelTopology->getLevel();
    const int coarseNodeSize = (1 << level);

    const bool recursive = false;

    if(recursive)
    {
        helper::vector<bool> fineChildren((int) coarseNodeSize*coarseNodeSize*coarseNodeSize, false);

        // condensate recursively each 8 children (if they exist)
        for(unsigned int i=0; i<fineElements.size(); ++i)
        {
            const MultilevelHexahedronSetTopologyContainer::Vec3i& voxelId = _multilevelTopology->getHexaIdxInFineRegularGrid(fineElements[i]);

            const int I = voxelId[0]%coarseNodeSize;
            const int J = voxelId[1]%coarseNodeSize;
            const int K = voxelId[2]%coarseNodeSize;

            fineChildren[I + coarseNodeSize * (J + K * coarseNodeSize)] = true;
        }

        computeMechanicalMatricesByCondensation_Recursive( K, M, totalMass, _material.K, _material.M, _material.mass, level, fineChildren);
    }
    else
    {
        for(unsigned int i=0; i<fineElements.size(); ++i)
        {
            const MultilevelHexahedronSetTopologyContainer::Vec3i& voxelId = _multilevelTopology->getHexaIdxInFineRegularGrid(fineElements[i]);

            const Mat88& H = __H[(voxelId[0]%coarseNodeSize) + coarseNodeSize * ((voxelId[1]%coarseNodeSize) + coarseNodeSize * (voxelId[2]%coarseNodeSize))];

            // add the fine element into the coarse
            this->addHtfineHtoCoarse(H, _material.K, K);
            this->addHtfineHtoCoarse(H, _material.M, M);
            totalMass += _material.mass;
        }
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation_Recursive( ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const ElementStiffness &K_fine,
        const ElementMass &M_fine,
        const Real& mass_fine,
        const int level,
        const helper::vector<bool>& fineChildren) const
{
    if(level==0)
    {
        K = K_fine;
        M = M_fine;
        totalMass = mass_fine;
    }
    else
    {
        const int nextLevel = level - 1;
        const int nodeSize = 1 << level;
        const int nextNodeSize = 1 << nextLevel;

        for(int i=0; i<8; ++i)
        {
            ElementStiffness K_tmp;
            ElementMass M_tmp;
            Real mass_tmp = 0;
            helper::vector<bool> children_tmp((int) nextNodeSize*nextNodeSize*nextNodeSize, false);

            int i_min(0), i_max(nodeSize), j_min(0), j_max(nodeSize), k_min(0), k_max(nodeSize);
            switch(i)
            {
            case 0:
                i_max = nextNodeSize;
                j_max = nextNodeSize;
                k_max = nextNodeSize;
                break;
            case 1:
                i_min = nextNodeSize;
                j_max = nextNodeSize;
                k_max = nextNodeSize;
                break;
            case 2:
                i_min = nextNodeSize;
                j_min = nextNodeSize;
                k_max = nextNodeSize;
                break;
            case 3:
                i_max = nextNodeSize;
                j_min = nextNodeSize;
                k_max = nextNodeSize;
                break;
            case 4:
                i_max = nextNodeSize;
                j_max = nextNodeSize;
                k_min = nextNodeSize;
                break;
            case 5:
                i_min = nextNodeSize;
                j_max = nextNodeSize;
                k_min = nextNodeSize;
                break;
            case 6:
                i_min = nextNodeSize;
                j_min = nextNodeSize;
                k_min = nextNodeSize;
                break;
            case 7:
                i_max = nextNodeSize;
                j_min = nextNodeSize;
                k_min = nextNodeSize;
                break;
            }

            bool allChildrenEmpty = true;

            for(int I=i_min; I<i_max; ++I)
                for(int J=j_min; J<j_max; ++J)
                    for(int K=k_min; K<k_max; ++K)
                    {
                        if(fineChildren[I + nodeSize * (J + K * nodeSize)])
                        {
                            children_tmp[(I-i_min) + nextNodeSize * ((J-j_min) + (K-k_min) * nextNodeSize)] = true;
                            allChildrenEmpty = false;
                        }
                    }

            if(!allChildrenEmpty)
            {
                computeMechanicalMatricesByCondensation_Recursive( K_tmp, M_tmp, mass_tmp, K_fine, M_fine, mass_fine, nextLevel, children_tmp);

                this->addHtfineHtoCoarse(_H[i], K_tmp, K);
                this->addHtfineHtoCoarse(_H[i], M_tmp, M);
                totalMass += mass_tmp;
            }
        }
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::addHtfineHtoCoarse(const Mat88& H,
        const ElementStiffness& fine,
        ElementStiffness& coarse) const
{
    ElementStiffness A;

    for(int i=0; i<24; ++i)
        for(int j=0; j<24; ++j)
            for(int k=0; k<24; ++k)
            {
                if(j%3==k%3)
                    A[i][j] += fine[i][k] * H[k/3][j/3];		// A = fine * H
            }

    for(int i=0; i<24; i++)
        for(int j=0; j<24; ++j)
            for(int k=0; k<24; ++k)
            {
                if(i%3==k%3)
                    coarse[i][j] += H[k/3][i/3] * A[k][j];		// HtfineH = Ht * A
            }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::subtractHtfineHfromCoarse(const Mat88& H,
        const ElementStiffness& fine,
        ElementStiffness& coarse) const
{
    ElementStiffness A;

    for(int i=0; i<24; ++i)
        for(int j=0; j<24; ++j)
            for(int k=0; k<24; ++k)
            {
                if(j%3==k%3)
                    A[i][j] += fine[i][k] * H[k/3][j/3];		// A = fine * H
            }

    for(int i=0; i<24; i++)
        for(int j=0; j<24; ++j)
            for(int k=0; k<24; ++k)
            {
                if(i%3==k%3)
                    coarse[i][j] -= H[k/3][i/3] * A[k][j];		// HtfineH = Ht * A
            }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeHtfineH(const Mat88& H,
        const ElementStiffness& fine,
        ElementStiffness& HtfineH) const
{
    HtfineH.clear();
    ElementStiffness A;

    for(int i=0; i<24; ++i)
        for(int j=0; j<24; ++j)
            for(int k=0; k<24; ++k)
            {
                if(j%3==k%3)
                    A[i][j] += fine[i][k] * H[k/3][j/3];		// A = fine * H
            }

    for(int i=0; i<24; i++)
        for(int j=0; j<24; ++j)
            for(int k=0; k<24; ++k)
            {
                if(i%3==k%3)
                    HtfineH[i][j] += H[k/3][i/3] * A[k][j];		// HtfineH = Ht * A
            }
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
