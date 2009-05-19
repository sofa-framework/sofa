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
    , _ignoreHexaModifEvent(core::objectmodel::Base::initData(&_ignoreHexaModifEvent, true, "ignoreHexaModifEvent", "Ignore hexa modif event."))
{
}

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

    // handle hexa events
    this->hexahedronInfo.handleTopologyEvents(itBegin,itEnd);
    this->_elementMasses.handleTopologyEvents(itBegin,itEnd);
    this->_elementTotalMass.handleTopologyEvents(itBegin,itEnd);

    for(std::list<const TopologyChange *>::const_iterator iter = itBegin;
        iter != itEnd; ++iter)
    {
        switch((*iter)->getChangeType())
        {
            // for added elements:
            // init element
            // add particle masses and lumped masses of adjacent particles
        case core::componentmodel::topology::HEXAHEDRAADDED:
        {
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
        }
        break;
        case ((core::componentmodel::topology::TopologyChangeType) component::topology::MultilevelModification::MULTILEVEL_MODIFICATION) :
        {

            //cerr<<"MULTILEVEL_MODIFICATION "<<(static_cast< const MultilevelModification *> (*iter))->getArray()<<endl;

            if(! _ignoreHexaModifEvent.getValue() ) // recompute everything
            {
                const VecElement& hexas = this->_topology->getHexas();
                const sofa::helper::vector<unsigned int> &hexaModif = (static_cast< const MultilevelModification *> (*iter))->getArray();

                // reinit modified elements: remove and add
                {
                    helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();

                    for(unsigned int i=0; i<hexaModif.size(); ++i)
                    {
                        const unsigned int hexaId = hexaModif[i];

                        Real mass = this->_elementTotalMass.getValue()[hexaId] * (Real) 0.125;

                        for(int w=0; w<8; ++w)
                            particleMasses[ hexas[hexaId][w] ] -= mass;
                    }

                    this->_particleMasses.endEdit();
                }

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
                {
                    helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();

                    for(unsigned int i=0; i<hexaModif.size(); ++i)
                    {
                        const unsigned int hexaId = hexaModif[i];

                        Real mass = this->_elementTotalMass.getValue()[hexaId] * (Real) 0.125;

                        for(int w=0; w<8; ++w)
                            particleMasses[ hexas[hexaId][w] ] += mass;
                    }

                    this->_particleMasses.endEdit();
                }

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
            else // newmethod -> just delete the fine hexa (only for removing and not adding of fine hexa)
            {
                // TODO: this is not working !!!

                //	const sofa::helper::vector<unsigned int> &fineRemovedHexa = (static_cast< const MultilevelModification *> (*iter))->getRemovedFineHexahedraArray();
                //	for(unsigned int i=0; i<fineRemovedHexa.size(); ++i)
                //	{
                //		removeFineHexa( fineRemovedHexa[i] );
                //	}
            }
        }
        break;
        default:
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

    hexahedronInf[i].stiffness = ElementStiffness(0);
    elementMasses[i] = ElementMass(0);
    elementTotalMass[i] = 0;

    //computeMechanicalMatricesByCondensation( hexahedronInf[i].stiffness,
    //								elementMasses[i],
    //								elementTotalMass[i], i);

    computeMechanicalMatricesByCondensationDirectlyFromFinestToCoarse( hexahedronInf[i].stiffness,
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

    hexahedronInf[i].stiffness = ElementStiffness(0);
    elementMasses[i] = ElementMass(0);
    elementTotalMass[i] = 0;

    // compute mechanichal matrices (mass and stiffness) by condensating from finest level
    //computeMechanicalMatricesByCondensation( hexahedronInf[i].stiffness,
    //								elementMasses[i],
    //								elementTotalMass[i], i);

    computeMechanicalMatricesByCondensationDirectlyFromFinestToCoarse( hexahedronInf[i].stiffness,
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
    const int level = _multilevelTopology->getLevel();
    const int coarseNodeSize = (1 << level);

    const VecCoord *X0=this->mstate->getX0();
    Vec<8,Coord> nodesCoarse;
    for(int w=0; w<8; ++w)
        nodesCoarse[w] = (*X0)[this->_topology->getHexa(elementIndex)[w]];

    Vec<8,Coord> nodesFine;
    for(int w=0; w<8; ++w)
        nodesFine[w] = (nodesCoarse[w] - nodesCoarse[0]) / coarseNodeSize;

    MaterialStiffness	C_fine;
    ElementStiffness	K_fine;
    ElementMass			M_fine;
    Real				mass_fine;

    HexahedralFEMForceField<T>::computeMaterialStiffness(C_fine, this->f_youngModulus.getValue(), this->f_poissonRatio.getValue());
    HexahedralFEMForceField<T>::computeElementStiffness(K_fine, C_fine, nodesFine);
    HexahedralFEMForceFieldAndMass<T>::computeElementMass(M_fine, mass_fine, nodesFine);

    // condensate recursively each 8 children (if they exist)

    helper::vector<unsigned int>	fineElements;
    _multilevelTopology->getHexaChildren(elementIndex, fineElements);

    helper::vector<bool> fineChildren((int) coarseNodeSize*coarseNodeSize*coarseNodeSize, false);

    for(unsigned int i=0; i<fineElements.size(); ++i)
    {
        const MultilevelHexahedronSetTopologyContainer::Vec3i& voxelId = _multilevelTopology->getHexaIdxInFineRegularGrid(fineElements[i]);

        const int I = voxelId[0]%coarseNodeSize;
        const int J = voxelId[1]%coarseNodeSize;
        const int K = voxelId[2]%coarseNodeSize;

        fineChildren[I + coarseNodeSize * (J + K * coarseNodeSize)] = true;
    }

    computeMechanicalMatricesByCondensation( K, M, totalMass, K_fine, M_fine, mass_fine, level, fineChildren);
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation( ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const ElementStiffness &K_fine,
        const ElementMass &M_fine,
        const Real& mass_fine,
        const int level,
        const helper::vector<bool>& fineChildren)
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
                computeMechanicalMatricesByCondensation( K_tmp, M_tmp, mass_tmp, K_fine, M_fine, mass_fine, nextLevel, children_tmp);
                this->addFineToCoarse(K_tmp, _H[i], K);
                this->addFineToCoarse(M_tmp, _H[i], M);
                totalMass += mass_tmp;
            }
        }
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensationDirectlyFromFinestToCoarse( ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const int elementIndex)
{
    const int level = _multilevelTopology->getLevel();
    const int coarseNodeSize = (1 << level);

    helper::vector<unsigned int>	fineElements;
    _multilevelTopology->getHexaChildren(elementIndex, fineElements);

    for(unsigned int i=0; i<fineElements.size(); ++i)
    {
        const MultilevelHexahedronSetTopologyContainer::Vec3i& voxelId = _multilevelTopology->getHexaIdxInFineRegularGrid(fineElements[i]);

        AFine actualfine;
        //actualfine.coarseHexaIdx = elementIndex;
        actualfine.mass = _material.mass;

        Mat88& H = __H[(voxelId[0]%coarseNodeSize) + coarseNodeSize * ((voxelId[1]%coarseNodeSize) + coarseNodeSize * (voxelId[2]%coarseNodeSize))];

        // add the fine element into the coarse
        this->computeHtfineHAndAddFineToCoarse(_material.K, H, K, actualfine.HtKH);
        this->computeHtfineHAndAddFineToCoarse(_material.M, H, M, actualfine.HtMH);
        totalMass += _material.mass;

        // save the fine values
        _mapFineToCorse[voxelId] = actualfine;
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::addFineToCoarse( const ElementStiffness& fine,
        const Mat88& H,
        ElementStiffness& coarse)
{
    ElementStiffness A;

    for(int i=0; i<24; ++i)
        for(int j=0; j<24; ++j)
        {
            A[i][j] = (Real) ((j%3==0) ? fine[i][0] * H[0][j/3] : 0.0);

            for(int k=1; k<24; ++k)
                A[i][j] += (Real) ((j%3==k%3) ? fine[i][k] * H[k/3][j/3] : 0.0);
        }

    for(int i=0; i<24; i++)
        for(int j=0; j<24; ++j)
        {
            for(int k=0; k<24; ++k)
                coarse[i][j] += (Real) ((i%3==k%3) ? H[k/3][i/3] * A[k][j] : 0.0);   // H[index] transposed
        }
}


template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeHtfineHAndAddFineToCoarse(const ElementStiffness& fine,
        const Mat88& H,
        ElementStiffness& coarse,
        ElementStiffness& HtfineH
                                                                                  )
{
    ElementStiffness A;

    for(int i=0; i<24; ++i)
        for(int j=0; j<24; ++j)
        {
            HtfineH[i][j] = 0.0;
            A[i][j] = (Real) ((j%3==0) ? fine[i][0] * H[0][j/3] : 0.0);

            for(int k=1; k<24; ++k)
                A[i][j] += (Real) ((j%3==k%3) ? fine[i][k] * H[k/3][j/3] : 0.0);
        }

    for(int i=0; i<24; i++)
        for(int j=0; j<24; ++j)
        {
            for(int k=0; k<24; ++k)
            {
                Real tmp = (Real) ((i%3==k%3) ? H[k/3][i/3] * A[k][j] : 0.0);
                HtfineH[i][j] += tmp;
                coarse[i][j] += tmp;
            }
        }
}


template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::removeFineHexa( const unsigned int fineIdx )
{
    const Vec3i& voxelId = _multilevelTopology->getHexaIdxInFineRegularGrid(fineIdx);
    AFine& actualfine = _mapFineToCorse[voxelId];
    unsigned coarseIdx = _multilevelTopology->getHexaParent(fineIdx);


    //cerr<<"removeFineHexa "<<fineIdx<< " in "<<coarseIdx<<" / "<<(*this->hexahedronInfo.beginEdit()).size()<<endl;
    //if( coarseIdx>=(*this->hexahedronInfo.beginEdit()).size() ) return; // the coarse hexa doesn't exist anymore

    helper::vector<typename HexahedralFEMForceFieldT::HexahedronInformation>& hexahedronInf = *this->hexahedronInfo.beginEdit();
    helper::vector<ElementMass>& elementMasses = *this->_elementMasses.beginEdit();
    helper::vector<Real>& elementTotalMass = *this->_elementTotalMass.beginEdit();


    ElementStiffness& K_coarse = hexahedronInf[coarseIdx].stiffness;
    ElementMass& M_coarse = elementMasses[coarseIdx];
    Real& mass_coarse = elementTotalMass[coarseIdx];

    K_coarse -= actualfine.HtKH;
    M_coarse -= actualfine.HtMH;
    mass_coarse -= actualfine.mass;


    const VecElement& hexas = this->_topology->getHexas();
    const Element& actualcoarsehexa = hexas[ coarseIdx ];

    helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();
    Real particleMass = (Real)actualfine.mass * (Real)0.125;
    for(int w=0; w<8; ++w)
        particleMasses[ actualcoarsehexa[w] ] -= particleMass;

    if( this->_useLumpedMass.getValue() )
    {
        helper::vector<Coord>&	lumpedMasses = *this->_lumpedMasses.beginEdit();

        for(int w=0; w<8; ++w)
        {
            lumpedMasses[ actualcoarsehexa[w] ][0] = 0;
            lumpedMasses[ actualcoarsehexa[w] ][1] = 0;
            lumpedMasses[ actualcoarsehexa[w] ][2] = 0;
            for(int j=0; j<8*3; ++j)
            {
                lumpedMasses[ actualcoarsehexa[w] ][0] += M_coarse[w*3  ][j];
                lumpedMasses[ actualcoarsehexa[w] ][1] += M_coarse[w*3+1][j];
                lumpedMasses[ actualcoarsehexa[w] ][2] += M_coarse[w*3+2][j];
            }
        }
        this->_lumpedMasses.endEdit();
    }


    this->_elementTotalMass.endEdit();
    this->_elementMasses.endEdit();
    this->hexahedronInfo.endEdit();
    this->_particleMasses.endEdit();
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
