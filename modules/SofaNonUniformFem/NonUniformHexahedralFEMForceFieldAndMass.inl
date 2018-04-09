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
#ifndef SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_INL
#define SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_INL

#include <SofaNonUniformFem/NonUniformHexahedralFEMForceFieldAndMass.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaNonUniformFem/MultilevelHexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/core/objectmodel/Base.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using std::set;

template <class DataTypes>
NonUniformHexahedralFEMForceFieldAndMass<DataTypes>::NonUniformHexahedralFEMForceFieldAndMass()
    : HexahedralFEMForceFieldAndMassT()
    , _bRecursive(core::objectmodel::Base::initData(&_bRecursive, false, "recursive", "Use recursive matrix computation"))
    , useMBK(initData(&useMBK, true, "useMBK", "compute MBK and use it in addMBKdx, instead of using addDForce and addMDx."))
{}

template <class DataTypes>
void NonUniformHexahedralFEMForceFieldAndMass<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();

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
    hexahedronInf.resize(this->_topology->getNbHexahedra());
    this->hexahedronInfo.endEdit();

    helper::vector<ElementMass>& elementMasses = *this->_elementMasses.beginEdit();
    elementMasses.resize( this->_topology->getNbHexahedra() );
    this->_elementMasses.endEdit();

    helper::vector<Real>& elementTotalMass = *this->_elementTotalMass.beginEdit();
    elementTotalMass.resize( this->_topology->getNbHexahedra() );
    this->_elementTotalMass.endEdit();

    const int level = _multilevelTopology->getLevel();
    const int coarseNodeSize = (1 << level);

    if (!this->_topology->getNbHexahedra())
    {
        serr << "Topology is empty !" << sendl;
        return;
    }

    const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    defaulttype::Vec<8,Coord> nodesCoarse;
    for(int w=0; w<8; ++w)
        nodesCoarse[w] = (X0)[this->_topology->getHexahedron(0)[w]];

    defaulttype::Vec<8,Coord> nodesFine;
    for(int w=0; w<8; ++w)
        nodesFine[w] = (nodesCoarse[w] - nodesCoarse[0]) / coarseNodeSize;

    HexahedralFEMForceField<T>::computeMaterialStiffness(_material.C, this->f_youngModulus.getValue(), this->f_poissonRatio.getValue());
    HexahedralFEMForceField<T>::computeElementStiffness(_material.K, _material.C, nodesFine);

    //	HexahedralFEMForceFieldAndMass<T>::computeElementMass(_material.M, _material.mass, nodesFine);

    const Real volume = (nodesFine[1]-nodesFine[0]).norm()*(nodesFine[3]-nodesFine[0]).norm()*(nodesFine[4]-nodesFine[0]).norm();
    _material.mass = volume * this->_density.getValue();

    Mat88	M;

    for(unsigned int u=0; u<8; ++u)
        for(unsigned int v=0; v<8; ++v)
        {
            M[u][v] = (Real) (_material.mass/216.0);

            const unsigned int q = u ^ v;
            if ((q&1)==((q&2) >> 1)) M[u][v] *= (Real) 2.0;
            if (!((q&2) >> 1))       M[u][v] *= (Real) 2.0;
            if (!(q>>2))             M[u][v] *= (Real) 2.0;

            for(unsigned int k=0; k<3; ++k)
                for(unsigned int j=0; j<3; ++j)
                    _material.M[3*u+k][3*v+j] = (Real)((k%3==j%3)?M[u][v]:0.0);
        }

    _H.resize(level+1);

    for(int currLevel = level; currLevel>=0; --currLevel)
    {
        const int currCoarseNodeSize = 1 << currLevel;
        const float fineNodeSize = 1.0f / (float) currCoarseNodeSize;
        int idx=0;

        _H[currLevel].resize(currCoarseNodeSize * currCoarseNodeSize * currCoarseNodeSize);

        for(int k=0; k<currCoarseNodeSize; ++k)
            for(int j=0; j<currCoarseNodeSize; ++j)
                for(int i=0; i<currCoarseNodeSize; ++i, ++idx)
                {
                    for(int w=0; w<8; ++w) // childNodeId
                    {
                        const float x = fineNodeSize * (i + ((w&1)!=((w&2) >> 1)) );
                        const float y = fineNodeSize * (j + ((w&2) >> 1) );
                        const float z = fineNodeSize * (k + (w>>2));

                        // entree dans la matrice pour le sommet w
                        _H[currLevel][idx][w][0] = (1-x) * (1-y) * (1-z);
                        _H[currLevel][idx][w][1] =   (x) * (1-y) * (1-z);
                        _H[currLevel][idx][w][3] = (1-x) *   (y) * (1-z);
                        _H[currLevel][idx][w][2] =   (x) *   (y) * (1-z);
                        _H[currLevel][idx][w][4] = (1-x) * (1-y) *   (z);
                        _H[currLevel][idx][w][5] =   (x) * (1-y) *   (z);
                        _H[currLevel][idx][w][7] = (1-x) *   (y) *   (z);
                        _H[currLevel][idx][w][6] =   (x) *   (y) *   (z);
                    }
                }
    }


    switch(this->method)
    {
    case HexahedralFEMForceFieldT::LARGE:
    {
        for (int i=0; i<this->_topology->getNbHexahedra(); ++i)
            initLarge(i);
    }
    break;
    case HexahedralFEMForceFieldT::POLAR:
    {
        for(int i=0; i<this->_topology->getNbHexahedra(); ++i)
            initPolar(i);
    }
    break;
    }

    HexahedralFEMForceFieldAndMassT::computeParticleMasses();
    HexahedralFEMForceFieldAndMassT::computeLumpedMasses();
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::handleTopologyChange(core::topology::Topology* t)
{
    if(t != this->_topology)
        return;
#ifdef TODOTOPO
    std::list<const TopologyChange *>::const_iterator itBegin=this->_topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=this->_topology->endChange();

    // handle point events
    //this->_particleMasses.handleTopologyEvents(itBegin,itEnd);

    //if( this->_useLumpedMass.getValue() )
    //    this->_lumpedMasses.handleTopologyEvents(itBegin,itEnd);

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
        case core::topology::HEXAHEDRAADDED:
        {
            // handle hexa events
            this->hexahedronInfo.handleTopologyEvents(iter, next_iter);
            this->_elementMasses.handleTopologyEvents(iter, next_iter);
            this->_elementTotalMass.handleTopologyEvents(iter, next_iter);

            handleHexaAdded(*(static_cast< const HexahedraAdded *> (*iter)));
        }
        break;

        // for removed elements:
        // subtract particle masses and lumped masses of adjacent particles
        case core::topology::HEXAHEDRAREMOVED:
        {
            handleHexaRemoved(*(static_cast< const HexahedraRemoved *> (*iter)));

            // handle hexa events
            this->hexahedronInfo.handleTopologyEvents(iter, next_iter);
            this->_elementMasses.handleTopologyEvents(iter, next_iter);
            this->_elementTotalMass.handleTopologyEvents(iter, next_iter);
        }
        break;

        // the structure of the coarse hexahedra has changed
        // subtract the contributions of removed voxels
        case ((core::topology::TopologyChangeType) component::topology::MultilevelModification::MULTILEVEL_MODIFICATION) :
        {
            handleMultilevelModif(*(static_cast< const MultilevelModification *> (*iter)));
        }
        break;
        default:
        {
        }
        break;
        }
    }
#endif
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::handleHexaAdded(const core::topology::HexahedraAdded& hexaAddedEvent)
{
    const sofa::helper::vector<unsigned int> &hexaModif = hexaAddedEvent.hexahedronIndexArray;

    dmsg_info() << "HEXAHEDRAADDED hexaId: " << hexaModif ;
    const VecElement& hexahedra = this->_topology->getHexahedra();

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
            particleMasses[ hexahedra[hexaId][w] ] += mass;
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
                    lumpedMasses[ hexahedra[hexaId][w] ][0] += mass[w*3  ][j];
                    lumpedMasses[ hexahedra[hexaId][w] ][1] += mass[w*3+1][j];
                    lumpedMasses[ hexahedra[hexaId][w] ][2] += mass[w*3+2][j];
                }
            }
        }

        this->_lumpedMasses.endEdit();
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::handleHexaRemoved(const core::topology::HexahedraRemoved& hexaRemovedEvent)
{
    const sofa::helper::vector<unsigned int> &hexaModif = hexaRemovedEvent.getArray();

    dmsg_info() << "HEXAHEDRAREMOVED hexaId: " << hexaModif ;

    const VecElement& hexahedra = this->_topology->getHexahedra();
    helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();

    for(unsigned int i=0; i<hexaModif.size(); ++i)
    {
        const unsigned int hexaId = hexaModif[i];

        Real mass = this->_elementTotalMass.getValue()[hexaId] * (Real) 0.125;

        for(int w=0; w<8; ++w)
            particleMasses[ hexahedra[hexaId][w] ] -= mass;
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
                    lumpedMasses[ hexahedra[hexaId][w] ][0] -= mass[w*3  ][j];
                    lumpedMasses[ hexahedra[hexaId][w] ][1] -= mass[w*3+1][j];
                    lumpedMasses[ hexahedra[hexaId][w] ][2] -= mass[w*3+2][j];
                }
            }
        }

        this->_lumpedMasses.endEdit();
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::handleMultilevelModif(const component::topology::MultilevelModification& modEvent)
{
    const sofa::helper::vector<unsigned int> &hexaModif = modEvent.getArray();

    dmsg_info() << "MULTILEVEL_MODIFICATION hexaId: " << hexaModif ;

    const VecElement& hexahedra = this->_topology->getHexahedra();

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

        const std::list<Vec3i>& removedVoxels = modEvent.getRemovedVoxels(hexaId);
        for(std::list<Vec3i>::const_iterator it = removedVoxels.begin(); it != removedVoxels.end(); ++it)
        {
            const Vec3i& voxelId = *it;

            const Mat88& H = _H[level][(voxelId[0]%coarseNodeSize) + coarseNodeSize * ((voxelId[1]%coarseNodeSize) + coarseNodeSize * (voxelId[2]%coarseNodeSize))];

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
            particleMasses[ hexahedra[hexaId][w] ] -= partMass;

        if( this->_useLumpedMass.getValue() )
        {
            for(int w=0; w<8; ++w)
            {
                for(int j=0; j<8*3; ++j)
                {
                    lumpedMasses[ hexahedra[hexaId][w] ][0] -= M[w*3  ][j];
                    lumpedMasses[ hexahedra[hexaId][w] ][1] -= M[w*3+1][j];
                    lumpedMasses[ hexahedra[hexaId][w] ][2] -= M[w*3+2][j];
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

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::initLarge( const int i)

{
    const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    defaulttype::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = (X0)[this->_topology->getHexahedron(i)[w]];

    // compute initial configuration in order to compute corotationnal deformations
    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    typename HexahedralFEMForceFieldT::Transformation R_0_1;
    this->computeRotationLarge(R_0_1, horizontal, vertical);

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
    const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    defaulttype::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = (X0)[this->_topology->getHexahedron(i)[j]];

    typename HexahedralFEMForceFieldT::Transformation R_0_1; // Rotation matrix (deformed and displaced Hexahedron/world)
    this->computeRotationPolar( R_0_1, nodes );


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
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation(
    ElementStiffness &K,
    ElementMass &M,
    Real& totalMass,
    const int elementIndex)
{
    K.clear();
    M.clear();
    totalMass = (Real) 0.0;

    const std::set<Vec3i>& voxels = _multilevelTopology->getHexaVoxels(elementIndex);

    const int level = _multilevelTopology->getLevel();

    if(_bRecursive.getValue())
    {
        const unsigned int coarseNodeSize = (1 << level);
        std::set<unsigned int> fineChildren;

        // condensate recursively each 8 children (if they exist)
        for(std::set<Vec3i>::const_iterator it = voxels.begin(); it != voxels.end(); ++it)
        {
            const Vec3i& voxelId = *it;

            fineChildren.insert(ijk2octree(voxelId[0]%coarseNodeSize, voxelId[1]%coarseNodeSize, voxelId[2]%coarseNodeSize));
        }

        computeMechanicalMatricesByCondensation_Recursive( K, M, totalMass, _material.K, _material.M, _material.mass, level, 0, fineChildren);
    }
    else
    {
        computeMechanicalMatricesByCondensation_IntervalAnalysis( K, M, totalMass, _material.K, _material.M, _material.mass, level, voxels);
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation_IntervalAnalysis(
    ElementStiffness &K,
    ElementMass &M,
    Real& totalMass,
    const ElementStiffness &K_fine,
    const ElementMass &M_fine,
    const Real& mass_fine,
    const unsigned int level,
    const std::set<Vec3i>& voxels) const
{
    if(voxels.size() == (unsigned int) (1 << (3*level))) // full element
    {
        K = K_fine;
        K *= (Real) (1 << level);

        M = M_fine;
        M *= (Real) voxels.size();

        totalMass = voxels.size() * mass_fine;
    }
    else if(voxels.size() < 8) // almost empty element
    {
        const int coarseNodeSize = (1 << level);

        for(std::set<Vec3i>::const_iterator it = voxels.begin(); it != voxels.end(); ++it)
        {
            const Vec3i& voxelId = *it;

            const Mat88& H = _H[level][(voxelId[0]%coarseNodeSize) + coarseNodeSize * ((voxelId[1]%coarseNodeSize) + coarseNodeSize * (voxelId[2]%coarseNodeSize))];

            // add the fine element into the coarse
            this->addHtfineHtoCoarse(H, K_fine, K);
            this->addHtfineHtoCoarse(H, M_fine, M);
            totalMass += mass_fine;
        }
    }
    else
    {
        unsigned int intervalLevels = 0;

        for(unsigned int l=1; l<=voxels.size(); l <<= 3)
            ++intervalLevels;

        typedef std::pair< unsigned int, unsigned int > t_int_interval;
        typedef std::list< t_int_interval > t_interval_list;

        helper::vector< t_interval_list >	intervals(intervalLevels);

        std::set<unsigned int> fineChildren;
        const unsigned int coarseNodeSize = (1 << level);

        for(std::set<Vec3i>::const_iterator it = voxels.begin(); it != voxels.end(); ++it)
        {
            const Vec3i& voxelId = *it;

            fineChildren.insert(ijk2octree(voxelId[0]%coarseNodeSize, voxelId[1]%coarseNodeSize, voxelId[2]%coarseNodeSize));
        }

        const std::set<unsigned int>::const_iterator itBegin = fineChildren.begin();
        const std::set<unsigned int>::const_iterator itEnd = fineChildren.end();

        std::set<unsigned int>::const_iterator it = itBegin;
        std::set<unsigned int>::const_iterator it_first = itBegin;
        std::set<unsigned int>::const_iterator it_last = itBegin;

        while(it != itEnd)
        {
            if((*it)-(*it_last) > 1) // found a gap
            {
                const unsigned int first = *it_first;
                const unsigned int length = (*it_last)-(*it_first)+1;

                unsigned int currLevel = 0;
                for(unsigned int l=8; l<length; l <<= 3)
                    ++currLevel;
                intervals[currLevel].push_back(t_int_interval(first, length));

                it_first = it;
            }

            it_last = it++;

            if(it == itEnd)
            {
                const unsigned int first = *it_first;
                const unsigned int length = (*it_last)-(*it_first)+1;

                unsigned int currLevel = 0;
                for(unsigned int l=8; l<length; l <<= 3)
                    ++currLevel;
                intervals[currLevel].push_back(t_int_interval(first, length));
            }
        }

        for(unsigned int intervalLevel=intervals.size()-1; intervalLevel>0; --intervalLevel)
        {
            const unsigned int intervalLength = 1 << (3*intervalLevel);

            for(t_interval_list::iterator iter = intervals[intervalLevel].begin(); iter != intervals[intervalLevel].end(); /* ++iter */)
            {
                t_int_interval&	currInterval = *iter;

                unsigned int& first = currInterval.first;
                unsigned int& length = currInterval.second;

                const unsigned int prefixLength = (intervalLength - (first % intervalLength)) % intervalLength;

                if(prefixLength > 0)
                {
                    unsigned int currLevel = 0;
                    for(unsigned int l=8; l<prefixLength; l <<= 3)
                        ++currLevel;
                    intervals[currLevel].push_back(t_int_interval(first, prefixLength));

                    first += prefixLength;
                    length -= prefixLength;
                }

                const unsigned int postfixLength = length % intervalLength;

                if(postfixLength > 0)
                {
                    unsigned int currLevel = 0;
                    for(unsigned int l=8; l<postfixLength; l <<= 3)
                        ++currLevel;
                    intervals[currLevel].push_back(t_int_interval(first+length-postfixLength, postfixLength));

                    length -= postfixLength;
                }

                if(length == 0)
                {
                    t_interval_list::iterator iter_tmp = iter++;
                    intervals[intervalLevel].erase(iter_tmp);
                }
                else
                {
                    ++iter;
                }
            }
        }

        ElementStiffness K_tmp;
        ElementMass M_tmp;
        Real mass_tmp = 0;

        for(unsigned int intervalLevel=0; intervalLevel<intervalLevels; ++intervalLevel)
        {
            if(! intervals[intervalLevel].empty())
            {
                const unsigned int intervalLength = 1 << (3*intervalLevel);

                K_tmp = K_fine;
                K_tmp *= (Real) (1 << intervalLevel);

                M_tmp = M_fine;
                M_tmp *= (Real) intervalLength;

                mass_tmp = intervalLength * mass_fine;

                const unsigned int diffLevel = level-intervalLevel;
                const unsigned int diffSize = 1 << diffLevel;

                for(t_interval_list::iterator iter = intervals[intervalLevel].begin(); iter != intervals[intervalLevel].end(); ++iter)
                {
                    const t_int_interval&	currInterval = *iter;

                    const unsigned int totalLength = currInterval.second;

                    for(unsigned int i=0; i * intervalLength < totalLength; ++i)
                    {
                        const unsigned int first = currInterval.first + i * intervalLength;

                        const Vec3i voxelId = octree2voxel(first >> (3 * intervalLevel));

                        const Mat88& H = _H[diffLevel][voxelId[0] + diffSize * (voxelId[1] + diffSize * voxelId[2])];

                        this->addHtfineHtoCoarse(H, K_tmp, K);
                        this->addHtfineHtoCoarse(H, M_tmp, M);
                        totalMass += mass_tmp;
                    }
                }
            }
        }
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation_Direct(
    ElementStiffness &K,
    ElementMass &M,
    Real& totalMass,
    const ElementStiffness &K_fine,
    const ElementMass &M_fine,
    const Real& mass_fine,
    const unsigned int level,
    const std::set<Vec3i>& voxels) const
{
    const int coarseNodeSize = (1 << level);

    for(std::set<Vec3i>::const_iterator it = voxels.begin(); it != voxels.end(); ++it)
    {
        const Vec3i& voxelId = *it;

        const Mat88& H = _H[level][(voxelId[0]%coarseNodeSize) + coarseNodeSize * ((voxelId[1]%coarseNodeSize) + coarseNodeSize * (voxelId[2]%coarseNodeSize))];

        // add the fine element into the coarse
        this->addHtfineHtoCoarse(H, K_fine, K);
        this->addHtfineHtoCoarse(H, M_fine, M);
        totalMass += mass_fine;
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation_Recursive(
    ElementStiffness &K,
    ElementMass &M,
    Real& totalMass,
    const ElementStiffness &K_fine,
    const ElementMass &M_fine,
    const Real& mass_fine,
    const unsigned int level,
    const unsigned int startIdx,
    const std::set<unsigned int>& fineChildren) const
{
    if(level == 0)
    {
        K = K_fine;
        M = M_fine;
        totalMass = mass_fine;
    }
    else
    {
        const unsigned int nextLevel = level - 1;
        const unsigned int delta = 1 << ( 3 * nextLevel);

        for(int child=0; child<8; ++child)
        {
            unsigned int minChildIdx = startIdx + child * delta;
            unsigned int maxChildIdx = minChildIdx + delta;

            std::set<unsigned int>::const_iterator itBegin = fineChildren.lower_bound(minChildIdx);
            std::set<unsigned int>::const_iterator itEnd = fineChildren.lower_bound(maxChildIdx);

            if(itBegin != itEnd) // there are some non empty voxels in the child area
            {
                ElementStiffness K_tmp;
                ElementMass M_tmp;
                Real mass_tmp = 0;

                computeMechanicalMatricesByCondensation_Recursive( K_tmp, M_tmp, mass_tmp, K_fine, M_fine, mass_fine, nextLevel, minChildIdx, fineChildren);

                this->addHtfineHtoCoarse(_H[1][child], K_tmp, K);
                this->addHtfineHtoCoarse(_H[1][child], M_tmp, M);
                totalMass += mass_tmp;

            }
        }
    }
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::addHtfineHtoCoarse(
    const Mat88& H,
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
void NonUniformHexahedralFEMForceFieldAndMass<T>::subtractHtfineHfromCoarse(
    const Mat88& H,
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
void NonUniformHexahedralFEMForceFieldAndMass<T>::computeHtfineH(
    const Mat88& H,
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


template<class T>
int NonUniformHexahedralFEMForceFieldAndMass<T>::ijk2octree(
    const int i,
    const int j,
    const int k) const
{
    unsigned int I = i;
    unsigned int J = j;
    unsigned int K = k;

    unsigned int octreeIdx = 0;

    unsigned int shift = 0;

    while( (I + J + K) > 0)
    {
        octreeIdx += ((I & 1) + (((J & 1) + ((K & 1) << 1)) << 1)) << shift;

        I >>= 1;
        J >>= 1;
        K >>= 1;

        shift += 3;
    }

    return octreeIdx;
}

template<class T>
void NonUniformHexahedralFEMForceFieldAndMass<T>::octree2ijk(
    const int octreeIdx,
    int &i,
    int &j,
    int &k) const
{
    i = j = k = 0;

    unsigned int idx = octreeIdx;

    unsigned int shift = 0;

    while( idx > 0 )
    {
        i +=  (idx & 1) << shift;
        j += ((idx & 2) >> 1) << shift;
        k += ((idx & 4) >> 2) << shift;

        idx >>= 3;

        ++shift;
    }
}

template<class T>
typename NonUniformHexahedralFEMForceFieldAndMass<T>::Vec3i NonUniformHexahedralFEMForceFieldAndMass<T>::octree2voxel(const int octreeIdx) const
{
    Vec3i voxelId;
    octree2ijk(octreeIdx, voxelId[0], voxelId[1], voxelId[2]);

    return voxelId;
}



template <class DataTypes>
void NonUniformHexahedralFEMForceFieldAndMass<DataTypes>::addMBKdx(const core::MechanicalParams* mparams, core::MultiVecDerivId dfId)
{
    Real mFactor=(Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());
    Real kFactor=(Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    helper::ReadAccessor < DataVecDeriv > dx = *mparams->readDx(this->mstate);
    helper::WriteAccessor< DataVecDeriv > df = *dfId[this->mstate.get(mparams)].write();
    const VecElement& hexahedra = this->_topology->getHexahedra();
    const helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = this->hexahedronInfo.getValue();

    // WARNING !  MBK is used not only in the equation matrix, but also in the right-hand term, with different coefficients.
    // We want to correct it only in the equation matrix, and we assume that mFactor<=0 correspond to the right-hand term.
    if( !useMBK.getValue() || mFactor<=0 )
    {
        // Do not compute the weighted sum of the matrices explicitly.
        matrixIsDirty = true;

        for(unsigned int i=0; i<hexahedra.size(); ++i)
        {
            Displacement rdx, rdf;
            const Mat33& Re = hexahedronInf[i].rotation;
            Mat33 Ret = Re.transposed();

            for(int k=0 ; k<8 ; ++k )
            {
                Deriv x_2 = Ret * dx[hexahedra[i][k]];

                int indice = k*3;
                for(int j=0 ; j<3 ; ++j )
                    rdx[indice+j] = x_2[j];
            }

            rdf = this->_elementMasses.getValue()[i] * rdx * mFactor - hexahedronInf[i].stiffness * rdx * kFactor;

            for(unsigned int w=0; w<8; ++w)
                df[hexahedra[i][w]] += Re * Deriv( rdf[w*3],  rdf[w*3+1],   rdf[w*3+2]  );
        }
        return;
    }

    // mFactor > 0 , so we assume this is a product of the equation matrix done by an implicit solver
    if( matrixIsDirty )
    {
        // Compute the matrix
        this->mbkMatrix.resize(hexahedra.size());

        for ( unsigned int e = 0; e < hexahedra.size(); ++e )
        {
            const ElementMass &Me = this->_elementMasses.getValue() [e];
            //const Element hexa = hexahedra[e];
            const ElementStiffness &Ke = hexahedronInf[e].stiffness;
            const Mat33& Re = hexahedronInf[e].rotation;
            Mat33 Ret = Re.transposed();
            ElementStiffness MBKe(0.);

            for ( unsigned n1 = 0; n1 < 8; n1++ )
            {
                for (unsigned n2=0; n2<8; n2++)
                {
                    // add M to matrix
                    Mat33 tmp( Deriv ( Me[3*n1+0][3*n2+0]*mFactor, Me[3*n1+0][3*n2+1]*mFactor, Me[3*n1+0][3*n2+2]*mFactor ),
                            Deriv ( Me[3*n1+1][3*n2+0]*mFactor, Me[3*n1+1][3*n2+1]*mFactor, Me[3*n1+1][3*n2+2]*mFactor ),
                            Deriv ( Me[3*n1+2][3*n2+0]*mFactor, Me[3*n1+2][3*n2+1]*mFactor, Me[3*n1+2][3*n2+2]*mFactor )
                             );

                    // sub K to matrix
                    tmp -= Mat33(
                            Deriv ( Ke[3*n1+0][3*n2+0]*kFactor, Ke[3*n1+0][3*n2+1]*kFactor, Ke[3*n1+0][3*n2+2]*kFactor ),
                            Deriv ( Ke[3*n1+1][3*n2+0]*kFactor, Ke[3*n1+1][3*n2+1]*kFactor, Ke[3*n1+1][3*n2+2]*kFactor ),
                            Deriv ( Ke[3*n1+2][3*n2+0]*kFactor, Ke[3*n1+2][3*n2+1]*kFactor, Ke[3*n1+2][3*n2+2]*kFactor ) );

                    // rotate the matrix
                    tmp = Re * tmp * Ret;

                    // store the matrix
                    MBKe[3*n1+0][3*n2+0] = tmp[0][0], MBKe[3*n1+0][3*n2+1] = tmp[0][1], MBKe[3*n1+0][3*n2+2] = tmp[0][2];
                    MBKe[3*n1+1][3*n2+0] = tmp[1][0], MBKe[3*n1+1][3*n2+1] = tmp[1][1], MBKe[3*n1+1][3*n2+2] = tmp[1][2];
                    MBKe[3*n1+2][3*n2+0] = tmp[2][0], MBKe[3*n1+2][3*n2+1] = tmp[2][1], MBKe[3*n1+2][3*n2+2] = tmp[2][2];
                }
            }

            // Filter singular values to (hopefully) avoid conditionning problems
            computeCorrection(MBKe);

            // store
            this->mbkMatrix[e] = MBKe;
        }

        this->matrixIsDirty = false;
    }

    // MBK matrix product
    for ( unsigned int e = 0; e < hexahedra.size(); ++e )
    {
        Displacement X;

        for(int w=0; w<8; ++w)
        {
            const Deriv& x_2 = dx[hexahedra[e][w]];
            X[w*3] = x_2[0];
            X[w*3+1] = x_2[1];
            X[w*3+2] = x_2[2];
        }
        Displacement F;
        F = this->mbkMatrix[e] * X;

        for(int w=0; w<8; ++w)
            df[hexahedra[e][w]] += Deriv( F[w*3],  F[w*3+1],  F[w*3+2]  );
    }

}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
