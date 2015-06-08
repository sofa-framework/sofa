#include "Topology/generic/mapImpl/mapCPH.hpp"

namespace CGoGN
{

MapCPH::MapCPH() :
    GenericMap(),
    m_curLevel(0u),
    m_maxLevel(0u),
    m_edgeIdCount(0u),
    m_faceIdCount(0u)
{
    m_dartLevel = m_attribs[DART].addAttribute< unsigned int >("dartLevel");
    m_edgeId  = m_attribs[DART].addAttribute< unsigned int >("edgeId");
    m_faceId =  m_attribs[DART].addAttribute< unsigned int >("faceId");
    m_nextLevelCell = m_attribs[DART].addAttribute<unsigned int>("nextLevelCell") ;
}

MapCPH::~MapCPH()
{
    m_attribs[DART].removeAttribute< unsigned int >("dartLevel");
    m_attribs[DART].removeAttribute< unsigned int >("edgeId");
    m_attribs[DART].removeAttribute< unsigned int >("faceId");
    m_attribs[DART].removeAttribute< unsigned int >("nextLevelCell");
}

void MapCPH::clear(bool removeAttrib)
{
    GenericMap::clear(removeAttrib) ;
    if (removeAttrib)
    {
        m_attribs[DART].removeAttribute< unsigned int >("dartLevel");
        m_attribs[DART].removeAttribute< unsigned int >("edgeId");
        m_attribs[DART].removeAttribute< unsigned int >("faceId");
        m_attribs[DART].removeAttribute< unsigned int >("nextLevelCell");
        m_permutation.clear();
        m_permutation_inv.clear();
        m_involution.clear();
        m_dartLevel = m_attribs[DART].addAttribute< unsigned int >("dartLevel");
        m_edgeId  = m_attribs[DART].addAttribute< unsigned int >("edgeId");
        m_faceId =  m_attribs[DART].addAttribute< unsigned int >("faceId");
        m_nextLevelCell = m_attribs[DART].addAttribute<unsigned int>("nextLevelCell") ;
    }
}

void MapCPH::initImplicitProperties()
{
    initEdgeId();
    initFaceId();
    for(unsigned int d = m_attribs[DART].begin(), end = m_attribs[DART].end() ; d != end; m_attribs[DART].next(d))
        m_nextLevelCell->operator[](d) = EMBNULL ;
}

void MapCPH::addInvolution()
{
    std::stringstream sstm;
    sstm << "involution_" << m_involution.size();
    m_involution.push_back(addRelation(sstm.str()));
}

void MapCPH::addPermutation()
{
    std::stringstream sstm;
    sstm << "permutation_" << m_permutation.size();
    m_permutation.push_back(addRelation(sstm.str()));
    std::stringstream sstm2;
    sstm2 << "permutation_inv_" << m_permutation_inv.size();
    m_permutation_inv.push_back(addRelation(sstm2.str()));
}

void MapCPH::removeLastInvolutionPtr()
{
    m_involution.pop_back();
}

void MapCPH::compactTopo()
{
    std::vector<unsigned int> oldnew;
    m_attribs[DART].compact(oldnew);

    for (unsigned int i = m_attribs[DART].begin(); i != m_attribs[DART].end(); m_attribs[DART].next(i))
    {
        for (unsigned int j = 0; j < m_permutation.size(); ++j)
        {
            Dart d = (*m_permutation[j])[i];
            if (d.index != oldnew[d.index])
                (*m_permutation[j])[i] = Dart(oldnew[d.index]);
        }
        for (unsigned int j = 0; j < m_permutation_inv.size(); ++j)
        {
            Dart d = (*m_permutation_inv[j])[i];
            if (d.index != oldnew[d.index])
                (*m_permutation_inv[j])[i] = Dart(oldnew[d.index]);
        }
        for (unsigned int j = 0; j < m_involution.size(); ++j)
        {
            Dart d = (*m_involution[j])[i];
            if (d.index != oldnew[d.index])
                (*m_involution[j])[i] = Dart(oldnew[d.index]);
        }
    }
}

bool MapCPH::saveMapBin(const std::string &filename) const
{
    // TODO
    return false;
}

bool MapCPH::loadMapBin(const std::string &filename)
{
    // TODO
    return false;
}

bool MapCPH::copyFrom(const GenericMap &map)
{
    // TODO
    return false;
}

void MapCPH::restore_topo_shortcuts()
{

}

void MapCPH::setNextLevelCell(Dart d, unsigned int emb)
{
    this->m_nextLevelCell->operator [](this->dartIndex(d)) = emb;
}


Dart MapCPH::newDart()
{
    const Dart d = GenericMap::newDart() ;

    for (unsigned int i = 0; i < m_permutation.size(); ++i)
        (*m_permutation[i])[d.index] = d ;
    for (unsigned int i = 0; i < m_permutation_inv.size(); ++i)
        (*m_permutation_inv[i])[d.index] = d ;
    for (unsigned int i = 0; i < m_involution.size(); ++i)
        (*m_involution[i])[d.index] = d ;

    setDartLevel(d, this->getCurrentLevel());
    setNextLevelCell(d, EMBNULL);
    return d;
}





} // namespace CGoGN


