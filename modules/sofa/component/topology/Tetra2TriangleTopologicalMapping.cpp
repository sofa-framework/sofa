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
#include <sofa/component/topology/Tetra2TriangleTopologicalMapping.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>

#include <sofa/component/topology/TriangleSetTopology.h>
#include <sofa/component/topology/TetrahedronSetTopology.h>

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology;
using namespace sofa::core::componentmodel::topology;

/// Input Topology
typedef BaseMeshTopology In;
/// Output Topology
typedef BaseMeshTopology Out;

SOFA_DECL_CLASS(Tetra2TriangleTopologicalMapping)

// Register in the Factory
int Tetra2TriangleTopologicalMappingClass = core::RegisterObject("Special case of mapping where TetrahedronSetTopology is converted to TriangleSetTopology")
        .add< Tetra2TriangleTopologicalMapping >()

        ;

// Implementation

Tetra2TriangleTopologicalMapping::Tetra2TriangleTopologicalMapping(In* from, Out* to)
    :
    fromModel(from), toModel(to),
    object1(initData(&object1, std::string("../.."), "object1", "First object to map")),
    object2(initData(&object2, std::string(".."), "object2", "Second object to map"))
{
}

Tetra2TriangleTopologicalMapping::~Tetra2TriangleTopologicalMapping()
{
}

In* Tetra2TriangleTopologicalMapping::getFromModel()
{
    return this->fromModel;
}

Out* Tetra2TriangleTopologicalMapping::getToModel()
{
    return this->toModel;
}

objectmodel::BaseObject* Tetra2TriangleTopologicalMapping::getFrom()
{
    return this->fromModel;
}

objectmodel::BaseObject* Tetra2TriangleTopologicalMapping::getTo()
{
    return this->toModel;
}

void Tetra2TriangleTopologicalMapping::setModels(In* from, Out* to)
{
    this->fromModel = from;
    this->toModel = to;
}

void Tetra2TriangleTopologicalMapping::init()
{
    //std::cout << "INFO_print : init Tetra2TriangleTopologicalMapping" << std::endl;

    // INITIALISATION of TRIANGULAR mesh from TETRAHEDRAL mesh :


    if (fromModel)
    {

        std::cout << "INFO_print : Tetra2TriangleTopologicalMapping - from = tetra" << std::endl;

        if (toModel)
        {

            std::cout << "INFO_print : Tetra2TriangleTopologicalMapping - to = triangle" << std::endl;

            TriangleSetTopologyModifier *to_tstm;
            toModel->getContext()->get(to_tstm);

            const sofa::helper::vector<Triangle> &triangleArray=fromModel->getTriangles();


            unsigned int nb_visible_triangles = 0;

            Loc2GlobVec.clear();
            Glob2LocMap.clear();

            for (unsigned int i=0; i<triangleArray.size(); ++i)
            {

                if (fromModel->getTetraTriangleShell(i).size()==1)
                {

                    to_tstm->addTriangle(triangleArray[i]);

                    Loc2GlobVec.push_back(i);
                    Glob2LocMap[i]=Loc2GlobVec.size()-1;

                    nb_visible_triangles+=1;
                }
            }

            TriangleSetTopologyAlgorithms<Vec3Types> *triangleAlg;
            toModel->getContext()->get(triangleAlg);
            triangleAlg->notifyEndingEvent();

        }

    }
}

unsigned int Tetra2TriangleTopologicalMapping::getFromIndex(unsigned int ind)
{

    if(fromModel->getTetraTriangleShell(ind).size()==1)
    {
        return fromModel->getTetraTriangleShell(ind)[0];
    }
    else
    {
        return -1;
    }
}

void Tetra2TriangleTopologicalMapping::updateTopologicalMapping()
{

    // INITIALISATION of TRIANGULAR mesh from TETRAHEDRAL mesh :

    if (fromModel)
    {

        TriangleSetTopologyModifier *to_tstm;
        toModel->getContext()->get(to_tstm);

        if (toModel)
        {

            std::list<const TopologyChange *>::const_iterator itBegin=fromModel->firstChange();
            std::list<const TopologyChange *>::const_iterator itEnd=fromModel->lastChange();

            while( itBegin != itEnd )
            {
                TopologyChangeType changeType = (*itBegin)->getChangeType();

                switch( changeType )
                {

                case core::componentmodel::topology::ENDING_EVENT:
                {
                    //std::cout << "INFO_print : Tetra2TriangleTopologicalMapping - ENDING_EVENT" << std::endl;
                    TriangleSetTopologyAlgorithms<Vec3Types> *triangleAlg;
                    toModel->getContext()->get(triangleAlg);
                    triangleAlg->notifyEndingEvent();
                    break;
                }

                case core::componentmodel::topology::TRIANGLESREMOVED:
                {
                    //std::cout << "INFO_print : Tetra2TriangleTopologicalMapping - TRIANGLESREMOVED" << std::endl;

                    int last;
                    int ind_last;

                    last= fromModel->getNbTriangles() - 1;

                    const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TrianglesRemoved *>( *itBegin ) )->getArray();

                    unsigned int ind_tmp;

                    unsigned int ind_real_last;
                    ind_last=toModel->getNbTriangles();

                    for (unsigned int i = 0; i <tab.size(); ++i)
                    {
                        unsigned int k = tab[i];
                        unsigned int ind_k;

                        std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                        if(iter_1 != Glob2LocMap.end())
                        {

                            ind_last = ind_last - 1;

                            ind_k = Glob2LocMap[k];
                            ind_real_last = ind_k;

                            std::map<unsigned int, unsigned int>::iterator iter_2 = Glob2LocMap.find(last);
                            if(iter_2 != Glob2LocMap.end())
                            {

                                ind_real_last = Glob2LocMap[last];

                                if((int) k != last)
                                {

                                    Glob2LocMap.erase(Glob2LocMap.find(k));
                                    Glob2LocMap[k] = ind_real_last;

                                    Glob2LocMap.erase(Glob2LocMap.find(last));
                                    Glob2LocMap[last] = ind_k;

                                    ind_tmp = Loc2GlobVec[ind_real_last];
                                    Loc2GlobVec[ind_real_last] = Loc2GlobVec[ind_k];
                                    Loc2GlobVec[ind_k] = ind_tmp;
                                }
                            }

                            if((int) ind_k != ind_last)
                            {

                                Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_last]));
                                Glob2LocMap[Loc2GlobVec[ind_last]] = ind_k;

                                Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_k]));
                                Glob2LocMap[Loc2GlobVec[ind_k]] = ind_last;

                                ind_tmp = Loc2GlobVec[ind_k];
                                Loc2GlobVec[ind_k] = Loc2GlobVec[ind_last];
                                Loc2GlobVec[ind_last] = ind_tmp;

                            }

                            Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[Loc2GlobVec.size() - 1]));
                            Loc2GlobVec.resize( Loc2GlobVec.size() - 1 );

                            sofa::helper::vector< unsigned int > triangles_to_remove;
                            triangles_to_remove.push_back(ind_k);

                            TriangleSetTopologyAlgorithms<Vec3Types> *triangleAlg;
                            toModel->getContext()->get(triangleAlg);
                            triangleAlg->removeTriangles(triangles_to_remove, true, false);

                        }
                        else
                        {

                            std::cout << "INFO_print : Tetra2TriangleTopologicalMapping - Glob2LocMap should have the visible triangle " << tab[i] << std::endl;
                            std::cout << "INFO_print : Tetra2TriangleTopologicalMapping - nb triangles = " << ind_last << std::endl;
                        }

                        --last;
                    }

                    break;
                }

                case core::componentmodel::topology::TETRAHEDRAREMOVED:
                {
                    //std::cout << "INFO_print : Tetra2TriangleTopologicalMapping - TETRAHEDRAREMOVED" << std::endl;

                    if (fromModel)
                    {

                        const sofa::helper::vector<Tetrahedron> &tetrahedronArray=fromModel->getTetras();

                        const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraRemoved *>( *itBegin ) )->getArray();

                        sofa::helper::vector< Triangle > triangles_to_create;
                        sofa::helper::vector< unsigned int > trianglesIndexList;
                        int nb_elems = toModel->getNbTriangles();

                        for (unsigned int i = 0; i < tab.size(); ++i)
                        {

                            for (unsigned int j = 0; j < 4; ++j)
                            {
                                unsigned int k = (fromModel->getTriangleTetraShell(tab[i]))[j];

                                if (fromModel->getTetraTriangleShell(k).size()==1)   // remove as visible the triangle indexed by k
                                {

                                    // do nothing

                                }
                                else   // fromModel->getTetraTriangleShell(k).size()==2 // add as visible the triangle indexed by k
                                {

                                    unsigned int ind_test;
                                    if(tab[i] == fromModel->getTetraTriangleShell(k)[0])
                                    {

                                        ind_test = fromModel->getTetraTriangleShell(k)[1];

                                    }
                                    else   // tab[i] == fromModel->getTetraTriangleShell(k)[1]
                                    {

                                        ind_test = fromModel->getTetraTriangleShell(k)[0];
                                    }

                                    bool is_present = false;
                                    unsigned int k0 = 0;
                                    while((!is_present) && k0 < i)
                                    {
                                        is_present = (ind_test == tab[k0]);
                                        k0+=1;
                                    }
                                    if(!is_present)
                                    {

                                        Triangle t;

                                        const Tetrahedron &te=tetrahedronArray[ind_test];
                                        int h = fromModel->getTriangleIndexInTetrahedron(fromModel->getTriangleTetraShell(ind_test),k);

                                        if (h%2)
                                        {
                                            t[0]=(int)(te[(h+1)%4]); t[1]=(int)(te[(h+2)%4]); t[2]=(int)(te[(h+3)%4]);
                                        }
                                        else
                                        {
                                            t[0]=(int)(te[(h+1)%4]); t[2]=(int)(te[(h+2)%4]); t[1]=(int)(te[(h+3)%4]);
                                        }

                                        // sort t such that t[0] is the smallest one
                                        while ((t[0]>t[1]) || (t[0]>t[2]))
                                        {
                                            int val=t[0]; t[0]=t[1]; t[1]=t[2]; t[2]=val;
                                        }

                                        triangles_to_create.push_back(t);
                                        trianglesIndexList.push_back(nb_elems);
                                        nb_elems+=1;

                                        Loc2GlobVec.push_back(k);
                                        std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                                        if(iter_1 != Glob2LocMap.end() )
                                        {
                                            std::cout << "INFO_print : Tetra2TriangleTopologicalMapping - fail to add triangle " << k << "which already exists" << std::endl;
                                            Glob2LocMap.erase(Glob2LocMap.find(k));
                                        }
                                        Glob2LocMap[k]=Loc2GlobVec.size()-1;
                                    }
                                }
                            }
                        }

                        to_tstm->addTrianglesProcess(triangles_to_create) ;
                        to_tstm->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList) ;

                    }

                    break;
                }

                case core::componentmodel::topology::POINTSREMOVED:
                {
                    //std::cout << "INFO_print : Tetra2TriangleTopologicalMapping - POINTSREMOVED" << std::endl;

                    const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

                    sofa::helper::vector<unsigned int> indices;

                    for(unsigned int i = 0; i < tab.size(); ++i)
                    {

                        indices.push_back(tab[i]);
                    }

                    sofa::helper::vector<unsigned int>& tab_indices = indices;

                    to_tstm->removePointsWarning(tab_indices, false);

                    toModel->propagateTopologicalChanges();
                    to_tstm->removePointsProcess(tab_indices, false);

                    break;
                }

                case core::componentmodel::topology::POINTSRENUMBERING:
                {
                    //std::cout << "INFO_print : Hexa2QuadTopologicalMapping - POINTSREMOVED" << std::endl;

                    const sofa::helper::vector<unsigned int> &tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
                    const sofa::helper::vector<unsigned int> &inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    sofa::helper::vector<unsigned int> indices;
                    sofa::helper::vector<unsigned int> inv_indices;

                    for(unsigned int i = 0; i < tab.size(); ++i)
                    {

                        //std::cout << "INFO_print : Hexa2QuadTopologicalMapping - point = " << tab[i] << std::endl;
                        indices.push_back(tab[i]);
                        inv_indices.push_back(inv_tab[i]);
                    }

                    sofa::helper::vector<unsigned int>& tab_indices = indices;
                    sofa::helper::vector<unsigned int>& inv_tab_indices = inv_indices;

                    to_tstm->renumberPointsWarning(tab_indices, inv_tab_indices, false);
                    toModel->propagateTopologicalChanges();
                    to_tstm->renumberPointsProcess(tab_indices, inv_tab_indices, false);

                    break;
                }
                default:
                    // Ignore events that are not Triangle  related.
                    break;
                };

                ++itBegin;
            }
            toModel->propagateTopologicalChanges();
        }
    }

    return;
}


} // namespace topology

} // namespace component

} // namespace sofa

