/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/Triangle.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/gl/gl.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Triangle)

int TriangleMeshModelClass = core::RegisterObject("collision model using a triangular mesh, as described in MeshTopology")
        .add< TriangleMeshModel >()
        .addAlias("TriangleModel")
        .addAlias("Triangle")
        ;

int TriangleSetModelClass = core::RegisterObject("collision model using a triangular mesh, as described in TriangleSetTopology")
        .add< TriangleSetModel >()
        .addAlias("TriangleSet")
        ;

TriangleModel::TriangleModel()
    : mstate(NULL)
{
    triangles = &mytriangles;
}

TriangleMeshModel::TriangleMeshModel()
    : meshRevision(-1), mesh(NULL)
{
}

TriangleSetModel::TriangleSetModel()
    : mesh(NULL)
{
}

void TriangleModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
    elems.resize(size);
}

void TriangleModel::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::componentmodel::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    elems.clear();

    if (mstate==NULL)
    {
        std::cerr << "ERROR: TriangleModel requires a Vec3 Mechanical Model.\n";
        return;
    }
}

void TriangleMeshModel::init()
{
    TriangleModel::init();
    mesh = dynamic_cast<Topology *>(getContext()->getTopology());
    if (!mesh)
    {
        std::cerr << "ERROR: TriangleMeshModel requires a MeshTopology.\n";
        return;
    }
    updateFromTopology();
    updateNormals();
}

void TriangleSetModel::init()
{
    TriangleModel::init();
    Loc2GlobVec.clear();
    Glob2LocMap.clear();
    mesh = dynamic_cast<Topology *>(getContext()->getMainTopology());
    if (!mesh)
    {
        std::cerr << "ERROR: TriangleSetModel requires a TriangleSetTopology.\n";
        return;
    }
    sofa::core::componentmodel::topology::BaseTopology* bt = mesh;

    sofa::core::componentmodel::topology::TopologyContainer *container=bt->getTopologyContainer();

    sofa::component::topology::TetrahedronSetTopologyContainer *testc= dynamic_cast<sofa::component::topology::TetrahedronSetTopologyContainer *>(container);
    if (testc)
    {

        const sofa::helper::vector<sofa::component::topology::Triangle> &triangleArray=testc->getTriangleArray();
        //resize(triangleArray.size());
        unsigned int nb_visible_triangles = 0;

        for (unsigned int i=0; i<triangleArray.size(); ++i)
        {
            if (testc->getTetrahedronTriangleShell(i).size()==1)
            {
                mytriangles.push_back(triangleArray[i]);
                Loc2GlobVec.push_back(i);
                Glob2LocMap[i]=Loc2GlobVec.size()-1;

                nb_visible_triangles+=1;
            }
        }
        triangles = & mytriangles;
        resize(nb_visible_triangles);

    }
    else
    {
        //std::cerr << "ERROR: Topology is not a TetrahedronSetTopology.\n";
        sofa::component::topology::TriangleSetTopologyContainer *tstc= mesh->getTriangleSetTopologyContainer();

        const sofa::helper::vector<sofa::component::topology::Triangle> &ta=tstc->getTriangleArray();

        triangles = &ta;

        resize(ta.size());
        //Loc2GlobVec.resize(ta.size());

        //for(unsigned int i=0;i<ta.size();++i) {
        //Loc2GlobVec.push_back(i);
        //Glob2LocMap[i]=i;
        //}
    }

    updateFlags();
    updateNormals();
    updateFromTopology();
}

void TriangleModel::updateNormals()
{
    for (int i=0; i<size; i++)
    {
        Triangle t(this,i);
        const Vector3& pt1 = t.p1();
        const Vector3& pt2 = t.p2();
        const Vector3& pt3 = t.p3();

        t.n() = cross(pt2-pt1,pt3-pt1);
        t.n().normalize();
    }
}

void TriangleModel::updateFromTopology()
{
    needsUpdate = false;
}

void TriangleMeshModel::updateFromTopology()
{
    const unsigned npoints = mstate->getX()->size();
    const unsigned ntris = mesh->getNbTriangles();
    const unsigned nquads = mesh->getNbQuads();
    const unsigned newsize = ntris+2*nquads;
    needsUpdate=true;

    int revision = mesh->getRevision();
    if (revision == meshRevision && newsize==(unsigned)size)
    {
        needsUpdate=false;
        return;
    }

    resize(newsize);

    if (newsize == ntris)
    {
        // no need to copy the triangle indices
        triangles = & mesh->getTriangles();
    }
    else
    {
        triangles = &mytriangles;
        mytriangles.resize(newsize);
        int index = 0;
        for (unsigned i=0; i<ntris; i++)
        {
            topology::MeshTopology::Triangle idx = mesh->getTriangle(i);
            if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints)
            {
                std::cerr << "ERROR: Out of range index in triangle "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" ( total points="<<npoints<<")\n";
                if (idx[0] >= npoints) idx[0] = npoints-1;
                if (idx[1] >= npoints) idx[1] = npoints-1;
                if (idx[2] >= npoints) idx[2] = npoints-1;
            }
            mytriangles[index] = idx;
            ++index;
        }
        for (unsigned i=0; i<nquads; i++)
        {
            topology::MeshTopology::Quad idx = mesh->getQuad(i);
            if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints || idx[3] >= npoints)
            {
                std::cerr << "ERROR: Out of range index in quad "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" "<<idx[3]<<" ( total points="<<npoints<<")\n";
                if (idx[0] >= npoints) idx[0] = npoints-1;
                if (idx[1] >= npoints) idx[1] = npoints-1;
                if (idx[2] >= npoints) idx[2] = npoints-1;
                if (idx[3] >= npoints) idx[3] = npoints-1;
            }
            mytriangles[index][0] = idx[1];
            mytriangles[index][1] = idx[2];
            mytriangles[index][2] = idx[0];
            ++index;
            mytriangles[index][0] = idx[3];
            mytriangles[index][1] = idx[0];
            mytriangles[index][2] = idx[2];
            ++index;
        }
    }
    updateFlags();
    meshRevision = revision;
}

void TriangleModel::updateFlags(int ntri)
{
    if (ntri < 0) ntri = triangles->size();
    //VecCoord& x = *mstate->getX();
    //VecDeriv& v = *mstate->getV();
    vector<bool> pflags(mstate->getSize());
    std::set<std::pair<int,int> > eflags;
    for (unsigned i=0; i<triangles->size(); i++)
    {
        int f = 0;
        topology::Triangle t = (*triangles)[i];
        if (!pflags[t[0]])
        {
            f |= FLAG_P1;
            pflags[t[0]] = true;
        }
        if (!pflags[t[1]])
        {
            f |= FLAG_P2;
            pflags[t[1]] = true;
        }
        if (!pflags[t[2]])
        {
            f |= FLAG_P3;
            pflags[t[2]] = true;
        }
        if (eflags.insert( (t[0]<t[1])?std::make_pair(t[0],t[1]):std::make_pair(t[1],t[0]) ).second)
        {
            f |= FLAG_E12;
        }
        if (i < (unsigned)ntri && eflags.insert( (t[1]<t[2])?std::make_pair(t[1],t[2]):std::make_pair(t[2],t[1]) ).second) // don't use the diagonal edge of quads
        {
            f |= FLAG_E23;
        }
        if (eflags.insert( (t[2]<t[0])?std::make_pair(t[2],t[0]):std::make_pair(t[0],t[2]) ).second)
        {
            f |= FLAG_E31;
        }
        elems[i].flags = f;
    }
}

void TriangleSetModel::updateFromTopology()
{
}

void TriangleSetModel::handleTopologyChange()
{
    if (triangles != &mytriangles)
    {
        // We use the same triangle array as the topology -> only resize and recompute flags
        resize(mesh->getTriangleSetTopologyContainer()->getNumberOfTriangles());
        needsUpdate = true;
        updateFlags();
        updateNormals(); // not strictly necessary but useful if we display the model before the next collision iteration
        return;
    }
    sofa::core::componentmodel::topology::BaseTopology* bt = mesh;
    if (bt)
    {

        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=bt->firstChange();
        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=bt->lastChange();


        while( itBegin != itEnd )
        {
            core::componentmodel::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();
            // Since we are using identifier, we can safely use C type casts.

            sofa::core::componentmodel::topology::TopologyContainer *container=bt->getTopologyContainer();

            sofa::component::topology::TriangleSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::TriangleSetTopologyContainer *>(container);
            sofa::component::topology::TetrahedronSetTopologyContainer *testc= dynamic_cast<sofa::component::topology::TetrahedronSetTopologyContainer *>(container);

            if((changeType == core::componentmodel::topology::TETRAHEDRAREMOVED) || (((!testc) && changeType == core::componentmodel::topology::TRIANGLESREMOVED)))
            {

                unsigned int my_size = 0;
                if(testc)
                {
                    my_size = testc->getTetrahedronTriangleShellArray().size();
                }
                else
                {
                    if(tstc)
                    {
                        my_size = tstc->getTriangleArray().size();
                    }
                }

                // TEST 1
                for(unsigned int i_check= 0; i_check <Loc2GlobVec.size(); ++i_check)
                {

                    if(i_check!=Glob2LocMap[Loc2GlobVec[i_check]])
                    {
                        std::cout << "INFO_print : Coll - Glob2LocMap fails at i_check = "<< i_check << std::endl;
                    }

                }

                // TEST 2
                std::map<unsigned int, unsigned int>::iterator iter_check = Glob2LocMap.begin();
                while(iter_check != Glob2LocMap.end())
                {

                    unsigned int my_glob = iter_check->first;
                    //unsigned int my_loc = iter_check->second;
                    iter_check++;

                    if(my_glob!=Loc2GlobVec[Glob2LocMap[my_glob]])
                    {
                        std::cout << "INFO_print : Coll - Loc2GlobVec fails at my_glob = "<< my_glob << std::endl;
                    }

                    if(my_glob>=my_size)
                    {
                        std::cout << "INFO_print : Coll - Glob2LocMap gives too big my_glob = "<< my_glob << std::endl;
                    }
                }

                // TEST 3
                if(testc)
                {

                    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=tstc->getTriangleVertexShellArray();
                    unsigned int last = tvsa.size() -1;
                    //std::cout << "INFO_print : Coll - last point = "<< last << std::endl;

                    for(unsigned int j_check= 0; j_check < my_size; ++j_check)
                    {

                        if(testc->getTetrahedronTriangleShell(j_check).size()==1)
                        {

                            std::map<unsigned int, unsigned int>::iterator iter_j = Glob2LocMap.find(j_check);
                            if(iter_j == Glob2LocMap.end() )
                            {
                                std::cout << "INFO_print : Coll - Glob2LocMap should have the visible triangle j_check = "<< j_check << std::endl;
                            }
                            else
                            {

                                if(mytriangles[Glob2LocMap[j_check]][0] > last)
                                {
                                    std::cout << "INFO_print : Coll !!! POINT 0 OUT for j_check = " << j_check << " , triangle = "<< Glob2LocMap[j_check] << " , point = " << mytriangles[Glob2LocMap[j_check]][0] << std::endl;
                                }
                                if(mytriangles[Glob2LocMap[j_check]][1] > last)
                                {
                                    std::cout << "INFO_print : Coll !!! POINT 1 OUT for j_check = " << j_check << " , triangle = "<< Glob2LocMap[j_check] << " , point = " << mytriangles[Glob2LocMap[j_check]][1] << std::endl;
                                }
                                if(mytriangles[Glob2LocMap[j_check]][2] > last)
                                {
                                    std::cout << "INFO_print : Coll !!! POINT 2 OUT for j_check = " << j_check << " , triangle = "<< Glob2LocMap[j_check] << " , point = " << mytriangles[Glob2LocMap[j_check]][2] << std::endl;
                                }
                            }

                        }
                        else
                        {

                            std::map<unsigned int, unsigned int>::iterator iter_j = Glob2LocMap.find(j_check);
                            if(iter_j != Glob2LocMap.end() )
                            {
                                std::cout << "INFO_print : Coll - Glob2LocMap should NOT have the INvisible triangle j_check = "<< j_check << std::endl;
                            }
                        }
                    }
                }

                // TEST_END

            }

            ///
            switch( changeType )
            {

            case core::componentmodel::topology::TRIANGLESADDED:
            {

                TriangleInfo t;
                const sofa::component::topology::TrianglesAdded *ta=dynamic_cast< const sofa::component::topology::TrianglesAdded * >( *itBegin );
                for (unsigned int i=0; i<ta->getNbAddedTriangles(); ++i)
                {
                    mytriangles.push_back(ta->triangleArray[i]);

                    unsigned int ind_triangle = Loc2GlobVec.size();//(ta->triangleIndexArray)[i];

                    Loc2GlobVec.push_back(ind_triangle);
                    Glob2LocMap[ind_triangle]=ind_triangle;

                }
                resize( mytriangles.size());
                //Loc2GlobVec.resize( mytriangles.size());
                needsUpdate=true;


                //init();
                break;
            }

            case core::componentmodel::topology::TRIANGLESREMOVED:
            {

                unsigned int last;
                unsigned int ind_last;

                if(testc)
                {
                    last= (testc->getTetrahedronTriangleShellArray()).size() - 1;
                }
                else
                {
                    if(tstc)
                    {
                        last= (tstc->getTriangleArray()).size() - 1;
                    }
                    else
                    {
                        last= elems.size() -1;
                    }
                }

                //if(!testc){

                const sofa::helper::vector<unsigned int> &tab = ( dynamic_cast< const sofa::component::topology::TrianglesRemoved *>( *itBegin ) )->getArray();

                TriangleInfo tmp;
                topology::Triangle tmp2;
                unsigned int ind_tmp;

                unsigned int ind_real_last;

                for (unsigned int i = 0; i <tab.size(); ++i)
                {

                    unsigned int k = tab[i];
                    unsigned int ind_k;

                    std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                    if(iter_1 != Glob2LocMap.end() )
                    {

                        ind_k = Glob2LocMap[k];
                        ind_real_last = ind_k;

                        std::map<unsigned int, unsigned int>::iterator iter_2 = Glob2LocMap.find(last);
                        if(iter_2 != Glob2LocMap.end())
                        {

                            //std::cout << "FOUND" << std::endl;

                            ind_real_last = Glob2LocMap[last];

                            tmp = elems[ind_k];
                            elems[ind_k] = elems[ind_real_last];
                            elems[ind_real_last] = tmp;

                            tmp2 = mytriangles[ind_k];
                            mytriangles[ind_k] = mytriangles[ind_real_last];
                            mytriangles[ind_real_last] = tmp2;

                        }

                        ind_last = elems.size() - 1;

                        if(ind_real_last != ind_last)
                        {

                            tmp = elems[ind_real_last];
                            elems[ind_real_last] = elems[ind_last];
                            elems[ind_last] = tmp;

                            tmp2 = mytriangles[ind_real_last];
                            mytriangles[ind_real_last] = mytriangles[ind_last];
                            mytriangles[ind_last] = tmp2;

                            Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_last]));
                            Glob2LocMap[Loc2GlobVec[ind_last]] = ind_real_last;
                            Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_real_last]));
                            Glob2LocMap[Loc2GlobVec[ind_real_last]] = ind_last;

                            ind_tmp = Loc2GlobVec[ind_real_last];
                            Loc2GlobVec[ind_real_last] = Loc2GlobVec[ind_last];
                            Loc2GlobVec[ind_last] = ind_tmp;

                        }

                        mytriangles.resize( elems.size() - 1 );
                        resize( elems.size() - 1 );
                        Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_last])); // OK ??? // Loc2GlobVec[ind_last]
                        Loc2GlobVec.resize( Loc2GlobVec.size() - 1 );

                    }
                    else
                    {

                        std::map<unsigned int, unsigned int>::iterator iter_2 = Glob2LocMap.find(last);
                        if(iter_2 != Glob2LocMap.end())
                        {

                            ind_real_last = Glob2LocMap[last];

                            //const sofa::helper::vector<sofa::component::topology::Triangle> &triangleArray=tstc->getTriangleArray();
                            //mytriangles[ind_real_last][0]=triangleArray[k][0];
                            //mytriangles[ind_real_last][1]=triangleArray[k][1];
                            //mytriangles[ind_real_last][2]=triangleArray[k][2];

                            Loc2GlobVec[ind_real_last] = k;

                            Glob2LocMap.erase(Glob2LocMap.find(last));
                            Glob2LocMap[k] = ind_real_last;

                        }

                        std::cout << "INFO_print : Coll - Glob2LocMap should have the visible triangle " << tab[i] << std::endl;
                    }

                    --last;
                }

                needsUpdate=true;

                //init();
                //}

                break;
            }

            case core::componentmodel::topology::TETRAHEDRAREMOVED:
            {
                if (testc)
                {

                    //const sofa::helper::vector<sofa::component::topology::Triangle> &triangleArray=testc->getTriangleArray();
                    const sofa::helper::vector<sofa::component::topology::Tetrahedron> &tetrahedronArray=testc->getTetrahedronArray();

                    const sofa::helper::vector<unsigned int> &tab = ( dynamic_cast< const sofa::component::topology::TetrahedraRemoved *>( *itBegin ) )->getArray();

                    for (unsigned int i = 0; i < tab.size(); ++i)
                    {

                        for (unsigned int j = 0; j < 4; ++j)
                        {
                            //unsigned int k = tetrahedronArray[tab[i]][j];
                            unsigned int k = (testc->getTetrahedronTriangles(tab[i]))[j];

                            if (testc->getTetrahedronTriangleShell(k).size()==1)   // remove as visible the triangle indexed by k
                            {


                            }
                            else   // testc->getTetrahedronTriangleShell(k).size()==2 // add as visible the triangle indexed by k
                            {

                                unsigned int ind_test;
                                if(tab[i] == testc->getTetrahedronTriangleShell(k)[0])
                                {

                                    ind_test = testc->getTetrahedronTriangleShell(k)[1];

                                }
                                else   // tab[i] == testc->getTetrahedronTriangleShell(k)[1]
                                {

                                    ind_test = testc->getTetrahedronTriangleShell(k)[0];
                                }

                                bool is_present = false;
                                unsigned int k0 = 0;
                                while((!is_present) && k0 < i)  // i // tab.size()
                                {
                                    is_present = (ind_test == tab[k0]);
                                    k0+=1;
                                }
                                if(!is_present)
                                {
                                    topology::Triangle t;

                                    const sofa::component::topology::Tetrahedron &te=tetrahedronArray[ind_test];
                                    int h = testc->getTriangleIndexInTetrahedron(testc->getTetrahedronTriangles(ind_test),k);

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

                                    unsigned int prev_size = mytriangles.size();

                                    mytriangles.push_back(t);

                                    Loc2GlobVec.push_back(k);
                                    Glob2LocMap[k]=Loc2GlobVec.size()-1;

                                    mytriangles.resize( prev_size + 1 );
                                    resize( prev_size + 1 );
                                }
                            }
                        }
                    }
                }

                needsUpdate=true;


                //init();
                break;
            }
            case core::componentmodel::topology::POINTSREMOVED:
            {

                if (tstc)
                {

                    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=tstc->getTriangleVertexShellArray();
                    unsigned int last = tvsa.size() -1;
                    unsigned int i,j;
                    const sofa::helper::vector<unsigned int> tab = ( dynamic_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

                    sofa::helper::vector<unsigned int> lastIndexVec; //= tab;
                    for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
                    {

                        lastIndexVec.push_back(last - i_init);
                    }

                    for ( i = 0; i < tab.size(); ++i)
                    {
                        unsigned int i_next = i;
                        bool is_reached = false;
                        while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
                        {

                            i_next += 1 ;
                            is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
                        }

                        if(is_reached)
                        {

                            lastIndexVec[i_next] = lastIndexVec[i];

                        }

                        const sofa::helper::vector<unsigned int> &shell=tvsa[lastIndexVec[i]]; // tvsa[last]; //
                        for (j=0; j<shell.size(); ++j)
                        {

                            std::map<unsigned int, unsigned int>::iterator iter = Glob2LocMap.find(shell[j]);
                            if(iter != Glob2LocMap.end() )
                            {

                                unsigned int ind_j =Glob2LocMap[shell[j]]; // shell[j];//

                                if ((unsigned)mytriangles[ind_j][0]==last)
                                    mytriangles[ind_j][0]=tab[i];
                                else if ((unsigned)mytriangles[ind_j][1]==last)
                                    mytriangles[ind_j][1]=tab[i];
                                else if ((unsigned)mytriangles[ind_j][2]==last)
                                    mytriangles[ind_j][2]=tab[i];
                            }
                            else
                            {
                                //std::cout << "INFO_print : Coll - triangle NOT FOUND in the map !!! global index = "  << shell[j] << std::endl;
                            }
                        }

                        if (testc)
                        {

                            for (unsigned int j_loc=0; j_loc<mytriangles.size(); ++j_loc)
                            {

                                bool is_forgotten = false;
                                if ((unsigned)mytriangles[j_loc][0]==last)
                                {
                                    mytriangles[j_loc][0]=tab[i];
                                    is_forgotten=true;

                                }
                                else
                                {
                                    if ((unsigned)mytriangles[j_loc][1]==last)
                                    {
                                        mytriangles[j_loc][1]=tab[i];
                                        is_forgotten=true;

                                    }
                                    else
                                    {
                                        if ((unsigned)mytriangles[j_loc][2]==last)
                                        {
                                            mytriangles[j_loc][2]=tab[i];
                                            is_forgotten=true;
                                        }
                                    }

                                }

                                if(is_forgotten)
                                {

                                    unsigned int ind_forgotten = Loc2GlobVec[j];
                                    std::map<unsigned int, unsigned int>::iterator iter = Glob2LocMap.find(ind_forgotten);

                                    if(iter == Glob2LocMap.end() )
                                    {
                                        //std::cout << "INFO_print : Coll - triangle is forgotten in MAP !!! global index = "  << ind_forgotten << std::endl;
                                        //Glob2LocMap[ind_forgotten] = j;
                                    }

                                    bool is_in_shell = false;
                                    for (unsigned int j_glob=0; j_glob<shell.size(); ++j_glob)
                                    {
                                        is_in_shell = is_in_shell || (shell[j_glob] == ind_forgotten);
                                    }

                                    if(!is_in_shell)
                                    {
                                        //std::cout << "INFO_print : Coll - triangle is forgotten in SHELL !!! global index = "  << ind_forgotten << std::endl;
                                    }

                                }

                            }
                        }

                        --last;
                    }

                    ///

                    //updateFromTopology();


                }

                //}

                needsUpdate=true;
                //init();
                break;
            }
            default:
                // Ignore events that are not Triangle  related.
                break;
            }; // switch( changeType )

            mytriangles.resize( elems.size() ); // not necessary
            resize( elems.size() ); // not necessary

            ++itBegin;
        } // while( changeIt != last; )
    }
    if (needsUpdate)
        updateFlags();
}

void TriangleModel::draw(int index)
{
    Triangle t(this,index);
    glBegin(GL_TRIANGLES);
    glNormal3dv(t.n().ptr());
    glVertex3dv(t.p1().ptr());
    glVertex3dv(t.p2().ptr());
    glVertex3dv(t.p3().ptr());
    glEnd();
}

void TriangleModel::draw()
{
    if (getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glEnable(GL_LIGHTING);
        //Enable<GL_BLEND> blending;
        //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

        glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, getColor4f());
        static const float emissive[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
        static const float specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
        glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive);
        glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular);
        glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, 20);

        for (int i=0; i<size; i++)
        {
            draw(i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

void TriangleModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();
    if (needsUpdate && !cubeModel->empty()) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate=false;
    Vector3 minElem, maxElem;

    cubeModel->resize(size);  // size = number of triangles
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Triangle t(this,i);
            const Vector3& pt1 = t.p1();
            const Vector3& pt2 = t.p2();
            const Vector3& pt3 = t.p3();

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];
            }

            // Also recompute normal vector
            t.n() = cross(pt2-pt1,pt3-pt1);
            t.n().normalize();

            cubeModel->setParentOf(i, minElem, maxElem); // define the bounding box of the current triangle
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void TriangleModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate=false;
    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Triangle t(this,i);
            const Vector3& pt1 = t.p1();
            const Vector3& pt2 = t.p2();
            const Vector3& pt3 = t.p3();
            const Vector3 pt1v = pt1 + t.v1()*dt;
            const Vector3 pt2v = pt2 + t.v2()*dt;
            const Vector3 pt3v = pt3 + t.v3()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];

                if (pt1v[c] > maxElem[c]) maxElem[c] = pt1v[c];
                else if (pt1v[c] < minElem[c]) minElem[c] = pt1v[c];
                if (pt2v[c] > maxElem[c]) maxElem[c] = pt2v[c];
                else if (pt2v[c] < minElem[c]) minElem[c] = pt2v[c];
                if (pt3v[c] > maxElem[c]) maxElem[c] = pt3v[c];
                else if (pt3v[c] < minElem[c]) minElem[c] = pt3v[c];
            }

            // Also recompute normal vector
            t.n() = cross(pt2-pt1,pt3-pt1);
            t.n().normalize();

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void TriangleModel::fillArrays( float *array_coord,float *array_identity, int *offset_coord, int Id)
{
    const float factor = 1.0f/128.0f;
    const float step_Id = 1.0f/((float) size);
    float Id_triangle = Id+1;
    for ( int i=0; i<size; i++)
    {
        const float valueId = Id_triangle*factor;
        //For each triangle of the model, we store the coordinates of the vertices and information about each of them
        Triangle t(this, i);
        //Point 1
        array_coord[(*offset_coord)  ]    = (float) t.p1()[0];
        array_coord[(*offset_coord)+1]    = (float) t.p1()[1];
        array_coord[(*offset_coord)+2]    = (float) t.p1()[2];

        array_identity[(*offset_coord)  ] = valueId;
        array_identity[(*offset_coord)+1] = 0.0f;
        array_identity[(*offset_coord)+2] = 0.0f;
        (*offset_coord) += 3;
        //Point 2
        array_coord[(*offset_coord)]   = (float) t.p2()[0];
        array_coord[(*offset_coord)+1] = (float) t.p2()[1];
        array_coord[(*offset_coord)+2] = (float) t.p2()[2];


        array_identity[(*offset_coord)  ] = valueId;
        array_identity[(*offset_coord)+1] = 1.0f;
        array_identity[(*offset_coord)+2] = 0.0f;
        (*offset_coord) += 3;
        //Point 3
        array_coord[(*offset_coord)  ] = (float) t.p3()[0];
        array_coord[(*offset_coord)+1] = (float) t.p3()[1];
        array_coord[(*offset_coord)+2] = (float) t.p3()[2];

        array_identity[(*offset_coord)  ] = valueId;
        array_identity[(*offset_coord)+1] = 0.0f;
        array_identity[(*offset_coord)+2] = 1.0f;
        (*offset_coord) += 3;

        Id_triangle +=  step_Id;
    }
}

void TriangleModel::buildOctree()
{
    /*
    	cerr<<"TriangleModel::buildOctree(), coords = "<<*mstate->getX()<<endl;
    	const int ntris = mesh->getNbTriangles();
    	const int nquads = mesh->getNbQuads();
    	for (int i=0; i<ntris; i++)
    	{
    		topology::MeshTopology::Triangle idx = mesh->getTriangle(i);
    		cerr<<"  triangle "<< idx[0] <<", "<<idx[1]<<", "<<idx[2]<<endl;;
    	}
    	for (int i=0; i<nquads; i++)
    	{
    		topology::MeshTopology::Quad idx = mesh->getQuad(i);
    		cerr<<"  triangle "<< idx[0] <<", "<<idx[1]<<", "<<idx[2]<<endl;;
    		cerr<<"  triangle "<< idx[0] <<", "<<idx[2]<<", "<<idx[3]<<endl;;
    	}
    */
}

} // namespace collision

} // namespace component

} // namespace sofa

