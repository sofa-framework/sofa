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
#if defined (__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
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

int TriangleModelClass = core::RegisterObject("collision model using a triangular mesh")
        .add< TriangleModel >()
        .addAlias("Triangle")
        ;

TriangleModel::TriangleModel()
    : meshRevision(-1), mstate(NULL), mesh(NULL)
{
}

void TriangleModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
    elems.resize(size);
    //Loc2GlobVec.resize(size);
}

void TriangleModel::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::componentmodel::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    elems.clear();
    Loc2GlobVec.resize(0);
    Glob2LocMap.clear();

    if (mstate==NULL)
    {
        std::cerr << "ERROR: TriangleModel requires a Vec3 Mechanical Model.\n";
        return;
    }

    sofa::core::componentmodel::topology::BaseTopology* bt = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());
    if (bt)
    {
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

                    /*
                    elems[i].i1=triangleArray[i][0];
                    elems[i].i2=triangleArray[i][1];
                    elems[i].i3=triangleArray[i][2];

                    */

                    TriangleInfo t;

                    t.i1=(int)(triangleArray[i])[0];
                    t.i2=(int)(triangleArray[i])[1];
                    t.i3=(int)(triangleArray[i])[2];
                    elems.push_back(t);

                    Loc2GlobVec.push_back(i);
                    Glob2LocMap[i]=Loc2GlobVec.size()-1;

                    nb_visible_triangles+=1;
                }
            }

            resize(nb_visible_triangles);

        }
        else
        {
            //std::cerr << "ERROR: Topology is not a TetrahedronSetTopology.\n";
            sofa::component::topology::TriangleSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::TriangleSetTopologyContainer *>(container);
            if (tstc)
            {

                const sofa::helper::vector<sofa::component::topology::Triangle> &ta=tstc->getTriangleArray();

                resize(ta.size());
                //Loc2GlobVec.resize(ta.size());

                for(unsigned int i=0; i<ta.size(); ++i)
                {

                    elems[i].i1=ta[i][0];
                    elems[i].i2=ta[i][1];
                    elems[i].i3=ta[i][2];

                    Loc2GlobVec.push_back(i);
                    Glob2LocMap[i]=i;
                }

            }
            else
            {
                std::cerr << "ERROR: Topology is not a TriangleSetTopology.\n";
                //return;
            }
            //return;
        }

    }
    else
    {
        mesh = dynamic_cast< topology::MeshTopology* > (getContext()->getTopology());

        if (mesh==NULL)
        {
            std::cerr << "ERROR: TriangleModel requires a Mesh Topology.\n";
            return;
        }
        updateFromTopology();
    }
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
    const int npoints = mstate->getX()->size();
    const int ntris = mesh->getNbTriangles();
    const int nquads = mesh->getNbQuads();
    const int newsize = ntris+2*nquads;
    needsUpdate=true;

    int revision = mesh->getRevision();
    if (revision == meshRevision && newsize==size)
    {
        needsUpdate=false;
        return;
    }

    resize(newsize);
    //Loc2GlobVec.resize(newsize);

    int index = 0;
    //VecCoord& x = *mstate->getX();
    //VecDeriv& v = *mstate->getV();
    vector<bool> pflags(npoints);
    std::set<std::pair<int,int> > eflags;
    for (int i=0; i<ntris; i++)
    {
        topology::MeshTopology::Triangle idx = mesh->getTriangle(i);
        if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints)
        {
            std::cerr << "ERROR: Out of range index in triangle "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" ( total points="<<npoints<<")\n";
            continue;
        }
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[1];
        elems[index].i3 = idx[2];

        //Loc2GlobVec[index]=i;
        //Glob2LocMap[i]=index;

        int f = 0;
        if (!pflags[elems[index].i1])
        {
            f |= FLAG_P1;
            pflags[elems[index].i1] = true;
        }
        if (!pflags[elems[index].i2])
        {
            f |= FLAG_P2;
            pflags[elems[index].i2] = true;
        }
        if (!pflags[elems[index].i3])
        {
            f |= FLAG_P3;
            pflags[elems[index].i3] = true;
        }
        if (eflags.insert( (elems[index].i1<elems[index].i2)?std::make_pair(elems[index].i1,elems[index].i2):std::make_pair(elems[index].i2,elems[index].i1) ).second)
        {
            f |= FLAG_E12;
        }
        if (eflags.insert( (elems[index].i2<elems[index].i3)?std::make_pair(elems[index].i2,elems[index].i3):std::make_pair(elems[index].i3,elems[index].i2) ).second)
        {
            f |= FLAG_E23;
        }
        if (eflags.insert( (elems[index].i3<elems[index].i1)?std::make_pair(elems[index].i3,elems[index].i1):std::make_pair(elems[index].i1,elems[index].i3) ).second)
        {
            f |= FLAG_E31;
        }
        elems[index].flags = f;

        //elems[index].i3 = idx[2];
        ++index;
    }
    for (int i=0; i<nquads; i++)
    {
        topology::MeshTopology::Quad idx = mesh->getQuad(i);
        if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints || idx[3] >= npoints)
        {
            std::cerr << "ERROR: Out of range index in quad "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" "<<idx[3]<<" ( total points="<<npoints<<")\n";
            continue;
        }
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[1];
        elems[index].i3 = idx[2];
        //Loc2GlobVec[index]=i;
        //Glob2LocMap[i]=index;

        int f = 0;
        if (!pflags[elems[index].i1])
        {
            f |= FLAG_P1;
            pflags[elems[index].i1] = true;
        }
        if (!pflags[elems[index].i2])
        {
            f |= FLAG_P2;
            pflags[elems[index].i2] = true;
        }
        if (!pflags[elems[index].i3])
        {
            f |= FLAG_P3;
            pflags[elems[index].i3] = true;
        }
        if (eflags.insert( (elems[index].i1<elems[index].i2)?std::make_pair(elems[index].i1,elems[index].i2):std::make_pair(elems[index].i2,elems[index].i1) ).second)
        {
            f |= FLAG_E12;
        }
        if (eflags.insert( (elems[index].i2<elems[index].i3)?std::make_pair(elems[index].i2,elems[index].i3):std::make_pair(elems[index].i3,elems[index].i2) ).second)
        {
            f |= FLAG_E23;
        }
        elems[index].flags = f;

        ++index;
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[2];
        elems[index].i3 = idx[3];
        //Loc2GlobVec[index]=i;
        //Glob2LocMap[i]=index;
        f = 0;
        if (!pflags[elems[index].i3])
        {
            f |= FLAG_P3;
            pflags[elems[index].i3] = true;
        }
        if (eflags.insert( (elems[index].i2<elems[index].i3)?std::make_pair(elems[index].i2,elems[index].i3):std::make_pair(elems[index].i3,elems[index].i2) ).second)
        {
            f |= FLAG_E23;
        }
        if (eflags.insert( (elems[index].i3<elems[index].i1)?std::make_pair(elems[index].i3,elems[index].i1):std::make_pair(elems[index].i1,elems[index].i3) ).second)
        {
            f |= FLAG_E31;
        }
        elems[index].flags = f;

        ++index;
    }
    meshRevision = revision;
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
void TriangleModel::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology* bt = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());
    if (bt)
    {

        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=bt->firstChange();
        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=bt->lastChange();


        while( itBegin != itEnd )
        {
            core::componentmodel::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();
            // Since we are using identifier, we can safely use C type casts.

            sofa::core::componentmodel::topology::BaseTopology* bt = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());
            sofa::core::componentmodel::topology::TopologyContainer *container=bt->getTopologyContainer();

            sofa::component::topology::TriangleSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::TriangleSetTopologyContainer *>(container);
            sofa::component::topology::TetrahedronSetTopologyContainer *testc= dynamic_cast<sofa::component::topology::TetrahedronSetTopologyContainer *>(container);

            if((changeType == core::componentmodel::topology::TETRAHEDRAREMOVED) || (((!testc) && changeType == core::componentmodel::topology::TRIANGLESREMOVED)))
            {

                unsigned int my_size;
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
                    unsigned int my_loc = iter_check->second;
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

                                if(elems[Glob2LocMap[j_check]].i1 > (int) last)
                                {
                                    std::cout << "INFO_print : Coll !!! POINT 0 OUT for j_check = " << j_check << " , triangle = "<< Glob2LocMap[j_check] << " , point = " << elems[Glob2LocMap[j_check]].i1 << std::endl;
                                }
                                if(elems[Glob2LocMap[j_check]].i2 > (int) last)
                                {
                                    std::cout << "INFO_print : Coll !!! POINT 1 OUT for j_check = " << j_check << " , triangle = "<< Glob2LocMap[j_check] << " , point = " << elems[Glob2LocMap[j_check]].i2 << std::endl;
                                }
                                if(elems[Glob2LocMap[j_check]].i3 > (int) last)
                                {
                                    std::cout << "INFO_print : Coll !!! POINT 2 OUT for j_check = " << j_check << " , triangle = "<< Glob2LocMap[j_check] << " , point = " << elems[Glob2LocMap[j_check]].i3 << std::endl;
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
                    t.i1=(int)(ta->triangleArray[i])[0];
                    t.i2=(int)(ta->triangleArray[i])[1];
                    t.i3=(int)(ta->triangleArray[i])[2];
                    elems.push_back(t);

                    unsigned int ind_triangle = Loc2GlobVec.size();//(ta->triangleIndexArray)[i];
                    Loc2GlobVec.push_back(ind_triangle);
                    Glob2LocMap[ind_triangle]=ind_triangle;

                }
                resize( elems.size());
                //Loc2GlobVec.resize( elems.size());
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

                        }

                        ind_last = elems.size() - 1;

                        if(ind_real_last != ind_last)
                        {

                            tmp = elems[ind_real_last];
                            elems[ind_real_last] = elems[ind_last];
                            elems[ind_last] = tmp;

                            Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_last]));
                            Glob2LocMap[Loc2GlobVec[ind_last]] = ind_real_last;
                            Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_real_last]));
                            Glob2LocMap[Loc2GlobVec[ind_real_last]] = ind_last;

                            ind_tmp = Loc2GlobVec[ind_real_last];
                            Loc2GlobVec[ind_real_last] = Loc2GlobVec[ind_last];
                            Loc2GlobVec[ind_last] = ind_tmp;

                        }

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
                            //elems[ind_real_last].i1=triangleArray[k][0];
                            //elems[ind_real_last].i2=triangleArray[k][1];
                            //elems[ind_real_last].i3=triangleArray[k][2];

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

                    const sofa::helper::vector<sofa::component::topology::Triangle> &triangleArray=testc->getTriangleArray();
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
                                    TriangleInfo t;

                                    const sofa::component::topology::Tetrahedron &te=tetrahedronArray[ind_test];
                                    int h = testc->getTriangleIndexInTetrahedron(testc->getTetrahedronTriangles(ind_test),k);

                                    if (h%2)
                                    {
                                        t.i1=(int)(te[(h+1)%4]); t.i2=(int)(te[(h+2)%4]); t.i3=(int)(te[(h+3)%4]);
                                    }
                                    else
                                    {
                                        t.i1=(int)(te[(h+1)%4]); t.i3=(int)(te[(h+2)%4]); t.i2=(int)(te[(h+3)%4]);
                                    }

                                    // sort t such that t.i1 is the smallest one
                                    while ((t.i1>t.i2) || (t.i1>t.i3))
                                    {
                                        int val=t.i1; t.i1=t.i2; t.i2=t.i3; t.i3=val;
                                    }

                                    unsigned int prev_size = elems.size();

                                    elems.push_back(t);

                                    Loc2GlobVec.push_back(k);
                                    Glob2LocMap[k]=Loc2GlobVec.size()-1;

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

                                if ((unsigned)elems[ind_j].i1==last)
                                    elems[ind_j].i1=tab[i];
                                else if ((unsigned)elems[ind_j].i2==last)
                                    elems[ind_j].i2=tab[i];
                                else if ((unsigned)elems[ind_j].i3==last)
                                    elems[ind_j].i3=tab[i];
                            }
                            else
                            {
                                //std::cout << "INFO_print : Coll - triangle NOT FOUND in the map !!! global index = "  << shell[j] << std::endl;
                            }
                        }

                        if (testc)
                        {

                            for (unsigned int j_loc=0; j_loc<elems.size(); ++j_loc)
                            {

                                bool is_forgotten = false;
                                if ((unsigned)elems[j_loc].i1==last)
                                {
                                    elems[j_loc].i1=tab[i];
                                    is_forgotten=true;

                                }
                                else
                                {
                                    if ((unsigned)elems[j_loc].i2==last)
                                    {
                                        elems[j_loc].i2=tab[i];
                                        is_forgotten=true;

                                    }
                                    else
                                    {
                                        if ((unsigned)elems[j_loc].i3==last)
                                        {
                                            elems[j_loc].i3=tab[i];
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

            resize( elems.size() ); // not necessary

            ++itBegin;
        } // while( changeIt != last; )
    }
}
void TriangleModel::draw()
{
    if (isActive() && getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glEnable(GL_LIGHTING);
        //Enable<GL_BLEND> blending;
        //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

        static const float color[4] = { 1.0f, 0.2f, 0.0f, 1.0f};
        static const float colorStatic[4] = { 0.5f, 0.5f, 0.5f, 1.0f};
        if (isStatic())
            glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, colorStatic);
        else
            glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
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
    if (isActive() && getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

void TriangleModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (mesh)
        updateFromTopology();
    if (needsUpdate && !cubeModel->empty()) cubeModel->resize(0);
    if (isStatic() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

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
unsigned int TriangleModel::getNbTriangles() const
{
    if (!mesh)
        return size;
    else
        return mesh->getNbTriangles();
}
void TriangleModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (mesh)
        updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (isStatic() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

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

void TriangleModel::buildOctree()
{
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
}

} // namespace collision

} // namespace component

} // namespace sofa

