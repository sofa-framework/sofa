/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_INL

#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/TriangleSetTopologyAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>

#include <sofa/component/container/MechanicalObject.h>

#include <algorithm>
#include <functional>

namespace sofa
{
namespace component
{
namespace topology
{

using namespace sofa::defaulttype;


template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::init()
{
    EdgeSetTopologyAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
}

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::reinit()
{
    if (!(m_listTriRemove.getValue () ).empty() && this->getContext()->getAnimate())
    {
        sofa::helper::vector< unsigned int > items = m_listTriRemove.getValue ();
        m_modifier->removeItems(items);

        m_modifier->propagateTopologicalChanges();
        items.clear();
    }

    if (!(m_listTriAdd.getValue () ).empty() && this->getContext()->getAnimate())
    {
        int nbrBefore = m_container->getNbTriangles();

        m_modifier->addTrianglesProcess(m_listTriAdd.getValue ());

        sofa::helper::vector< TriangleID > new_triangles_id;

        for (unsigned int i = 0; i < (m_listTriAdd.getValue ()).size(); i++)
            new_triangles_id.push_back (m_container->getNbTriangles()-(m_listTriAdd.getValue ()).size()+i);

        //	  std::cout << "new tri ID: "<< new_triangles_id<<std::endl;
        //	  std::cout << "params: " << (m_listTriAdd.getValue ()).size()<< m_listTriAdd.getValue () <<  new_triangles_id << std::endl;

        if (nbrBefore != m_container->getNbTriangles()) // Triangles have been added
        {
            m_modifier->addTrianglesWarning((m_listTriAdd.getValue ()).size(), m_listTriAdd.getValue (), new_triangles_id);
            m_modifier->propagateTopologicalChanges();
        }
        else
        {
            std::cout << " Nothing added " << std::endl;
        }

    }

}



// Move and fix the two closest points of two triangles to their median point
template<class DataTypes>
bool TriangleSetTopologyAlgorithms< DataTypes >::Suture2Points(unsigned int ind_ta, unsigned int ind_tb,
        unsigned int &ind1, unsigned int &ind2)
{
    // Access the topology
    m_geometryAlgorithms->computeClosestIndexPair(ind_ta, ind_tb, ind1, ind2);

    //this->sout << "INFO_print : ind1 = " << ind1 << this->sendl;
    //this->sout << "INFO_print : ind2 = " << ind2 << this->sendl;

    Vec<3,double> point_created = m_geometryAlgorithms->computeBaryEdgePoint(ind1, ind2, 0.5);

    sofa::helper::vector< double > x_created;
    x_created.push_back((double) point_created[0]);
    x_created.push_back((double) point_created[1]);
    x_created.push_back((double) point_created[2]);

    ///TODO:Cast into a MechanicalObject ?
    MechanicalState<DataTypes>* state = m_geometryAlgorithms->getDOF();
    MechanicalObject<DataTypes> *temp = dynamic_cast<MechanicalObject<DataTypes> *>(state);
    temp->forcePointPosition(ind1, x_created);
    temp->forcePointPosition(ind2, x_created);

    return true;
}

// Removes triangles along the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::RemoveAlongTrianglesList(const Vec<3,double>& a,
        const Vec<3,double>& b,
        const unsigned int ind_ta,
        const unsigned int ind_tb)
{
    std::cout << "TriangleSetTopologyAlgorithms< DataTypes >::RemoveAlongTrianglesList" << std::endl;

    sofa::helper::vector< unsigned int > triangles_list;
    sofa::helper::vector< unsigned int > edges_list;
    sofa::helper::vector< double > coords_list;

    bool is_intersected=false;

    unsigned int ind_tb_final;

    bool is_on_boundary;

    ind_tb_final=ind_tb;
    is_intersected = m_geometryAlgorithms->computeIntersectedPointsList(a, b, ind_ta, ind_tb_final, triangles_list, edges_list, coords_list, is_on_boundary);

    if(is_intersected)
    {
        //sofa::helper::vector< unsigned int > triangles;
        //for (unsigned int i=0; i<triangles_list.size(); ++i)
        //{
        //	triangles.push_back(triangles_list[i]);
        //}
        m_modifier->removeTriangles(triangles_list, true, true);
    }
}


// Incises along the list of points (ind_edge,coord) intersected by the sequence of input segments (list of input points) and the triangular mesh

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::InciseAlongLinesList(const sofa::helper::vector< Vec<3,double> >& input_points,
        const sofa::helper::vector< unsigned int > &input_triangles)
{
    // HYP : input_points.size() == input_triangles.size()

    unsigned int points_size = input_points.size();

    // Initialization for INTERSECTION method
    sofa::helper::vector< unsigned int > triangles_list;
    sofa::helper::vector< unsigned int > edges_list;
    sofa::helper::vector< double > coords_list;

    unsigned int ind_tb_final;

    bool is_on_boundary;

    const Vec<3,double> a = input_points[0];
    unsigned int ind_ta = input_triangles[0];

    unsigned int j = 0;
    bool is_validated=true;
    for(j = 0; is_validated && j < points_size - 1; ++j)
    {
        const Vec<3,double> pa = input_points[j];
        const Vec<3,double> pb = input_points[j+1];
        unsigned int ind_tpa = input_triangles[j];
        unsigned int ind_tpb = input_triangles[j+1];

        bool is_distinct = (pa!=pb && ind_tpa!=ind_tpb);

        if(is_distinct)
        {
            // Call the method "computeIntersectedPointsList" to get the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
            ind_tb_final=ind_tpb;
            bool is_intersected = m_geometryAlgorithms->computeIntersectedPointsList(pa, pb, ind_tpa, ind_tb_final, triangles_list, edges_list, coords_list, is_on_boundary);
            is_validated=is_intersected;
        }
        else
        {
            is_validated=false;
        }
    }

    const Vec<3,double> b = input_points[j];
    unsigned int ind_tb = input_triangles[j];

    const Triangle &ta=m_container->getTriangle(ind_ta);
    const Triangle &tb=m_container->getTriangle(ind_tb);

    //const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();
    unsigned int nb_points =  m_container->getTriangleVertexShellArray().size() - 1; //vect_c.size() -1;

    const sofa::helper::vector<Triangle> &vect_t=m_container->getTriangleArray();
    unsigned int nb_triangles =  vect_t.size() -1;

    // Variables to accumulate the number of elements registered to be created (so as to remember their indices)
    unsigned int acc_nb_points=nb_points;
    unsigned int acc_nb_triangles=nb_triangles;

    // Variables to accumulate the elements registered to be created or to be removed
    sofa::helper::vector< sofa::helper::vector< unsigned int > > p_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs;
    sofa::helper::vector< Triangle > triangles_to_create;
    sofa::helper::vector< unsigned int > trianglesIndexList;
    sofa::helper::vector< unsigned int > triangles_to_remove;

    unsigned int ta_to_remove;
    unsigned int tb_to_remove;

    // Initialization for SNAPPING method

    bool is_snap_a0=false;
    bool is_snap_a1=false;
    bool is_snap_a2=false;

    bool is_snap_b0=false;
    bool is_snap_b1=false;
    bool is_snap_b2=false;

    double epsilon = 0.2; // INFO : epsilon is a threshold in [0,1] to control the snapping of the extremities to the closest vertex

    sofa::helper::vector< double > a_baryCoefs = m_geometryAlgorithms->computeTriangleBarycoefs(ind_ta, (const Vec<3,double> &) a);
    snapping_test_triangle(epsilon, a_baryCoefs[0], a_baryCoefs[1], a_baryCoefs[2],
            is_snap_a0, is_snap_a1, is_snap_a2);

    double is_snapping_a = is_snap_a0 || is_snap_a1 || is_snap_a2;

    sofa::helper::vector< double > b_baryCoefs = m_geometryAlgorithms->computeTriangleBarycoefs(ind_tb, (const Vec<3,double> &) b);
    snapping_test_triangle(epsilon, b_baryCoefs[0], b_baryCoefs[1], b_baryCoefs[2],
            is_snap_b0, is_snap_b1, is_snap_b2);

    double is_snapping_b = is_snap_b0 || is_snap_b1 || is_snap_b2;

    /*
      if(is_snapping_a){
      this->sout << "INFO_print : is_snapping_a" <<  this->sendl;
      }
      if(is_snapping_b){
      this->sout << "INFO_print : is_snapping_b" <<  this->sendl;
      }
    */

    if(is_validated) // intersection successfull
    {
        /// force the creation of TriangleEdgeShellArray
        m_container->getTriangleEdgeShellArray();
        /// force the creation of TriangleVertexShellArray
        m_container->getTriangleVertexShellArray();

        // Initialization for the indices of the previous intersected edge
        unsigned int p1_prev=0;
        unsigned int p2_prev=0;

        unsigned int p1_a=m_container->getEdge(edges_list[0])[0];
        unsigned int p2_a=m_container->getEdge(edges_list[0])[1];
        unsigned int p1_b=m_container->getEdge(edges_list[edges_list.size()-1])[0];
        unsigned int p2_b=m_container->getEdge(edges_list[edges_list.size()-1])[1];

        // Plan to remove triangles indexed by ind_ta and ind_tb
        triangles_to_remove.push_back(ind_ta); triangles_to_remove.push_back(ind_tb);

        // Treatment of particular case for first extremity a

        sofa::helper::vector< unsigned int > a_first_ancestors;
        sofa::helper::vector< double > a_first_baryCoefs;

        if(!is_snapping_a)
        {
            /// Register the creation of point a

            a_first_ancestors.push_back(ta[0]);
            a_first_ancestors.push_back(ta[1]);
            a_first_ancestors.push_back(ta[2]);
            p_ancestors.push_back(a_first_ancestors);
            p_baryCoefs.push_back(a_baryCoefs);

            acc_nb_points=acc_nb_points+1;

            /// Register the creation of triangles incident to point a

            unsigned int ind_a =  acc_nb_points; // last point registered to be created

            sofa::helper::vector< Triangle > a_triangles;
            Triangle t_a01 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_a,
                    (unsigned int)ta[0],
                    (unsigned int) ta[1]));
            Triangle t_a12 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_a,
                    (unsigned int)ta[1],
                    (unsigned int) ta[2]));
            Triangle t_a20 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_a,
                    (unsigned int)ta[2],
                    (unsigned int) ta[0]));
            triangles_to_create.push_back(t_a01);
            triangles_to_create.push_back(t_a12);
            triangles_to_create.push_back(t_a20);

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles+1);
            trianglesIndexList.push_back(acc_nb_triangles+2);
            acc_nb_triangles=acc_nb_triangles+3;

            /// Register the removal of triangles incident to point a

            if(ta[0]!=p1_a && ta[0]!=p2_a)
            {
                ta_to_remove=acc_nb_triangles-1;
            }
            else
            {
                if(ta[1]!=p1_a && ta[1]!=p2_a)
                {
                    ta_to_remove=acc_nb_triangles;
                }
                else // (ta[2]!=p1_a && ta[2]!=p2_a)
                {
                    ta_to_remove=acc_nb_triangles-2;
                }
            }
            triangles_to_remove.push_back(ta_to_remove);

            Triangle t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,
                    (unsigned int) ind_a,
                    (unsigned int) p1_a));
            Triangle t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,
                    (unsigned int) p2_a,
                    (unsigned int)ind_a));
            triangles_to_create.push_back(t_pa1);
            triangles_to_create.push_back(t_pa2);

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles+1);
            acc_nb_triangles=acc_nb_triangles+2;
        }
        else // snapping a to the vertex indexed by ind_a, which is the closest to point a
        {
            // localize the closest vertex
            unsigned int ind_a;
            unsigned int p0_a;

            if(ta[0]!=p1_a && ta[0]!=p2_a)
            {
                p0_a=ta[0];
            }
            else
            {
                if(ta[1]!=p1_a && ta[1]!=p2_a)
                {
                    p0_a=ta[1];
                }
                else// ta[2]!=p1_a && ta[2]!=p2_a
                {
                    p0_a=ta[2];
                }
            }

            if(is_snap_a0) // is_snap_a1 == false and is_snap_a2 == false
            {
                /// VERTEX 0
                ind_a=ta[0];
            }
            else
            {
                if(is_snap_a1) // is_snap_a0 == false and is_snap_a2 == false
                {
                    /// VERTEX 1
                    ind_a=ta[1];
                }
                else // is_snap_a2 == true and (is_snap_a0 == false and is_snap_a1 == false)
                {
                    /// VERTEX 2
                    ind_a=ta[2];
                }
            }

            /// Register the creation of triangles incident to point indexed by ind_a

            if(ind_a==p1_a)
            {
                Triangle t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,
                        (unsigned int) p0_a,
                        (unsigned int) p1_a));
                Triangle t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,
                        (unsigned int) p2_a,
                        (unsigned int) p0_a));
                triangles_to_create.push_back(t_pa1);
                triangles_to_create.push_back(t_pa2);
            }
            else
            {
                if(ind_a==p2_a)
                {
                    Triangle t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,
                            (unsigned int) p0_a,
                            (unsigned int) p1_a));
                    Triangle t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,
                            (unsigned int) p2_a,
                            (unsigned int) p0_a));
                    triangles_to_create.push_back(t_pa1);
                    triangles_to_create.push_back(t_pa2);
                }
                else
                {
                    Triangle t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,
                            (unsigned int) ind_a,
                            (unsigned int) p1_a));
                    Triangle t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,
                            (unsigned int) p2_a,
                            (unsigned int)ind_a));
                    triangles_to_create.push_back(t_pa1);
                    triangles_to_create.push_back(t_pa2);
                }
            }

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles+1);
            acc_nb_triangles+=2;
        }

        // Traverse the loop of interected edges

        for (unsigned int i=0; i<edges_list.size(); ++i)
        {
            /// Register the creation of the two points (say current "duplicated points") localized on the current interected edge
            unsigned int p1 = m_container->getEdge(edges_list[i])[0];
            unsigned int p2 = m_container->getEdge(edges_list[i])[1];

            sofa::helper::vector< unsigned int > p_first_ancestors;
            p_first_ancestors.push_back(p1);
            p_first_ancestors.push_back(p2);
            p_ancestors.push_back(p_first_ancestors);
            p_ancestors.push_back(p_first_ancestors);

            sofa::helper::vector< double > p_first_baryCoefs;
            p_first_baryCoefs.push_back(1.0 - coords_list[i]);
            p_first_baryCoefs.push_back(coords_list[i]);
            p_baryCoefs.push_back(p_first_baryCoefs);
            p_baryCoefs.push_back(p_first_baryCoefs);

            acc_nb_points=acc_nb_points+2;

            if(i>0) // not to treat particular case of first extremitiy
            {
                // SNAPPING TEST

                double gamma = 0.3;
                bool is_snap_p1;
                bool is_snap_p2;

                snapping_test_edge(gamma, 1.0 - coords_list[i], coords_list[i], is_snap_p1, is_snap_p2);
                double is_snapping_p = is_snap_p1 || is_snap_p2;

                unsigned int ind_p;

                if(is_snapping_p && i<edges_list.size()-1) // not to treat particular case of last extremitiy
                {
                    if(is_snap_p1)
                    {
                        /// VERTEX 0
                        ind_p=p1;
                    }
                    else // is_snap_p2 == true
                    {
                        /// VERTEX 1
                        ind_p=p2;
                    }

                    //this->sout << "INFO_print : is_snapping_p, i = " << i << " on vertex " << ind_p <<  this->sendl;

                    sofa::helper::vector< unsigned int > triangles_list_1;

                    sofa::helper::vector< unsigned int > triangles_list_2;

                    //this->sout << "INFO_print : DO Prepare_VertexDuplication " <<  this->sendl;
                    m_geometryAlgorithms->prepareVertexDuplication(ind_p, triangles_list[i], triangles_list[i+1], m_container->getEdge(edges_list[i-1]), coords_list[i-1], m_container->getEdge(edges_list[i+1]), coords_list[i+1], triangles_list_1, triangles_list_2);
                    //this->sout << "INFO_print : DONE Prepare_VertexDuplication " <<  this->sendl;

                    //this->sout << "INFO_print : triangles_list_1.size() = " << triangles_list_1.size() <<  this->sendl;
                    //for (unsigned int k=0;k<triangles_list_1.size();k++){
                    //		this->sout << "INFO_print : triangles_list_1 number " << k << " = " << triangles_list_1[k] <<  this->sendl;
                    //}

                    //this->sout << "INFO_print : triangles_list_2.size() = " << triangles_list_2.size() <<  this->sendl;
                    //for (unsigned int k=0;k<triangles_list_2.size();k++){
                    //		this->sout << "INFO_print : triangles_list_2 number " << k << " = " << triangles_list_2[k] <<  this->sendl;
                    //}
                }

                /// Register the removal of the current triangle

                triangles_to_remove.push_back(triangles_list[i]);

                /// Register the creation of triangles incident to the current "duplicated points" and to the previous "duplicated points"

                unsigned int p1_created=acc_nb_points - 3;
                unsigned int p2_created=acc_nb_points - 2;

                unsigned int p1_to_create=acc_nb_points - 1;
                unsigned int p2_to_create=acc_nb_points;

                unsigned int p0_t = m_container->getTriangle(triangles_list[i])[0];
                unsigned int p1_t = m_container->getTriangle(triangles_list[i])[1];
                unsigned int p2_t = m_container->getTriangle(triangles_list[i])[2];

                Triangle t_p1 = Triangle(helper::make_array<unsigned int>((unsigned int) p1_created,(unsigned int) p1_prev,(unsigned int) p1_to_create));
                Triangle t_p2 = Triangle(helper::make_array<unsigned int>((unsigned int) p2_created,(unsigned int) p2_to_create,(unsigned int) p2_prev));

                Triangle t_p3;

                if(p0_t!=p1_prev && p0_t!=p2_prev)
                {
                    if(p0_t==p1)
                    {
                        t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p0_t,(unsigned int) p1_to_create,(unsigned int) p1_prev));

                    }
                    else // p0_t==p2
                    {
                        t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p0_t,(unsigned int) p2_prev,(unsigned int) p2_to_create));
                    }
                }
                else
                {
                    if(p1_t!=p1_prev && p1_t!=p2_prev)
                    {
                        if(p1_t==p1)
                        {
                            t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p1_t,(unsigned int) p1_to_create,(unsigned int) p1_prev));
                        }
                        else // p1_t==p2
                        {
                            t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p1_t,(unsigned int) p2_prev,(unsigned int) p2_to_create));
                        }
                    }
                    else // (p2_t!=p1_prev && p2_t!=p2_prev)
                    {
                        if(p2_t==p1)
                        {
                            t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p2_t,(unsigned int) p1_to_create,(unsigned int) p1_prev));
                        }
                        else // p2_t==p2
                        {
                            t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p2_t,(unsigned int) p2_prev,(unsigned int) p2_to_create));
                        }
                    }
                }

                triangles_to_create.push_back(t_p1);
                triangles_to_create.push_back(t_p2);
                triangles_to_create.push_back(t_p3);

                trianglesIndexList.push_back(acc_nb_triangles);
                trianglesIndexList.push_back(acc_nb_triangles+1);
                trianglesIndexList.push_back(acc_nb_triangles+2);
                acc_nb_triangles=acc_nb_triangles+3;
            }

            // Update the previous "duplicated points"
            p1_prev=p1;
            p2_prev=p2;
        }

        // Treatment of particular case for second extremity b
        sofa::helper::vector< unsigned int > b_first_ancestors;
        sofa::helper::vector< double > b_first_baryCoefs;

        if(!is_snapping_b)
        {
            /// Register the creation of point b

            b_first_ancestors.push_back(tb[0]);
            b_first_ancestors.push_back(tb[1]);
            b_first_ancestors.push_back(tb[2]);
            p_ancestors.push_back(b_first_ancestors);
            p_baryCoefs.push_back(b_baryCoefs);

            acc_nb_points=acc_nb_points+1;

            /// Register the creation of triangles incident to point b

            unsigned int ind_b =  acc_nb_points; // last point registered to be created

            sofa::helper::vector< Triangle > b_triangles;
            Triangle t_b01 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_b,
                    (unsigned int)tb[0],
                    (unsigned int) tb[1]));
            Triangle t_b12 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_b,
                    (unsigned int)tb[1],
                    (unsigned int) tb[2]));
            Triangle t_b20 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_b,
                    (unsigned int)tb[2],
                    (unsigned int) tb[0]));
            triangles_to_create.push_back(t_b01);
            triangles_to_create.push_back(t_b12);
            triangles_to_create.push_back(t_b20);

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles+1);
            trianglesIndexList.push_back(acc_nb_triangles+2);
            acc_nb_triangles=acc_nb_triangles+3;

            /// Register the removal of triangles incident to point b

            if(tb[0]!=p1_b && tb[0]!=p2_b)
            {
                tb_to_remove=acc_nb_triangles-1;
            }
            else
            {
                if(tb[1]!=p1_b && tb[1]!=p2_b)
                {
                    tb_to_remove=acc_nb_triangles;
                }
                else // (tb[2]!=p1_b && tb[2]!=p2_b)
                {
                    tb_to_remove=acc_nb_triangles-2;
                }
            }
            triangles_to_remove.push_back(tb_to_remove);

            Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 2,
                    (unsigned int) p1_b,
                    (unsigned int)ind_b));
            Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,
                    (unsigned int)ind_b,
                    (unsigned int) p2_b));
            triangles_to_create.push_back(t_pb1);
            triangles_to_create.push_back(t_pb2);

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles+1);
            acc_nb_triangles=acc_nb_triangles+2;

        }
        else // snapping b to the vertex indexed by ind_b, which is the closest to point b
        {
            // localize the closest vertex
            unsigned int ind_b;
            unsigned int p0_b;

            if(tb[0]!=p1_b && tb[0]!=p2_b)
            {
                p0_b=tb[0];
            }
            else
            {
                if(tb[1]!=p1_b && tb[1]!=p2_b)
                {
                    p0_b=tb[1];
                }
                else// tb[2]!=p1_b && tb[2]!=p2_b
                {
                    p0_b=tb[2];
                }
            }

            if(is_snap_b0) // is_snap_b1 == false and is_snap_b2 == false
            {
                /// VERTEX 0
                ind_b=tb[0];
            }
            else
            {
                if(is_snap_b1) // is_snap_b0 == false and is_snap_b2 == false
                {
                    /// VERTEX 1
                    ind_b=tb[1];
                }
                else // is_snap_b2 == true and (is_snap_b0 == false and is_snap_b1 == false)
                {
                    /// VERTEX 2
                    ind_b=tb[2];
                }
            }

            /// Register the creation of triangles incident to point indexed by ind_b

            if(ind_b==p1_b)
            {
                Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,(unsigned int) p1_b,(unsigned int) p0_b));
                Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,(unsigned int) p0_b, (unsigned int) p2_b));
                triangles_to_create.push_back(t_pb1);
                triangles_to_create.push_back(t_pb2);

            }
            else
            {
                if(ind_b==p2_b)
                {
                    Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int) p1_b,(unsigned int) p0_b));
                    Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int) p0_b, (unsigned int) p2_b));
                    triangles_to_create.push_back(t_pb1);
                    triangles_to_create.push_back(t_pb2);
                }
                else
                {
                    Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int) p1_b,(unsigned int) ind_b));
                    Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,(unsigned int) ind_b, (unsigned int)p2_b));
                    triangles_to_create.push_back(t_pb1);
                    triangles_to_create.push_back(t_pb2);
                }
            }

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles+1);
            acc_nb_triangles+=2;
        }

        // Create all the points registered to be created
        m_modifier->addPointsProcess((const unsigned int) acc_nb_points - nb_points);

        // Warn for the creation of all the points registered to be created
        m_modifier->addPointsWarning((const unsigned int) acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

        // Create all the triangles registered to be created
        m_modifier->addTrianglesProcess((const sofa::helper::vector< Triangle > &) triangles_to_create) ; // WARNING called after the creation process by the method "addTrianglesProcess"

        // Warn for the creation of all the triangles registered to be created
        m_modifier->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList);

        // Propagate the topological changes *** not necessary
        //m_modifier->propagateTopologicalChanges();

        // Remove all the triangles registered to be removed
        m_modifier->removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

        // Propagate the topological changes *** not necessary
        //m_modifier->propagateTopologicalChanges();
    }
}

template<class DataTypes>
int TriangleSetTopologyAlgorithms<DataTypes>::SplitAlongPath(unsigned int pa, Coord& a, unsigned int pb, Coord& b,
        sofa::helper::vector<TriangleID>& triangles_list, sofa::helper::vector<EdgeID>& edges_list,
        sofa::helper::vector<double>& coords_list, sofa::helper::vector<EdgeID>& new_edges, bool snap)
{
    std::cout << "TriangleSetTopologyAlgorithms<DataTypes>::SplitAlongPath" << std::endl;


    //////// STEP 1 : MODIFY PATH IF SNAP = TRUE (don't change border case here)
    (void)snap;

    /*	sofa::helper::vector< double > points2Snap;
    if (snap)
    {
      SnapAlongPath (triangles_list, edges_list, coords_list, points2Snap);
      }*/







    unsigned int nb_edges = edges_list.size();
    sofa::helper::vector< sofa::helper::vector< PointID > > p_ancestors; p_ancestors.reserve(nb_edges+2);
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs; p_baryCoefs.reserve(nb_edges+2);
    PointID next_point = m_container->getNbPoints();
    TriangleID next_triangle = m_container->getNbTriangles();
    if (triangles_list.empty()) return 0;
    sofa::helper::vector< PointID > new_edge_points; // new points created on each edge
    sofa::helper::vector< Triangle > new_triangles;
    sofa::helper::vector< TriangleID > new_triangles_id;
    sofa::helper::vector< TriangleID > removed_triangles;

    //////// STEP 1 : Create points

    //// STEP 1a : Create start point if necessary

    if (pa == (PointID)-1 && triangles_list.front() != (TriangleID)-1)
    {
        // first point is inside a triangle
        Triangle t = m_container->getTriangle(triangles_list.front());
        p_ancestors.resize(p_ancestors.size()+1);
        sofa::helper::vector< PointID >& ancestors = p_ancestors.back();
        p_baryCoefs.resize(p_baryCoefs.size()+1);
        sofa::helper::vector< double >& baryCoefs = p_baryCoefs.back();
        ancestors.push_back(t[0]);
        ancestors.push_back(t[1]);
        ancestors.push_back(t[2]);
        Vec<3, double> p; p = a;
        baryCoefs = m_geometryAlgorithms->compute3PointsBarycoefs(p, t[0], t[1], t[2]);
        std::cout << "Creating first point in triangle "<<triangles_list.front()<<" barycoefs "<<baryCoefs<<std::endl;
        pa = (next_point); ++next_point;
    }

    //// STEP 1b : Create a point on each crossed edge

    for (unsigned int i = 0; i < nb_edges; ++i)
    {
        Edge e = m_container->getEdge(edges_list[i]);
        p_ancestors.resize(p_ancestors.size()+1);
        sofa::helper::vector< PointID >& ancestors = p_ancestors.back();
        p_baryCoefs.resize(p_baryCoefs.size()+1);
        sofa::helper::vector< double >& baryCoefs = p_baryCoefs.back();
        ancestors.push_back(e[0]);
        ancestors.push_back(e[1]);
        baryCoefs.push_back(1.0 - coords_list[i]);
        baryCoefs.push_back(coords_list[i]);
        new_edge_points.push_back(next_point); ++next_point;
    }

    //// STEP 1c : Create last point if necessary

    if (pb == (PointID)-1 && triangles_list.back() != (TriangleID)-1)
    {
        // last point is inside a triangle
        Triangle t = m_container->getTriangle(triangles_list.back());
        p_ancestors.resize(p_ancestors.size()+1);
        sofa::helper::vector< PointID >& ancestors = p_ancestors.back();
        p_baryCoefs.resize(p_baryCoefs.size()+1);
        sofa::helper::vector< double >& baryCoefs = p_baryCoefs.back();
        ancestors.push_back(t[0]);
        ancestors.push_back(t[1]);
        ancestors.push_back(t[2]);
        Vec<3, double> p; p = b;
        baryCoefs = m_geometryAlgorithms->compute3PointsBarycoefs(p, t[0], t[1], t[2]);
        std::cout << "Creating last point in triangle "<<triangles_list.back()<<" barycoefs "<<baryCoefs<<std::endl;
        pb = (next_point); ++next_point;
    }

    //// STEP 2 : Create new triangles, spliting old ones along the new path

    for (unsigned int i = 0; i < triangles_list.size() ; ++i)
    {
        TriangleID tid = triangles_list[i];
        if (tid == (TriangleID) -1) continue;
        Triangle t = m_container->getTriangle(tid);
        // The triangle can be split either :
        // 1- between a vertex and the opposite edge (creating 2 triangles),
        // 2- between two edges (creating 3 triangles),
        // 3- between an inside point and an edge (creating 4 triangles),
        // 4- or between two inside points (creating 5 triangles).
        // The last case is currently not handled.
        if (i == 0 || i == triangles_list.size()-1)
        {
            // point + edge case (1 or 3)
            PointID p = (i == 0) ? pa : pb;
            EdgeID e = (i == 0) ? edges_list.front() : edges_list.back();
            PointID split_p = (i == 0) ? new_edge_points.front() : new_edge_points.back();
            Edge edge = m_container->getEdge(e);
            // find the corner opposite the given edge
            int corner;
            for (corner = 0; corner < 3 && (edge[0]==t[corner] || edge[1]==t[corner]); ++corner) {}
            if (corner == 3)
            {
                this->serr << "ERROR: Degenerate triangle " << tid << " : " << t << this->sendl;
                continue;
            }
            if (p != t[corner])
            {
                // case 3 : create two triangles linking p with the corner
                new_triangles.push_back(Triangle(p, t[corner], t[(corner+1)%3]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(p, t[(corner+2)%3], t[corner]));
                new_triangles_id.push_back(next_triangle++);
            }
            // create two triangles linking p with the splitted edge
            new_triangles.push_back(Triangle(p, t[(corner+1)%3], split_p));
            new_triangles_id.push_back(next_triangle++);
            new_triangles.push_back(Triangle(p, split_p, t[(corner+2)%3]));
            new_triangles_id.push_back(next_triangle++);
        }
        else
        {
            PointID p1 = new_edge_points[i-1];
            EdgeID e1 = edges_list[i-1];
            Edge edge1 = m_container->getEdge(e1);
            Vec<3,double> pos1 = m_geometryAlgorithms->computeBaryEdgePoint(edge1, coords_list[i-1]);
            PointID p2 = new_edge_points[i];
            EdgeID e2 = edges_list[i];
            Edge edge2 = m_container->getEdge(e2);
            Vec<3,double> pos2 = m_geometryAlgorithms->computeBaryEdgePoint(edge2, coords_list[i]);
            // find the corner common to the two edges
            int corner;
            for (corner = 0; corner < 3 && ((edge1[0]!=t[corner] && edge1[1]!=t[corner]) || (edge2[0]!=t[corner] && edge2[1]!=t[corner])); ++corner) {}
            if (corner == 3)
            {
                this->serr << "ERROR: triangle " << tid << " ( " << t << " ) does not contain edges " << e1 << " ( " << edge1 << " ) and " << e2 << " ( " << edge2 << " )." << this->sendl;
                continue;
            }
            PointID p = t[corner];
            // reorder the indices within each edge to put the common corner first
            if (edge1[0] != p)
            {
                edge1[1] = edge1[0];
                edge1[0] = p;
            }
            if (edge2[0] != p)
            {
                edge2[1] = edge2[0];
                edge2[0] = p;
            }
            // swap the edges so that the edge1 is the first edge after p in the order of the triangle indices
            if (edge1[1] != t[(corner+1)%3])
            {
                EdgeID t_e = e1; e1 = e2; e2 = t_e;
                Edge t_edge = edge1; edge1 = edge2; edge2 = t_edge;
                PointID t_p = p1; p1 = p2; p2 = t_p;
                Vec<3, double> t_pos = pos1; pos1 = pos2; pos2 = t_pos;
            }

            // Create the triangle around p
            new_triangles.push_back(Triangle(p, p1, p2));
            new_triangles_id.push_back(next_triangle++);

            // Triangularize the remaining quad according to the delaunay criteria
            Vec<3,double> pos_e1; pos_e1 = m_geometryAlgorithms->getPointPosition(edge1[1]);
            Vec<3,double> pos_e2; pos_e2 = m_geometryAlgorithms->getPointPosition(edge2[1]);
            if (m_geometryAlgorithms->isQuadDeulaunayOriented(pos1, pos_e1, pos_e2, pos2))
            {
                new_triangles.push_back(Triangle(edge1[1], edge2[1], p1));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(p2, p1, edge2[1]));
                new_triangles_id.push_back(next_triangle++);
            }
            else
            {
                new_triangles.push_back(Triangle(edge2[1], p2, edge1[1]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(p1, edge1[1], p2));
                new_triangles_id.push_back(next_triangle++);
            }
        }
        removed_triangles.push_back(tid);
    }

    // FINAL STEP : Apply changes

    // Create all the points registered to be created
    m_modifier->addPointsProcess(p_ancestors.size());

    // Warn for the creation of all the points registered to be created
    m_modifier->addPointsWarning(p_ancestors.size(), p_ancestors, p_baryCoefs);

    // Create all the triangles registered to be created
    m_modifier->addTrianglesProcess(new_triangles); // WARNING called after the creation process by the method "addTrianglesProcess"

    // Warn for the creation of all the triangles registered to be created
    m_modifier->addTrianglesWarning(new_triangles.size(), new_triangles, new_triangles_id);

    // Propagate the topological changes *** not necessary
    m_modifier->propagateTopologicalChanges();

    // Remove all the triangles registered to be removed
    m_modifier->removeTriangles(removed_triangles, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

    // Propagate the topological changes *** not necessary
    //m_modifier->propagateTopologicalChanges();


    for (unsigned int i = 0; i < triangles_list.size() ; ++i)
    {
        TriangleID tid = triangles_list[i];
        if (tid == (TriangleID) -1) continue;
        PointID p1, p2;
        if (i == 0)
        {
            p1 = pa;
            p2 = new_edge_points.front();
        }
        else if (i == triangles_list.size()-1)
        {
            p1 = new_edge_points.back();
            p2 = pb;
        }
        else
        {
            p1 = new_edge_points[i-1];
            p2 = new_edge_points[i];
        }
        EdgeID e = m_container->getEdgeIndex(p1, p2);
        if (e == (EdgeID)-1)
            this->serr << "ERROR: Edge " << p1 << " - " << p2 << " NOT FOUND." << this->sendl;
        else
            new_edges.push_back(e);
    }
    return p_ancestors.size();
}



template<class DataTypes>
void TriangleSetTopologyAlgorithms<DataTypes>::SnapAlongPath (sofa::helper::vector< sofa::core::componentmodel::topology::TopologyObjectType>& topoPath_list,
        sofa::helper::vector<unsigned int>& indices_list, sofa::helper::vector< Vec<3, double> >& coords_list,
        sofa::helper::vector< sofa::helper::vector<double> >& points2Snap)
{

    std::cout << "TriangleSetTopologyAlgorithms::SnapAlongPath()" << std::endl;

    std::cout << "*** Inputs: ***" << std::endl;
    std::cout << "topoPath_list: " << topoPath_list << std::endl;
    std::cout << "indices_list: " << indices_list << std::endl;
    std::cout << "coords_list: " << coords_list << std::endl;
    std::cout << "****************" << std::endl;

    std::map <PointID, sofa::helper::vector<unsigned int> > map_point2snap;
    std::map <PointID, sofa::helper::vector<unsigned int> >::iterator it;
    std::map <PointID, Vec<3,double> > map_point2bary;
    float epsilon = 0.25;

    //// STEP 1 - First loop to find concerned points
    for (unsigned int i = 0; i < indices_list.size(); i++)
    {
        switch ( topoPath_list[i] )
        {
            // New case to handle other topological object can be added.
            // Default: if object is a POINT , nothing has to be done.

        case core::componentmodel::topology::EDGE:
        {
            PointID Vertex2Snap;

            if ( coords_list[i][0] < epsilon )  // This point has to be snaped
            {
                Vertex2Snap = m_container->getEdge(indices_list[i])[0];
                it = map_point2snap.find (Vertex2Snap);
            }
            else if ( coords_list[i][0] > (1.0 - epsilon) )
            {
                Vertex2Snap = m_container->getEdge(indices_list[i])[1];
                it = map_point2snap.find (Vertex2Snap);
            }
            else
            {
                break;
            }

            if (it == map_point2snap.end()) // First time this point is encounter
            {
                map_point2snap[Vertex2Snap] = sofa::helper::vector <unsigned int> ();
                map_point2bary[Vertex2Snap] = Vec<3,double> ();
            }

            break;
        }
        case core::componentmodel::topology::TRIANGLE:  // TODO: NOT TESTED YET!
        {
            PointID Vertex2Snap;
            Vec<3, double>& barycoord = coords_list[i];
            bool TriFind = false;

            for (unsigned int j = 0; j < 3; j++)
            {
                if ( barycoord[j] > (1.0 - epsilon) )  // This point has to be snaped
                {
                    Vertex2Snap = m_container->getTriangleArray()[indices_list[i]][j];
                    it = map_point2snap.find (Vertex2Snap);
                    TriFind = true;
                    break;
                }
            }

            if ( TriFind && (it == map_point2snap.end()) ) // First time this point is encounter
            {
                map_point2snap[Vertex2Snap] = sofa::helper::vector <unsigned int> ();
                map_point2bary[Vertex2Snap] = Vec<3,double> ();
            }

            break;
        }
        default:
            break;
        }
    }


    //// STEP 2 - Test if snaping is needed
    if (map_point2snap.empty())
    {
        std::cout << "EXIT" << std::endl;  // TODO: remove this
        return;
    }

    typename DataTypes::VecCoord& coords = *(m_geometryAlgorithms->getDOF()->getX());


    //// STEP 3 - Second loop necessary to find object on the neighborhood of a snaped point
    for (unsigned int i = 0; i < indices_list.size(); i++)
    {
        switch ( topoPath_list[i] )
        {
        case core::componentmodel::topology::POINT:
        {
            if ( map_point2snap.find (indices_list[i]) != map_point2snap.end() )
            {
                map_point2snap[ indices_list[i] ].push_back(i);

                for (unsigned int j = 0; j<3; j++)
                    map_point2bary[ indices_list[i] ][j] += coords[ indices_list[i] ][j];
            }
            break;
        }
        case core::componentmodel::topology::EDGE:
        {
            Edge theEdge = m_container->getEdge(indices_list[i]);
            bool PointFind = false;

            for (unsigned int indEdge = 0; indEdge < 2; indEdge++)
            {
                PointID thePoint = theEdge[ indEdge ];
                if ( map_point2snap.find (thePoint) != map_point2snap.end() )
                {
                    PointFind = true;
                    map_point2snap[ thePoint ].push_back(i);
                    // Compute new position.
                    // Step 1/3: Compute real coord of incision point on the edge
                    const Vec<3,double>& coord_bary = m_geometryAlgorithms->computeBaryEdgePoint (theEdge[(indEdge+1)%2], thePoint, coords_list[i][0]);

                    // Step 2/3: Sum the different incision point position.
                    for (unsigned int j = 0; j<3; j++)
                        map_point2bary[ thePoint ][j] += coord_bary[j];
                }

                if (PointFind)
                    break;
            }
            break;
        }
        case core::componentmodel::topology::TRIANGLE: // TODO: NOT TESTED YET!
        {
            Triangle theTriangle = m_container->getTriangleArray()[indices_list[i]];
            bool PointFind = false;

            for (unsigned int indTri = 0; indTri < 3; indTri++)
            {
                PointID thePoint = theTriangle[ indTri ];
                if ( map_point2snap.find (thePoint) != map_point2snap.end() )
                {
                    PointFind = true;
                    map_point2snap[ thePoint ].push_back(i);
                    // TODO: check if it is the good function (optional: add comments in header...)
                    const sofa::helper::vector< double >& coord_bary = m_geometryAlgorithms->computeTriangleBarycoefs (indices_list[i], coords_list[i]);

                    for (unsigned int j = 0; j<3; j++)
                        map_point2bary[ thePoint ][j] += coord_bary[j];
                }

                if (PointFind)
                    break;
            }
            break;
        }
        default:
            break;
        }
    }


    //// STEP 4 - Compute new coordinates of point to be snaped, and inform path that point has to be snaped
    sofa::helper::vector<unsigned int> field2remove;
    points2Snap.resize (map_point2snap.size());
    unsigned int cpt = 0;

    for (it = map_point2snap.begin(); it != map_point2snap.end(); ++it)
    {
        points2Snap[ cpt ].push_back ((*it).first); // points2Snap[X][0] => id point to snap

        unsigned int size = ((*it).second).size();
        Vec<3,double> newCoords;

        // Step 3/3: Compute mean value of all incision point position.
        for (unsigned int j = 0; j<3; j++)
        {
            points2Snap[ cpt ].push_back ( map_point2bary[(*it).first][j]/size ); // points2Snap[X][1 2 3] => real coord of point to snap
        }

        cpt++;

        // Change enum of the first object to snap to POINT, change id and label it as snaped
        topoPath_list[ ((*it).second)[0]] = core::componentmodel::topology::POINT;
        indices_list[ ((*it).second)[0]] = (*it).first;
        coords_list[ ((*it).second)[0]][0] = -1.0;

        // If more objects are concerned, remove them from the path  (need to stock and get out of the loop to delete them)
        if (size > 1 )
            for (unsigned int i = 1; i <size; i++)
                field2remove.push_back ((*it).second[i]);
    }


    //// STEP 5 - Modify incision path
    //TODO: verify that one object can't be snaped and considered at staying at the same time
    sort (field2remove.begin(), field2remove.end());

    for (unsigned int i = 1; i <= field2remove.size(); i++) //Delete in reverse order
    {
        topoPath_list.erase (topoPath_list.begin()+field2remove[field2remove.size()-i]);
        indices_list.erase (indices_list.begin()+1+field2remove[field2remove.size()-i]);
        coords_list.erase (coords_list.begin()+field2remove[field2remove.size()-i]);
    }

    return;
}





/** \brief Duplicates the given edges. Only works if at least the first or last point is adjacent to a border.
 * @returns true if the incision succeeded.
 */
template<class DataTypes>
bool TriangleSetTopologyAlgorithms<DataTypes>::InciseAlongEdgeList(const sofa::helper::vector<unsigned int>& edges, sofa::helper::vector<unsigned int>& new_points, sofa::helper::vector<unsigned int>& end_points)
{
    sofa::helper::vector< sofa::helper::vector< PointID > > p_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs;
    PointID next_point = m_container->getNbPoints();
    TriangleID next_triangle = m_container->getNbTriangles();
    sofa::helper::vector< Triangle > new_triangles;
    sofa::helper::vector< TriangleID > new_triangles_id;
    sofa::helper::vector< TriangleID > removed_triangles;

    int nbEdges = edges.size();
    if (nbEdges == 0) return true;
    sofa::helper::vector<PointID> init_points;
    Edge edge;
    edge = m_container->getEdge(edges[0]);
    init_points.push_back(edge[0]);
    init_points.push_back(edge[1]);
    if (nbEdges > 1)
    {
        edge = m_container->getEdge(edges[1]);
        if (init_points[0] == edge[0] || init_points[0] == edge[1])
        {
            // swap the first points
            PointID t = init_points[0];
            init_points[0] = init_points[1];
            init_points[1] = t;
        }
        // add the rest of the points
        for (int i=1; i<nbEdges; ++i)
        {
            edge = m_container->getEdge(edges[i]);
            if (edge[0] == init_points.back())
                init_points.push_back(edge[1]);
            else if (edge[1] == init_points.back())
                init_points.push_back(edge[0]);
            else
            {
                this->serr << "ERROR: edges are not connected after number " << i-1 << " : " << edges << this->sendl;
                return false;
            }
        }
    }

    this->sout << "Points on the path: " << init_points << this->sendl;

    sofa::helper::vector< std::pair<TriangleID,TriangleID> > init_triangles;
    for (int i=0; i<nbEdges; ++i)
    {
        const sofa::helper::vector<TriangleID>& shell = m_container->getTriangleEdgeShell(edges[i]);
        if (shell.size() != 2)
        {
            this->serr << "ERROR: cannot split an edge with " << shell.size() << "!=2 attached triangles." << this->sendl;
            return false;
        }
        init_triangles.push_back(std::make_pair(shell[0],shell[1]));
    }

    bool beginOnBorder = (m_container->getTriangleVertexShell(init_points.front()).size() < m_container->getEdgeVertexShell(init_points.front()).size());
    bool endOnBorder = (m_container->getTriangleVertexShell(init_points.back()).size() < m_container->getEdgeVertexShell(init_points.back()).size());
    if (!beginOnBorder && !endOnBorder && nbEdges == 1)
    {
        this->serr << "ERROR: cannot split a single edge not on the border." << this->sendl;
        return false;
    }

    if (!beginOnBorder) end_points.push_back(init_points.front());
    if (!endOnBorder) end_points.push_back(init_points.back());
    this->sout << "End points : " << end_points << this->sendl;

    /// STEP 1: Create the new points corresponding the one of the side of the now separated edges
    int first_new_point = beginOnBorder ? 0 : 1;
    int last_new_point = endOnBorder ? init_points.size()-1 : init_points.size()-2;
    std::map<PointID, PointID> splitMap;
    for (int i = first_new_point ; i <= last_new_point ; ++i)
    {
        PointID p = init_points[i];
        p_ancestors.resize(p_ancestors.size()+1);
        sofa::helper::vector< PointID >& ancestors = p_ancestors.back();
        p_baryCoefs.resize(p_baryCoefs.size()+1);
        sofa::helper::vector< double >& baryCoefs = p_baryCoefs.back();
        ancestors.push_back(p);
        baryCoefs.push_back(1.0);
        new_points.push_back(next_point);
        splitMap[p] = next_point;
        ++next_point;
    }

    // STEP 2: Find all triangles that need to be attached to the new points
    std::set<TriangleID> updatedTriangles;

    TriangleID t0 = m_container->getTriangleEdgeShell(edges[0])[0];
    if (beginOnBorder)
    {
        // STEP 2a: Find the triangles linking the first edge to the border
        TriangleID tid = t0;
        PointID p0 = init_points[0];
        PointID p1 = init_points[1];
        for(;;)
        {
            updatedTriangles.insert(tid);
            Triangle t = m_container->getTriangle(tid);
            PointID p2 = getOtherPointInTriangle(t, p0, p1);
            EdgeID e = m_container->getEdgeIndex(p0, p2);
            const sofa::core::componentmodel::topology::BaseMeshTopology::EdgeTriangles& etri = m_container->getTriangleEdgeShell(e);
            if (etri.size() != 2) break; // border or non-manifold edge
            if (etri[0] == tid)
                tid = etri[1];
            else
                tid = etri[0];
            p1 = p2;
        }
    }

    // STEP 2b: Find the triangles linking each edge to the next, by starting from the last triangle, rotate around each point until the next point is reached

    for (int i = 0 ; i < nbEdges-1 ; ++i)
    {
        PointID p1 = init_points[i];
        PointID p0 = init_points[i+1];
        PointID pnext = init_points[i+2];
        TriangleID tid = t0;
        for (;;)
        {
            updatedTriangles.insert(tid);
            Triangle t = m_container->getTriangle(tid);
            PointID p2 = getOtherPointInTriangle(t, p0, p1);
            if (p2 == pnext) break;
            EdgeID e = m_container->getEdgeIndex(p0, p2);
            const sofa::core::componentmodel::topology::BaseMeshTopology::EdgeTriangles& etri = m_container->getTriangleEdgeShell(e);
            if (etri.size() < 2) break; // border or non-manifold edge
            if (etri[0] == tid)
                tid = etri[1];
            else
                tid = etri[0];
            p1 = p2;
        }
        t0 = tid;
    }

    if (endOnBorder)
    {
        // STEP 2c: Find the triangles linking the last edge to the border
        TriangleID tid = t0;
        PointID p0 = init_points[nbEdges];
        PointID p1 = init_points[nbEdges-1];
        for(;;)
        {
            updatedTriangles.insert(tid);
            Triangle t = m_container->getTriangle(tid);
            PointID p2 = getOtherPointInTriangle(t, p0, p1);
            EdgeID e = m_container->getEdgeIndex(p0, p2);
            const sofa::core::componentmodel::topology::BaseMeshTopology::EdgeTriangles& etri = m_container->getTriangleEdgeShell(e);
            if (etri.size() != 2) break; // border or non-manifold edge
            if (etri[0] == tid)
                tid = etri[1];
            else
                tid = etri[0];
            p1 = p2;
        }
    }

    // STEP 3: Create new triangles by replacing indices of split points in the list of triangles to update

    for (std::set<TriangleID>::const_iterator it = updatedTriangles.begin(), itend = updatedTriangles.end(); it != itend; ++it)
    {
        TriangleID tid = *it;
        Triangle t = m_container->getTriangle(tid);
        bool changed = false;
        for (int c = 0; c < 3; ++c)
        {
            std::map<PointID, PointID>::iterator itsplit = splitMap.find(t[c]);
            if (itsplit != splitMap.end())
            {
                t[c] = itsplit->second;
                changed = true;
            }
        }
        if (!changed)
        {
            this->serr << "ERROR: Triangle " << tid << " ( " << t << " ) was flagged as updated but no change was found." << this->sendl;
        }
        else
        {
            new_triangles.push_back(t);
            new_triangles_id.push_back(next_triangle++);
            removed_triangles.push_back(tid);
        }
    }

    // FINAL STEP : Apply changes

    // Create all the points registered to be created
    m_modifier->addPointsProcess(p_ancestors.size());

    // Warn for the creation of all the points registered to be created
    m_modifier->addPointsWarning(p_ancestors.size(), p_ancestors, p_baryCoefs);

    // Create all the triangles registered to be created
    m_modifier->addTrianglesProcess(new_triangles); // WARNING called after the creation process by the method "addTrianglesProcess"

    // Warn for the creation of all the triangles registered to be created
    m_modifier->addTrianglesWarning(new_triangles.size(), new_triangles, new_triangles_id);

    // Propagate the topological changes *** not necessary
    m_modifier->propagateTopologicalChanges();

    // Remove all the triangles registered to be removed
    m_modifier->removeTriangles(removed_triangles, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

    // Propagate the topological changes *** not necessary
    //m_modifier->propagateTopologicalChanges();

    return true;
}



// Duplicate the given edge. Only works of at least one of its points is adjacent to a border.
template<class DataTypes>
int TriangleSetTopologyAlgorithms<DataTypes>::InciseAlongEdge(unsigned int ind_edge, int* createdPoints)
{
    const Edge & edge0=m_container->getEdge(ind_edge);
    unsigned ind_pa = edge0[0];
    unsigned ind_pb = edge0[1];

    const helper::vector<unsigned>& triangles0 = m_container->getTriangleEdgeShell(ind_edge);
    if (triangles0.size() != 2)
    {
        this->serr << "InciseAlongEdge: ERROR edge "<<ind_edge<<" is not attached to 2 triangles." << this->sendl;
        return -1;
    }

    // choose one triangle
    unsigned ind_tri0 = triangles0[0];

    unsigned ind_tria = ind_tri0;
    unsigned ind_trib = ind_tri0;
    unsigned ind_edgea = ind_edge;
    unsigned ind_edgeb = ind_edge;

    helper::vector<unsigned> list_tria;
    helper::vector<unsigned> list_trib;

    for (;;)
    {
        const TriangleEdges& te = m_container->getTriangleEdge(ind_tria);

        // find the edge adjacent to a that is not ind_edgea
        int j=0;
        for (j=0; j<3; ++j)
        {
            if (te[j] != ind_edgea && (m_container->getEdge(te[j])[0] == ind_pa || m_container->getEdge(te[j])[1] == ind_pa))
                break;
        }
        if (j == 3)
        {
            this->serr << "InciseAlongEdge: ERROR in triangle "<<ind_tria<<this->sendl;
            return -1;
        }

        ind_edgea = te[j];
        if (ind_edgea == ind_edge)
            break; // full loop

        const helper::vector<unsigned>& tes = m_container->getTriangleEdgeShell(ind_edgea);
        if(tes.size() < 2)
            break; // border edge

        if (tes[0] == ind_tria)
            ind_tria = tes[1];
        else
            ind_tria = tes[0];
        list_tria.push_back(ind_tria);
    }

    for (;;)
    {
        const TriangleEdges& te = m_container->getTriangleEdge(ind_trib);

        // find the edge adjacent to b that is not ind_edgeb
        int j=0;
        for (j=0; j<3; ++j)
        {
            if (te[j] != ind_edgeb && (m_container->getEdge(te[j])[0] == ind_pb || m_container->getEdge(te[j])[1] == ind_pb))
                break;
        }
        if (j == 3)
        {
            this->serr << "InciseAlongEdge: ERROR in triangle "<<ind_trib<<this->sendl;
            return -1;
        }

        ind_edgeb = te[j];
        if (ind_edgeb == ind_edge)
            break; // full loop

        const helper::vector<unsigned>& tes = m_container->getTriangleEdgeShell(ind_edgeb);
        if(tes.size() < 2)
            break; // border edge

        if (tes[0] == ind_trib)
            ind_trib = tes[1];
        else
            ind_trib = tes[0];
        list_trib.push_back(ind_trib);
    }

    bool pa_is_on_border = (ind_edgea != ind_edge);
    bool pb_is_on_border = (ind_edgeb != ind_edge);

    if (!pa_is_on_border && !pb_is_on_border)
    {
        this->serr << "InciseAlongEdge: ERROR edge "<<ind_edge<<" is not on border." << this->sendl;
        return -1;
    }

    // now we can split the edge

    /// force the creation of TriangleEdgeShellArray
    m_container->getTriangleEdgeShellArray();
    /// force the creation of TriangleVertexShellArray
    m_container->getTriangleVertexShellArray();

    //const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();
    unsigned int nb_points =  m_container->getTriangleVertexShellArray().size(); //vect_c.size();
    const sofa::helper::vector<Triangle> &vect_t=m_container->getTriangleArray();
    unsigned int nb_triangles =  vect_t.size();

    // Variables to accumulate the number of elements registered to be created (so as to remember their indices)
    unsigned int acc_nb_points=nb_points;
    unsigned int acc_nb_triangles=nb_triangles;

    // Variables to accumulate the elements registered to be created or to be removed
    sofa::helper::vector< sofa::helper::vector< unsigned int > > p_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs;
    sofa::helper::vector< Triangle > triangles_to_create;
    sofa::helper::vector< unsigned int > trianglesIndexList;
    sofa::helper::vector< unsigned int > triangles_to_remove;

    sofa::helper::vector<double> defaultCoefs; defaultCoefs.push_back(1.0);

    unsigned new_pa, new_pb;

    if (pa_is_on_border)
    {
        sofa::helper::vector<unsigned int> ancestors;
        new_pa = acc_nb_points++;
        ancestors.push_back(ind_pa);
        p_ancestors.push_back(ancestors);
        p_baryCoefs.push_back(defaultCoefs);
        if (createdPoints) *(createdPoints++) = new_pa;
    }
    else
        new_pa = ind_pa;

    sofa::helper::vector<unsigned int> ancestors(1);

    if (pb_is_on_border)
    {
        new_pb = acc_nb_points++;
        ancestors[0] = ind_pb;
        p_ancestors.push_back(ancestors);
        p_baryCoefs.push_back(defaultCoefs);
        if (createdPoints) *(createdPoints++) = new_pb;
    }
    else
        new_pb = ind_pb;

    // we need to recreate at least tri0
    Triangle new_tri0 = m_container->getTriangle(ind_tri0);
    for (unsigned i=0; i<3; i++)
    {
        if (new_tri0[i] == ind_pa)
            new_tri0[i] = new_pa;
        else if (new_tri0[i] == ind_pb)
            new_tri0[i] = new_pb;
    }

    triangles_to_remove.push_back(ind_tri0);
    ancestors[0] = ind_tri0;
    triangles_to_create.push_back(new_tri0);

    trianglesIndexList.push_back(acc_nb_triangles);
    acc_nb_triangles += 1;

    // recreate list_tria iff pa is new
    if (new_pa != ind_pa)
    {
        for (unsigned j=0; j<list_tria.size(); j++)
        {
            unsigned ind_tri = list_tria[j];
            Triangle new_tri = m_container->getTriangle(ind_tri);
            for (unsigned i=0; i<3; i++)
                if (new_tri[i] == ind_pa) new_tri[i] = new_pa;
            triangles_to_remove.push_back(ind_tri);
            ancestors[0] = ind_tri;
            triangles_to_create.push_back(new_tri);

            trianglesIndexList.push_back(acc_nb_triangles);
            acc_nb_triangles+=1;
        }
    }

    // recreate list_trib iff pb is new
    if (new_pb != ind_pb)
    {
        for (unsigned j=0; j<list_trib.size(); j++)
        {
            unsigned ind_tri = list_trib[j];
            Triangle new_tri = m_container->getTriangle(ind_tri);
            for (unsigned i=0; i<3; i++)
                if (new_tri[i] == ind_pb) new_tri[i] = new_pb;
            triangles_to_remove.push_back(ind_tri);
            ancestors[0] = ind_tri;
            triangles_to_create.push_back(new_tri);

            trianglesIndexList.push_back(acc_nb_triangles);
            acc_nb_triangles+=1;
        }
    }

    // Create all the points registered to be created
    m_modifier->addPointsProcess((const unsigned int) acc_nb_points - nb_points);

    // Warn for the creation of all the points registered to be created
    m_modifier->addPointsWarning((const unsigned int) acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

    // Create all the triangles registered to be created
    m_modifier->addTrianglesProcess((const sofa::helper::vector< Triangle > &) triangles_to_create) ; // WARNING called after the creation process by the method "addTrianglesProcess"

    // Warn for the creation of all the triangles registered to be created
    m_modifier->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList);

    // Propagate the topological changes *** not necessary
    //m_modifier->propagateTopologicalChanges();

    // Remove all the triangles registered to be removed
    m_modifier->removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

    // Propagate the topological changes *** not necessary
    //m_modifier->propagateTopologicalChanges();

    return (pb_is_on_border?1:0)+(pa_is_on_border?1:0); // todo: get new edge indice
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TriangleSetTOPOLOGY_INL
