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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_INL

#include <sofa/component/topology/TriangleSetTopologyAlgorithms.h>
#include <sofa/component/topology/TriangleSetTopology.h>

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
void TriangleSetTopologyAlgorithms< DataTypes >::removeTriangles(sofa::helper::vector< unsigned int >& triangles,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    /// add the topological changes in the queue
    m_modifier->removeTrianglesWarning(triangles);
    // inform other objects that the triangles are going to be removed
    m_container->propagateTopologicalChanges();
    // now destroy the old triangles.
    m_modifier->removeTrianglesProcess(  triangles ,removeIsolatedEdges, removeIsolatedPoints);

    m_container->checkTopology();
}

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeTriangles(items, true, true);
}

template<class DataTypes>
void  TriangleSetTopologyAlgorithms<DataTypes>::renumberPoints( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index)
{
    /// add the topological changes in the queue
    m_modifier->renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    m_container->propagateTopologicalChanges();
    // now renumber the points
    m_modifier->renumberPointsProcess(index, inv_index);

    m_container->checkTopology();
}


// Move and fix the two closest points of two triangles to their median point
template<class DataTypes>
bool TriangleSetTopologyAlgorithms< DataTypes >::Suture2Points(unsigned int ind_ta, unsigned int ind_tb,
        unsigned int &ind1, unsigned int &ind2)
{
    // Access the topology
    m_geometryAlgorithms->closestIndexPair(ind_ta, ind_tb, ind1, ind2);

    //std::cout << "INFO_print : ind1 = " << ind1 << std::endl;
    //std::cout << "INFO_print : ind2 = " << ind2 << std::endl;

    sofa::helper::vector< unsigned int > indices;
    indices.push_back(ind1); indices.push_back(ind2);

    Vec<3,Real> point_created=(Vec<3,double>) m_geometryAlgorithms->computeBaryEdgePoint(indices, (double) 0.5);

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

// Incises along the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
// Point a belongs to the triangle sindexed by ind_ta
// Point b belongs to the triangle sindexed by ind_tb

template<class DataTypes>
bool TriangleSetTopologyAlgorithms< DataTypes >::InciseAlongPointsList(bool is_first_cut,
        const Vec<3,double>& a,
        const Vec<3,double>& b,
        const unsigned int ind_ta,
        const unsigned int ind_tb,
        unsigned int& a_last,
        sofa::helper::vector< unsigned int > &a_p12_last,
        sofa::helper::vector< unsigned int > &a_i123_last,
        unsigned int& b_last,
        sofa::helper::vector< unsigned int > &b_p12_last,
        sofa::helper::vector< unsigned int > &b_i123_last,
        sofa::helper::vector< sofa::helper::vector<unsigned int> > &new_points,
        sofa::helper::vector< sofa::helper::vector<unsigned int> > &closest_vertices)
{

    double epsilon = 0.2; // INFO : epsilon is a threshold in [0,1] to control the snapping of the extremities to the closest vertex

    unsigned int x_i1 = 0;
    unsigned int x_i2 = 0;
    unsigned int x_i3 = 0;
    unsigned int x_i1_to = 0;
    unsigned int x_i2_to = 0;
    unsigned int x_p1 = 0;
    unsigned int x_p2 = 0;
    unsigned int x_p1_to = 0;
    unsigned int x_p2_to = 0;

    Vec<3,Real> a_new = a;
    unsigned int ind_ta_new = ind_ta;
    Vec<3,Real> b_new = b;
    unsigned int ind_tb_new = ind_tb;

    unsigned int ind_ta_test_init;
    unsigned int ind_tb_test_init;

    unsigned int &ind_ta_test = ind_ta_test_init;
    unsigned int &ind_tb_test = ind_tb_test_init;

    unsigned int ind_tb_final_init;
    unsigned int &ind_tb_final = ind_tb_final_init;

    bool is_on_boundary_init=false;
    bool &is_on_boundary=is_on_boundary_init;

    if(is_first_cut)
    {
        bool is_a_inside = m_geometryAlgorithms->is_PointinTriangle(true, a_new, ind_ta_new, ind_ta_test);
        if(is_a_inside)
        {
            //std::cout << "a is inside" <<  std::endl;
        }
        else
        {
            //std::cout << "a is NOT inside !!!" <<  std::endl;
            if(ind_ta_new == ind_ta_test) // fail
            {
                //std::cout << "fail !!!" <<  std::endl;
                return false;
            }
            else
            {
                ind_ta_new=ind_ta_test;
            }
        }
    }

    bool is_b_inside = m_geometryAlgorithms->is_PointinTriangle(true, b_new, ind_tb_new, ind_tb_test);
    if(is_b_inside)
    {
        //std::cout << "b is inside" <<  std::endl;
    }
    else
    {
        //std::cout << "b is NOT inside !!!" <<  std::endl;
        if(ind_tb_new==ind_tb_test) // fail
        {
            //std::cout << "fail !!!" <<  std::endl;
            return false;
        }
        else
        {
            ind_tb_new=ind_tb_test;
        }
    }

    if(ind_ta_new==ind_tb_new)
    {
        return false;
    }

    //const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();
    unsigned int nb_points =  m_container->getTriangleVertexShellArray().size() - 1; //vect_c.size() -1;

    const sofa::helper::vector<Triangle> &vect_t = m_container->getTriangleArray();
    unsigned int nb_triangles =  vect_t.size() -1;

    // Variables to accumulate the number of elements registered to be created (so as to remember their indices)
    unsigned int acc_nb_points = nb_points;
    unsigned int acc_nb_triangles = nb_triangles;

    // Variables to accumulate the elements registered to be created or to be removed
    sofa::helper::vector< sofa::helper::vector< unsigned int > > p_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs;
    sofa::helper::vector< Triangle > triangles_to_create;
    sofa::helper::vector< unsigned int > trianglesIndexList;
    sofa::helper::vector< unsigned int > triangles_to_remove;

    // Initialization for INTERSECTION method
    sofa::helper::vector< unsigned int > triangles_list_init;
    sofa::helper::vector< unsigned int > &triangles_list = triangles_list_init;

    sofa::helper::vector< sofa::helper::vector<unsigned int> > indices_list_init;
    sofa::helper::vector< sofa::helper::vector<unsigned int> > &indices_list = indices_list_init;

    sofa::helper::vector< double > coords_list_init;
    sofa::helper::vector< double >& coords_list=coords_list_init;

    bool is_intersected=false;

    // Pre-treatment if is_first_cut ==false :

    if(!is_first_cut)
    {
        x_p1 = b_p12_last[0];
        x_p2 = b_p12_last[1];
        x_i1 = b_i123_last[0];
        x_i2 = b_i123_last[1];
        x_i3 = b_i123_last[2];

        const typename DataTypes::Coord& b_point_last = m_geometryAlgorithms->getPositionPoint(b_last); //vect_c[b_last];

        a_new[0]= (Real) b_point_last[0];
        a_new[1]= (Real) b_point_last[1];
        a_new[2]= (Real) b_point_last[2];

        bool is_crossed = false;
        double coord_kmin = 0.0;

        const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=m_container->getTriangleVertexShellArray();
        if (tvsa.size()>0)
        {
            sofa::helper::vector< unsigned int > shell_b =(sofa::helper::vector< unsigned int >) (tvsa[b_last]);
            unsigned int ind_t_test;
            unsigned int i=0;

            if(shell_b.size()>0)
            {

                while(i < shell_b.size())
                {
                    ind_t_test=shell_b[i];
                    triangles_to_remove.push_back(ind_t_test);

                    sofa::helper::vector<unsigned int> c_indices_init; sofa::helper::vector<unsigned int> &c_indices=c_indices_init;
                    double c_baryCoef_init; double &c_baryCoef=c_baryCoef_init;
                    double c_coord_k_init; double &c_coord_k = c_coord_k_init;

                    bool is_intersection_found = m_geometryAlgorithms
                            ->computeSegmentTriangleIntersection(false,
                                    (const Vec<3,double>&) a_new,
                                    (const Vec<3,double>&) b,
                                    ind_t_test,
                                    c_indices, c_baryCoef, c_coord_k);

                    if(is_intersection_found)
                    {
                        is_intersection_found=is_intersection_found && (c_indices[0] != b_last && c_indices[1] != b_last);
                    }

                    if(is_intersection_found && c_coord_k>coord_kmin)
                    {
                        ind_ta_new=ind_t_test;
                        coord_kmin=c_coord_k;
                    }

                    is_crossed = is_crossed || is_intersection_found;

                    i++;
                }

                if(is_crossed)
                {
                    if(ind_ta_new==ind_tb_new)
                    {
                        return false;
                    }

                    ind_tb_final=ind_tb_new;
                    is_intersected = m_geometryAlgorithms
                            ->computeIntersectedPointsList((const Vec<3,double>&) a_new, b,
                                    ind_ta_new, ind_tb_final,
                                    triangles_list, indices_list,
                                    coords_list, is_on_boundary);
                }
                else
                {
                    return false;
                }
            }
        }
    }
    else
    {

        // Call the method "computeIntersectedPointsList" to get the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
        ind_tb_final=ind_tb_new;
        is_intersected = m_geometryAlgorithms
                ->computeIntersectedPointsList(a, b, ind_ta_new, ind_tb_final,
                        triangles_list, indices_list,
                        coords_list, is_on_boundary);
    }

    unsigned int elem_size = triangles_list.size();

    if(elem_size>0) // intersection successfull
    {
        /// force the creation of TriangleEdgeShellArray
        m_container->getTriangleEdgeShellArray();
        /// force the creation of TriangleVertexShellArray
        m_container->getTriangleVertexShellArray();

        // Initialization for the indices of the previous intersected edge
        unsigned int p1_prev = 0;
        unsigned int p2_prev = 0;

        // Treatment of particular case for first extremity a

        const Triangle &ta = m_container->getTriangle(ind_ta_new);
        unsigned int ta_to_remove;
        unsigned int p1_a = indices_list[0][0];
        unsigned int p2_a = indices_list[0][1];

        // Plan to remove triangles indexed by ind_ta_new
        if(is_first_cut)
        {
            triangles_to_remove.push_back(ind_ta_new);
        }

        // Initialization for SNAPPING method for point a

        bool is_snap_a0_init=false; bool is_snap_a1_init=false; bool is_snap_a2_init=false;
        bool& is_snap_a0=is_snap_a0_init;
        bool& is_snap_a1=is_snap_a1_init;
        bool& is_snap_a2=is_snap_a2_init;

        sofa::helper::vector< double > a_baryCoefs = m_geometryAlgorithms
                ->computeTriangleBarycoefs((const Vec<3,double> &) a_new, ind_ta_new);
        snapping_test_triangle(epsilon, a_baryCoefs[0], a_baryCoefs[1], a_baryCoefs[2],
                is_snap_a0, is_snap_a1, is_snap_a2);

        double is_snapping_a = is_snap_a0 || is_snap_a1 || is_snap_a2;

        //std::cout << "a_baryCoefs = " << a_baryCoefs[0] << ", " << a_baryCoefs[1] << ", " << a_baryCoefs[2] <<  std::endl;

        sofa::helper::vector< unsigned int > a_first_ancestors;
        sofa::helper::vector< double > a_first_baryCoefs;

        if((is_first_cut) && (!is_snapping_a))
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

            a_last=ind_a; // OUPTUT

            a_p12_last.clear();
            a_p12_last.push_back(acc_nb_points+2); // OUPTUT
            a_p12_last.push_back(acc_nb_points+1); // OUPTUT

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

            a_i123_last.clear();
            a_i123_last.push_back(p2_a); // OUPTUT
            a_i123_last.push_back(p1_a); // OUPTUT

            if(ta[0]!=p1_a && ta[0]!=p2_a)
            {
                ta_to_remove=acc_nb_triangles-1;
                a_i123_last.push_back(ta[0]); // OUPTUT

            }
            else
            {
                if(ta[1]!=p1_a && ta[1]!=p2_a)
                {
                    ta_to_remove=acc_nb_triangles;
                    a_i123_last.push_back(ta[1]); // OUPTUT
                }
                else
                {
                    // (ta[2]!=p1_a && ta[2]!=p2_a)
                    ta_to_remove=acc_nb_triangles-2;
                    a_i123_last.push_back(ta[2]); // OUPTUT
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
        else
        {
            // (is_first_cut == false) or : snapping a to the vertex indexed by ind_a, which is the closest to point a
            x_p1_to = acc_nb_points + 1;
            x_p2_to = acc_nb_points + 2;

            x_i1_to = p1_a;
            x_i2_to = p2_a;

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
                else // ta[2]!=p1_a && ta[2]!=p2_a
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

            Triangle t_pa1;
            Triangle t_pa2;

            if(ind_a==p1_a)
            {
                t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,
                        (unsigned int) p0_a,
                        (unsigned int) p1_a));
                t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,
                        (unsigned int) p2_a,
                        (unsigned int) p0_a));
                triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);
            }
            else
            {
                if(ind_a==p2_a)
                {
                    t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,
                            (unsigned int) p0_a,
                            (unsigned int) p1_a));
                    t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,
                            (unsigned int) p2_a,
                            (unsigned int) p0_a));
                    triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);
                }
                else
                {
                    t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,
                            (unsigned int) ind_a,
                            (unsigned int) p1_a));
                    t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,
                            (unsigned int) p2_a,
                            (unsigned int)ind_a));
                    triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);
                }
            }

            trianglesIndexList.push_back(acc_nb_triangles); trianglesIndexList.push_back(acc_nb_triangles+1);
            acc_nb_triangles+=2;

            if(!is_first_cut)
            {
                triangles_to_remove.push_back(acc_nb_triangles-1);
                triangles_to_remove.push_back(acc_nb_triangles);
            }
        }

        // Traverse the loop of interected edges

        for (unsigned int i=0; i<indices_list.size(); i++)
        {
            /// Register the creation of the two points (say current "duplicated points") localized on the current interected edge

            unsigned int p1 = indices_list[i][0];
            unsigned int p2 = indices_list[i][1];

            sofa::helper::vector< unsigned int > p_first_ancestors;
            p_first_ancestors.push_back(p1); p_first_ancestors.push_back(p2);
            p_ancestors.push_back(p_first_ancestors); p_ancestors.push_back(p_first_ancestors);

            sofa::helper::vector< double > p_first_baryCoefs;
            p_first_baryCoefs.push_back(1.0 - coords_list[i]); p_first_baryCoefs.push_back(coords_list[i]);
            p_baryCoefs.push_back(p_first_baryCoefs); p_baryCoefs.push_back(p_first_baryCoefs);

            acc_nb_points=acc_nb_points+2;

            sofa::helper::vector<unsigned int> new_points_current;
            new_points_current.push_back(acc_nb_points-1); new_points_current.push_back(acc_nb_points);
            new_points.push_back(new_points_current);
            closest_vertices.push_back(indices_list[i]);

            if(i>0) // not to treat particular case of first extremitiy
            {
                // SNAPPING TEST

                double gamma = 0.3;
                bool is_snap_p1;
                bool is_snap_p2;

                snapping_test_edge(gamma, 1.0 - coords_list[i], coords_list[i], is_snap_p1, is_snap_p2);
                double is_snapping_p = is_snap_p1 || is_snap_p2;

                unsigned int ind_p;

                if(is_snapping_p && i<indices_list.size()-1) // not to treat particular case of last extremitiy
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

                    // std::cout << "INFO_print : DO is_snapping_p, i = " << i << " on vertex " << ind_p <<  std::endl;

                    sofa::helper::vector< unsigned int > triangles_list_1_init;
                    sofa::helper::vector< unsigned int > &triangles_list_1 = triangles_list_1_init;

                    sofa::helper::vector< unsigned int > triangles_list_2_init;
                    sofa::helper::vector< unsigned int > &triangles_list_2 = triangles_list_2_init;

                    // std::cout << "INFO_print : DO Prepare_VertexDuplication " <<  std::endl;
                    m_geometryAlgorithms->Prepare_VertexDuplication(ind_p, triangles_list[i], triangles_list[i+1], indices_list[i-1], coords_list[i-1], indices_list[i+1], coords_list[i+1], triangles_list_1, triangles_list_2);
                    // std::cout << "INFO_print : DONE Prepare_VertexDuplication " <<  std::endl;

                    // std::cout << "INFO_print : triangles_list_1.size() = " << triangles_list_1.size() <<  std::endl;
                    //for (unsigned int k=0;k<triangles_list_1.size();k++){
                    // std::cout << "INFO_print : triangles_list_1 number " << k << " = " << triangles_list_1[k] <<  std::endl;
                    //}

                    // std::cout << "INFO_print : triangles_list_2.size() = " << triangles_list_2.size() <<  std::endl;
                    //for (unsigned int k=0;k<triangles_list_2.size();k++){
                    // std::cout << "INFO_print : triangles_list_2 number " << k << " = " << triangles_list_2[k] <<  std::endl;
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

                Triangle t_p1;
                Triangle t_p2;
                Triangle t_p3;

                unsigned int ind_quad;
                Vec<3,Real> point_created=(Vec<3,double>) m_geometryAlgorithms->computeBaryEdgePoint(indices_list[i-1], coords_list[i-1]);
                Vec<3,Real> point_to_create=(Vec<3,double>) m_geometryAlgorithms->computeBaryEdgePoint(indices_list[i], coords_list[i]);

                if(p0_t!=p1_prev && p0_t!=p2_prev)
                {
                    ind_quad=p0_t;
                }
                else
                {
                    if(p1_t!=p1_prev && p1_t!=p2_prev)
                    {
                        ind_quad=p1_t;
                    }
                    else // (p2_t!=p1_prev && p2_t!=p2_prev)
                    {
                        ind_quad=p2_t;
                    }
                }

                if(ind_quad==p1) // *** p1_to_create - p1_created - p1_prev - ind_quad
                {

                    t_p1 = Triangle(helper::make_array<unsigned int>(p2_created, p2_to_create, p2_prev));
                    if(m_geometryAlgorithms->isQuadDeulaunayOriented(point_to_create, point_created, p1_prev, ind_quad))
                    {
                        t_p2 = Triangle(helper::make_array<unsigned int>(p1_to_create, p1_created, p1_prev));
                        t_p3 = Triangle(helper::make_array<unsigned int>(p1_prev, ind_quad, p1_to_create));
                    }
                    else
                    {
                        t_p2 = Triangle(helper::make_array<unsigned int>(p1_created, p1_prev, ind_quad));
                        t_p3 = Triangle(helper::make_array<unsigned int>(ind_quad, p1_to_create, p1_created));
                    }
                }
                else // ind_quad==p2 // *** p2_created - p2_to_create - ind_quad - p2_prev
                {
                    t_p1 = Triangle(helper::make_array<unsigned int>(p1_to_create, p1_created, p1_prev));
                    if(m_geometryAlgorithms->isQuadDeulaunayOriented(point_created, point_to_create, ind_quad, p2_prev))
                    {
                        t_p2 = Triangle(helper::make_array<unsigned int>(p2_created, p2_to_create, ind_quad));
                        t_p3 = Triangle(helper::make_array<unsigned int>(ind_quad, p2_prev, p2_created));
                    }
                    else
                    {
                        t_p2 = Triangle(helper::make_array<unsigned int>(p2_to_create, ind_quad, p2_prev));
                        t_p3 = Triangle(helper::make_array<unsigned int>(p2_prev, p2_created, p2_to_create));
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

        if(is_intersected || !is_on_boundary)
        {
            ind_tb_new=ind_tb_final;

            b_p12_last.clear();
            b_p12_last.push_back(acc_nb_points-1); // OUPTUT
            b_p12_last.push_back(acc_nb_points); // OUPTUT

            // Treatment of particular case for second extremity b

            const Triangle &tb=m_container->getTriangle(ind_tb_new);
            unsigned int tb_to_remove;
            unsigned int p1_b=indices_list[indices_list.size()-1][0];
            unsigned int p2_b=indices_list[indices_list.size()-1][1];

            b_i123_last.clear();
            b_i123_last.push_back(p1_b); // OUPTUT
            b_i123_last.push_back(p2_b); // OUPTUT

            // Plan to remove triangles indexed by ind_tb_new
            triangles_to_remove.push_back(ind_tb_new);

            // Initialization for SNAPPING method for point b

            bool is_snap_b0_init=false; bool is_snap_b1_init=false; bool is_snap_b2_init=false;
            bool& is_snap_b0=is_snap_b0_init;
            bool& is_snap_b1=is_snap_b1_init;
            bool& is_snap_b2=is_snap_b2_init;

            sofa::helper::vector< double > b_baryCoefs = m_geometryAlgorithms->computeTriangleBarycoefs((const Vec<3,double> &) b, ind_tb_new);

            if(!is_intersected && !is_on_boundary)
            {
                b_baryCoefs[0] = (double) (1.0/3.0);
                b_baryCoefs[1] = (double) (1.0/3.0);
                b_baryCoefs[2] = (double) (1.0 - (b_baryCoefs[0] + b_baryCoefs[1]));
            }

            snapping_test_triangle(epsilon, b_baryCoefs[0], b_baryCoefs[1], b_baryCoefs[2],
                    is_snap_b0, is_snap_b1, is_snap_b2);

            double is_snapping_b = is_snap_b0 || is_snap_b1 || is_snap_b2;

            //std::cout << "b_baryCoefs = " << b_baryCoefs[0] << ", " << b_baryCoefs[1] << ", " << b_baryCoefs[2] <<  std::endl;

            sofa::helper::vector< unsigned int > b_first_ancestors;
            sofa::helper::vector< double > b_first_baryCoefs;

            is_snapping_b = false; // COMMENT : point b will not be snapped

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

                b_last=ind_b; // OUPTUT

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
                    b_i123_last.push_back(tb[0]); // OUTPUT
                }
                else
                {
                    if(tb[1]!=p1_b && tb[1]!=p2_b)
                    {
                        tb_to_remove=acc_nb_triangles;
                        b_i123_last.push_back(tb[1]); // OUTPUT
                    }
                    else // (tb[2]!=p1_b && tb[2]!=p2_b)
                    {
                        tb_to_remove=acc_nb_triangles-2;
                        b_i123_last.push_back(tb[2]); // OUTPUT
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
                    b_i123_last.push_back(tb[0]); // OUTPUT
                }
                else
                {
                    if(tb[1]!=p1_b && tb[1]!=p2_b)
                    {
                        p0_b=tb[1];
                        b_i123_last.push_back(tb[1]); // OUTPUT
                    }
                    else// tb[2]!=p1_b && tb[2]!=p2_b
                    {
                        p0_b=tb[2];
                        b_i123_last.push_back(tb[2]); // OUTPUT
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

                b_last=ind_b; // OUTPUT

                /// Register the creation of triangles incident to point indexed by ind_b

                if(ind_b==p1_b)
                {
                    Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,
                            (unsigned int) p1_b,
                            (unsigned int) p0_b));
                    Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,
                            (unsigned int) p0_b,
                            (unsigned int) p2_b));
                    triangles_to_create.push_back(t_pb1);
                    triangles_to_create.push_back(t_pb2);
                }
                else
                {
                    if(ind_b==p2_b)
                    {
                        Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,
                                (unsigned int) p1_b,
                                (unsigned int) p0_b));
                        Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,
                                (unsigned int) p0_b,
                                (unsigned int) p2_b));
                        triangles_to_create.push_back(t_pb1);
                        triangles_to_create.push_back(t_pb2);
                    }
                    else
                    {
                        Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,
                                (unsigned int) p1_b,
                                (unsigned int) ind_b));
                        Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,
                                (unsigned int) ind_b,
                                (unsigned int)p2_b));
                        triangles_to_create.push_back(t_pb1);
                        triangles_to_create.push_back(t_pb2);
                    }
                }

                trianglesIndexList.push_back(acc_nb_triangles);
                trianglesIndexList.push_back(acc_nb_triangles+1);
                acc_nb_triangles+=2;
            }
        }

        // POINT SEPARATING

        if(!is_first_cut)
        {
            sofa::helper::vector< unsigned int > bb_first_ancestors;
            bb_first_ancestors.push_back(x_i1);
            bb_first_ancestors.push_back(x_i2);
            bb_first_ancestors.push_back(x_i3);
            sofa::helper::vector< double > bb_baryCoefs = m_geometryAlgorithms->compute3PointsBarycoefs((const Vec<3,double> &) a_new, x_i1, x_i2, x_i3);

            // Add point B1
            p_ancestors.push_back(bb_first_ancestors);
            p_baryCoefs.push_back(bb_baryCoefs);
            acc_nb_points=acc_nb_points+1;
            unsigned int B1 = acc_nb_points;

            // Add point B2
            p_ancestors.push_back(bb_first_ancestors);
            p_baryCoefs.push_back(bb_baryCoefs);
            acc_nb_points=acc_nb_points+1;
            unsigned int B2 = acc_nb_points;

            Triangle T1 = Triangle(helper::make_array<unsigned int>(B1, x_p1, x_i1));
            Triangle T1_to = Triangle(helper::make_array<unsigned int>(B1, x_i1_to, x_p1_to));

            Triangle T2 = Triangle(helper::make_array<unsigned int>(B2, x_i2, x_p2));
            Triangle T2_to = Triangle(helper::make_array<unsigned int>(B2, x_p2_to, x_i2_to));

            Triangle Ti1 = Triangle(helper::make_array<unsigned int>(B1, x_i1, x_i1_to));
            Triangle Ti2 = Triangle(helper::make_array<unsigned int>(B2, x_i2_to, x_i2));

            Triangle Tp1 = Triangle(helper::make_array<unsigned int>(B1, x_p1, x_p1_to));
            Triangle Tp2 = Triangle(helper::make_array<unsigned int>(B2, x_p2_to, x_p2));

            Triangle T1_13 = Triangle(helper::make_array<unsigned int>(B1, x_i1, x_i3));
            Triangle T1_23 = Triangle(helper::make_array<unsigned int>(B1, x_i3, x_i2));
            Triangle T2_13 = Triangle(helper::make_array<unsigned int>(B2, x_i1, x_i3));
            Triangle T2_23 = Triangle(helper::make_array<unsigned int>(B2, x_i3, x_i2));

            if(x_i1 == x_i1_to)
            {
                triangles_to_create.push_back(T1);
                triangles_to_create.push_back(T1_to);
                triangles_to_create.push_back(T2);
                triangles_to_create.push_back(T2_to);
                triangles_to_create.push_back(Ti2);

            }
            else
            {
                if(x_i2 == x_i2_to)
                {
                    triangles_to_create.push_back(T1);
                    triangles_to_create.push_back(T1_to);
                    triangles_to_create.push_back(T2);
                    triangles_to_create.push_back(T2_to);
                    triangles_to_create.push_back(Ti1);
                }
                else // (x_i1 == x_i2_to) or (x_i2 == x_i1_to)
                {
                    if(x_i1 == x_i2_to)
                    {
                        triangles_to_create.push_back(Tp1);
                        triangles_to_create.push_back(T2_13);
                        triangles_to_create.push_back(T2_23);
                        triangles_to_create.push_back(T2);
                        triangles_to_create.push_back(T2_to);
                    }
                    else // x_i2 == x_i1_to
                    {
                        triangles_to_create.push_back(Tp2);
                        triangles_to_create.push_back(T1_13);
                        triangles_to_create.push_back(T1_23);
                        triangles_to_create.push_back(T1);
                        triangles_to_create.push_back(T1_to);
                    }
                }
            }

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles+1);
            trianglesIndexList.push_back(acc_nb_triangles+2);
            trianglesIndexList.push_back(acc_nb_triangles+3);
            trianglesIndexList.push_back(acc_nb_triangles+4);
            acc_nb_triangles = acc_nb_triangles + 5;
        }


        // Create all the points registered to be created
        m_modifier->addPointsProcess((const unsigned int) acc_nb_points - nb_points);

        // Warn for the creation of all the points registered to be created
        m_modifier->addPointsWarning((const unsigned int) acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

        // Create all the triangles registered to be created
        m_modifier->addTrianglesProcess((const sofa::helper::vector< Triangle > &) triangles_to_create) ; // WARNING called after the creation process by the method "addTrianglesProcess"

        //cout<<"INFO, number to create = "<< triangles_to_create.size() <<endl;

        // Warn for the creation of all the triangles registered to be created
        m_modifier->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList);

        // Propagate the topological changes *** not necessary
        //m_container->propagateTopologicalChanges();

        // Remove all the triangles registered to be removed
        removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

        //cout<<"INFO, number to remove = "<< triangles_to_remove.size() <<endl;

        // Propagate the topological changes *** not necessary
        //m_container->propagateTopologicalChanges();

    }

    return is_intersected && (elem_size>0);
}

// Removes triangles along the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::RemoveAlongTrianglesList(const Vec<3,double>& a,
        const Vec<3,double>& b,
        const unsigned int ind_ta,
        const unsigned int ind_tb)
{
    sofa::helper::vector< unsigned int > triangles_list_init;
    sofa::helper::vector< unsigned int > &triangles_list = triangles_list_init;

    sofa::helper::vector< sofa::helper::vector< unsigned int> > indices_list_init;
    sofa::helper::vector< sofa::helper::vector< unsigned int> > &indices_list = indices_list_init;

    sofa::helper::vector< double > coords_list_init;
    sofa::helper::vector< double >& coords_list=coords_list_init;

    bool is_intersected=false;

    unsigned int ind_tb_final_init;
    unsigned int& ind_tb_final=ind_tb_final_init;

    bool is_on_boundary_init=false;
    bool &is_on_boundary=is_on_boundary_init;

    ind_tb_final=ind_tb;
    is_intersected = m_geometryAlgorithms->computeIntersectedPointsList(a, b, ind_ta, ind_tb_final, triangles_list, indices_list, coords_list, is_on_boundary);

    if(is_intersected)
    {
        sofa::helper::vector< unsigned int > triangles;

        for (unsigned int i=0; i<triangles_list.size(); ++i)
        {
            triangles.push_back(triangles_list[i]);
        }
        removeTriangles(triangles, true, true);
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
    sofa::helper::vector< unsigned int > triangles_list_init;
    sofa::helper::vector< unsigned int > &triangles_list = triangles_list_init;

    sofa::helper::vector< sofa::helper::vector<unsigned int> > indices_list_init;
    sofa::helper::vector< sofa::helper::vector<unsigned int> > &indices_list = indices_list_init;

    sofa::helper::vector< double > coords_list_init;
    sofa::helper::vector< double >& coords_list=coords_list_init;

    unsigned int ind_tb_final_init;
    unsigned int &ind_tb_final=ind_tb_final_init;

    bool is_on_boundary_init=false;
    bool &is_on_boundary=is_on_boundary_init;

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
            bool is_intersected = m_geometryAlgorithms->computeIntersectedPointsList(pa, pb, ind_tpa, ind_tb_final, triangles_list, indices_list, coords_list, is_on_boundary);
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

    bool is_snap_a0_init=false; bool is_snap_a1_init=false; bool is_snap_a2_init=false;
    bool& is_snap_a0=is_snap_a0_init;
    bool& is_snap_a1=is_snap_a1_init;
    bool& is_snap_a2=is_snap_a2_init;

    bool is_snap_b0_init=false; bool is_snap_b1_init=false; bool is_snap_b2_init=false;
    bool& is_snap_b0=is_snap_b0_init;
    bool& is_snap_b1=is_snap_b1_init;
    bool& is_snap_b2=is_snap_b2_init;

    double epsilon = 0.2; // INFO : epsilon is a threshold in [0,1] to control the snapping of the extremities to the closest vertex

    sofa::helper::vector< double > a_baryCoefs = m_geometryAlgorithms->computeTriangleBarycoefs((const Vec<3,double> &) a, ind_ta);
    snapping_test_triangle(epsilon, a_baryCoefs[0], a_baryCoefs[1], a_baryCoefs[2],
            is_snap_a0, is_snap_a1, is_snap_a2);

    double is_snapping_a = is_snap_a0 || is_snap_a1 || is_snap_a2;

    sofa::helper::vector< double > b_baryCoefs = m_geometryAlgorithms->computeTriangleBarycoefs((const Vec<3,double> &) b, ind_tb);
    snapping_test_triangle(epsilon, b_baryCoefs[0], b_baryCoefs[1], b_baryCoefs[2],
            is_snap_b0, is_snap_b1, is_snap_b2);

    double is_snapping_b = is_snap_b0 || is_snap_b1 || is_snap_b2;

    /*
    if(is_snapping_a){
    std::cout << "INFO_print : is_snapping_a" <<  std::endl;
    }
    if(is_snapping_b){
    std::cout << "INFO_print : is_snapping_b" <<  std::endl;
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

        unsigned int p1_a=indices_list[0][0];
        unsigned int p2_a=indices_list[0][1];
        unsigned int p1_b=indices_list[indices_list.size()-1][0];
        unsigned int p2_b=indices_list[indices_list.size()-1][1];

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

        for (unsigned int i=0; i<indices_list.size(); ++i)
        {
            /// Register the creation of the two points (say current "duplicated points") localized on the current interected edge
            unsigned int p1 = indices_list[i][0];
            unsigned int p2 = indices_list[i][1];

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

                if(is_snapping_p && i<indices_list.size()-1) // not to treat particular case of last extremitiy
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

                    //std::cout << "INFO_print : is_snapping_p, i = " << i << " on vertex " << ind_p <<  std::endl;

                    sofa::helper::vector< unsigned int > triangles_list_1_init;
                    sofa::helper::vector< unsigned int > &triangles_list_1 = triangles_list_1_init;

                    sofa::helper::vector< unsigned int > triangles_list_2_init;
                    sofa::helper::vector< unsigned int > &triangles_list_2 = triangles_list_2_init;

                    //std::cout << "INFO_print : DO Prepare_VertexDuplication " <<  std::endl;
                    m_geometryAlgorithms->Prepare_VertexDuplication(ind_p, triangles_list[i], triangles_list[i+1], indices_list[i-1], coords_list[i-1], indices_list[i+1], coords_list[i+1], triangles_list_1, triangles_list_2);
                    //std::cout << "INFO_print : DONE Prepare_VertexDuplication " <<  std::endl;

                    //std::cout << "INFO_print : triangles_list_1.size() = " << triangles_list_1.size() <<  std::endl;
                    //for (unsigned int k=0;k<triangles_list_1.size();k++){
                    //		std::cout << "INFO_print : triangles_list_1 number " << k << " = " << triangles_list_1[k] <<  std::endl;
                    //}

                    //std::cout << "INFO_print : triangles_list_2.size() = " << triangles_list_2.size() <<  std::endl;
                    //for (unsigned int k=0;k<triangles_list_2.size();k++){
                    //		std::cout << "INFO_print : triangles_list_2 number " << k << " = " << triangles_list_2[k] <<  std::endl;
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
        //m_container->propagateTopologicalChanges();

        // Remove all the triangles registered to be removed
        removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

        // Propagate the topological changes *** not necessary
        //m_container->propagateTopologicalChanges();
    }
}

// Duplicate the given edge. Only works of at least one of its points is adjacent to a border.
template<class DataTypes>
int TriangleSetTopologyAlgorithms<DataTypes>::InciseAlongEdge(unsigned int ind_edge)
{
    const Edge & edge0=m_container->getEdge(ind_edge);
    unsigned ind_pa = edge0[0];
    unsigned ind_pb = edge0[1];

    const helper::vector<unsigned>& triangles0 = m_container->getTriangleEdgeShell(ind_edge);
    if (triangles0.size() != 2)
    {
        std::cerr << "InciseAlongEdge: ERROR edge "<<ind_edge<<" is not attached to 2 triangles." << std::endl;
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
            std::cerr << "InciseAlongEdge: ERROR in triangle "<<ind_tria<<std::endl;
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
            std::cerr << "InciseAlongEdge: ERROR in triangle "<<ind_trib<<std::endl;
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
        std::cerr << "InciseAlongEdge: ERROR edge "<<ind_edge<<" is not on border." << std::endl;
        return -1;
    }

    // now we can split the edge

    /// force the creation of TriangleEdgeShellArray
    m_container->getTriangleEdgeShellArray();
    /// force the creation of TriangleVertexShellArray
    m_container->getTriangleVertexShellArray();

    //const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();
    unsigned int nb_points =  m_container->getTriangleVertexShellArray().size() - 1; //vect_c.size();
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

    sofa::helper::vector<double> defaultCoefs; defaultCoefs.push_back(1.0);

    unsigned new_pa, new_pb;

    if (pa_is_on_border)
    {
        sofa::helper::vector<unsigned int> ancestors;
        new_pa = acc_nb_points++;
        ancestors.push_back(ind_pa);
        p_ancestors.push_back(ancestors);
        p_baryCoefs.push_back(defaultCoefs);
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
    //m_container->propagateTopologicalChanges();

    // Remove all the triangles registered to be removed
    removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

    // Propagate the topological changes *** not necessary
    //m_container->propagateTopologicalChanges();

    return (pb_is_on_border?1:0)+(pa_is_on_border?1:0); // todo: get new edge indice
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TriangleSetTOPOLOGY_INL
