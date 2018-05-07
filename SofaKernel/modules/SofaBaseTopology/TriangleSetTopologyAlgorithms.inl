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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_INL

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>

#include <SofaBaseMechanics/MechanicalObject.h>

#include <algorithm>
#include <functional>

namespace sofa
{
namespace component
{
namespace topology
{

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
            new_triangles_id.push_back((unsigned int)(m_container->getNbTriangles()-(m_listTriAdd.getValue ()).size()+i));

        if (nbrBefore != m_container->getNbTriangles()) // Triangles have been added
        {
            m_modifier->addTrianglesWarning((unsigned int)m_listTriAdd.getValue().size(), m_listTriAdd.getValue(), new_triangles_id);
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

    sofa::defaulttype::Vec<3,double> point_created = m_geometryAlgorithms->computeBaryEdgePoint(ind1, ind2, 0.5);

    sofa::helper::vector< double > x_created;
    x_created.push_back((double) point_created[0]);
    x_created.push_back((double) point_created[1]);
    x_created.push_back((double) point_created[2]);

    core::behavior::MechanicalState<DataTypes>* state = m_geometryAlgorithms->getDOF();

    sofa::helper::WriteAccessor< Data<VecCoord> > x_wA = *state->write(core::VecCoordId::position());
    sofa::helper::WriteAccessor< Data<VecDeriv> > v_wA = *state->write(core::VecDerivId::velocity());

    DataTypes::set(x_wA[ind1], x_created[0], x_created[1], x_created[2]);
    DataTypes::set(v_wA[ind1], (Real) 0.0, (Real) 0.0, (Real) 0.0);

    DataTypes::set(x_wA[ind2], x_created[0], x_created[1], x_created[2]);
    DataTypes::set(v_wA[ind2], (Real) 0.0, (Real) 0.0, (Real) 0.0);

    return true;
}

// Removes triangles along the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::RemoveAlongTrianglesList(const sofa::defaulttype::Vec<3,double>& a,
        const sofa::defaulttype::Vec<3,double>& b,
        const unsigned int ind_ta,
        const unsigned int ind_tb)
{
    sofa::helper::vector< unsigned int > triangles_list;
    sofa::helper::vector< unsigned int > edges_list;
    sofa::helper::vector< double > coords_list;

    bool is_intersected=false;

    unsigned int ind_tb_final;

    bool is_on_boundary;

    ind_tb_final=ind_tb;
    unsigned int ind_ta_final=ind_ta;
    is_intersected = m_geometryAlgorithms->computeIntersectedPointsList(core::topology::BaseMeshTopology::InvalidID,a, b, ind_ta_final, ind_tb_final, triangles_list, edges_list, coords_list, is_on_boundary);

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
void TriangleSetTopologyAlgorithms< DataTypes >::InciseAlongLinesList(
        const sofa::helper::vector< sofa::defaulttype::Vec<3,double> >& input_points,
        const sofa::helper::vector< unsigned int > &input_triangles)
{
    // HYP : input_points.size() == input_triangles.size()

    size_t points_size = input_points.size();

    // Initialization for INTERSECTION method
    sofa::helper::vector< unsigned int > triangles_list;
    sofa::helper::vector< unsigned int > edges_list;
    sofa::helper::vector< double > coords_list;

    unsigned int ind_tb_final;

    bool is_on_boundary;

    const sofa::defaulttype::Vec<3,double> a = input_points[0];
    unsigned int ind_ta = input_triangles[0];

    unsigned int j = 0;
    bool is_validated=true;
    for(j = 0; is_validated && j < points_size - 1; ++j)
    {
        const sofa::defaulttype::Vec<3,double> pa = input_points[j];
        const sofa::defaulttype::Vec<3,double> pb = input_points[j+1];
        unsigned int ind_tpa = input_triangles[j];
        unsigned int ind_tpb = input_triangles[j+1];

        bool is_distinct = (pa!=pb && ind_tpa!=ind_tpb);

        if(is_distinct)
        {
            // Call the method "computeIntersectedPointsList" to get the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
            ind_tb_final=ind_tpb;
            bool is_intersected = m_geometryAlgorithms->computeIntersectedPointsList(core::topology::BaseMeshTopology::InvalidID,pa, pb, ind_tpa, ind_tb_final, triangles_list, edges_list, coords_list, is_on_boundary);
            is_validated=is_intersected;
        }
        else
        {
            is_validated=false;
        }
    }

    const sofa::defaulttype::Vec<3,double> b = input_points[j];
    unsigned int ind_tb = input_triangles[j];

    const Triangle &ta=m_container->getTriangle(ind_ta);
    const Triangle &tb=m_container->getTriangle(ind_tb);

    //const typename DataTypes::VecCoord& vect_c =topology->getDOF()->read(core::ConstVecCoordId::position())->getValue();
    const unsigned int nb_points =  (unsigned int)m_container->getTrianglesAroundVertexArray().size() - 1; //vect_c.size() -1;

    const sofa::helper::vector<Triangle> &vect_t=m_container->getTriangleArray();
    const unsigned int nb_triangles = (unsigned int)vect_t.size() -1;

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

    sofa::helper::vector< double > a_baryCoefs =
            m_geometryAlgorithms->computeTriangleBarycoefs(ind_ta, (const sofa::defaulttype::Vec<3,double> &) a);
    snapping_test_triangle(epsilon, a_baryCoefs[0], a_baryCoefs[1], a_baryCoefs[2],
            is_snap_a0, is_snap_a1, is_snap_a2);

    double is_snapping_a = is_snap_a0 || is_snap_a1 || is_snap_a2;

    sofa::helper::vector< double > b_baryCoefs =
            m_geometryAlgorithms->computeTriangleBarycoefs(ind_tb, (const sofa::defaulttype::Vec<3,double> &) b);
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
        /// force the creation of TrianglesAroundEdgeArray
        m_container->getTrianglesAroundEdgeArray();
        /// force the creation of TrianglesAroundVertexArray
        m_container->getTrianglesAroundVertexArray();

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
            Triangle t_a01 = Triangle((unsigned int)ind_a,
                    (unsigned int)ta[0],
                    (unsigned int) ta[1]);
            Triangle t_a12 = Triangle((unsigned int)ind_a,
                    (unsigned int)ta[1],
                    (unsigned int) ta[2]);
            Triangle t_a20 = Triangle((unsigned int)ind_a,
                    (unsigned int)ta[2],
                    (unsigned int) ta[0]);
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

            Triangle t_pa1 = Triangle((unsigned int) acc_nb_points + 1,
                    (unsigned int) ind_a,
                    (unsigned int) p1_a);
            Triangle t_pa2 = Triangle((unsigned int) acc_nb_points + 2,
                    (unsigned int) p2_a,
                    (unsigned int)ind_a);
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
                Triangle t_pa1 = Triangle((unsigned int) acc_nb_points + 2,
                        (unsigned int) p0_a,
                        (unsigned int) p1_a);
                Triangle t_pa2 = Triangle((unsigned int) acc_nb_points + 2,
                        (unsigned int) p2_a,
                        (unsigned int) p0_a);
                triangles_to_create.push_back(t_pa1);
                triangles_to_create.push_back(t_pa2);
            }
            else
            {
                if(ind_a==p2_a)
                {
                    Triangle t_pa1 = Triangle((unsigned int) acc_nb_points + 1,
                            (unsigned int) p0_a,
                            (unsigned int) p1_a);
                    Triangle t_pa2 = Triangle((unsigned int) acc_nb_points + 1,
                            (unsigned int) p2_a,
                            (unsigned int) p0_a);
                    triangles_to_create.push_back(t_pa1);
                    triangles_to_create.push_back(t_pa2);
                }
                else
                {
                    Triangle t_pa1 = Triangle((unsigned int) acc_nb_points + 1,
                            (unsigned int) ind_a,
                            (unsigned int) p1_a);
                    Triangle t_pa2 = Triangle((unsigned int) acc_nb_points + 2,
                            (unsigned int) p2_a,
                            (unsigned int)ind_a);
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

                Triangle t_p1 = Triangle((unsigned int) p1_created,(unsigned int) p1_prev,(unsigned int) p1_to_create);
                Triangle t_p2 = Triangle((unsigned int) p2_created,(unsigned int) p2_to_create,(unsigned int) p2_prev);

                Triangle t_p3;

                if(p0_t!=p1_prev && p0_t!=p2_prev)
                {
                    if(p0_t==p1)
                    {
                        t_p3=Triangle((unsigned int) p0_t,(unsigned int) p1_to_create,(unsigned int) p1_prev);

                    }
                    else // p0_t==p2
                    {
                        t_p3=Triangle((unsigned int) p0_t,(unsigned int) p2_prev,(unsigned int) p2_to_create);
                    }
                }
                else
                {
                    if(p1_t!=p1_prev && p1_t!=p2_prev)
                    {
                        if(p1_t==p1)
                        {
                            t_p3=Triangle((unsigned int) p1_t,(unsigned int) p1_to_create,(unsigned int) p1_prev);
                        }
                        else // p1_t==p2
                        {
                            t_p3=Triangle((unsigned int) p1_t,(unsigned int) p2_prev,(unsigned int) p2_to_create);
                        }
                    }
                    else // (p2_t!=p1_prev && p2_t!=p2_prev)
                    {
                        if(p2_t==p1)
                        {
                            t_p3=Triangle((unsigned int) p2_t,(unsigned int) p1_to_create,(unsigned int) p1_prev);
                        }
                        else // p2_t==p2
                        {
                            t_p3=Triangle((unsigned int) p2_t,(unsigned int) p2_prev,(unsigned int) p2_to_create);
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
            Triangle t_b01 = Triangle((unsigned int)ind_b,
                    (unsigned int)tb[0],
                    (unsigned int) tb[1]);
            Triangle t_b12 = Triangle((unsigned int)ind_b,
                    (unsigned int)tb[1],
                    (unsigned int) tb[2]);
            Triangle t_b20 = Triangle((unsigned int)ind_b,
                    (unsigned int)tb[2],
                    (unsigned int) tb[0]);
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

            Triangle t_pb1 = Triangle((unsigned int) acc_nb_points - 2,
                    (unsigned int) p1_b,
                    (unsigned int)ind_b);
            Triangle t_pb2 = Triangle((unsigned int) acc_nb_points - 1,
                    (unsigned int)ind_b,
                    (unsigned int) p2_b);
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
                Triangle t_pb1 = Triangle((unsigned int) acc_nb_points,(unsigned int) p1_b,(unsigned int) p0_b);
                Triangle t_pb2 = Triangle((unsigned int) acc_nb_points,(unsigned int) p0_b, (unsigned int) p2_b);
                triangles_to_create.push_back(t_pb1);
                triangles_to_create.push_back(t_pb2);

            }
            else
            {
                if(ind_b==p2_b)
                {
                    Triangle t_pb1 = Triangle((unsigned int) acc_nb_points - 1,(unsigned int) p1_b,(unsigned int) p0_b);
                    Triangle t_pb2 = Triangle((unsigned int) acc_nb_points - 1,(unsigned int) p0_b, (unsigned int) p2_b);
                    triangles_to_create.push_back(t_pb1);
                    triangles_to_create.push_back(t_pb2);
                }
                else
                {
                    Triangle t_pb1 = Triangle((unsigned int) acc_nb_points - 1,(unsigned int) p1_b,(unsigned int) ind_b);
                    Triangle t_pb2 = Triangle((unsigned int) acc_nb_points,(unsigned int) ind_b, (unsigned int)p2_b);
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
        m_modifier->addTrianglesWarning((unsigned int)triangles_to_create.size(), triangles_to_create, trianglesIndexList);

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
        sofa::helper::vector< sofa::core::topology::TopologyObjectType>& topoPath_list,
        sofa::helper::vector<unsigned int>& indices_list,
        sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& coords_list,
        sofa::helper::vector<EdgeID>& new_edges, double epsilonSnapPath, double epsilonSnapBorder)
{
    //////// STEP 1.a : MODIFY PATH IF SNAP = TRUE (don't change border case here if they are near an edge)
    if (indices_list.empty()) return 0;

    sofa::helper::vector< sofa::helper::vector<double> > points2Snap;

    //	double epsilon = 0.25; // to change to an input for snaping

    if (epsilonSnapPath != 0.0)
        SnapAlongPath (topoPath_list, indices_list, coords_list, points2Snap, epsilonSnapPath);

    //STEP 1.b : Modify border case path if snap = true
    if (epsilonSnapBorder != 0.0)
        SnapBorderPath (pa, a, pb, b, topoPath_list, indices_list, coords_list, points2Snap, epsilonSnapBorder);

    // Output declarations:
    const size_t nb_points = indices_list.size();
    sofa::helper::vector< sofa::helper::vector< PointID > > p_ancestors; p_ancestors.reserve(nb_points);// WARNING
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs; p_baryCoefs.reserve(nb_points);
    PointID next_point = m_container->getNbPoints();
    TriangleID next_triangle = m_container->getNbTriangles();
    sofa::helper::vector< PointID > new_edge_points; // new points created on each edge
    sofa::helper::vector< Triangle > new_triangles;
    sofa::helper::vector< TriangleID > new_triangles_id;
    sofa::helper::vector< TriangleID > removed_triangles;
    sofa::helper::vector< sofa::helper::vector< TriangleID > >  triangles_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > >  triangles_barycoefs;


    helper::vector< core::topology::PointAncestorElem > srcElems;

    //////// STEP 1 : Create points

    for (unsigned int i = 0; i < nb_points; i++)
    {

        p_ancestors.resize(p_ancestors.size()+1);
        sofa::helper::vector< PointID >& ancestors = p_ancestors.back();
        p_baryCoefs.resize(p_baryCoefs.size()+1);
        sofa::helper::vector< double >& baryCoefs = p_baryCoefs.back();


        switch ( topoPath_list[i] )
        {

        case core::topology::POINT:
        {
            // qlq chose a faire?
            new_edge_points.push_back(indices_list[i]);

            p_ancestors.resize(p_ancestors.size()-1);
            p_baryCoefs.resize(p_baryCoefs.size()-1);

            // For snaping:
            if ( (epsilonSnapPath != 0.0) || (!points2Snap.empty()))
                for (unsigned int j = 0; j<points2Snap.size(); j++)
                    if (points2Snap[j][0] == indices_list[i])
                    {
                        if (i == 0 || i == nb_points-1) //should not append, 0 and nb_points-1 correspond to bordersnap
                        {
                            unsigned int the_point = indices_list[i];
                            const sofa::helper::vector<EdgeID>& shell = m_container->getEdgesAroundVertex (the_point);
                            unsigned int cptSnap = 0;

                            for (unsigned int k = 0; k<shell.size(); k++)
                            {
                                const Edge& the_edge = m_container->getEdge (shell[k]);
                                if (the_edge[0] == the_point)
                                    points2Snap[j].push_back (the_edge[1]);
                                else
                                    points2Snap[j].push_back (the_edge[0]);

                                cptSnap++;
                                if (cptSnap == 3)
                                    break;
                            }

                            if (cptSnap != 3)
                                std::cout << "Error: In snaping border, missing elements to compute barycoefs!" << std::endl;

                            break;
                        }

                        points2Snap[j].push_back (next_point-1);
                        points2Snap[j].push_back (next_point);

                        if (topoPath_list[i-1] == core::topology::POINT) //second dof has to be moved, first acestor must be pa
                            points2Snap[j][4] = indices_list[i-1];

                        if (topoPath_list[i+1] == core::topology::POINT) //second dof has to be moved, first acestor must be pa
                            points2Snap[j][5] = indices_list[i+1];

                        break;
                    }

            break;
        }

        case core::topology::EDGE:
        {
            Edge theEdge = m_container->getEdge(indices_list[i]);
            ancestors.push_back(theEdge[0]);
            ancestors.push_back(theEdge[1]);

            baryCoefs.push_back(1.0 - coords_list[i][0]);
            baryCoefs.push_back(coords_list[i][0]);

            srcElems.push_back(core::topology::PointAncestorElem(core::topology::EDGE, indices_list[i],
                core::topology::PointAncestorElem::LocalCoords(coords_list[i][0], 0, 0)));

            new_edge_points.push_back(next_point);
            ++next_point;
            break;
        }
        case core::topology::TRIANGLE:
        {

            Triangle theTriangle = m_container->getTriangle(indices_list[i]);

            ancestors.push_back(theTriangle[0]);
            ancestors.push_back(theTriangle[1]);
            ancestors.push_back(theTriangle[2]);

            baryCoefs.push_back(coords_list[i][0]);
            baryCoefs.push_back(coords_list[i][1]);
            baryCoefs.push_back(coords_list[i][2]);

            srcElems.push_back(core::topology::PointAncestorElem(core::topology::TRIANGLE, indices_list[i],
                core::topology::PointAncestorElem::LocalCoords(coords_list[i][1], coords_list[i][2], 0)));

            new_edge_points.push_back(next_point);// hum...? pour les edges to split
            ++next_point;
            break;
        }
        default:
            break;

        }
    }

    bool error = false;

    // STEP 2: Computing triangles along path

    for (unsigned int i = 0; i<indices_list.size()-1; ++i)
    {
        unsigned int firstObject = indices_list[i];

        switch ( topoPath_list[i] )
        {
        case core::topology::POINT:
        {
            PointID thePointFirst = firstObject;

            switch ( topoPath_list[i+1] )
            {
            case core::topology::POINT: // Triangle to create: 0 / Triangle to remove: 0
            {
                PointID thePointSecond = indices_list[i+1];
                sofa::helper::vector <unsigned int> edgevertexshell = m_container->getEdgesAroundVertex (thePointSecond);
                bool test = false;

                for (unsigned int j = 0; j <edgevertexshell.size(); j++)
                {
                    Edge e = m_container->getEdge (edgevertexshell[j]);

                    if ( ((e[0] == thePointSecond) && (e[1] == thePointFirst)) || ((e[1] == thePointSecond) && (e[0] == thePointFirst)))
                    {
                        test = true;
                        break;
                    }
                }
                if(!test)
                {
#ifndef NDEBUG
                    std::cout << " Error: SplitAlongPath: error in POINT::EDGE case, the edge between these points has not been found." << std::endl;
#endif
                    error = true;
                }

                break;
            }
            case core::topology::EDGE: // Triangle to create: 2 / Triangle to remove: 1
            {
                EdgeID edgeIDSecond = indices_list[i+1];
                TriangleID triId;
                Triangle tri;

                sofa::helper::vector <unsigned int> triangleedgeshell = m_container->getTrianglesAroundEdge (edgeIDSecond);

                for (unsigned int j = 0; j<triangleedgeshell.size(); j++)
                {
                    triId = triangleedgeshell[j];
                    tri = m_container->getTriangle (triangleedgeshell[j]);

                    if ( (tri[0] == thePointFirst) || (tri[1] == thePointFirst) || (tri[2] == thePointFirst) )
                    {
                        triangles_ancestors.resize (triangles_ancestors.size()+2);
                        triangles_barycoefs.resize (triangles_barycoefs.size()+2);

                        triangles_ancestors[triangles_ancestors.size()-2].push_back (triId);
                        triangles_barycoefs[triangles_barycoefs.size()-2].push_back (1.0);
                        triangles_ancestors[triangles_ancestors.size()-1].push_back (triId);
                        triangles_barycoefs[triangles_barycoefs.size()-1].push_back (1.0);
                        //found = true;

                        break;
                    }
                }

                int vertxInTriangle = m_container->getVertexIndexInTriangle (tri, thePointFirst);

                if (vertxInTriangle == -1)
                {
#ifndef NDEBUG
                    std::cout << " Error: SplitAlongPath: error in triangle in POINT::EDGE case" << std::endl;

                    std::cout << "*********************************" << std::endl;
                    std::cout << "topoPath_list: " << topoPath_list << std::endl;
                    std::cout << "indices_list: " << indices_list << std::endl;
                    std::cout << "new_edge_points: " << new_edge_points << std::endl;
                    std::cout << "nb new points: " << p_ancestors.size() << std::endl;
                    std::cout << "ancestors: " << p_ancestors << std::endl;
                    std::cout << "baryCoefs: " << p_baryCoefs << std::endl;
                    std::cout << "points2Snap: " << points2Snap << std::endl;
                    std::cout << "*********************************" << std::endl;
#endif

                    error = true;

                    break;
                }


                new_triangles.push_back (Triangle ( tri[vertxInTriangle], new_edge_points[i+1] , tri[(vertxInTriangle+2)%3]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( tri[vertxInTriangle], tri[(vertxInTriangle+1)%3],  new_edge_points[i+1]));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triId);

                break;
            }
            case core::topology::TRIANGLE: // Triangle to create: 3 / Triangle to remove: 1
            {
                TriangleID triangleIDSecond = indices_list[i+1];
                Triangle theTriangleSecond = m_container->getTriangle(triangleIDSecond);

                triangles_ancestors.resize (triangles_ancestors.size()+3);
                triangles_barycoefs.resize (triangles_barycoefs.size()+3);

                for (unsigned int j = 0; j<3; j++)
                {
                    triangles_ancestors[triangles_ancestors.size()-j-1].push_back (triangleIDSecond);
                    triangles_barycoefs[triangles_barycoefs.size()-j-1].push_back (1.0);
                }

                new_triangles.push_back (Triangle ( theTriangleSecond[0], theTriangleSecond[1], new_edge_points[i+1]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( theTriangleSecond[1], theTriangleSecond[2], new_edge_points[i+1]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( theTriangleSecond[2], theTriangleSecond[0], new_edge_points[i+1]));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triangleIDSecond);

                break;
            }
            default:
                break;

            }
            break;
        }

        case core::topology::EDGE:
        {
            PointID p1 = new_edge_points[i];
            EdgeID edgeIDFirst = firstObject;
            Edge theEdgeFirst = m_container->getEdge(firstObject);
            sofa::defaulttype::Vec<3,double> pos1 = m_geometryAlgorithms->computeBaryEdgePoint(theEdgeFirst, coords_list[i][0]);

            switch ( topoPath_list[i+1] )
            {

            case core::topology::POINT: // Triangle to create: 2 / Triangle to remove: 1
            {
                PointID thePointSecond = indices_list[i+1];

                TriangleID triId;
                Triangle tri;

                sofa::helper::vector <unsigned int> triangleedgeshell = m_container->getTrianglesAroundEdge (edgeIDFirst);

                for (unsigned int j = 0; j<triangleedgeshell.size(); j++)
                {
                    triId = triangleedgeshell[j];
                    tri = m_container->getTriangle (triangleedgeshell[j]);

                    if ( (tri[0] == thePointSecond) || (tri[1] == thePointSecond) || (tri[2] == thePointSecond) )
                    {
                        triangles_ancestors.resize (triangles_ancestors.size()+2);
                        triangles_barycoefs.resize (triangles_barycoefs.size()+2);

                        triangles_ancestors[triangles_ancestors.size()-2].push_back (triId);
                        triangles_barycoefs[triangles_barycoefs.size()-2].push_back (1.0);
                        triangles_ancestors[triangles_ancestors.size()-1].push_back (triId);
                        triangles_barycoefs[triangles_barycoefs.size()-1].push_back (1.0);

                        break;
                    }
                }

                int vertxInTriangle = m_container->getVertexIndexInTriangle (tri, thePointSecond);

                if (vertxInTriangle == -1)
                {
#ifndef NDEBUG
                    std::cout << " Error: SplitAlongPath: error in triangle in EDGE::POINT case" << std::endl;
#endif
                    error = true;
                    break;
                }

                new_triangles.push_back (Triangle ( thePointSecond, p1 , tri[(vertxInTriangle+2)%3]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( thePointSecond, tri[(vertxInTriangle+1)%3],  p1));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triId);

                break;
            }
            case core::topology::EDGE: // Triangle to create: 3 / Triangle to remove: 1
            {
                PointID p2 = new_edge_points[i+1];
                EdgeID edgeIDSecond = indices_list[i+1];
                Edge theEdgeSecond = m_container->getEdge(edgeIDSecond);
                sofa::defaulttype::Vec<3,double> pos2 = m_geometryAlgorithms->computeBaryEdgePoint(theEdgeSecond, coords_list[i+1][0]);

                TriangleID triId;
                Triangle tri;

                sofa::helper::vector <unsigned int> triangleedgeshell = m_container->getTrianglesAroundEdge (edgeIDFirst);

                for (unsigned int j = 0; j<triangleedgeshell.size(); j++)
                {
                    triId = triangleedgeshell[j];
                    tri = m_container->getTriangle (triangleedgeshell[j]);
                    const EdgesInTriangle triedge = m_container->getEdgesInTriangle (triangleedgeshell[j]);

                    if ( (triedge[0] == edgeIDSecond) || (triedge[1] == edgeIDSecond) || (triedge[2] == edgeIDSecond) )
                    {
                        triangles_ancestors.resize (triangles_ancestors.size()+3);
                        triangles_barycoefs.resize (triangles_barycoefs.size()+3);

                        for (unsigned int k = 0; k<3; k++)
                        {
                            triangles_ancestors[triangles_ancestors.size()-k-1].push_back (triId);
                            triangles_barycoefs[triangles_barycoefs.size()-k-1].push_back (1.0);
                        }
                        break;
                    }
                }


                // Find common corner and find incision direction in triangle
                unsigned int cornerInEdge1 = ((theEdgeFirst[0] == theEdgeSecond[0]) || (theEdgeFirst[0] == theEdgeSecond[1])) ? 0 : 1;
                int vertxInTriangle = m_container->getVertexIndexInTriangle (tri, theEdgeFirst[cornerInEdge1]);

                PointID vertexOrder[5]; //corner, p1, tri+1, tri+2, p2
                vertexOrder[0] = theEdgeFirst[cornerInEdge1]; vertexOrder[2] = tri[ (vertxInTriangle+1)%3 ]; vertexOrder[3] = tri[ (vertxInTriangle+2)%3 ];
                Coord posOrder[4];

                if ( tri[ (vertxInTriangle+1)%3 ] == theEdgeFirst[ (cornerInEdge1+1)%2 ] )
                {
                    vertexOrder[1] = p1; vertexOrder[4] = p2;
                    posOrder[0] = pos1; posOrder[3] = pos2;
                    posOrder[1] = m_geometryAlgorithms->getPointPosition( tri[ (vertxInTriangle+1)%3 ] );
                    posOrder[2] = m_geometryAlgorithms->getPointPosition( tri[ (vertxInTriangle+2)%3 ] );
                }
                else
                {
                    vertexOrder[1] = p2; vertexOrder[4] = p1;
                    posOrder[0] = pos2; posOrder[3] = pos1;
                    posOrder[1] = m_geometryAlgorithms->getPointPosition( tri[ (vertxInTriangle+2)%3 ] );
                    posOrder[2] = m_geometryAlgorithms->getPointPosition( tri[ (vertxInTriangle+1)%3 ] );
                }

                // Create the triangle around corner
                new_triangles.push_back(Triangle(vertexOrder[0], vertexOrder[1], vertexOrder[4]));
                new_triangles_id.push_back(next_triangle++);


                // Triangularize the remaining quad according to the delaunay criteria
                if (m_geometryAlgorithms->isQuadDeulaunayOriented(posOrder[0], posOrder[1], posOrder[2], posOrder[3]))
                {
                    new_triangles.push_back(Triangle(vertexOrder[1], vertexOrder[2], vertexOrder[3]));
                    new_triangles_id.push_back(next_triangle++);
                    new_triangles.push_back(Triangle(vertexOrder[4], vertexOrder[1], vertexOrder[3]));
                    new_triangles_id.push_back(next_triangle++);
                }
                else
                {
                    new_triangles.push_back(Triangle(vertexOrder[1], vertexOrder[2], vertexOrder[4]));
                    new_triangles_id.push_back(next_triangle++);
                    new_triangles.push_back(Triangle(vertexOrder[2], vertexOrder[3], vertexOrder[4]));
                    new_triangles_id.push_back(next_triangle++);
                }

                removed_triangles.push_back(triId);
                break;
            }
            case core::topology::TRIANGLE: // Triangle to create: 4 / Triangle to remove: 1
            {
                PointID p2 = new_edge_points[i+1];
                TriangleID triangleIDSecond = indices_list[i+1];
                Triangle theTriangleSecond = m_container->getTriangle(triangleIDSecond);

                const EdgesInTriangle triedge = m_container->getEdgesInTriangle (triangleIDSecond);
                int edgeInTriangle = m_container->getEdgeIndexInTriangle (triedge, edgeIDFirst);

                if (edgeInTriangle == -1)
                {
#ifndef NDEBUG
                    std::cout << " Error: SplitAlongPath: error in triangle in EDGE::TRIANGLE case" << std::endl;
#endif
                    error = true;
                    break;
                }

                triangles_ancestors.resize (triangles_ancestors.size()+4);
                triangles_barycoefs.resize (triangles_barycoefs.size()+4);

                for (unsigned int j = 0; j<4; j++)
                {
                    triangles_ancestors[triangles_ancestors.size()-j-1].push_back (triangleIDSecond);
                    triangles_barycoefs[triangles_barycoefs.size()-j-1].push_back (1.0);
                }


                // create two triangles linking p with the corner
                new_triangles.push_back (Triangle ( p2, theTriangleSecond[edgeInTriangle], theTriangleSecond[(edgeInTriangle+1)%3]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( p2, theTriangleSecond[(edgeInTriangle+2)%3], theTriangleSecond[edgeInTriangle]));
                new_triangles_id.push_back(next_triangle++);


                // create two triangles linking p with the splitted edge
                new_triangles.push_back (Triangle ( p2, theTriangleSecond[(edgeInTriangle+1)%3], p1));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( p2, p1, theTriangleSecond[(edgeInTriangle+2)%3]));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triangleIDSecond);
                break;

            }
            default:
                break;

            }
            break;
        }
        case core::topology::TRIANGLE:
        {
            Triangle theTriangleFirst = m_container->getTriangle(firstObject);
            TriangleID triangleIDFirst = indices_list[i];
            PointID p1 = new_edge_points[i];
            PointID p2 = new_edge_points[i+1];

            switch ( topoPath_list[i+1] )
            {
            case core::topology::POINT: // Triangle to create: 3 / Triangle to remove: 1
            {
                triangles_ancestors.resize (triangles_ancestors.size()+3);
                triangles_barycoefs.resize (triangles_barycoefs.size()+3);

                for (unsigned int j = 0; j<3; j++)
                {
                    triangles_ancestors[triangles_ancestors.size()-j-1].push_back (triangleIDFirst);
                    triangles_barycoefs[triangles_barycoefs.size()-j-1].push_back (1.0);
                }

                new_triangles.push_back (Triangle ( p1, theTriangleFirst[0], theTriangleFirst[1]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( p1, theTriangleFirst[1], theTriangleFirst[2]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( p1, theTriangleFirst[2], theTriangleFirst[0]));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triangleIDFirst);

                break;
            }
            case core::topology::EDGE: // Triangle to create: 4 / Triangle to remove: 1
            {
                EdgeID edgeIDSecond = indices_list[i+1];

                const EdgesInTriangle triedge = m_container->getEdgesInTriangle (triangleIDFirst);
                int edgeInTriangle = m_container->getEdgeIndexInTriangle (triedge, edgeIDSecond);

                if (edgeInTriangle == -1)
                {
#ifndef NDEBUG
                    std::cout << " Error: SplitAlongPath: error in triangle in TRIANGLE::EDGE case" << std::endl;
#endif
                    error = true;
                    break;
                }

                triangles_ancestors.resize (triangles_ancestors.size()+4);
                triangles_barycoefs.resize (triangles_barycoefs.size()+4);

                for (unsigned int j = 0; j<4; j++)
                {
                    triangles_ancestors[triangles_ancestors.size()-j-1].push_back (triangleIDFirst);
                    triangles_barycoefs[triangles_barycoefs.size()-j-1].push_back (1.0);
                }

                // create two triangles linking p with the corner
                new_triangles.push_back (Triangle ( p1, theTriangleFirst[edgeInTriangle], theTriangleFirst[(edgeInTriangle+1)%3]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( p1, theTriangleFirst[(edgeInTriangle+2)%3], theTriangleFirst[edgeInTriangle]));
                new_triangles_id.push_back(next_triangle++);


                // create two triangles linking p with the splitted edge
                new_triangles.push_back (Triangle ( p1, theTriangleFirst[(edgeInTriangle+1)%3], p2));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back (Triangle ( p1, p2, theTriangleFirst[(edgeInTriangle+2)%3]));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triangleIDFirst);
                break;
            }
            case core::topology::TRIANGLE: // Triangle to create: 5 / Triangle to remove: 1
            {
                TriangleID triangleIDSecond = indices_list[i+1];

                if (triangleIDSecond != triangleIDFirst)
                {
#ifndef NDEBUG
                    std::cout << " Error: SplitAlongPath: incision not in the mesh plan not supported yet, in TRIANGLE::TRIANGLE case" << std::endl;
#endif
                    error = true;
                    break;
                }

                PointID quad[2][4];
                PointID Tri1[3];
                double tmp1 = 0.0; unsigned int cornerP1[3]= {0,0,0};
                double tmp2 = 0.0; unsigned int cornerP2[3]= {0,0,0};

                for (unsigned int j = 0; j<3;  j++) // find first corners
                {
                    if (coords_list[i][j] > tmp1)
                    {
                        tmp1 = coords_list[i][j];
                        cornerP1[0] = j;
                    }

                    if (coords_list[i+1][j] > tmp2)
                    {
                        tmp2 = coords_list[i+1][j];
                        cornerP2[0] = j;
                    }
                }

                // sort other corners by decreasing barycoef
                if (coords_list[i][ (cornerP1[0]+1)%3 ] > coords_list[i][ (cornerP1[0]+2)%3 ])
                {
                    cornerP1[1] = (cornerP1[0]+1)%3;
                    cornerP1[2] = (cornerP1[0]+2)%3;
                }
                else
                {
                    cornerP1[1] = (cornerP1[0]+2)%3;
                    cornerP1[2] = (cornerP1[0]+1)%3;
                }

                if (coords_list[i+1][ (cornerP2[0]+1)%3 ] > coords_list[i+1][ (cornerP2[0]+2)%3 ])
                {
                    cornerP2[1] = (cornerP2[0]+1)%3;
                    cornerP2[2] = (cornerP2[0]+2)%3;
                }
                else
                {
                    cornerP2[1] = (cornerP2[0]+2)%3;
                    cornerP2[2] = (cornerP2[0]+1)%3;
                }


                if (cornerP1[0] != cornerP2[0])
                {
                    unsigned int cornerP1InTriangle = cornerP1[0];
                    unsigned int cornerP2InTriangle = cornerP2[0];

                    if ( (cornerP1InTriangle+1)%3 == cornerP2InTriangle ) // in the right direction
                    {
                        quad[0][0] = p1; quad[0][1] = theTriangleFirst[cornerP1InTriangle];
                        quad[0][3] = p2; quad[0][2] = theTriangleFirst[cornerP2InTriangle];

                        if (coords_list[i][(cornerP1InTriangle+2)%3] > coords_list[i+1][(cornerP1InTriangle+2)%3]) // second quad in other direction
                        {
                            quad[1][0] = p2; quad[1][1] = theTriangleFirst[(cornerP1InTriangle+1)%3];
                            quad[1][3] = p1; quad[1][2] = theTriangleFirst[(cornerP1InTriangle+2)%3];
                            Tri1[0] = p1; Tri1[1] =  theTriangleFirst[(cornerP1InTriangle+2)%3]; Tri1[2] =  theTriangleFirst[cornerP1InTriangle];
                        }
                        else
                        {
                            quad[1][0] = p2; quad[1][1] = theTriangleFirst[(cornerP1InTriangle+2)%3];
                            quad[1][3] = p1; quad[1][2] = theTriangleFirst[cornerP1InTriangle];
                            Tri1[0] =  p2; Tri1[1] = theTriangleFirst[(cornerP1InTriangle+1)%3]; Tri1[2] = theTriangleFirst[(cornerP1InTriangle+2)%3];
                        }
                    }
                    else     // switch order due to incision direction
                    {
                        quad[0][0] = p2; quad[0][1] = theTriangleFirst[cornerP2InTriangle];
                        quad[0][3] = p1; quad[0][2] = theTriangleFirst[cornerP1InTriangle];

                        if (coords_list[i][(cornerP2InTriangle+2)%3] > coords_list[i+1][(cornerP2InTriangle+2)%3]) // second quad in other direction
                        {
                            quad[1][0] = p1; quad[1][1] = theTriangleFirst[(cornerP1InTriangle+1)%3];
                            quad[1][3] = p2; quad[1][2] = theTriangleFirst[(cornerP1InTriangle+2)%3];
                            Tri1[0] = p1; Tri1[1] = theTriangleFirst[cornerP1InTriangle]; Tri1[2] = theTriangleFirst[(cornerP1InTriangle+1)%3];
                        }
                        else
                        {
                            quad[1][0] = p1; quad[1][1] = theTriangleFirst[cornerP1InTriangle];
                            quad[1][3] = p2; quad[1][2] = theTriangleFirst[(cornerP1InTriangle+1)%3];
                            Tri1[0] = p2; Tri1[1] = theTriangleFirst[(cornerP1InTriangle+1)%3]; Tri1[2] = theTriangleFirst[(cornerP1InTriangle+2)%3];
                        }
                    }
                }
                else
                {
                    unsigned int closest, second;
                    int cornerInTriangle;

                    if (tmp1 > tmp2)
                    {
                        closest = p1; second = p2;
                        cornerInTriangle = cornerP1[0];
                    }
                    else
                    {
                        closest = p2; second = p1;
                        cornerInTriangle = cornerP2[0];
                    }

                    quad[0][0] = closest; quad[0][1] = theTriangleFirst[cornerInTriangle];
                    quad[0][3] = second; quad[0][2] = theTriangleFirst[(cornerInTriangle+1)%3];

                    quad[1][0] = second; quad[1][1] = theTriangleFirst[(cornerInTriangle+2)%3];
                    quad[1][3] = closest; quad[1][2] = theTriangleFirst[cornerInTriangle];

                    Tri1[0] = second; Tri1[1] = theTriangleFirst[(cornerInTriangle+1)%3]; Tri1[2] = theTriangleFirst[(cornerInTriangle+2)%3];
                }

                new_triangles.push_back(Triangle(Tri1[0], Tri1[1], Tri1[2]));
                new_triangles_id.push_back(next_triangle++);

                // Triangularize the remaining quad according to the delaunay criteria
                const typename DataTypes::VecCoord& coords =(m_geometryAlgorithms->getDOF()->read(core::ConstVecCoordId::position())->getValue());
                for (unsigned int j = 0; j<2; j++)
                {
                    //Vec<3,double> pos[4];
                    Coord pos[4];
                    for (unsigned int k = 0; k<4; k++)
                    {
                        if (quad[j][k] == p1)
                            for (unsigned int u = 0; u<3; u++)
                                for (unsigned int v = 0; v<3; v++)
                                    pos[k][v] = pos[k][v] + coords[theTriangleFirst[u]][v]*(Real)coords_list[i][u];
                        else if (quad[j][k] == p2)
                            for (unsigned int u = 0; u<3; u++)
                                for (unsigned int v = 0; v<3; v++)
                                    pos[k][v] = pos[k][v] + coords[theTriangleFirst[u]][v]*(Real)coords_list[i+1][u];
                        else
                            pos[k]= coords[quad[j][k]];

                    }

                    if (m_geometryAlgorithms->isQuadDeulaunayOriented(pos[0], pos[1], pos[2], pos[3]))
                    {
                        new_triangles.push_back(Triangle(quad[j][1], quad[j][2], quad[j][0]));
                        new_triangles_id.push_back(next_triangle++);
                        new_triangles.push_back(Triangle(quad[j][3], quad[j][0], quad[j][2]));
                        new_triangles_id.push_back(next_triangle++);
                    }
                    else
                    {
                        new_triangles.push_back(Triangle(quad[j][2], quad[j][3], quad[j][1]));
                        new_triangles_id.push_back(next_triangle++);
                        new_triangles.push_back(Triangle(quad[j][0], quad[j][1], quad[j][3]));
                        new_triangles_id.push_back(next_triangle++);
                    }

                }

                triangles_ancestors.resize (triangles_ancestors.size()+5);
                triangles_barycoefs.resize (triangles_barycoefs.size()+5);

                for (unsigned int j = 0; j<5; j++)
                {
                    triangles_ancestors[triangles_ancestors.size()-j-1].push_back (triangleIDFirst);
                    triangles_barycoefs[triangles_barycoefs.size()-j-1].push_back (1.0);
                }

                removed_triangles.push_back(triangleIDFirst);

                break;
            }
            default:
                break;
            }
            break;
        }

        default:
            break;
        }

        if (error)
        {
#ifndef NDEBUG
            std::cout << "ERROR: in the incision path. " << std::endl;
#endif
            return -1;
        }
    }

    // FINAL STEP : Apply changes
    PointID newP0 = next_point - (PointID)srcElems.size();
    m_modifier->addPoints((unsigned int)srcElems.size(), srcElems);

    // m_modifier->propagateTopologicalChanges();

    // Create new edges with full ancestry information
    std::set<Edge> edges_processed;
    sofa::helper::vector<Edge> edges_added;
    sofa::helper::vector<core::topology::EdgeAncestorElem> edges_src;
    for (unsigned int ti = 0; ti < new_triangles.size(); ++ti)
    {
        Triangle t = new_triangles[ti];
        for (int tpi = 0; tpi < 3; ++tpi)
        {
            Edge e(t[tpi],t[(tpi+1)%3]);
            if (e[0] > e[1]) { unsigned int tmp = e[0]; e[0] = e[1]; e[1] = tmp; }
            if (e[0] < newP0 && e[1] < newP0 && m_container->getEdgeIndex(e[0], e[1]) != -1)
                continue; // existing edge
            if (!edges_processed.insert(e).second)
                continue; // this edge was already processed
            core::topology::EdgeAncestorElem src;
            for (unsigned int k = 0; k < 2; ++k)
            {
                if (e[k] < newP0)
                { // previous point
                    src.pointSrcElems[k].type = core::topology::POINT;
                    src.pointSrcElems[k].index = e[k];
                }
                else
                {
                    src.pointSrcElems[k] = srcElems[e[k]-newP0];
                }
            }
            // Source element could be an edge if both points are from it or from its endpoints
            if (src.pointSrcElems[0].type != core::topology::TRIANGLE
                && src.pointSrcElems[1].type != core::topology::TRIANGLE
                && (src.pointSrcElems[0].type == core::topology::EDGE
                    || src.pointSrcElems[1].type == core::topology::EDGE)
                && (src.pointSrcElems[0].type == core::topology::POINT
                    || src.pointSrcElems[1].type == core::topology::POINT
                    || src.pointSrcElems[0].index == src.pointSrcElems[1].index))
            {
                unsigned int src_eid = (src.pointSrcElems[0].type == core::topology::EDGE)
                    ? src.pointSrcElems[0].index : src.pointSrcElems[1].index;
                Edge src_e = m_container->getEdge(src_eid);
                if ((src.pointSrcElems[0].type != core::topology::POINT
                    || src.pointSrcElems[0].index == src_e[0]
                    || src.pointSrcElems[0].index == src_e[1])
                    && (src.pointSrcElems[1].type != core::topology::POINT
                    || src.pointSrcElems[1].index == src_e[0]
                    || src.pointSrcElems[1].index == src_e[1]))
                {
                    src.srcElems.push_back(core::topology::TopologyElemID(core::topology::EDGE,
                        src_eid));
                }
            }
            if (src.srcElems.empty()) // within the initial triangle by default
                src.srcElems.push_back(core::topology::TopologyElemID(core::topology::TRIANGLE,
                    triangles_ancestors[ti][0]));
            edges_added.push_back(e);
            edges_src.push_back(src);
        }
    }
    m_modifier->addEdges(edges_added, edges_src);

    unsigned int nbEdges = m_container->getNbEdges();

    //Add and remove triangles lists
    m_modifier->addRemoveTriangles((unsigned int)new_triangles.size(), new_triangles, new_triangles_id, triangles_ancestors, triangles_barycoefs, removed_triangles);

    unsigned int nbEdges2 = m_container->getNbEdges();

    if (nbEdges2 > nbEdges)
    {
        serr << "SplitAlongPath: auto created edges up to " << nbEdges << ", while ended up with " << nbEdges2 << sendl;
    }

    //WARNING can produce error TODO: check it
    if ( !points2Snap.empty())
    {
        sofa::helper::vector <unsigned int> id2Snap;
        sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestors2Snap; ancestors2Snap.resize(points2Snap.size());
        sofa::helper::vector< sofa::helper::vector< double > > coefs2Snap; coefs2Snap.resize(points2Snap.size());

        for (unsigned int i = 0; i<points2Snap.size(); i++)
        {

            sofa::defaulttype::Vec<3,double> SnapedCoord;
            unsigned int firstAncestor = (unsigned int)points2Snap[i][4];
            unsigned int secondAncestor = (unsigned int)points2Snap[i][5];

            for (unsigned int j = 0; j<3; j++)
                SnapedCoord[j] = points2Snap[i][j+1];

            id2Snap.push_back ((unsigned int)points2Snap[i][0]);

            ancestors2Snap[i].push_back (firstAncestor); //coefs2Snap[i].push_back (bary_coefs[0]);
            ancestors2Snap[i].push_back (secondAncestor); //coefs2Snap[i].push_back (bary_coefs[1]);


            if (points2Snap[i].size() == 7)
            {
                coefs2Snap[i] = m_geometryAlgorithms->compute3PointsBarycoefs (SnapedCoord , firstAncestor, secondAncestor, (unsigned int)points2Snap[i][6]);
                ancestors2Snap[i].push_back ((unsigned int)points2Snap[i][6]);
            }
            else
                coefs2Snap[i] = m_geometryAlgorithms->compute2PointsBarycoefs (SnapedCoord , firstAncestor, secondAncestor);
        }
        m_modifier->movePointsProcess ( id2Snap, ancestors2Snap, coefs2Snap);
    }

    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    for (unsigned int i = 0; i < new_edge_points.size()-1; ++i)
    {
        EdgeID e = m_container->getEdgeIndex(new_edge_points[i], new_edge_points[i+1]);

        if (e == (EdgeID)-1)
            e = m_container->getEdgeIndex(new_edge_points[i+1], new_edge_points[i]);

        if (e == (EdgeID)-1)
            this->serr << "ERROR: Edge " << new_edge_points[i] << " - " << new_edge_points[i+1] << " NOT FOUND." << this->sendl;
        else
            new_edges.push_back(e);
    }
    return (int)p_ancestors.size();
}



template<class DataTypes>
void TriangleSetTopologyAlgorithms<DataTypes>::SnapAlongPath (sofa::helper::vector< sofa::core::topology::TopologyObjectType>& topoPath_list,
        sofa::helper::vector<unsigned int>& indices_list, sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& coords_list,
        sofa::helper::vector< sofa::helper::vector<double> >& points2Snap,
        double epsilonSnapPath)
{
    std::map <PointID, sofa::helper::vector<unsigned int> > map_point2snap;
    std::map <PointID, sofa::helper::vector<unsigned int> >::iterator it;
    std::map <PointID, sofa::defaulttype::Vec<3,double> > map_point2bary;
    double epsilon = epsilonSnapPath;

    //// STEP 1 - First loop to find concerned points
    for (unsigned int i = 0; i < indices_list.size(); i++)
    {
        switch ( topoPath_list[i] )
        {
            // New case to handle other topological object can be added.
            // Default: if object is a POINT , nothing has to be done.

        case core::topology::EDGE:
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
                map_point2bary[Vertex2Snap] = sofa::defaulttype::Vec<3,double> ();
            }

            break;
        }
        case core::topology::TRIANGLE:
        {
            PointID Vertex2Snap;
            sofa::defaulttype::Vec<3, double>& barycoord = coords_list[i];
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
                map_point2bary[Vertex2Snap] = sofa::defaulttype::Vec<3,double> ();
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
        return;
    }

    const typename DataTypes::VecCoord& coords =(m_geometryAlgorithms->getDOF()->read(core::ConstVecCoordId::position())->getValue());


    //// STEP 3 - Second loop necessary to find object on the neighborhood of a snaped point
    for (unsigned int i = 0; i < indices_list.size(); i++)
    {
        switch ( topoPath_list[i] )
        {
        case core::topology::POINT:
        {
            if ( map_point2snap.find (indices_list[i]) != map_point2snap.end() )
            {
                map_point2snap[ indices_list[i] ].push_back(i);

                for (unsigned int j = 0; j<3; j++)
                    map_point2bary[ indices_list[i] ][j] += coords[ indices_list[i] ][j];
            }
            break;
        }
        case core::topology::EDGE:
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
                    const sofa::defaulttype::Vec<3,double>& coord_bary = m_geometryAlgorithms->computeBaryEdgePoint (theEdge, coords_list[i][0]);

                    // Step 2/3: Sum the different incision point position.
                    for (unsigned int j = 0; j<3; j++)
                        map_point2bary[ thePoint ][j] += coord_bary[j];
                }

                if (PointFind)
                    break;
            }
            break;
        }
        case core::topology::TRIANGLE:
        {
            Triangle theTriangle = m_container->getTriangleArray()[indices_list[i]];
            bool PointFind = false;

            for (unsigned int indTri = 0; indTri < 3; indTri++)
            {
                PointID thePoint = theTriangle[ indTri ];

                if ( (map_point2snap.find (thePoint) != map_point2snap.end()) && (coords_list[i][indTri] > (1-epsilon)))
                {
                    PointFind = true;
                    map_point2snap[ thePoint ].push_back(i);

                    const sofa::defaulttype::Vec<3,double>& coord_bary = m_geometryAlgorithms->computeBaryTrianglePoint (theTriangle, coords_list[i]);

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

    //Pre-treatment to avoid snaping near a border:
    sofa::helper::vector<unsigned int> field2remove;
    for (it = map_point2snap.begin(); it != map_point2snap.end(); ++it)
    {
        const sofa::helper::vector <EdgeID>& shell = m_container->getEdgesAroundVertex ((*it).first);
        for (unsigned int i = 0; i< shell.size(); i++)
            if ( (m_container->getTrianglesAroundEdge (shell[i])).size() == 1)
            {
                field2remove.push_back ((*it).first);
                break;
            }
    }

    //deleting point on border:
    for (unsigned int i = 0; i< field2remove.size(); i++)
    {
        it = map_point2snap.find (field2remove[i]);
        map_point2snap.erase (it);
    }


    //// STEP 4 - Compute new coordinates of point to be snaped, and inform path that point has to be snaped
    field2remove.clear();
    points2Snap.resize (map_point2snap.size());
    unsigned int cpt = 0;
    for (it = map_point2snap.begin(); it != map_point2snap.end(); ++it)
    {
        const size_t size = ((*it).second).size();
        if (size == 1) // for border case or reincision
        {
            points2Snap.resize (points2Snap.size()-1);
            continue;
        }

        points2Snap[ cpt ].push_back ((*it).first); // points2Snap[X][0] => id point to snap
        sofa::defaulttype::Vec<3,double> newCoords;

        // Step 3/3: Compute mean value of all incision point position.
        for (unsigned int j = 0; j<3; j++)
        {
            points2Snap[ cpt ].push_back ( map_point2bary[(*it).first][j]/size ); // points2Snap[X][1 2 3] => real coord of point to snap
        }
        cpt++;

        // Change enum of the first object to snap to POINT, change id and label it as snaped
        topoPath_list[ ((*it).second)[0]] = core::topology::POINT;
        indices_list[ ((*it).second)[0]] = (*it).first;
        coords_list[ ((*it).second)[0]][0] = -1.0;

        // If more objects are concerned, remove them from the path  (need to stock and get out of the loop to delete them)
        for (unsigned int i = 1; i <size; i++)
            field2remove.push_back ((*it).second[i]);
    }

    //// STEP 5 - Modify incision path
    //TODO: verify that one object can't be snaped and considered at staying at the same time
    sort (field2remove.begin(), field2remove.end());

    for (unsigned int i = 1; i <= field2remove.size(); i++) //Delete in reverse order
    {
        topoPath_list.erase (topoPath_list.begin()+field2remove[field2remove.size()-i]);
        indices_list.erase (indices_list.begin()+field2remove[field2remove.size()-i]);
        coords_list.erase (coords_list.begin()+field2remove[field2remove.size()-i]);
    }

    return;
}


template<class DataTypes>
void TriangleSetTopologyAlgorithms<DataTypes>::SnapBorderPath (unsigned int pa, Coord& a, unsigned int pb, Coord& b,
        sofa::helper::vector< sofa::core::topology::TopologyObjectType>& topoPath_list,
        sofa::helper::vector<unsigned int>& indices_list,
        sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& coords_list,
        sofa::helper::vector< sofa::helper::vector<double> >& points2Snap,
        double epsilonSnapBorder)
{
    bool snap_a = false;
    bool snap_b = false;
    bool intersected = true;
    double epsilon = epsilonSnapBorder;

    // Test if point has not already been snap on a point
    for (unsigned int i = 0; i <points2Snap.size(); i++)
    {
        if (points2Snap[i][0] == pa)
            snap_a = true;
        else if (points2Snap[i][0] == pb)
            snap_b = true;

        if (snap_a & snap_b)
            break;
    }

    // Test if point need to be snap on an edge
    if (!snap_a  && topoPath_list[0] == core::topology::TRIANGLE) // this means a is not close to a point, but could be close to an edge
    {
        for (unsigned int i = 0; i<3; i++)
        {
            if (coords_list[0][i] < epsilon)
            {
                const EdgeID theEdge = m_container->getEdgesInTriangle ( indices_list[0])[i];
                bool find = false;
                bool allDone = false;
                bool pointDone = false;
                PointID thePoint = 0;
                if( (m_container->getTrianglesAroundEdge (theEdge)).size() > 1) //snap to point and not edge
                {
                    for (unsigned int j = 0; j<3; j++)
                        if (coords_list[0][j] > 1-epsilon)
                        {
                            thePoint = m_container->getTriangle ( indices_list[0])[j];
                            topoPath_list[0] = core::topology::POINT;
                            indices_list[0] = thePoint;
                            find = true;
                            break;
                        }

                    if(topoPath_list.size() <= 2)
                        break;

                    while (find)
                    {
                        pointDone = true;
                        allDone = true;
                        if (topoPath_list[1] == core::topology::EDGE) // just remove or need to projection?
                        {
                            const sofa::helper::vector <EdgeID>& shell = m_container->getEdgesAroundVertex (thePoint);
                            for (unsigned int k = 0; k< shell.size(); k++)
                            {
                                if (shell[k] == indices_list[1])
                                {
                                    //std::cout << indices_list[1] << std::endl;
                                    topoPath_list.erase (topoPath_list.begin()+1);
                                    indices_list.erase (indices_list.begin()+1);
                                    coords_list.erase (coords_list.begin()+1);
                                    allDone = false;
                                    break;
                                }
                            }
                        }
                        else if (topoPath_list[1] == core::topology::POINT)
                        {
                            if (indices_list[1] == thePoint)
                            {
                                topoPath_list.erase (topoPath_list.begin()+1);
                                indices_list.erase (indices_list.begin()+1);
                                coords_list.erase (coords_list.begin()+1);
                                pointDone = false;
                            }
                        }
                        else
                            find = false;

                        if (pointDone && allDone) //nor one not the other
                            find = false;
                    }
                    break;
                }


                if ((indices_list[1] == theEdge) && (topoPath_list[1] == core::topology::EDGE)) // Only keep this one? or need to project?
                {
                    std::cout <<"************* Just wonder if it is possible!!" << std::endl;
                    topoPath_list.erase (topoPath_list.begin());
                    indices_list.erase (indices_list.begin());
                    coords_list.erase (coords_list.begin());
                    break;
                }
                else // need to create this point by projection
                {
                    sofa::defaulttype::Vec<3,double> thePoint; DataTypes::get(thePoint[0], thePoint[1], thePoint[2], a);

                    sofa::helper::vector< double > new_coord =  m_geometryAlgorithms->computePointProjectionOnEdge (theEdge, thePoint, intersected);

                    if (!intersected)
                        std::cout << " Error: TriangleSetTopologyAlgorithms::SnapBorderPath orthogonal projection failed" << std::endl;

                    topoPath_list[0] = core::topology::EDGE;

                    indices_list[0] = theEdge;
                    coords_list[0][0] = new_coord[1];  // not the same order as barycoef in the incision path
                    coords_list[0][1] = new_coord[0];
                    coords_list[0][2] = 0.0;

                    Edge theEdgeFirst = m_container->getEdge(theEdge);
                    sofa::defaulttype::Vec<3,double> pos1 = m_geometryAlgorithms->computeBaryEdgePoint(theEdgeFirst, new_coord[1]);
                    for (unsigned int j = 0; j<3; j++)
                        a[j]=(float)pos1[j];

                    break;
                }
            }
        }
    }

    // Same for last point
    if (!snap_b  && topoPath_list.back() == core::topology::TRIANGLE) // this means a is not close to a point, but could be close to an edge
    {
        for (unsigned int i = 0; i<3; i++)
        {
            if (coords_list.back()[i] < epsilon)
            {
                const EdgeID theEdge = m_container->getEdgesInTriangle ( indices_list.back())[i];
                bool find = false;
                bool allDone = false;
                bool pointDone = false;
                PointID thePoint = 0;

                if( (m_container->getTrianglesAroundEdge (theEdge)).size() > 1) //snap to point and not edge
                {
                    for (unsigned int j = 0; j<3; j++)
                        if (coords_list.back()[j] > 1-epsilon)
                        {
                            thePoint = m_container->getTriangle ( indices_list.back())[j];
                            topoPath_list.back() = core::topology::POINT;
                            indices_list.back() = thePoint;
                            find = true;
                            break;
                        }

                    if(topoPath_list.size() <= 2)
                        break;

                    while (find)
                    {
                        const size_t pos = topoPath_list.size()-2;
                        pointDone = true;
                        allDone = true;
                        if (topoPath_list[pos] == core::topology::EDGE) // just remove or need to projection?
                        {
                            const sofa::helper::vector <EdgeID> &shell = m_container->getEdgesAroundVertex (thePoint);
                            for (unsigned int k = 0; k< shell.size(); k++)
                            {
                                if (shell[k] == indices_list[pos])
                                {
                                    topoPath_list.erase (topoPath_list.begin()+pos);
                                    indices_list.erase (indices_list.begin()+pos);
                                    coords_list.erase (coords_list.begin()+pos);
                                    allDone = false;
                                    break;
                                }
                            }
                        }
                        else if (topoPath_list[pos] == core::topology::POINT)
                        {
                            if (indices_list[pos] == thePoint)
                            {
                                topoPath_list.erase (topoPath_list.begin()+pos);
                                indices_list.erase (indices_list.begin()+pos);
                                coords_list.erase (coords_list.begin()+pos);
                                pointDone = false;
                            }
                        }
                        else
                            find = false;

                        if (pointDone && allDone) //nor one not the other
                            find = false;
                    }

                    break;
                }


                if ((indices_list[indices_list.size()-2] == theEdge) && (topoPath_list[topoPath_list.size()-2] == core::topology::EDGE)) // Only keep this one? or need to projection?
                {
                    std::cout <<"************* Just wonder if it is possible!!" << std::endl;
                    topoPath_list.pop_back();
                    indices_list.pop_back();
                    coords_list.pop_back();
                    break;
                }
                else
                {
                    sofa::defaulttype::Vec<3,double> thePoint; DataTypes::get(thePoint[0], thePoint[1], thePoint[2], b);
                    sofa::helper::vector< double > new_coord =  m_geometryAlgorithms->computePointProjectionOnEdge (theEdge, thePoint, intersected);

                    if (!intersected)
                        std::cout << " Error: TriangleSetTopologyAlgorithms::SnapBorderPath orthogonal projection failed" << std::endl;

                    topoPath_list.back() = core::topology::EDGE;
                    indices_list.back() = theEdge;
                    coords_list.back()[0] = new_coord[1];
                    coords_list.back()[1] = new_coord[0];
                    coords_list.back()[2] = 0.0;

                    Edge theEdgeLast = m_container->getEdge(theEdge);
                    sofa::defaulttype::Vec<3,double> pos1 = m_geometryAlgorithms->computeBaryEdgePoint(theEdgeLast, new_coord[1]);
                    for (unsigned int j = 0; j<3; j++)
                        a[j]=(float)pos1[j];

                    break;
                }
            }
        }
    }
    return;
}


/** \brief Duplicates the given edges. Only works if at least the first or last point is adjacent to a border.
 * @returns true if the incision succeeded.
 */
template<class DataTypes>
bool TriangleSetTopologyAlgorithms<DataTypes>::InciseAlongEdgeList(const sofa::helper::vector<unsigned int>& edges,
        sofa::helper::vector<unsigned int>& new_points,
        sofa::helper::vector<unsigned int>& end_points,
        bool& reachBorder)
{
    sofa::helper::vector< sofa::helper::vector< PointID > > p_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs;
    PointID next_point = m_container->getNbPoints();
    TriangleID next_triangle = m_container->getNbTriangles();
    sofa::helper::vector< Triangle > new_triangles;
    sofa::helper::vector< TriangleID > new_triangles_id;
    sofa::helper::vector< TriangleID > removed_triangles;
    sofa::helper::vector< sofa::helper::vector< TriangleID > >  triangles_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > >  triangles_barycoefs;


    const size_t nbEdges = edges.size();
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
        for (size_t i=1; i<nbEdges; ++i)
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

    sofa::helper::vector< std::pair<TriangleID,TriangleID> > init_triangles;
    for (size_t i=0; i<nbEdges; ++i)
    {
        const sofa::helper::vector<TriangleID>& shell = m_container->getTrianglesAroundEdge(edges[i]);
        if (shell.size() != 2)
        {
            this->serr << "ERROR: cannot split an edge with " << shell.size() << "!=2 attached triangles. Around edge: " << edges[i] << this->sendl;
            this->serr << "Which is composed of vertex: "<< m_container->getEdge (edges[i]) << this->sendl;
            return false;
        }
        init_triangles.push_back(std::make_pair(shell[0],shell[1]));
    }

    bool beginOnBorder = (m_container->getTrianglesAroundVertex(init_points.front()).size() < m_container->getEdgesAroundVertex(init_points.front()).size());
    bool endOnBorder = (m_container->getTrianglesAroundVertex(init_points.back()).size() < m_container->getEdgesAroundVertex(init_points.back()).size());

    if (!beginOnBorder) end_points.push_back(init_points.front());
    if (!endOnBorder) end_points.push_back(init_points.back());
    else
        reachBorder=true;

    /// STEP 1: Create the new points corresponding the one of the side of the now separated edges
    const size_t first_new_point = beginOnBorder ? 0 : 1;
    const size_t last_new_point = endOnBorder ? init_points.size()-1 : init_points.size()-2;
    std::map<PointID, PointID> splitMap;
    for (size_t i = first_new_point ; i <= last_new_point ; ++i)
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

    //TODO : WARNING THERE SEEMS TO BE A SEG FAULT HERE
    TriangleID t0 = m_container->getTrianglesAroundEdge(edges[0])[0];
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
            const sofa::core::topology::BaseMeshTopology::TrianglesAroundEdge& etri = m_container->getTrianglesAroundEdge(e);
            if (etri.size() != 2) break; // border or non-manifold edge
            if (etri[0] == tid)
                tid = etri[1];
            else
                tid = etri[0];
            p1 = p2;
        }
    }

    // STEP 2b: Find the triangles linking each edge to the next, by starting from the last triangle, rotate around each point until the next point is reached
    for (size_t i = 0 ; i < nbEdges-1 ; ++i)
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

            const sofa::core::topology::BaseMeshTopology::TrianglesAroundEdge& etri = m_container->getTrianglesAroundEdge(e);
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
            const sofa::core::topology::BaseMeshTopology::TrianglesAroundEdge& etri = m_container->getTrianglesAroundEdge(e);
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

            // Taking into account ancestors for adding triangles
            triangles_ancestors.resize (triangles_ancestors.size()+1);
            triangles_barycoefs.resize (triangles_barycoefs.size()+1);

            triangles_ancestors[triangles_ancestors.size()-1].push_back (tid);
            triangles_barycoefs[triangles_barycoefs.size()-1].push_back (1.0); //that is the question... ??
        }
    }

    // FINAL STEP : Apply changes
    // Create all the points registered to be created
    m_modifier->addPointsProcess((unsigned int)p_ancestors.size());

    // Warn for the creation of all the points registered to be created
    m_modifier->addPointsWarning((unsigned int)p_ancestors.size(), p_ancestors, p_baryCoefs);

    //Add and remove triangles lists
    m_modifier->addRemoveTriangles((unsigned int)new_triangles.size(), new_triangles, new_triangles_id, triangles_ancestors, triangles_barycoefs, removed_triangles);

    return true;
}



// Duplicate the given edge. Only works of at least one of its points is adjacent to a border.
template<class DataTypes>
int TriangleSetTopologyAlgorithms<DataTypes>::InciseAlongEdge(unsigned int ind_edge, int* createdPoints)
{
    const Edge & edge0=m_container->getEdge(ind_edge);
    unsigned ind_pa = edge0[0];
    unsigned ind_pb = edge0[1];

    const helper::vector<unsigned>& triangles0 = m_container->getTrianglesAroundEdge(ind_edge);
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
        const EdgesInTriangle& te = m_container->getEdgesInTriangle(ind_tria);

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

        const helper::vector<unsigned>& tes = m_container->getTrianglesAroundEdge(ind_edgea);
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
        const EdgesInTriangle& te = m_container->getEdgesInTriangle(ind_trib);

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

        const helper::vector<unsigned>& tes = m_container->getTrianglesAroundEdge(ind_edgeb);
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

    /// force the creation of TrianglesAroundEdgeArray
    m_container->getTrianglesAroundEdgeArray();
    /// force the creation of TrianglesAroundVertexArray
    m_container->getTrianglesAroundVertexArray();

    //const typename DataTypes::VecCoord& vect_c =topology->getDOF()->read(core::ConstVecCoordId::position())->getValue();
    const unsigned int nb_points = (unsigned int)m_container->getTrianglesAroundVertexArray().size(); //vect_c.size();
    const sofa::helper::vector<Triangle> &vect_t=m_container->getTriangleArray();
    const unsigned int nb_triangles = (unsigned int)vect_t.size();

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
    m_modifier->addTrianglesWarning((unsigned int)triangles_to_create.size(), triangles_to_create, trianglesIndexList);

    // Propagate the topological changes *** not necessary
    //m_modifier->propagateTopologicalChanges();

    // Remove all the triangles registered to be removed
    m_modifier->removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

    return (pb_is_on_border?1:0)+(pa_is_on_border?1:0); // todo: get new edge indice
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TriangleSetTOPOLOGY_INL
