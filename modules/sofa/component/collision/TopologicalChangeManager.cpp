#include "TopologicalChangeManager.h"

#include <sofa/component/collision/TriangleModel.h>

#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>

#include <sofa/core/componentmodel/topology/TopologicalMapping.h>

/*
// Includes to distinghuish specific topologies

#include <sofa/component/topology/TetrahedronSetTopology.h>
#include <sofa/component/topology/TriangleSetTopology.h>
#include <sofa/component/topology/HexahedronSetTopology.h>
#include <sofa/component/topology/QuadSetTopology.h>

// Includes to generate, from the input file mesh, the ouptut file mesh with the optimal vertex permutation according to the Reverse CuthillMckee algorithm
// See APPLICATION 1
#include <iostream>
#include <fstream>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/defaulttype/Vec.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/core/objectmodel/Base.h>

// Includes to handle two points suturing from a tetrahedral mesh
// See APPLICATION 2
#include <sofa/component/constraint/FixedConstraint.h>

*/


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::component::topology;

TopologicalChangeManager::TopologicalChangeManager()
{
}

TopologicalChangeManager::~TopologicalChangeManager()
{
}

void TopologicalChangeManager::removeItemsFromTriangleModel(sofa::core::CollisionElementIterator elem2) const
{
    TriangleModel* my_triangle_model = (dynamic_cast<TriangleModel*>(elem2.getCollisionModel()));
    if (my_triangle_model)
    {
        sofa::core::componentmodel::topology::BaseTopology *topo_curr = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(elem2.getCollisionModel()->getContext()->getMainTopology());
        unsigned int ind_curr = elem2.getIndex();

        simulation::tree::GNode *node_curr = dynamic_cast<simulation::tree::GNode*>(topo_curr->getContext());

        std::set< unsigned int > items;
        items.insert(ind_curr);

        bool is_topoMap = true;

        while(is_topoMap)
        {

            is_topoMap = false;
            for(simulation::tree::GNode::ObjectIterator it = node_curr->object.begin(); it != node_curr->object.end(); ++it)
            {

                sofa::core::componentmodel::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping *>(*it);
                if(topoMap != NULL)
                {

                    is_topoMap = true;
                    //unsigned int ind_glob = topoMap->getGlobIndex(ind_curr);
                    //ind_curr = topoMap->getFromIndex(ind_glob);
                    std::set< unsigned int > loc_items = items;
                    items.clear();
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        unsigned int ind_glob = topoMap->getGlobIndex(*it);
                        unsigned int ind = topoMap->getFromIndex(ind_glob);
                        std::cout << *it << " -> "<<ind_glob << " -> "<<ind<<std::endl;
                        items.insert(ind);
                    }

                    topo_curr = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(topoMap->getFrom());
                    node_curr = dynamic_cast<simulation::tree::GNode*>(topo_curr->getContext());

                    break;
                }
            }
        }
        sofa::helper::vector<unsigned int> vitems;
        vitems.reserve(items.size());
        vitems.insert(vitems.end(), items.rbegin(), items.rend());
        topo_curr->getTopologyAlgorithms()->removeItems(vitems);
        /*
        for (std::set< unsigned int >::const_reverse_iterator it=items.rbegin(); it != items.rend(); ++it)
        {
            vitems.clear();
            vitems.push_back(*it);
            std::cout << "removeItems("<<vitems<<")"<<std::endl;
            topo_curr->getTopologyAlgorithms()->removeItems(vitems);
        topo_curr->getTopologyAlgorithms()->notifyEndingEvent();
            topo_curr->propagateTopologicalChanges();
        }
        */

        topo_curr->getTopologyAlgorithms()->notifyEndingEvent();
        topo_curr->propagateTopologicalChanges();


        /*
        //// For APPLICATION 1 : generate, from the input file mesh, the ouptut file mesh with the optimal vertex permutation according to the Reverse CuthillMckee algorithm

        sofa::helper::vector<int> init_inverse_permutation;
        sofa::helper::vector<int>& inverse_permutation = init_inverse_permutation;
        topology::EdgeSetTopology< Vec3Types >* esp = dynamic_cast< topology::EdgeSetTopology< Vec3Types >* >( topo_curr );
        esp->getEdgeSetTopologyAlgorithms()->resortCuthillMckee(inverse_permutation);

        //std::cout << "inverse_permutation : " << std::endl;
        sofa::helper::vector<int> permutation;
        permutation.resize(inverse_permutation.size());
        for (unsigned int i=0; i<inverse_permutation.size(); ++i){
        	permutation[inverse_permutation[i]]=i;
        	//std::cout << i << " -> " << inverse_permutation[i] << std::endl;
        }

        //std::cout << "permutation : " << std::endl;
        //for (unsigned int i=0; i<permutation.size(); ++i){
        	//std::cout << i << " -> " << permutation[i] << std::endl;
        //}



        topo_curr->getTopologyAlgorithms()->renumberPoints((const sofa::helper::vector<unsigned int> &) inverse_permutation, (const sofa::helper::vector<unsigned int> &) permutation);



        std::ofstream myfile;
        myfile.open ((const char *) "D:/PROJ/trunk/Sofa/scenes/Topology/outputMesh.msh");


        std::string fname = topo_curr->getFilename();

        topology::PointSetTopology< Vec3Types >* psp = dynamic_cast< topology::PointSetTopology< Vec3Types >* >( topo_curr );
        using namespace sofa::core::componentmodel::behavior;
        sofa::helper::vector< sofa::defaulttype::Vec<3,double> > p = *psp->getDOF()->getX();

        std::cout << "fname = " << fname << std::endl;

        myfile << "$NOD\n";
        myfile << inverse_permutation.size() <<"\n";

        for (unsigned int i=0; i<inverse_permutation.size(); ++i){

        	double x = (double) p[inverse_permutation[i]][0];
        	double y = (double) p[inverse_permutation[i]][1];
        	double z = (double) p[inverse_permutation[i]][2];

        	myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
        }

        myfile << "$ENDNOD\n";
        myfile << "$ELM\n";

        topology::TetrahedronSetTopology< Vec3Types >* tesp = dynamic_cast< topology::TetrahedronSetTopology< Vec3Types >* >( topo_curr );
        topology::TriangleSetTopology< Vec3Types >* tsp = dynamic_cast< topology::TriangleSetTopology< Vec3Types >* >( topo_curr );
        topology::HexahedronSetTopology< Vec3Types >* hesp = dynamic_cast< topology::HexahedronSetTopology< Vec3Types >* >( topo_curr );
        topology::QuadSetTopology< Vec3Types >* qsp = dynamic_cast< topology::QuadSetTopology< Vec3Types >* >( topo_curr );

        if(tesp){
        	topology::TetrahedronSetTopologyContainer * c_tesp = static_cast< TetrahedronSetTopologyContainer* >(tesp->getTopologyContainer());
        	const sofa::helper::vector<Tetrahedron> &tea=c_tesp->getTetrahedronArray();

        	myfile << tea.size() <<"\n";

        	for (unsigned int i=0; i<tea.size(); ++i){
        		myfile << i+1 << " 4 1 1 4 " << permutation[tea[i][0]]+1 << " " << permutation[tea[i][1]]+1 << " " << permutation[tea[i][2]]+1 << " " << permutation[tea[i][3]]+1 <<"\n";
        	}
        }else{
        	if(tsp){
        		topology::TriangleSetTopologyContainer * c_tsp = static_cast< TriangleSetTopologyContainer* >(tsp->getTopologyContainer());
        		const sofa::helper::vector<topology::Triangle> &ta=c_tsp->getTriangleArray();

        		myfile << ta.size() <<"\n";

        		for (unsigned int i=0; i<ta.size(); ++i){
        			myfile << i+1 << " 2 6 6 3 " << permutation[ta[i][0]]+1 << " " << permutation[ta[i][1]]+1 << " " << permutation[ta[i][2]]+1 <<"\n";
        		}
        	}else{
        		if(hesp){
        			topology::HexahedronSetTopologyContainer * c_hesp = static_cast< HexahedronSetTopologyContainer* >(hesp->getTopologyContainer());
        			const sofa::helper::vector<Hexahedron> &hea=c_hesp->getHexahedronArray();

        			myfile << hea.size() <<"\n";

        			for (unsigned int i=0; i<hea.size(); ++i){
        				myfile << i+1 << " 5 1 1 8 "
        					<< permutation[hea[i][4]]+1 << " " << permutation[hea[i][5]]+1 << " " << permutation[hea[i][1]]+1 << " " << permutation[hea[i][0]]+1 << " "
        					<< permutation[hea[i][7]]+1 << " " << permutation[hea[i][6]]+1 << " " << permutation[hea[i][2]]+1 << " " << permutation[hea[i][3]]+1
        				<<"\n";
        			}
        		}else{
        			if(qsp){
        				topology::QuadSetTopologyContainer * c_qsp = static_cast< QuadSetTopologyContainer* >(qsp->getTopologyContainer());
        				const sofa::helper::vector<Quad> &qa=c_qsp->getQuadArray();

        				myfile << qa.size() <<"\n";

        				for (unsigned int i=0; i<qa.size(); ++i){
        					myfile << i+1 << " 3 1 1 4 " << permutation[qa[i][0]]+1 << " " << permutation[qa[i][1]]+1 << " " << permutation[qa[i][2]]+1 << " " << permutation[qa[i][3]]+1 <<"\n";
        				}
        			}
        		}
        	}
        }

        myfile << "$ENDELM\n";

        myfile.close();

        */


    }
}

// Handle Removing of topological element (from any type of topology)
void TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionElementIterator elem2) const
{
    if(dynamic_cast<TriangleModel*>(elem2.getCollisionModel())!= NULL)
    {
        removeItemsFromTriangleModel(elem2);
    }
}


// Intermediate method to handle cutting
bool TopologicalChangeManager::incisionTriangleSetTopology(topology::TriangleSetTopology< Vec3Types >* tsp)
{
    const Vec<3,double>& a= (const Vec<3,double>) incision.a_init;
    const Vec<3,double>& b= (const Vec<3,double>) incision.b_init;

    const unsigned int &ind_ta = (const unsigned int) incision.ind_ta_init;
    const unsigned int &ind_tb = (const unsigned int) incision.ind_tb_init;

    unsigned int& a_last = incision.a_last_init;
    sofa::helper::vector< unsigned int >& a_p12_last = incision.a_p12_last_init;
    sofa::helper::vector< unsigned int >& a_i123_last = incision.a_i123_last_init;

    unsigned int& b_last = incision.b_last_init;
    sofa::helper::vector< unsigned int >& b_p12_last = incision.b_p12_last_init;
    sofa::helper::vector< unsigned int >& b_i123_last = incision.b_i123_last_init;

    bool is_prepared=!((a[0]==b[0] && a[1]==b[1] && a[2]==b[2]) || (incision.ind_ta_init == incision.ind_tb_init));

    if(is_prepared)
    {
        sofa::helper::vector< sofa::helper::vector<unsigned int> > new_points_init;
        sofa::helper::vector< sofa::helper::vector<unsigned int> > closest_vertices_init;
        sofa::helper::vector< sofa::helper::vector<unsigned int> > &new_points = new_points_init;
        sofa::helper::vector< sofa::helper::vector<unsigned int> > &closest_vertices = closest_vertices_init;

        bool is_fully_cut = tsp->getTriangleSetTopologyAlgorithms()->InciseAlongPointsList(incision.is_first_cut, a, b, ind_ta, ind_tb, a_last, a_p12_last, a_i123_last, b_last, b_p12_last, b_i123_last, new_points, closest_vertices);

        // notify the end for the current sequence of topological change events
        tsp->getTriangleSetTopologyAlgorithms()->notifyEndingEvent();

        tsp->propagateTopologicalChanges();

        incision.is_first_cut = false;

        return is_fully_cut;
    }
    else
    {
        return false;
    }
}

// Handle Cutting (activated only for a triangular topology), using global variables to register the two last input points
bool TopologicalChangeManager::incisionCollisionModel(sofa::core::CollisionElementIterator elem2, Vector3& pos,
        const bool firstInput, const bool isCut)
{
    if(dynamic_cast<TriangleModel*>(elem2.getCollisionModel())!= NULL)
    {
        return incisionTriangleModel(elem2, pos, firstInput, isCut);
    }

    return false;
}

bool TopologicalChangeManager::incisionTriangleModel(sofa::core::CollisionElementIterator elem2, Vector3& pos,
        const bool firstInput, const bool isCut)
{
    Triangle triangle(elem2);
    TriangleModel* model2 = triangle.getCollisionModel();

    // Test if a TopologicalMapping (by default from TetrahedronSetTopology to TriangleSetTopology) exists :

    bool is_TopologicalMapping = false;
    //bool is_FixedConstraint = false; // for Application 2
    sofa::core::componentmodel::topology::BaseTopology *topo_curr = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(elem2.getCollisionModel()->getContext()->getMainTopology());

    simulation::tree::GNode* parent2 = dynamic_cast<simulation::tree::GNode*>(model2->getContext());

    for (simulation::tree::GNode::ObjectIterator it = parent2->object.begin(); it != parent2->object.end(); ++it)
    {
        //std::cout << "INFO : name of GNode = " << (*it)->getName() <<  std::endl;

        if (dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping *>(*it)!= NULL)
        {
            is_TopologicalMapping=true;
            sofa::core::componentmodel::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping *>(*it);
            topo_curr = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(topoMap->getFrom());
        }

    }

    // try to catch the topology associated to the detected object (a TriangleSetTopology is expected)
    topology::TriangleSetTopology< Vec3Types >* tsp = dynamic_cast< topology::TriangleSetTopology< Vec3Types >* >( elem2.getCollisionModel()->getContext()->getMainTopology() );

    if(!is_TopologicalMapping)
    {
        // no TopologicalMapping

        if (tsp) // TriangleSetTopology
        {
            if (firstInput)
            {
                incision.a_init[0] = pos[0];
                incision.a_init[1] = pos[1];
                incision.a_init[2] = pos[2];
                incision.ind_ta_init = elem2.getIndex();

                incision.is_first_cut = true;
            }
            else if (isCut)
            {
                incision.b_init[0] = pos[0];
                incision.b_init[1] = pos[1];
                incision.b_init[2] = pos[2];

                incision.ind_tb_init = elem2.getIndex();

                if(incisionTriangleSetTopology(tsp))
                {
                    // full cut
                    incision.a_init[0] = pos[0];
                    incision.a_init[1] = pos[1];
                    incision.a_init[2] = pos[2];
                    incision.ind_ta_init = elem2.getIndex();

                    sofa::helper::vector<unsigned int> components_init;
                    sofa::helper::vector<unsigned int>& components = components_init;
                    int num = tsp->getEdgeSetTopologyContainer()->getNumberConnectedComponents(components);
                    std::cout << "Number of connected components : " << num << endl;
                    //sofa::helper::vector<int>::size_type i;
                    //for (i = 0; i != components.size(); ++i)
                    //  std::cout << "Vertex " << i <<" is in component " << components[i] << endl;
                }
                else
                {
                    sofa::helper::vector<unsigned int> components_init;
                    sofa::helper::vector<unsigned int>& components = components_init;
                    int num = tsp->getEdgeSetTopologyContainer()->getNumberConnectedComponents(components);
                    std::cout << "Number of connected components : " << num << endl;
                    //sofa::helper::vector<int>::size_type i;
                    //for (i = 0; i != components.size(); ++i)
                    //  std::cout << "Vertex " << i <<" is in component " << components[i] << endl;
                    return true; // change state to ATTACHED;
                }
            }
        }
    }
    else
    {
        // there is a TetrahedronSetTopology over the TriangleSetTopology

        /*
        //// For APPLICATION 2 : handle two points suturing from a tetrahedral mesh

        if (tsp) // TriangleSetTopology
        {
        	if (firstInput)
        	{
        		incision.a_init[0] = pos[0];
        		incision.a_init[1] = pos[1];
        		incision.a_init[2] = pos[2];
        		incision.ind_ta_init = elem2.getIndex();

        		incision.is_first_cut = true;
        	}
        	else if (isCut)
        	{
        		incision.b_init[0] = pos[0];
        		incision.b_init[1] = pos[1];
        		incision.b_init[2] = pos[2];

        		incision.ind_tb_init = elem2.getIndex();

        		unsigned int ind1_init = 0; unsigned int ind2_init = 0;
        		unsigned int &ind1 = ind1_init;
        		unsigned int &ind2 = ind2_init;

        		bool is_fully_cut = tsp->getTriangleSetTopologyAlgorithms()->Suture2Points(incision.ind_ta_init, incision.ind_tb_init, ind1, ind2);

        		parent2 = dynamic_cast<simulation::tree::GNode*>(topo_curr->getContext());
        		for (simulation::tree::GNode::ObjectIterator it = parent2->object.begin(); it != parent2->object.end(); ++it)
        		{
        			//std::cout << "INFO : name of GNode = " << (*it)->getName() <<  std::endl;

        			if (dynamic_cast<sofa::component::constraint::FixedConstraint<Vec3Types> *>(*it)!= NULL)
        			{

        				is_FixedConstraint=true;
        				sofa::component::constraint::FixedConstraint<Vec3Types> *fixConstraint = dynamic_cast<sofa::component::constraint::FixedConstraint<Vec3Types> *>(*it);

        				fixConstraint->addConstraint(ind1);
        				tsp->getTriangleSetTopologyAlgorithms()->notifyEndingEvent();
        				tsp->propagateTopologicalChanges();

        				fixConstraint->addConstraint(ind2);
        				tsp->getTriangleSetTopologyAlgorithms()->notifyEndingEvent();
        				tsp->propagateTopologicalChanges();

        			}
        		}

        	}
        }

        */



    }

    return false;
}


} // namespace collision

} // namespace component

} // namespace sofa
