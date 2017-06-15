/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTopologyMapping/Hexa2TetraTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>

#include <SofaBaseTopology/GridTopology.h>

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
using namespace sofa::core::topology;

SOFA_DECL_CLASS(Hexa2TetraTopologicalMapping)

// Register in the Factory
int Hexa2TetraTopologicalMappingClass = core::RegisterObject("Special case of mapping where HexahedronSetTopology is converted to TetrahedronSetTopology")
        .add< Hexa2TetraTopologicalMapping >()

        ;

// Implementation

Hexa2TetraTopologicalMapping::Hexa2TetraTopologicalMapping()
    : swapping(initData(&swapping, false, "swapping","Boolean enabling to swapp hexa-edges\n in order to avoid bias effect"))
{
}

Hexa2TetraTopologicalMapping::~Hexa2TetraTopologicalMapping()
{
}

void Hexa2TetraTopologicalMapping::init()
{
    //sout << "INFO_print : init Hexa2TetraTopologicalMapping" << sendl;

    // INITIALISATION of TETRAHEDRAL mesh from HEXAHEDRAL mesh :

    if (fromModel)
    {

        sout << "INFO_print : Hexa2TetraTopologicalMapping - from = hexa" << sendl;

        if (toModel)
        {

            sout << "INFO_print : Hexa2TetraTopologicalMapping - to = tetra" << sendl;

            TetrahedronSetTopologyContainer *to_tstc;
            toModel->getContext()->get(to_tstc);
            to_tstc->clear();

            toModel->setNbPoints(fromModel->getNbPoints());

            TetrahedronSetTopologyModifier *to_tstm;
            toModel->getContext()->get(to_tstm);

            sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

            Loc2GlobVec.clear();
            Glob2LocMap.clear();

#ifdef SOFA_NEW_HEXA
            int nbcubes = fromModel->getNbHexahedra();
#else
            int nbcubes = fromModel->getNbCubes();
#endif
            // These values are only correct if the mesh is a grid topology
            int nx = 2;
            int ny = 1;
            //int nz = 1;
            {
                topology::GridTopology* grid = dynamic_cast<topology::GridTopology*>(fromModel.get());
                if (grid != NULL)
                {
                    nx = grid->getNx()-1;
                    ny = grid->getNy()-1;
                    //nz = grid->getNz()-1;
                }
            }

            // Tesselation of each cube into 6 tetrahedra
            for (int i=0; i<nbcubes; i++)
            {
#ifdef SOFA_NEW_HEXA
                core::topology::BaseMeshTopology::Hexa c = fromModel->getHexahedron(i);
#define swap(a,b) { int t = a; a = b; b = t; }
                // TODO : swap indexes where needed (currently crash in TriangleSetContainer)
                bool swapped = false;

                if(swapping.getValue())
                {
                    if (!((i%nx)&1))
                    {
                        // swap all points on the X edges
                        swap(c[0],c[1]);
                        swap(c[3],c[2]);
                        swap(c[4],c[5]);
                        swap(c[7],c[6]);
                        swapped = !swapped;
                    }
                    if (((i/nx)%ny)&1)
                    {
                        // swap all points on the Y edges
                        swap(c[0],c[3]);
                        swap(c[1],c[2]);
                        swap(c[4],c[7]);
                        swap(c[5],c[6]);
                        swapped = !swapped;
                    }
                    if ((i/(nx*ny))&1)
                    {
                        // swap all points on the Z edges
                        swap(c[0],c[4]);
                        swap(c[1],c[5]);
                        swap(c[2],c[6]);
                        swap(c[3],c[7]);
                        swapped = !swapped;
                    }
                }
#undef swap
                typedef core::topology::BaseMeshTopology::Tetra Tetra;

                if(!swapped)
                {
                    to_tstm->addTetrahedronProcess(Tetra(c[0],c[5],c[1],c[6]));
                    to_tstm->addTetrahedronProcess(Tetra(c[0],c[1],c[3],c[6]));
                    to_tstm->addTetrahedronProcess(Tetra(c[1],c[3],c[6],c[2]));
                    to_tstm->addTetrahedronProcess(Tetra(c[6],c[3],c[0],c[7]));
                    to_tstm->addTetrahedronProcess(Tetra(c[6],c[7],c[0],c[5]));
                    to_tstm->addTetrahedronProcess(Tetra(c[7],c[5],c[4],c[0]));
                }
                else
                {
                    to_tstm->addTetrahedronProcess(Tetra(c[0],c[5],c[6],c[1]));
                    to_tstm->addTetrahedronProcess(Tetra(c[0],c[1],c[6],c[3]));
                    to_tstm->addTetrahedronProcess(Tetra(c[1],c[3],c[2],c[6]));
                    to_tstm->addTetrahedronProcess(Tetra(c[6],c[3],c[7],c[0]));
                    to_tstm->addTetrahedronProcess(Tetra(c[6],c[7],c[5],c[0]));
                    to_tstm->addTetrahedronProcess(Tetra(c[7],c[5],c[0],c[4]));
                }
#else
                core::topology::BaseMeshTopology::Cube c = fromModel->getCube(i);
                int sym = 0;
                if (!((i%nx)&1)) sym+=1;
                if (((i/nx)%ny)&1) sym+=2;
                if ((i/(nx*ny))&1) sym+=4;
                typedef core::topology::BaseMeshTopology::Tetra Tetra;
                to_tstm->addTetrahedronProcess(Tetra(c[0^sym],c[5^sym],c[1^sym],c[7^sym]));
                to_tstm->addTetrahedronProcess(Tetra(c[0^sym],c[1^sym],c[2^sym],c[7^sym]));
                to_tstm->addTetrahedronProcess(Tetra(c[1^sym],c[2^sym],c[7^sym],c[3^sym]));
                to_tstm->addTetrahedronProcess(Tetra(c[7^sym],c[2^sym],c[0^sym],c[6^sym]));
                to_tstm->addTetrahedronProcess(Tetra(c[7^sym],c[6^sym],c[0^sym],c[5^sym]));
                to_tstm->addTetrahedronProcess(Tetra(c[6^sym],c[5^sym],c[4^sym],c[0^sym]));
#endif
                for(int j=0; j<6; j++)
                    Loc2GlobVec.push_back(i);
                Glob2LocMap[i]=Loc2GlobVec.size()-1;
            }

            //to_tstm->propagateTopologicalChanges();
            to_tstm->notifyEndingEvent();
            //to_tstm->propagateTopologicalChanges();
            Loc2GlobDataVec.endEdit();
        }

    }
}

unsigned int Hexa2TetraTopologicalMapping::getFromIndex(unsigned int /*ind*/)
{

    return 0;
}

void Hexa2TetraTopologicalMapping::updateTopologicalMappingTopDown()
{
// TODO...
}


} // namespace topology

} // namespace component

} // namespace sofa
