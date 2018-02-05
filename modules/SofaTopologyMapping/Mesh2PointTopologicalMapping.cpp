/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaTopologyMapping/Mesh2PointTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <sofa/core/topology/TopologyChange.h>

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
using helper::vector;

SOFA_DECL_CLASS ( Mesh2PointTopologicalMapping )

// Register in the Factory
int Mesh2PointTopologicalMappingClass = core::RegisterObject ( "This class maps any mesh primitive (point, edge, triangle...) into a point using a relative position from the primitive" )
        .add< Mesh2PointTopologicalMapping >()
        ;

// Implementation
Mesh2PointTopologicalMapping::Mesh2PointTopologicalMapping ()
    : pointBaryCoords ( initData ( &pointBaryCoords, "pointBaryCoords", "Coordinates for the points of the output topology created from the points of the input topology" ) ),
      edgeBaryCoords ( initData ( &edgeBaryCoords, "edgeBaryCoords", "Coordinates for the points of the output topology created from the edges of the input topology" ) ),
      triangleBaryCoords ( initData ( &triangleBaryCoords, "triangleBaryCoords", "Coordinates for the points of the output topology created from the triangles of the input topology" ) ),
      quadBaryCoords ( initData ( &quadBaryCoords, "quadBaryCoords", "Coordinates for the points of the output topology created from the quads of the input topology" ) ),
      tetraBaryCoords ( initData ( &tetraBaryCoords, "tetraBaryCoords", "Coordinates for the points of the output topology created from the tetra of the input topology" ) ),
      hexaBaryCoords ( initData ( &hexaBaryCoords, "hexaBaryCoords", "Coordinates for the points of the output topology created from the hexa of the input topology" ) ),
      copyEdges ( initData ( &copyEdges, false, "copyEdges", "Activate mapping of input edges into the output topology (requires at least one item in pointBaryCoords)" ) ),
      copyTriangles ( initData ( &copyTriangles, false, "copyTriangles", "Activate mapping of input triangles into the output topology (requires at least one item in pointBaryCoords)" ) ),
      copyTetrahedra ( initData ( &copyTetrahedra, false, "copyTetrahedra", "Activate mapping of input tetrahedra into the output topology (requires at least one item in pointBaryCoords)" ) ),
       initDone(false)
{
    pointBaryCoords.setGroup("BaryCoords");
    edgeBaryCoords.setGroup("BaryCoords");
    triangleBaryCoords.setGroup("BaryCoords");
    quadBaryCoords.setGroup("BaryCoords");
    tetraBaryCoords.setGroup("BaryCoords");
    hexaBaryCoords.setGroup("BaryCoords");
}

void Mesh2PointTopologicalMapping::init()
{
    initDone = true;
    if(fromModel)
    {
        if(toModel)
        {
            int toModelLastPointIndex = 0;
            toModel->clear();

            PointSetTopologyModifier *toPointMod = NULL;
            toModel->getContext()->get(toPointMod, sofa::core::objectmodel::BaseContext::Local);
            EdgeSetTopologyModifier *toEdgeMod = NULL;
            toModel->getContext()->get(toEdgeMod, sofa::core::objectmodel::BaseContext::Local);
            TriangleSetTopologyModifier *toTriangleMod = NULL;
            toModel->getContext()->get(toTriangleMod, sofa::core::objectmodel::BaseContext::Local);
			TetrahedronSetTopologyModifier *toTetrahedronMod = NULL;
            toModel->getContext()->get(toTetrahedronMod, sofa::core::objectmodel::BaseContext::Local);
            //QuadSetTopologyModifier *toQuadMod = NULL;
            //TetrahedronSetTopologyModifier *toTetrahedronMod = NULL;
            //HexahedronSetTopologyModifier *toHexahedronMod = NULL;


            if (copyEdges.getValue() && pointBaryCoords.getValue().empty())
            {
                serr << "copyEdges requires at least one item in pointBaryCoords" << sendl;
                copyEdges.setValue(false);
            }

            if (copyTriangles.getValue() && pointBaryCoords.getValue().empty())
            {
                serr << "copyTriangles requires at least one item in pointBaryCoords" << sendl;
                copyTriangles.setValue(false);
            }
           if (copyTetrahedra.getValue() && pointBaryCoords.getValue().empty())
            {
                serr << "copyTetrahedra requires at least one item in pointBaryCoords" << sendl;
                copyTetrahedra.setValue(false);
            }

            // point to point mapping
            if (!pointBaryCoords.getValue().empty())
            {
                pointsMappedFrom[POINT].resize(fromModel->getNbPoints());
                for (int i=0; i<fromModel->getNbPoints(); i++)
                {
                    toModelLastPointIndex+=addInputPoint(i);
                }
            }

            // edge to point mapping
            if (!edgeBaryCoords.getValue().empty())
            {
                pointsMappedFrom[EDGE].resize(fromModel->getNbEdges());
                for (int i=0; i<fromModel->getNbEdges(); i++)
                {
                    addInputEdge(i, NULL);
                }
            }

            // edge to edge identity mapping
            if (copyEdges.getValue())
            {
                sout << "Copying " << fromModel->getNbEdges() << " edges" << sendl;
                for (int i=0; i<fromModel->getNbEdges(); i++)
                {
                    Edge e = fromModel->getEdge(i);
                    for (unsigned int j=0; j<e.size(); ++j)
                        e[j] = pointsMappedFrom[POINT][e[j]][0];
                    if (toEdgeMod)
                        toEdgeMod->addEdgeProcess(e);
                    else
                        toModel->addEdge(e[0],e[1]);
                }
            }

            // triangle to point mapping
            if (!triangleBaryCoords.getValue().empty())
            {
                pointsMappedFrom[TRIANGLE].resize(fromModel->getNbTriangles());
                for (int i=0; i<fromModel->getNbTriangles(); i++)
                {
                    addInputTriangle(i, NULL);
                }
            }

            // triangle to triangle identity mapping
            if (copyTriangles.getValue())
            {
                sout << "Copying " << fromModel->getNbTriangles() << " triangles" << sendl;
                for (int i=0; i<fromModel->getNbTriangles(); i++)
                {
                    Triangle t = fromModel->getTriangle(i);
                    for (unsigned int j=0; j<t.size(); ++j)
                        t[j] = pointsMappedFrom[POINT][t[j]][0];
                    if (toTriangleMod)
                        toTriangleMod->addTriangleProcess(t);
                    else
                        toModel->addTriangle(t[0],t[1],t[2]);
                }
            }

            // quad to point mapping
            if (!quadBaryCoords.getValue().empty())
            {
                pointsMappedFrom[QUAD].resize(fromModel->getNbQuads());
                for (int i=0; i<fromModel->getNbQuads(); i++)
                {
                    for (unsigned int j=0; j<quadBaryCoords.getValue().size(); j++)
                    {
                        Quad q = fromModel->getQuad(i);

                        Vec3d p0(fromModel->getPX(q[0]), fromModel->getPY(q[0]), fromModel->getPZ(q[0]));
                        Vec3d p1(fromModel->getPX(q[1]), fromModel->getPY(q[1]), fromModel->getPZ(q[1]));
                        Vec3d p2(fromModel->getPX(q[2]), fromModel->getPY(q[2]), fromModel->getPZ(q[2]));
                        Vec3d p3(fromModel->getPX(q[3]), fromModel->getPY(q[3]), fromModel->getPZ(q[3]));

                        double fx = quadBaryCoords.getValue()[j][0];
                        double fy = quadBaryCoords.getValue()[j][1];

                        Vec3d result =  p0 * ((1-fx) * (1-fy))
                                + p1 * ((  fx) * (1-fy))
                                + p2 * ((1-fx) * (  fy))
                                + p3 * ((  fx) * (  fy));

                        toModel->addPoint(result[0], result[1], result[2]);

                        pointsMappedFrom[QUAD][i].push_back(toModelLastPointIndex);
                        pointSource.push_back(std::make_pair(QUAD,i));
                        toModelLastPointIndex++;
                    }
                }
            }

            // tetrahedron to point mapping
            if (!tetraBaryCoords.getValue().empty())
            {
                pointsMappedFrom[TETRA].resize(fromModel->getNbTetrahedra());
                for (int i=0; i<fromModel->getNbTetrahedra(); i++)
                {
					addInputTetrahedron(i, NULL);
                }
            }
			// triangle to triangle identity mapping
            if (copyTetrahedra.getValue())
            {

                sout << "Copying " << fromModel->getNbTetrahedra() << " tetrahedra" << sendl;
                for (int i=0; i<fromModel->getNbTetrahedra(); i++)
                {
                    Tetrahedron t = fromModel->getTetrahedron(i);
                    for (unsigned int j=0; j<t.size(); ++j)
                        t[j] = pointsMappedFrom[POINT][t[j]][0];
                    if (toTetrahedronMod)
                        toTetrahedronMod->addTetrahedronProcess(t);
                    else
                        toModel->addTetra(t[0],t[1],t[2],t[3]);
                }
            }
            // hexahedron to point mapping
            if (!hexaBaryCoords.getValue().empty())
            {
                pointsMappedFrom[HEXA].resize(fromModel->getNbHexahedra());
                for (int i=0; i<fromModel->getNbHexahedra(); i++)
                {
                    for (unsigned int j=0; j<hexaBaryCoords.getValue().size(); j++)
                    {
                        Hexahedron h = fromModel->getHexahedron(i);

                        Vec3d p0(fromModel->getPX(h[0]), fromModel->getPY(h[0]), fromModel->getPZ(h[0]));
                        Vec3d p1(fromModel->getPX(h[1]), fromModel->getPY(h[1]), fromModel->getPZ(h[1]));
						Vec3d p2(fromModel->getPX(h[3]), fromModel->getPY(h[3]), fromModel->getPZ(h[3]));
						Vec3d p3(fromModel->getPX(h[2]), fromModel->getPY(h[2]), fromModel->getPZ(h[2]));
                        Vec3d p4(fromModel->getPX(h[4]), fromModel->getPY(h[4]), fromModel->getPZ(h[4]));
                        Vec3d p5(fromModel->getPX(h[5]), fromModel->getPY(h[5]), fromModel->getPZ(h[5]));
						Vec3d p6(fromModel->getPX(h[7]), fromModel->getPY(h[7]), fromModel->getPZ(h[7]));
						Vec3d p7(fromModel->getPX(h[6]), fromModel->getPY(h[6]), fromModel->getPZ(h[6]));

                        double fx = hexaBaryCoords.getValue()[j][0];
                        double fy = hexaBaryCoords.getValue()[j][1];
                        double fz = hexaBaryCoords.getValue()[j][2];

                        Vec3d result =  p0 * ((1-fx) * (1-fy) * (1-fz))
                                + p1 * ((  fx) * (1-fy) * (1-fz))
                                + p2 * ((1-fx) * (  fy) * (1-fz))
                                + p3 * ((  fx) * (  fy) * (1-fz))
                                + p4 * ((1-fx) * (1-fy) * (  fz))
                                + p5 * ((  fx) * (1-fy) * (  fz))
								+ p6 * ((1-fx) * (  fy) * (  fz))
								+ p7 * ((  fx) * (  fy) * (  fz));

                        toModel->addPoint(result[0], result[1], result[2]);

                        pointsMappedFrom[HEXA][i].push_back(toModelLastPointIndex);
                        pointSource.push_back(std::make_pair(HEXA,i));
                        toModelLastPointIndex++;
                    }
                }
            }
            internalCheck("init");
        }
    }
}

/// Check consistency of internal maps and output topology
bool Mesh2PointTopologicalMapping::internalCheck(const char* step, const helper::fixed_array <int, NB_ELEMENTS >& nbInputRemoved)
{
    bool ok = true;
    unsigned int nbPOut = (unsigned int)toModel->getNbPoints();
    if (nbPOut != pointSource.size())
    {
        serr << "Internal Error after " << step << ": pointSource size " << pointSource.size() << " != output topology size " << nbPOut << sendl;
        ok = false;
    }
    unsigned int nbPMapped = 0;
    for (int type=0; type<NB_ELEMENTS; ++type)
    {
        const vector< vector<int> >& pointsMapped = pointsMappedFrom[type];
        std::string typestr;
        unsigned int nbEIn = 0;
        unsigned int nbEPOut = 0;
        switch (type)
        {
        case POINT :    typestr="Point";    nbEIn = fromModel->getNbPoints();     nbEPOut = pointBaryCoords.getValue().size(); break;
        case EDGE :     typestr="Edge";     nbEIn = fromModel->getNbEdges();      nbEPOut = edgeBaryCoords.getValue().size(); break;
        case TRIANGLE : typestr="Triangle"; nbEIn = fromModel->getNbTriangles();  nbEPOut = triangleBaryCoords.getValue().size(); break;
        case QUAD :     typestr="Quad";     nbEIn = fromModel->getNbQuads();      nbEPOut = quadBaryCoords.getValue().size(); break;
        case TETRA :    typestr="Tetra";    nbEIn = fromModel->getNbTetrahedra(); nbEPOut = tetraBaryCoords.getValue().size(); break;
        case HEXA :     typestr="Hexa";     nbEIn = fromModel->getNbHexahedra();  nbEPOut = hexaBaryCoords.getValue().size(); break;
        default :       typestr="Unknown";  break;
        }
        nbEIn -= nbInputRemoved[type];
        if (pointsMapped.empty())
        {
            if (nbEIn && nbEPOut)
            {
                serr << "Internal Error after " << step << ": pointsMappedFrom" << typestr << " is empty while there should be " << nbEPOut << " generated points per input " << typestr << sendl;
                ok = false;
            }
            continue;
        }

        if (nbEIn != pointsMapped.size())
        {
            serr << "Internal Error after " << step << ": pointsMappedFrom" << typestr << " size " << pointsMapped.size() << " != input topology size " << nbEIn;
            if (nbInputRemoved[type]) serr << " (including " << nbInputRemoved[type] << " removed input elements)";
            serr << sendl;
            ok = false;
        }
        for (unsigned int es = 0; es < pointsMapped.size(); ++es)
        {
            if (pointsMapped[es].size() != nbEPOut)
            {
                serr << "Internal Error after " << step << ":     pointsMappedFrom" << typestr << "[" << es << "] size " << pointsMapped[es].size() << " != barycoords size " << nbEPOut << sendl;
                ok = false;
            }
            for (unsigned int j = 0; j < pointsMapped[es].size(); ++j)
            {
                if ((unsigned)pointsMapped[es][j] >= nbPOut)
                {
                    serr << "Internal Error after " << step << ":     pointsMappedFrom" << typestr << "[" << es << "][" << j << "] = " << pointsMapped[es][j] << " >= " << nbPOut << sendl;
                    ok = false;
                }
            }
        }
        nbPMapped += nbEIn * nbEPOut;
    }
    if (nbPOut != nbPMapped + pointsToRemove.size())
    {
        serr << "Internal Error after " << step << ": " << nbPOut << " mapped points + " << pointsToRemove.size() << " removed points != output topology size " << nbPOut << sendl;
        ok = false;
    }
    if (copyEdges.getValue())
    {
        if (fromModel->getNbEdges() - nbInputRemoved[EDGE] != toModel->getNbEdges())
        {
            serr << "Internal Error after " << step << ": edges were copied, yet output edges size " << toModel->getNbEdges() << " - " << nbInputRemoved[EDGE] << " != input edges size " << fromModel->getNbEdges();
            if (nbInputRemoved[EDGE]) serr << " - " << nbInputRemoved[EDGE];
            serr << sendl;
            ok = false;
        }
    }
    if (copyTriangles.getValue())
    {
        if (fromModel->getNbTriangles() - nbInputRemoved[TRIANGLE] != toModel->getNbTriangles())
        {
            serr << "Internal Error after " << step << ": triangles were copied, yet output triangles size " << toModel->getNbTriangles() << " != input triangles size " << fromModel->getNbTriangles();
            if (nbInputRemoved[TRIANGLE]) serr << " - " << nbInputRemoved[TRIANGLE];
            serr << sendl;
            ok = false;
        }
    }
	if (copyTetrahedra.getValue())
    {
        if (fromModel->getNbTetrahedra() - nbInputRemoved[TETRA] != toModel->getNbTetrahedra())
        {
            serr << "Internal Error after " << step << ": tetrahedra were copied, yet output tetrahedra size " << toModel->getNbTetrahedra() << " != input tetrahedra size " << fromModel->getNbTetrahedra();
            if (nbInputRemoved[TETRA]) serr << " - " << nbInputRemoved[TETRA];
            serr << sendl;
            ok = false;
        }
    }
    sout << "Internal check done after " << step << ", " << fromModel->getNbPoints();
    if (nbInputRemoved[POINT]) sout << " - " << nbInputRemoved[POINT];
    sout << " input points, " << nbPOut;
    if (pointsToRemove.size()) sout << " - " << pointsToRemove.size();
    sout << " generated points";
    if (copyEdges.getValue()) sout << ", " << toModel->getNbEdges() << " generated edges";
    if (copyTriangles.getValue()) sout << ", " << toModel->getNbTriangles() << " generated triangles";
    if (copyTetrahedra.getValue()) sout << ", " << toModel->getNbTetrahedra() << " generated tetrahedra";
    sout << "." << sendl;
    return ok;
}


size_t Mesh2PointTopologicalMapping::addInputPoint(unsigned int i, PointSetTopologyModifier* toPointMod)
{
    if( pointsMappedFrom[POINT].size() < i+1)
        pointsMappedFrom[POINT].resize(i+1);
    else
        pointsMappedFrom[POINT][i].clear();

    const vector< Vec3d > &pBaryCoords = pointBaryCoords.getValue();

    if (toPointMod)
    {
        toPointMod->addPointsProcess(pBaryCoords.size());
    }
    else
    {
        for (unsigned int j = 0; j < pBaryCoords.size(); j++)
        {        
            toModel->addPoint(fromModel->getPX(i) + pBaryCoords[j][0], fromModel->getPY(i) + pBaryCoords[j][1], fromModel->getPZ(i) + pBaryCoords[j][2]);
        }
    }
    
    for (unsigned int j = 0; j < pBaryCoords.size(); j++)
    {        
        pointsMappedFrom[POINT][i].push_back(pointSource.size());
        pointSource.push_back(std::make_pair(POINT, i));
    }
    
    if (toPointMod)
    {  
        helper::vector< helper::vector< unsigned int > > ancestors;
        helper::vector< helper::vector< double       > > coefs;
        toPointMod->addPointsWarning(pBaryCoords.size(), ancestors, coefs);
    }
    
    return pointBaryCoords.getValue().size();

}

void Mesh2PointTopologicalMapping::addInputEdge(unsigned int i, PointSetTopologyModifier* toPointMod)
{
    if (pointsMappedFrom[EDGE].size() < i + 1)
        pointsMappedFrom[EDGE].resize(i + 1);
    else
        pointsMappedFrom[EDGE][i].clear();

    Edge e = fromModel->getEdge(i);
    const vector< Vec3d > &eBaryCoords = edgeBaryCoords.getValue();

    Vec3d p0(fromModel->getPX(e[0]), fromModel->getPY(e[0]), fromModel->getPZ(e[0]));
    Vec3d p1(fromModel->getPX(e[1]), fromModel->getPY(e[1]), fromModel->getPZ(e[1]));

    for (unsigned int j = 0; j < eBaryCoords.size(); j++)
    {
        pointsMappedFrom[EDGE][i].push_back(pointSource.size());
        pointSource.push_back(std::make_pair(EDGE, i));
    }

    if (toPointMod)
    {
        toPointMod->addPointsProcess(eBaryCoords.size());
    }
    else
    {
        for (unsigned int j = 0; j < eBaryCoords.size(); j++)
        {
            double fx = eBaryCoords[j][0];

            Vec3d result = p0 * (1 - fx) + p1 * fx;

            toModel->addPoint(result[0], result[1], result[2]);
        }
    }

    if (toPointMod)
    {
        helper::vector< helper::vector< unsigned int > > ancestors;
        helper::vector< helper::vector< double       > > coefs;
        toPointMod->addPointsWarning(eBaryCoords.size(), ancestors, coefs);
    }
}

void Mesh2PointTopologicalMapping::addInputTriangle(unsigned int i, PointSetTopologyModifier* toPointMod)
{
    if (pointsMappedFrom[TRIANGLE].size() < i+1)
        pointsMappedFrom[TRIANGLE].resize(i+1);
    else
        pointsMappedFrom[TRIANGLE][i].clear();

    Triangle t = fromModel->getTriangle(i);
    const vector< Vec3d > &tBaryCoords = triangleBaryCoords.getValue();

    Vec3d p0(fromModel->getPX(t[0]), fromModel->getPY(t[0]), fromModel->getPZ(t[0]));
    Vec3d p1(fromModel->getPX(t[1]), fromModel->getPY(t[1]), fromModel->getPZ(t[1]));
    Vec3d p2(fromModel->getPX(t[2]), fromModel->getPY(t[2]), fromModel->getPZ(t[2]));

    for (unsigned int j = 0; j < tBaryCoords.size(); j++)
    {
        pointsMappedFrom[TRIANGLE][i].push_back(pointSource.size());
        pointSource.push_back(std::make_pair(TRIANGLE,i));
    }

    if (toPointMod)
    {
        toPointMod->addPointsProcess(tBaryCoords.size());
    }
    else
    {
        for (unsigned int j = 0; j < tBaryCoords.size(); j++)
        {
            double fx = tBaryCoords[j][0];
            double fy = tBaryCoords[j][1];

            Vec3d result =  p0 * (1-fx-fy) + p1 * fx + p2 * fy;         

            toModel->addPoint(result[0], result[1], result[2]);
        }
    }

    if (toPointMod)
    {
        helper::vector< helper::vector< unsigned int > > ancestors;
        helper::vector< helper::vector< double       > > coefs;
        toPointMod->addPointsWarning(tBaryCoords.size(), ancestors, coefs);
    }
}


void Mesh2PointTopologicalMapping::addInputTetrahedron(unsigned int i, PointSetTopologyModifier* toPointMod)
{
    if (pointsMappedFrom[TETRA].size() < i+1)
        pointsMappedFrom[TETRA].resize(i+1);
    else
        pointsMappedFrom[TETRA][i].clear();

    Tetrahedron t = fromModel->getTetrahedron(i);
    const vector< Vec3d > &tBaryCoords = tetraBaryCoords.getValue();

    Vec3d p0(fromModel->getPX(t[0]), fromModel->getPY(t[0]), fromModel->getPZ(t[0]));
    Vec3d p1(fromModel->getPX(t[1]), fromModel->getPY(t[1]), fromModel->getPZ(t[1]));
    Vec3d p2(fromModel->getPX(t[2]), fromModel->getPY(t[2]), fromModel->getPZ(t[2]));
    Vec3d p3(fromModel->getPX(t[3]), fromModel->getPY(t[3]), fromModel->getPZ(t[3]));

    for (unsigned int j = 0; j < tBaryCoords.size(); j++)
    {
        pointsMappedFrom[TETRA][i].push_back(pointSource.size());
        pointSource.push_back(std::make_pair(TETRA,i));
    }

    if (toPointMod)
    {
        toPointMod->addPointsProcess(tBaryCoords.size());
    }
    else
    {
        for (unsigned int j = 0; j < tBaryCoords.size(); j++)
        {
            double fx = tBaryCoords[j][0];
            double fy = tBaryCoords[j][1];
	        double fz = tBaryCoords[j][2];

            Vec3d result =  p0 * (1-fx-fy-fz) + p1 * fx + p2 * fy +p3*fz;         

            toModel->addPoint(result[0], result[1], result[2]);
        }
    }

    if (toPointMod)
    {
        helper::vector< helper::vector< unsigned int > > ancestors;
        helper::vector< helper::vector< double       > > coefs;
        toPointMod->addPointsWarning(tBaryCoords.size(), ancestors, coefs);
    }
}

void Mesh2PointTopologicalMapping::updateTopologicalMappingTopDown()
{
    if(fromModel && toModel && initDone)
    {
        std::list<const TopologyChange *>::const_iterator changeIt=fromModel->beginChange();
        std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

        PointSetTopologyModifier *toPointMod = NULL;
        EdgeSetTopologyModifier *toEdgeMod = NULL;
        TriangleSetTopologyModifier *toTriangleMod = NULL;
        TetrahedronSetTopologyModifier *toTetrahedronMod = NULL;
        //QuadSetTopologyModifier *toQuadMod = NULL;
        //HexahedronSetTopologyModifier *toHexahedronMod = NULL;
        toModel->getContext()->get(toPointMod, sofa::core::objectmodel::BaseContext::Local);
        bool check = false;
        helper::fixed_array <int, NB_ELEMENTS > nbInputRemoved;
        nbInputRemoved.assign(0);
        std::string laststep = "";
        while( changeIt != itEnd )
        {
            TopologyChangeType changeType = (*changeIt)->getChangeType();
            laststep += " ";
            laststep += sofa::core::topology::parseTopologyChangeTypeToString(changeType);
            switch( changeType )
            {
            case core::topology::POINTSINDICESSWAP:
            {
                unsigned int i1 = ( static_cast< const PointsIndicesSwap * >( *changeIt ) )->index[0];
                unsigned int i2 = ( static_cast< const PointsIndicesSwap* >( *changeIt ) )->index[1];
				sout << "INPUT SWAP POINTS "<<i1 << " " << i2 << sendl;
                swapInput(POINT,i1,i2);
                check = true;
                break;
            }
            case core::topology::POINTSADDED:
            {
                const sofa::helper::vector<unsigned int>& tab= ( static_cast< const PointsAdded *>( *changeIt ) )->pointIndexArray;
				sout << "INPUT ADD POINTS " << tab << sendl;
                for (unsigned int i=0; i<tab.size(); i++)
                {
                    addInputPoint(tab[i], toPointMod);
                }
                check = true;
                break;
            }
            case core::topology::POINTSREMOVED:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRemoved * >( *changeIt ) )->getArray();
				 sout << "INPUT REMOVE POINTS "<<tab << sendl;
                removeInput(POINT, tab );
                check = true;
                nbInputRemoved[POINT] += tab.size();
                break;
            }
            case core::topology::POINTSRENUMBERING:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRenumbering * >( *changeIt ) )->getinv_IndexArray();
				 sout << "INPUT RENUMBER POINTS "<<tab << sendl;
                renumberInput(POINT, tab );
                check = true;
                break;
            }
            case core::topology::EDGESADDED:
            {
                const EdgesAdded *eAdd = static_cast< const EdgesAdded * >( *changeIt );
                const sofa::helper::vector< unsigned int > &tab = eAdd->edgeIndexArray;
//				sout << "INPUT ADD EDGES " << tab << sendl;
                for (unsigned int i=0; i < tab.size(); i++)
                    addInputEdge(tab[i], toPointMod);
                toPointMod->propagateTopologicalChanges();
                if (copyEdges.getValue())
                {
                    if (!toEdgeMod) toModel->getContext()->get(toEdgeMod, sofa::core::objectmodel::BaseContext::Local);
                    if (toEdgeMod)
                    {
                        sout << "EDGESADDED : " << eAdd->getNbAddedEdges() << sendl;
                        const sofa::helper::vector<Edge>& fromArray = eAdd->edgeArray;
                        sofa::helper::vector<Edge> toArray;
                        toArray.resize(fromArray.size());
                        for (unsigned int i=0; i<fromArray.size(); ++i)
                            for (unsigned int j=0; j<fromArray[i].size(); ++j)
                                toArray[i][j] = pointsMappedFrom[POINT][fromArray[i][j]][0];
                        toEdgeMod->addEdgesProcess(toArray);
                        toEdgeMod->addEdgesWarning(eAdd->getNbAddedEdges(), toArray, eAdd->edgeIndexArray, eAdd->ancestorsList, eAdd->coefs);
                        toEdgeMod->propagateTopologicalChanges();
                    }
                }
                check = true;
                break;
            }
            case core::topology::EDGESREMOVED:
            {
                const EdgesRemoved *eRem = static_cast< const EdgesRemoved * >( *changeIt );
                const sofa::helper::vector<unsigned int> &tab = eRem->getArray();
                if (copyEdges.getValue())
                {
                    if (!toEdgeMod) toModel->getContext()->get(toEdgeMod, sofa::core::objectmodel::BaseContext::Local);
                    if (toEdgeMod)
                    {
                        sout << "EDGESREMOVED : " << eRem->getNbRemovedEdges() << sendl;
                        sofa::helper::vector<unsigned int> toArray = tab;
                        toEdgeMod->removeEdgesWarning(toArray);
                        toEdgeMod->propagateTopologicalChanges();
                        toEdgeMod->removeEdgesProcess(tab, false);
                    }
                }
//				sout << "INPUT REMOVE EDGES "<<tab << sendl;
                removeInput(EDGE, tab );
                check = true;
                nbInputRemoved[EDGE] += tab.size();
                break;
            }
            case core::topology::TRIANGLESADDED:
            {
                const TrianglesAdded *tAdd = static_cast< const TrianglesAdded * >( *changeIt );
                const sofa::helper::vector<unsigned int> &tab = tAdd->getArray();
//				sout << "INPUT ADD TRIANGLES " << tab << sendl;
                for (unsigned int i=0; i < tab.size(); i++)
                    addInputTriangle(tab[i], toPointMod);
                toPointMod->propagateTopologicalChanges();
                if (copyTriangles.getValue())
                {
                    if (!toTriangleMod) toModel->getContext()->get(toTriangleMod, sofa::core::objectmodel::BaseContext::Local);
                    if (toTriangleMod)
                    {
                        sout << "TRIANGLESADDED : " << tAdd->getNbAddedTriangles() << sendl;
                        const sofa::helper::vector<Triangle>& fromArray = tAdd->triangleArray;
                        sofa::helper::vector<Triangle> toArray;
                        toArray.resize(fromArray.size());
                        for (unsigned int i=0; i<fromArray.size(); ++i)
                            for (unsigned int j=0; j<fromArray[i].size(); ++j)
                                toArray[i][j] = pointsMappedFrom[POINT][fromArray[i][j]][0];
                        sout << "<IN: " << fromModel->getNbTriangles() << " OUT: " << toModel->getNbTriangles() << sendl;
                        sout << "     ToArray : " << toArray.size() << " : " << toArray << sendl;
                        toTriangleMod->addTrianglesProcess(toArray);
                        sout << "     triangleIndexArray : " << tAdd->triangleIndexArray.size() << " : " << tAdd->triangleIndexArray << sendl;
                        toTriangleMod->addTrianglesWarning(tAdd->getNbAddedTriangles(), toArray, tAdd->triangleIndexArray, tAdd->ancestorsList, tAdd->coefs);
                        toTriangleMod->propagateTopologicalChanges();
                        sout << ">IN: " << fromModel->getNbTriangles() << " OUT: " << toModel->getNbTriangles() << sendl;
                    }
                }
                check = true;
                break;
            }
            case core::topology::TRIANGLESREMOVED:
            {
                const TrianglesRemoved *tRem = static_cast< const TrianglesRemoved * >( *changeIt );
                const sofa::helper::vector<unsigned int> &tab = tRem->getArray();
                if (copyTriangles.getValue())
                {
                    if (!toTriangleMod) toModel->getContext()->get(toTriangleMod, sofa::core::objectmodel::BaseContext::Local);
                    if (toTriangleMod)
                    {
                        sout << "TRIANGLESREMOVED : " << tRem->getNbRemovedTriangles() << " : " << tab << sendl;
                        sofa::helper::vector<unsigned int> toArray = tab;
                        toTriangleMod->removeTrianglesWarning(toArray);
                        toTriangleMod->propagateTopologicalChanges();
                        toTriangleMod->removeTrianglesProcess(tab, false);
                    }
                }
//				sout << "INPUT REMOVE TRIANGLES "<<tab << sendl;
                removeInput(TRIANGLE, tab );
                check = true;
                nbInputRemoved[TRIANGLE] += tab.size();
                break;
            }
            case core::topology::QUADSADDED:
            {
                /// @todo
                break;
            }
            case core::topology::QUADSREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const QuadsRemoved *>( *changeIt ) )->getArray();
//				sout << "INPUT REMOVE QUADS "<<tab << sendl;
                removeInput(QUAD, tab );
                check = true;
                nbInputRemoved[QUAD] += tab.size();
                break;
            }
            case core::topology::TETRAHEDRAADDED:
            {
				const TetrahedraAdded *tAdd = static_cast< const TetrahedraAdded * >( *changeIt );
                const sofa::helper::vector<unsigned int> &tab = tAdd->getArray();
//				sout << "INPUT ADD TETRAHEDRA " << tab << sendl;
                for (unsigned int i=0; i < tab.size(); i++)
                    addInputTetrahedron(tab[i], toPointMod);
                toPointMod->propagateTopologicalChanges();
                if (copyTetrahedra.getValue())
                {
                    if (!toTetrahedronMod) toModel->getContext()->get(toTetrahedronMod, sofa::core::objectmodel::BaseContext::Local);
                    if (toTetrahedronMod)
                    {
                        sout << "TETRAHEDRAADDED : " << tAdd->getNbAddedTetrahedra() << sendl;
                        const sofa::helper::vector<Tetrahedron>& fromArray = tAdd->tetrahedronArray;
                        sofa::helper::vector<Tetrahedron> toArray;
                        toArray.resize(fromArray.size());
                        for (unsigned int i=0; i<fromArray.size(); ++i)
                            for (unsigned int j=0; j<fromArray[i].size(); ++j)
                                toArray[i][j] = pointsMappedFrom[POINT][fromArray[i][j]][0];
                        sout << "<IN: " << fromModel->getNbTetrahedra() << " OUT: " << toModel->getNbTetrahedra() << sendl;
                        sout << "     ToArray : " << toArray.size() << " : " << toArray << sendl;
                        toTetrahedronMod->addTetrahedraProcess(toArray);
                        sout << "     tetrahedronIndexArray : " << tAdd->tetrahedronIndexArray.size() << " : " << tAdd->tetrahedronIndexArray << sendl;
                        toTetrahedronMod->addTetrahedraWarning(tAdd->getNbAddedTetrahedra(), toArray, tAdd->tetrahedronIndexArray, tAdd->ancestorsList, tAdd->coefs);
                        toTetrahedronMod->propagateTopologicalChanges();
                        sout << ">IN: " << fromModel->getNbTetrahedra() << " OUT: " << toModel->getNbTetrahedra() << sendl;
                    }
                }
                check = true;
                break;
            }
            case core::topology::TETRAHEDRAREMOVED:
            {
				const TetrahedraRemoved *tRem = static_cast< const TetrahedraRemoved * >( *changeIt );
                const sofa::helper::vector<unsigned int> &tab = tRem->getArray();
                if (copyTetrahedra.getValue())
                {
                    if (!toTetrahedronMod) toModel->getContext()->get(toTetrahedronMod, sofa::core::objectmodel::BaseContext::Local);
                    if (toTetrahedronMod)
                    {
                        sout << "TETRAHEDRAREMOVED : " << tRem->getNbRemovedTetrahedra() << " : " << tab << sendl;
                        sofa::helper::vector<unsigned int> toArray = tab;
                        toTetrahedronMod->removeTetrahedraWarning(toArray);
                        toTetrahedronMod->propagateTopologicalChanges();
                        toTetrahedronMod->removeTetrahedraProcess(tab, false);
                    }
                }
//				sout << "INPUT REMOVE TETRAHEDRA "<<tab << sendl;
                removeInput(TETRA, tab );
                nbInputRemoved[TETRA] += tab.size();
                check = true;
                break;
            }
            case core::topology::HEXAHEDRAADDED:
            {
                /// @TODO
                break;
            }
            case core::topology::HEXAHEDRAREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const HexahedraRemoved *>( *changeIt ) )->getArray();
//				sout << "INPUT REMOVE HEXAHEDRA "<<tab << sendl;
                removeInput(HEXA, tab );
                check = true;
                nbInputRemoved[TETRA] += tab.size();
                break;
            }
            case core::topology::ENDING_EVENT:
            {
//			    sout << "ENDING EVENT" << sendl;
                pointsToRemove.erase(BaseMeshTopology::InvalidID);
                if (toPointMod != NULL && !pointsToRemove.empty())
                {
                    // TODO: This will fail to work if add and
                    // remove changes are combined and removes are
                    // signaled prior to adds! The indices will mix
                    // up.

                    sofa::helper::vector<unsigned int> vitems;
                    vitems.reserve(pointsToRemove.size());
                    vitems.insert(vitems.end(), pointsToRemove.rbegin(), pointsToRemove.rend());

                    toPointMod->removePointsWarning(vitems);
                    toPointMod->propagateTopologicalChanges();

                    removeOutputPoints(vitems);

                    toPointMod->removePointsProcess(vitems);

                    toPointMod->propagateTopologicalChanges();
                    toPointMod->notifyEndingEvent();
                    toPointMod->propagateTopologicalChanges();

                    pointsToRemove.clear();
                }
                check = true;
                break;
            }

            default:
                sout << "IGNORING " << sofa::core::topology::parseTopologyChangeTypeToString(changeType) << sendl;
                break;

            }
            ++changeIt;
        }
        if (check)
            internalCheck(laststep.c_str(), nbInputRemoved);
    }
}

void Mesh2PointTopologicalMapping::swapInput(Element elem, int i1, int i2)
{
    if (pointsMappedFrom[elem].empty()) return;
    vector<int> i1Map = pointsMappedFrom[elem][i1];
    vector<int> i2Map = pointsMappedFrom[elem][i2];

    pointsMappedFrom[elem][i1] = i2Map;
    for(unsigned int i = 0; i < i2Map.size(); ++i)
    {
        if (i2Map[i] != -1) pointSource[i2Map[i]].second = i1;
    }

    pointsMappedFrom[elem][i2] = i1Map;
    for(unsigned int i = 0; i < i1Map.size(); ++i)
    {
        if (i1Map[i] != -1) pointSource[i1Map[i]].second = i2;
    }
}

void Mesh2PointTopologicalMapping::removeInput(Element elem,  const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFrom[elem].empty()) return;
    unsigned int last = pointsMappedFrom[elem].size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        if (index[i] != last)
        {
            swapInput(elem, index[i], last );
        }
        for (unsigned int j = 0; j < pointsMappedFrom[elem][last].size(); ++j)
        {
            int map = pointsMappedFrom[elem][last][j];
            if (map != -1)
            {
                pointsToRemove.insert(map);
                pointSource[map].second = -1;
            }
        }
        --last;
    }

    pointsMappedFrom[elem].resize( last + 1 );
}

void Mesh2PointTopologicalMapping::renumberInput(Element elem, const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFrom[elem].empty()) return;
    helper::vector< vector<int> > copy = pointsMappedFrom[elem];
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        const vector<int>& map = copy[index[i]];
        pointsMappedFrom[elem][i] = map;
        for (unsigned int j = 0; j < map.size(); ++j)
        {
            int m = map[j];
            if (m != -1)
                pointSource[m].second = i;
        }
    }
}

void Mesh2PointTopologicalMapping::swapOutputPoints(int i1, int i2, bool removeLast)
{
    std::pair<Element, int> i1Source = pointSource[i1];
    std::pair<Element, int> i2Source = pointSource[i2];
    pointSource[i1] = i2Source;
    pointSource[i2] = i1Source;
    if (i1Source.second != -1)
    {
        // replace i1 by i2 in pointsMappedFrom[i1Source.first][i1Source.second]
        vector<int> & pts = pointsMappedFrom[i1Source.first][i1Source.second];
        for (unsigned int j = 0; j < pts.size(); ++j)
        {
            if (pts[j] == i1)
            {
                if (removeLast)
                    pts[j] = -1;
                else
                    pts[j] = i2;
            }
        }
    }
    if (i2Source.second != -1)
    {
        // replace i2 by i1 in pointsMappedFrom[i2Source.first][i1Source.second]
        vector<int> & pts = pointsMappedFrom[i2Source.first][i2Source.second];
        for (unsigned int j = 0; j < pts.size(); ++j)
        {
            if (pts[j] == i2)
                pts[j] = i1;
        }
    }
}

void Mesh2PointTopologicalMapping::removeOutputPoints( const sofa::helper::vector<unsigned int>& index )
{
    unsigned int last = pointSource.size() - 1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapOutputPoints( index[i], last, true );
        --last;
    }

    pointSource.resize(last + 1);
}

} // namespace topology
} // namespace component
} // namespace sofa

