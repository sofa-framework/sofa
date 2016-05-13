/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "STEPExporter.h"

#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>


namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(STEPExporter)

int STEPExporterClass = core::RegisterObject("Read State vectors from file at each timestep")
        .add< STEPExporter >();

STEPExporter::STEPExporter()
    : stepCounter(0)
    , stepFilename( initData(&stepFilename, "filename", "output STEP file name"))
    , m_fileFormat( initData(&m_fileFormat, (bool)false, "shortformat", "if true, save in STEP format with short names to save space"))
    , m_position( initData(&m_position, "position", "points coordinates"))
    , m_triangle( initData(&m_triangle, "triangles", "triangles indices"))
    , m_quad( initData(&m_quad, "quads", "quads indices"))
    , exportEveryNbSteps( initData(&exportEveryNbSteps, (unsigned int)0, "exportEveryNumberOfSteps", "export file only at specified number of steps (0=disable)"))
    , exportAtBegin( initData(&exportAtBegin, (bool)false, "exportAtBegin", "export file at the initialization"))
    , exportAtEnd( initData(&exportAtEnd, (bool)false, "exportAtEnd", "export file when the simulation is finished"))
{
}

STEPExporter::~STEPExporter()
{
    if (outfile)
        delete outfile;
}

void STEPExporter::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    
    context->get(topology, sofa::core::objectmodel::BaseContext::Local);
    context->get(mstate, sofa::core::objectmodel::BaseContext::Local);
    context->get(vmodel, sofa::core::objectmodel::BaseContext::Local);
    
    if(!topology && !vmodel)
    {
        serr << "STEPExporter : error, no topology ." << sendl;
        return;
    }
    
    // Test if the position has not been modified
    if(!m_position.isSet())
    {
        sofa::core::objectmodel::BaseData* pos = NULL;
        sofa::core::objectmodel::BaseData* tri = NULL;
        sofa::core::objectmodel::BaseData* qua = NULL;
        
        if(!pos && vmodel)
            pos = vmodel->findField("position");
        
        if(!pos && mstate)
            pos = mstate->findField("position");
        
        if(!pos && topology)
            pos = topology->findField("position");
        
        if(vmodel)
        {
            tri = vmodel->findField("triangles");
            qua = vmodel->findField("quads");
        }
        
        if(!tri && !qua && topology)
        {
            tri = topology->findField("triangles");
            qua = topology->findField("quads");
        }
        
        if(!tri)
        {
            if(!qua)
            {
                serr << "STEPExporter : error, neither triangles nor quads" << sendl;
                return;
            }
            m_quad.setParent(qua);
        }
        else
        {
            m_triangle.setParent(tri);
        }
        
        if(pos)
        {
            m_position.setParent(pos);
        }
        else
        {
            serr << "STEPExporter : error, no positions" << sendl;
            return;
        }
    }

    // Activate the listening to the event in order to be able to export file at the nth-step
    if(exportEveryNbSteps.getValue() != 0)
        this->f_listening.setValue(true);
    
    nbFiles = 0;

}

void STEPExporter::writeSTEP()
{
    std::string filename = stepFilename.getFullPath();
    if (maxStep)
    {
        std::ostringstream oss;
        oss.width(5);
        oss.fill('0');
        oss << nbFiles;
        filename += oss.str();
    }
    filename += ".stp";

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file " << filename << sendl;
        delete outfile;
        outfile = NULL;
        return;
    }
    
    helper::ReadAccessor< Data< vector< core::topology::Triangle > > > triangleIndices = m_triangle;
    helper::ReadAccessor< Data< vector< core::topology::Quad > > > quadIndices = m_quad;
    helper::ReadAccessor< Data< defaulttype::Vec3Types::VecCoord > > positionIndices = m_position;
    
    outfile->precision(7);
    /**********/
    
    /* KEYWORD */
    *outfile << "ISO-10303-21;" << std::endl;
    
    // Get the file's creation date -- ISO 8601
    time_t rawtime;
    struct tm * timeinfo;
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    char date [80];
    strftime (date, sizeof(date), "%Y-%m-%dT%T%z", timeinfo);
    
    /* HEADER */
    *outfile << "HEADER;" << std::endl;
    *outfile << "FILE_DESCRIPTION(('Sofa STEP Exporter'),'2;1');" << std::endl;
    *outfile << std::endl;
    *outfile << "FILE_NAME(" << std::endl;
    *outfile << "/* name */ '" << filename << "'," << std::endl; 
    *outfile << "/* time_stamp */ '" << date << "'," << std::endl; 
    *outfile << "/* author */ ('" << "unknown" << "')," << std::endl; 
    *outfile << "/*organization*/ ('" << "unknown" << "')," << std::endl; 
    *outfile << "/* preprocessor_version */ '" << "Sofa version 1.0 RC 1" << "'," << std::endl; 
    *outfile << "/* originating_system */ '" << "unknown" << "'," << std::endl; 
    *outfile << "/* authorization */ '" << "unknown" << "');"  << std::endl;
    
    *outfile << std::endl;
    // CONFIG_CONTROL_DESIGN specify the schema for AP203
    // see : http://www.steptools.com/support/stdev_docs/express/ap203/html/schema.html
    *outfile << "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));" << std::endl;
    *outfile << std::endl;
    *outfile << "ENDSEC;" << std::endl;
    
    /* DATA */
    *outfile << "DATA;" << std::endl;
    
    unsigned int i=1;
    char year [80], month [80], day [80], hours [80], minutes [80], seconds [80], timezone [80];
    strftime (year, sizeof(year), "%Y", timeinfo);
    strftime (month, sizeof(month), "%m", timeinfo);
    strftime (day, sizeof(day), "%d"/*"%e"*/, timeinfo);
    strftime (hours, sizeof(hours), "%H", timeinfo);
    strftime (minutes, sizeof(minutes), "%M", timeinfo);
    strftime (seconds, sizeof(seconds), "%S", timeinfo);
    strftime (timezone, sizeof(timezone), "%z", timeinfo);
    int hoursOffset = std::atoi(timezone)/100;
    int minutesOffset = std::atoi(timezone)%100;
    unsigned int appCon = 0, secClas = 0, designCon = 0, mechCon = 0, persOrg = 0, dateTime = 0, prodDef = 0, prodDefSS = 0, prod = 0, geoCon = 0, defShape = 0;
    
    // The following lines are mandatory for the AP203 protocol
    // see : http://www.steptools.com/support/stdev_docs/express/ap203/recprac203v8.pdf
    
    /* Context and definition */
    *outfile << '#' << i << "=APPLICATION_CONTEXT('configuration controlled 3D design of mechanical parts and assemblies');" << std::endl;appCon = i;i++;
    *outfile << '#' << i << "=DESIGN_CONTEXT(' ',#" << appCon << ",'design');" << std::endl;designCon = i;i++;
    *outfile << '#' << i << "=MECHANICAL_CONTEXT(' ',#" << appCon << ",'mechanical');" << std::endl;mechCon = i;i++;
    *outfile << '#' << i << "=APPLICATION_PROTOCOL_DEFINITION('international standard','config_control_design',1994,#" << appCon << ");" << std::endl;i++;

    /* Product */
    *outfile << '#' << i << "=PRODUCT('1','object','converted by STEPExporter',(#" << mechCon << "));" << std::endl;prod = i;i++;
    *outfile << '#' << i << "=PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('A',' ',#" << i-1 << ",.NOT_KNOWN.);" << std::endl;prodDefSS = i;i++;  
    *outfile << '#' << i << "=PRODUCT_CATEGORY('part',$) ;" << std::endl;i++;
    *outfile << '#' << i << "=PRODUCT_RELATED_PRODUCT_CATEGORY('detail',$,(#" << i-3 << ")) ;" << std::endl;i++;
    *outfile << '#' << i << "=PRODUCT_CATEGORY_RELATIONSHIP(' ',' ',#" << i-2 << ",#" << i-1 << ") ;" << std::endl;i++;
    *outfile << '#' << i << "=PRODUCT_DEFINITION(' ',' ',#" << i-4 << ",#" << designCon << ");" << std::endl;prodDef = i;i++;
    *outfile << '#' << i << "=PRODUCT_DEFINITION_SHAPE(' ',' ',#" << i-1 << ") ;" << std::endl;defShape = i;i++;
    
    /* Security */
    *outfile << '#' << i << "=SECURITY_CLASSIFICATION_LEVEL('unclassified') ;" << std::endl;i++;
    *outfile << '#' << i << "=SECURITY_CLASSIFICATION(' ',' ',#" << i-1 << ") ;" << std::endl;secClas = i;i++;
    *outfile << '#' << i << "=CC_DESIGN_SECURITY_CLASSIFICATION(#" << i-1 << ",(#" << prodDef << "));" << std::endl;i++;
    
    /* Date and time */
    *outfile << '#' << i << "=DATE_TIME_ROLE('classification_date');" << std::endl;i++;
    *outfile << '#' << i << "=DATE_TIME_ROLE('creation_date') ;" << std::endl;i++;
    *outfile << '#' << i << "=CALENDAR_DATE(" << year << "," << day << "," << month << ");" << std::endl;i++;
    *outfile << '#' << i << "=LOCAL_TIME(" << hours << "," << minutes << "," << seconds << ".,#" << i+1 << ");" << std::endl;i++;
    *outfile << '#' << i << "=DATE_AND_TIME(#" << i-2 << ",#" << i-1 << ");" << std::endl;dateTime = i;i++;
    *outfile << '#' << i << "=COORDINATED_UNIVERSAL_TIME_OFFSET(" << hoursOffset << "," << minutesOffset << ",.AHEAD.);" << std::endl;i++;
    *outfile << '#' << i << "=CC_DESIGN_DATE_AND_TIME_ASSIGNMENT(#" << i-2 << ",#" << i-6 << ",(#" << secClas << "));" << std::endl;i++;
    *outfile << '#' << i << "=CC_DESIGN_DATE_AND_TIME_ASSIGNMENT(#" << i-3 << ",#" << i-6 << ",(#" << prodDef << "));" << std::endl;i++;
    
    /* Person and organisation */
    *outfile << '#' << i << "=PERSON_AND_ORGANIZATION_ROLE('design_owner');" << std::endl;i++;
    *outfile << '#' << i << "=PERSON_AND_ORGANIZATION_ROLE('design_supplier');" << std::endl;i++;
    *outfile << '#' << i << "=PERSON_AND_ORGANIZATION_ROLE('classification_officer');" << std::endl;i++;
    *outfile << '#' << i << "=PERSON_AND_ORGANIZATION_ROLE('creator');" << std::endl;i++;
    *outfile << '#' << i << "=PERSON(' ',' ',' ',$,$,$);" << std::endl;i++;
    *outfile << '#' << i << "=ORGANIZATION(' ',' ',' ');" << std::endl;i++;
    *outfile << '#' << i << "=PERSONAL_ADDRESS(' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',(#" << i-2 << "),' ');" << std::endl;i++;
    *outfile << '#' << i << "=PERSON_AND_ORGANIZATION(#" << i-3 << ",#" << i-2 << ") ;" << std::endl;persOrg = i;i++;
    *outfile << '#' << i << "=CC_DESIGN_PERSON_AND_ORGANIZATION_ASSIGNMENT(#" << i-1 << ",#" << i-8 << ",(#" << prod << "));" << std::endl;i++;
    *outfile << '#' << i << "=CC_DESIGN_PERSON_AND_ORGANIZATION_ASSIGNMENT(#" << i-2 << ",#" << i-8 << ",(#" << prodDefSS << "));" << std::endl;i++;
    *outfile << '#' << i << "=CC_DESIGN_PERSON_AND_ORGANIZATION_ASSIGNMENT(#" << i-3 << ",#" << i-8 << ",(#" << secClas << "));" << std::endl;i++;
    *outfile << '#' << i << "=CC_DESIGN_PERSON_AND_ORGANIZATION_ASSIGNMENT(#" << i-4 << ",#" << i-8 << ",(#" << prodDefSS << ",#" << prodDef << "));" << std::endl;i++;
    
    /* Approval */
    *outfile << '#' << i << "=APPROVAL_ROLE('APPROVER');" << std::endl;i++;
    *outfile << '#' << i << "=APPROVAL_STATUS('not_yet_approved');" << std::endl;i++;
    *outfile << '#' << i << "=APPROVAL(#" << i-1 << ",' ');" << std::endl;i++;
    *outfile << '#' << i << "=APPROVAL_PERSON_ORGANIZATION(#" << persOrg << ",#" << i-1 << ",#" << i-3 << ");" << std::endl;i++;
    *outfile << '#' << i << "=APPROVAL_DATE_TIME(#" << dateTime << ",#" << i-2 << ");" << std::endl;i++;
    *outfile << '#' << i << "=CC_DESIGN_APPROVAL(#" << i-3 << ",(#" << secClas << ",#" << prodDefSS << ",#" << prodDef << ")) ;" << std::endl;i++;
    
    /* Measure units */
    *outfile << '#' << i << "=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));" << std::endl;i++;
    *outfile << '#' << i << "=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));" << std::endl;i++;
    *outfile << '#' << i << "=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());" << std::endl;i++;
    *outfile << '#' << i << "=PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(0.0174532925199),#" << i-2 << ");" << std::endl;i++;
    *outfile << '#' << i << "=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.005),#" << i-4 << ",'TOL_CRV','CONFUSED CURVE UNCERTAINTY');" << std::endl;i++;
    *outfile << '#' << i << "=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#" << i-1 << "))GLOBAL_UNIT_ASSIGNED_CONTEXT((#" << i-5 << ",#" << i-4 << ",#" << i-3 << "))REPRESENTATION_CONTEXT(' ',' ')) ;" << std::endl;geoCon = i;i++;
    
    vector<unsigned int> facetIndices;
    
    if(!triangleIndices.empty())
    {
        for(unsigned int j=0;j<triangleIndices.size();j++)
        {
            for(int k=0;k<3;k++)
            {
                /* points */
                double pointA[3], pointB[3];
                if(k==2)
                {
                    pointB[0] = positionIndices[triangleIndices[j][0]][0];
                    pointB[1] = positionIndices[triangleIndices[j][0]][1];
                    pointB[2] = positionIndices[triangleIndices[j][0]][2];
                }
                else
                {
                    pointB[0] = positionIndices[triangleIndices[j][k+1]][0];
                    pointB[1] = positionIndices[triangleIndices[j][k+1]][1];
                    pointB[2] = positionIndices[triangleIndices[j][k+1]][2];
                }
                pointA[0] = positionIndices[triangleIndices[j][k]][0];
                pointA[1] = positionIndices[triangleIndices[j][k]][1];
                pointA[2] = positionIndices[triangleIndices[j][k]][2];
                *outfile << '#' << i << "=DIRECTION('direction',(";
                *outfile << pointB[0]-pointA[0] << ',';
                *outfile << pointB[1]-pointA[1] << ',';
                *outfile << pointB[2]-pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VECTOR('vector',#" << i-1 << ",1.);" << std::endl;i++;
                *outfile << '#' << i << "=CARTESIAN_POINT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=LINE('line',#" << i-1 << ",#" << i-2 << ");" << std::endl;i++;
                *outfile << '#' << i << "=CARTESIAN_POINT('cartesian point',(" << std::fixed << pointB[0] << "," << std::fixed << pointB[1] << "," << std::fixed << pointB[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VERTEX_POINT('vertex',#" << i-1 << ");" << std::endl;i++;
                *outfile << '#' << i << "=CARTESIAN_POINT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VERTEX_POINT('vertex',#" << i-1 << ");" << std::endl;i++;
                
                /* Edge */
                *outfile << '#' << i << "=EDGE_CURVE('edge',#" << i-3 << ",#" << i-1 << ",#" << i-5 << ",.T.);" << std::endl;i++;
                *outfile << '#' << i << "=ORIENTED_EDGE('oriented edge',*,*,#" << i-1 << ",.T.);" << std::endl;i++;
            }
        
        /* Wireframe facet */
        *outfile << '#' << i << "=EDGE_LOOP('edge loop',(#" << i-21 << ",#" << i-11 << ",#" << i-1 << "));" << std::endl;i++;
        *outfile << '#' << i << "=FACE_OUTER_BOUND('outer bound',#" << i-1 << ",.F.);" << std::endl;i++;
        
        /* Facet */
        double pointA[3], pointB[3], pointC[3];
        pointA[0] = positionIndices[triangleIndices[j][0]][0];
        pointA[1] = positionIndices[triangleIndices[j][0]][1];
        pointA[2] = positionIndices[triangleIndices[j][0]][2];
        pointB[0] = positionIndices[triangleIndices[j][1]][0];
        pointB[1] = positionIndices[triangleIndices[j][1]][1];
        pointB[2] = positionIndices[triangleIndices[j][1]][2];
        pointC[0] = positionIndices[triangleIndices[j][2]][0];
        pointC[1] = positionIndices[triangleIndices[j][2]][1];
        pointC[2] = positionIndices[triangleIndices[j][2]][2];
        *outfile << '#' << i << "=CARTESIAN_POINT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
        *outfile << '#' << i << "=DIRECTION('direction',(" << std::fixed << pointB[0]-pointA[0] << "," << std::fixed << pointB[1]-pointA[1] << "," << std::fixed << pointB[2]-pointA[2] << "));" << std::endl;i++;
        *outfile << '#' << i << "=DIRECTION('direction',(" << std::fixed << pointC[0]-pointA[0] << "," << std::fixed << pointC[1]-pointA[1] << "," << std::fixed << pointB[2]-pointA[2] << "));" << std::endl;i++;
        *outfile << '#' << i << "=AXIS2_PLACEMENT_3D('axis placement',#" << i-3 << ",#" << i-2 << ",#" << i-1 << ");" << std::endl;i++;
        *outfile << '#' << i << "=PLANE('plane',#" << i-1 << ");" << std::endl;i++;
        *outfile << '#' << i << "=ADVANCED_FACE('Advanced Face',(#" << i-6 << "),#" << i-1 << ",.T.);" << std::endl;
        facetIndices.push_back(i);i++;
        }
    }
    
    if(!quadIndices.empty())
    {
        for(unsigned int j=0;j<quadIndices.size();j++)
        {
            for(int k=0;k<4;k++)
            {
                /* points */
                double pointA[3], pointB[3];
                if(k==3)
                {
                    pointB[0] = positionIndices[quadIndices[j][0]][0];
                    pointB[1] = positionIndices[quadIndices[j][0]][1];
                    pointB[2] = positionIndices[quadIndices[j][0]][2];
                }
                else
                {
                    pointB[0] = positionIndices[quadIndices[j][k+1]][0];
                    pointB[1] = positionIndices[quadIndices[j][k+1]][1];
                    pointB[2] = positionIndices[quadIndices[j][k+1]][2];
                }
                pointA[0] = positionIndices[quadIndices[j][k]][0];
                pointA[1] = positionIndices[quadIndices[j][k]][1];
                pointA[2] = positionIndices[quadIndices[j][k]][2];
                
                *outfile << '#' << i << "=DIRECTION('direction',(";
                *outfile << pointB[0]-pointA[0] << ',';
                *outfile << pointB[1]-pointA[1] << ',';
                *outfile << pointB[2]-pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VECTOR('vector',#" << i-1 << ",1.);" << std::endl;i++;
                *outfile << '#' << i << "=CARTESIAN_POINT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=LINE('line',#" << i-1 << ",#" << i-2 << ");" << std::endl;i++;
                *outfile << '#' << i << "=CARTESIAN_POINT('cartesian point',(" << std::fixed << pointB[0] << "," << std::fixed << pointB[1] << "," << std::fixed << pointB[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VERTEX_POINT('vertex',#" << i-1 << ");" << std::endl;i++;
                *outfile << '#' << i << "=CARTESIAN_POINT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VERTEX_POINT('vertex',#" << i-1 << ");" << std::endl;i++;
                
                /* Edge */
                *outfile << '#' << i << "=EDGE_CURVE('edge',#" << i-3 << ",#" << i-1 << ",#" << i-5 << ",.T.);" << std::endl;i++;
                *outfile << '#' << i << "=ORIENTED_EDGE('oriented edge',*,*,#" << i-1 << ",.T.);" << std::endl;i++;
            }
            
            /* Wireframe facet */
            *outfile << '#' << i << "=EDGE_LOOP('edge loop',(#" << i-31 << ",#" << i-21 << ",#" << i-11 << ",#" << i-1 << "));" << std::endl;i++;
            *outfile << '#' << i << "=FACE_OUTER_BOUND('outer bound',#" << i-1 << ",.F.);" << std::endl;i++;
            
            /* Facet */
            double pointA[3], pointB[3], pointC[3];
            pointA[0] = positionIndices[quadIndices[j][0]][0];
            pointA[1] = positionIndices[quadIndices[j][0]][1];
            pointA[2] = positionIndices[quadIndices[j][0]][2];
            pointB[0] = positionIndices[quadIndices[j][1]][0];
            pointB[1] = positionIndices[quadIndices[j][1]][1];
            pointB[2] = positionIndices[quadIndices[j][1]][2];
            pointC[0] = positionIndices[quadIndices[j][2]][0];
            pointC[1] = positionIndices[quadIndices[j][2]][1];
            pointC[2] = positionIndices[quadIndices[j][2]][2];
            *outfile << '#' << i << "=CARTESIAN_POINT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
            *outfile << '#' << i << "=DIRECTION('direction',(" << std::fixed << pointB[0]-pointA[0] << "," << std::fixed << pointB[1]-pointA[1] << "," << std::fixed << pointB[2]-pointA[2] << "));" << std::endl;i++;
            *outfile << '#' << i << "=DIRECTION('direction',(" << std::fixed << pointC[0]-pointA[0] << "," << std::fixed << pointC[1]-pointA[1] << "," << std::fixed << pointB[2]-pointA[2] << "));" << std::endl;i++;
            *outfile << '#' << i << "=AXIS2_PLACEMENT_3D('axis placement',#" << i-3 << ",#" << i-2 << ",#" << i-1 << ");" << std::endl;i++;
            *outfile << '#' << i << "=PLANE('plane',#" << i-1 << ");" << std::endl;i++;
            *outfile << '#' << i << "=ADVANCED_FACE('Advanced Face',(#" << i-6 << "),#" << i-1 << ",.T.);" << std::endl;
            facetIndices.push_back(i);i++;
        }
    }
    
    /* Shape */
    *outfile << '#' << i << "=CLOSED_SHELL('Closed Shell',(";
    for(unsigned int i=0;i<facetIndices.size()-1;i++)
    {
        *outfile << "#" << facetIndices[i] << ',';
    }
    *outfile << facetIndices[facetIndices.size()] << "));" << std::endl;i++;
    *outfile << '#' << i << "=MANIFOLD_SOLID_BREP('Manifold Brep',#" << i-1 << ");" << std::endl;i++;
    *outfile << '#' << i << "=ADVANCED_BREP_SHAPE_REPRESENTATION('adv brep shape',(#" << i-1 << "),#" << geoCon << ");" << std::endl;i++;
    *outfile << '#' << i << "=SHAPE_DEFINITION_REPRESENTATION(#" << defShape << ",#" << i-1 << ");" << std::endl;i++;

    
    *outfile << std::endl;
    *outfile << "ENDSEC;" << std::endl;
    
    /* END KEYWORD */
    *outfile << "END-ISO-10303-21;" << std::endl;
    
    /**********/
    
    outfile->close();
    sout << filename << " written" << sendl;
    nbFiles++;
}

void STEPExporter::writeSTEPShort()
{
    std::string filename = stepFilename.getFullPath();
    if (maxStep)
    {
        std::ostringstream oss;
        oss.width(5);
        oss.fill('0');
        oss << nbFiles;
        filename += oss.str();
    }
    filename += ".stp";
    
    outfile = new std::ofstream(filename.c_str(), std::ios::out | std::ios::binary);
    if( !outfile->is_open() )
    {
        serr << "Error creating file " << filename << sendl;
        delete outfile;
        outfile = NULL;
        return;
    }
    
    
    helper::ReadAccessor< Data< vector< core::topology::Triangle > > > triangleIndices = m_triangle;
    helper::ReadAccessor< Data< vector< core::topology::Quad > > > quadIndices = m_quad;
    helper::ReadAccessor< Data< defaulttype::Vec3Types::VecCoord > > positionIndices = m_position;
    
    outfile->precision(7);
    /**********/
    
    /* KEYWORD */
    *outfile << "ISO-10303-21;" << std::endl;
    
    // Get the file's creation date -- ISO 8601
    time_t rawtime;
    struct tm * timeinfo;
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    char date [80];
    strftime (date, sizeof(date), "%Y-%m-%dT%T%z", timeinfo);
    
    /* HEADER */
    *outfile << "HEADER;" << std::endl;
    *outfile << "FILE_DESCRIPTION(('Sofa STEP Exporter'),'2;1');" << std::endl;
    *outfile << std::endl;
    *outfile << "FILE_NAME(" << std::endl;
    *outfile << "/* name */ '" << filename << "'," << std::endl; 
    *outfile << "/* time_stamp */ '" << date << "'," << std::endl; 
    *outfile << "/* author */ ('" << "unknown" << "')," << std::endl; 
    *outfile << "/*organization*/ ('" << "unknown" << "')," << std::endl; 
    *outfile << "/* preprocessor_version */ '" << "Sofa version 1.0 RC 1" << "'," << std::endl; 
    *outfile << "/* originating_system */ '" << "unknown" << "'," << std::endl; 
    *outfile << "/* authorization */ '" << "unknown" << "');"  << std::endl;
    
    *outfile << std::endl;
    // CONFIG_CONTROL_DESIGN specify the schema for AP203
    // see : http://www.steptools.com/support/stdev_docs/express/ap203/html/schema.html
    *outfile << "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));" << std::endl;
    *outfile << std::endl;
    *outfile << "ENDSEC;" << std::endl;
    
    /* DATA */
    *outfile << "DATA;" << std::endl;
    
    unsigned int i=1;
    char year [80], month [80], day [80], hours [80], minutes [80], seconds [80], timezone [80];
    strftime (year, sizeof(year), "%Y", timeinfo);
    strftime (month, sizeof(month), "%m", timeinfo);
    strftime (day, sizeof(day), "%d"/*"%e"*/, timeinfo);
    strftime (hours, sizeof(hours), "%H", timeinfo);
    strftime (minutes, sizeof(minutes), "%M", timeinfo);
    strftime (seconds, sizeof(seconds), "%S", timeinfo);
    strftime (timezone, sizeof(timezone), "%z", timeinfo);
    int hoursOffset = std::atoi(timezone)/100;
    int minutesOffset = std::atoi(timezone)%100;
    unsigned int appCon = 0, secClas = 0, designCon = 0, mechCon = 0, persOrg = 0, dateTime = 0, prodDef = 0, prodDefSS = 0, prod = 0, geoCon = 0, defShape = 0;
    
    // The following lines are mandatory for the AP203 protocol
    // see : http://www.steptools.com/support/stdev_docs/express/ap203/recprac203v8.pdf
    
    /* Context and definition */
    *outfile << '#' << i << "=APPCNT('configuration controlled 3D design of mechanical parts and assemblies');" << std::endl;appCon = i;i++;
    *outfile << '#' << i << "=DSGCNT(' ',#" << appCon << ",'design');" << std::endl;designCon = i;i++;
    *outfile << '#' << i << "=MCHCNT(' ',#" << appCon << ",'mechanical');" << std::endl;mechCon = i;i++;
    *outfile << '#' << i << "=APPRDF('international standard','config_control_design',1994,#" << appCon << ");" << std::endl;i++;
    
    /* Product */
    *outfile << '#' << i << "=PRDCT('1','object','converted by STEPExporter',(#" << mechCon << "));" << std::endl;prod = i;i++;
    *outfile << '#' << i << "=PDFWSS('A',' ',#" << i-1 << ",.NOT_KNOWN.);" << std::endl;prodDefSS = i;i++;  
    *outfile << '#' << i << "=PRDCTG('part',$) ;" << std::endl;i++;
    *outfile << '#' << i << "=PRPC('detail',$,(#" << i-3 << ")) ;" << std::endl;i++;
    *outfile << '#' << i << "=PRCTRL(' ',' ',#" << i-2 << ",#" << i-1 << ") ;" << std::endl;i++;
    *outfile << '#' << i << "=PRDDFN(' ',' ',#" << i-4 << ",#" << designCon << ");" << std::endl;prodDef = i;i++;
    *outfile << '#' << i << "=PRDFSH(' ',' ',#" << i-1 << ") ;" << std::endl;defShape = i;i++;
    
    /* Security */
    *outfile << '#' << i << "=SCCLLV('unclassified') ;" << std::endl;i++;
    *outfile << '#' << i << "=SCRCLS(' ',' ',#" << i-1 << ") ;" << std::endl;secClas = i;i++;
    *outfile << '#' << i << "=CDSC(#" << i-1 << ",(#" << prodDef << "));" << std::endl;i++;
    
    /* Date and time */
    *outfile << '#' << i << "=DTTMRL('classification_date');" << std::endl;i++;
    *outfile << '#' << i << "=DTTMRL('creation_date') ;" << std::endl;i++;
    *outfile << '#' << i << "=CLNDT(" << year << "," << day << "," << month << ");" << std::endl;i++;
    *outfile << '#' << i << "=LCLTM(" << hours << "," << minutes << "," << seconds << ".,#" << i+1 << ");" << std::endl;i++;
    *outfile << '#' << i << "=DTANTM(#" << i-2 << ",#" << i-1 << ");" << std::endl;dateTime = i;i++;
    *outfile << '#' << i << "=CUTO(" << hoursOffset << "," << minutesOffset << ",.AHEAD.);" << std::endl;i++;
    *outfile << '#' << i << "=CDDATA(#" << i-2 << ",#" << i-6 << ",(#" << secClas << "));" << std::endl;i++;
    *outfile << '#' << i << "=CDDATA(#" << i-3 << ",#" << i-6 << ",(#" << prodDef << "));" << std::endl;i++;
    
    /* Person and organisation */
    *outfile << '#' << i << "=PAOR('design_owner');" << std::endl;i++;
    *outfile << '#' << i << "=PAOR('design_supplier');" << std::endl;i++;
    *outfile << '#' << i << "=PAOR('classification_officer');" << std::endl;i++;
    *outfile << '#' << i << "=PAOR('creator');" << std::endl;i++;
    *outfile << '#' << i << "=PERSON(' ',' ',' ',$,$,$);" << std::endl;i++;
    *outfile << '#' << i << "=ORGNZT(' ',' ',' ');" << std::endl;i++;
    *outfile << '#' << i << "=PRSADD(' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',(#" << i-2 << "),' ');" << std::endl;i++;
    *outfile << '#' << i << "=PRANOR(#" << i-3 << ",#" << i-2 << ") ;" << std::endl;persOrg = i;i++;
    *outfile << '#' << i << "=CDPAOA(#" << i-1 << ",#" << i-8 << ",(#" << prod << "));" << std::endl;i++;
    *outfile << '#' << i << "=CDPAOA(#" << i-2 << ",#" << i-8 << ",(#" << prodDefSS << "));" << std::endl;i++;
    *outfile << '#' << i << "=CDPAOA(#" << i-3 << ",#" << i-8 << ",(#" << secClas << "));" << std::endl;i++;
    *outfile << '#' << i << "=CDPAOA(#" << i-4 << ",#" << i-8 << ",(#" << prodDefSS << ",#" << prodDef << "));" << std::endl;i++;
    
    /* Approval */
    *outfile << '#' << i << "=APPRL('APPROVER');" << std::endl;i++;
    *outfile << '#' << i << "=APPSTT('not_yet_approved');" << std::endl;i++;
    *outfile << '#' << i << "=APPRVL(#" << i-1 << ",' ');" << std::endl;i++;
    *outfile << '#' << i << "=APPROR(#" << persOrg << ",#" << i-1 << ",#" << i-3 << ");" << std::endl;i++;
    *outfile << '#' << i << "=APDTTM(#" << dateTime << ",#" << i-2 << ");" << std::endl;i++;
    *outfile << '#' << i << "=CCDSAP(#" << i-3 << ",(#" << secClas << ",#" << prodDefSS << ",#" << prodDef << ")) ;" << std::endl;i++;
    
    /* Measure units */
    *outfile << '#' << i << "=(LNGUNT()NMDUNT(*)SUNT(.MILLI.,.METRE.));" << std::endl;i++;
    *outfile << '#' << i << "=(NMDUNT(*)PLANUN()SUNT($,.RADIAN.));" << std::endl;i++;
    *outfile << '#' << i << "=(NMDUNT(*)SUNT($,.STERADIAN.)SLANUN());" << std::endl;i++;
    *outfile << '#' << i << "=PAMWU(PLANE_ANGLE_MEASURE(0.0174532925199),#" << i-2 << ");" << std::endl;i++;
    *outfile << '#' << i << "=UMWU(LENGTH_MEASURE(0.005),#" << i-4 << ",'TOL_CRV','CONFUSED CURVE UNCERTAINTY');" << std::endl;i++;
    *outfile << '#' << i << "=(GMRPCN(3)GC((#" << i-1 << "))GUAC((#" << i-5 << ",#" << i-4 << ",#" << i-3 << "))RPRCNT(' ',' ')) ;" << std::endl;geoCon = i;i++;
    
    vector<unsigned int> facetIndices;
    
    if(!triangleIndices.empty())
    {
        for(unsigned int j=0;j<triangleIndices.size();j++)
        {
            for(int k=0;k<3;k++)
            {
                /* points */
                double pointA[3], pointB[3];
                if(k==2)
                {
                    pointB[0] = positionIndices[triangleIndices[j][0]][0];
                    pointB[1] = positionIndices[triangleIndices[j][0]][1];
                    pointB[2] = positionIndices[triangleIndices[j][0]][2];
                }
                else
                {
                    pointB[0] = positionIndices[triangleIndices[j][k+1]][0];
                    pointB[1] = positionIndices[triangleIndices[j][k+1]][1];
                    pointB[2] = positionIndices[triangleIndices[j][k+1]][2];
                }
                pointA[0] = positionIndices[triangleIndices[j][k]][0];
                pointA[1] = positionIndices[triangleIndices[j][k]][1];
                pointA[2] = positionIndices[triangleIndices[j][k]][2];
                *outfile << '#' << i << "=DRCTN('direction',(";
                *outfile << pointB[0]-pointA[0] << ',';
                *outfile << pointB[1]-pointA[1] << ',';
                *outfile << pointB[2]-pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VECTOR('vector',#" << i-1 << ",1.);" << std::endl;i++;
                *outfile << '#' << i << "=CRTPNT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=LINE('line',#" << i-1 << ",#" << i-2 << ");" << std::endl;i++;
                *outfile << '#' << i << "=CRTPNT('cartesian point',(" << std::fixed << pointB[0] << "," << std::fixed << pointB[1] << "," << std::fixed << pointB[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VRTPNT('vertex',#" << i-1 << ");" << std::endl;i++;
                *outfile << '#' << i << "=CRTPNT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VRTPNT('vertex',#" << i-1 << ");" << std::endl;i++;
                
                /* Edge */
                *outfile << '#' << i << "=EDGCRV('edge',#" << i-3 << ",#" << i-1 << ",#" << i-5 << ",.T.);" << std::endl;i++;
                *outfile << '#' << i << "=ORNEDG('oriented edge',*,*,#" << i-1 << ",.T.);" << std::endl;i++;
            }
            
            /* Wireframe facet */
            *outfile << '#' << i << "=EDGLP('edge loop',(#" << i-21 << ",#" << i-11 << ",#" << i-1 << "));" << std::endl;i++;
            *outfile << '#' << i << "=FCOTBN('outer bound',#" << i-1 << ",.F.);" << std::endl;i++;
            
            /* Facet */
            double pointA[3], pointB[3], pointC[3];
            pointA[0] = positionIndices[triangleIndices[j][0]][0];
            pointA[1] = positionIndices[triangleIndices[j][0]][1];
            pointA[2] = positionIndices[triangleIndices[j][0]][2];
            pointB[0] = positionIndices[triangleIndices[j][1]][0];
            pointB[1] = positionIndices[triangleIndices[j][1]][1];
            pointB[2] = positionIndices[triangleIndices[j][1]][2];
            pointC[0] = positionIndices[triangleIndices[j][2]][0];
            pointC[1] = positionIndices[triangleIndices[j][2]][1];
            pointC[2] = positionIndices[triangleIndices[j][2]][2];
            *outfile << '#' << i << "=CRTPNT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
            *outfile << '#' << i << "=DRCTN('direction',(" << std::fixed << pointB[0]-pointA[0] << "," << std::fixed << pointB[1]-pointA[1] << "," << std::fixed << pointB[2]-pointA[2] << "));" << std::endl;i++;
            *outfile << '#' << i << "=DRCTN('direction',(" << std::fixed << pointC[0]-pointA[0] << "," << std::fixed << pointC[1]-pointA[1] << "," << std::fixed << pointB[2]-pointA[2] << "));" << std::endl;i++;
            *outfile << '#' << i << "=A2PL3D('axis placement',#" << i-3 << ",#" << i-2 << ",#" << i-1 << ");" << std::endl;i++;
            *outfile << '#' << i << "=PLANE('plane',#" << i-1 << ");" << std::endl;i++;
            *outfile << '#' << i << "=ADVFC('Advanced Face',(#" << i-6 << "),#" << i-1 << ",.T.);" << std::endl;
            facetIndices.push_back(i);i++;
        }
    }
    
    if(!quadIndices.empty())
    {
        for(unsigned int j=0;j<quadIndices.size();j++)
        {
            for(int k=0;k<4;k++)
            {
                /* points */
                double pointA[3], pointB[3];
                if(k==3)
                {
                    pointB[0] = positionIndices[quadIndices[j][0]][0];
                    pointB[1] = positionIndices[quadIndices[j][0]][1];
                    pointB[2] = positionIndices[quadIndices[j][0]][2];
                }
                else
                {
                    pointB[0] = positionIndices[quadIndices[j][k+1]][0];
                    pointB[1] = positionIndices[quadIndices[j][k+1]][1];
                    pointB[2] = positionIndices[quadIndices[j][k+1]][2];
                }
                pointA[0] = positionIndices[quadIndices[j][k]][0];
                pointA[1] = positionIndices[quadIndices[j][k]][1];
                pointA[2] = positionIndices[quadIndices[j][k]][2];
                
                *outfile << '#' << i << "=DRCTN('direction',(";
                *outfile << pointB[0]-pointA[0] << ',';
                *outfile << pointB[1]-pointA[1] << ',';
                *outfile << pointB[2]-pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VECTOR('vector',#" << i-1 << ",1.);" << std::endl;i++;
                *outfile << '#' << i << "=CRTPNT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=LINE('line',#" << i-1 << ",#" << i-2 << ");" << std::endl;i++;
                *outfile << '#' << i << "=CRTPNT('cartesian point',(" << std::fixed << pointB[0] << "," << std::fixed << pointB[1] << "," << std::fixed << pointB[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VRTPNT('vertex',#" << i-1 << ");" << std::endl;i++;
                *outfile << '#' << i << "=CRTPNT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
                *outfile << '#' << i << "=VRTPNT('vertex',#" << i-1 << ");" << std::endl;i++;
                
                /* Edge */
                *outfile << '#' << i << "=EDGCRV('edge',#" << i-3 << ",#" << i-1 << ",#" << i-5 << ",.T.);" << std::endl;i++;
                *outfile << '#' << i << "=ORNEDG('oriented edge',*,*,#" << i-1 << ",.T.);" << std::endl;i++;
            }
            
            /* Wireframe facet */
            *outfile << '#' << i << "=EDGLP('edge loop',(#" << i-31 << ",#" << i-21 << ",#" << i-11 << ",#" << i-1 << "));" << std::endl;i++;
            *outfile << '#' << i << "=FCOTBN('outer bound',#" << i-1 << ",.F.);" << std::endl;i++;
            
            /* Facet */
            double pointA[3], pointB[3], pointC[3];
            pointA[0] = positionIndices[quadIndices[j][0]][0];
            pointA[1] = positionIndices[quadIndices[j][0]][1];
            pointA[2] = positionIndices[quadIndices[j][0]][2];
            pointB[0] = positionIndices[quadIndices[j][1]][0];
            pointB[1] = positionIndices[quadIndices[j][1]][1];
            pointB[2] = positionIndices[quadIndices[j][1]][2];
            pointC[0] = positionIndices[quadIndices[j][2]][0];
            pointC[1] = positionIndices[quadIndices[j][2]][1];
            pointC[2] = positionIndices[quadIndices[j][2]][2];
            *outfile << '#' << i << "=CRTPNT('cartesian point',(" << std::fixed << pointA[0] << "," << std::fixed << pointA[1] << "," << std::fixed << pointA[2] << "));" << std::endl;i++;
            *outfile << '#' << i << "=DRCTN('direction',(" << std::fixed << pointB[0]-pointA[0] << "," << std::fixed << pointB[1]-pointA[1] << "," << std::fixed << pointB[2]-pointA[2] << "));" << std::endl;i++;
            *outfile << '#' << i << "=DRCTN('direction',(" << std::fixed << pointC[0]-pointA[0] << "," << std::fixed << pointC[1]-pointA[1] << "," << std::fixed << pointB[2]-pointA[2] << "));" << std::endl;i++;
            *outfile << '#' << i << "=A2PL3D('axis placement',#" << i-3 << ",#" << i-2 << ",#" << i-1 << ");" << std::endl;i++;
            *outfile << '#' << i << "=PLANE('plane',#" << i-1 << ");" << std::endl;i++;
            *outfile << '#' << i << "=ADVFC('Advanced Face',(#" << i-6 << "),#" << i-1 << ",.T.);" << std::endl;
            facetIndices.push_back(i);i++;
        }
    }
    
    /* Shape */
    *outfile << '#' << i << "=CLSSHL('Closed Shell',(";
    for(unsigned int i=0;i<facetIndices.size()-1;i++)
    {
        *outfile << "#" << facetIndices[i] << ',';
    }
    *outfile << facetIndices[facetIndices.size()] << "));" << std::endl;i++;
    *outfile << '#' << i << "=MNSLBR('Manifold Brep',#" << i-1 << ");" << std::endl;i++;
    *outfile << '#' << i << "=ABSR('adv brep shape',(#" << i-1 << "),#" << geoCon << ");" << std::endl;i++;
    *outfile << '#' << i << "=SHDFRP(#" << defShape << ",#" << i-1 << ");" << std::endl;i++;
    
    
    *outfile << std::endl;
    *outfile << "ENDSEC;" << std::endl;
    
    /* END KEYWORD */
    *outfile << "END-ISO-10303-21;" << std::endl;
    
    /**********/
    
    outfile->close();
    sout << filename << " written" << sendl;
    nbFiles++;
}

void STEPExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        std::cout << "key pressed " << std::endl;
        switch(ev->getKey())
        {

        case 'E':
        case 'e':
            if(m_fileFormat.getValue())
                writeSTEPShort();
            else
                writeSTEP();
            break;

        case 'F':
        case 'f':
            break;
        }
    }


    if ( /*simulation::AnimateEndEvent* ev =*/  dynamic_cast<simulation::AnimateEndEvent*>(event))
    {
        maxStep = exportEveryNbSteps.getValue();
        if (maxStep == 0) return;

        stepCounter++;
        if(stepCounter % maxStep == 0)
        {
            if(m_fileFormat.getValue())
                writeSTEPShort();
            else
                writeSTEP();
        }
    }
}

void STEPExporter::cleanup()
{
    if (exportAtEnd.getValue())
        (m_fileFormat.getValue()) ? writeSTEPShort() : writeSTEP();
}

void STEPExporter::bwdInit()
{
    if (exportAtBegin.getValue())
        (m_fileFormat.getValue()) ? writeSTEPShort() : writeSTEP();
}

} // namespace misc

} // namespace component

} // namespace sofa

