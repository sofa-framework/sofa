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
#ifndef SOFA_COMPONENT_MISC_MONITOR_INL
#define SOFA_COMPONENT_MISC_MONITOR_INL

#include <sofa/component/misc/Monitor.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/gl/Axis.h>
#include <fstream>
#include <sofa/defaulttype/Vec.h>
#include <cmath>



namespace sofa
{

namespace component
{

namespace misc
{

using namespace sofa::defaulttype;
using namespace std;


///////////////////////////// Monitor /////////////////////////////////////
template <class DataTypes>
Monitor<DataTypes>::Monitor()
    : saveXToGnuplot ( initData ( &saveXToGnuplot, false, "ExportPositions", "export monitored positions as gnuplot file" ) )
    , saveVToGnuplot ( initData ( &saveVToGnuplot, false, "ExportVelocities", "export monitored velocities as gnuplot file" ) )
    , saveFToGnuplot ( initData ( &saveFToGnuplot, false, "ExportForces", "export monitored forces as gnuplot file" ) )
    , monitoring( initData (&monitoring, "MonitoredParticles", "monitoring of desired particles"))
    ,showPositions (initData (&showPositions, false, "ShowPositions", "see the monitored positions"))
    ,positionsColor (initData (&positionsColor,Vector3 (1.0, 1.0, 0.0), "PositionsColor", "define the color of positions"))

    ,showVelocities (initData (&showVelocities, false, "ShowVelocities", "see the monitored velocities"))
    ,velocitiesColor(initData (&velocitiesColor,Vector3 (1.0, 1.0, 0.0), "VelocitiesColor", "define the color of velocities"))

    ,showForces (initData (&showForces, false, "ShowForces", "see the monitored forces"))
    ,forcesColor (initData (&forcesColor,Vector3 (1.0, 1.0, 0.0), "ForcesColor", "define the color of forces"))
    ,showMinThreshold (initData (&showMinThreshold, 0.01 ,"ShowMinThreshold", "under this value, vectors are not represented"))
    ,showTrajectories (initData (&showTrajectories, false ,"ShowTrajectories", "print the trajectory of monitored particles"))
    ,trajectoriesPrecision (initData (&trajectoriesPrecision, 0.1,"TrajectoriesPrecision", "set the dt between to save of positions"))
    ,trajectoriesColor(initData (&trajectoriesColor,Vector3 (1.0, 1.0, 0.0), "TrajectoriesColor", "define the color of the trajectories"))
    ,saveGnuplotX ( NULL ), saveGnuplotV ( NULL ), saveGnuplotF ( NULL )
    ,internalDt(0.0)
{}
/////////////////////////// end Monitor ///////////////////////////////////



////////////////////////////// ~Monitor ///////////////////////////////////
template <class DataTypes>
Monitor<DataTypes>::~Monitor()
{
    if (saveGnuplotX) delete ( saveGnuplotX );
    if (saveGnuplotV) delete ( saveGnuplotV );
    if (saveGnuplotF) delete ( saveGnuplotF );
}
///////////////////////////// end~Monitor /////////////////////////////////



///////////////////////////// setIndPos ///////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::setIndPos ( sofa::helper::vector < int > &_IdxPos )
{
    monitoring.beginEdit() -> setIndPos (_IdxPos);
    monitoring.endEdit();
}
/////////////////////////// end setIndPos /////////////////////////////////



///////////////////////////// setIndVels //////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::setIndVels ( sofa::helper::vector < int > &_IdxVels )
{
    monitoring.beginEdit() -> setIndVels (_IdxVels);
    monitoring.endEdit();
}
/////////////////////////// end setIndVels ////////////////////////////////



//////////////////////////// setIndForces /////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::setIndForces ( sofa::helper::vector < int > &_IdxForces )
{
    monitoring.beginEdit() -> setIndForces (_IdxForces);
    monitoring.endEdit();
}
////////////////////////// end setIndForces ///////////////////////////////



////////////////////////////// init () ////////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::init()
{
    mmodel =
        dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>
        ( this->getContext()->getMechanicalState() );

    monitoring.beginEdit() ->setValues (mmodel -> getV(), mmodel -> getF(), mmodel -> getX());
    sofa::helper::vector < int > initialPosIndices = monitoring.beginEdit()->getIndPos();
    sofa::helper::vector < int > initialVelsIndices = monitoring.beginEdit()->getIndVels();
    sofa::helper::vector < int > initialForcesIndices = monitoring.beginEdit()->getIndForces();

    monitoring.beginEdit()->setIndPosInit (initialPosIndices);
    monitoring.beginEdit()->setIndVelsInit (initialVelsIndices);
    monitoring.beginEdit()->setIndForcesInit (initialForcesIndices);
    monitoring.endEdit();


    initGnuplot ("./");
}
///////////////////////////// end init () /////////////////////////////////



///////////////////////////// reset () ////////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::reset()
{
    monitoring.beginEdit()->clearVecIndices();
    monitoring.beginEdit() ->setValues (mmodel -> getV(), mmodel -> getF(), mmodel -> getX());
    sofa::helper::vector < int > initialPosIndices = monitoring.beginEdit()->getIndPosInit();
    sofa::helper::vector < int > initialVelsIndices = monitoring.beginEdit()->getIndVelsInit();
    sofa::helper::vector < int > initialForcesIndices = monitoring.beginEdit()->getIndForcesInit();

    monitoring.beginEdit()->setIndPos (initialPosIndices);
    monitoring.beginEdit()->setIndVels (initialVelsIndices);
    monitoring.beginEdit()->setIndForces (initialForcesIndices);
    monitoring.endEdit();

    monitoring.beginEdit()->getSavePos()->clear();
    internalDt = 0.0;
}
//////////////////////////// end reset () /////////////////////////////////



//////////////////////////// reinit () ////////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::reinit()
{
    initGnuplot ( "./" );
}
/////////////////////////// end reinit () /////////////////////////////////



/////////////////////////// fwdDraw () ////////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::fwdDraw ( Pass pass = Std )
{
    if ( pass == core::VisualModel::Std )
    {
        monitoring.beginEdit() ->setValues (mmodel -> getV(), mmodel -> getF(), mmodel -> getX());
        monitoring.endEdit();

        if ( saveXToGnuplot.getValue() || saveVToGnuplot.getValue() || saveFToGnuplot.getValue() )
            exportGnuplot ( (Real) this ->getTime() );

    }
    glDisable(GL_LIGHTING);
    if (showPositions.getValue())
    {
        glPointSize(5.0);
        glBegin (GL_POINTS);
        glColor3d ((*positionsColor.beginEdit())[0], (*positionsColor.beginEdit())[1], (*positionsColor.beginEdit())[2]);
        for (unsigned int i=0; i < monitoring.beginEdit() -> sizeIdxPos(); i++)
        {
            Coord posvertex = monitoring.beginEdit() ->getPos(i);
            glVertex3d (posvertex[0], posvertex[1], posvertex[2]);
        }
        glEnd ();
    }

    if (showVelocities.getValue())
    {
        const VecCoord* mechPos = monitoring.beginEdit() -> getMechPos();
        for (unsigned int i=0; i < monitoring.beginEdit() -> sizeIdxVels(); i++)
        {
            sofa::helper::vector <int> posOfVel = monitoring.beginEdit() -> getIndVels ();


            Coord baseVelVertex = mechPos [0] [posOfVel[i]];
            Deriv topVelVertex = monitoring.beginEdit() -> getVel(i);

            if (vectorNorm(topVelVertex) > showMinThreshold.getValue())
            {
                for (unsigned short int j = 0; j < 3; j++)
                {
                    topVelVertex[j] = baseVelVertex[j] + topVelVertex[j];
                }

                glColor3d ((*velocitiesColor.beginEdit())[0], (*velocitiesColor.beginEdit())[1], (*velocitiesColor.beginEdit())[2]);
                helper::gl::Axis::draw(baseVelVertex, topVelVertex, 0.1);
            }
        }
    }

    if (showForces.getValue() && monitoring.beginEdit() -> getSizeVecForces())
    {
        const VecCoord* mechPos = monitoring.beginEdit() -> getMechPos();
        for (unsigned int i=0; i < monitoring.beginEdit() -> sizeIdxForces(); i++)
        {
            sofa::helper::vector <int> posOfForces = monitoring.beginEdit() -> getIndForces ();

            Coord baseVelVertex = mechPos [0] [posOfForces[i]];
            Deriv topVelVertex = monitoring.beginEdit() -> getVel(i);
            if (vectorNorm(topVelVertex) > showMinThreshold.getValue())
            {
                topVelVertex = baseVelVertex + topVelVertex;

                glColor3d ((*forcesColor.beginEdit())[0], (*forcesColor.beginEdit())[1], (*forcesColor.beginEdit())[2]);
                helper::gl::Axis::draw(baseVelVertex, topVelVertex, 0.2);
            }
        }
    }

    if (showTrajectories.getValue())
    {
        internalDt += this -> getContext()->getDt();
        VecCoord instantPositions;

        if (trajectoriesPrecision.getValue() <= internalDt)
        {
            internalDt = 0.0;
            for (unsigned int i=0; i < monitoring.beginEdit() -> sizeIdxPos(); i++)
            {
                instantPositions.push_back(monitoring.beginEdit() ->getPos(i));
            }
            monitoring.beginEdit()-> getSavePos()->push_back(instantPositions);
        }
        const unsigned int instantPositionsSize =  (*monitoring.beginEdit()->getMechPos())[0].size();//instantPositions.size();

        //printing those positions
        glLineWidth (1);
        glColor3d ((*trajectoriesColor.beginEdit())[0], (*trajectoriesColor.beginEdit())[1], (*trajectoriesColor.beginEdit())[2]);
        glBegin (GL_LINES);
        for (unsigned int i = 1; i < monitoring.beginEdit()->getSavePos()->size(); i++)
        {
            for (unsigned int j = 0; j < instantPositionsSize; j++)
            {
                glVertex3d (((*monitoring.beginEdit()->getSavePos())[i-1][j][0]),
                        ((*monitoring.beginEdit()->getSavePos())[i-1][j][1]),
                        ((*monitoring.beginEdit()->getSavePos())[i-1][j][2]));

                glVertex3d (((*monitoring.beginEdit()->getSavePos())[i][j][0]),
                        ((*monitoring.beginEdit()->getSavePos())[i][j][1]),
                        ((*monitoring.beginEdit()->getSavePos())[i][j][2]));
            }
        }
        glEnd();

    }

}
/////////////////////////// end fwdDraw () ////////////////////////////////







/////////////////////////// initGnuplot () ////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::initGnuplot ( const std::string path )
{
    if ( !this->getName().empty() )
    {
        if ( saveXToGnuplot.getValue() )
        {
            if ( saveGnuplotX != NULL ) delete saveGnuplotX;
            saveGnuplotX = new std::ofstream ( ( path + this->getName() +"_x.txt" ).c_str() );
            ( *saveGnuplotX ) << "# Gnuplot File : positions of "
                    << monitoring.beginEdit()->sizeIdxPos() << " particle(s) monitored"
                    <<  endl;
            ( *saveGnuplotX ) << "# 1st Column : time, others : particle(s) number ";

            sofa::helper::vector< int > posIdx = monitoring.beginEdit()->getIndPos();
            for (unsigned int i = 0; i < posIdx.size(); i++)
                ( *saveGnuplotX ) << posIdx.at(i) << " ";
            ( *saveGnuplotX ) << endl;

            monitoring.endEdit();
        }

        if ( saveVToGnuplot.getValue() )
        {
            if ( saveGnuplotV != NULL ) delete saveGnuplotV;

            saveGnuplotV = new std::ofstream ( ( path + this->getName() +"_v.txt" ).c_str() );
            ( *saveGnuplotV ) << "# Gnuplot File : velocities of "
                    << monitoring.beginEdit()->sizeIdxVels() << " particle(s) monitored"
                    <<  endl;
            ( *saveGnuplotV ) << "# 1st Column : time, others : particle(s) number ";

            sofa::helper::vector< int > velsIdx = monitoring.beginEdit()->getIndVels();
            for (unsigned int i = 0; i < velsIdx.size(); i++)
                ( *saveGnuplotV ) << velsIdx.at(i) << " ";
            ( *saveGnuplotV ) << endl;
            monitoring.endEdit();


        }



        if ( saveFToGnuplot.getValue() )
        {
            if ( saveGnuplotF != NULL ) delete saveGnuplotF;
            saveGnuplotF = new std::ofstream ( ( path + this->getName() +"_f.txt" ).c_str() );
            ( *saveGnuplotF ) << "# Gnuplot File : forces of "
                    << monitoring.beginEdit()->sizeIdxForces() << " particle(s) monitored"
                    <<  endl;
            ( *saveGnuplotF ) << "# 1st Column : time, others : particle(s) number ";

            sofa::helper::vector< int > forcesIdx = monitoring.beginEdit()->getIndForces();
            for (unsigned int i = 0; i < forcesIdx.size(); i++)
                ( *saveGnuplotF ) << forcesIdx.at(i) << " ";
            ( *saveGnuplotF ) << endl;
            monitoring.endEdit();
        }

    }
}
////////////////////////// end initGnuplot () /////////////////////////////



///////////////////////// exportGnuplot () ////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::exportGnuplot ( Real time )
{
    if ( saveXToGnuplot.getValue() )
    {
        ( *saveGnuplotX ) << time <<"\t" ;

        for (unsigned int i = 0; i < monitoring.beginEdit()->sizeIdxPos(); i++)
            ( *saveGnuplotX ) << monitoring.beginEdit() -> getPos(i) << "\t";
        ( *saveGnuplotX ) << endl;
        monitoring.endEdit();
    }
    if ( saveVToGnuplot.getValue() && monitoring.beginEdit()->getSizeVecVels() )
    {
        ( *saveGnuplotV ) << time <<"\t";

        for (unsigned int i = 0; i < monitoring.beginEdit()->sizeIdxVels(); i++)
            ( *saveGnuplotV ) << monitoring.beginEdit() -> getVel(i) << "\t";
        ( *saveGnuplotV ) << endl;
        monitoring.endEdit();
    }

    if ( saveFToGnuplot.getValue() && monitoring.beginEdit()->getSizeVecForces())
    {
        ( *saveGnuplotF ) << time <<"\t";

        for (unsigned int i = 0; i < monitoring.beginEdit()->sizeIdxForces(); i++)
            ( *saveGnuplotF ) << monitoring.beginEdit() -> getForce (i) << "\t";
        ( *saveGnuplotF ) << endl;
        monitoring.endEdit();
    }
}
///////////////////////////////////////////////////////////////////////////

} // namespace misc

} // namespace component

} // namespace sofa

#endif
