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
#ifndef SOFA_COMPONENT_MISC_MONITOR_INL
#define SOFA_COMPONENT_MISC_MONITOR_INL

#include <SofaValidation/Monitor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/objectmodel/Data.h>
#include <fstream>
#include <sofa/defaulttype/Vec.h>
#include <cmath>




namespace sofa
{

namespace component
{

namespace misc
{


///////////////////////////// Monitor /////////////////////////////////////
template <class DataTypes>
Monitor<DataTypes>::Monitor()
    : indices ( initData ( &indices, "indices", "MechanicalObject points indices to monitor" ) )
    , saveXToGnuplot ( initData ( &saveXToGnuplot, false, "ExportPositions", "export Monitored positions as gnuplot file" ) )
    , saveVToGnuplot ( initData ( &saveVToGnuplot, false, "ExportVelocities", "export Monitored velocities as gnuplot file" ) )
    , saveFToGnuplot ( initData ( &saveFToGnuplot, false, "ExportForces", "export Monitored forces as gnuplot file" ) )
    ,showPositions (initData (&showPositions, false, "showPositions", "see the Monitored positions"))
    ,positionsColor (initData (&positionsColor, "PositionsColor", "define the color of positions"))
    ,showVelocities (initData (&showVelocities, false, "showVelocities", "see the Monitored velocities"))
    ,velocitiesColor(initData (&velocitiesColor, "VelocitiesColor", "define the color of velocities"))
    ,showForces (initData (&showForces, false, "showForces", "see the Monitored forces"))
    ,forcesColor (initData (&forcesColor, "ForcesColor", "define the color of forces"))
    ,showMinThreshold (initData (&showMinThreshold, 0.01 ,"showMinThreshold", "under this value, vectors are not represented"))
    ,showTrajectories (initData (&showTrajectories, false ,"showTrajectories", "print the trajectory of Monitored particles"))
    ,trajectoriesPrecision (initData (&trajectoriesPrecision, 0.1,"TrajectoriesPrecision", "set the dt between to save of positions"))
    ,trajectoriesColor(initData (&trajectoriesColor, "TrajectoriesColor", "define the color of the trajectories"))
    ,showSizeFactor(initData (&showSizeFactor, 1.0, "sizeFactor", "factor to multiply to arrows"))
    ,saveGnuplotX ( NULL ), saveGnuplotV ( NULL ), saveGnuplotF ( NULL )
    ,X (NULL), V(NULL), F(NULL)
    ,internalDt(0.0)
{
    if (!f_listening.isSet()) f_listening.setValue(true);

    positionsColor=RGBAColor::yellow();
    velocitiesColor=RGBAColor::yellow();
    forcesColor=RGBAColor::yellow();
    trajectoriesColor=RGBAColor::yellow();
}
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




////////////////////////////// init () ////////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::init()
{
    core::behavior::MechanicalState<DataTypes>* mmodel = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>( this->getContext()->getMechanicalState() );

    if(!mmodel)
    {
        serr<<"Monitor error : no MechanicalObject found"<<sendl;
        return;
    }

    X = &mmodel->read(core::ConstVecCoordId::position())->getValue();
    V = &mmodel->read(core::ConstVecDerivId::velocity())->getValue();
    F = &mmodel->read(core::ConstVecDerivId::force())->getValue();


    initGnuplot ("./");

    savedPos.resize(indices.getValue().size());
}
///////////////////////////// end init () /////////////////////////////////



///////////////////////////// reset () ////////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::reset()
{
    internalDt = 0.0;
    for(unsigned int i=0 ; i<indices.getValue().size() ; ++i)
        savedPos[i].clear();
}
//////////////////////////// end reset () /////////////////////////////////



//////////////////////////// reinit () ////////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::reinit()
{
    init();
}
/////////////////////////// end reinit () /////////////////////////////////



template<class DataTypes>
void Monitor<DataTypes>::handleEvent( core::objectmodel::Event* ev )
{
    if (sofa::simulation::AnimateEndEvent::checkEventType(ev))
    {
        if ( saveXToGnuplot.getValue() || saveVToGnuplot.getValue() || saveFToGnuplot.getValue() )
            exportGnuplot ( (Real) this ->getTime() );

        if (showTrajectories.getValue())
        {
            internalDt += this -> getContext()->getDt();

            if (trajectoriesPrecision.getValue() <= internalDt)
            {
                internalDt = 0.0;
                for (unsigned int i=0; i < indices.getValue().size(); ++i)
                {
                    savedPos[i].push_back( (*X)[indices.getValue()[i]] );
                }
            }
        }
    }
}

/////////////////////////// draw () ////////////////////////////////////
template<class DataTypes>
void Monitor<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    vparams->drawTool()->setLightingEnabled(false);
    if (showPositions.getValue())
    {
        helper::vector<defaulttype::Vector3> points;
        for (unsigned int i=0; i < indices.getValue().size(); ++i)
        {
            Coord posvertex = (*X)[indices.getValue()[i]];
            points.push_back(defaulttype::Vector3(posvertex[0],posvertex[1],posvertex[2]));
        }
        vparams->drawTool()->drawPoints(points, (float)(showSizeFactor.getValue())*2.0f, positionsColor.getValue());

    }

    if (showVelocities.getValue())
    {
        for (unsigned int i=0; i < indices.getValue().size(); ++i)
        {
            Coord posVertex = (*X)[indices.getValue()[i]];
            defaulttype::Vector3 p1(posVertex[0],posVertex[1],posVertex[2]);
            Deriv velVertex = (*V)[indices.getValue()[i]];
            defaulttype::Vector3 p2(showSizeFactor.getValue()*velVertex[0],showSizeFactor.getValue()*velVertex[1],showSizeFactor.getValue()*velVertex[2]);

            if(p2.norm() > showMinThreshold.getValue())
                vparams->drawTool()->drawArrow(p1, p1+p2, (float)(showSizeFactor.getValue()*p2.norm()/20.0), velocitiesColor.getValue());
        }
    }

    if (showForces.getValue() && F->size()>0)
    {
        for (unsigned int i=0; i < indices.getValue().size(); ++i)
        {
            Coord posVertex = (*X)[indices.getValue()[i]];
            defaulttype::Vector3 p1(posVertex[0],posVertex[1],posVertex[2]);
            Deriv forceVertex = (*F)[indices.getValue()[i]];
            defaulttype::Vector3 p2(showSizeFactor.getValue()*forceVertex[0],showSizeFactor.getValue()*forceVertex[1],showSizeFactor.getValue()*forceVertex[2]);

            if(p2.norm() > showMinThreshold.getValue())
                vparams->drawTool()->drawArrow(p1, p1+p2, (float)(showSizeFactor.getValue()*p2.norm()/20.0), forcesColor.getValue());
        }
    }

    if (showTrajectories.getValue())
    {
        internalDt += this -> getContext()->getDt();
        for (unsigned int i=0; i < indices.getValue().size(); ++i)
        {
            helper::vector<defaulttype::Vector3> points;
            Coord point;
            for (unsigned int j=0 ; j<savedPos[i].size() ; ++j)
            {
                point = savedPos[i][j];
                points.push_back(defaulttype::Vector3(point[0], point[1], point[2]));
                if(j!=0)
                    points.push_back(defaulttype::Vector3(point[0], point[1], point[2]));
            }
            vparams->drawTool()->drawLines(points, (float)(showSizeFactor.getValue()*0.2), trajectoriesColor.getValue());
        }
    }
}
/////////////////////////// end draw () ////////////////////////////////







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
                    << indices.getValue().size() << " particle(s) Monitored"
                    <<  std::endl;
            ( *saveGnuplotX ) << "# 1st Column : time, others : particle(s) number ";

            for (unsigned int i = 0; i < indices.getValue().size(); i++)
                ( *saveGnuplotX ) << indices.getValue()[i] << " ";
            ( *saveGnuplotX ) << std::endl;

        }

        if ( saveVToGnuplot.getValue() )
        {
            if ( saveGnuplotV != NULL ) delete saveGnuplotV;

            saveGnuplotV = new std::ofstream ( ( path + this->getName() +"_v.txt" ).c_str() );
            ( *saveGnuplotV ) << "# Gnuplot File : velocities of "
                    << indices.getValue().size() << " particle(s) Monitored"
                    <<  std::endl;
            ( *saveGnuplotV ) << "# 1st Column : time, others : particle(s) number ";

            for (unsigned int i = 0; i < indices.getValue().size(); i++)
                ( *saveGnuplotV ) << indices.getValue()[i] << " ";
            ( *saveGnuplotV ) << std::endl;
        }



        if ( saveFToGnuplot.getValue() )
        {
            if ( saveGnuplotF != NULL ) delete saveGnuplotF;
            saveGnuplotF = new std::ofstream ( ( path + this->getName() +"_f.txt" ).c_str() );
            ( *saveGnuplotF ) << "# Gnuplot File : forces of "
                    << indices.getValue().size() << " particle(s) Monitored"
                    <<  std::endl;
            ( *saveGnuplotF ) << "# 1st Column : time, others : particle(s) number ";

            for (unsigned int i = 0; i < indices.getValue().size(); i++)
                ( *saveGnuplotF ) << indices.getValue()[i] << " ";
            ( *saveGnuplotF ) << std::endl;
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

        for (unsigned int i = 0; i < indices.getValue().size(); i++)
            ( *saveGnuplotX ) << (*X)[indices.getValue()[i]] << "\t";
        ( *saveGnuplotX ) << std::endl;
    }
    if ( saveVToGnuplot.getValue() && V->size()>0 )
    {
        ( *saveGnuplotV ) << time <<"\t";

        for (unsigned int i = 0; i < indices.getValue().size(); i++)
            ( *saveGnuplotV ) << (*V)[indices.getValue()[i]] << "\t";
        ( *saveGnuplotV ) << std::endl;
    }

    if ( saveFToGnuplot.getValue() && F->size()>0)
    {
        ( *saveGnuplotF ) << time <<"\t";

        for (unsigned int i = 0; i < indices.getValue().size(); i++)
            ( *saveGnuplotF ) << (*F)[indices.getValue()[i]] << "\t";
        ( *saveGnuplotF ) << std::endl;
    }
}
///////////////////////////////////////////////////////////////////////////

} // namespace misc

} // namespace component

} // namespace sofa

#endif
