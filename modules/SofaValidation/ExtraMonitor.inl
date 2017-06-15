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
#ifndef SOFA_COMPONENT_MISC_EXTRAMONITOR_INL
#define SOFA_COMPONENT_MISC_EXTRAMONITOR_INL

#include <SofaValidation/ExtraMonitor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/MechanicalComputeEnergyVisitor.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/objectmodel/Data.h>
#include <fstream>
#include <sofa/defaulttype/Vec.h>
#include <cmath>
#include <limits>


namespace sofa
{

namespace component
{

namespace misc
{


///////////////////////////// Monitor /////////////////////////////////////
template <class DataTypes>
ExtraMonitor<DataTypes>::ExtraMonitor()
    : Inherit()
    , saveWcinToGnuplot ( initData ( &saveWcinToGnuplot, false, "ExportWcin", "export Wcin of the monitored dofs as gnuplot file" ) )
    , saveWextToGnuplot ( initData ( &saveWextToGnuplot, false, "ExportWext", "export Wext of the monitored dofs as gnuplot file" ) )
    , resultantF( initData( &resultantF, true, "resultantF", "export force resultant of the monitored dofs as gnuplot file instead of all dofs") )
    , minX( initData( &minX, -1, "minCoord", "export minimum displacement on the given coordinate as gnuplot file instead of positions of all dofs" ) )
    , maxX( initData( &maxX, -1, "maxCoord", "export minimum displacement on the given coordinate as gnuplot file instead of positions of all dofs" ) )
    , disp( initData( &disp, -1, "dispCoord", "export displacement on the given coordinate as gnuplot file" ) )
{
    saveGnuplotWcin = NULL;
    saveGnuplotWext = NULL;
}
/////////////////////////// end Monitor ///////////////////////////////////

////////////////////////////// init () ////////////////////////////////////
template<class DataTypes>
void ExtraMonitor<DataTypes>::init()
{
    Inherit::init();
    if (minX.getValue() != -1)
    {
        initialMinPos.resize(this->indices.getValue().size());
        for(unsigned i=0; i<this->indices.getValue().size(); i++)
            initialMinPos[i] = (*this->X)[this->indices.getValue()[i]][minX.getValue()];
    }
    if (maxX.getValue() != -1)
    {
        initialMaxPos.resize(this->indices.getValue().size());
        for(unsigned i=0; i<this->indices.getValue().size(); i++)
            initialMaxPos[i] = (*this->X)[this->indices.getValue()[i]][maxX.getValue()];
    }
    if (disp.getValue() != -1)
    {
        initialPos.resize(this->indices.getValue().size());
        for(unsigned i=0; i<this->indices.getValue().size(); i++)
            initialPos[i] = (*this->X)[this->indices.getValue()[i]][disp.getValue()];
    }
}
///////////////////////////// end init () /////////////////////////////////

/////////////////////////// initGnuplot () ////////////////////////////////
template<class DataTypes>
void ExtraMonitor<DataTypes>::initGnuplot ( const std::string path )
{
    if ( !this->getName().empty() )
    {
        if ( this->saveXToGnuplot.getValue() )
        {
            if ( this->saveGnuplotX != NULL ) delete this->saveGnuplotX;
            this->saveGnuplotX = new std::ofstream ( ( path + this->getName() +"_x.txt" ).c_str() );
            if (minX.getValue() == -1 && maxX.getValue() == -1)
            {
                if (disp.getValue() == -1)
                {
                    ( *this->saveGnuplotX ) << "# Gnuplot File : positions of ";
                }
                else
                {
                    ( *this->saveGnuplotX ) << "# Gnuplot File : displacement of ";
                }
                ( *this->saveGnuplotX )<< this->indices.getValue().size() << " particle(s) Monitored"
                        <<  std::endl;
                ( *this->saveGnuplotX ) << "# 1st Column : time, others : particle(s) number ";

                for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                    ( *this->saveGnuplotX ) << this->indices.getValue()[i] << " ";
                ( *this->saveGnuplotX ) << std::endl;
            }
            else
            {
                ( *this->saveGnuplotX ) << "# Gnuplot File : resultant of the positions of "
                        << this->indices.getValue().size() << " particle(s) Monitored"
                        << std::endl;
                ( *this->saveGnuplotX ) << "# 1st Column : time";
                if (minX.getValue() != -1)
                {
                    ( *this->saveGnuplotX )<<", minimum displacement on "<<minX.getValue()<<" coordinate";
                }
                if (maxX.getValue() != -1)
                {
                    ( *this->saveGnuplotX )<<", maximum displacement on "<<maxX.getValue()<<" coordinate";
                }
                ( *this->saveGnuplotX )<< std::endl;
            }
        }

        if ( this->saveVToGnuplot.getValue() )
        {
            if ( this->saveGnuplotV != NULL ) delete this->saveGnuplotV;

            this->saveGnuplotV = new std::ofstream ( ( path + this->getName() +"_v.txt" ).c_str() );
            ( *this->saveGnuplotV ) << "# Gnuplot File : velocities of "
                    << this->indices.getValue().size() << " particle(s) Monitored"
                    <<  std::endl;
            ( *this->saveGnuplotV ) << "# 1st Column : time, others : particle(s) number ";

            for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                ( *this->saveGnuplotV ) << this->indices.getValue()[i] << " ";
            ( *this->saveGnuplotV ) << std::endl;
        }

        if ( this->saveFToGnuplot.getValue() )
        {
            if ( this->saveGnuplotF != NULL ) delete this->saveGnuplotF;
            this->saveGnuplotF = new std::ofstream ( ( path + this->getName() +"_f.txt" ).c_str() );
            if (!resultantF.getValue())
            {
                ( *this->saveGnuplotF ) << "# Gnuplot File : forces of "
                        << this->indices.getValue().size() << " particle(s) Monitored"
                        <<  std::endl;
                ( *this->saveGnuplotF ) << "# 1st Column : time, others : particle(s) number ";

                for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                    ( *this->saveGnuplotF ) << this->indices.getValue()[i] << " ";
                ( *this->saveGnuplotF ) << std::endl;
            }
            else
            {
                ( *this->saveGnuplotF ) << "# Gnuplot File : resultant of the forces of "
                        << this->indices.getValue().size() << " particle(s) Monitored"
                        << std::endl;
                ( *this->saveGnuplotF ) << "# 1st Column : time, other : resultant force "
                        << std::endl;
            }
        }

        if ( this->saveWcinToGnuplot.getValue() )
        {
            if ( this->saveGnuplotWcin != NULL ) delete this->saveGnuplotWcin;
            this->saveGnuplotWcin = new std::ofstream ( ( path + this->getName() + "_wcin.txt" ).c_str() );
            ( *this->saveGnuplotWcin ) << "# Gnuplot File : kinetic energy of the system "<<std::endl;
            ( *this->saveGnuplotWcin ) << "# 1st Column : time, 2nd : kinetic energy"<< std::endl;
        }// saveWcinToGnuplot

        if ( this->saveWextToGnuplot.getValue() )
        {
            if ( this->saveGnuplotWext != NULL ) delete this->saveGnuplotWext;
            this->saveGnuplotWext = new std::ofstream ( ( path + this->getName() + "_wext.txt" ).c_str() );
            ( *this->saveGnuplotWext ) << "# Gnuplot File : external energy of the system "<<std::endl;
            ( *this->saveGnuplotWext ) << "# 1st Column : time, 2nd : external energy"<< std::endl;
        }// saveWextToGnuplot
    }
}
////////////////////////// end initGnuplot () /////////////////////////////

template<class DataTypes>
void ExtraMonitor<DataTypes>::handleEvent( core::objectmodel::Event* ev )
{
    if (sofa::simulation::AnimateEndEvent::checkEventType(ev))
    {
        if ( this->saveXToGnuplot.getValue() || this->saveVToGnuplot.getValue() || this->saveFToGnuplot.getValue() || saveWcinToGnuplot.getValue() || saveWextToGnuplot.getValue() )
            exportGnuplot ( (Real) this ->getTime() );

        if (this->showTrajectories.getValue())
        {
            this->internalDt += this -> getContext()->getDt();

            if (this->trajectoriesPrecision.getValue() <= this->internalDt)
            {
                this->internalDt = 0.0;
                for (unsigned int i=0; i < this->indices.getValue().size(); ++i)
                {
                    this->savedPos[i].push_back( (*this->X)[this->indices.getValue()[i]] );
                }
            }
        }
    }
}


///////////////////////// exportGnuplot () ////////////////////////////////
template<class DataTypes>
void ExtraMonitor<DataTypes>::exportGnuplot ( Real time )
{
    if ( this->saveXToGnuplot.getValue() )
    {
        ( *this->saveGnuplotX ) << time <<"\t" ;

        if ((minX.getValue() == -1) && (maxX.getValue() == -1))
        {
            if (disp.getValue() == -1)
            {
                for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                    ( *this->saveGnuplotX ) << (*this->X)[this->indices.getValue()[i]] << "\t";
            }
            else
            {
                for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                    ( *this->saveGnuplotX ) << (*this->X)[this->indices.getValue()[i]][disp.getValue()] - initialPos[i]<< "\t";
            }
            ( *this->saveGnuplotX ) << std::endl;
        }
        else
        {
            if (minX.getValue() != -1)
            {
                Real min = std::numeric_limits<Real>::max();
                Real displ = 0.0;
                for (unsigned i = 0; i < this->indices.getValue().size(); i++)
                {
                    displ = (*this->X)[this->indices.getValue()[i]][minX.getValue()] - initialMinPos[i];
                    displ = fabs(displ); // TODO to be read again
                    if (displ < min)
                        min = displ;
                }
                ( *this->saveGnuplotX ) << min << "\t";
            }
            if (maxX.getValue() != -1)
            {
                Real max = std::numeric_limits<Real>::min();
                Real displ = 0.0;
                for (unsigned i = 0; i < this->indices.getValue().size(); i++)
                {
                    displ = (*this->X)[this->indices.getValue()[i]][maxX.getValue()] - initialMaxPos[i];
                    displ = fabs(displ); // TODO to be read again
                    if (displ > max)
                        max = displ;
                }
                ( *this->saveGnuplotX ) << max;
            }
            ( *this->saveGnuplotX ) << std::endl;
        }
    }
    if ( this->saveVToGnuplot.getValue() && this->V->size()>0 )
    {
        ( *this->saveGnuplotV ) << time <<"\t";

        for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
            ( *this->saveGnuplotV ) << (*this->V)[this->indices.getValue()[i]] << "\t";
        ( *this->saveGnuplotV ) << std::endl;
    }

    if ( this->saveFToGnuplot.getValue() && this->F->size()>0)
    {
        ( *this->saveGnuplotF ) << time <<"\t";

        if (!resultantF.getValue())
        {
            for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                ( *this->saveGnuplotF ) << (*this->F)[this->indices.getValue()[i]] << "\t";
            ( *this->saveGnuplotF ) << std::endl;
        }
        else
        {
            Deriv resultant;
            for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                resultant += (*this->F)[this->indices.getValue()[i]];

            (*this->saveGnuplotF ) << resultant << std::endl;
        }
    }

    if ( this->saveWcinToGnuplot.getValue() )
    {
        sofa::simulation::MechanicalComputeEnergyVisitor *kineticEnergy = new sofa::simulation::MechanicalComputeEnergyVisitor(core::MechanicalParams::defaultInstance());
        kineticEnergy->execute( this->getContext() );
        ( *this->saveGnuplotWcin ) << time <<"\t";
        ( *this->saveGnuplotWcin ) << kineticEnergy->getKineticEnergy() << std::endl;
    }// export kinetic energy

    if ( this->saveWextToGnuplot.getValue() )
    {
        sofa::simulation::MechanicalComputeEnergyVisitor *potentialEnergy= new sofa::simulation::MechanicalComputeEnergyVisitor(core::MechanicalParams::defaultInstance());
        potentialEnergy->execute( this->getContext() );
        ( *this->saveGnuplotWext ) << time <<"\t";
        ( *this->saveGnuplotWext ) << potentialEnergy->getPotentialEnergy() << std::endl;
    }// export external energy

}
///////////////////////////////////////////////////////////////////////////

} // namespace misc

} // namespace component

} // namespace sofa

#endif
