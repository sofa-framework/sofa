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
        initialMinPos.resize(this->d_indices.getValue().size());
        for(unsigned i=0; i<this->d_indices.getValue().size(); i++)
            initialMinPos[i] = (*this->m_X)[this->d_indices.getValue()[i]][minX.getValue()];
    }
    if (maxX.getValue() != -1)
    {
        initialMaxPos.resize(this->d_indices.getValue().size());
        for(unsigned i=0; i<this->d_indices.getValue().size(); i++)
            initialMaxPos[i] = (*this->m_X)[this->d_indices.getValue()[i]][maxX.getValue()];
    }
    if (disp.getValue() != -1)
    {
        initialPos.resize(this->d_indices.getValue().size());
        for(unsigned i=0; i<this->d_indices.getValue().size(); i++)
            initialPos[i] = (*this->m_X)[this->d_indices.getValue()[i]][disp.getValue()];
    }
}
///////////////////////////// end init () /////////////////////////////////

/////////////////////////// initGnuplot () ////////////////////////////////
template<class DataTypes>
void ExtraMonitor<DataTypes>::initGnuplot ( const std::string path )
{
    if ( !this->getName().empty() )
    {
        if ( this->d_saveXToGnuplot.getValue() )
        {
            if ( this->m_saveGnuplotX != NULL ) delete this->m_saveGnuplotX;
            this->m_saveGnuplotX = new std::ofstream ( ( path +"_x.txt" ).c_str() );
            if (minX.getValue() == -1 && maxX.getValue() == -1)
            {
                if (disp.getValue() == -1)
                {
                    ( *this->m_saveGnuplotX ) << "# Gnuplot File : positions of ";
                }
                else
                {
                    ( *this->m_saveGnuplotX ) << "# Gnuplot File : displacement of ";
                }
                ( *this->m_saveGnuplotX )<< this->d_indices.getValue().size() << " particle(s) Monitored"
                                         <<  std::endl;
                ( *this->m_saveGnuplotX ) << "# 1st Column : time, others : particle(s) number ";

                for (unsigned int i = 0; i < this->d_indices.getValue().size(); i++)
                    ( *this->m_saveGnuplotX ) << this->d_indices.getValue()[i] << " ";
                ( *this->m_saveGnuplotX ) << std::endl;
            }
            else
            {
                ( *this->m_saveGnuplotX ) << "# Gnuplot File : resultant of the positions of "
                                          << this->d_indices.getValue().size() << " particle(s) Monitored"
                                          << std::endl;
                ( *this->m_saveGnuplotX ) << "# 1st Column : time";
                if (minX.getValue() != -1)
                {
                    ( *this->m_saveGnuplotX )<<", minimum displacement on "<<minX.getValue()<<" coordinate";
                }
                if (maxX.getValue() != -1)
                {
                    ( *this->m_saveGnuplotX )<<", maximum displacement on "<<maxX.getValue()<<" coordinate";
                }
                ( *this->m_saveGnuplotX )<< std::endl;
            }
        }

        if ( this->d_saveVToGnuplot.getValue() )
        {
            if ( this->m_saveGnuplotV != NULL ) delete this->m_saveGnuplotV;

            this->m_saveGnuplotV = new std::ofstream ( ( path +"_v.txt" ).c_str() );
            ( *this->m_saveGnuplotV ) << "# Gnuplot File : velocities of "
                                      << this->d_indices.getValue().size() << " particle(s) Monitored"
                                      <<  std::endl;
            ( *this->m_saveGnuplotV ) << "# 1st Column : time, others : particle(s) number ";

            for (unsigned int i = 0; i < this->d_indices.getValue().size(); i++)
                ( *this->m_saveGnuplotV ) << this->d_indices.getValue()[i] << " ";
            ( *this->m_saveGnuplotV ) << std::endl;
        }

        if ( this->d_saveFToGnuplot.getValue() )
        {
            if ( this->m_saveGnuplotF != NULL ) delete this->m_saveGnuplotF;
            this->m_saveGnuplotF = new std::ofstream ( ( path +"_f.txt" ).c_str() );
            if (!resultantF.getValue())
            {
                ( *this->m_saveGnuplotF ) << "# Gnuplot File : forces of "
                                          << this->d_indices.getValue().size() << " particle(s) Monitored"
                                          <<  std::endl;
                ( *this->m_saveGnuplotF ) << "# 1st Column : time, others : particle(s) number ";

                for (unsigned int i = 0; i < this->d_indices.getValue().size(); i++)
                    ( *this->m_saveGnuplotF ) << this->d_indices.getValue()[i] << " ";
                ( *this->m_saveGnuplotF ) << std::endl;
            }
            else
            {
                ( *this->m_saveGnuplotF ) << "# Gnuplot File : resultant of the forces of "
                                          << this->d_indices.getValue().size() << " particle(s) Monitored"
                                          << std::endl;
                ( *this->m_saveGnuplotF ) << "# 1st Column : time, other : resultant force "
                                          << std::endl;
            }
        }

        if ( this->saveWcinToGnuplot.getValue() )
        {
            if ( this->saveGnuplotWcin != NULL ) delete this->saveGnuplotWcin;
            this->saveGnuplotWcin = new std::ofstream ( ( path + "_wcin.txt" ).c_str() );
            ( *this->saveGnuplotWcin ) << "# Gnuplot File : kinetic energy of the system "<<std::endl;
            ( *this->saveGnuplotWcin ) << "# 1st Column : time, 2nd : kinetic energy"<< std::endl;
        }// saveWcinToGnuplot

        if ( this->saveWextToGnuplot.getValue() )
        {
            if ( this->saveGnuplotWext != NULL ) delete this->saveGnuplotWext;
            this->saveGnuplotWext = new std::ofstream ( ( path + "_wext.txt" ).c_str() );
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
        if ( this->d_saveXToGnuplot.getValue() || this->d_saveVToGnuplot.getValue() || this->d_saveFToGnuplot.getValue() || saveWcinToGnuplot.getValue() || saveWextToGnuplot.getValue() )
            exportGnuplot ( (Real) this ->getTime() );

        if (this->d_showTrajectories.getValue())
        {
            this->m_internalDt += this->getContext()->getDt();

            if (this->d_trajectoriesPrecision.getValue() <= this->m_internalDt)
            {
                this->m_internalDt = 0.0;
                for (unsigned int i=0; i < this->d_indices.getValue().size(); ++i)
                {
                    this->m_savedPos[i].push_back( (*this->m_X)[this->d_indices.getValue()[i]] );
                }
            }
        }
    }
}


///////////////////////// exportGnuplot () ////////////////////////////////
template<class DataTypes>
void ExtraMonitor<DataTypes>::exportGnuplot ( Real time )
{
    if ( this->d_saveXToGnuplot.getValue() )
    {
        ( *this->m_saveGnuplotX ) << time <<"\t" ;

        if ((minX.getValue() == -1) && (maxX.getValue() == -1))
        {
            if (disp.getValue() == -1)
            {
                for (unsigned int i = 0; i < this->d_indices.getValue().size(); i++)
                    ( *this->m_saveGnuplotX ) << (*this->m_X)[this->d_indices.getValue()[i]] << "\t";
            }
            else
            {
                for (unsigned int i = 0; i < this->d_indices.getValue().size(); i++)
                    ( *this->m_saveGnuplotX ) << (*this->m_X)[this->d_indices.getValue()[i]][disp.getValue()] - initialPos[i]<< "\t";
            }
            ( *this->m_saveGnuplotX ) << std::endl;
        }
        else
        {
            if (minX.getValue() != -1)
            {
                Real min = std::numeric_limits<Real>::max();
                Real displ = 0.0;
                for (unsigned i = 0; i < this->d_indices.getValue().size(); i++)
                {
                    displ = (*this->m_X)[this->d_indices.getValue()[i]][minX.getValue()] - initialMinPos[i];
                    displ = fabs(displ); // TODO to be read again
                    if (displ < min)
                        min = displ;
                }
                ( *this->m_saveGnuplotX ) << min << "\t";
            }
            if (maxX.getValue() != -1)
            {
                Real max = std::numeric_limits<Real>::min();
                Real displ = 0.0;
                for (unsigned i = 0; i < this->d_indices.getValue().size(); i++)
                {
                    displ = (*this->m_X)[this->d_indices.getValue()[i]][maxX.getValue()] - initialMaxPos[i];
                    displ = fabs(displ); // TODO to be read again
                    if (displ > max)
                        max = displ;
                }
                ( *this->m_saveGnuplotX ) << max;
            }
            ( *this->m_saveGnuplotX ) << std::endl;
        }
    }
    if ( this->d_saveVToGnuplot.getValue() && this->m_V->size()>0 )
    {
        ( *this->m_saveGnuplotV ) << time <<"\t";

        for (unsigned int i = 0; i < this->d_indices.getValue().size(); i++)
            ( *this->m_saveGnuplotV ) << (*this->m_V)[this->d_indices.getValue()[i]] << "\t";
        ( *this->m_saveGnuplotV ) << std::endl;
    }

    if ( this->d_saveFToGnuplot.getValue() && this->m_F->size()>0)
    {
        ( *this->m_saveGnuplotF ) << time <<"\t";

        if (!resultantF.getValue())
        {
            for (unsigned int i = 0; i < this->d_indices.getValue().size(); i++)
                ( *this->m_saveGnuplotF ) << (*this->m_F)[this->d_indices.getValue()[i]] << "\t";
            ( *this->m_saveGnuplotF ) << std::endl;
        }
        else
        {
            Deriv resultant;
            for (unsigned int i = 0; i < this->d_indices.getValue().size(); i++)
                resultant += (*this->m_F)[this->d_indices.getValue()[i]];

            (*this->m_saveGnuplotF ) << resultant << std::endl;
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
