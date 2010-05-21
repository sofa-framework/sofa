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
#ifndef SOFA_COMPONENT_MISC_EXTRAMONITOR_INL
#define SOFA_COMPONENT_MISC_EXTRAMONITOR_INL

#include <sofa/component/misc/ExtraMonitor.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/objectmodel/Data.h>
#include <fstream>
#include <sofa/defaulttype/Vec.h>
#include <cmath>
#include <limits>
#include <sofa/helper/gl/DrawManager.h>



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
ExtraMonitor<DataTypes>::ExtraMonitor()
    : Inherit()
    , saveWcinToGnuplot ( initData ( &saveWcinToGnuplot, false, "ExportWcin", "export Wcin of the monitored dofs as gnuplot file" ) )
    , saveWextToGnuplot ( initData ( &saveWextToGnuplot, false, "ExportWext", "export Wext of the monitored dofs as gnuplot file" ) )
    , resultantF( initData( &resultantF, true, "resultantF", "export force resultant of the monitored dofs as gnuplot file instead of all dofs") )
    , minX( initData( &minX, -1, "minCoord", "export minimum displacement on the given coordinate as gnuplot file instead of positions of all dofs" ) )
    , maxX( initData( &maxX, -1, "maxCoord", "export minimum displacement on the given coordinate as gnuplot file instead of positions of all dofs" ) )
{
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
    std::cout<<"maxX.getValue() = "<<maxX.getValue()<<std::endl;
    std::cout<<"minX.getValue() = "<<minX.getValue()<<std::endl;
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
                ( *this->saveGnuplotX ) << "# Gnuplot File : positions of "
                        << this->indices.getValue().size() << " particle(s) Monitored"
                        <<  endl;
                ( *this->saveGnuplotX ) << "# 1st Column : time, others : particle(s) number ";

                for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                    ( *this->saveGnuplotX ) << this->indices.getValue()[i] << " ";
                ( *this->saveGnuplotX ) << endl;
            }
            else
            {
                ( *this->saveGnuplotX ) << "# Gnuplot File : resultant of the positions of "
                        << this->indices.getValue().size() << " particle(s) Monitored"
                        << endl;
                ( *this->saveGnuplotX ) << "# 1st Column : time";
                if (minX.getValue() != -1)
                {
                    ( *this->saveGnuplotX )<<", minimum displacement on "<<minX.getValue()<<" coordinate";
                }
                if (maxX.getValue() != -1)
                {
                    ( *this->saveGnuplotX )<<", maximum displacement on "<<maxX.getValue()<<" coordinate";
                }
                ( *this->saveGnuplotX )<< endl;
            }
        }

        if ( this->saveVToGnuplot.getValue() )
        {
            if ( this->saveGnuplotV != NULL ) delete this->saveGnuplotV;

            this->saveGnuplotV = new std::ofstream ( ( path + this->getName() +"_v.txt" ).c_str() );
            ( *this->saveGnuplotV ) << "# Gnuplot File : velocities of "
                    << this->indices.getValue().size() << " particle(s) Monitored"
                    <<  endl;
            ( *this->saveGnuplotV ) << "# 1st Column : time, others : particle(s) number ";

            for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                ( *this->saveGnuplotV ) << this->indices.getValue()[i] << " ";
            ( *this->saveGnuplotV ) << endl;
        }

        if ( this->saveFToGnuplot.getValue() )
        {
            if ( this->saveGnuplotF != NULL ) delete this->saveGnuplotF;
            this->saveGnuplotF = new std::ofstream ( ( path + this->getName() +"_f.txt" ).c_str() );
            if (!resultantF.getValue())
            {
                ( *this->saveGnuplotF ) << "# Gnuplot File : forces of "
                        << this->indices.getValue().size() << " particle(s) Monitored"
                        <<  endl;
                ( *this->saveGnuplotF ) << "# 1st Column : time, others : particle(s) number ";

                for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                    ( *this->saveGnuplotF ) << this->indices.getValue()[i] << " ";
                ( *this->saveGnuplotF ) << endl;
            }
            else
            {
                ( *this->saveGnuplotF ) << "# Gnuplot File : resultant of the forces of "
                        << this->indices.getValue().size() << " particle(s) Monitored"
                        << endl;
                ( *this->saveGnuplotF ) << "# 1st Column : time, other : resultant force "
                        << endl;
            }
        }

    }
}
////////////////////////// end initGnuplot () /////////////////////////////



///////////////////////// exportGnuplot () ////////////////////////////////
template<class DataTypes>
void ExtraMonitor<DataTypes>::exportGnuplot ( Real time )
{
    if ( this->saveXToGnuplot.getValue() )
    {
        ( *this->saveGnuplotX ) << time <<"\t" ;

        if ((minX.getValue() == -1) && (maxX.getValue() == -1))
        {
            for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                ( *this->saveGnuplotX ) << (*this->X)[this->indices.getValue()[i]] << "\t";
            ( *this->saveGnuplotX ) << endl;
        }
        else
        {
            if (minX.getValue() != -1)
            {
                Real min = numeric_limits<Real>::max();
                Real displ = 0.0;
                for (unsigned i = 0; i < this->indices.getValue().size(); i++)
                {
                    displ = (*this->X)[this->indices.getValue()[i]][minX.getValue()] - initialMinPos[i];
                    displ = fabs(displ); // TODO to be read again
                    if (displ < min)
                        min = displ;
                }
                ( *this->saveGnuplotX ) << displ << "\t";
            }
            if (maxX.getValue() != -1)
            {
                Real max = numeric_limits<Real>::min();
                Real displ = 0.0;
                for (unsigned i = 0; i < this->indices.getValue().size(); i++)
                {
                    displ = (*this->X)[this->indices.getValue()[i]][maxX.getValue()] - initialMaxPos[i];
                    displ = fabs(displ); // TODO to be read again
                    if (displ > max)
                        max = displ;
                }
                ( *this->saveGnuplotX ) << displ;
            }
            ( *this->saveGnuplotX ) << endl;
        }
    }
    if ( this->saveVToGnuplot.getValue() && this->V->size()>0 )
    {
        ( *this->saveGnuplotV ) << time <<"\t";

        for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
            ( *this->saveGnuplotV ) << (*this->V)[this->indices.getValue()[i]] << "\t";
        ( *this->saveGnuplotV ) << endl;
    }

    if ( this->saveFToGnuplot.getValue() && this->F->size()>0)
    {
        ( *this->saveGnuplotF ) << time <<"\t";

        if (!resultantF.getValue())
        {
            for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                ( *this->saveGnuplotF ) << (*this->F)[this->indices.getValue()[i]] << "\t";
            ( *this->saveGnuplotF ) << endl;
        }
        else
        {
            Deriv resultant;
            for (unsigned int i = 0; i < this->indices.getValue().size(); i++)
                resultant += (*this->F)[this->indices.getValue()[i]];

            (*this->saveGnuplotF ) << resultant << endl;
        }
    }
}
///////////////////////////////////////////////////////////////////////////

} // namespace misc

} // namespace component

} // namespace sofa

#endif
