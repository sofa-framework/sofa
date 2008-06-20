/****************************************************************************
*																			*
*		Copyright: See COPYING file that comes with this distribution		*
*																			*
****************************************************************************/
#ifndef SOFA_COMPONENT_MISC_MONITOR_INL
#define SOFA_COMPONENT_MISC_MONITOR_INL

#include <sofa/component/misc/Monitor.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/Data.h>
#include <iostream>
#include <fstream>



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
    ,saveGnuplotX ( NULL ), saveGnuplotV ( NULL ), saveGnuplotF ( NULL )
{}
/////////////////////////// end Monitor ///////////////////////////////////



////////////////////////////// ~Monitor ///////////////////////////////////
template <class DataTypes>
Monitor<DataTypes>::~Monitor()
{
    delete ( saveGnuplotX );
    delete ( saveGnuplotV );
    delete ( saveGnuplotF );
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
