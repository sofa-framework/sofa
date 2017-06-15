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
#ifndef SOFA_HELPER_UnitTest_H
#define SOFA_HELPER_UnitTest_H

#include <iostream>
#include <sstream>
#include <limits>
#include <cstdlib>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace helper
{


/** Base class for performing series of unit tests and issuing test reports.

  There are three types of reports: the total number of unit tests, the number of warnings, and the number of errors.
  Warnings can be used as alternative to errors.

  Log and warning/error messages can be output using output streams given by  sout() and serr(), respectively.
  The messages are displayed or not, depending on the verbosity level.

  To implement a new test, derive this class and implement method void runTests( unsigned& numTests, unsigned& numWarnings, unsigned& numErrors ).
  */
class UnitTest
{

public:
    typedef enum {SILENT,WARNINGS_ONLY,ALL_MESSAGES} VerbosityLevel;

    UnitTest( std::string shortTestName, VerbosityLevel verb );
    virtual ~UnitTest() {}

    /** Run a series of unit tests.
      Increment @param numTests with the number of unit tests performed.
      Increment @param numWarnings with the number of warnings issued.
      Increment @param numErrors with the number of errors detected.
      For each unit test, method checkIf(bool testSucceeded, std::string testDescription, unsigned& numTests, unsigned& numErrors)
        can be used to automatically increment numTests and numErrors, as well as issuing log and error messages.
    */
    virtual void runTests( unsigned& numTests, unsigned& numWarnings, unsigned& numErrors )=0;

protected:
    /** Helper to register the result of unit test.
      @param testSucceeded is the result of the test, true if successfull
      @param testDescription is the description of the test, printed depending on the test result and verbosity level
      @param numTests is incremented by 1
      @param numErrors is incremented if testSucceeded is false
      */
    bool checkIf( bool testSucceeded, std::string testDescription, unsigned& numTests, unsigned& numErrors );

    /// Basic output stream, displayed only if the verbosity level is ALL_MESSAGES
    virtual std::ostream& sout() { std::ostream& s = verbosityLevel()>WARNINGS_ONLY ? std::cerr: skippedMsgs; s<<getName()<<": "; return s;}

    /// Error and warning output stream, skipped only if the verbosity level is SILENT
    virtual std::ostream& serr() { std::ostream& s = verbosityLevel()>SILENT ? std::cerr: skippedMsgs;  s<<getName()<<": "; return s;}

    VerbosityLevel verbosityLevel() { return verbose; }
    const std::string& getName() const { return name; }

    /// General-purpose initializations, such as the random number generator.
    static void initClass();

    /** @name Helpers
     *  Helper Functions to more easily create tests and check the results.
     */
    //@{
    /// A very small value. Can be used to check if an error is small enough.
    virtual double epsilon() const { return 1.0e-10; }

    /// random double in [0,1[. It is recommended to use initClass() to initialize the random number generator before using this method.
    static double drand() { return (double)rand() / RAND_MAX; }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    template<typename Matrix1, typename Matrix2>
    static bool matricesAreEqual( const Matrix1& m1, const Matrix2& m2, double tolerance=std::numeric_limits<double>::epsilon() )
    {
        if(m1.rowSize()!=m2.rowSize() || m2.colSize()!=m1.colSize()) return false;
        for( unsigned i=0; i<m1.rowSize(); i++ )
            for( unsigned j=0; j<m1.colSize(); j++ )
                if( fabs(m1.element(i,j)-m2.element(i,j))>tolerance  ) return false;
        return true;
    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    template< typename Vector1, typename Vector2>
    static bool vectorsAreEqual( const Vector1& m1, const Vector2& m2, double tolerance=std::numeric_limits<double>::epsilon() )
    {
        if( m1.size()!=m2.size() ) return false;
        for( unsigned i=0; i<m1.size(); i++ )
            if( fabs(m1.element(i)-m2.element(i))>tolerance  ) return false;
        return true;
    }



    //@}

private:
    VerbosityLevel verbose;  ///< Condition for printing test names, comments and results of the tests.
    std::string name;     ///< Test name, preferably short. Detailed test descriptions can be given to the detectErrors method.
    std::ostringstream  skippedMsgs;  ///< Contains all the skipped messages, depending on the verbosity level.

};







} // namespace helper

} // namespace sofa

#endif
