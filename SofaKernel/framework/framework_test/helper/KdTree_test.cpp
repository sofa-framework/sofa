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
#include <limits>
#include <sofa/helper/kdTree.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/random.h>

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;


namespace sofa {


/**  Test suite for KdTree.
 * @author Benjamin Gilles
 * @date 2014
 */

struct KdTreeTest: public BaseTest
{
    typedef SReal Real;
    typedef defaulttype::Vec<3,Real> Coord;
    typedef std::vector<Coord> VecCoord;

    typedef sofa::helper::kdTree<Coord> kdT;
    typedef kdT::distanceToPoint distanceToPoint;
    typedef kdT::distanceSet distanceSet;


    void generateRandomPoint(VecCoord &position,const unsigned int nbp, const Real range)
    {
        position.clear();
        position.reserve(nbp);
        for(unsigned int i=0;i<nbp;i++) position.push_back(Coord(Real(helper::drand(range)),Real(helper::drand(range)),Real(helper::drand(range))));
    }

    /// brute detection = gold standard
    void getClosetNPoints(distanceSet &cl,const Coord& p, const VecCoord &position,const unsigned int N)
    {
        for(unsigned int i=0;i<position.size();i++)
        {
            Real d=(p-position[i]).norm2();
            cl.insert(distanceToPoint(d,i));
            if(cl.size()>N) {distanceSet::iterator it=cl.end(); it--; cl.erase(it);}
        }
    }

    /// test if kdtree finds the right closest point for source to target
    void testPointPointCorrespondences(const unsigned int nbp_source, const unsigned int nbp_target,const Real range)
    {
        VecCoord sourceposition;
        generateRandomPoint(sourceposition,nbp_source,range);
        VecCoord targetposition;
        generateRandomPoint(targetposition,nbp_target,range);

        kdT KDT;
        KDT.build(targetposition);

        for(unsigned int i=0;i<nbp_source;i++)
        {
            unsigned int closest_kdt=KDT.getClosest(sourceposition[i],targetposition);
            distanceSet closest_brute; getClosetNPoints(closest_brute,sourceposition[i],targetposition,1);
            ASSERT_EQ( closest_brute.begin()->second , closest_kdt);
        }
    }

    /// test if kdtree finds the right N closest points for source to target
    void testPointNPointsCorrespondences(const unsigned int nbp_source, const unsigned int nbp_target,const Real range, const unsigned int N)
    {
        VecCoord sourceposition;
        generateRandomPoint(sourceposition,nbp_source,range);
        VecCoord targetposition;
        generateRandomPoint(targetposition,nbp_target,range);

        kdT KDT;
        KDT.build(targetposition);

        for(unsigned int i=0;i<nbp_source;i++)
        {
            distanceSet closest_kdt; KDT.getNClosest(closest_kdt, sourceposition[i],targetposition,N);
            distanceSet closest_brute; getClosetNPoints(closest_brute,sourceposition[i],targetposition,N);
            distanceSet::iterator closestKdt=closest_kdt.begin();
            for(distanceSet::iterator closestBrute=closest_brute.begin();closestBrute!=closest_brute.end();++closestBrute)
            {
                ASSERT_EQ( closestBrute->second , closestKdt->second);
                closestKdt++;
            }
        }
    }

    /// move a source point nb times and test if distance caching works to find the right closest point to target
    void testCachedPointPointCorrespondences(const unsigned int nb, const unsigned int nbp_target,const Real range, const Real dprange, const unsigned int N)
    {
        VecCoord sourceposition;
        generateRandomPoint(sourceposition,1,range);
        VecCoord sourcedisplacement;
        generateRandomPoint(sourcedisplacement,nb,dprange);
        VecCoord targetposition;
        generateRandomPoint(targetposition,nbp_target,range);

        kdT KDT;
        KDT.build(targetposition);

        Coord x = sourceposition[0],previousX;
        distanceSet closest_kdt;
        distanceToPoint cacheThresh[2];
        for(unsigned int i=0;i<nb;i++)
        {
            x+=sourcedisplacement[i];
            KDT.getNClosestCached(closest_kdt,cacheThresh[0],cacheThresh[1],previousX,x,targetposition,N);
            distanceSet closest_brute; getClosetNPoints(closest_brute,x,targetposition,1);
            ASSERT_EQ( closest_brute.begin()->second , closest_kdt.begin()->second);
        }


    }

};

TEST_F(KdTreeTest, point_point ) {    testPointPointCorrespondences(100,100,10); }
TEST_F(KdTreeTest, point_Npoints ) {   testPointNPointsCorrespondences(100,100,10,10); }
TEST_F(KdTreeTest, cached_point_point ) {   testCachedPointPointCorrespondences(100,100,10,0.5,5); }


} // namespace sofa
