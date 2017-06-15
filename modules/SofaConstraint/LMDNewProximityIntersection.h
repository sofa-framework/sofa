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
#ifndef SOFA_COMPONENT_COLLISION_LMDNEWPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_LMDNEWPROXIMITYINTERSECTION_H
#include "config.h"

#include <SofaBaseCollision/BaseProximityIntersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaUserInteraction/RayModel.h>

namespace sofa
{

namespace component
{

namespace collision
{


/// I guess LMD is for Local Min Distance?
/// What is the specificity of this approach?
/// What are exactly doing the filters?
class SOFA_CONSTRAINT_API LMDNewProximityIntersection : public BaseProximityIntersection
{
public:
    SOFA_CLASS(LMDNewProximityIntersection,BaseProximityIntersection);

    Data<bool> useLineLine;
protected:
    LMDNewProximityIntersection();
public:
    virtual void init();

    /// Returns true if algorithm uses proximity
    virtual bool useProximity() const { return true; }

    bool testIntersection(Cube& ,Cube&);
    bool testIntersection(Point&, Point&);
    template<class T> bool testIntersection(TSphere<T>&, Point&);
    template<class T1, class T2> bool testIntersection(TSphere<T1>&, TSphere<T2>&);
    bool testIntersection(Line&, Point&);
    template<class T> bool testIntersection(Line&, TSphere<T>&);
    bool testIntersection(Line&, Line&);
    bool testIntersection(Triangle&, Point&);
    template<class T> bool testIntersection(Triangle&, TSphere<T>&);
    bool testIntersection(Triangle&, Line&);
    bool testIntersection(Triangle&, Triangle&);
    bool testIntersection(Ray&, Triangle&);

    int computeIntersection(Cube&, Cube&, OutputVector*);
    int computeIntersection(Point&, Point&, OutputVector*);
    template<class T> int computeIntersection(TSphere<T>&, Point&, OutputVector*);
    template<class T1, class T2> int computeIntersection(TSphere<T1>&, TSphere<T2>&, OutputVector*);
    int computeIntersection(Line&, Point&, OutputVector*);
    template<class T> int computeIntersection(Line&, TSphere<T>&, OutputVector*);
    int computeIntersection(Line&, Line&, OutputVector*);
    int computeIntersection(Triangle&, Point&, OutputVector*);
    template<class T> int computeIntersection(Triangle&, TSphere<T>&, OutputVector*);
    int computeIntersection(Triangle&, Line&, OutputVector*);
    int computeIntersection(Triangle&, Triangle&, OutputVector*);
    int computeIntersection(Ray&, Triangle&, OutputVector*);

    template< class TFilter1, class TFilter2 >
    static inline int doIntersectionLineLine(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q1, const defaulttype::Vector3& q2, OutputVector* contacts, int id, int indexLine1, int indexLine2, TFilter1 &f1, TFilter2 &f2);

    template< class TFilter1, class TFilter2 >
    static inline int doIntersectionLinePoint(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q, OutputVector* contacts, int id, int indexLine1, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems = false);

    template< class TFilter1, class TFilter2 >
    static inline int doIntersectionPointPoint(double dist2, const defaulttype::Vector3& p, const defaulttype::Vector3& q, OutputVector* contacts, int id, int indexPoint1, int indexPoint2, TFilter1 &f1, TFilter2 &f2);

    template< class TFilter1, class TFilter2 >
    static inline int doIntersectionTrianglePoint(double dist2, int flags, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& p3, const defaulttype::Vector3& n, const defaulttype::Vector3& q, OutputVector* contacts, int id, Triangle &e1, unsigned int *edgesIndices, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems = false);


//	static inline int doIntersectionLineLine(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q1, const defaulttype::Vector3& q2, OutputVector* contacts, int id);

//	static inline int doIntersectionLinePoint(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q, OutputVector* contacts, int id, bool swapElems = false);

//	static inline int doIntersectionPointPoint(double dist2, const defaulttype::Vector3& p, const defaulttype::Vector3& q, OutputVector* contacts, int id);

//	static inline int doIntersectionTrianglePoint(double dist2, int flags, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& p3, const defaulttype::Vector3& n, const defaulttype::Vector3& q, OutputVector* contacts, int id, bool swapElems = false);


    /**
     * @brief Method called at the beginning of the collision detection between model1 and model2.
     * Checks if LMDFilters are associated to the CollisionModels.
     * @TODO Optimization.
     */
//	int beginIntersection(TriangleModel* /*model1*/, TriangleModel* /*model2*/, DiscreteIntersection::OutputVector* /*contacts*/);

protected:
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_COLLISION_LMDNEWPROXIMITYINTERSECTION_H
