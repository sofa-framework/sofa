#include <sofa/component/collision/MeshIntTool.h>

namespace sofa
{
namespace component
{
namespace collision
{
using namespace sofa::defaulttype;
using namespace sofa::core::collision;

int MeshIntTool::computeIntersection(Capsule & cap, Point & pnt,double alarmDist,double contactDist,OutputVector* contacts){
    if(doCapPointInt(cap,pnt.p(),alarmDist,contactDist,contacts)){
        DetectionOutput *detection = &*(contacts->end()-1);

        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, pnt);

        return 1;
    }

    return 0;
}


int MeshIntTool::doCapPointInt(Capsule& cap, const Vector3& q,double alarmDist,double contactDist,OutputVector* contacts){
    const Vector3 p1 = cap.point1();
    const Vector3 p2 = cap.point2();
    const Vector3 AB = p2-p1;
    const Vector3 AQ = q -p1;
    double A;
    double b;
    A = AB*AB;
    b = AQ*AB;
    double cap_rad = cap.radius();

    double alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;//projection of the point on the capsule segment such as the projected point P = p1 + AB * alpha
        //if (alpha < 0.000001 || alpha > 0.999999)
        //        return 0;
        if (alpha < 0.0) alpha = 0.0;//if the projection is out the segment, we associate it to a segment apex
        else if (alpha > 1.0) alpha = 1.0;
    }

    Vector3 p,pq;
    p = p1 + AB * alpha;
    pq = q-p;

    double enough_to_touch = alarmDist + cap_rad;
    if (pq.norm2() >= enough_to_touch * enough_to_touch)
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal = pq;

    detection->value = detection->normal.norm();
    detection->normal /= detection->value;

    detection->value -= (contactDist + cap_rad);

    return 1;
}


int MeshIntTool::computeIntersection(Capsule & cap, Line & lin,double alarmDist,double contactDist,OutputVector* contacts)
{
    double cap_rad = cap.radius();
    const Vector3 p1 = cap.point1();
    const Vector3 p2 = cap.point2();
    const Vector3 q1 = lin.p1();
    const Vector3 q2 = lin.p2();

    if(doCapLineInt(p1,p2,cap_rad,q1,q2,alarmDist,contactDist,contacts)){
        OutputVector::iterator detection = contacts->end()-1;
        //detection->id = cap.getCollisionModel()->getSize() > lin.getCollisionModel()->getSize() ? cap.getIndex() : lin.getIndex();
        detection->id = cap.getIndex();
        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, lin);
        return 1;
    }

    return 0;
}


int MeshIntTool::doCapLineInt(const Vector3 & p1,const Vector3 & p2,double cap_rad,
                         const Vector3 & q1, const Vector3 & q2,double alarmDist,double contactDist,OutputVector *contacts, bool ignore_p1, bool ignore_p2){
    const Vector3 AB = p2-p1;//capsule segment
    const Vector3 CD = q2-q1;//line segment
    const Vector3 AC = q1-p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    if (det < -0.000000000001 || det > 0.000000000001)//AB and CD are not on the same plane
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;

        if(alpha < 0)
            alpha = 0;
        else if(alpha > 1)
            alpha = 1;

        if(beta < 0)
            beta = 0;
        else if(beta > 1)
            beta = 1;
    }
    else{//Segments on a same plane. Here the idea to find the nearest points
        //is to project segment apexes on the other segment.
        //Visual example with semgents AB and CD :
        //            A----------------B
        //                     C----------------D
        //After projection :
        //            A--------c-------B
        //                     C-------b--------D
        //So the nearest points are p and q which are respecively in the middle of cB and Cb:
        //            A--------c---p---B
        //                     C---q---b--------D

        Vector3 AD = q2 - p1;
        Vector3 CB = p2 - q1;

        double AB_norm2 = AB.norm2();
        double CD_norm2 = CD.norm2();
        double c_proj= b[0]/AB_norm2;//alpha = (AB * AC)/AB_norm2
        double d_proj = (AB * AD)/AB_norm2;
        double a_proj = b[1];//beta = (-CD*AC)/CD_norm2
        double b_proj= (CD*CB)/CD_norm2;

        if(c_proj >= 0 && c_proj <= 1){//projection of C on AB is lying on AB
            if(d_proj > 1){//case :
                           //             A----------------B
                           //                      C---------------D
                alpha = (1.0 + c_proj)/2.0;
                beta = b_proj/2.0;
            }
            else if(d_proj < 0){//case :
                                //             A----------------B
                                //     D----------------C
                alpha = c_proj/2.0;
                beta = (1 + a_proj)/2.0;
            }
            else{//case :
                //             A----------------B
                //                 C------D
                alpha = (c_proj + d_proj)/2.0;
                beta  = 0.5;
            }
        }
        else if(d_proj >= 0 && d_proj <= 1){
            if(c_proj < 0){//case :
                           //             A----------------B
                           //     C----------------D
                alpha = d_proj /2.0;
                beta = (1 + a_proj)/2.0;
            }
            else{//case :
                 //          A---------------B
                 //                 D-------------C
                alpha = (1 + d_proj)/2.0;
                beta = b_proj/2.0;
            }
        }
        else{
            if(c_proj * d_proj < 0){//case :
                                    //           A--------B
                                    //       D-----------------C
                alpha = 0.5;
                beta = (a_proj + b_proj)/2.0;
            }
            else{
                if(c_proj < 0){//case :
                               //                    A---------------B
                               // C-------------D

                    alpha = 0;
                }
                else{
                    alpha = 1;
                }

                if(a_proj < 0){//case :
                               // A---------------B
                               //                     C-------------D
                }
                else{//case :
                     //                     A---------------B
                     //   C-------------D
                    beta = 1;
                }
            }
        }
    }

    if(ignore_p1 && beta == 0)
        return 0;
    if(ignore_p2 && beta == 1)
        return 0;

    double enough_to_touch = alarmDist + cap_rad;
    Vector3 p,q,pq;
    p = p1 + AB * alpha;
    q = q1 + CD * beta;
    pq = q-p;
    if (pq.norm2() >= enough_to_touch*enough_to_touch)
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    //detection->id = cap.getCollisionModel()->getSize() > lin.getCollisionModel()->getSize() ? cap.getIndex() : lin.getIndex();
    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal=pq;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= (contactDist + cap_rad);

    ///!\ CAUTION : uninitialized fields detection->elem and detection->id

    return 1;
}

int MeshIntTool::doCapLineInt(Capsule & cap,const Vector3 & q1,const Vector3 & q2 ,double alarmDist,double contactDist,OutputVector* contacts, bool ignore_p1, bool ignore_p2)
{
    double cap_rad = cap.radius();
    const Vector3 p1 = cap.point1();
    const Vector3 p2 = cap.point2();

    return doCapLineInt(p1,p2,cap_rad,q1,q2,alarmDist,contactDist,contacts,ignore_p1,ignore_p2);
}


int MeshIntTool::doIntersectionTrianglePoint(double dist2, int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3,const Vector3& q, OutputVector* contacts,bool swapElems)
{
    const Vector3 AB = p2-p1;
    const Vector3 AC = p3-p1;
    const Vector3 AQ = q -p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    //if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        //if (alpha < 0.000001 ||
        //    beta  < 0.000001 ||
        //    alpha + beta  > 0.999999)
        //        return 0;
        if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
        {
            // nearest point is on an edge or corner
            // barycentric coordinate on AB
            double pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
            // barycentric coordinate on AC
            double pAC = b[1] / A[1][1]; // AQ*AB / AB*AB
            if (pAB < 0.000001 && pAC < 0.0000001)
            {
                // closest point is A
                if (!(flags&TriangleModel::FLAG_P1)) return 0; // this corner is not considered
                alpha = 0.0;
                beta = 0.0;
            }
            else if (pAB < 0.999999 && beta < 0.000001)
            {
                // closest point is on AB
                if (!(flags&TriangleModel::FLAG_E12)) return 0; // this edge is not considered
                alpha = pAB;
                beta = 0.0;
            }
            else if (pAC < 0.999999 && alpha < 0.000001)
            {
                // closest point is on AC
                if (!(flags&TriangleModel::FLAG_E12)) return 0; // this edge is not considered
                alpha = 0.0;
                beta = pAC;
            }
            else
            {
                // barycentric coordinate on BC
                // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
                double pBC = (b[1] - b[0] + A[0][0] - A[1][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
                if (pBC < 0.000001)
                {
                    // closest point is B
                    if (!(flags&TriangleModel::FLAG_P2)) return 0; // this edge is not considered
                    alpha = 1.0;
                    beta = 0.0;
                }
                else if (pBC > 0.999999)
                {
                    // closest point is C
                    if (!(flags&TriangleModel::FLAG_P3)) return 0; // this edge is not considered
                    alpha = 0.0;
                    beta = 1.0;
                }
                else
                {
                    // closest point is on BC
                    if (!(flags&TriangleModel::FLAG_E31)) return 0; // this edge is not considered
                    alpha = 1.0-pBC;
                    beta = pBC;
                }
            }
        }
    }

    Vector3 p, pq;
    p = p1 + AB * alpha + AC * beta;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq;
    }
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;

    ///!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value, you have to substract contactDist

    return 1;
}


int MeshIntTool::computeIntersection(Capsule& cap, Triangle& tri,double alarmDist,double contactDist,OutputVector* contacts){
    const int tri_flg = tri.flags();

    int id = cap.getIndex();
    int n = 0;

    const Vector3 cap_p1 = cap.point1();
    const Vector3 cap_p2 = cap.point2();
    double cap_rad = cap.radius();
    double dist2 = (alarmDist + cap_rad) * (alarmDist + cap_rad);

    const Vector3 tri_p1 = tri.p1();
    const Vector3 tri_p2 = tri.p2();
    const Vector3 tri_p3 = tri.p3();

    double substract_dist = contactDist + cap_rad;
    n += doIntersectionTrianglePoint(dist2,tri_flg,tri_p1,tri_p2,tri_p3,cap_p1,contacts,true);
    n += doIntersectionTrianglePoint(dist2,tri_flg,tri_p1,tri_p2,tri_p3,cap_p2,contacts,true);

    if(n == 2){
        OutputVector::iterator detection1 = contacts->end() - 2;
        OutputVector::iterator detection2 = contacts->end() - 1;

        if(detection1->value > detection2->value - 1e-15 && detection1->value < detection2->value + 1e-15){
            detection1->point[0] = (detection1->point[0] + detection2->point[0])/2.0;
            detection1->point[1] = (detection1->point[1] + detection2->point[1])/2.0;
            detection1->normal = (detection1->normal + detection2->normal)/2.0;
            detection1->value = (detection1->value + detection2->value)/2.0 - substract_dist;
            detection1->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);

            contacts->pop_back();
            n = 1;
        }
        else{
            for(OutputVector::iterator detection = contacts->end() - n; detection != contacts->end() ; ++detection){
                detection->value -= substract_dist;
                detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
                detection->id = id;
            }
        }
    }
    else{
        for(OutputVector::iterator detection = contacts->end() - n; detection != contacts->end() ; ++detection){
            detection->value -= substract_dist;
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
            detection->id = id;
        }
    }

    int old_n = n;
    n = 0;

    if (tri_flg&TriangleModel::FLAG_E12)
        n += doCapLineInt(cap_p1,cap_p2,cap_rad,tri_p1,tri_p2,alarmDist,contactDist,contacts,!(tri_flg&TriangleModel::FLAG_P1),!(tri_flg&TriangleModel::FLAG_P2));
    if (tri_flg&TriangleModel::FLAG_E23)
        n += doCapLineInt(cap_p1,cap_p2,cap_rad,tri_p2,tri_p3,alarmDist,contactDist,contacts,!(tri_flg&TriangleModel::FLAG_P2),!(tri_flg&TriangleModel::FLAG_P3));
    if (tri_flg&TriangleModel::FLAG_E31)
        n += doCapLineInt(cap_p1,cap_p2,cap_rad,tri_p3,tri_p1,alarmDist,contactDist,contacts,!(tri_flg&TriangleModel::FLAG_P3),!(tri_flg&TriangleModel::FLAG_P1));

    for(OutputVector::iterator detection = contacts->end()-n ; detection != contacts->end() ; ++detection){
        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, tri);
        detection->id = id;
    }

    return n + old_n;
}


int MeshIntTool::computeIntersection(Triangle& tri,int flags,OBB & obb,double alarmDist,double contactDist,OutputVector* contacts){
    IntrTriangleOBB intr(tri,obb);
    if(intr.Find(alarmDist,flags)){
        OBB::Real dist2 = (intr.pointOnFirst() - intr.pointOnSecond()).norm2();
        if((!intr.colliding()) && dist2 > alarmDist * alarmDist)
            return 0;

        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);

        detection->normal = intr.separatingAxis();
        detection->point[0] = intr.pointOnFirst();
        detection->point[1] = intr.pointOnSecond();

        if(intr.colliding())
            detection->value = -helper::rsqrt(dist2) - contactDist;
        else
            detection->value = helper::rsqrt(dist2) - contactDist;

        detection->elem.first = tri;
        detection->elem.second = obb;
        //detection->id = (tri.getCollisionModel()->getSize() > obb.getCollisionModel()->getSize()) ? tri.getIndex() : obb.getIndex();
        detection->id = tri.getIndex();

        return 1;
    }

    return 0;
}

int MeshIntTool::projectPointOnTriangle(int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3, Vector3 & to_be_projected)
{
    const Vector3 AB = p2-p1;
    const Vector3 AC = p3-p1;
    const Vector3 AQ = to_be_projected -p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    //if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        //if (alpha < 0.000001 ||
        //    beta  < 0.000001 ||
        //    alpha + beta  > 0.999999)
        //        return 0;
        if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
        {
            // nearest point is on an edge or corner
            // barycentric coordinate on AB
            double pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
            // barycentric coordinate on AC
            double pAC = b[1] / A[1][1]; // AQ*AB / AB*AB
            if (pAB < 0.000001 && pAC < 0.0000001)
            {
                // closest point is A
                if (!(flags&TriangleModel::FLAG_P1)) return 0; // this corner is not considered
                alpha = 0.0;
                beta = 0.0;
            }
            else if (pAB < 0.999999 && beta < 0.000001)
            {
                // closest point is on AB
                if (!(flags&TriangleModel::FLAG_E12)) return 0; // this edge is not considered
                alpha = pAB;
                beta = 0.0;
            }
            else if (pAC < 0.999999 && alpha < 0.000001)
            {
                // closest point is on AC
                if (!(flags&TriangleModel::FLAG_E12)) return 0; // this edge is not considered
                alpha = 0.0;
                beta = pAC;
            }
            else
            {
                // barycentric coordinate on BC
                // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
                double pBC = (b[1] - b[0] + A[0][0] - A[1][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
                if (pBC < 0.000001)
                {
                    // closest point is B
                    if (!(flags&TriangleModel::FLAG_P2)) return 0; // this edge is not considered
                    alpha = 1.0;
                    beta = 0.0;
                }
                else if (pBC > 0.999999)
                {
                    // closest point is C
                    if (!(flags&TriangleModel::FLAG_P3)) return 0; // this edge is not considered
                    alpha = 0.0;
                    beta = 1.0;
                }
                else
                {
                    // closest point is on BC
                    if (!(flags&TriangleModel::FLAG_E31)) return 0; // this edge is not considered
                    alpha = 1.0-pBC;
                    beta = pBC;
                }
            }
        }
    }

    to_be_projected = p1 + AB * alpha + AC * beta;

    return 1;
}

class SOFA_MESH_COLLISION_API MeshIntTool;

}
}
}
