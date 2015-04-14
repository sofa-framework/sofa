/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

// #define DEBUG

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace MovingObjects
{

//#define DELTA 0.00001
//static const float DELTA=0.00001;


template <typename PFP>
void ParticleCell3D<PFP>::display()
{
    //	std::cout << "position : " << this->m_position << std::endl;
}

template <typename PFP>
void ParticleCell3D<PFP>:: reset_positionFace()
{
    if(this->face_center)
    {
        this->m_positionFace=(*face_center)[d];
    }
    else
    {
        this->m_positionFace=Algo::Surface::Geometry::faceCentroid<PFP>(m,d,position);
    }

}

template <typename PFP>
void ParticleCell3D<PFP>:: reset_positionVolume()
{
    if(this->volume_center)
    {
        this->m_positionVolume=(*volume_center)[d];
    }
    else
    {
        this->m_positionVolume=Algo::Surface::Geometry::volumeCentroid<PFP>(m,d,position);

    }
}

template <typename PFP>
void ParticleCell3D<PFP>::resetParticule(){
    VEC3 oldPos=this->m_position;
    reset_positionFace();
    reset_positionVolume();
    CGoGN::Algo::MovingObjects::ParticleBase<PFP>::move(m_positionVolume);

#ifdef DEBUG
    CGoGNout<<"part moved to centroid ,d : "<< this->getPosition()<<" || "<<this->getCell()<<CGoGNendl;

#endif
    this->setState(VOLUME) ;
    move(oldPos);
}


template <typename PFP>
inline Geom::Orientation3D ParticleCell3D<PFP>::whichSideOfFace(const VEC3& c, Dart da) //renvoie la position par rapport au plan défini apr le triangle défini par l'arete visée et le centre de la face
{
        Dart da2 = m.phi1(da);
        Geom::Plane3D<typename PFP::REAL> plan(position[da],position[da2],this->m_positionFace);
#ifdef DEBUG
    std::cout << "Test side of Face (obj,d,position[d], test)" <<c<<" || "<<da<<" || "<<position[da]<<" ||" <<plan.orient(c)<< std::endl;
#endif
//    return Algo::Surface::Geometry::facePlane<PFP>(m,da,position).orient(c);

    return plan.orient(c);
}
template <typename PFP>
Geom::Orientation3D ParticleCell3D<PFP>::orientationPlan(const VEC3& c,const VEC3& p1, const VEC3& p2, const VEC3& p3)
{
     return Geom::Plane3D<typename PFP::REAL>(p3, p1, p2).orient(c);
}


//template <typename PFP>
//int ParticleCell3D<PFP>::whichSideOfPlan(const VEC3& c, Dart da, const VEC3& base, const VEC3& normal) // orientation par rapport au plan de gauche de l'arete visée
//{
//    const VEC3 v2(position[da] -base);
//    const VEC3 np(normal ^ v2);
//#ifdef DEBUG
//    std::cout << "Test plan Face (obj,d,position[d], positionFace, test)" <<c<<" || "<<da<<" || "<<position[da]<<" || "<<base<<" || "<<Geom::Plane3D<typename PFP::REAL>(np,base).orient(c)<< std::endl;
//#endif
//    return Geom::Plane3D<typename PFP::REAL>(np,base).orient(c);

//}

//template <typename PFP>
//Dart ParticleCell3D<PFP>::nextDartOfVertexNotMarked(Dart d, CellMarkerStore<MAP, FACE>& mark)
//{
//    // lock a marker
//    Dart d1;
//    DartMarkerNoUnmark<MAP> markCC(m);

//    // init algo with parameter dart
//    std::list<Dart> darts_list;
//    darts_list.push_back(d);
//    markCC.mark(d);

//    // use iterator for begin of not yet treated darts
//    std::list<Dart>::iterator beg = darts_list.begin();

//    // until all darts treated
//    while (beg != darts_list.end())
//    {
//        d1 = *beg;
//        // add phi1, phi2 and phi3 successor if they are not yet marked
//        Dart d2 = m.phi1(m.phi2(d1));
//        Dart d3 = m.phi1(m.phi3(d1));

//        if (!markCC.isMarked(d2)) {
//            darts_list.push_back(d2);
//            markCC.mark(d2);
//        }

//        if (!markCC.isMarked(d3)) {
//            darts_list.push_back(d3);
//            markCC.mark(d3);
//        }

//        beg++;

//        // apply functor
//        if (!mark.isMarked(d1)) {
//            for (std::list<Dart>::iterator it=darts_list.begin(); it!=darts_list.end(); ++it)
//                markCC.unmark(*it);
//            return d1;
//        }
//    }

//    // clear markers
//    for (std::list<Dart>::iterator it=darts_list.begin(); it!=darts_list.end(); ++it)
//        markCC.unmark(*it);

//    return d;
//}

template <typename PFP>
Geom::Orientation3D ParticleCell3D<PFP>::whichSideOfEdge(const VEC3& c, Dart d) // orientation par rapport au plan orthogonal a la face passant par l'arete visée
{
    VEC3 p1 = position[m.phi1(d)];
    VEC3 p2 = position[d];
    VEC3 p3 = m_positionFace;
    const Geom::Plane3D<typename PFP::REAL> pl (p1,p2,p3);
    VEC3 norm = pl.normal();
    VEC3 n2 = norm ^ VEC3(p1-p2);
#ifdef DEBUG
    std::cout << "Test side of edge (obj,d,position[d], position(phi1), test)" <<c<<" || "<<d<<" || "<<position[d]<<" || "<<position[m.phi1(d)]<<" || "<<Geom::Plane3D<typename PFP::REAL>(n2,p1).orient(c)<< std::endl;
#endif
    return Geom::Plane3D<typename PFP::REAL>(n2,p1).orient(c);
}

template <typename PFP>
bool ParticleCell3D<PFP>::isOnHalfEdge(VEC3 c, Dart d) // booleen : vrai si on est sur l'arete mais pas sur le sommet
{
    VEC3 p1 = position[d];
    VEC3 p2 = position[m.phi1(d)];

    VEC3 norm(p2-p1);
    norm.normalize();

    Geom::Plane3D<typename PFP::REAL> pl(norm,p1);
#ifdef DEBUG
    std::cout << "is on half efge (obj , d, p1, p2, orient,points equal)"<< c <<  d <<  " || " << p1 <<  " || " << p2 <<  " || " << pl.orient(c) <<  " || " << Geom::arePointsEquals(c,p1) << std::endl;

#endif
    return pl.orient(c)==Geom::OVER && !Geom::arePointsEquals(c,p1);
}

/**when the ParticleCell3D trajectory go through a vertex
*  searching the good volume "umbrella" where the ParticleCell3D is
*  if the ParticleCell3D is on the vertex, do nothing */
template <typename PFP>
void ParticleCell3D<PFP>::vertexState(const VEC3& current)
{
#ifdef DEBUG
    std::cout << "vertexState" << d <<  " " <<position[d] <<  " " << this->m_position<<  " " << this->m_positionFace<< std::endl;
#endif

    crossCell = CROSS_OTHER ;

    VEC3 som = position[d];

    if(Geom::arePointsEquals(current, som)) { // si on est sur le sommet on s'arrete
#ifdef DEBUG
        std::cout << "points equal vrai :" << current << " || "<<this->m_position<<" || "<<d<<" || "<<position[d]<< std::endl;
#endif
        this->m_position = som;
        this->m_positionFace = som;
        this->setState(VERTEX);
        return;
    }

    //sinon on tourne autour du sommet pour chercher le bon volume a interroger
    // on interroge les volumes sans tenter de préciser plus a cause des faces non convexes
    Dart depart =d;
    bool reset =false;
    do
    {
        if(reset)
        {
            reset=false;
            depart=d;
        }
        switch (orientationPlan(current,position[d],position[m.phi1(d)],position[m.phi_1(d)]))
        {
            case Geom::UNDER : d=m.phi3(d); reset =true;
                            break;
            default : d=m.phi1(m.phi2(d)); // over ou ON on tourne sur les faces
                            break;

        }

    }while(d!=depart);

    reset_positionFace();
    this->m_position = this->m_positionFace ;
    volumeState(current);

//    Dart dd=d;
//    Geom::Orientation3D wsof;
//    CellMarkerStore<MAP, FACE> mark(m);

//    do {
//        VEC3 dualsp = (som+ Algo::Surface::Geometry::vertexNormal<PFP>(m,d,position));
//        Dart ddd=d;

//        mark.mark(d);

//        //searching the good orientation in a volume
//        if(isLeftENextVertex(current,d,dualsp)!=Geom::UNDER) {

//            d=m.phi1(m.phi2(d));
//            while(isLeftENextVertex(current,d,dualsp)!=Geom::UNDER && ddd!=d)
//                d=m.phi1(m.phi2(d));

//            if(ddd==d) {
//                if(isnan(current[0]) || isnan(current[1]) || isnan(current[2])) {
//                    std::cout << __FILE__ << " " << __LINE__ << " NaN !" << std::endl;
//                }

//                bool verif = true;
//                do {
//                    reset_positionFace();
//                    if(whichSideOfFace(current,d)!=Geom::OVER)
//                        verif = false;
//                    else
//                        d=m.alpha1(d);
//                } while(verif && d!=ddd);

//                if(verif) {
//                    reset_positionVolume();
//                    volumeState(current);
//                    return;
//                }
//            }
//        }
//        else {
//            while(isRightVertex(current,d,dualsp) && dd!=m.alpha_1(d))
//                d=m.phi2(m.phi_1(d));
//        }
//        reset_positionFace();
//        wsof = whichSideOfFace(current,d);

//        //if c is before the vertex on the edge, we have to change of umbrella, we are symetric to the good umbrella
//        if(wsof != Geom::OVER)
//        {
//            VEC3 p1=position[d];
//            VEC3 p2=position[m.phi1(d)];
//            VEC3 norm(p2-p1);
//            norm.normalize();
//            Geom::Plane3D<typename PFP::REAL> plane(norm,p1);

//            wsof = plane.orient(current);
//        }

//        //if c is on the other side of the face, we have to change umbrella
//        //if the umbrella has already been tested, we just take another arbitrary one
//        if(wsof == 1)
//        {
//            mark.mark(d);

//            if(!mark.isMarked(m.alpha1(d)))
//                d=m.alpha1(d);
//            else
//            {
//                Dart dtmp=d;
//                d = nextDartOfVertexNotMarked(d,mark);
//                if(dtmp==d) {
//                    std::cout << "numerical rounding ?" << std::endl;

//                    d = dd;
//                    reset_positionFace();
//                    reset_positionVolume();
//                    this->m_position = this->m_positionFace ;
//                    volumeState(current);
//                    return;
//                }
//            }
//        }
//    } while(wsof == Geom::OVER);

//    if(wsof !=  Geom::ON)
//    {
//        reset_positionFace();
//        reset_positionVolume();
//        this->m_position = this->m_positionFace ;
//        volumeState(current);
//    }
//    else
//    {
//        //Dart ddd=d;
//        edgeState(current);
//    }
}

template <typename PFP>
void ParticleCell3D<PFP>::edgeState(const VEC3& current)
{
#ifdef DEBUG
    std::cout << "edgeState" <<  d <<  " " << this->m_position<<  " " << this->m_positionFace << std::endl;

#endif

    crossCell = CROSS_OTHER ;
    reset_positionFace();
    Dart dd=d;
    bool aDroite=false;
    bool aGauche=false;
    bool onEdge=false;
    // on recherche dans quel secteur on se trouve en calculant son orientation apr rapport a des plans
    // jusqu'a etre a gauche d'un plan et a droite du suivant
    do {
        switch(whichSideOfFace(current,d))// on tourne pour trouver le volume ou on se trouve
        {

            case Geom::UNDER : aDroite=true; if(!aGauche){ d=m.alpha2(d); reset_positionFace(); }
                            break;
            case Geom::OVER :aGauche=true;d=m.alpha_2(d); reset_positionFace();
                            break;
            default : //ON
                            switch (whichSideOfEdge(current,d)) {
                            case Geom::OVER :       // on est sur la face en question
                                d=m.phi1(d);
                                reset_positionVolume();
                                faceState(current);
                                return;
                            case Geom::ON :// on est sur l'arete, il faut tester les sommets
                                onEdge=true;
                                break;
                            default : //UNDER  on est de l'autre coté de l'arête, il faut commencer de tourner (on ne peut pas finir ici sauf si on commence en face)
                                aGauche=true;d=m.alpha_2(d);reset_positionFace();
                                break;
                            }
                            break;
        }
    } while ((!aDroite||!aGauche)&&d!=dd&&!onEdge);

    if(d==dd && (!aDroite || !aGauche )&&!onEdge)
    {
        CGoGNout<<"on a fait un tour sans trouver ..."<<__LINE__<<__FILE__<<CGoGNendl;
        return;
    }

    if(onEdge)
    {   // on est sur l'arete visée
        if(isOnHalfEdge(current,d)) //
        {
            if (isOnHalfEdge(current,m.phi2(d))) { // sur l'arete
                this->setState(EDGE);
                this->m_position  = current;
            }
            else {  // sur le sommet de l'arete
                d=m.phi1(d);
                vertexState(current);
            }
        }
        else {  // sur l'autre sommet de l'arete
            vertexState(current);
        }
    }
    else { // on a trouvé le camembert et on est dans le volume

        d=m.phi1(d);
        this->m_position = this->m_positionFace ;
        volumeState(current);
        return;
    }
}

template <typename PFP>
void ParticleCell3D<PFP>::faceState(const VEC3& current, Geom::Orientation3D wsof)
{
#ifdef DEBUG
    std::cout << "faceState" <<  d <<  " " << this->m_position<<  " " << this->m_positionFace<< std::endl;
#endif
    reset_positionFace();
    crossCell = CROSS_FACE ;

    Dart dd=d;
    bool aDroite=false;
    bool aGauche=false;
    // on recherche dans quel secteur on se trouve en calculant son orientation apr rapport a des plans
    // jusqu'a etre a gauche d'un plan et a droite du suivant
    do {
        switch(orientationPlan(current,position[d],this->m_positionFace,this->m_positionVolume))
        {

            case Geom::OVER : aDroite=true; if(!aGauche)d=m.phi1(d);
                            break;
            case Geom::UNDER :aGauche=true;d=m.phi_1(d);
                            break;
            case Geom::ON :
                            do
                            {  // on est obligé de faire le tour car il y a plusieurs cas possible quand on est sur le plan
                                switch (whichSideOfEdge(current,d)) {
                                case Geom::OVER : d=m.phi_1(d);
                                    break;
                                default :
                                    this->m_position=position[d]; // on est sur le sommet ou de l'autre coté, dans tous les cas c'est au vertex de voir
                                    vertexState(current);
                                    return;
                                }
                            } while(d!=dd);

                            this->m_position = this->m_positionFace = current;
                            this->setState(FACE);
                            return;
            default : CGoGNout<<"erreur detection orientation"<<__LINE__<<__FILE__<<CGoGNendl; break;

        }
    } while ((!aDroite||!aGauche)&&d!=dd);

    if(d==dd && (!aDroite || !aGauche ))
    {
        CGoGNout<<"on a fait un tour sans trouver ..."<<d<<" "<<aDroite<<" "<<aGauche<<" "<<orientationPlan(current,position[d],this->m_positionFace,this->m_positionVolume)<<__LINE__<<__FILE__<<CGoGNendl;
        return;
    }
   // ici on a trouvé le secteur de la face qui contient l'objectif



        wsof = whichSideOfFace(current,d);

    switch(wsof) // on peut maintenant tester si on est bien sur la face ou dans un des volumes
    {
        case Geom::OVER : // dans l'autre volume

            d = m.phi3(d);

            volumeState(current);
            break;

        case Geom::UNDER : // dans le volume
            volumeState(current);

            break;

        default : // sur la face et dans le secteur
                switch (whichSideOfEdge(current,d))
                {
                    case Geom::OVER : // on est sur la face dans ce secteur

                        this->m_position = current;
                        this->setState(FACE);
                        break;
                    case Geom::ON : // on est sur l'edge défini par ce secteur mais pas sur les sommets

                        this->m_position = current;
                        this->setState(EDGE);
                        break;
                    default :   // on est de l'autre côté de l'edge on laisse donc l'edge décider
                        this->m_position = (position[m.phi1(d)]+position[d])/2;

                        edgeState(current);
                }
                break;

    }





}

template <typename PFP>
void ParticleCell3D<PFP>::placeOnRightFaceAndRightEdge(const VEC3& current,bool * casON, bool * enDessous)
{
    Dart dd=d;
    *casON=false;
    *enDessous=false;
    bool aDroite=false;
    bool aGauche=false;
    // on recherche dans quel secteur on se trouve en calculant son orientation apr rapport a des plans
    // jusqu'a etre a gauche d'un plan et a droite du suivant


    do {
        switch(orientationPlan(current,position[d],this->m_positionFace,this->m_positionVolume))
        {

            case Geom::UNDER :
                            aDroite=true; if(!aGauche)d=m.phi1(d);
                            break;
            case Geom::OVER :
                            aGauche=true;d=m.phi_1(d);
                            break;
            case Geom::ON : if(aGauche ||aDroite)//si on est ON des le debut on ne peut pas etre sur de la face
                                *casON=true;
                            else                // donc on tourne dans une direction pour donner un nouveau départ
                                d=m.phi1(d);
                            break;
            default : CGoGNout<<"erreur detection orientation"<<__LINE__<<__FILE__<<CGoGNendl; break;
        }
    } while ((!aDroite||!aGauche)&&d!=dd&&!*casON);

    switch(orientationPlan(current,position[m.phi1(d)],position[d],this->m_positionVolume)) // ensuite on teste par rapport au plan defini par le centre du volume et l'arete visée
    {
        case Geom::OVER : d=m.phi2(d); // si on est au dessus on change de face
                          reset_positionFace();
                          placeOnRightFaceAndRightEdge(current,casON,enDessous);
                          return;
        case Geom::UNDER :*enDessous=true; // si on est en dessous on l'indique
                        break;
        case Geom::ON :         // si on c'est qu'on aura pas enDessous a vrai
                        break;
        default : CGoGNout<<"erreur detection orientation"<<__LINE__<<__FILE__<<CGoGNendl; break;

    }

}


template <typename PFP>
void ParticleCell3D<PFP>::volumeState(const VEC3& current)
{
#ifdef DEBUG
    std::cout << "volumeState " <<  d <<  " " << this->m_position<<  " " << this->m_positionFace<< std::endl;
#endif
    reset_positionVolume();
    bool casON=false;
    bool enDessous=false;

    placeOnRightFaceAndRightEdge(current,&casON,&enDessous);
    Geom::Orientation3D wsof =whichSideOfFace(current,d);
    switch (wsof) {
        case Geom::UNDER :
            this->m_position = current;
            this->setState(VOLUME);
            break;
        case Geom::ON : // on est sur la face
            this->m_position = current;
            if(enDessous)// on teste par rapport a la derniere face du tetra si en dessous on est sur la face, sinon c'est qu'on était "on" et on est sur l'edge
            {
               this->setState(FACE);
            }
            else
            {
                if(casON)
                {
                  this->setState(VERTEX); // on se permet de faire un set si on est sur
                }
                else
                {
                  this->setState(EDGE);// on se permet de faire un set si on est sur
                }

            }
            return;
        default : // over l'obj est de l'autre coté de la face
            this->m_position=m_positionFace;
            if(enDessous)
            {
                faceState(current,wsof);
            }
            else
            {
                if(casON)
                {
                  vertexState(current); // on est de l'autre coté du vertex
                }
                else
                {
                  edgeState(current); // on est de l'autre coté de l'edge
                }
            }
            return;
    }
}


} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
