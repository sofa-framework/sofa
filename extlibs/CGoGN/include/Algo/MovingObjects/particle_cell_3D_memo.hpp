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

template <typename PFP>
std::vector<Dart> ParticleCell3DMemo<PFP>::move(const VEC3& newCurrent)
{
    this->crossCell = NO_CROSS ;

//	if(!Geom::arePointsEquals(newCurrent, this->getPosition()))
	{
        CellMarkerMemo<MAP, VOLUME> memo_cross(this->m);
        switch(this->getState()) {
		case VERTEX : vertexState(newCurrent,memo_cross); break;
		case EDGE : 	edgeState(newCurrent,memo_cross);   break;
		case FACE : 	faceState(newCurrent,memo_cross);   break;
		case VOLUME : volumeState(newCurrent,memo_cross);   break;
		}
		return memo_cross.get_markedCells();
	}
//	else
//        this->Algo::MovingObjects::ParticleBase<PFP>::move(newCurrent) ;

	std::vector<Dart> res;
	res.push_back(this->d);
	return res;
}

template <typename PFP>
std::vector<Dart> ParticleCell3DMemo<PFP>::move(const VEC3& newCurrent, CellMarkerMemo<MAP, VOLUME>& memo_cross)
{
    this->crossCell = NO_CROSS ;

//	if(!Geom::arePointsEquals(newCurrent, this->getPosition()))
	{
        switch(this->getState()) {
		case VERTEX : vertexState(newCurrent,memo_cross); break;
		case EDGE : 	edgeState(newCurrent,memo_cross);   break;
		case FACE : 	faceState(newCurrent,memo_cross);   break;
		case VOLUME : volumeState(newCurrent,memo_cross);   break;
		}
		return memo_cross.get_markedCells();
	}
//	else
//        this->Algo::MovingObjects::ParticleBase<PFP>::move(newCurrent) ;

	std::vector<Dart> res;
	res.push_back(this->d);
	return res;
}


/**when the ParticleCell3D trajectory go through a vertex
*  searching the good volume "umbrella" where the ParticleCell3D is
*  if the ParticleCell3D is on the vertex, do nothing */
template <typename PFP>
void ParticleCell3DMemo<PFP>::vertexState(const VEC3& current, CellMarkerMemo<MAP, VOLUME>& memo_cross)
{
    #ifdef DEBUG
    std::cout << "vertexStateMemo" << this->d <<  std::endl;
    #endif
    if(!memo_cross.isMarked(this->d)) memo_cross.mark(this->d);
    this->crossCell = CROSS_OTHER ;

    VEC3 som = this->position[this->d];

    if(Geom::arePointsEquals(current, som)) {
#ifdef DEBUG
        std::cout << "points equal vrai :" << current << " || "<<this->m_position<<" || "<<this->d<<" || "<<this->position[this->d]<< std::endl;
#endif
        this->m_position = som;
        this->m_positionFace = som;
        this->setState(VERTEX);
        return;
    }

    //sinon on tourne autour du sommet pour chercher le bon volume a interroger
    // on interroge les volumes sans tenter de préciser plus a cause des faces non convexes
    Dart depart =this->d;
    bool reset =false;
    do
    {
        if(reset)
        {
            reset=false;
            depart=this->d;
        }
        switch (this->orientationPlan(current,this->position[this->d],this->position[this->m.phi1(this->d)],this->position[this->m.phi_1(this->d)]))
        {
            case Geom::UNDER : this->d=this->m.phi3(this->d); reset =true;
                            break;
            default : this->d=this->m.phi1(this->m.phi2(this->d)); // over ou ON on tourne sur les faces
                            break;

        }

    }while(this->d!=depart);

    this->reset_positionFace();
    this->m_position = this->m_positionFace ;
    volumeState(current,memo_cross);





//    Dart dd=this->d;
//    Geom::Orientation3D wsof;
//    CellMarkerStore<MAP, FACE> mark(this->m);

//    do {
//        VEC3 dualsp = (som+ Algo::Surface::Geometry::vertexNormal<PFP>(this->m,this->d,this->position));
//        Dart ddd=this->d;

//        mark.mark(this->d);

//        //searching the good orientation in a volume
//        if(this->isLeftENextVertex(current,this->d,dualsp)!=Geom::UNDER) {

//            this->d=this->m.phi1(this->m.phi2(this->d));
//            while(this->isLeftENextVertex(current,this->d,dualsp)!=Geom::UNDER && ddd!=this->d)
//                this->d=this->m.phi1(this->m.phi2(this->d));

//            if(ddd==this->d) {
//                if(isnan(current[0]) || isnan(current[1]) || isnan(current[2])) {
//                    std::cout << __FILE__ << " " << __LINE__ << " NaN !" << std::endl;
//                }

//                bool verif = true;
//                do {
//                    if(this->whichSideOfFace(current,this->d)!=Geom::OVER)
//                        verif = false;
//                    else
//                        this->d=this->m.alpha1(this->d);
//                } while(verif && this->d!=ddd);

//                if(verif) {
//                    this->reset_positionVolume();
//                    volumeState(current,memo_cross);
//                    return;
//                }
//            }
//        }
//        else {
//            while(this->isRightVertex(current,this->d,dualsp) && dd!=this->m.alpha_1(this->d))
//                this->d=this->m.phi2(this->m.phi_1(this->d));
//        }

//        wsof = this->whichSideOfFace(current,this->d);

//        //if c is before the vertex on the edge, we have to change of umbrella, we are symetric to the good umbrella
//        if(wsof != Geom::OVER)
//        {
//            VEC3 p1=this->position[this->d];
//            VEC3 p2=this->position[this->m.phi1(this->d)];
//            VEC3 norm(p2-p1);
//            norm.normalize();
//            Geom::Plane3D<typename PFP::REAL> plane(norm,p1);

//            wsof = plane.orient(current);
//        }

//        //if c is on the other side of the face, we have to change umbrella
//        //if the umbrella has already been tested, we just take another arbitrary one
//        if(wsof == 1)
//        {
//            mark.mark(this->d);

//            if(!mark.isMarked(this->m.alpha1(this->d)))
//                this->d=this->m.alpha1(this->d);
//            else
//            {
//                Dart dtmp=this->d;
//                this->d = this->nextDartOfVertexNotMarked(this->d,mark);
//                if(dtmp==this->d) {
//                    std::cout << "numerical rounding ?" << std::endl;

//                    this->d = dd;
//                    this->reset_positionFace();
//                    this->reset_positionVolume();
//                    this->m_position = this->m_positionFace ;
//                    volumeState(current,memo_cross);
//                    return;
//                }
//            }
//        }
//    } while(wsof == 1);

//    if(wsof != 0)
//    {
//        this->reset_positionFace();
//        this->reset_positionVolume();
//        this->m_position = this->m_positionFace ;
//        volumeState(current,memo_cross);
//    }
//    else
//    {
//        //Dart ddd=d;
//        edgeState(current, memo_cross);
//    }
}


template <typename PFP>
void ParticleCell3DMemo<PFP>::edgeState(const VEC3& current, CellMarkerMemo<MAP, VOLUME>& memo_cross)
{
     if(!memo_cross.isMarked(this->d)) memo_cross.mark(this->d);
#ifdef DEBUG
    std::cout << "edgeStateMemo" <<  this->d <<  " " << this->m_position<<  " " << this->m_positionFace << std::endl;

#endif

    this->crossCell = CROSS_OTHER ;

    Dart dd=this->d;
    this->reset_positionFace();
    bool aDroite=false;
    bool aGauche=false;
    bool onEdge=false;
    // on recherche dans quel secteur on se trouve en calculant son orientation apr rapport a des plans
    // jusqu'a etre a gauche d'un plan et a droite du suivant
    do {

        switch(this->whichSideOfFace(current,this->d))// on tourne pour trouver le volume ou on se trouve
        {

            case Geom::UNDER : aDroite=true; if(!aGauche){ this->d=this->m.alpha2(this->d); this->reset_positionFace();}
                            break;
            case Geom::OVER :aGauche=true;this->d=this->m.alpha_2(this->d);this->reset_positionFace();
                            break;
            default : //ON
                            switch (this->whichSideOfEdge(current,this->d)) {
                            case Geom::OVER :       // on est sur la face en question
                                this->d=this->m.phi1(this->d);
                                 this->reset_positionVolume();
                                faceState(current,memo_cross);
                                return;
                            case Geom::ON :// on est sur l'arete, il faut tester les sommets
                                onEdge=true;
                                break;
                            default : //UNDER  on est de l'autre coté de l'arête, il faut commencer de tourner (on ne peut pas finir ici sauf si on commence en face)
                                if(aGauche||aDroite) CGoGNout<< "PROBLEME |||||||||||||||||||||||||||||||||||||"<<CGoGNendl;
                                aGauche=true;this->d=this->m.alpha_2(this->d); this->reset_positionFace();
                                break;
                            }
                            break;
        }
    } while ((!aDroite||!aGauche)&&this->d!=dd&&!onEdge);

    if(this->d==dd && (!aDroite || !aGauche )&&!onEdge)
    {
        CGoGNout<<"on a fait un tour sans trouver ..."<<__LINE__<<__FILE__<<CGoGNendl;
        return;
    }

    if(onEdge)
    {   // on est sur l'arete visée
        if(this->isOnHalfEdge(current,this->d)) //
        {
            if (this->isOnHalfEdge(current,this->m.phi2(this->d))) { // sur l'arete
                this->setState(EDGE);
                this->m_position  = current;
            }
            else {  // sur le sommet de l'arete
                this->d=this->m.phi1(this->d);
                vertexState(current,memo_cross);
            }
        }
        else {  // sur l'autre sommet de l'arete
            vertexState(current,memo_cross);
        }
    }
    else { // on a trouvé le camembert et on est dans le volume

//        d = nextNonPlanar(m.phi1(d));
        this->d=this->m.phi1(this->d);
        this->m_position = this->m_positionFace ;
        volumeState(current,memo_cross);
        return;
    }
}

 template <typename PFP>
 void ParticleCell3DMemo<PFP>::faceState(const VEC3& current, CellMarkerMemo<MAP, VOLUME>& memo_cross, Geom::Orientation3D wsof)
{
	#ifdef DEBUG
    std::cout << "faceStateMemo" <<  this->d <<  " " << this->m_position<<  " " << this->m_positionFace<< std::endl;
	#endif
     if(!memo_cross.isMarked(this->d)) memo_cross.mark(this->d);


    this->reset_positionFace();
    this->crossCell = CROSS_FACE ;

    Dart dd=this->d;
    bool aDroite=false;
    bool aGauche=false;
    // on recherche dans quel secteur on se trouve en calculant son orientation apr rapport a des plans
    // jusqu'a etre a gauche d'un plan et a droite du suivant
    do {
        switch(this->orientationPlan(current,this->position[this->d],this->m_positionFace,this->m_positionVolume))
        {

            case Geom::OVER : aDroite=true; if(!aGauche)this->d=this->m.phi1(this->d);
                            break;
            case Geom::UNDER :aGauche=true;this->d=this->m.phi_1(this->d);
                            break;
            case Geom::ON :
                            do
                            {  // on est obligé de faire le tour car il y a plusieurs cas possible quand on est sur le plan
                                switch (this->whichSideOfEdge(current,this->d)) {
                                case Geom::OVER : this->d=this->m.phi_1(this->d);
                                    break;
                                default :
                                    this->m_position=this->position[this->d]; // on est sur le sommet ou de l'autre coté, dans tous les cas c'est au vertex de voir
                                    vertexState(current,memo_cross);
                                    return;
                                }
                            } while(this->d!=dd);

                            this->m_position = this->m_positionFace = current;
                            this->setState(FACE);
                            return;
            default : CGoGNout<<"erreur detection orientation"<<__LINE__<<__FILE__<<CGoGNendl; break;

        }
    } while ((!aDroite||!aGauche)&&this->d!=dd);

    if(this->d==dd && (!aDroite || !aGauche ))
    {
        CGoGNout<<"on a fait un tour sans trouver ..."<<this->d<<" "<<aDroite<<" "<<aGauche<<" "<<this->orientationPlan(current,this->position[this->d],this->m_positionFace,this->m_positionVolume)<<__LINE__<<__FILE__<<CGoGNendl;
        return;
    }
   // ici on a trouvé le secteur de la face qui contient l'objectif



        wsof = this->whichSideOfFace(current,this->d);

    switch(wsof) // on peut maintenant tester si on est bien sur la face ou dans un des volumes
    {
        case Geom::OVER : // dans l'autre volume
            this->d = this->m.phi3(this->d);

            volumeState(current,memo_cross);
            break;

        case Geom::UNDER : // dans le volume
            volumeState(current,memo_cross);

            break;

        default : // sur la face et dans le secteur
                switch (this->whichSideOfEdge(current,this->d))
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
                        this->m_position = (this->position[this->m.phi1(this->d)]+this->position[this->d])/2;

                        edgeState(current,memo_cross);
                        break;
                }
                break;

    }
}

template <typename PFP>
void ParticleCell3DMemo<PFP>::volumeState(const VEC3& current, CellMarkerMemo<MAP, VOLUME>& memo_cross)
{
	#ifdef DEBUG
    std::cout << "volumeStateMemo " <<  this->d <<  " " << this->m_position<<  " " << this->m_positionFace<< std::endl;
	#endif
    if(!memo_cross.isMarked(this->d)) memo_cross.mark(this->d);
    this->reset_positionVolume();
    bool casON=false;
    bool enDessous=false;

    this->placeOnRightFaceAndRightEdge(current,&casON,&enDessous);
    Geom::Orientation3D wsof =this->whichSideOfFace(current,this->d);
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
                  this->setState(VERTEX);
                }
                else
                {
                  this->setState(EDGE);
                }

            }
            return;
        default : // over l'obj est de l'autre coté de la face
            this->m_position=this->m_positionFace;
            if(enDessous)
            {
                this->reset_positionFace();
                faceState(current,memo_cross,wsof);
            }
            else
            {
                if(casON)
                {
                  vertexState(current,memo_cross);
                }
                else
                {
                  edgeState(current,memo_cross);
                }
            }
            return;
    }
}


} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
