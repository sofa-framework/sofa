#ifndef PARTCELL_3D_H
#define PARTCELL_3D_H

#include "Algo/MovingObjects/particle_base.h"

#include "Algo/Geometry/inclusion.h"
#include "Geometry/intersection.h"
#include "Geometry/orientation.h"
#include "Geometry/plane_3d.h"
#include "Topology/ihmap/ihm3.h"
#include <iostream>

/* A particle cell is a particle base within a map, within a precise cell, the displacement function should indicate
   after each displacement wherein lies the new position of the particle */

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace MovingObjects
{

enum {
    NO_CROSS,
    CROSS_FACE,
    CROSS_OTHER
};

template <typename PFP>
class ParticleCell3D : public Algo::MovingObjects::ParticleBase<PFP>
{
public :
    typedef Algo::MovingObjects::ParticleBase<PFP> Inherit;
    typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;
//    typedef AttributeHandler<VEC3, VERTEX, MAP> TAB_POS;
    //    typedef AttributeHandler<VEC3, VOLUME, MAP> TAB_VOL;
    //    typedef AttributeHandler<VEC3,FACE, MAP> TAB_FACE;
    typedef typename AttributeHandler_Traits< VEC3, VERTEX, MAP>::Handler TAB_POS;
    typedef typename AttributeHandler_Traits< VEC3, FACE, MAP>::Handler TAB_FACE;
    typedef typename AttributeHandler_Traits< VEC3, VOLUME, MAP>::Handler TAB_VOL;

    MAP& m;

    const TAB_POS& position;
    const TAB_FACE* face_center;
    const TAB_VOL* volume_center;



    VEC3 m_positionFace;
    VEC3 m_positionVolume;


    unsigned int crossCell ;
    bool newVol;
    ParticleCell3D(MAP& map) : m(map)
    {}

    ParticleCell3D(MAP& map, Dart belonging_cell, VEC3 pos, const TAB_POS& tabPos,const TAB_FACE * fa_center = NULL,
    const TAB_VOL * vol_center=NULL) :
        Algo::MovingObjects::ParticleBase<PFP>(pos),
        m(map),
        position(tabPos),
        face_center(fa_center),
        volume_center(vol_center),
        d(belonging_cell)

    {
        newVol = false;
        crossCell=NO_CROSS;
        reset_positionFace();
        reset_positionVolume();
        this->setState(VOLUME);
    }

    void display();

    inline Dart getCell() const
    {
        return d;
    }

    inline void setCell(Dart cell)
    {
        d = cell;
    }

//    inline Geom::Orientation3D isLeftENextVertex(const VEC3& c, Dart d, const VEC3& base);

//    inline bool isRightVertex(const VEC3& c, Dart d, const VEC3& base);

    inline Geom::Orientation3D whichSideOfFace(const VEC3& c, Dart d);

//    inline Geom::Orientation3D whichSideOfPlanVolume(const VEC3& c, Dart d, const VEC3& base, const VEC3& top);

//    inline int whichSideOfPlan(const VEC3& c, Dart d, const VEC3& base, const VEC3& normal); // orientation par rapport au plan de gauche de l'arete visée
    inline Geom::Orientation3D orientationPlan(const VEC3& c,const VEC3& p1, const VEC3& p2, const VEC3& p3);

//    Dart nextDartOfVertexNotMarked(Dart d, CellMarkerStore<MAP, FACE>& mark);

//    Dart nextNonPlanar(Dart d);

//    Dart nextFaceNotMarked(Dart d, CellMarkerStore<MAP, FACE>& mark);

    Geom::Orientation3D whichSideOfEdge(const VEC3& c, Dart d);

    bool isOnHalfEdge(const VEC3& c, Dart d);

    void vertexState(const VEC3& current);

    void edgeState(const VEC3& current);

    void faceState(const VEC3& current, Geom::Orientation3D sideOfFace = Geom::ON);

    void volumeState(const VEC3& current);

    void placeOnRightFaceAndRightEdge(const VEC3& current,bool * casON, bool * enDessous);

    void reset_positionFace(); // remet a jour la positionFace
    void reset_positionVolume(); // remet a jour la positionVolume
    void resetParticuleSimplif(Dart newD); // a appeler pour changer la dart visée par une particule et réinitialiser ses positionsFace et Volume ( en cas de simplif )
    void resetParticuleSubdiv(); // a appeler pour replacer correctement une particule (apres une subdivision par exemple)
    void move(const VEC3& newCurrent)
    {
        crossCell = NO_CROSS ;
        newVol = false;
//        if(!Geom::arePointsEquals(newCurrent, this->getPosition()))
        {
            switch(this->getState()) {
            case VERTEX : vertexState(newCurrent); break;
            case EDGE : 	edgeState(newCurrent);   break;
            case FACE : 	faceState(newCurrent);   break;
            case VOLUME : volumeState(newCurrent);   break;
            }

            display();
        }
    }


protected:
    Dart d;
    Dart lastCrossed;
};

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/MovingObjects/particle_cell_3D.hpp"

#endif
