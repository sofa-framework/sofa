#include <sofa/component/collision/CubeModel.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <GL/gl.h>
#include <algorithm>
#include <math.h>


namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Cube)

using namespace sofa::defaulttype;

void create(CubeModel*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    obj = new CubeModel;
    if (obj!=NULL)
    {
        obj->setStatic(atoi(arg->getAttribute("static","0"))!=0);
        //if (arg->getAttribute("scale")!=NULL)
        //	obj->applyScale(atof(arg->getAttribute("scale","1.0")));
        //if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
        //	obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
    }
}

Creator<simulation::tree::xml::ObjectFactory, CubeModel > CubeModelClass("Cube");

CubeModel::CubeModel()
    : static_(false)
{
}

void CubeModel::resize(int size)
{
    int size0 = this->size;
    if (size == size0) return;
    // reset parent
    CollisionModel* parent = getPrevious();
    while(parent != NULL)
    {
        parent->resize(0);
        parent = parent->getPrevious();
    }
    this->core::CollisionModel::resize(size);
    this->elems.resize(size);
    this->parentOf.resize(size);
    // set additional indices
    for (int i=size0; i<size; ++i)
    {
        this->elems[i].leaf = core::CollisionElementIterator(getNext(), i);
        this->parentOf[i] = i;
    }
}

void CubeModel::setParentOf(int childIndex, const Vector3& min, const Vector3& max)
{
    int i = parentOf[childIndex];
    elems[i].minBBox = min;
    elems[i].maxBBox = max;
}

int CubeModel::addCube(Cube subcellsBegin, Cube subcellsEnd)
{
    int i = size;
    this->core::CollisionModel::resize(size+1);
    elems.resize(size+1);
    elems[i].subcells = std::make_pair(subcellsBegin, subcellsEnd);
    elems[i].leaf = core::CollisionElementIterator();
    updateCube(i);
    return i;
}

void CubeModel::updateCube(int index)
{
    const std::pair<Cube,Cube>& subcells = elems[index].subcells;
    if (subcells.first != subcells.second)
    {
        Cube c = subcells.first;
        Vector3 minBBox = c.minVect();
        Vector3 maxBBox = c.maxVect();
        ++c;
        while(c != subcells.second)
        {
            const Vector3& cmin = c.minVect();
            const Vector3& cmax = c.maxVect();
            for (int j=0; j<3; j++)
            {
                if (cmax[j] > maxBBox[j]) maxBBox[j] = cmax[j];
                if (cmin[j] < minBBox[j]) minBBox[j] = cmin[j];
            }
            ++c;
        }
        elems[index].minBBox = minBBox;
        elems[index].maxBBox = maxBBox;
    }
}

void CubeModel::updateCubes()
{
    for (int i=0; i<size; i++)
        updateCube(i);
}

void CubeModel::draw(int index)
{
    const Vector3& vmin = elems[index].minBBox;
    const Vector3& vmax = elems[index].maxBBox;

    glBegin(GL_LINES);
    {
        glVertex3d(vmin[0], vmin[1], vmin[2]);
        glVertex3d(vmin[0], vmin[1], vmax[2]);
        glVertex3d(vmin[0], vmax[1], vmin[2]);
        glVertex3d(vmin[0], vmax[1], vmax[2]);
        glVertex3d(vmax[0], vmin[1], vmin[2]);
        glVertex3d(vmax[0], vmin[1], vmax[2]);
        glVertex3d(vmax[0], vmax[1], vmin[2]);
        glVertex3d(vmax[0], vmax[1], vmax[2]);

        glVertex3d(vmin[0], vmin[1], vmin[2]);
        glVertex3d(vmin[0], vmax[1], vmin[2]);
        glVertex3d(vmin[0], vmin[1], vmax[2]);
        glVertex3d(vmin[0], vmax[1], vmax[2]);
        glVertex3d(vmax[0], vmin[1], vmin[2]);
        glVertex3d(vmax[0], vmax[1], vmin[2]);
        glVertex3d(vmax[0], vmin[1], vmax[2]);
        glVertex3d(vmax[0], vmax[1], vmax[2]);

        glVertex3d(vmin[0], vmin[1], vmin[2]);
        glVertex3d(vmax[0], vmin[1], vmin[2]);
        glVertex3d(vmin[0], vmax[1], vmin[2]);
        glVertex3d(vmax[0], vmax[1], vmin[2]);
        glVertex3d(vmin[0], vmin[1], vmax[2]);
        glVertex3d(vmax[0], vmin[1], vmax[2]);
        glVertex3d(vmin[0], vmax[1], vmax[2]);
        glVertex3d(vmax[0], vmax[1], vmax[2]);
    }
    glEnd();
}

void CubeModel::draw()
{
    if (!isActive() || !((getNext()==NULL)?getContext()->getShowCollisionModels():getContext()->getShowBoundingCollisionModels())) return;
    glDisable(GL_LIGHTING);
    int level=0;
    CollisionModel* m = getPrevious();
    float color = 1.0f;
    while (m!=NULL)
    {
        m = m->getPrevious();
        ++level;
        color *= 0.5f;
    }
    if (isStatic())
        glColor4f(1.0f, 1.0f, 1.0f, color);
    else
        glColor4f(1.0f, 1.0f, 0.0f, color);
    if (color < 1.0f)
    {
        glEnable(GL_BLEND);
        glDepthMask(0);
    }
    for (int i=0; i<size; i++)
    {
        draw(i);
    }
    if (color < 1.0f)
    {
        glDisable(GL_BLEND);
        glDepthMask(1);
    }
    if (getPrevious()!=NULL && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

std::pair<core::CollisionElementIterator,core::CollisionElementIterator> CubeModel::getInternalChildren(int index) const
{
    return elems[index].subcells;
}

std::pair<core::CollisionElementIterator,core::CollisionElementIterator> CubeModel::getExternalChildren(int index) const
{
    core::CollisionElementIterator i1 = elems[index].leaf;
    if (!i1.valid())
    {
        return std::make_pair(core::CollisionElementIterator(),core::CollisionElementIterator());
    }
    else
    {
        core::CollisionElementIterator i2 = i1; ++i2;
        return std::make_pair(i1,i2);
    }
}

bool CubeModel::isLeaf( int index ) const
{
    return elems[index].leaf.valid();
}

class CubeModel::CubeSortPredicate
{
    int axis;
public:
    CubeSortPredicate(int axis) : axis(axis) {}
    bool operator()(const CubeData& c1,const CubeData& c2) const
    {
        double v1 = c1.minBBox[axis]+c1.maxBBox[axis];
        double v2 = c2.minBBox[axis]+c2.maxBBox[axis];
        return v1 < v2;
    }
    template<int Axis>
    static int sortCube(const void* p1, const void* p2)
    {
        const CubeModel::CubeData* c1 = (const CubeModel::CubeData*)p1;
        const CubeModel::CubeData* c2 = (const CubeModel::CubeData*)p2;
        double v1 = c1->minBBox[Axis] + c1->maxBBox[Axis];
        double v2 = c2->minBBox[Axis] + c2->maxBBox[Axis];

        if (v1 < v2)
            return -1;
        else if (v1 > v2)
            return 1;
        else
            return 0;
    }
};

void CubeModel::computeBoundingTree(int maxDepth)
{
    //std::cout << ">CubeModel::computeBoundingTree("<<maxDepth<<")"<<std::endl;
    std::list<CubeModel*> levels;
    levels.push_front(createPrevious<CubeModel>());
    for (int i=0; i<maxDepth; i++)
        levels.push_front(levels.front()->createPrevious<CubeModel>());
    CubeModel* root = levels.front();
    if (isStatic() && root->getPrevious() == NULL && !root->empty()) return; // No need to recompute BBox if immobile

    if (root->empty() || root->getPrevious() != NULL)
    {
        // Tree must be reconstructed
        //std::cout << "Building Tree with depth "<<maxDepth<<" from "<<size<<" elements.\n";
        // First remove extra levels
        while(root->getPrevious()!=NULL)
        {
            core::CollisionModel* m = root->getPrevious();
            root->setPrevious(m->getPrevious());
            delete m;
        }
        // Then clear all existing levels
        {
            for (std::list<CubeModel*>::iterator it = levels.begin(); it != levels.end(); ++it)
                (*it)->resize(0);
        }
        // Then build root cell
        //std::cout << "CubeModel: add root cube"<<std::endl;
        root->addCube(Cube(this,0),Cube(this,size));
        // Construct tree by splitting cells along their biggest dimension
        std::list<CubeModel*>::iterator it = levels.begin();
        CubeModel* level = *it;
        ++it;
        int lvl = 0;
        while(it != levels.end())
        {
            //std::cout << "CubeModel: split level "<<lvl<<std::endl;
            CubeModel* clevel = *it;
            for(Cube cell = Cube(level->begin()); level->end() != cell; ++cell)
            {
                std::pair<Cube,Cube> subcells = cell.subcells();
                int ncells = subcells.second.getIndex() - subcells.first.getIndex();
                if (ncells > 4)
                {
                    // Only split cells with more than 4 childs
                    // Find the biggest dimension
                    int splitAxis;
                    Vector3 l = cell.maxVect()-cell.minVect();
                    int middle = subcells.first.getIndex()+(ncells+1)/2;
                    if(l[0]>l[1])
                        if (l[0]>l[2])
                            splitAxis = 0;
                        else
                            splitAxis = 2;
                    else if (l[1]>l[2])
                        splitAxis = 1;
                    else
                        splitAxis = 2;

                    // Separate cells on each side of the median cell

#if defined(GCC_VERSION) && GCC_VERSION == 40101
                    // there is apparently a bug in std::sort with GCC 4.1.1
                    if (splitAxis == 0)
                        qsort(&(elems[subcells.first.getIndex()]), subcells.second.getIndex()-subcells.first.getIndex(), sizeof(elems[0]), CubeSortPredicate::sortCube<0>);
                    else if (splitAxis == 1)
                        qsort(&(elems[subcells.first.getIndex()]), subcells.second.getIndex()-subcells.first.getIndex(), sizeof(elems[0]), CubeSortPredicate::sortCube<1>);
                    else
                        qsort(&(elems[subcells.first.getIndex()]), subcells.second.getIndex()-subcells.first.getIndex(), sizeof(elems[0]), CubeSortPredicate::sortCube<2>);
#else
                    CubeSortPredicate sortpred(splitAxis);
                    //std::nth_element(elems.begin()+subcells.first.getIndex(),elems.begin()+middle,elems.begin()+subcells.second.getIndex(), sortpred);
                    std::sort(elems.begin()+subcells.first.getIndex(),elems.begin()+subcells.second.getIndex(), sortpred);
#endif

                    // Create the two new subcells
                    Cube cmiddle(this, middle);
                    int c1 = clevel->addCube(subcells.first, cmiddle);
                    int c2 = clevel->addCube(cmiddle, subcells.second);
                    level->elems[cell.getIndex()].subcells = std::make_pair(Cube(clevel,c1),Cube(clevel,c2+1));
                    //std::cout << "L"<<lvl<<" cell "<<cell.getIndex()<<" split along "<<(splitAxis==0?'X':splitAxis==1?'Y':'Z')<<" in cell "<<c1<<" size "<<middle-subcells.first.getIndex()<<" and cell "<<c2<<" size "<<subcells.second.getIndex()-middle<<".\n";
                }
            }
            ++it;
            level = clevel;
            ++lvl;
        }
        // Finally update parentOf to reflect new cell order
        for (int i=0; i<size; i++)
            parentOf[elems[i].leaf.getIndex()] = i;
    }
    else
    {
        // Simply update the existing tree, starting from the bottom
        int lvl = 0;
        for (std::list<CubeModel*>::reverse_iterator it = levels.rbegin(); it != levels.rend(); it++)
        {
            //std::cout << "CubeModel: update level "<<lvl<<std::endl;
            (*it)->updateCubes();
            ++lvl;
        }
    }
    //std::cout << "<CubeModel::computeBoundingTree("<<maxDepth<<")"<<std::endl;
}

} // namespace collision

} // namespace component

} // namespace sofa

