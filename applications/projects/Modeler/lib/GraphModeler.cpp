/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v16.08                  *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//History of operations management
//TODO: not use the factory to create the elements!
bool GraphModeler::cut(std::string path)
{
    bool resultCopy=copy(path);
    if (resultCopy)
    {
        deleteComponent();
        return true;
    }
    return false;
}

bool GraphModeler::copy(std::string path)
{
    helper::vector< QTreeWidgetItem*> items; getSelectedItems(items);

    if (!items.empty())
    {
        saveComponents(items,path);
        return true;
    }
    return false;
}

bool GraphModeler::paste(std::string path)
{
    helper::vector< QTreeWidgetItem*> items;
    getSelectedItems(items);
    if (!items.empty())
    {
        //Get the last item of the node: the new items will be inserted AFTER this last item.
        QTreeWidgetItem *last=items.front();
        //while(last->nextSibling()) last=last->nextSibling();
        last = last->child(last->childCount()-1);

        Node *node = getNode(items.front());
        //Load the paste buffer
        loadNode(node, path);
        QTreeWidgetItem *pasteItem=items.front();

        //Find all the QListViewItem inserted
        helper::vector< QTreeWidgetItem* > insertedItems;
//        QTreeWidgetItem *insertedItem=last;
        for(int i=0 ; i<last->parent()->childCount() ; i++)
        {
            QTreeWidgetItem *insertedItem = last->parent()->child(i);
            if(insertedItem != last)
                insertedItems.push_back(insertedItem);
        }

        //Initialize their position in the node
        for (unsigned int i=0; i<insertedItems.size(); ++i) initItem(insertedItems[i], pasteItem);
    }

    return !items.empty();
}
}
}
}
