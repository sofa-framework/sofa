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
void GraphListenerQListView::addDatas(sofa::core::objectmodel::BaseObject *parent)
{
    if (frozen) return;
    QTreeWidgetItem* new_item;
    std::string name;
    BaseData* data = NULL;
    if(items.count(parent))
    {
        const sofa::core::objectmodel::Base::VecData& fields = parent->getDataFields();
        for( sofa::core::objectmodel::Base::VecData::const_iterator it = fields.begin();
                it!=fields.end();
                ++it)
        {
            data = (*it);
            if(!datas.count(data))
            {
                static QPixmap pixData((const char**)icondata_xpm);
                new_item = createItem(items[parent]);
                name += "  ";
                name += data->getName();
                datas.insert(std::pair<BaseData*,QTreeWidgetItem*>(data,new_item));
                new_item->setText(0, name.c_str());
                new_item->setIcon(0, QIcon(pixData));
//                widget->ensureItemVisible(new_item);
                widget->scrollToItem(new_item);
                name.clear();
            }
        }
    }
}




} //qt
} //gui
} //sofa
