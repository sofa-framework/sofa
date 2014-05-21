import QtQuick 2.0
import Qt.labs.folderlistmodel 2.0

ListView  {
    id: fileBrowser
    width: 200; height: 400

    FolderListModel  {
        id: folderModel
        nameFilters: ["*.xml *.scn *.pscn *.py *.simu"]
    }

    Component  {
        id: fileDelegate
        Text  { text: fileName }
    }

    model: folderModel
    delegate: fileDelegate
    property string basePath: "../examples"
}
