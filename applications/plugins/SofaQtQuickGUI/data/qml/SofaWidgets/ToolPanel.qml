import QtQuick 2.0
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.1
import SofaBasics 1.0
import SofaTools 1.0
import Qt.labs.folderlistmodel 2.1

Rectangle {
    id: root
    clip: true
    color: "lightgrey"

    property Scene scene

    FolderListModel {
        id: folderListModel
        nameFilters: ["*.qml"]
        showDirs: false
        showFiles: true
        sortField: FolderListModel.Name

        Component.onCompleted: {
            folder = "qrc:/SofaTools";
        }

        onCountChanged: refresh()
    }

    function refresh() {
        for(var i = 0; i < loaderLocation.children.length; ++i)
            loaderLocation.children[i].destroy();

        var contentList = [];
        for(var i = 0; i < folderListModel.count; ++i) {
            var source = "qrc" + folderListModel.get(i, "filePath");

            var contentComponent = Qt.createComponent(source);
            if(contentComponent.status === Component.Error) {
                console.log("LOADING ERROR:", contentComponent.errorString());
            } else {
                contentList.push(contentComponent.createObject(root, {"Layout.fillWidth": true, "scene": scene}));
            }
        }

        contentList.sort(function(a, b) {
            if(undefined !== a.priority && undefined !== b.priority)
                return a.priority > b.priority ? -1 : 1;

            if(undefined === a.priority && undefined !== b.priority)
                return 1;

            if(undefined !== a.priority && undefined === b.priority)
                return -1;

            return 0;
        });

        for(var i = 0; i < contentList.length; ++i)
            contentList[i].parent = loaderLocation;
    }

    ScrollView {
        id: scrollView
        anchors.fill: parent

        Flickable {
            id: flickable
            anchors.fill: parent
            contentHeight: loaderLocation.implicitHeight

            ColumnLayout {
                id: loaderLocation
                width: parent.width - 1
                spacing: 0
            }
        }
    }
}
