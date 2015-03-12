import QtQuick 2.0
import QtQuick.Controls 1.3
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
                console.error("LOADING ERROR:", contentComponent.errorString());
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

        contextMenu = Qt.createQmlObject("import QtQuick.Controls 1.3; Menu {title: 'Tools'}", root, "contextMenu");
        for(var i = 0; i < contentList.length; ++i)
        {
            contentList[i].parent = loaderLocation;
            var menuItem = contextMenu.addItem(contentList[i].title);
            menuItem.checkable = true;
            menuItem.checked = true;

            var menuSlot = function(content) {return function(checked) {content.visible = checked;}} (contentList[i]);
            menuItem.toggled.connect(menuSlot);
        }
    }

    property Menu contextMenu

    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.RightButton
        onClicked: if(contextMenu) contextMenu.popup()

        ScrollView {
            id: scrollView
            anchors.fill: parent
            //horizontalScrollBarPolicy: Qt.ScrollBarAsNeeded

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
}
