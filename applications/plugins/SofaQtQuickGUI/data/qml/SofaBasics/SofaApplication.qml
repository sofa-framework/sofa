pragma Singleton
import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.3
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0
import SofaInteractors 1.0
import Qt.labs.folderlistmodel 2.1
import "qrc:/SofaCommon/SofaCommonScript.js" as SofaCommonScript

QtObject {
    id: root

////////////////////////////////////////////////// PRIVATE
    property QtObject d: QtObject {

        property var sceneListModel: null

    }

////////////////////////////////////////////////// SCENE

    readonly property var scene: Scene {}

//    // create a sceneListModel only if needed
//    function sceneListModel() {
//        if(null === d.sceneListModel) {
//            console.log("+");
//            d.sceneListModel = SofaCommonScript.InstanciateComponent(SceneListModel, root, {"scene": root.scene});
//            console.log("-");
//        }

//        return d.sceneListModel;
//    }

////////////////////////////////////////////////// SETTINGS

    //property UISettings uiSettings: UISettings {}
    //property RecentSettings uiSettings: RecentSettings {}

////////////////////////////////////////////////// INTERACTOR

    property string defaultInteractorName: "Selection"
    readonly property string interactorName: {
        if(interactorComponent)
            for(var key in interactorComponentMap)
                if(interactorComponentMap.hasOwnProperty(key) && interactorComponent === interactorComponentMap[key])
                    return key;

        return "";
    }

    property Component interactorComponent: null
    property var interactorComponentMap: null

    property var interactorFolderListModel: FolderListModel {
        id: interactorFolderListModel
        nameFilters: ["*.qml"]
        showDirs: false
        showFiles: true
        sortField: FolderListModel.Name

        Component.onCompleted: refresh();
        onCountChanged: update();

        property var refreshOnSceneLoaded: Connections {
            target: root.scene
            onLoaded: interactorFolderListModel.refresh();
        }

        function refresh() {
            folder = "";
            folder = "qrc:/SofaInteractors";
        }

        function update() {
            if(root.interactorComponentMap)
                for(var key in root.interactorComponentMap)
                    if(root.interactorComponentMap.hasOwnProperty(key))
                        root.interactorComponentMap[key].destroy();

            var interactorComponentMap = [];
            for(var i = 0; i < count; ++i)
            {
                var fileBaseName = get(i, "fileBaseName");
                var filePath = get(i, "filePath").toString();
                if(-1 !== folder.toString().indexOf("qrc:"))
                    filePath = "qrc" + filePath;

                var name = fileBaseName.slice(fileBaseName.indexOf("_") + 1);
                var interactorComponent = Qt.createComponent(filePath);
                interactorComponentMap[name] = interactorComponent;
            }

            if(null === root.interactorComponent)
                if(interactorComponentMap.hasOwnProperty(root.defaultInteractorName))
                    root.interactorComponent = interactorComponentMap[root.defaultInteractorName];

            if(null === root.interactorComponent)
                for(var key in interactorComponentMap)
                    if(interactorComponentMap.hasOwnProperty(key)) {
                        root.interactorComponent = interactorComponentMap[key];
                        break;
                    }

            root.interactorComponentMap = interactorComponentMap;
        }
    }

//////////////////////////////////////////////////


}
