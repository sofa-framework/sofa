import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0
import "qrc:/SofaCommon/SofaSettingsScript.js" as SofaSettingsScript
import "qrc:/SofaCommon/SofaToolsScript.js" as SofaToolsScript

Item {
    id: root
    clip: true

    property int uiId: 0
    property int previousUiId: uiId
    onUiIdChanged: {
        SofaSettingsScript.Ui.replace(previousUiId, uiId);
    }

    property real splitterThickness: 1
    property real splitterMagnetizeThreshold: 5
    property real splitterMarginThreshold: 30
    property real splittingMarginThreshold: 10

    property url source
    property Component sourceComponent

    Settings {
        id: uiSettings
        category: 0 !== root.uiId ? "ui_" + root.uiId : "dummy"

        property string splitterIds
        property string viewIds
    }

    property bool isLoaded: false
    function load() {
        if(0 === uiId)
            return;

        for(var i = 0; i < root.children.length; ++i) {
            var item = root.children[i];
            if(!item.isSplitter && !item.isView)
                continue;

            item.load();
        }

        for(var i = 0; i < root.children.length; ++i) {
            var item = root.children[i];
            if(!item.isSplitter && !item.isView)
                continue;

            item.init();
        }

        isLoaded = true;
    }

    Component.onCompleted: {
        if(0 === uiId) {
            uiId = SofaSettingsScript.Ui.generate();
        } else {
            // try to load a previous configuration
            var viewIdArray = uiSettings.viewIds.split(';');
            for(var i = 0; i < viewIdArray.length; ++i) {
                if(0 === viewIdArray[i].length)
                    continue;

                createView({"uiId": viewIdArray[i]});
            }

            var splitterIdArray = uiSettings.splitterIds.split(';');
            for(var i = 0; i < splitterIdArray.length; ++i) {
                if(0 === splitterIdArray[i].length)
                    continue;

                createSplitter({"uiId": splitterIdArray[i]});
            }

            load();
        }

        // use the default settings if there is no saved one
        if(0 === children.length) {
            var view = createView();

            /*width = 100;
            height = 100;

            var splitter = splitView(view, Qt.Horizontal, Qt.point(0, 0));
            splitter.relativeX = 0.75;

            for(var i = 0; i < root.children.length; ++i) {
                var item = root.children[i];
                if(!item.isSplitter && !item.isView)
                    continue;

                if(item.isView) {
                    var loaderItem = item.item;
                    if(loaderItem.isDynamicContent) {
                        loaderItem.defaultContentName = "ToolPanel";
                        break;
                    }
                }
            }*/
        }
    }

    onChildrenChanged: {
        if(!isLoaded)
            return;

        uiSettings.splitterIds = "";
        uiSettings.viewIds = "";

        for(var j = 0; j < root.children.length; ++j) {
            var item = root.children[j];

            if(item.isSplitter)
                uiSettings.splitterIds += item.uiId + ";";

            if(item.isView)
                uiSettings.viewIds += item.uiId + ";";
        }
    }

    onWidthChanged: {
        for(var i = 0; i < root.children.length; ++i) {
            var item = root.children[i];
            if(!item.isSplitter)
                continue;

            var splitter = item;
            if(Qt.Vertical === splitter.orientation)
                splitter.x = width * splitter.relativeX;
        }

        updateSplitters();
    }

    onHeightChanged: {
        for(var i = 0; i < root.children.length; ++i) {
            var item = root.children[i];
            if(!item.isSplitter)
                continue;

            var splitter = item;
            if(Qt.Horizontal === splitter.orientation)
                splitter.y = height * splitter.relativeY;
        }

        updateSplitters();
    }

    function getSplitterByUiId(uiId) {
        if(0 === uiId)
            return null;

        for(var i = 0; i < children.length; ++i) {
            var item = children[i];
            if(!item.isSplitter)
                continue;

            if(uiId === item.uiId)
                return item;
        }

        return null;
    }

    function getViewByUiId(uiId) {
        if(0 === uiId)
            return null;

        for(var i = 0; i < children.length; ++i) {
            var item = children[i];
            if(!item.isView)
                continue;

            if(uiId === item.uiId)
                return item;
        }

        return null;
    }

    function createView(properties) {
        if(undefined === properties)
            properties = {};

        var view = viewComponent.createObject(root, properties);
        var corner = cornerComponent.createObject(root, {"view": view});

        view.corner = corner;

        return view;
    }

    function splitView(view, orientation, position) {
        var splitter = null;

        if(Qt.Horizontal === orientation) {
            splitter = createSplitter({"x": position.x, "topLeftEdge": view.topEdge, "bottomRightEdge": view.bottomEdge, "orientation": Qt.Vertical});

            var newView;
            if(position.x > view.x + view.width /2) {
                newView = createView({"topEdge": view.topEdge, "bottomEdge": view.bottomEdge, "leftEdge": splitter, "rightEdge": view.rightEdge});
                view.rightEdge = splitter;
            } else {
                newView = createView({"topEdge": view.topEdge, "bottomEdge": view.bottomEdge, "leftEdge": view.leftEdge, "rightEdge": splitter});
                view.leftEdge = splitter;
            }
        } else {
            splitter = createSplitter({"y": position.y, "topLeftEdge": view.leftEdge, "bottomRightEdge": view.rightEdge, "orientation": Qt.Horizontal});

            var newView;
            if(position.y < view.y + view.height /2) {
                newView = createView({"topEdge": view.topEdge, "bottomEdge": splitter, "leftEdge": view.leftEdge, "rightEdge": view.rightEdge});
                view.topEdge = splitter;
            } else {
                newView = createView({"topEdge": splitter, "bottomEdge": view.bottomEdge, "leftEdge": view.leftEdge, "rightEdge": view.rightEdge});
                view.bottomEdge = splitter;
            }
        }

        return splitter;
    }

    function evaluateViewMerging(view, location) {
        var mergingItem = childAt(location.x, location.y);
        if(!mergingItem)
            return;

        var mergingView;
        if(mergingItem.isCorner) {
            var mergingCorner = mergingItem;
            mergingView = mergingCorner.view;
        } else if(mergingItem.isView) {
            mergingView = mergingItem;
        }

        if(!view || !view.isView || !mergingView || !mergingView.isView || view === mergingView)
            return false;

        if(view.topEdge && view.topEdge === mergingView.bottomEdge && view.leftEdge === mergingView.leftEdge && view.rightEdge === mergingView.rightEdge)
            return true;

        if(view.bottomEdge && view.bottomEdge === mergingView.topEdge && view.leftEdge === mergingView.leftEdge && view.rightEdge === mergingView.rightEdge)
            return true;

        if(view.rightEdge && view.rightEdge === mergingView.leftEdge && view.bottomEdge === mergingView.bottomEdge && view.topEdge === mergingView.topEdge)
            return true;

        if(view.leftEdge && view.leftEdge === mergingView.rightEdge && view.bottomEdge === mergingView.bottomEdge && view.topEdge === mergingView.topEdge)
            return true;

        return false;
    }

    function mergeView(view, location) {
        if(!view || !view.isView)
            return;

        var mergingItem = childAt(location.x, location.y);
        if(!mergingItem)
            return;

        var mergingView;
        if(mergingItem.isCorner) {
            var mergingCorner = mergingItem;
            mergingView = mergingCorner.view;
        } else if(mergingItem.isView) {
            mergingView = mergingItem;
        }

        if(!mergingView || !mergingView.isView || view === mergingView)
            return;

        // merge vertically - bottom to top
        if(view.topEdge && view.topEdge === mergingView.bottomEdge && view.leftEdge === mergingView.leftEdge && view.rightEdge === mergingView.rightEdge) {
            // splitter is shared only by the two views
            var splitter = view.topEdge;
            if(splitter.topLeftEdge === view.leftEdge && splitter.bottomRightEdge === view.rightEdge) {
                splitter.destroyByUser();
                view.topEdge = mergingView.topEdge;
                mergingView.destroyByUser();
            } else {
                if(splitter.topLeftEdge === view.leftEdge) { // splitter is shared by the views and also by another view on the right
                    splitter.topLeftEdge = view.rightEdge;
                    view.topEdge = mergingView.topEdge;
                    mergingView.destroyByUser();
                } else if(splitter.bottomRightEdge === view.rightEdge) { // splitter is shared by two views and also by another view on the left
                    splitter.bottomRightEdge = view.leftEdge;
                    view.topEdge = mergingView.topEdge;
                    mergingView.destroyByUser();
                } else { // splitter is shared by two views and also by other views on the left and on the right
                    view.topEdge = mergingView.topEdge;
                    var leftSplitter = createSplitter({"y": splitter.y, "topLeftEdge": splitter.topLeftEdge, "bottomRightEdge": view.leftEdge, "orientation": splitter.orientation});
                    var rightSplitter = createSplitter({"y": splitter.y, "topLeftEdge": view.rightEdge, "bottomRightEdge": splitter.bottomRightEdge, "orientation": splitter.orientation});
                    replaceWithSuitableSplitter(splitter, leftSplitter, rightSplitter);
                    mergingView.destroyByUser();
                    splitter.destroyByUser();
                }
            }
        }

        // merge vertically - top to bottom
        if(view.bottomEdge && view.bottomEdge === mergingView.topEdge && view.leftEdge === mergingView.leftEdge && view.rightEdge === mergingView.rightEdge) {
            // splitter is shared only by the two views
            var splitter = view.bottomEdge;
            if(splitter.topLeftEdge === view.leftEdge && splitter.bottomRightEdge === view.rightEdge) {
                splitter.destroyByUser();
                view.bottomEdge = mergingView.bottomEdge;
                mergingView.destroyByUser();
            } else {
                if(splitter.topLeftEdge === view.leftEdge) { // splitter is shared by the views and also by another view on the right
                    splitter.topLeftEdge = view.rightEdge;
                    view.bottomEdge = mergingView.bottomEdge;
                    mergingView.destroyByUser();
                } else if(splitter.bottomRightEdge === view.rightEdge) { // splitter is shared by two views and also by another view on the left
                    splitter.bottomRightEdge = view.leftEdge;
                    view.bottomEdge = mergingView.bottomEdge;
                    mergingView.destroyByUser();
                } else { // splitter is shared by two views and also by other views on the left and on the right
                    view.bottomEdge = mergingView.bottomEdge;
                    var leftSplitter = createSplitter({"y": splitter.y, "topLeftEdge": splitter.topLeftEdge, "bottomRightEdge": view.leftEdge, "orientation": splitter.orientation});
                    var rightSplitter = createSplitter({"y": splitter.y, "topLeftEdge": view.rightEdge, "bottomRightEdge": splitter.bottomRightEdge, "orientation": splitter.orientation});
                    replaceWithSuitableSplitter(splitter, leftSplitter, rightSplitter);
                    mergingView.destroyByUser();
                    splitter.destroyByUser();
                }
            }
        }

        // merge horizontally - right to left
        if(view.rightEdge && view.rightEdge === mergingView.leftEdge && view.bottomEdge === mergingView.bottomEdge && view.topEdge === mergingView.topEdge) {
            // splitter is shared only by the two views
            var splitter = view.rightEdge;
            if(splitter.bottomRightEdge === view.bottomEdge && splitter.topLeftEdge === view.topEdge) {
                splitter.destroyByUser();
                view.rightEdge = mergingView.rightEdge;
                mergingView.destroyByUser();
            } else {
                if(splitter.bottomRightEdge === view.bottomEdge) { // splitter is shared by two views and also by another view on the right
                    splitter.bottomRightEdge = view.topEdge;
                    view.rightEdge = mergingView.rightEdge;
                    mergingView.destroyByUser();
                } else if(splitter.topLeftEdge === view.topEdge) { // splitter is shared by two views and also by another view on the left
                    splitter.topLeftEdge = view.bottomEdge;
                    view.rightEdge = mergingView.rightEdge;
                    mergingView.destroyByUser();
                } else { // splitter is shared by two views and also by other views on the left and on the right
                    view.rightEdge = mergingView.rightEdge;
                    var bottomSplitter = createSplitter({"x": splitter.x, "topLeftEdge": view.bottomEdge, "bottomRightEdge": splitter.bottomRightEdge, "orientation": splitter.orientation});
                    var topSplitter = createSplitter({"x": splitter.x, "topLeftEdge": splitter.topLeftEdge, "bottomRightEdge": view.topEdge, "orientation": splitter.orientation});
                    replaceWithSuitableSplitter(splitter, bottomSplitter, topSplitter);
                    mergingView.destroyByUser();
                    splitter.destroyByUser();
                }
            }
        }

        // merge horizontally - left to right
        if(view.leftEdge && view.leftEdge === mergingView.rightEdge && view.bottomEdge === mergingView.bottomEdge && view.topEdge === mergingView.topEdge) {
            // splitter is shared only by the two views
            var splitter = view.leftEdge;
            if(splitter.bottomRightEdge === view.bottomEdge && splitter.topLeftEdge === view.topEdge) {
                splitter.destroyByUser();
                view.leftEdge = mergingView.leftEdge;
                mergingView.destroyByUser();
            } else {
                if(splitter.bottomRightEdge === view.bottomEdge) { // splitter is shared by two views and also by another view on the right
                    splitter.bottomRightEdge = view.topEdge;
                    view.leftEdge = mergingView.leftEdge;
                    mergingView.destroyByUser();
                } else if(splitter.topLeftEdge === view.topEdge) { // splitter is shared by two views and also by another view on the left
                    splitter.topLeftEdge = view.bottomEdge;
                    view.leftEdge = mergingView.leftEdge;
                    mergingView.destroyByUser();
                } else { // splitter is shared by two views and also by other views on the left and on the right
                    view.leftEdge = mergingView.leftEdge;
                    var bottomSplitter = createSplitter({"x": splitter.x, "topLeftEdge": view.bottomEdge, "bottomRightEdge": splitter.bottomRightEdge, "orientation": splitter.orientation});
                    var topSplitter = createSplitter({"x": splitter.x, "topLeftEdge": splitter.topLeftEdge, "bottomRightEdge": view.topEdge, "orientation": splitter.orientation});
                    replaceWithSuitableSplitter(splitter, bottomSplitter, topSplitter);
                    mergingView.destroyByUser();
                    splitter.destroyByUser();
                }
            }
        }
    }

    function evaluateViewSwapping(view, location) {
        if(!view || !view.isView)
            return false;

        var swappingItem = childAt(location.x, location.y);
        if(!swappingItem)
            return false;

        var swappingView;
        if(swappingItem.isCorner) {
            var swappingCorner = swappingItem;
            swappingView = swappingCorner.view;
        } else if(swappingItem.isView) {
            swappingView = swappingItem;
        }

        if(!swappingView || !swappingView.isView || view === swappingView)
            return false;

        return true;
    }

    function swapView(view, location) {
        if(!view || !view.isView)
            return;

        var swappingItem = childAt(location.x, location.y);
        if(!swappingItem)
            return;

        var swappingView;
        if(swappingItem.isCorner) {
            var swappingCorner = swappingItem;
            swappingView = swappingCorner.view;
        } else if(swappingItem.isView) {
            swappingView = swappingItem;
        }

        if(!swappingView || !swappingView.isView || view === swappingView)
            return;

        var topEdge     = swappingView.topEdge;
        var bottomEdge  = swappingView.bottomEdge;
        var leftEdge    = swappingView.leftEdge;
        var rightEdge   = swappingView.rightEdge;

        swappingView.topEdge    = view.topEdge;
        swappingView.bottomEdge = view.bottomEdge;
        swappingView.leftEdge   = view.leftEdge;
        swappingView.rightEdge  = view.rightEdge;

        view.topEdge    = topEdge;
        view.bottomEdge = bottomEdge;
        view.leftEdge   = leftEdge;
        view.rightEdge  = rightEdge;
    }

    // View Component
    Component {
        id: viewComponent

        Loader {
            id: view
            active: false

            z: 0
            readonly property bool isView: true

            property int uiId: 0
            property int previousUiId: uiId
            onUiIdChanged: {
                SofaSettingsScript.Ui.replace(previousUiId, uiId);
                init();
            }

            property int contentUiId: 0

            Settings {
                id: uiSettings
                category: 0 !== view.uiId ? "ui_" + view.uiId : "dummy"
                property int contentUiId    : 0

                property int topEdgeUiId    : 0
                property int bottomEdgeUiId : 0
                property int leftEdgeUiId   : 0
                property int rightEdgeUiId  : 0

                Component.onCompleted: {
                    view.contentUiId = uiSettings.contentUiId;

                    view.active = true;

                    uiSettings.contentUiId = Qt.binding(function() {return null !== view.item && undefined !== view.item.uiId ? view.item.uiId : 0;});
                }
            }

            function init() {
                uiSettings.contentUiId     = Qt.binding(function() {return view.contentUiId;});
                uiSettings.topEdgeUiId     = Qt.binding(function() {return view.topEdge    ? view.topEdge.uiId     : 0;});
                uiSettings.bottomEdgeUiId  = Qt.binding(function() {return view.bottomEdge ? view.bottomEdge.uiId  : 0;});
                uiSettings.leftEdgeUiId    = Qt.binding(function() {return view.leftEdge   ? view.leftEdge.uiId    : 0;});
                uiSettings.rightEdgeUiId   = Qt.binding(function() {return view.rightEdge  ? view.rightEdge.uiId   : 0;});
            }

            function load() {
                if(0 === uiId)
                    return;

                view.contentUiId    = uiSettings.contentUiId;
                view.topEdge        = root.getSplitterByUiId(uiSettings.topEdgeUiId);
                view.bottomEdge     = root.getSplitterByUiId(uiSettings.bottomEdgeUiId);
                view.leftEdge       = root.getSplitterByUiId(uiSettings.leftEdgeUiId);
                view.rightEdge      = root.getSplitterByUiId(uiSettings.rightEdgeUiId);
            }

            Component.onCompleted: {
                if(0 === uiId) {
                    uiId = SofaSettingsScript.Ui.generate();
                    init();
                }
            }

            Component.onDestruction: {
                if(corner)
                    corner.destroy();

                if(isUserDestroyed) {
                    var previousUiId = uiId;
                    uiId = 0;
                    SofaSettingsScript.Ui.remove(previousUiId);
                }
            }

            property bool isUserDestroyed: false
            function destroyByUser() {
                isUserDestroyed = true;

                if(item && undefined !== item.setNoSettings)
                    item.setNoSettings();

                destroy();
            }

            property Item corner

            property Item topEdge
            property Item bottomEdge
            property Item leftEdge
            property Item rightEdge

            anchors.top: topEdge ? topEdge.bottom : root.top
            anchors.bottom: bottomEdge ? bottomEdge.top : root.bottom
            anchors.left: leftEdge ? leftEdge.right : root.left
            anchors.right: rightEdge ? rightEdge.left : root.right

            source: root.source
            sourceComponent: root.sourceComponent
        }
    }

    Component {
        id: cornerComponent

        Item {
            id: corner

            readonly property bool isCorner: true

            property Item view
            anchors.fill: view
            z: 2

            // Top Right Hand Corner
            Image {
                anchors.top: parent.top
                anchors.right: parent.right
                source: "qrc:/icon/trCorner.png"
                width: 12
                height: width

                MouseArea {
                    anchors.fill: parent
                    acceptedButtons: Qt.LeftButton | Qt.RightButton

                    onReleased: {
                        if(!dragTarget)
                        {
                            if(Qt.LeftButton === mouse.button) {
                                // merging ?
                                if(mouse.x >= width && mouse.y >= 0) {
                                    mergeView(view, mapToItem(root, mouse.x, mouse.y));
                                }
                                else if(mouse.x <= width && mouse.y <= 0) {
                                    mergeView(view, mapToItem(root, mouse.x, mouse.y));
                                }
                            } else if(Qt.RightButton === mouse.button) {
                                // swap ?
                                swapView(view, mapToItem(root, mouse.x, mouse.y));
                            }
                        }

                        SofaToolsScript.Tools.overrideCursorShape = 0;
                        dragTarget = null;
                    }

                    onPositionChanged: {
                        if(!dragTarget && (pressedButtons & Qt.LeftButton)) {
                            if(mouse.x <= -splittingMarginThreshold && mouse.y >= 0 && mouse.y <= height + splittingMarginThreshold && view.width > splitterMarginThreshold * 2) {
                                var splitter = splitView(view, Qt.Horizontal, mapToItem(root, mouse.x, mouse.y));
                                dragTarget = splitter;
                            } else if(mouse.x >= -splittingMarginThreshold && mouse.x <= width && mouse.y >= height + splittingMarginThreshold && view.height > splitterMarginThreshold * 2) {
                                var splitter = splitView(view, Qt.Vertical, mapToItem(root, mouse.x, mouse.y));
                                dragTarget = splitter;
                            }
                        }

                        if(dragTarget) {
                            if(dragTarget.isSplitter) {
                                var splitter = dragTarget;
                                if(Qt.Vertical === splitter.orientation) {
                                    splitter.x = mapToItem(root, mouse.x, mouse.y).x;
                                    var dragBoundary = computeDragXBoundary(splitter);
                                    var dragMinimumX = dragBoundary.min;
                                    var dragMaximumX = dragBoundary.max;
                                    splitter.x = Math.max(dragMinimumX, Math.min(splitter.x, dragMaximumX));
                                } else {
                                    splitter.y = mapToItem(root, mouse.x, mouse.y).y;
                                    var dragBoundary = computeDragYBoundary(splitter);
                                    var dragMinimumY = dragBoundary.min;
                                    var dragMaximumY = dragBoundary.max;
                                    splitter.y = Math.max(dragMinimumY, Math.min(splitter.y, dragMaximumY));
                                }
                            }
                        } else {
                            if((pressedButtons & Qt.LeftButton) && evaluateViewMerging(view, mapToItem(root, mouse.x, mouse.y)))
                                SofaToolsScript.Tools.overrideCursorShape = Qt.OpenHandCursor;
                            else if((pressedButtons & Qt.RightButton) && evaluateViewSwapping(view, mapToItem(root, mouse.x, mouse.y)))
                                SofaToolsScript.Tools.overrideCursorShape = Qt.OpenHandCursor;
                            else
                                SofaToolsScript.Tools.overrideCursorShape = Qt.ForbiddenCursor;
                        }
                    }

                    property Item dragTarget

                    cursorShape: Qt.CrossCursor

                    onDragTargetChanged: {
                        if(dragTarget) {
                            if(dragTarget.isSplitter) {
                                var splitter = dragTarget;
                                if(Qt.Horizontal !== splitter.orientation)
                                    SofaToolsScript.Tools.overrideCursorShape = Qt.SplitHCursor;
                                else
                                    SofaToolsScript.Tools.overrideCursorShape = Qt.SplitVCursor;
                            }
                        } else {
                            SofaToolsScript.Tools.overrideCursorShape = 0;
                            updateSplitters();
                        }
                    }
                }
            }

            // Bottom Left Hand Corner
            Image {
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                source: "qrc:/icon/blCorner.png"
                width: 12
                height: width

                MouseArea {
                    anchors.fill: parent
                    acceptedButtons: Qt.LeftButton | Qt.RightButton

                    onReleased: {
                        if(!dragTarget)
                        {
                            // merging ?
                            if(Qt.LeftButton === mouse.button) {
                                if(mouse.x <= 0 && mouse.y <= height) {
                                    mergeView(view, mapToItem(root, mouse.x, mouse.y));
                                }
                                else if(mouse.x >= 0 && mouse.y >= height) {
                                    mergeView(view, mapToItem(root, mouse.x, mouse.y));
                                }
                            } else if(Qt.RightButton === mouse.button) {
                                // swap ?
                                swapView(view, mapToItem(root, mouse.x, mouse.y));
                            }
                        }

                        SofaToolsScript.Tools.overrideCursorShape = 0;
                        dragTarget = null;
                    }

                    onPositionChanged: {
                        if(!dragTarget && (pressedButtons & Qt.LeftButton)) {
                            if(mouse.x >= width + splittingMarginThreshold && mouse.y >= -splittingMarginThreshold && mouse.y <= height && view.width > splitterMarginThreshold * 2) {
                                var splitter = splitView(view, Qt.Horizontal, mapToItem(root, mouse.x, mouse.y));
                                dragTarget = splitter;
                            } else if(mouse.x >= 0 && mouse.x <= width + splittingMarginThreshold && mouse.y <= -splittingMarginThreshold && view.height > splitterMarginThreshold * 2) {
                                var splitter = splitView(view, Qt.Vertical, mapToItem(root, mouse.x, mouse.y));
                                dragTarget = splitter;
                            }
                        }

                        if(dragTarget) {
                            if(dragTarget.isSplitter) {
                                var splitter = dragTarget;
                                if(Qt.Vertical === splitter.orientation) {
                                    splitter.x = mapToItem(root, mouse.x, mouse.y).x;
                                    var dragBoundary = computeDragXBoundary(splitter);
                                    var dragMinimumX = dragBoundary.min;
                                    var dragMaximumX = dragBoundary.max;
                                    splitter.x = Math.max(dragMinimumX, Math.min(splitter.x, dragMaximumX));
                                } else {
                                    splitter.y = mapToItem(root, mouse.x, mouse.y).y;
                                    var dragBoundary = computeDragYBoundary(splitter);
                                    var dragMinimumY = dragBoundary.min;
                                    var dragMaximumY = dragBoundary.max;
                                    splitter.y = Math.max(dragMinimumY, Math.min(splitter.y, dragMaximumY));
                                }
                            }
                        } else {
                            if((pressedButtons & Qt.LeftButton) && evaluateViewMerging(view, mapToItem(root, mouse.x, mouse.y)))
                                SofaToolsScript.Tools.overrideCursorShape = Qt.OpenHandCursor;
                            else if((pressedButtons & Qt.RightButton) && evaluateViewSwapping(view, mapToItem(root, mouse.x, mouse.y)))
                                SofaToolsScript.Tools.overrideCursorShape = Qt.OpenHandCursor;
                            else
                                SofaToolsScript.Tools.overrideCursorShape = Qt.ForbiddenCursor;
                        }
                    }

                    property Item dragTarget

                    cursorShape: Qt.CrossCursor

                    onDragTargetChanged: {
                        if(dragTarget) {
                            if(dragTarget.isSplitter) {
                                var splitter = dragTarget;
                                if(Qt.Horizontal !== splitter.orientation)
                                    SofaToolsScript.Tools.overrideCursorShape = Qt.SplitHCursor;
                                else
                                    SofaToolsScript.Tools.overrideCursorShape = Qt.SplitVCursor;
                            }
                        } else {
                            SofaToolsScript.Tools.overrideCursorShape = 0;
                            updateSplitters();
                        }
                    }
                }
            }
        }
    }

    function createSplitter(properties) {
        if(undefined === properties)
            properties = {};

        var splitter = splitterComponent.createObject(root, properties);

        return splitter;
    }

    function updateSplitters() {
        // try to merge splitters that should be
        for(var j = 0; j < root.children.length; ++j) {
            var item = root.children[j];
            if(!item.isSplitter || item.isDestroying)
                continue;

            var splitter = item;
            for(var i = 0; i < root.children.length; ++i) {
                var otherItem = root.children[i];
                if(!otherItem.isSplitter || otherItem.isDestroying || item === otherItem)
                    continue;

                var otherSplitter = otherItem;
                mergeSplitter(splitter, otherSplitter);
            }
        }
    }

    function replaceSplitter(splitter, replacementSplitter) {
        for(var i = 0; i < root.children.length; ++i) {
            var otherItem = root.children[i];
            if(splitter === otherItem)
                continue;

            if(otherItem.isSplitter) {
                var otherSplitter = otherItem;
                if(otherSplitter.topLeftEdge === splitter)
                    otherSplitter.topLeftEdge = replacementSplitter;
                if(otherSplitter.bottomRightEdge === splitter)
                    otherSplitter.bottomRightEdge = replacementSplitter;
            } else if(otherItem.isView) {
                var otherView = otherItem;
                if(otherView.topEdge === splitter)
                    otherView.topEdge = replacementSplitter;
                if(otherView.bottomEdge === splitter)
                    otherView.bottomEdge = replacementSplitter;
                if(otherView.leftEdge === splitter)
                    otherView.leftEdge = replacementSplitter;
                if(otherView.rightEdge === splitter)
                    otherView.rightEdge = replacementSplitter;
            }
        }
    }

    function chooseSuitableSplitter(item, splitter, replacementSplitter0, replacementSplitter1) {
        if(Qt.Horizontal === splitter.orientation) {
            if(replacementSplitter0.x <= item.x && replacementSplitter0.x + replacementSplitter0.width >= item.x + item.width)
                return replacementSplitter0;
            if(replacementSplitter1.x <= item.x && replacementSplitter1.x + replacementSplitter1.width >= item.x + item.width)
                return replacementSplitter1;
        } else {
            if(replacementSplitter0.y <= item.y && replacementSplitter0.y + replacementSplitter0.height >= item.y + item.height)
                return replacementSplitter0;
            if(replacementSplitter1.y <= item.y && replacementSplitter1.y + replacementSplitter1.height >= item.y + item.height)
                return replacementSplitter1;
        }

        return null;
    }

    function replaceWithSuitableSplitter(splitter, replacementSplitter0, replacementSplitter1)
    {
        for(var i = 0; i < root.children.length; ++i) {
            var otherItem = root.children[i];
            if(splitter === otherItem)
                continue;

            var replacementSplitter = chooseSuitableSplitter(otherItem, splitter, replacementSplitter0, replacementSplitter1);

            if(otherItem.isSplitter) {
                var otherSplitter = otherItem;
                if(otherSplitter.topLeftEdge === splitter)
                    otherSplitter.topLeftEdge = replacementSplitter;
                if(otherSplitter.bottomRightEdge === splitter)
                    otherSplitter.bottomRightEdge = replacementSplitter;
            } else if(otherItem.isView) {
                var otherView = otherItem;
                if(otherView.topEdge === splitter)
                    otherView.topEdge = replacementSplitter;
                if(otherView.bottomEdge === splitter)
                    otherView.bottomEdge = replacementSplitter;
                if(otherView.leftEdge === splitter)
                    otherView.leftEdge = replacementSplitter;
                if(otherView.rightEdge === splitter)
                    otherView.rightEdge = replacementSplitter;
            }
        }
    }

    function shareEdge(splitter, otherSplitter) {
        if((null !== splitter.bottomRightEdge && splitter.bottomRightEdge === otherSplitter.topLeftEdge) ||
           (null !== splitter.topLeftEdge && splitter.topLeftEdge === otherSplitter.bottomRightEdge)) {
            return true;
        }

        return false;
    }

    function mergeSplitter(splitter, otherSplitter) {
        if(splitter !== otherSplitter && shareEdge(splitter, otherSplitter) && (Math.abs(splitter.x - otherSplitter.x) < splitterMagnetizeThreshold || Math.abs(splitter.y - otherSplitter.y) < splitterMagnetizeThreshold)) {
            if(null !== otherSplitter.bottomRightEdge && otherSplitter.bottomRightEdge === splitter.topLeftEdge) {
                otherSplitter.bottomRightEdge = splitter.bottomRightEdge;
            } else if(null !== otherSplitter.topLeftEdge && otherSplitter.topLeftEdge === splitter.bottomRightEdge) {
                otherSplitter.topLeftEdge = splitter.topLeftEdge;
            }

            replaceSplitter(splitter, otherSplitter);
            splitter.destroyByUser();

            return otherSplitter;
        }

        return splitter;
    }

    function computeDragXBoundary(splitter) {
        var min = 0;
        var max = root.width;

        var closestSplitterFound = false;
        var closestSplitterX = 0;
        for(var i = 0; i < root.children.length; ++i) {
            var item = root.children[i];
            if(splitter === item)
                continue;

            if(item.isSplitter && splitterMagnetizeThreshold > 0) {
                var otherSplitter = item;

                if(splitter.orientation === otherSplitter.orientation) { // TODO: check if they have a common edge
                    var otherSplitterX = Math.round(otherSplitter.x);
                    if(closestSplitterFound && Math.abs(splitter.x - otherSplitterX) < Math.abs(splitter.x - closestSplitterX)) {
                        closestSplitterX = otherSplitterX;
                    } else if(!closestSplitterFound) {
                        closestSplitterFound = true;
                        closestSplitterX = otherSplitterX;
                    }
                }
            }

            if(!item.isView)
                continue;

            var view = item;
            if(splitter === view.rightEdge) {
                min = Math.max(min, view.x);
            } else if(splitter === view.leftEdge) {
                max = Math.min(max, view.x + view.width);
            }
        }
        min += splitterMarginThreshold;
        max -= splitter.width + splitterMarginThreshold;

        // magnetize with closest splitter
        if(closestSplitterFound && Math.abs(splitter.x - closestSplitterX) <= splitterMagnetizeThreshold && closestSplitterX >= min && closestSplitterX <= max)
            splitter.x = closestSplitterX;

        return {"min": min, "max": max};
    }

    function computeDragYBoundary(splitter) {
        var min = 0;
        var max = root.height;

        var closestSplitterFound = false;
        var closestSplitterY = 0;
        for(var i = 0; i < root.children.length; ++i) {
            var item = root.children[i];
            if(splitter === item)
                continue;

            if(item.isSplitter && splitterMagnetizeThreshold > 0) {
                var otherSplitter = item;

                if(splitter.orientation === otherSplitter.orientation) {
                    var otherSplitterY = Math.round(otherSplitter.y);
                    if(closestSplitterFound && Math.abs(splitter.y - otherSplitterY) < Math.abs(splitter.y - closestSplitterY)) {
                        closestSplitterY = otherSplitterY;
                    } else if(!closestSplitterFound) {
                        closestSplitterFound = true;
                        closestSplitterY = otherSplitterY;
                    }
                }
            }

            if(!item.isView)
                continue;

            var view = item;
            if(splitter === view.bottomEdge) {
                min = Math.max(min, view.y);
            } else if(splitter === view.topEdge) {
                max = Math.min(max, view.y + view.height);
            }
        }
        min += splitterMarginThreshold;
        max -= splitter.height + splitterMarginThreshold;

        // magnetize with closest splitter
        if(closestSplitterFound && Math.abs(splitter.y - closestSplitterY) <= splitterMagnetizeThreshold && closestSplitterY >= min && closestSplitterY <= max)
            splitter.y = closestSplitterY;

        return {"min": min, "max": max};
    }

    // Splitter Component
    Component {
        id: splitterComponent

        Rectangle {
            id: splitter
            z: 1
            color: "grey"

            property Item topLeftEdge
            property Item bottomRightEdge
            property real relativeX
            property real relativeY
            property int  orientation

            readonly property bool isSplitter: true
            property bool isDestroying: false

            property int uiId: 0
            property int previousUiId: uiId
            onUiIdChanged: {
                SofaSettingsScript.Ui.replace(previousUiId, uiId);
            }

            Settings {
                id: uiSettings
                category: 0 !== splitter.uiId ? "ui_" + splitter.uiId : "dummy"

                property int topLeftEdgeUiId:       0
                property int bottomRightEdgeUiId:   0

                property real relativeX
                property real relativeY
                property int orientation
            }

            function init() {
                uiSettings.topLeftEdgeUiId          = Qt.binding(function() {return splitter.topLeftEdge     ? splitter.topLeftEdge.uiId : 0;});
                uiSettings.bottomRightEdgeUiId      = Qt.binding(function() {return splitter.bottomRightEdge ? splitter.bottomRightEdge.uiId : 0;});
                uiSettings.relativeX                = Qt.binding(function() {return splitter.relativeX;});
                uiSettings.relativeY                = Qt.binding(function() {return splitter.relativeY;});
                uiSettings.orientation              = Qt.binding(function() {return splitter.orientation;});
            }

            function load() {
                if(0 === uiId)
                    return;

                splitter.topLeftEdge        = root.getSplitterByUiId(uiSettings.topLeftEdgeUiId);
                splitter.bottomRightEdge    = root.getSplitterByUiId(uiSettings.bottomRightEdgeUiId);
                splitter.relativeX          = uiSettings.relativeX;
                splitter.relativeY          = uiSettings.relativeY;
                splitter.orientation        = uiSettings.orientation;
            }

            Component.onCompleted: {
                if(0 === uiId) {
                    uiId = SofaSettingsScript.Ui.generate();
                    init();
                }
            }

            property bool isUserDestroyed: false
            Component.onDestruction: {
                if(isUserDestroyed) {
                    var previousUiId = uiId;
                    uiId = 0;
                    SofaSettingsScript.Ui.remove(previousUiId);
                }
            }

            QtObject {
                id: d

                property real relativeX: splitter.x / root.width
                property real relativeY: splitter.y / root.height

                onRelativeXChanged: {
                    if(root.width > 1.0)
                        splitter.relativeX = relativeX;
                }

                onRelativeYChanged: {
                    if(root.height> 1.0)
                        splitter.relativeY = relativeY;
                }
            }

            // if destroyed by user => do not save the item settings
            function destroyByUser() {
                isDestroying = true;
                isUserDestroyed = true;

                destroy();
            }

            width: if(Qt.Vertical === orientation) return root.splitterThickness;
            height: if(Qt.Horizontal === orientation) return root.splitterThickness;

            anchors.top: Qt.Horizontal === orientation ? undefined : (topLeftEdge ? topLeftEdge.bottom : root.top)
            anchors.bottom: Qt.Horizontal === orientation ? undefined : (bottomRightEdge ? bottomRightEdge.top : root.bottom)
            anchors.left: Qt.Vertical === orientation ? undefined : (topLeftEdge ? topLeftEdge.right : root.left)
            anchors.right: Qt.Vertical === orientation ? undefined : (bottomRightEdge ? bottomRightEdge.left : root.right)

            Drag.dragType: Drag.Automatic

            MouseArea {
                id: splitterMouseArea
                anchors.fill: parent

                property real hoverMargin: -5
                anchors.leftMargin: splitter.orientation === Qt.Vertical     ? hoverMargin : 0
                anchors.rightMargin: splitter.orientation === Qt.Vertical    ? hoverMargin : 0
                anchors.topMargin: splitter.orientation === Qt.Horizontal    ? hoverMargin : 0
                anchors.bottomMargin: splitter.orientation === Qt.Horizontal ? hoverMargin : 0

                onPressed: {
                    dragTarget = parent
                }

                onReleased: {
                    dragTarget = null;
                }

                onPositionChanged: {
                    if(dragTarget) {
                        var splitter = dragTarget;
                        if(Qt.Vertical === splitter.orientation) {
                            splitter.x = mapToItem(root, mouse.x, mouse.y).x;
                            var dragBoundary = computeDragXBoundary(splitter);
                            var dragMinimumX = dragBoundary.min;
                            var dragMaximumX = dragBoundary.max;
                            splitter.x = Math.max(dragMinimumX, Math.min(splitter.x, dragMaximumX));
                        } else {
                            splitter.y = mapToItem(root, mouse.x, mouse.y).y;
                            var dragBoundary = computeDragYBoundary(splitter);
                            var dragMinimumY = dragBoundary.min;
                            var dragMaximumY = dragBoundary.max;
                            splitter.y = Math.max(dragMinimumY, Math.min(splitter.y, dragMaximumY));
                        }
                    }
                }

                property Item dragTarget

                cursorShape: splitter.orientation === Qt.Vertical ? Qt.SizeHorCursor : Qt.SizeVerCursor

                onDragTargetChanged: {
                    if(dragTarget) {
                        var splitter = dragTarget;
                        if(Qt.Horizontal !== splitter.orientation)
                            SofaToolsScript.Tools.overrideCursorShape = Qt.SizeHorCursor;
                        else
                            SofaToolsScript.Tools.overrideCursorShape = Qt.SizeVerCursor;
                    } else {
                        SofaToolsScript.Tools.overrideCursorShape = 0;
                        updateSplitters();
                    }
                }
            }
        }
    }
}
