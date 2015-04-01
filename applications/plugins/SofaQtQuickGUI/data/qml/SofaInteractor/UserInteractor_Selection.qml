import QtQuick 2.0
import SofaBasics 1.0
import "qrc:/SofaCommon/SofaToolsScript.js" as SofaToolsScript

UserInteractor {
    id: root

    property var previousX: -1
    property var previousY: -1

    property real moveSpeed: 0.00133
    property real turnSpeed: 20.0
    property real zoomSpeed: 1.0

    function init() {
        addMousePressedMapping(Qt.LeftButton, function(mouse) {
            var nearPosition = viewer.mapToWorld(Qt.vector3d(mouse.x + 0.5, mouse.y + 0.5, 0.0));
            var farPosition = viewer.mapToWorld(Qt.vector3d(mouse.x + 0.5, mouse.y + 0.5, 1.0));
            if(scene.pickingInteractor.pick(nearPosition, farPosition.minus(nearPosition))) {
                var z = viewer.camera.computeDepth(scene.pickingInteractor.pickedPointPosition());
                var position = viewer.camera.projectOnViewPlane(nearPosition, z);
                scene.pickingInteractor.position = position;

                setMouseMoveMapping(function(mouse) {
                    var nearPosition = viewer.mapToWorld(Qt.vector3d(mouse.x + 0.5, mouse.y + 0.5, 0.0));
                    var farPosition = viewer.mapToWorld(Qt.vector3d(mouse.x + 0.5, mouse.y + 0.5, 1.0));
                    var z = viewer.camera.computeDepth(scene.pickingInteractor.pickedPointPosition());
                    var position = viewer.camera.projectOnViewPlane(nearPosition, z);
                    scene.pickingInteractor.position = position;
                });
            }
        });

        addMouseReleasedMapping(Qt.LeftButton, function(mouse) {
            scene.pickingInteractor.release();

            setMouseMoveMapping(null);
        });

        addMouseDoubleClickedMapping(Qt.LeftButton, function(mouse) {
            var position = viewer.projectOnGeometry(Qt.point(mouse.x + 0.5, mouse.y + 0.5));
            if(1.0 === position.w) {
                viewer.camera.target = position.toVector3d();
                crosshairGizmo.pop();
            }
        });

        addMousePressedMapping(Qt.RightButton, function(mouse) {
            previousX = mouse.x;
            previousY = mouse.y;

            crosshairGizmo.show();
            SofaToolsScript.Tools.overrideCursorShape = Qt.ClosedHandCursor;

            setMouseMoveMapping(function(mouse) {
                if(!viewer.camera)
                    return;

                var angleAroundX = 0.0;
                var angleAroundY = 0.0;
                var angleAroundZ = 0.0;

                if(Qt.ControlModifier & mouse.modifiers) {
                    angleAroundZ = (previousX - mouse.x) / 180.0 * Math.PI * turnSpeed;
                } else {
                    angleAroundX = (previousY - mouse.y) / 180.0 * Math.PI * turnSpeed;
                    angleAroundY = (previousX - mouse.x) / 180.0 * Math.PI * turnSpeed;
                }

                viewer.camera.turn(angleAroundX, angleAroundY, angleAroundZ);

                previousX = mouse.x;
                previousY = mouse.y;
            });
        });

        addMouseReleasedMapping(Qt.RightButton, function(mouse) {
            setMouseMoveMapping(null);

            SofaToolsScript.Tools.overrideCursorShape = 0;
            crosshairGizmo.hide();
        });

        addMousePressedMapping(Qt.MiddleButton, function(mouse) {
            previousX = mouse.x;
            previousY = mouse.y;

            crosshairGizmo.show();
            SofaToolsScript.Tools.overrideCursorShape = Qt.ClosedHandCursor;

            setMouseMoveMapping(function(mouse) {
                if(!viewer.camera)
                    return;

                var screenToScene = viewer.camera.target.minus(viewer.camera.eye()).length();

                var moveX = (mouse.x - previousX) * screenToScene * moveSpeed;
                var moveY = (mouse.y - previousY) * screenToScene * moveSpeed;
                viewer.camera.move(-moveX, moveY, 0.0);

                previousX = mouse.x;
                previousY = mouse.y;
            });
        });

        addMouseReleasedMapping(Qt.MiddleButton, function(mouse) {
            setMouseMoveMapping(null);

            SofaToolsScript.Tools.overrideCursorShape = 0;
            crosshairGizmo.hide();
        });

        setMouseWheelMapping(function(wheel) {
            if(!viewer.camera)
                return;

            if(0 === wheel.angleDelta.y)
                return;

            var boundary = 2.0;
            var factor = Math.max(-boundary, Math.min(wheel.angleDelta.y / 120.0, boundary)) / boundary;
            if(factor < 0.0) {
                factor = 1.0 + 0.5 * factor;
                factor /= zoomSpeed;
            }
            else {
                factor = 1.0 + factor;
                factor *= zoomSpeed;
            }

            viewer.camera.zoom(factor);

            wheel.accepted = true;
        });
    }
}
