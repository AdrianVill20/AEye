import cv2
import mediapipe as mp
import numpy as np
import time

class HeadPoseEstimator:
    LANDMARK_IDS = [33, 263, 1, 61, 291, 199]

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def _preprocess(self, frame):
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        return image

    def _extract_landmarks(self, results, img_w, img_h):
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        face_2d, face_3d = [], []
        nose_2d = nose_3d = None

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in self.LANDMARK_IDS:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        return face_2d, face_3d, nose_2d, nose_3d, face_landmarks

    def _solve_pose(self, face_2d, face_3d, img_w, img_h):
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1],
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        return x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix

    def _classify_direction(self, x, y, threshold=10):
        if y < -threshold:
            return "Looking Left"
        elif y > threshold:
            return "Looking Right"
        elif x < -threshold:
            return "Looking Down"
        elif x > threshold:
            return "Looking Up"
        return "Forward"

    def get_pose(self, frame, threshold=10):
        image = self._preprocess(frame)
        results = self.face_mesh.process(image)
        img_h, img_w, _ = frame.shape

        extracted = self._extract_landmarks(results, img_w, img_h)
        if extracted is None:
            return None

        face_2d, face_3d, nose_2d, nose_3d, face_landmarks = extracted
        x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix = self._solve_pose(
            face_2d, face_3d, img_w, img_h
        )
        direction = self._classify_direction(x, y, threshold)

        nose_3d_projection, _ = cv2.projectPoints(
            nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
        )
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

        return {
            "direction": direction,
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2),
            "nose_2d": p1,
            "nose_end_2d": p2,
            "landmarks": face_landmarks, 
        }

    def draw_overlay(self, frame, pose):
        if pose is None:
            return frame
        cv2.line(frame, pose["nose_2d"], pose["nose_end_2d"], (255, 0, 0), 3)
        cv2.putText(frame, pose["direction"], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(frame, f'x: {pose["x"]}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'y: {pose["y"]}', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'z: {pose["z"]}', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=pose["landmarks"],
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.drawing_spec,
        )
        return frame

    def run_demo(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            start = time.time()
            display = cv2.flip(frame, 1)

            pose = self.get_pose(frame)
            display = self.draw_overlay(display, pose)

            fps = 1 / (time.time() - start)
            cv2.putText(display, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('Head Pose Estimation', display)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    HeadPoseEstimator().run_demo()