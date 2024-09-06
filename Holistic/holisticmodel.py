import cv2
import mediapipe as mp


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break

        # mediapipe için BGR görüntüyü RGB'ye çevir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # el, yüz ve vücut pozisyonu tespiti
        results = holistic.process(image)


        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Yüz çizimi
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Sağ el çizimi
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS)

        # Sol el çizimi
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS)

        # Vücut çizimi
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('Mediapipe Holistic Modeli', image)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
