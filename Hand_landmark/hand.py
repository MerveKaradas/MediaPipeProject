import cv2
import mediapipe as mp

#Videoyu başlat
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands # Bu modül, el algılama ve takip işlemleri için kullanılır.
hands = mpHands.Hands(
    static_image_mode=False, #False=video True=Image için
    max_num_hands=2, # Tespit edilecek max el sayısı
    min_detection_confidence=0.5, # El algılama işlemi sırasında modelin ne kadar güvenle bir eli algılaması gerektiğini belirler.
    min_tracking_confidence=0.5 #Tespit edilen eli takip ederken kullanılan güven düzeyi.

) #ellerin tespit edilmesi ve el üzerindeki 21 anahtar noktanın bulunması için kullanılır.
mpDraw = mp.solutions.drawing_utils # elde edilen noktaları frame üzerinden çizmemizi sağlar

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #mediapipe kullanımı RGB ile oluyor

    handLandmarks = hands.process(frameRGB) # Belirlenen noktaları bulur
  #  print(handLandmarks.multi_hand_landmarks) Tespit edilem noktaların kordinatlarını görmek istersen

    if handLandmarks.multi_hand_landmarks : #Eğer gerçekten el tespit ederse işlemleri gerçekleştirir
        for hlm in handLandmarks.multi_hand_landmarks: # Her bir nokta için
            mpDraw.draw_landmarks(frame, hlm, mpHands.HAND_CONNECTIONS) # noktaların bağlantısını çizer

            for index, lm in enumerate(hlm.landmark):

                height, width, channel = frame.shape
                positionX, positionY = int(lm.x * width), int(lm.y * height)

                # Landmark numarası
                cv2.putText(frame,
                            str(index),
                            (positionX, positionY),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()