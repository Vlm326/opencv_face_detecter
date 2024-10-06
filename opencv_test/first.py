import cv2

# Загрузка каскада для полного тела
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование в черно-белый формат
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение людей
    bodies = body_cascade.detectMultiScale(gray, 1.1, 3)

    # Рисование прямоугольников вокруг обнаруженных людей
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Показ изображения
    cv2.imshow('Body Detection', frame)

    # Прерывание по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()