import cv2
import asyncio
from time import sleep


async def detect_faces(faces):
    if len(faces) > 0:
        print(f'detected faces {len(faces)}')
        await asyncio.sleep(0.1)
    

face_cascade = cv2.CascadeClassifier('/home/vladislav/Загрузки/haarcascade_frontalface_default.xml')
full_body = cv2.CascadeClassifier('/home/vladislav/Загрузки/haarcascade_fullbody.xml')

cap = cv2.VideoCapture(0)

async def main():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц и тел
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        body = full_body.detectMultiScale(gray, 1.1, 4)

        # Вызов асинхронной функции
        await detect_faces(faces)

        # Отрисовка прямоугольников вокруг обнаруженных объектов
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x, y, w, h) in body:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Body tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
# Запуск асинхронной функции
if __name__ == "__main__":
    asyncio.run(main())
