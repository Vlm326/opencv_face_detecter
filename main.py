import cv2
from telegram import Bot
from telegram import Update
from telegram import Application, CommandHandler, ContextTypes, MessageHandler



face_cascade = cv2.CascadeClassifier('/home/vladislav/Загрузки/haarcascade_frontalface_default.xml')
full_body = cv2.CascadeClassifier('/home/vladislav/Загрузки/haarcascade_fullbody.xml')



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет хозяин\n напиши /help чтобы узнать команды\n я робот, который любит сосать' )

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Это бот для отслеживания лиц')

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Пока')
    await context.application.shutdown()
async def send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, path: str) -> None:
    await update.message.reply_photo(photo=open('path', 'rb'))
  
async def main():
    cap = cv2.VideoCapture(0)
    token = "7912167270:AAF2yHk_-1t6eOqL4UG3z5R8uVWsPQaFOSo"  
    application = Application.builder().token(token).build()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        body = full_body.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x, y, w, h) in body:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Body tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(faces) > 0:
            cv2.imwrite('face.png', frame)
            await send_photo('face.png')
            
            
    cap.release()
    cv2.destroyAllWindows()




    
if __name__ == '__main__':
    main()