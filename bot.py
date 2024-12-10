import telebot
from telebot import types
from ultralytics import YOLO
import os
import datetime

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "best.pt")

# Загрузите модель
model = YOLO(model_path)

API_TOKEN = '7602626066:AAHKb0BWmiPral734oHGJeNWDDK4JBqlKjo'
bot = telebot.TeleBot(API_TOKEN)

# Словарь для хранения данных сессии пользователей
user_sessions = {}


def detect_objects(image_path):
    # Запуск инференса на изображении
    results = model([image_path])

    # Обработка результатов
    result = results[0]

    # Извлечение классов из результатов
    classes_detected = result.boxes.cls.tolist()
    class_names = [model.names[int(cls)] for cls in classes_detected]

    # Вывод класса проблемы
    class_probabilities = result.boxes.conf.tolist()
    class_probabilities = [f"{cls_name} ({100 * prob:.2f}%)" for cls, prob, cls_name in zip(classes_detected, class_probabilities, class_names)]

    annotated_image_path = "annotated_image.jpg"
    result.save(annotated_image_path)

    return annotated_image_path, class_probabilities


def save_for_retraining(annotated_image_path, new_filename):
    # Сохраните изображение для дообучения
    retraining_path = "retraining_images/"
    if not os.path.exists(retraining_path):
        os.makedirs(retraining_path)
    os.rename(annotated_image_path, os.path.join(retraining_path, new_filename))


@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button1 = types.KeyboardButton('Загрузить фото')
    button2 = types.KeyboardButton('Информация')
    button3 = types.KeyboardButton('Тех. поддержка')
    markup.add(button1, button2, button3)
    bot.send_message(message.chat.id, 'Я телеграм бот для определения брака ноутбуков', reply_markup=markup)


@bot.message_handler(content_types=['text'])
def get_text_message(message):
    if message.text == 'бот':
        bot.send_message(message.chat.id, 'нажми /start')
    elif message.text == 'Загрузить фото':
        bot.send_message(message.chat.id, 'Пожалуйста, отправьте фото ноутбука для анализа.')
    elif message.text == 'Информация':
        bot.send_message(message.chat.id,
                         'Привет, я бот который найдет поломку в твоем ноутбуке. Чтобы я нашел ошибку загрузи картинку/видео и при помощи нейросетей я выдам тебе результат с возможными поломками.')
    elif message.text == 'Тех. поддержка':
        bot.send_message(message.chat.id, '[Администратор](https://t.me/hantik_X)', parse_mode='Markdown')


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
  file_info = bot.get_file(message.photo[-1].file_id)
  downloaded_file = bot.download_file(file_info.file_path)

  photo_path = "received_photo.jpg"
  with open(photo_path, 'wb') as new_file:
    new_file.write(downloaded_file)

  annotated_image_path, class_probabilities = detect_objects(photo_path)

  class_probabilities_text = '\n'.join(class_probabilities)
  response_text = f"На изображении обнаружены следующие поломки:\n{class_probabilities_text}"
  bot.send_message(message.chat.id, response_text)

  with open(annotated_image_path, 'rb') as img_file:
    bot.send_photo(message.chat.id, img_file)

  # Создаем кнопки "Да" и "Нет"
  markup = types.InlineKeyboardMarkup()
  button_yes = types.InlineKeyboardButton("Да", callback_data="confirm_yes")
  button_no = types.InlineKeyboardButton("Нет", callback_data="confirm_no")
  markup.add(button_yes, button_no)

  question = "Пожалуйста, подтвердите, правильно ли определены поломки:"
  bot.send_message(message.chat.id, question, reply_markup=markup)

  user_sessions[message.chat.id] = annotated_image_path

@bot.callback_query_handler(func=lambda call: True)
def handle_feedback(call):
  annotated_image_path = user_sessions.get(call.message.chat.id)
  if annotated_image_path:
    if call.data == "confirm_yes":
      filename, file_extension = os.path.splitext(annotated_image_path)
      new_filename = f"{filename}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{file_extension}"
      save_for_retraining(annotated_image_path, new_filename)
      bot.send_message(call.message.chat.id, "Спасибо за подтверждение! Изображение сохранено для дообучения.")
    elif call.data == "confirm_no":
      # Здесь можно развить систему с пользовательской аннотацией или просто ответить
      bot.send_message(call.message.chat.id, "Спасибо за ваш отзыв! Мы будем работать над улучшением.")

    bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)


if __name__ == "__main__":
    bot.polling(none_stop=True)