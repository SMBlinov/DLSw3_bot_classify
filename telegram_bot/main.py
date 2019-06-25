from model import ClassPredictor
from telegram_token import token
import torch
from config import reply_texts
import numpy as np
from PIL import Image
from io import BytesIO
import telebot
import pandas as pd

model = ClassPredictor()
wiki = pd.read_csv('D:/DLSw3/bot_classificator/DLSw3_bot_classify/dataset/res.csv')


def _info(pred_):
	mess_='Стиль данной картины похож: \n'
	val_, id_=pred_[2].topk(3)
	for i in range(0,3):
		pain_=wiki[wiki['count']==int(id_[i])].name.values[0]
		percent_=int(val_[i]*100)
		mess_+=str(pain_)+' '+str(percent_)+'%\n' 
	return mess_

def send_prediction_on_photo(bot, update):
	#print(update)
	if update.message==None:
		type_chat='channel_post'
	else:
		type_chat='message'
	chat_id = update[type_chat].chat_id
	print("Got image from {}".format(chat_id))

    # получаем информацию о картинке
	image_info = update[type_chat].photo[-1]
	image_file = bot.get_file(image_info)
	image_stream = BytesIO()
	image_file.download(out=image_stream)
	
	pred_ = model.predict(image_stream)
	update[type_chat].reply_text(_info(pred_))

    # теперь отправим результат
	print("Sent Answer to user, predicted: {}".format(pred_[0]))


if __name__ == '__main__':
	from telegram.ext import Updater, MessageHandler, Filters, CommandHandler, CallbackQueryHandler
	from telegram import InlineKeyboardButton, InlineKeyboardMarkup
	import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
	logging.basicConfig(
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		level=logging.INFO)
    # используем прокси, так как без него у меня ничего не работало(
	updater = Updater(token=token)
	updater.dispatcher.add_handler(MessageHandler(Filters.photo, send_prediction_on_photo))
	updater.start_polling()
