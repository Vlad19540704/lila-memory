import os
import logging
import pinecone
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Логи
logging.basicConfig(level=logging.INFO)

# Ключи
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Инициализация Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Создание индекса (если не существует)
INDEX_NAME = "lila-memory"
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(name=INDEX_NAME, dimension=1536)

# Подключение к индексу
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeStore.from_existing_index(INDEX_NAME, embeddings)

# Цепочка с памятью
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Flask
app = Flask(__name__)

# Telegram команды
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я Лила 2.0 с памятью. Пиши что угодно.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_history = context.chat_data.get("history", [])

    # Обработка запроса через LangChain
    result = qa({"question": user_text, "chat_history": chat_history})
    answer = result["answer"]

    # Сохраняем историю
    chat_history.append((user_text, answer))
    context.chat_data["history"] = chat_history

    await update.message.reply_text(answer)

def run_telegram_bot():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == "__main__":
    run_telegram_bot()


