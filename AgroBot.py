import discord
import torch
from discord.ext import commands
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from settings import settings_bot

PATH = 'khvatov/ru_toxicity_detector'
tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSequenceClassification.from_pretrained(PATH)
model.to(torch.device("cpu"))

warning_threshold = 60
user_warnings = {}

intents = discord.Intents.all()
bot = commands.Bot(command_prefix='*', intents=intents)


@bot.event
async def on_ready():
    print(f'Залогинился как {bot.user.name}')


def get_toxicity_probs(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.nn.functional.softmax(model(**inputs).logits, dim=1).cpu().numpy()
    return proba[0]


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith('*'):
        return

    toxicity_probs = get_toxicity_probs(message.content)
    toxicity_perce = toxicity_probs[1] * 100

    print(f'Сообщение: "{message.content}" - Токсичность: {toxicity_perce:.2f}%')

    if toxicity_perce >= warning_threshold:
        author_id = str(message.author.id)
        if author_id not in user_warnings:
            user_warnings[author_id] = 0

        user_warnings[author_id] += 1
        await message.delete()

        user = message.author
        warning_msg = f'{user.mention} Ваше сообщение считается токсичным.'
        await user.send(warning_msg)

    await bot.process_commands(message)


bot_token = settings_bot['token']
bot.run(bot_token)
