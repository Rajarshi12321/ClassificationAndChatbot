from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
import json
import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HUGGINGFACEHUB_APT_TOKEN = os.getenv("HUGGINGFACEHUB_APT_TOKEN")

ANIMAL_API_TOKEN = os.getenv("ANIMAL_API_TOKEN")


# endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

Animal_danger_classification = {
    "Antelope": "Vulnerable",
    "Badger": "Least Concern",
    "Bat": "Vulnerable",
    "Bear": "Vulnerable",
    "Bee": "Least Concern",
    "Beetle": "Least Concern",
    "Bison": "Least Concern",
    "Boar": "Least Concern",
    "Butterfly": "Vulnerable",
    "Cat": "Least Concern",
    "Caterpillar": "Least Concern",
    "Chimpanzee": "Endangered",
    "Cockroach": "Least Concern",
    "Cow": "Least Concern",
    "Coyote": "Least Concern",
    "Crab": "Least Concern",
    "Crow": "Least Concern",
    "Deer": "Least Concern",
    "Dog": "Least Concern",
    "Dolphin": "Least Concern",
    "Donkey": "Least Concern",
    "Dragonfly": "Vulnerable",
    "Duck": "Least Concern",
    "Eagle": "Least Concern",
    "Elephant": "Endangered",
    "Flamingo": "Vulnerable",
    "Fly": "Least Concern",
    "Fox": "Least Concern",
    "Goat": "Least Concern",
    "Goldfish": "Least Concern",
    "Goose": "Least Concern",
    "Gorilla": "Endangered",
    "Grasshopper": "Least Concern",
    "Hamster": "Least Concern",
    "Hare": "Least Concern",
    "Hedgehog": "Least Concern",
    "Hippopotamus": "Vulnerable",
    "Hornbill": "Vulnerable",
    "Horse": "Least Concern",
    "Hummingbird": "Least Concern",
    "Hyena": "Least Concern",
    "Jellyfish": "Least Concern",
    "Kangaroo": "Least Concern",
    "Koala": "Vulnerable",
    "Ladybugs": "Least Concern",
    "Leopard": "Vulnerable",
    "Lion": "Vulnerable",
    "Lizard": "Least Concern",
    "Lobster": "Least Concern",
    "Mosquito": "Least Concern",
    "Moth": "Least Concern",
    "Mouse": "Least Concern",
    "Octopus": "Least Concern",
    "Okapi": "Endangered",
    "Orangutan": "Endangered",
    "Otter": "Vulnerable",
    "Owl": "Least Concern",
    "Ox": "Least Concern",
    "Oyster": "Least Concern",
    "Panda": "Endangered",
    "Parrot": "Least Concern",
    "Pelecaniformes": "Least Concern",
    "Penguin": "Vulnerable",
    "Pig": "Least Concern",
    "Pigeon": "Least Concern",
    "Porcupine": "Least Concern",
    "Possum": "Least Concern",
    "Raccoon": "Least Concern",
    "Rat": "Least Concern",
    "Reindeer": "Least Concern",
    "Rhinoceros": "Vulnerable",
    "Sandpiper": "Least Concern",
    "Seahorse": "Least Concern",
    "Seal": "Vulnerable",
    "Shark": "Least Concern",
    "Sheep": "Least Concern",
    "Snake": "Least Concern",
    "Sparrow": "Least Concern",
    "Squid": "Least Concern",
    "Squirrel": "Least Concern",
    "Starfish": "Least Concern",
    "Swan": "Least Concern",
    "Tiger": "Vulnerable",
    "Turkey": "Least Concern",
    "Turtle": "Vulnerable",
    "Whale": "Vulnerable",
    "Wolf": "Least Concern",
    "Wombat": "Least Concern",
    "Woodpecker": "Least Concern",
    "Zebra": "Least Concern"
}


ANIMAL_NAMES = ['Antelope',
                'Badger',
                'Bat',
                'Bear',
                'Bee',
                'Beetle',
                'Bison',
                'Boar',
                'Butterfly',
                'Cat',
                'Caterpillar',
                'Chimpanzee',
                'Cockroach',
                'Cow',
                'Coyote',
                'Crab',
                'Crow',
                'Deer',
                'Dog',
                'Dolphin',
                'Donkey',
                'Dragonfly',
                'Duck',
                'Eagle',
                'Elephant',
                'Flamingo',
                'Fly',
                'Fox',
                'Goat',
                'Goldfish',
                'Goose',
                'Gorilla',
                'Grasshopper',
                'Hamster',
                'Hare',
                'Hedgehog',
                'Hippopotamus',
                'Hornbill',
                'Horse',
                'Hummingbird',
                'Hyena',
                'Jellyfish',
                'Kangaroo',
                'Koala',
                'Ladybugs',
                'Leopard',
                'Lion',
                'Lizard',
                'Lobster',
                'Mosquito',
                'Moth',
                'Mouse',
                'Octopus',
                'Okapi',
                'Orangutan',
                'Otter',
                'Owl',
                'Ox',
                'Oyster',
                'Panda',
                'Parrot',
                'Pelecaniformes',
                'Penguin',
                'Pig',
                'Pigeon',
                'Porcupine',
                'Possum',
                'Raccoon',
                'Rat',
                'Reindeer',
                'Rhinoceros',
                'Sandpiper',
                'Seahorse',
                'Seal',
                'Shark',
                'Sheep',
                'Snake',
                'Sparrow',
                'Squid',
                'Squirrel',
                'Starfish',
                'Swan',
                'Tiger',
                'Turkey',
                'Turtle',
                'Whale',
                'Wolf',
                'Wombat',
                'Woodpecker',
                'Zebra']


MODEL = tf.keras.models.load_model("animal_classification_model.h5")


@app.get("/")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    return image


def animal_data(predicted_class):
    api_url = 'https://api.api-ninjas.com/v1/animals?name={}'.format(
        predicted_class)
    response = requests.get(
        api_url, headers={'X-Api-Key': ANIMAL_API_TOKEN})
    if response.status_code == requests.codes.ok:
        # print(response.text)
        pass

    else:
        print("Error:", response.status_code, response.text)

    data = response.text
    dict = json.loads(data)

    data = [i for i in dict if i["name"].lower() ==
            predicted_class.lower()]

    return data


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    bytes = await file.read()

    IMAGE_SIZE = (256, 256)

    # pass
    image = read_file_as_image(bytes)
    print(image)

    # Convert the file contents to a TensorFlow tensor

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    # resized_image.shape
    resized_image = tf.image.resize(img_array, IMAGE_SIZE)

    # model prediction
    predictions = MODEL.predict(resized_image)

    # processing predicted output to give valid result
    predicted_class = ANIMAL_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    print(predicted_class)

    data = animal_data(predicted_class)
    classification = Animal_danger_classification[predicted_class]

    return {
        "class": predicted_class,
        "confidence": float(confidence),
        "response": data,
        "Danger Classification": classification
    }


@app.post("/chatbot")
async def chat(data: dict):
    # Generate empty lists for generated and user.
    # Assistant Response
    print(data['text'])

    # get user input

    def get_text(data):

        return data['text']

    # Applying the user input box

    def chain_setup():

        template = """<|prompter|>{question}<|endoftext|>
        <|assistant|>"""

        prompt = PromptTemplate(
            template=template, input_variables=["question"])

        llm = HuggingFaceHub(
            repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"max_new_tokens": 1200}, huggingfacehub_api_token=HUGGINGFACEHUB_APT_TOKEN)

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt
        )
        return llm_chain

    # generate response

    def generate_response(question, llm_chain):
        response = llm_chain.run(question)
        return response

    # load LLM
    llm_chain = chain_setup()

    # main loop
    input_text = get_text(
        data) + 'tell me a first aid and provide me best possible solution in 10 points for this situation'
    response = generate_response(input_text, llm_chain)
    print(response)

    return response


# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)

'''
These are some of the animals whose exact data is not available in the api

{'name': 'boar', 'present': 0}
{'name': 'cat', 'present': 0}
{'name': 'dog', 'present': 0}
{'name': 'ladybugs', 'present': 0}
{'name': 'orangutan', 'present': 0}
{'name': 'panda', 'present': 0}
{'name': 'pelecaniformes', 'present': 0}
{'name': 'sandpiper', 'present': 0}
{'name': 'turtle', 'present': 0}
{'name': 'whale', 'present': 0}
'''
