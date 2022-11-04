import os, io
import uuid
import numpy as np
import pandas as pd
from random_word import RandomWords
import time
import json

# from flask import Flask, render_template
# from flask_socketio import SocketIO, send, emit, join_room, leave_room
from fastapi import FastAPI, Request, Form, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datetime import datetime

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-uzsdaoJoOkGAwWtp9LpL1D1TgNRBv6vwMKhCezVTp4RguNlR'

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], 
    verbose=True,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

players = 4
rooms = []
r = RandomWords()

# ROUTES
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/join/{pid}')
def join(pid):
    print('player: ', pid)
    for room in rooms:
        if len(room['players']) < 4 and room['prompting'] == False and room['finished'] == False:
            player = {'id':pid, 'last_seen': datetime.now()}
            room['players'].append(player)
            room['updated'] = datetime.now()
            if len(room['players']) == 4:
                room['started'] = True
                room['updated'] = datetime.now()
                return room
            return room
    # if no room found, create a new room and join it
    room = {
        'id': random_words(3),
        'players': [{'id':pid, 'last_seen': datetime.now()}],
        'prompts': [],
            # '0': {'text': '', 'image': '', 'ready': False},
            # '1': {'text': '', 'image': '', 'ready': False},
            # '2': {'text': '', 'image': '', 'ready': False},
            # '3': {'text': '', 'image': '', 'ready': False},,
        'started': False,
        'prompting': False,
        'finished': False,
        'full': '',
        'created': datetime.now(),
        'updated': datetime.now()
    }
    rooms.append(room)
    return room

@app.get('/room/{rid}/{pid}')
def get_rooms(rid, pid):
    if len(rooms) == 0:
        return "no rooms"
    room = [room for room in rooms if room['id'] == rid][0]
    # update last seen
    player = [player for player in room['players'] if player['id'] == pid][0]
    player['last_seen'] = datetime.now()
    # check if other players are still in room
    for player in room['players']:
        if (datetime.now() - player['last_seen']).seconds > 10:
            room['players'].remove(player)
            # if prompts havent been submitted then unstart the room
            if not room['prompting']:
                room['started'] = False
    room['updated'] = datetime.now()
    return room

@app.get('/rooms')
def get_rooms():
    return rooms

@app.get('/mask')
def mask():
    make_mask()
    make_full_mask()
    return "ok"

@app.get("/test")
def testing():
    test_prompt_loop()
    return "ok"

class Prompt(BaseModel):
    text: str
    pid: str
    rid: str

@app.post('/prompt/{rid}/{pid}')
def post_prompt(rid, pid, background_tasks: BackgroundTasks, prompt: str = Form(...)):
    if len(rooms) == 0:
        return "no rooms, refresh"
    print('prompt: ', prompt, ' from player: ', pid, ' in room: ', rid)
    # add prompt that matches players position in array
    room = [room for room in rooms if room['id'] == rid][0]
    # the index of room.players where id == pid
    p = {'pid': pid, 'text': prompt, 'image': '', 'ready': False}
    # check if room prompts alreay has a prompt from this player
    player_has = False
    if len(room['prompts']) > 0:
        for i, rp in enumerate(room['prompts']):
            if rp['pid'] == pid:
                player_has = True
                room['prompts'][i] = p
    if not player_has:
        room['prompts'].append(p)
    # if all prompts are ready, emit ready event
    if len(room['prompts']) == 4:
        room['prompting'] = True
        room['updated'] = datetime.now()
        background_tasks.add_task(run_room, room)
        return room
    return room

# @app.route('/mask')
# def mask():
#     make_full_mask()
#     return "ok"

def random_string(len):
    import random
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(len))

def random_words(num):
    words = ''
    for i in range(num):
        word = r.get_random_word()
        if i == num-1:
            words += word
        else:
            words += word + '-'
    return words

# FUNCTIONS
def test_one_prompt():
    prompt = "head, cartoon, line drawing, of human head"
    generate_image(prompt)
    return

def test_prompt_loop():
    one = "head, head only, cartoon, line drawing, of human head, SFW, safe for work"
    two = "neck and shoulders only, a soldier body in uniform photorealistic, facing camera, SFW, safe for work"
    three = "torso only, body only, arm, no head, no legs, of an elephant 3d render, SFW, safe for work"
    four = "two legs only, no body, no head, no arms, the legs of a giant, SFW, safe for work"
    prompts = [one, two, three, four]
    images = []
    for i, p in enumerate(prompts):
        if i == 0:
            img = generate_image(p)
        else:
            init = next_image(images[len(images)-1])
            img = generate_image(p, init_image=init)
        images.append(img)
        print(images)
    full, nm = join_images(images)
    full_prompt = one + two + three + four
    rerender = generate_image(prompt=full_prompt, init_image=full, is_rerender=True)
    return nm

def run_room(room):
    images = []
    for i, prompt in enumerate(room['prompts']):
        if i == 0:
            p = prompt['text'] + 'head, head only,  exquisite, no blur, clear'
            prompt['image'] = generate_image(p)
            prompt['ready'] = True
            img = prompt['image']
        else:
            if i == 1:
                p = prompt['text'] + 'neck and shoulders, art, neck only, exquisite, no blur, clear', 
            elif i == 2:
                p = prompt['text'] + 'body, torso, middle, art, exquisite, no blur, clear'
            elif i == 3:
                p = prompt['text'] + 'legs and feet, legs and feet only, art, exquisite, no blur, clear'
            init = next_image(images[len(images)-1])
            prompt['image'] = generate_image(p, init_image=init)
            img = prompt['image']
        images.append(img)
    full, nm = join_images(images)
    room['finished'] = True
    room['prompting'] = False
    room['full'] = nm
    room['updated'] = datetime.now()

    full_prompt = room['prompts'][0]['text'] + room['prompts'][1]['text'] + room['prompts'][2]['text'] + room['prompts'][3]['text']
    rerender_path = generate_image(prompt=full_prompt, init_image=full, is_rerender=True)
    print("IMAGE RENDERED", rerender_path)
    room['full'] = rerender_path
    room['updated'] = datetime.now()
    # save a json file with the room data
    # with open('rooms/' + room['id'] + '.json', 'w') as f:
    #     # json.dump(room, f)
    #     json.dumps(room, f, cls=DateTimeEncoder)
    return


def join_images(images):
    # work backwards through list
    head = Image.open(images[0])
    head_cropped = head.crop((0, 0, 512, 384))
    neck = Image.open(images[1])
    neck_cropped = neck.crop((0, 128, 512, 384))
    torso = Image.open(images[2])
    torso_cropped = torso.crop((0, 128, 512, 384))
    legs = Image.open(images[3])
    legs_cropped = legs.crop((0, 128, 512, 512))
    # combine cropped into one image.
    full_height = head_cropped.height + neck_cropped.height + torso_cropped.height + legs_cropped.height
    full = Image.new('RGB', (512, full_height))
    full.paste(head_cropped, (0, 0))
    full.paste(neck_cropped, (0, head_cropped.height))
    full.paste(torso_cropped, (0, head_cropped.height+neck_cropped.height))
    full.paste(legs_cropped, (0, head_cropped.height+neck_cropped.height+torso_cropped.height))    
    # crop full so height is divisible by 64
    full = full.crop((0, 0, full.width, full.height - (full.height % 64)))
    nm = f'full-{str(uuid.uuid4())}.png'
    full.save(f'images/{nm}')
    return full, nm


def make_mask():
    w, h = 512, 512
    # image of w,h size. the top third of the image is black and the bottom two thirds are white
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[:int(h/2), :, :] = 255
    mask[int(h/2):, :, :] = 0
    # convert to PIL image
    mask = Image.fromarray(mask)
    # make it blurry
    # import ImageFilter form PIL
    from PIL import ImageFilter
    mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
    # save image with name mask.png and return the path to the file
    mask.save('images/mask.png')

def make_full_mask():
    import random
    w, h = 512, 1280
    # image of w,h size. the top third of the image is black and the bottom two thirds are white
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    # make whole image white
    mask[:, :, :] = 255
    # random 10% of pixels are black
    for i in range(int((w*h)*0.5)):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        mask[y, x, :] = 0
    # convert to PIL image
    mask = Image.fromarray(mask)
    # make it blurry
    # import ImageFilter form PIL
    from PIL import ImageFilter
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    # save image with name mask.png and return the path to the file
    mask.save('images/mask_full.png')

def next_image(prev_image_src):
    print("next image is ", prev_image_src)
    image = Image.open(prev_image_src)
    # get the bottom 1/2 of image
    bot = image.crop((0, int((image.height/2)), image.width, image.height))
    # make noise image of same size
    noise_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    # convert to PIL image
    noise_image = Image.fromarray(noise_image)
    # past bot over the top of noise image
    noise_image.paste(bot, (0, 0))
    folder, filename = prev_image_src.split('/')
    noise_image.save(f'{folder}/next-{filename}', 'PNG')
    # return noise image    
    return noise_image
    

def generate_image(prompt, init_image=False, is_rerender=False):
    print("Generating image...")
    if init_image:
        if is_rerender:
            print("rerendering")
            mask_image = Image.open('images/mask_full.png')
            answers = stability_api.generate(
                prompt=prompt + " SFW, detailed, 4k, sharp, not blurry",
                steps=50,
                init_image=init_image,
                mask_image=mask_image,
                start_schedule=0.2
            )
        else:
            mask_image = Image.open('images/mask.png')
            print("...with initial image...")
            answers = stability_api.generate(
                prompt=prompt,
                steps=35,
                init_image=init_image,
                start_schedule=1,
                mask_image=mask_image
            )
    else:
        print("...with first prompt...")
        answers = stability_api.generate(
            prompt=prompt,
            steps=35
        )

    
    dir = 'images'
    ty = '.png'
    nm = uuid.uuid1()
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                print('hit nsfw filter')
            if artifact.type == generation.ARTIFACT_IMAGE:
                if is_rerender:
                    path = f'renders/{nm}{ty}'
                else:
                    path = os.path.join(dir, f'{nm}{ty}')
                im = Image.open(io.BytesIO(artifact.binary))
                im.save(path)
                return path
        
class DateTimeEncoder(json.JSONEncoder):
    def default(self, z):
        if isinstance(z, datetime.datetime):
            return (str(z))
        else:
            return super().default(z)