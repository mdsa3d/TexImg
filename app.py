from functools import cmp_to_key
import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline

import random, string
# create the app

app = tk.Tk()
app.geometry("532x632") # set size of the actual app
app.title("TexImg") # title of the app
ctk.set_appearance_mode("dark")

# text entry fields
prompt = ctk.CTkEntry(
                        height=40, width=512, # size of the prompt box
                        text_font=("Arial", 20),
                        text_color="black", # set color of the prompt box
                        fg_color= "white" # set the color of actual box
                        )
prompt.place(x=10, y=10) # place the prompt box on the app

# placeholder for the image
lmain = ctk.CTkLabel(height=512, width=512) # this is usual size stable diffusion model returns
lmain.place(x=10, y=110)

###### load the model ###########
# model id, you can try different models
modelID = "CompVis/stable-diffusion-v1-4"
device= "cpu"
# set the pipeline
if device == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(modelID, 
                                                        revision="fp16", # this revision allows you to work with smaller VRAM - 4gb atleast
                                                        torch_dtype=torch.float16, # datatytpe
                                                        use_auth_token=auth_token # token to access the hugging face model
                                                        )
elif device =="cpu":
        pipe = StableDiffusionPipeline.from_pretrained(modelID, 
                                                        revision="fp16", # this revision allows you to work with smaller VRAM - 4gb atleast
                                                        use_auth_token=auth_token # token to access the hugging face model
                                                        )                                        
pipe.to(device)



def generate():

    with autocast(): # send things to gpu
        image = pipe(prompt.get(), # get the prompt from the prompt box
                guidance_scale=8.5 # how closely you want stable diffusion to follow written text in prompt, higher the value the image will be more closer to the text
                )["sample"][0] # first passing through the sample and then extract the first image
        r = ''.join(random.choice(string.digits) for _ in range(3)) # create a random number of 3 digits 
        image.save('{}generatedimage.png'.format(r))# save the iamge
        # create the image
        img = ImageTk.PhotoImage(image)
        # set the frame
        lmain.configure(image=img)
#create a button
trigger =  ctk.CTkButton(
                        height=40, width=120, 
                        text_font=("Arial", 20),
                        text_color="black", # text color
                        fg_color= "brown", # color of actual box
                        command=generate # function it is going to trigger
                        )
trigger.configure(text="Generate") # text in the button
trigger.place(x=206, y=60)

app.mainloop() # to run the app