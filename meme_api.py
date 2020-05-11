#!/usr/bin/env python
# coding: utf-8

# In[28]:


from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import (
    ImageDataBunch,
    cnn_learner,
    open_image,
    get_transforms,
    models,
    get_image_files,
    load_learner,
    error_rate,
)
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio


# In[7]:


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


# In[8]:


app = Starlette()


# In[21]:


cat_images_path = Path("/tmp/meme_classifier")
# cat_fnames = get_image_files(cat_images_path)


# In[26]:


# cat_data = ImageDataBunch.from_name_re(
#     cat_images_path,
#     cat_fnames,
#     r"/([^/]+).jpg$",
#     valid_pct=0,
#     ds_tfms=get_transforms(),
#     size=224,
# )


# In[29]:


# cat_learner = cnn_learner(cat_data, models.resnet34, metrics=error_rate)
cat_learner = load_learner(cat_images_path)


# In[30]:


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


# In[31]:


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


# In[32]:


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = cat_learner.predict(img)
    return JSONResponse({
        "predictions": str(pred_class) 
    })


# In[33]:


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


# In[34]:


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)


# In[ ]:




