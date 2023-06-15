from PIL import Image
import torch
from fastapi import FastAPI, UploadFile
import io
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this based on your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# img = Image.open('16.jpeg').convert("RGB")
# out = face2paint(model, img)
# out.save('output_anime.jpg')
# Add CORS middleware


@app.post("/process_image")
async def process_image(file: UploadFile):
    # Read the uploaded image
    img = Image.open(file.file).convert("RGB")

    # Process the image with AnimeGAN2
    out = face2paint(model, img)

    # Save the output image to a BytesIO object
    output_image = io.BytesIO()
    out.save(output_image, format="JPEG")
    output_image.seek(0)

    return StreamingResponse(output_image, media_type="image/jpeg")
