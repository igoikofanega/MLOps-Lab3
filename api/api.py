"""FastAPI application for image processing operations."""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io

from mylib.operations import (
    predict_class,
    resize_image,
    convert_to_grayscale,
    get_image_info,
)


app = FastAPI(title="Image Processing API", version="1.0.0")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Render the main home page with the upload form.

    Args:
        request: The incoming HTTP request.

    Returns:
        HTMLResponse: Rendered home.html template.
    """
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    """Predict the class of an uploaded image using the loaded model.

    Args:
        file: Uploaded image file (required).

    Returns:
        dict: JSON response containing:
            - filename: Name of the uploaded file.
            - predicted_class: Predicted class name.
            - image_info: Dictionary with size, width, height, and mode.

    Raises:
        HTTPException: 400 if the file is not a valid image or processing fails.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        predicted_class = predict_class(image)
        info = get_image_info(image)

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "image_info": info,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/resize")
async def resize_endpoint(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
) -> StreamingResponse:
    """Resize an uploaded image to the specified dimensions.

    Args:
        file: Uploaded image file (required).
        width: Desired width in pixels (must be > 0).
        height: Desired height in pixels (must be > 0).

    Returns:
        StreamingResponse: Resized image as a downloadable file.

    Raises:
        HTTPException: 400 if dimensions are invalid or image processing fails.
    """
    try:
        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400, detail="Width and height must be positive integers"
            )

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        resized_image = resize_image(image, width, height)

        img_byte_arr = io.BytesIO()
        save_format = image.format or "PNG"
        resized_image.save(img_byte_arr, format=save_format)
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type=f"image/{save_format.lower()}",
            headers={
                "Content-Disposition": f'attachment; filename="resized_{file.filename}"'
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid dimensions: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/grayscale")
async def grayscale_endpoint(file: UploadFile = File(...)) -> StreamingResponse:
    """Convert an uploaded image to grayscale.

    Args:
        file: Uploaded image file (required).

    Returns:
        StreamingResponse: Grayscale version of the image as a downloadable file.

    Raises:
        HTTPException: 400 if the file is not a valid image or processing fails.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        gray_image = convert_to_grayscale(image)

        img_byte_arr = io.BytesIO()
        save_format = image.format or "PNG"
        gray_image.save(img_byte_arr, format=save_format)
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type=f"image/{save_format.lower()}",
            headers={
                "Content-Disposition": f'attachment; filename="gray_{file.filename}"'
            },
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/info")
async def info_endpoint(file: UploadFile = File(...)) -> dict:
    """Retrieve metadata about an uploaded image.

    Args:
        file: Uploaded image file (required).

    Returns:
        dict: JSON response containing:
            - filename: Name of the uploaded file.
            - image_info: Dictionary with size, width, height, and mode.

    Raises:
        HTTPException: 400 if the file is not a valid image.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        info = get_image_info(image)

        return {
            "filename": file.filename,
            "image_info": info,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
