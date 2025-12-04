import gradio as gr
import requests
from PIL import Image
import io
import os

# URL del API en Render (debes cambiar esto por tu URL real)
API_URL = os.getenv("API_URL", "https://igoikofanegamlops-lab2.onrender.com")

def predict_image(image):
    """Predice la clase de la imagen"""
    if image is None:
        return "Por favor, sube una imagen"
    
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('predicted_class', 'Error desconocido')
            return f"Predicción: {prediction}"
        else:
            return f"Error en el API: {response.status_code}"
    
    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"

def grayscale_image(image):
    """Convierte la imagen a escala de grises"""
    if image is None:
        return None, "Por favor, sube una imagen"
    
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{API_URL}/grayscale", files=files)
        
        if response.status_code == 200:
            # La respuesta es una imagen en bytes
            img = Image.open(io.BytesIO(response.content))
            return img, "Image converted"
        else:
            return None, f"Error en el API: {response.status_code}"
    
    except Exception as e:
        return None, f"Error al procesar la imagen: {str(e)}"

def resize_image(image, width, height):
    """Redimensiona la imagen"""
    if image is None:
        return None, "Por favor, sube una imagen"
    
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        data = {'width': int(width), 'height': int(height)}
        response = requests.post(f"{API_URL}/resize", files=files, data=data)
        
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            return img, f"Imagen redimensionada a {width}x{height}"
        else:
            return None, f"Error en el API: {response.status_code}"
    
    except Exception as e:
        return None, f"Error al procesar la imagen: {str(e)}"

def get_image_info(image):
    """Obtiene información de la imagen"""
    if image is None:
        return "Por favor, sube una imagen"
    
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{API_URL}/info", files=files)
        
        if response.status_code == 200:
            result = response.json()
            filename = result.get('filename', {})
            image_info = result.get('image_info', {})
            height = str(image_info.get('height', {}))
            width = str(image_info.get('width', {}))
            mode = image_info.get('mode', {})
            formate = image_info.get('format', {})
            
            # Formatear la información de forma legible
            info_text = f"📊 Information of the image: {filename}\n\n"
            info_text += "📏 Dimensions and format:\n"
            info_text += f"width: {width} px\n"
            info_text += f"height: {height} px\n"
            info_text += f"Mode of color: {mode}\n"
            info_text += f"Format: {formate}\n"
            
            
            return info_text
        else:
            return f"Error en el API: {response.status_code}"
    
    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"

# Crear interfaz Gradio con pestañas
with gr.Blocks(title="Random Model - Image Processing") as demo:
    gr.Markdown("# 🖼️ Random Model - Image Processing Suite")
    gr.Markdown("Tools for image processing and analysis")
    
    with gr.Tabs():
        # Pestaña 1: Predicción
        with gr.TabItem("🔮 Prediction"):
            with gr.Row():
                pred_image = gr.Image(type="pil", label="Imagen de entrada")
            with gr.Row():
                pred_button = gr.Button("Predict", variant="primary")
            with gr.Row():
                pred_output = gr.Textbox(label="Resultado", interactive=False)
            
            pred_button.click(fn=predict_image, inputs=pred_image, outputs=pred_output)
        
        # Pestaña 2: Escala de grises
        with gr.TabItem("⚫⚪ Escale of gray"):
            with gr.Row():
                gray_image = gr.Image(type="pil", label="Imagen de entrada")
            with gr.Row():
                gray_button = gr.Button("Convertir a gris", variant="primary")
            with gr.Row():
                gray_output_img = gr.Image(label="Imagen convertida")
                gray_output_msg = gr.Textbox(label="Mensaje", interactive=False)
            
            gray_button.click(fn=grayscale_image, inputs=gray_image, outputs=[gray_output_img, gray_output_msg])
        
        # Pestaña 3: Redimensionar
        with gr.TabItem("📐 Resize"):
            with gr.Row():
                resize_image_input = gr.Image(type="pil", label="Imagen de entrada")
            with gr.Row():
                resize_width = gr.Number(value=200, label="Ancho", precision=0)
                resize_height = gr.Number(value=200, label="Alto", precision=0)
            with gr.Row():
                resize_button = gr.Button("Resize", variant="primary")
            with gr.Row():
                resize_output_img = gr.Image(label="Image resized")
                resize_output_msg = gr.Textbox(label="Message", interactive=False)
            
            resize_button.click(fn=resize_image, inputs=[resize_image_input, resize_width, resize_height], outputs=[resize_output_img, resize_output_msg])
        
        # Pestaña 4: Información
        with gr.TabItem("ℹ️ Information"):
            with gr.Row():
                info_image = gr.Image(type="pil", label="Imagen de entrada")
            with gr.Row():
                info_button = gr.Button("Obtain information", variant="primary")
            with gr.Row():
                info_output = gr.Textbox(label="Information of the image", interactive=False, lines=8)
            
            info_button.click(fn=get_image_info, inputs=info_image, outputs=info_output)

if __name__ == "__main__":
    demo.launch()
