from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage

# Create your views here.

from uuid import uuid4
from . import inference
from . models import ImageContents
from . forms import ImageContentsForm
import base64

def img_detect(target_img):

    target_image_path = str(target_img)
    target_img = 'media/' + target_image_path

    inference.img_inference(target_img)

@ require_http_methods(['GET', 'POST'])
def main(request):
    if request.method == 'POST':
        img_form = ImageContentsForm(request.POST, request.FILES)
        if img_form.is_valid():
            image = img_form.save(commit=False)
            image.save()

            img_detect(image.image)
            
            return redirect('img_erase:inference_img' ,uuid=image.image_uuid)
            
    else:
        img_form = ImageContentsForm()
    
    return render(request, 'img_erase/main.html', {'img_form': img_form})

def inference_image(request, uuid):
    image = get_object_or_404(ImageContents, image_uuid=uuid)
    img_path = str(image.image)
    inference_path = img_path.replace('images', 'inferenced_images')
    
    return render(request, 'img_erase/img_inference.html', {
        'img_path' : img_path,
        'inference_path' : inference_path,
    })