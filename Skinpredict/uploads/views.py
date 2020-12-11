from django.shortcuts import render

from django.http import HttpResponse 
from django.shortcuts import render, redirect 
from .forms import *
from uploads.models import skin
  
# Create your views here. 
def  image_view(request): 
    if request.method == 'POST': 
        form = ImageForm(request.POST, request.FILES) 
        if form.is_valid(): 
            form.save()   
            return redirect('success') 
    else: 
        form = ImageForm() 
    return render(request, 'image_form.html', {'form' : form}) 
  
  
def success(request): 
    return HttpResponse('successfully uploaded') 

def skin_images(request): 
  
    if request.method == 'GET': 
  
        # getting all the objects of hotel. 
        skinimages = skin.objects.all()
        return render(request, 'skin_images.html', {'skin_images' : skinimages[len(skinimages)-1:len(skinimages)]}) 
