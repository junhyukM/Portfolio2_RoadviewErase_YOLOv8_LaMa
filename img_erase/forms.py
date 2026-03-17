from django import forms
from . models import ImageContents

# models 에서 정의한 내용 html 에서 form 으로 정의
class ImageContentsForm(forms.ModelForm):
    class Meta:
        model = ImageContents
        fields = ('image', )
