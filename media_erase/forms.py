from django import forms
from . models import MediaContents

class MediaContentsForm(forms.ModelForm):
    class Meta:
        model = MediaContents
        fields = ('media', )
