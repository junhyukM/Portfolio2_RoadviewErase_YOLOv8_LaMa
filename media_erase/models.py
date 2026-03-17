from django.db import models
from django.conf import settings
import uuid

# Create your models here.

class MediaContents(models.Model):
    media = models.FileField(upload_to='videos/')

    media_uuid = models.UUIDField(unique=True, default=uuid.uuid4, editable=False)
