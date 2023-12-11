from django.db import models

# Create your models here.
# models.py
"""
from django.db import models

class Feedback(models.Model):
    RATING_CHOICES = (
        (1, '1 Star'),
        (2, '2 Stars'),
        (3, '3 Stars'),
        (4, '4 Stars'),
        (5, '5 Stars'),
    )

    rating = models.PositiveSmallIntegerField(choices=RATING_CHOICES)
    feedback_text = models.TextField()

    def __str__(self):
        return f'Rating: {self.rating} - Feedback: {self.feedback_text}"""
