
from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name="home"),
    path('contact.html',views.contact, name="contact"),
    path('services.html',views.services, name="services"),
    path('blog.html',views.blog, name="blog"),
    path('about.html',views.about, name="about"),
    path('getstart.html',views.getstart, name="getstart"),
    path('doctors.html',views.doctors, name="doctors"),
    path('camera.html',views.camera, name="camera"),
    path('question.html',views.question, name="question"),
    path('happy.html',views.happy, name="happy"),
    path('sad.html',views.sad, name="sad"),
    path('neutral.html',views.neutral, name="neutral"),
    path('rate.html',views.rate, name="rate"),
    path('angry.html',views.angry, name="angry"),
    path('fear.html',views.angry, name="fear"),
]
