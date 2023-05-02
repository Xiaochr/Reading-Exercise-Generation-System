from django.urls import path

from . import views

urlpatterns = [
    path('gen_essay/', views.gen_essay, name='index0'),
    path('gen_questions/', views.gen_questions, name='index4'),
]
