from django.shortcuts import render
from django.views import View
from classifier.NNmodel import get_predict
import numpy as np


class IndexClass(View):
    context = {}

    def get(self, request):
        self.context['submit'] = "NO"
        if request.GET:
            v_1 = int(self.request.GET.get('1_s'))
            v_2 = int(self.request.GET.get('2_s'))
            v_3 = int(self.request.GET.get('3_s'))
            v_4 = int(self.request.GET.get('4_s'))
            v_5 = int(self.request.GET.get('5_s'))
            v_6 = int(self.request.GET.get('6_s'))
            usr_input = [v_1, v_2, v_3, v_4, v_5, v_6]
            p = get_predict(np.array([usr_input]))
            self.context['usr_input'] = usr_input
            self.context['predict_p'] = "{:.3f}%".format(p[0][0]*100)
            self.context['submit'] = "YES"
        return render(request, 'index.html', self.context)
