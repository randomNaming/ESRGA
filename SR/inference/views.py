from django.shortcuts import render
from inferenceSR import inferenceSR
from django.http import FileResponse,HttpResponse
from django.views.decorators.cache import never_cache
import time
# Create your views here.
@never_cache
def home(request):
    if request.method == "GET":
        return render(request,"index.html")
    arg = request.POST
    print(arg)
    if len(arg) <=3 and arg['download'] == '下载图片':
        #在游览器中唤起下载图片
        response = FileResponse(open('static/result/out.png', 'rb'))
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment;filename="out.png"'
        return response
    elif arg['format'] == '0':
        pic = request.FILES.get("picture")
        if pic is None:
            return render(request, "index.html", {"error": "请选择要上传的图片文件"})
        f = open('input/inference.png',mode='wb')
        for chunk in pic.chunks():
            f.write(chunk)
        f.close()
        scale = int(arg['scale'])
        try:
            inferenceSR('input/inference.png',scale)
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"处理图像时出错: {error_msg}")
            print(traceback.format_exc())
            return render(request, "index.html", {"error": f"处理图像时出错: {error_msg}"})
        return render(request,"index.html")

@never_cache
def ZH(request):
    if request.method == "GET":
        return render(request,"index.zh-CN.html")
    arg = request.POST
    print(arg)
    if len(arg) <=3 and arg['download'] == 'Download':
        #在游览器中唤起下载图片
        response = FileResponse(open('static/result/out.png', 'rb'))
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment;filename="out.png"'
        return response
    elif arg['format'] == '0':
        pic = request.FILES.get("picture")
        if pic is None:
            return render(request, "index.zh-CN.html", {"error": "请选择要上传的图片文件"})
        f = open('input/inference.png',mode='wb')
        for chunk in pic.chunks():
            f.write(chunk)
        f.close()
        try:
            scale = int(arg.get('scale', 4))  # 默认scale为4
            inferenceSR('input/inference.png', scale)
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"处理图像时出错: {error_msg}")
            print(traceback.format_exc())
            return render(request, "index.zh-CN.html", {"error": f"处理图像时出错: {error_msg}"})
        return render(request,"index.zh-CN.html")