# tf_project/Dockerfile
FROM tensorflow/tensorflow:2.18.0-gpu

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    "numpy<2.0.0" \
    pandas \
    matplotlib \
    scikit-learn \
    tqdm \
    tf-keras \
    tensorflow-probability \
    jupyterlab \ 
    ipywidgets
    

RUN pip install --no-cache-dir "numpy<2.0.0" --force-reinstall
RUN pip install tensorflow[and-cuda]==2.18.0

# 設定工作目錄
WORKDIR /tf

# 開放 8888 端口
EXPOSE 8888

# 啟動 Jupyter Lab
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--allow-root", \
     "--no-browser", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''", \
     "--ServerApp.allow_origin='*'", \
     "--ServerApp.root_dir=/workspace"]